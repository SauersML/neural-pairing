"""Trains the two linked models on the games: LoRA on both models plus both halves of the link.

Each step draws a batch of episodes (duplex games and, with probability DIALOGUE, dialogue games), lays out both
models' teacher-forced token streams in time (session.build), reads both models with 2 or 3 Gauss-Seidel rounds
(link.Pair.coupled), and minimizes the weighted cross-entropy of every target token plus a small cost per message sent.
Cooperative games train everything on the sum of both models' losses. For the first WARM of the steps every gate is
open and sending is free, so the receive adapters learn to read the messages before the gates can close (an untrained
message is noise to the receiver); then the gates are learned, and the send cost rises to its full value over RAMP. Every COMP-th step is competitive: in 'resist'
the sender tries to steer a receiver that was told to ignore it, in 'impersonate' the sender tries to make the
receiver introduce itself as the sender, and in 'bluff' a model thinking of one concept tries to make its partner name
another. Each competitive episode becomes two sequences: the receiver's own target, whose loss trains only the
receiver (its LoRA and its half of the link), and the sender's goal (the steered reply, the false identity, the decoy),
whose loss trains only the sender.

Under torchrun (one process per GPU) every rank draws the same global batch, keeps one group of episodes of similar
length (ddp.partition), and the gradients are summed over the ranks after the backward pass (ddp.sum_grads); losses
use the denominators of the whole batch, so the update equals the single-process update on that batch. Rank 0 logs and
saves; the quick evaluation is split over the ranks by game.

Benchmark questions (data/bench.json): with --qa P a fraction P of the episodes are training questions that both
models see and answer at once over the link. The controls train on the same questions: --solo trains each model alone
(the link cut), --text_pair trains the pair to answer one after the other in text with the link closed.

Usage: python train.py --replies OUT/replies.json --out OUT/run [--steps 3000] [--init RUN/step_N]
       torchrun --nproc_per_node 4 train.py ... --ckpt none
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

import ddp
import games as G
import session as S
from link import DEVICE, MODELS, SIDES, Pair, find_layers, load

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lr_link', type=float, default=1e-3)
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--send_cost', type=float, default=0.01)
ap.add_argument('--comp', type=int, default=5, help='every comp-th step is competitive (0: never)')
ap.add_argument('--p3', type=float, default=0.3, help='probability of 3 rounds instead of 2')
ap.add_argument('--lr_frac', type=float, default=0.3, help='receive layer, fraction of depth')
ap.add_argument('--ls_frac', type=float, default=0.67, help='send layer, fraction of depth')
ap.add_argument('--size', type=float, default=0.3, help='write size, in units of the residual RMS')
ap.add_argument('--warm', type=float, default=0.4, help='fraction of steps with every gate open and no send cost')
ap.add_argument('--ramp', type=float, default=0.2, help='fraction of steps over which the send cost then rises')
ap.add_argument('--eval_every', type=int, default=500)
ap.add_argument('--first_eval', type=int, default=20, help='an extra early evaluation, to check the pipeline')
ap.add_argument('--save_every', type=int, default=1000)
ap.add_argument('--max_digits', type=int, default=16)
ap.add_argument('--ckpt', default='both', choices=('both', 'none', 'quill', 'ember'), help='gradient checkpointing')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dialogue', type=float, default=0.0, help='probability that an episode is a dialogue game')
ap.add_argument('--init', default=None, help='a saved step to start from (LoRA and link)')
ap.add_argument('--quick_n', type=int, default=32)
ap.add_argument('--qa', type=float, default=0.0, help='probability that an episode is a benchmark question')
ap.add_argument('--solo', action='store_true', help='benchmark questions only, each model alone (link cut)')
ap.add_argument('--text_pair', action='store_true', help='benchmark questions only, answered in turn in text')
ap.add_argument('--payload_every', type=int, default=5,
                help='every N steps, the payload loss also with the link cut, on the same batch')
ap.add_argument('--recv_init', default='zero', choices=('zero', 'random'),
                help="the receive adapter's last layer at the start: zero, or PyTorch's random init")
ap.add_argument('--inbox', action='store_true', help='messages stay in the stream (link.Pair.inbox)')
ap.add_argument('--soft', action='store_true', help='the soft-token read (link.Pair.soft): each message is a token of '
                'its own in the partner\'s input, starting as the sender\'s next-token distribution as text')
ap.add_argument('--payload_eval', type=int, default=0,
                help='payload loss with the link and cut on N fixed never-seen episodes, at step 0 and every evaluation')
ap.add_argument('--halt', type=int, default=0, help='stop after this step (probe runs; the schedule is still --steps)')
ap.add_argument('--grad_check', default=None, help='save the summed gradients of every step to PATH.STEP.pt')
args = ap.parse_args()
ddp.init()
if not ddp.MAIN:
    sys.stdout = open(os.devnull, 'w')
ONLY_QA = args.solo or args.text_pair
if ONLY_QA:
    args.comp, args.send_cost = 0, 0.0
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
t0 = time.time()

toks, models, lr, ls = {}, {}, {}, {}
for x in SIDES:
    toks[x], m = load(x)
    n = len(find_layers(m))
    lr[x], ls[x] = round(args.lr_frac * n), round(args.ls_frac * n)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    m = get_peft_model(m, cfg)
    if args.init:
        set_peft_model_state_dict(m, load_file(os.path.join(args.init, x, 'adapter_model.safetensors')))
    if args.ckpt in ('both', x):
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.config.use_cache = False
    models[x] = m
    print(f'{x}: {MODELS[x]["hf"]}, {n} blocks, receive after block {lr[x]}, send from block {ls[x]}', flush=True)

replies = json.load(open(args.replies))
train_games = G.Games(replies, 'train', seed=args.seed, max_digits=args.max_digits)
test_games = G.Games(replies, 'test', seed=1000 + args.seed, max_digits=args.max_digits)


@torch.no_grad()
def embedding_rms(x):
    """Median RMS of model x's input embeddings over the tokens of a few training prompts."""
    E = models[x].get_input_embeddings().weight
    ids = torch.tensor([t for _ in range(8) for t in S.build(train_games.episode('plain'), toks)[x]['ids']])
    return float(E[ids.to(E.device)].float().pow(2).mean(-1).sqrt().median())


@torch.no_grad()
def residual_rms(x):
    """Median per-token RMS of the residual after block lr[x] on a few training prompts."""
    model = models[x]
    eps = [train_games.episode('plain') for _ in range(8)]
    vals = []
    h = {}
    hook = find_layers(model)[lr[x]].register_forward_hook(lambda mod, i, o: h.update(v=o[0] if isinstance(o, tuple) else o))
    for ep in eps:
        model(input_ids=torch.tensor([S.build(ep, toks)[x]['ids']], device=DEVICE))
        vals.append(h['v'][0, 1:].float().pow(2).mean(-1).sqrt())
    hook.remove()
    return float(torch.cat(vals).median())


if args.init:
    ck = torch.load(os.path.join(args.init, 'link.pt'), weights_only=False)
    assert ck['lr'] == lr and ck['ls'] == ls, (ck['lr'], ck['ls'])
    sizes = ck['sizes']
    rms = ck.get('rms') or {x: sizes[x] / ck['args']['size'] for x in SIDES}
    pair = Pair(models, toks, lr, ls, sizes, soft=ck.get('soft', False)).to(DEVICE)
    pair.side.load_state_dict(ck['link'])
    print('started from', args.init, flush=True)
else:
    rms = {x: embedding_rms(x) if args.soft else residual_rms(x) for x in SIDES}
    sizes = {x: args.size * rms[x] for x in SIDES}
    print('initial write size per element:', {x: round(v, 3) for x, v in sizes.items()}, flush=True)
    pair = Pair(models, toks, lr, ls, sizes, zero_write=args.recv_init == 'zero', soft=args.soft).to(DEVICE)
for x in SIDES:
    models[x].train()

params = {x: [p for p in models[x].parameters() if p.requires_grad] + list(pair.side[x].parameters()) for x in SIDES}
ALL = [p for x in SIDES for p in params[x]]
pair.inbox = args.inbox or (bool(args.init) and ck.get('inbox', False))
ddp.broadcast(ALL)
opt = torch.optim.AdamW([{'params': [p for p in models[x].parameters() if p.requires_grad], 'lr': args.lr}
                         for x in SIDES] + [{'params': list(pair.side.parameters()), 'lr': args.lr_link}],
                        weight_decay=0.0, betas=(0.9, 0.98))
base_lr = [g['lr'] for g in opt.param_groups]


BOTH = frozenset(SIDES)
PADS = {x: toks[x].pad_token_id for x in SIDES}


def adversary(ep):
    """The sender's goal as a target sequence: the receiver's answer replaced by meta['adversary'], nothing else."""
    s, r = ep['meta']['sender'], ep['meta']['receiver']
    if ep['mode'] == 'duplex':
        return dict(ep, reply={s: (ep['reply'][s][0], 0.0), r: (ep['meta']['adversary'], 1.0)})
    *head, (sp, _, _) = ep['turns']
    return dict(ep, turns=[(a, t, 0.0) for a, t, _ in head] + [(sp, ep['meta']['adversary'], 1.0)])


BENCH = G.bench('train') if (args.qa > 0 or ONLY_QA) else None
VAL = G.bench('val') if BENCH else None


def draw(rng, games):
    if args.solo:
        return G.qa_episode(rng.choice(BENCH), 'solo')
    if args.text_pair:
        return G.qa_episode(rng.choice(BENCH), 'text', order=rng.sample(SIDES, 2))
    if rng.random() < args.qa:
        return G.qa_episode(rng.choice(BENCH), 'pair')
    if rng.random() < args.dialogue:
        return games.episode(rng.choice(G.DIALOGUE))
    return games.episode()


def make_rows(competitive, rng, games):
    """(episode, {x: the sides whose parameters x's losses train})"""
    if not competitive:
        return [(draw(rng, games), {x: BOTH for x in SIDES}) for _ in range(args.bs)]
    rows = []
    names = G.COMPETITIVE + (G.DIALOGUE_COMPETITIVE if args.dialogue > 0 else ())
    for _ in range(args.bs // 2):
        ep = games.episode(rng.choice(names))
        s, r = ep['meta']['sender'], ep['meta']['receiver']
        rows.append((ep, {s: frozenset([s]), r: frozenset([r])}))
        rows.append((adversary(ep), {s: frozenset(), r: frozenset([s])}))
    return rows


rng = random.Random(args.seed)


def payload(ep):
    """{x: [(text, start, end)]}: the characters of x's target reply that only the partner can supply (the relayed
    value, the partner's number and the sum, the partner's half of the fact, the looked-up word). Duplex games only."""
    if ep['mode'] != 'duplex':
        return {}
    g, m = ep['game'], ep['meta']
    text = {x: ep['reply'][x][0] for x in SIDES}

    def at(x, sub, after=''):
        i = text[x].find(after + sub)
        return [(x, i + len(after), i + len(after) + len(sub))] if i >= 0 else []
    if g.startswith('relay'):
        return {m['receiver']: at(m['receiver'], m['value'])}
    if g == 'merge_sum':
        v = m['values']
        return {x: at(x, str(v[S.partner(x)]), "'s is ") + at(x, str(m['sum']), 'sum is ') for x in SIDES}
    if g == 'merge_fact':
        n = m['has_name']
        return {n: at(n, m['city'], 'lives in '), S.partner(n): at(S.partner(n), m['name'])}
    if g == 'lookup':
        return {m['asker']: at(m['asker'], m['value'], 'the word ')}
    return {}


def payload_labels(ep):
    """{x: indices into x's labels of the payload tokens} (in a duplex episode label j predicts reply token j)."""
    out = {}
    for x, spans in payload(ep).items():
        if not spans:
            continue
        enc = toks[x](ep['reply'][x][0], add_special_tokens=False, return_offsets_mapping=True)
        out[x] = [j for j, (a, b) in enumerate(enc.offset_mapping) if any(a < e and b > s0 for _, s0, e in spans)]
    return out


def prepare(step, rng=rng, games=train_games):
    """The global batch of a step: the same on every rank."""
    competitive = args.comp > 0 and step % args.comp == 0
    rows = make_rows(competitive, rng, games)
    built = [S.build(ep, toks) for ep, _ in rows]
    for (ep, _), e in zip(rows, built):
        for x, js in payload_labels(ep).items():
            e[x]['payload'] = js
    rounds = 1 if ONLY_QA else 3 if rng.random() < args.p3 else 2
    return competitive, rows, built, rounds


def forward(rows, built, rounds, cost_scale=1.0):
    """This rank's share of the global batch: losses (with the denominators of the whole batch) and statistics."""
    mine = ddp.partition([sum(len(e[x]['ids']) for x in SIDES) for e in built])[ddp.RANK]
    denom = {x: max(1.0, sum(wt for e in built for _, _, wt in e[x]['labels'])) for x in SIDES}
    n_send = {x: max(1, sum(t >= 0 for e in built for t in e[x]['time'])) for x in SIDES}
    rows, built = [rows[i] for i in mine], [built[i] for i in mine]
    b = S.collate(built, PADS, [o for _, o in rows], soft=pair.soft)
    last, msgs = pair.coupled(b, rounds=rounds, mode='zero' if ONLY_QA else None)
    loss_for = {x: torch.zeros((), device=DEVICE) for x in SIDES}
    total = torch.zeros((), device=DEVICE)
    tally = {}
    for x in SIDES:
        logits = pair.token_logits(x, last[x], b[x]['idx'])
        ce = F.cross_entropy(logits, b[x]['lab'], reduction='none')
        w = b[x]['w']
        for o in SIDES:
            loss_for[o] = loss_for[o] + (ce * w * b[x]['own_mask'][o]).sum() / denom[x]
        total = total + (ce * w).sum() / denom[x]
        # per-episode teacher-forced token accuracy and loss on answer tokens (weight 1), summed on the GPU
        sel = (w >= 1.0).float()
        hit = (logits.argmax(-1) == b[x]['lab']).float()
        z = torch.zeros(3, len(rows), device=DEVICE)
        tally[x] = z.index_add_(1, b[x]['row'], torch.stack([hit * sel, ce.detach() * sel, sel]))
    sent = torch.zeros(len(SIDES), device=DEVICE)
    for j, x in enumerate(SIDES):
        g, p = msgs[x][1], msgs[x][2]
        cost = cost_scale * args.send_cost * p.sum() / n_send[x]
        loss_for[x] = loss_for[x] + cost
        total = total + cost
        sent[j] = (g.detach() > 0.5).float().sum() / n_send[x]
    return rows, built, b, last, loss_for, total, sent, tally


@torch.no_grad()
def payload_ce(rows, built, b, rounds, last, episodes=None, ids=None):
    """{game/receiving model: [summed payload cross-entropy with the link, with the link cut, payload tokens]} on this
    rank's batch. episodes: a list that gets one record per episode and receiving model (ids: the episodes' ids)."""
    sel = {x: [] for x in SIDES}
    games = {x: [] for x in SIDES}
    rec = {}
    for x in SIDES:
        off = 0
        for k, ((ep, owners), e) in enumerate(zip(rows, built)):
            if owners[x] == BOTH:            # cooperative episodes only
                for j in e[x].get('payload', []):
                    sel[x].append(off + j)
                    games[x].append(f"{ep['game']}/{x}")
                    rec.setdefault((k, x), []).append(len(sel[x]) - 1)
            off += len(e[x]['labels'])
    if not any(sel.values()):
        return {}
    pair.clear()
    keep = dict(pair.write_ms)
    cut, _ = pair.coupled(b, rounds=rounds, mode='zero')
    pair.write_ms = keep
    out = {}
    for x in SIDES:
        if not sel[x]:
            continue
        i = torch.tensor(sel[x], device=DEVICE)
        ce = [F.cross_entropy(pair.token_logits(x, h[x], b[x]['idx'][i]), b[x]['lab'][i], reduction='none').tolist()
              for h in (last, cut)]
        for g, a, z in zip(games[x], *ce):
            o = out.setdefault(g, [0.0, 0.0, 0])
            o[0] += a; o[1] += z; o[2] += 1
        if episodes is not None:
            for (k, xx), js in rec.items():
                if xx == x:
                    episodes.append(dict(episode=ids[k], game=rows[k][0]['game'], receiver=x, n_tokens=len(js),
                                         link_ce=sum(ce[0][j] for j in js) / len(js),
                                         cut_ce=sum(ce[1][j] for j in js) / len(js)))
    pair.clear()
    return out


PAYLOAD_GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')


def payload_eval(n, step):
    """{game/receiving model: [mean payload cross-entropy with the link, with the link cut, payload tokens]} on n fixed
    episodes of the never-seen split (every gate as in training at this step), split over the ranks. Per episode and
    receiving model (mean over its payload tokens) in OUT/payload_step{step}.json."""
    g = G.Games(replies, 'test', seed=4242, max_digits=args.max_digits)
    rows = [(g.episode(PAYLOAD_GAMES[i % len(PAYLOAD_GAMES)]), {x: BOTH for x in SIDES}) for i in range(n)]
    built = [S.build(ep, toks) for ep, _ in rows]
    for (ep, _), e in zip(rows, built):
        for x, js in payload_labels(ep).items():
            e[x]['payload'] = js
    mine = ddp.partition([sum(len(e[x]['ids']) for x in SIDES) for e in built])[ddp.RANK]
    out, eps = {}, []
    with torch.no_grad():
        for k in range(0, len(mine), args.bs):
            part = mine[k:k + args.bs]
            r, bl = [rows[i] for i in part], [built[i] for i in part]
            b = S.collate(bl, PADS, [o for _, o in r], soft=pair.soft)
            last, _ = pair.coupled(b, rounds=3)
            pair.clear()
            out = merge([out, payload_ce(r, bl, b, 3, last, eps, part)])
    out = merge(ddp.gather(out))
    eps = sorted([e for part in ddp.gather(eps) for e in part], key=lambda e: (e['episode'], e['receiver']))
    if ddp.MAIN:
        json.dump(eps, open(os.path.join(args.out, f'payload_step{step}.json'), 'w'))
    return {k: [round(a[0] / a[2], 3), round(a[1] / a[2], 3), a[2]] for k, a in sorted(out.items())}


def backward(competitive, loss_for, total):
    opt.zero_grad(set_to_none=True)
    if competitive:
        torch.autograd.backward(loss_for['quill'], inputs=params['quill'], retain_graph=True)
        torch.autograd.backward(loss_for['ember'], inputs=params['ember'])
    else:
        total.backward()
    ddp.sum_grads(ALL)


def probe():
    """Peak memory of one step on the costliest global batch of 64 draws with 3 rounds (an out-of-memory error here
    means: turn gradient checkpointing on)."""
    prng, pgames = random.Random(args.seed + 7), G.Games(replies, 'train', seed=args.seed + 7, max_digits=args.max_digits)
    rows = [r for _ in range(4) for r in make_rows(False, prng, pgames)]
    built = [S.build(ep, toks) for ep, _ in rows]
    top = sorted(range(len(rows)), key=lambda i: -sum(len(built[i][x]['ids']) for x in SIDES))[:args.bs]
    torch.cuda.reset_peak_memory_stats(DEVICE)
    _, _, _, _, loss_for, total, _, _ = forward([rows[i] for i in top], [built[i] for i in top], 1 if ONLY_QA else 3)
    backward(False, loss_for, total)
    opt.zero_grad(set_to_none=True)
    pair.clear()
    peak = ddp.gather(round(torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30, 1))
    longest = max(len(built[i][x]['ids']) for i in top for x in SIDES)
    print(f'peak memory on the costliest batch (longest sequence {longest} tokens), GiB per rank: {peak}', flush=True)
    torch.cuda.reset_peak_memory_stats(DEVICE)


if DEVICE.startswith('cuda'):
    probe()


def merge(parts):
    out = {}
    for st in parts:
        for k, a in st.items():
            m = out.setdefault(k, [0.0, 0.0, 0])
            for j in range(3):
                m[j] += a[j]
    return out


log = []
stats, pstats = {}, {}
if args.payload_eval:
    pair.open = args.warm > 0
    pe = payload_eval(args.payload_eval, 0)
    print('payload step 0: ' + json.dumps(pe), flush=True)
    log.append({'step': 0, 'payload_eval': pe})
nxt = prepare(1)
for step in range(1, args.steps + 1):
    frac = step / args.steps
    for g, lr0 in zip(opt.param_groups, base_lr):
        g['lr'] = lr0 * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    competitive, rows, built, rounds = nxt
    pair.open = frac <= args.warm
    scale = 0.0 if pair.open else min(1.0, (frac - args.warm) / args.ramp) if args.ramp > 0 else 1.0
    rows, built, b, last, loss_for, total, sent, tally = forward(rows, built, rounds, scale)
    backward(competitive, loss_for, total)
    if (step - 1) % args.payload_every == 0 and not competitive:
        for g, a in payload_ce(rows, built, b, rounds, {x: last[x].detach() for x in SIDES}).items():
            o = pstats.setdefault(g, [0.0, 0.0, 0])
            for j in range(3):
                o[j] += a[j]
    del last
    if step % 20 == 0 or step <= 3:      # gradient norms of the link halves (summed over ranks, before clipping)
        gn = {x: [round(float(torch.norm(torch.stack([q.grad.norm() for q in mod.parameters() if q.grad is not None])))
                        if any(q.grad is not None for q in mod.parameters()) else 0.0, 5)
                  for mod in (pair.side[x].send, pair.side[x].recv, pair.side[x].gate)] for x in SIDES}
    if args.grad_check and ddp.MAIN:
        named = [(f'{x}.{n}', p) for x in SIDES for n, p in models[x].named_parameters() if p.requires_grad] + \
                [(f'link.{n}', p) for n, p in pair.side.named_parameters()]
        torch.save({n: p.grad.detach().cpu().clone() for n, p in named if p.grad is not None},
                   f'{args.grad_check}.{step}.pt')
    torch.nn.utils.clip_grad_norm_(ALL, 1.0)
    opt.step()
    pair.clear()
    if step < args.steps:
        nxt = prepare(step + 1)          # on the CPU while the GPU finishes this step
    for x in SIDES:
        for i, (h, c, n) in enumerate(tally[x].T.tolist()):
            ep, owners = rows[i]
            if not owners[x] or n == 0:
                continue
            k = ep['game'] + ('/adv' if owners[x] != BOTH and x not in owners[x] else '')
            a = stats.setdefault(k, [0.0, 0.0, 0])
            a[0] += h / n; a[1] += c / n; a[2] += 1
    if step % 20 == 0 or step <= 3:
        v = ddp.sum_tensor(torch.cat([total.detach()[None], sent])).tolist()
        merged = merge(ddp.gather(stats))
        pay = merge(ddp.gather(pstats))
        mem = max(ddp.gather(round(torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30, 1))) \
            if DEVICE.startswith('cuda') else 0
        line = {'step': step, 'loss': round(v[0], 4), 'rounds': rounds,
                'sent': {x: round(v[1 + j], 3) for j, x in enumerate(SIDES)},
                'size': {x: round(float(pair.side[x].log_size.exp()), 3) for x in SIDES},
                'games': {k: [round(a[0] / a[2], 3), round(a[1] / a[2], 3)] for k, a in sorted(merged.items())},
                'payload': {k: [round(a[0] / a[2], 3), round(a[1] / a[2], 3), a[2]] for k, a in sorted(pay.items())},
                'write': {x: round(float(pair.write_ms[x]) ** 0.5 / rms[x], 4) if x in pair.write_ms else 0.0
                          for x in SIDES},
                'gate_open': pair.open, 'cost': round(scale, 3), 'grad': gn,
                'time': round(time.time() - t0, 1), 'mem': mem}
        if step % 20 == 0:
            stats, pstats = {}, {}
            log.append(line)
        print(json.dumps(line), flush=True)
    if args.grad_check:
        continue
    if step % args.eval_every == 0 or step == args.steps or step == args.first_eval:
        import evaluate as E
        for x in SIDES:
            models[x].eval()
        res = {} if ONLY_QA else E.quick(pair, toks, test_games, n=args.quick_n, dialogue=args.dialogue > 0)
        if BENCH:
            style = 'solo' if args.solo else 'text' if args.text_pair else 'pair'
            res['qa_val'] = E.quick_qa(pair, VAL[:args.quick_n * 4], style)
        if args.payload_eval:
            res['payload'] = payload_eval(args.payload_eval, step)
        for x in SIDES:
            models[x].train()
        print(f'eval step {step}: ' + json.dumps(res), flush=True)
        log.append({'step': step, 'eval': res})
    if ddp.MAIN and (step % args.save_every == 0 or step == args.steps or step == args.halt):
        d = os.path.join(args.out, f'step_{step}')
        for x in SIDES:
            models[x].save_pretrained(os.path.join(d, x))
        torch.save({'link': pair.side.state_dict(), 'lr': lr, 'ls': ls, 'sizes': sizes, 'rms': rms, 'inbox': pair.inbox,
                    'soft': pair.soft,
                    'args': vars(args)},
                   os.path.join(d, 'link.pt'))
        json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'))
        print(f'saved {d}', flush=True)
    if step == args.halt:
        break
