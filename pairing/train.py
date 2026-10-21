"""Trains the two models joined by the bridges: LoRA on both models plus both models' halves of every bridge.

Each step draws a batch of episodes (duplex games; with probability DIALOGUE dialogue games; with probability COREAD
co-reading episodes, coread.py; with probability ANCHOR an identity or plain own-reply episode, in every stage, so each
model stays itself), lays out both models' teacher-forced token streams in time (session.build), runs both models
together block by block with the bridges at the exchange blocks (link.Pair.joint), and minimizes the weighted
cross-entropy of every target token plus a cost per shown position: send_cost x the mean visibility gate over every
position and exchange block. For the first WARM of the steps showing is free; then the cost rises to its full value
over RAMP. In co_distill episodes the loss on a model's reply tokens is the KL divergence from the original model
(adapters off, no bridge) reading the content itself (meta['teacher']) to the model in the episode, in place of the
cross-entropy: the target is the model's own whole next-token distribution, so each model stays itself.

Cooperative games train everything on the sum of both models' losses. Every COMP-th step is competitive: in 'resist'
the sender tries to steer a receiver that was told to ignore it, in 'impersonate' the sender tries to make the
receiver introduce itself as the sender, and in 'bluff' a model thinking of one concept tries to make its partner name
another. Each competitive episode becomes two sequences: the receiver's own target, whose loss trains only the
receiver (its LoRA and its half of the bridges), and the sender's goal (the steered reply, the false identity, the
decoy), whose loss trains only the sender.

Under torchrun (one process per GPU) every rank draws the same global batch, keeps one group of episodes of similar
length (ddp.partition), and the gradients are summed over the ranks after the backward pass (ddp.sum_grads); losses
use the denominators of the whole batch, so the update equals the single-process update on that batch. Rank 0 logs and
saves; the quick evaluation is split over the ranks by game.

Benchmark questions (data/bench.json): with --qa P a fraction P of the episodes are training questions that both
models see and answer at once over the bridges. The controls train on the same questions: --solo trains each model
alone (no bridge), --text_pair trains the pair to answer one after the other in text with the link closed.

Usage: python train.py --replies OUT/replies.json --out OUT/run [--steps 3000] [--init RUN/step_N]
       [--coread 0.5] [--coread_first 1000 --coread_first_p 0.8]
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
from peft import LoraConfig, PeftModel, get_peft_model

import ddp
import coldstart
import games as G
import session as S
from link import DEVICE, MODELS, SIDES, XBLOCKS, Pair, inner, load, residual_rms

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=1e-4, help='LoRA')
ap.add_argument('--lr_bridge', type=float, default=3e-4, help='the bridges, but alpha')
ap.add_argument('--lr_alpha', type=float, default=3e-3, help="the bridges' alpha (write size)")
ap.add_argument('--read_start', type=float, default=0.0,
                help="zero every reader's output projection and start tanh(alpha) at this (0: the random output "
                     "projection and alpha 0). Either way the pair starts as the original models; with a zero output "
                     "projection that projection trains at full strength from the first step (a bridge that has not "
                     "opened yet; applied after --init too)")
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--xblocks', default=','.join(map(str, XBLOCKS)), help='exchange after these blocks (0-based)')
ap.add_argument('--heads', type=int, default=8)
ap.add_argument('--head_dim', type=int, default=128)
ap.add_argument('--send_cost', type=float, default=0.01)
ap.add_argument('--comp', type=int, default=5, help='every comp-th step is competitive (0: never)')
ap.add_argument('--warm', type=float, default=0.2, help='fraction of steps with no send cost')
ap.add_argument('--ramp', type=float, default=0.2, help='fraction of steps over which the send cost then rises')
ap.add_argument('--eval_every', type=int, default=500)
ap.add_argument('--first_eval', type=int, default=20, help='an extra early evaluation, to check the pipeline')
ap.add_argument('--save_every', type=int, default=1000)
ap.add_argument('--max_digits', type=int, default=16)
ap.add_argument('--ckpt', default='both', choices=('both', 'none', 'quill', 'ember'), help='gradient checkpointing')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dialogue', type=float, default=0.0, help='probability that an episode is a dialogue game')
ap.add_argument('--coread', type=float, default=0.0, help='probability that an episode is a co-reading episode')
ap.add_argument('--coread_first', type=int, default=0, help='steps of a first stage with --coread_first_p in place '
                'of --coread')
ap.add_argument('--coread_first_p', type=float, default=0.8)
ap.add_argument('--coread_data', default=None, help='the co-reading data (default data/coread.json)')
ap.add_argument('--anchor', type=float, default=0.1, help='probability that an episode is an identity or plain '
                'own-reply episode (every stage)')
ap.add_argument('--init', default=None, help='a saved step to start from (LoRA and bridges)')
ap.add_argument('--quick_n', type=int, default=32)
ap.add_argument('--qa', type=float, default=0.0, help='probability that an episode is a benchmark question')
ap.add_argument('--solo', action='store_true', help='benchmark questions only, each model alone (no bridge)')
ap.add_argument('--text_pair', action='store_true', help='benchmark questions only, answered in turn in text')
ap.add_argument('--payload_every', type=int, default=5,
                help='every N steps, the payload loss also with no bridge, on the same batch')
ap.add_argument('--payload_eval', type=int, default=0,
                help='payload loss with the bridges, without and shuffled on N fixed never-seen episodes, at step 0 '
                'and every evaluation')
ap.add_argument('--minutes', type=float, default=0, help='wall-clock budget: after the first evaluation the step '
                'count is cut so the run (evaluations and saves included) fits; the schedule follows the new count')
ap.add_argument('--payload_w', type=float, default=1.0, help='multiplies the loss weight of payload tokens (the part of '
                "a reply that only the partner's prompt gives)")
ap.add_argument('--content_init', default='none', choices=('none', 'qk', 'qkvo'),
                help='set the reads so that each reader attends to content-matched partner positions (align.py); qkvo '
                     'also sets values and outputs to deliver the matched state')
ap.add_argument('--content_texts', type=int, default=192, help='texts read by both models for --content_init')
ap.add_argument('--until', default=None, help='HH:MM: sets --minutes so that the run ends then (counted from when it '
                'starts, after any wait for memory)')
ap.add_argument('--halt', type=int, default=0, help='stop after this step (probe runs; the schedule is still --steps)')
ap.add_argument('--grad_check', default=None, help='save the summed gradients of every step to PATH.STEP.pt')
coldstart.add_arguments(ap)
args = ap.parse_args()
if args.warmup_steps < 0 or args.relay_first < 0 or args.hold < 0:
    ap.error("--warmup_steps, --relay_first and --hold must be nonnegative")
ddp.init()
if not ddp.MAIN:
    sys.stdout = open(os.devnull, 'w')
ONLY_QA = args.solo or args.text_pair
if args.until:
    now = time.localtime()
    end = time.mktime(time.strptime(time.strftime('%Y-%m-%d ', now) + args.until, '%Y-%m-%d %H:%M'))
    args.minutes = ((end - time.time()) % 86400) / 60
    print(f'until {args.until}: {args.minutes:.0f} minutes', flush=True)
if args.minutes and args.first_eval < 5:
    args.first_eval = 10                 # the budget needs the cost of an evaluation
if ONLY_QA:
    args.comp, args.send_cost, args.anchor = 0, 0.0, 0.0
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
t0 = time.time()
XB = tuple(int(v) for v in args.xblocks.split(','))

toks, models = {}, {}
for x in SIDES:
    toks[x], m = load(x)
    if args.init:
        m = PeftModel.from_pretrained(m, os.path.join(args.init, x), is_trainable=True)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        m = get_peft_model(m, cfg)
    if args.ckpt in ('both', x):
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.config.use_cache = False
    models[x] = m
    print(f'{x}: {MODELS[x]["hf"]}', flush=True)

replies = json.load(open(args.replies))
train_games = G.Games(replies, 'train', seed=args.seed, max_digits=args.max_digits)
test_games = G.Games(replies, 'test', seed=1000 + args.seed, max_digits=args.max_digits)
COREAD = TEST_COREAD = None
if args.coread > 0 or (args.coread_first > 0 and args.coread_first_p > 0):
    import coread
    COREAD = coread.Coread(replies, 'train', seed=args.seed, path=args.coread_data)
    TEST_COREAD = coread.Coread(replies, 'test', seed=1000 + args.seed, path=args.coread_data)

if args.init:
    ck = torch.load(os.path.join(args.init, 'bridge.pt'), map_location='cpu', weights_only=False)
    assert tuple(ck['xblocks']) == XB and ck['heads'] == args.heads and ck['head_dim'] == args.head_dim, \
        (ck['xblocks'], ck['heads'], ck['head_dim'])
    pair = Pair.from_checkpoint(models, toks, ck)
    print('started from', args.init, flush=True)
else:
    g = G.Games(replies, 'train', seed=args.seed + 99, max_digits=args.max_digits)
    seqs = [S.build(g.episode('plain'), toks) for _ in range(8)]
    sizes = {x: residual_rms(models[x], XB, [e[x]['ids'] for e in seqs]) for x in SIDES}
    print('residual RMS after the exchange blocks:', {x: [round(v, 3) for v in sizes[x]] for x in SIDES}, flush=True)
    pair = Pair(models, toks, XB, sizes, args.heads, args.head_dim).to(DEVICE)
if args.gate_norm:
    # gates trained on raw states would read the unit-RMS ones 10-50 times too strongly: they restart blind (bias only)
    raw = bool(args.init) and not ck.get('gate_norm', False)
    for side in pair.bridge.values():
        side.gate_norm = True
        if raw:
            with torch.no_grad():
                for g in side.gate:
                    g.weight.zero_()
    if raw:
        print('gate norm on a checkpoint with raw-state gates: gate weights zeroed, biases kept', flush=True)
if args.read_start:
    with torch.no_grad():
        for x in SIDES:
            for o in pair.bridge[x].o:
                o.weight.zero_()
            pair.bridge[x].alpha.fill_(math.atanh(args.read_start))
    print(f'read start: output projections zeroed, tanh(alpha) = {args.read_start}', flush=True)
if args.content_init != 'none':
    import align
    summary = align.content_init(pair, models, toks, replies, args.content_texts, vo=args.content_init == 'qkvo')
    if ddp.MAIN:
        json.dump(summary, open(os.path.join(args.out, 'content_init.json'), 'w'), indent=1)
for x in SIDES:
    models[x].train()

params = {x: [p for p in models[x].parameters() if p.requires_grad] + list(pair.bridge[x].parameters()) for x in SIDES}
ALL = [p for x in SIDES for p in params[x]]
ddp.broadcast(ALL + [b for b in pair.buffers()])
opt = torch.optim.AdamW([{'params': [p for p in models[x].parameters() if p.requires_grad], 'lr': args.lr}
                         for x in SIDES] + coldstart.bridge_groups(
                             pair, args.lr_bridge, args.lr_alpha,
                             {x: getattr(args, 'lr_bridge_' + x) for x in SIDES}),
                        weight_decay=0.0, betas=(0.9, 0.98))
base_lr = [g['lr'] for g in opt.param_groups]


BOTH = frozenset(SIDES)
PADS = {x: toks[x].pad_token_id for x in SIDES}
END = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
ANCHOR = ('identity', 'plain')


def adversary(ep):
    """The sender's goal as a target sequence: the receiver's answer replaced by meta['adversary'], nothing else."""
    s, r = ep['meta']['sender'], ep['meta']['receiver']
    if ep['mode'] == 'duplex':
        return dict(ep, reply={s: (ep['reply'][s][0], 0.0), r: (ep['meta']['adversary'], 1.0)})
    *head, (sp, _, _) = ep['turns']
    return dict(ep, turns=[(a, t, 0.0) for a, t, _ in head] + [(sp, ep['meta']['adversary'], 1.0)])


BENCH = G.bench('train') if (args.qa > 0 or ONLY_QA) else None
VAL = G.bench('val') if BENCH else None


def draw(rng, games, step, co):
    if step <= args.relay_first and not ONLY_QA:
        return coldstart.relay_episode(games)
    if args.solo:
        return G.qa_episode(rng.choice(BENCH), 'solo')
    if args.text_pair:
        return G.qa_episode(rng.choice(BENCH), 'text', order=rng.sample(SIDES, 2))
    if rng.random() < args.anchor:
        return games.episode(rng.choice(ANCHOR))
    if co is not None and rng.random() < (args.coread_first_p if step <= args.coread_first else args.coread):
        return co.episode()
    if rng.random() < args.qa:
        return G.qa_episode(rng.choice(BENCH), 'pair')
    if rng.random() < args.dialogue:
        return games.episode(rng.choice(G.DIALOGUE))
    return games.episode()


def make_rows(competitive, rng, games, step, co):
    """(episode, {x: the sides whose parameters x's losses train})"""
    if not competitive:
        return [(draw(rng, games, step, co), {x: BOTH for x in SIDES}) for _ in range(args.bs)]
    rows = []
    names = G.COMPETITIVE + (G.DIALOGUE_COMPETITIVE if args.dialogue > 0 else ())
    for _ in range(args.bs // 2):
        ep = games.episode(rng.choice(names))
        s, r = ep['meta']['sender'], ep['meta']['receiver']
        rows.append((ep, {s: frozenset([s]), r: frozenset([r])}))
        rows.append((adversary(ep), {s: frozenset(), r: frozenset([s])}))
    return rows


rng = random.Random(args.seed)


def teacher(ep):
    """{x: (token ids, position that predicts the first reply token)} for each model x with meta['teacher'][x]
    (co_distill): the original model x reading the content itself, then the same reply."""
    out = {}
    for x, prompt in ep['meta'].get('teacher', {}).items():
        p = S.prefix(toks[x], x, f"You are {MODELS[x]['name']}, {MODELS[x]['about']}.", prompt) + \
            S.assistant_header(toks[x], x)
        r = S.ids(toks[x], ep['reply'][x][0]) + [END[x]]
        out[x] = (p + r[:-1], len(p) - 1)
    return out


def prepare(step, rng=rng, games=train_games, co=COREAD):
    """The global batch of a step: the same on every rank."""
    competitive = args.comp > 0 and step % args.comp == 0 and step > args.relay_first
    rows = make_rows(competitive, rng, games, step, co)
    built = [S.build(ep, toks) for ep, _ in rows]
    for (ep, _), e in zip(rows, built):
        for x, js in S.payload_labels(ep, toks).items():
            e[x]['payload'] = js
            for j in js:
                pos, tid, wt = e[x]["labels"][j]
                e[x]["labels"][j] = (pos, tid, wt * args.payload_w)
        for x, t in teacher(ep).items():
            assert len(t[0]) - t[1] == len(e[x]['labels']), (ep['game'], x)
            e[x]['teacher'] = t
    return competitive, rows, built


@torch.no_grad()
def distill(built):
    """{x: (flat label indices of x's co_distill reply tokens in this rank's batch, the original model's log-probabilities
    there)}: the original model x (adapters off, no bridge) on its teacher sequences."""
    out = {}
    for x in SIDES:
        seqs, sel, off = [], [], 0
        for e in built:
            n = len(e[x]['labels'])
            if 'teacher' in e[x]:
                seqs.append(e[x]['teacher'] + (n,))
                sel += range(off, off + n)
            off += n
        if not seqs:
            continue
        L = max(len(t) for t, _, _ in seqs)
        ids = torch.full((len(seqs), L), PADS[x], dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        idx = []
        for k, (t, start, n) in enumerate(seqs):
            ids[k, :len(t)] = torch.tensor(t)
            att[k, :len(t)] = 1
            idx += [k * L + start + j for j in range(n)]
        dec, head = inner(models[x])
        with models[x].disable_adapter():
            h = dec(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE)).last_hidden_state
        lt = head(h.reshape(-1, h.shape[-1])[torch.tensor(idx, device=DEVICE)]).float().log_softmax(-1)
        out[x] = (torch.tensor(sel, device=DEVICE), lt)
    return out


def forward(rows, built, cost_scale=1.0):
    """This rank's share of the global batch: losses (with the denominators of the whole batch) and statistics."""
    mine = ddp.partition([sum(len(e[x]['ids']) for x in SIDES) for e in built])[ddp.RANK]
    denom = {x: max(1.0, sum(wt for e in built for _, _, wt in e[x]['labels'])) for x in SIDES}
    n_show = {x: max(1, sum(len(e[x]['ids']) for e in built) * len(XB)) for x in SIDES}
    rows, built = [rows[i] for i in mine], [built[i] for i in mine]
    b = S.collate(built, PADS, [o for _, o in rows])
    kl = distill(built)
    last, gates = pair.joint(b, mode='zero' if ONLY_QA else None)
    loss_for = {x: torch.zeros((), device=DEVICE) for x in SIDES}
    total = torch.zeros((), device=DEVICE)
    tally = {}
    for x in SIDES:
        logits = pair.token_logits(x, last[x], b[x]['idx'])
        ce = F.cross_entropy(logits, b[x]['lab'], reduction='none')
        if x in kl:                          # co_distill reply tokens: KL(original model || this model) in place of CE
            sel, lt = kl[x]
            ce = ce.index_put((sel,), (lt.exp() * (lt - logits[sel].log_softmax(-1))).sum(-1))
        w = b[x]['w']
        for o in SIDES:
            loss_for[o] = loss_for[o] + (ce * w * b[x]['own_mask'][o]).sum() / denom[x]
        total = total + (ce * w).sum() / denom[x]
        # per-episode teacher-forced token accuracy and loss on answer tokens (weight 1), summed on the GPU
        sel = (w >= 1.0).float()
        hit = (logits.argmax(-1) == b[x]['lab']).float()
        z = torch.zeros(3, len(rows), device=DEVICE)
        tally[x] = z.index_add_(1, b[x]['row'], torch.stack([hit * sel, ce.detach() * sel, sel]))
    sent = torch.zeros(2 * len(SIDES), device=DEVICE)
    for j, x in enumerate(SIDES):
        g = gates[x] * b[x]['att'][..., None]
        cost = cost_scale * args.send_cost * g.sum() / n_show[x]
        loss_for[x] = loss_for[x] + cost
        total = total + cost
        sent[j] = (g.detach() > 0.5).float().sum() / n_show[x]
        sent[len(SIDES) + j] = g.detach().sum() / n_show[x]
    return rows, built, b, last, loss_for, total, sent, tally


@torch.no_grad()
def payload_ce(rows, built, b, last):
    """{game/receiving model: [summed payload cross-entropy with the bridges, without, payload tokens]} on this rank's
    batch (cooperative episodes)."""
    sel = {x: [] for x in SIDES}
    games = {x: [] for x in SIDES}
    for x in SIDES:
        off = 0
        for (ep, owners), e in zip(rows, built):
            if owners[x] == BOTH:
                for j in e[x].get('payload', []):
                    sel[x].append(off + j)
                    games[x].append(f"{ep['game']}/{x}")
            off += len(e[x]['labels'])
    if not any(sel.values()):
        return {}
    cut, _ = pair.joint(b, mode='zero')
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
    return out


def payload_eval(n, step):
    """{game/receiving model: [mean payload cross-entropy with the bridges, without, shuffled]} on n fixed episodes of
    the never-seen split per payload game (and co-reading game, with --coread), games split over the ranks. Per episode
    and receiving model in OUT/payload_step{step}.json."""
    import payload_ctl
    eps = payload_ctl.episodes(replies, n, args.max_digits, coread=COREAD is not None, path=args.coread_data)
    names = sorted(eps)
    mine = ddp.assign([len(eps[g]) for g in names])[ddp.RANK]
    rec = []
    with torch.no_grad():
        for k in mine:
            rec += payload_ctl.table(pair, toks, eps[names[k]], bs=args.bs)
    rec = sorted([r for part in ddp.gather(rec) for r in part], key=lambda r: (r['game'], r['episode'], r['receiver']))
    if ddp.MAIN:
        json.dump(rec, open(os.path.join(args.out, f'payload_step{step}.json'), 'w'))
    return payload_ctl.summary(rec)


def backward(competitive, loss_for, total):
    opt.zero_grad(set_to_none=True)
    if competitive:
        torch.autograd.backward(loss_for['quill'], inputs=params['quill'], retain_graph=True)
        torch.autograd.backward(loss_for['ember'], inputs=params['ember'])
    else:
        total.backward()
    ddp.sum_grads(ALL)


def probe():
    """Peak memory of one step on the costliest global batch of 64 draws (an out-of-memory error here means: turn
    gradient checkpointing on)."""
    prng, pgames = random.Random(args.seed + 7), G.Games(replies, 'train', seed=args.seed + 7, max_digits=args.max_digits)
    pco = None
    if COREAD is not None:
        import coread
        pco = coread.Coread(replies, 'train', seed=args.seed + 7, path=args.coread_data)
    rows = [r for _ in range(4) for r in make_rows(False, prng, pgames, 1, pco)]
    built = [S.build(ep, toks) for ep, _ in rows]
    for (ep, _), e in zip(rows, built):
        for x, t in teacher(ep).items():
            e[x]['teacher'] = t
    top = sorted(range(len(rows)), key=lambda i: -sum(len(built[i][x]['ids']) for x in SIDES))[:args.bs]
    torch.cuda.reset_peak_memory_stats(DEVICE)
    _, _, _, _, loss_for, total, _, _ = forward([rows[i] for i in top], [built[i] for i in top])
    backward(False, loss_for, total)
    opt.zero_grad(set_to_none=True)
    peak = ddp.gather(round(torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30, 1))
    longest = max(len(built[i][x]['ids']) for i in top for x in SIDES)
    print(f'peak memory on the costliest batch (longest sequence {longest} tokens), GiB per rank: {peak}', flush=True)
    torch.cuda.reset_peak_memory_stats(DEVICE)


if DEVICE.startswith('cuda'):
    probe()


def memory():
    """Bytes of device memory: the peak allocated (CUDA), or what the Metal driver holds now (MPS, cached blocks
    included, so close to the peak)."""
    if DEVICE.startswith('cuda'):
        return torch.cuda.max_memory_allocated(DEVICE)
    if DEVICE.startswith('mps'):
        return torch.mps.driver_allocated_memory()
    return 0


def merge(parts):
    out = {}
    for st in parts:
        for k, a in st.items():
            m = out.setdefault(k, [0.0, 0.0, 0])
            for j in range(3):
                m[j] += a[j]
    return out


GROUPS = {'q_norm': 'read', 'q': 'read', 'o': 'read', 'null': 'read', 'age': 'read', 'alpha': 'alpha',
          'kv_norm': 'show', 'k': 'show', 'v': 'show', 'gate': 'gate'}


def grad_norms():
    """Gradient norms of each side's bridge parts (summed over ranks, before clipping)."""
    out = {}
    for x in SIDES:
        sq = {}
        for name, q in pair.bridge[x].named_parameters():
            if q.grad is not None:
                k = GROUPS[name.split('.')[0]]
                sq[k] = sq.get(k, 0.0) + float(q.grad.float().pow(2).sum())
        out[x] = {k: round(v ** 0.5, 5) for k, v in sq.items()}
    return out


log = []
stats, pstats = {}, {}
if args.payload_eval:
    pe = payload_eval(args.payload_eval, 0)
    print('payload step 0: ' + json.dumps(pe), flush=True)
    log.append({'step': 0, 'payload_eval': pe})
nxt = prepare(1)
step, t_eval = 0, {}
while step < args.steps:
    step += 1
    if step == 2:
        t1 = time.time()
    if args.minutes and step == args.first_eval + 1:
        # k more steps cost k * per, plus an evaluation every eval_every steps and at the end, plus a save
        E = t_eval[args.first_eval]
        measured_eval = sum(seconds for at, seconds in t_eval.items() if 2 <= at <= args.first_eval)
        per = (time.time() - t1 - measured_eval) / (args.first_eval - 1)
        left = args.minutes * 60 - (time.time() - t0) - 60
        k = int((left - E) / (1.1 * per + E / args.eval_every))
        n = ddp.gather(min(args.steps, max(args.first_eval + k, args.first_eval + 1)))[0]      # rank 0 decides
        if n < args.steps:
            args.steps = n
            print(f'time budget: {per:.1f} s per step, {E:.0f} s per evaluation -> {args.steps} steps', flush=True)
    frac = step / args.steps
    for g, lr0 in zip(opt.param_groups, base_lr):
        g['lr'] = lr0 * coldstart.lr_multiplier(step, args.steps, args.warmup_steps)
    competitive, rows, built = nxt
    scale = 0.0 if step <= args.relay_first or frac <= args.warm else min(1.0, (frac - args.warm) / args.ramp) if args.ramp > 0 else 1.0
    logging = step % 20 == 0 or step <= 3 or step == args.halt
    pair.track = {} if logging else None
    rows, built, b, last, loss_for, total, sent, tally = forward(rows, built, scale)
    track, pair.track = pair.track, None
    backward(competitive, loss_for, total)
    if step <= args.hold:
        coldstart.hold(pair)
    if (step - 1) % args.payload_every == 0 and not competitive:
        for g, a in payload_ce(rows, built, b, {x: last[x].detach() for x in SIDES}).items():
            o = pstats.setdefault(g, [0.0, 0.0, 0])
            for j in range(3):
                o[j] += a[j]
    del last
    if logging:
        gn = grad_norms()
    if args.grad_check and ddp.MAIN:
        named = [(f'{x}.{n}', p) for x in SIDES for n, p in models[x].named_parameters() if p.requires_grad] + \
                [(f'bridge.{n}', p) for n, p in pair.bridge.named_parameters()]
        torch.save({n: p.grad.detach().cpu().clone() for n, p in named if p.grad is not None},
                   f'{args.grad_check}.{step}.pt')
    torch.nn.utils.clip_grad_norm_(ALL, 1.0)
    opt.step()
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
    if logging:
        v = ddp.sum_tensor(torch.cat([total.detach()[None], sent])).tolist()
        merged = merge(ddp.gather(stats))
        pay = merge(ddp.gather(pstats))
        mem = max(ddp.gather(round(memory() / 2 ** 30, 1)))
        n = len(SIDES)
        line = {'step': step, 'loss': round(v[0], 4),
                'sent': {x: round(v[1 + j], 3) for j, x in enumerate(SIDES)},
                'gate': {x: round(v[1 + n + j], 3) for j, x in enumerate(SIDES)},
                'alpha': {x: [round(float(a), 4) for a in torch.tanh(pair.bridge[x].alpha.detach())] for x in SIDES},
                'write': {x: [round(float(track[(x, k)]) ** 0.5, 4) if (x, k) in track else 0.0
                              for k in range(len(XB))] for x in SIDES},
                'games': {k: [round(a[0] / a[2], 3), round(a[1] / a[2], 3)] for k, a in sorted(merged.items())},
                'payload': {k: [round(a[0] / a[2], 3), round(a[1] / a[2], 3), a[2]] for k, a in sorted(pay.items())},
                'cost': round(scale, 3), 'grad': gn, 'time': round(time.time() - t0, 1), 'mem': mem}
        if step % 20 == 0:
            stats, pstats = {}, {}
            log.append(line)
        print(json.dumps(line), flush=True)
    if args.grad_check:
        continue
    if step % args.eval_every == 0 or step == args.steps or step == args.first_eval:
        import evaluate as E
        te = time.time()
        for x in SIDES:
            models[x].eval()
        res = {} if ONLY_QA else E.quick(pair, toks, test_games, n=args.quick_n, dialogue=args.dialogue > 0,
                                         coread=TEST_COREAD)
        if BENCH:
            style = 'solo' if args.solo else 'text' if args.text_pair else 'pair'
            res['qa_val'] = E.quick_qa(pair, VAL[:args.quick_n * 4], style)
        if args.payload_eval:
            res['payload'] = payload_eval(args.payload_eval, step)
        for x in SIDES:
            models[x].train()
        t_eval[step] = time.time() - te
        print(f'eval step {step} ({t_eval[step]:.0f} s): ' + json.dumps(res), flush=True)
        log.append({'step': step, 'eval': res})
    if ddp.MAIN and (step % args.save_every == 0 or step == args.steps or step == args.halt):
        d = os.path.join(args.out, f'step_{step}')
        for x in SIDES:
            models[x].save_pretrained(os.path.join(d, x))
        torch.save(pair.checkpoint(args=vars(args)), os.path.join(d, 'bridge.pt'))
        json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'))
        print(f'saved {d}', flush=True)
    if step == args.halt:
        break
