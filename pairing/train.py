"""Trains the two linked models on the games: LoRA on both models plus both halves of the link.

Each step draws a batch of episodes, reads both models on their teacher-forced text with 2 or 3 Gauss-Seidel rounds
(link.Pair.coupled), and minimizes the weighted cross-entropy of every target token plus a small cost per message sent.
Cooperative games train everything on the sum of both models' losses. Every COMP-th step is competitive: in 'resist'
the sender tries to steer a receiver that was told to ignore it, and in 'impersonate' the sender tries to make the
receiver introduce itself as the sender. Each competitive episode becomes two sequences: the receiver's own target,
whose loss trains only the receiver (its LoRA and its half of the link), and the sender's goal (the steered reply or
the false identity), whose loss trains only the sender.

Usage: python train.py --replies OUT/replies.json --out OUT/run [--steps 3000]
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import games as G
from link import DEVICE, MODELS, SIDES, Pair, find_layers, load, partner

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lr_link', type=float, default=5e-4)
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--send_cost', type=float, default=0.01)
ap.add_argument('--comp', type=int, default=5, help='every comp-th step is competitive (0: never)')
ap.add_argument('--p3', type=float, default=0.3, help='probability of 3 rounds instead of 2')
ap.add_argument('--lr_frac', type=float, default=0.3, help='receive layer, fraction of depth')
ap.add_argument('--ls_frac', type=float, default=0.67, help='send layer, fraction of depth')
ap.add_argument('--size', type=float, default=0.3, help='initial write size, in units of the residual RMS')
ap.add_argument('--eval_every', type=int, default=500)
ap.add_argument('--first_eval', type=int, default=20)
ap.add_argument('--save_every', type=int, default=1000)
ap.add_argument('--max_digits', type=int, default=16)
ap.add_argument('--no_ckpt', action='store_true')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
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
    if not args.no_ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.config.use_cache = False
    models[x] = m
    print(f'{x}: {MODELS[x]["hf"]}, {n} blocks, receive after block {lr[x]}, send from block {ls[x]}', flush=True)

replies = json.load(open(args.replies))
train_games = G.Games(replies, 'train', seed=args.seed, max_digits=args.max_digits)
test_games = G.Games(replies, 'test', seed=1000 + args.seed, max_digits=args.max_digits)


@torch.no_grad()
def residual_rms(x):
    """Median per-token RMS of the residual after block lr[x] on a few training prompts."""
    tok, model = toks[x], models[x]
    eps = [train_games.episode('plain') for _ in range(8)]
    vals = []
    h = {}
    hook = find_layers(model)[lr[x]].register_forward_hook(lambda mod, i, o: h.update(v=o[0] if isinstance(o, tuple) else o))
    for ep in eps:
        p, r, _ = G.encode(tok, ep[x])
        model(input_ids=torch.tensor([p + r], device=DEVICE))
        vals.append(h['v'][0, 1:].float().pow(2).mean(-1).sqrt())
    hook.remove()
    return float(torch.cat(vals).median())


sizes = {x: args.size * residual_rms(x) for x in SIDES}
print('initial write size per element:', {x: round(v, 3) for x, v in sizes.items()}, flush=True)
pair = Pair(models, toks, lr, ls, sizes).to(DEVICE)
for x in SIDES:
    models[x].train()

params = {x: [p for p in models[x].parameters() if p.requires_grad] + list(pair.side[x].parameters()) for x in SIDES}
opt = torch.optim.AdamW([{'params': [p for p in models[x].parameters() if p.requires_grad], 'lr': args.lr}
                         for x in SIDES] + [{'params': list(pair.side.parameters()), 'lr': args.lr_link}],
                        weight_decay=0.0, betas=(0.9, 0.98))
base_lr = [g['lr'] for g in opt.param_groups]


def collate(rows):
    """rows: list of (episode, {x: owners}) -> batch dict per side, with flat label indices and loss owners.

    Model x's ticks are its positions start .. start + n - 1, where start is its last prompt token and n the length of
    its reply: position start + i reads reply token i - 1 (the prompt's last token at i = 0) and predicts reply token i."""
    b = {}
    for x in SIDES:
        enc = [G.encode(toks[x], ep[x]) for ep, _ in rows]
        L = max(len(p) + len(r) for p, r, _ in enc)
        pad = toks[x].pad_token_id
        ids = torch.full((len(rows), L), pad, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        idx, lab, w, own = [], [], [], []
        for i, ((p, r, wt), (_, owners)) in enumerate(zip(enc, rows)):
            seq = p + r
            ids[i, :len(seq)] = torch.tensor(seq)
            att[i, :len(seq)] = 1
            for j, t in enumerate(r):
                idx.append(i * L + len(p) - 1 + j)
                lab.append(t)
                w.append(wt)
                own.append(owners[x])
        b[x] = dict(ids=ids.to(DEVICE), att=att.to(DEVICE),
                    start=torch.tensor([len(p) - 1 for p, _, _ in enc], device=DEVICE),
                    n=torch.tensor([len(r) for _, r, _ in enc], device=DEVICE),
                    idx=torch.tensor(idx, device=DEVICE), lab=torch.tensor(lab, device=DEVICE),
                    w=torch.tensor(w, device=DEVICE, dtype=torch.float),
                    own=own, row=torch.tensor([k // L for k in idx], device=DEVICE))
    return b


BOTH = frozenset(SIDES)


def make_rows(competitive):
    if not competitive:
        return [(train_games.episode(), {x: BOTH for x in SIDES}) for _ in range(args.bs)]
    rows = []
    for _ in range(args.bs // 2):
        ep = train_games.episode(random.choice(G.COMPETITIVE))
        s, r = ep['meta']['sender'], ep['meta']['receiver']
        rows.append((ep, {s: frozenset([s]), r: frozenset([r])}))
        adv = dict(ep, **{r: dict(ep[r], target=ep['meta']['adversary'])})
        adv[s] = dict(ep[s], weight=0.0)
        rows.append((adv, {s: frozenset(), r: frozenset([s])}))
    return rows


log = []
stats = {}
for step in range(1, args.steps + 1):
    frac = step / args.steps
    for g, lr0 in zip(opt.param_groups, base_lr):
        g['lr'] = lr0 * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    competitive = args.comp > 0 and step % args.comp == 0
    rows = make_rows(competitive)
    b = collate(rows)
    rounds = 3 if random.random() < args.p3 else 2
    last, msgs = pair.coupled(b, rounds=rounds)
    loss_for = {x: torch.zeros((), device=DEVICE) for x in SIDES}
    total = torch.zeros((), device=DEVICE)
    for x in SIDES:
        logits = pair.token_logits(x, last[x], b[x]['idx'])
        ce = F.cross_entropy(logits, b[x]['lab'], reduction='none')
        w = b[x]['w']
        denom = w.sum().clamp_min(1.0)
        for o in SIDES:
            mask = torch.tensor([o in own for own in b[x]['own']], device=DEVICE, dtype=torch.float)
            loss_for[o] = loss_for[o] + (ce * w * mask).sum() / denom
        total = total + (ce * w).sum() / denom
        # per-game teacher-forced token accuracy on answer tokens (weight 1)
        hit = (logits.argmax(-1) == b[x]['lab']).float()
        for i, (ep, owners) in enumerate(rows):
            if not owners[x] or ep[x]['weight'] < 1.0:
                continue
            sel = b[x]['row'] == i
            k = ep['game'] + ('/adv' if owners[x] == frozenset([partner(x)]) else '')
            a = stats.setdefault(k, [0.0, 0.0, 0])
            a[0] += float(hit[sel].mean()); a[1] += float(ce[sel].detach().mean()); a[2] += 1
    sent = {}
    for x in SIDES:
        _, g, p, valid = msgs[x]
        cost = args.send_cost * p.sum() / valid.sum().clamp_min(1)
        loss_for[x] = loss_for[x] + cost
        total = total + cost
        sent[x] = float((g.detach() > 0.5).float().sum() / valid.sum().clamp_min(1))
    opt.zero_grad(set_to_none=True)
    if competitive:
        torch.autograd.backward(loss_for['quill'], inputs=params['quill'], retain_graph=True)
        torch.autograd.backward(loss_for['ember'], inputs=params['ember'])
    else:
        total.backward()
    torch.nn.utils.clip_grad_norm_([p for x in SIDES for p in params[x]], 1.0)
    opt.step()
    pair.clear()
    if step % 20 == 0:
        line = {'step': step, 'loss': round(float(total), 4), 'rounds': rounds, 'sent': {x: round(v, 3) for x, v in sent.items()},
                'size': {x: round(float(pair.side[x].log_size.exp()), 3) for x in SIDES},
                'games': {k: [round(v[0] / v[2], 3), round(v[1] / v[2], 3)] for k, v in sorted(stats.items())},
                'time': round(time.time() - t0)}
        stats = {}
        log.append(line)
        print(json.dumps(line), flush=True)
    if step % args.eval_every == 0 or step == args.steps or step == args.first_eval:
        import evaluate as E
        for x in SIDES:
            models[x].eval()
        res = E.quick(pair, toks, test_games, n=32)
        for x in SIDES:
            models[x].train()
        print(f'eval step {step}: ' + json.dumps(res), flush=True)
        log.append({'step': step, 'eval': res})
    if step % args.save_every == 0 or step == args.steps:
        d = os.path.join(args.out, f'step_{step}')
        for x in SIDES:
            models[x].save_pretrained(os.path.join(d, x))
        torch.save({'link': pair.side.state_dict(), 'lr': lr, 'ls': ls, 'sizes': sizes, 'args': vars(args)},
                   os.path.join(d, 'link.pt'))
        json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'))
        print(f'saved {d}', flush=True)
