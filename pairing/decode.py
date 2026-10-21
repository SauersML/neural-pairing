"""Which side of the link is the bottleneck? Decodes the payload from the sender's messages with a ridge classifier.

merge_sum: the sender's own number (50 values) from its message at the tick that reads the number's last token.
relay_digits: the first digit (10 values) from the sender's messages at ticks 0-3.
Each read is a teacher-forced coupled forward (gates open), no gradients. The decoder is fit on 75% of the episodes and
scored on the other 25%. A decodable payload with no drop in the receiver's payload loss puts the fault on the
receiving side; an undecodable one, on the sending side.

Usage: python decode.py --replies OUT/replies.json [--run RUN/step_N | --fresh] [--n 512]
"""
import argparse
import json
import os

import torch

import games as G
import session as S
from link import DEVICE, SIDES, Pair, find_layers, load

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--run', default=None)
ap.add_argument('--fresh', action='store_true', help='the original models and a newly made link (step 0)')
ap.add_argument('--n', type=int, default=512)
ap.add_argument('--bs', type=int, default=32)
args = ap.parse_args()

if args.fresh:
    toks, models, lr, ls = {}, {}, {}, {}
    for x in SIDES:
        toks[x], models[x] = load(x)
        n = len(find_layers(models[x]))
        lr[x], ls[x] = round(0.3 * n), round(0.67 * n)
    torch.manual_seed(0)
    pair = Pair(models, toks, lr, ls, {x: 1.0 for x in SIDES}).to(DEVICE).eval()
else:
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
pair.open = True
games = G.Games(json.load(open(args.replies)), 'train', seed=4242)
pads = {x: toks[x].pad_token_id for x in SIDES}


def own_tick(ep, x):
    """The tick at which x reads the last token of its own number in its merge_sum reply."""
    text = ep['reply'][x][0]
    a = str(ep['meta']['values'][x])
    i = text.find('Mine is ' + a) + len('Mine is ')
    enc = toks[x](text, add_special_tokens=False, return_offsets_mapping=True)
    last = max(j for j, (s, e) in enumerate(enc.offset_mapping) if s < i + len(a) and e > i)
    return last + 1                             # reply token j is read at tick j + 1


@torch.no_grad()
def messages(eps):
    out = []
    for k in range(0, len(eps), args.bs):
        chunk = eps[k:k + args.bs]
        b = S.collate([S.build(e, toks) for e in chunk], pads, soft=pair.soft)
        _, msgs = pair.coupled(b, rounds=2)
        pair.clear()
        out += [{x: msgs[x][0][i].float().cpu() for x in SIDES} for i in range(len(chunk))]
    return out


def ridge(X, y, classes):
    """Held-out accuracy of a ridge classifier (one-hot targets), the penalty picked on the training part."""
    n = len(y)
    cut = int(0.75 * n)
    X = (X - X[:cut].mean(0)) / (X[:cut].std(0) + 1e-6)
    X = torch.cat([X, torch.ones(n, 1)], 1).double()
    Y = torch.nn.functional.one_hot(y, classes).double()
    best = None
    for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
        f = int(0.8 * cut)                      # penalty chosen on a split of the training part
        W = torch.linalg.solve(X[:f].T @ X[:f] + lam * torch.eye(X.shape[1], dtype=X.dtype), X[:f].T @ Y[:f])
        acc = float(((X[f:cut] @ W).argmax(1) == y[f:cut]).float().mean())
        if best is None or acc > best[0]:
            best = (acc, lam)
    W = torch.linalg.solve(X[:cut].T @ X[:cut] + best[1] * torch.eye(X.shape[1], dtype=X.dtype), X[:cut].T @ Y[:cut])
    return round(float(((X[cut:] @ W).argmax(1) == y[cut:]).float().mean()), 3)


res = {}
eps = [games.episode('merge_sum') for _ in range(args.n)]
ms = messages(eps)
for x in SIDES:
    X = torch.stack([m[x][own_tick(ep, x)] for ep, m in zip(eps, ms)])
    y = torch.tensor([ep['meta']['values'][x] for ep in eps])
    res[f'merge_sum/{x} sends its number'] = ridge(X, y, 50)
eps = [games.episode('relay_digits') for _ in range(args.n)]
ms = messages(eps)
y = torch.tensor([int(ep['meta']['value'].split()[0]) for ep in eps])
for t in range(4):
    X = torch.stack([m[ep['meta']['sender']][t] for ep, m in zip(eps, ms)])
    res[f'relay_digits/first digit at tick {t}'] = ridge(X, y, 10)
print(json.dumps({'run': args.run or 'fresh', 'held_out_accuracy': res, 'chance': {'merge_sum': 0.02, 'relay_digits': 0.1}}),
      flush=True)
