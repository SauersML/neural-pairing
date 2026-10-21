"""Which side of the bridges is the bottleneck? Decodes a payload with a ridge classifier from what one model shows at
each exchange block, and from what the other model's bridge writes.

merge_sum: each model's own number (50 values) from its shown values (gate x V) at the number's last token in its own
prompt ('shows'), and from the partner's write into its last prompt token, which reads the whole of that prompt
('receives').
relay_digits: the first digit (10 values) in the same two places (sender shows, receiver receives).
Each read is a teacher-forced joint pass with the recording hook (link.Pair.record), no gradients. The decoder is fit on
75% of the episodes and scored on the other 25%. A payload the sender shows that the receiver's write does not carry
puts the fault on the reading side; one the sender does not show, on the showing side.

Usage: python decode.py --replies OUT/replies.json [--run RUN/step_N | --fresh] [--n 512]
"""
import argparse
import json

import numpy as np
import scipy.linalg
import torch

import games as G
import session as S
from link import DEVICE, SIDES, XBLOCKS, Pair, load, partner

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--run', default=None)
ap.add_argument('--fresh', action='store_true', help='the original models and new bridges with alpha = 1 (step 0)')
ap.add_argument('--n', type=int, default=512)
ap.add_argument('--bs', type=int, default=16)
args = ap.parse_args()

if args.fresh:
    toks, models = {}, {}
    for x in SIDES:
        toks[x], models[x] = load(x)
    torch.manual_seed(0)
    pair = Pair(models, toks, XBLOCKS).to(DEVICE).eval()
    for x in SIDES:
        pair.bridge[x].alpha.data.fill_(1.0)       # so the writes are not zero
else:
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
games = G.Games(json.load(open(args.replies)), 'train', seed=4242)
pads = {x: toks[x].pad_token_id for x in SIDES}


def value_position(ep, x, value):
    """The position in x's prompt of the token that holds the last character of the first occurrence of value in x's
    task."""
    sys_, task = ep['system'][x], ep['task'][x]
    s = f'<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n'
    end = s.index(task) + task.index(value) + len(value) - 1
    enc = toks[x](s, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == S.prefix(toks[x], x, sys_, task)
    return max(j for j, (a, b) in enumerate(enc.offset_mapping) if a <= end < b)


def last_prompt(ep, x):
    """The position of x's last prompt token (the end of its assistant header, read at tick 0)."""
    return len(S.prefix(toks[x], x, ep['system'][x], ep['task'][x])) + len(S.assistant_header(toks[x], x)) - 1


@torch.no_grad()
def reads(eps, places):
    """Per episode: {(x, block): (shown at places[i][x][0], write at places[i][x][1])} as float32 numpy vectors."""
    out = []
    for a, z in S.batches(len(eps), args.bs):
        chunk = eps[a:z]
        b = S.collate([S.build(e, toks) for e in chunk], pads)
        pair.record = {}
        pair.joint(b)
        rec, pair.record = pair.record, None
        for i in range(len(chunk)):
            d = {}
            for (x, blk), (r,) in rec.items():
                sp, wp = places[a + i][x]
                d[(x, blk)] = (r['shown'][i, sp].numpy(), r['write'][i, wp].numpy())
            out.append(d)
    return out


def ridge(X, y, classes):
    """Never-seen accuracy of a ridge classifier (one-hot targets), the penalty picked on the training part."""
    n = len(y)
    cut = int(0.75 * n)
    X = (X - X[:cut].mean(0)) / (X[:cut].std(0) + 1e-6)
    X = np.concatenate([X, np.ones((n, 1))], 1).astype(np.float64)
    Y = np.eye(classes)[y]

    def fit(m, lam):
        return scipy.linalg.solve(X[:m].T @ X[:m] + lam * np.eye(X.shape[1]), X[:m].T @ Y[:m], assume_a='pos')
    f = int(0.8 * cut)                           # penalty chosen on a split of the training part
    best = max((float(((X[f:cut] @ fit(f, lam)).argmax(1) == y[f:cut]).mean()), lam)
               for lam in (0.1, 1.0, 10.0, 100.0, 1000.0))
    return round(float(((X[cut:] @ fit(cut, best[1])).argmax(1) == y[cut:]).mean()), 3)


res = {}
eps = [games.episode('merge_sum') for _ in range(args.n)]
places = [{x: (value_position(ep, x, str(ep['meta']['values'][x])), last_prompt(ep, x)) for x in SIDES} for ep in eps]
rd = reads(eps, places)
for blk in pair.xblocks:
    for x in SIDES:
        y = np.array([ep['meta']['values'][x] for ep in eps])
        res[f'merge_sum/{x} shows its number, block {blk}'] = ridge(np.stack([r[(x, blk)][0] for r in rd]), y, 50)
        res[f'merge_sum/{partner(x)} receives it, block {blk}'] = \
            ridge(np.stack([r[(partner(x), blk)][1] for r in rd]), y, 50)
eps = [games.episode('relay_digits') for _ in range(args.n)]
places = []
for ep in eps:
    s, r = ep['meta']['sender'], ep['meta']['receiver']
    places.append({s: (value_position(ep, s, ep['meta']['value'].split()[0]), last_prompt(ep, s)),
                   r: (0, last_prompt(ep, r))})
rd = reads(eps, places)
y = np.array([int(ep['meta']['value'].split()[0]) for ep in eps])
for blk in pair.xblocks:
    res[f'relay_digits/sender shows the first digit, block {blk}'] = \
        ridge(np.stack([r[(ep['meta']['sender'], blk)][0] for ep, r in zip(eps, rd)]), y, 10)
    res[f'relay_digits/receiver receives it, block {blk}'] = \
        ridge(np.stack([r[(ep['meta']['receiver'], blk)][1] for ep, r in zip(eps, rd)]), y, 10)
print(json.dumps({'run': args.run or 'fresh', 'never_seen_accuracy': res, 'chance': {'merge_sum': 0.02, 'relay_digits': 0.1}}),
      flush=True)
