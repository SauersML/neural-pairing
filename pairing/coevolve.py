"""Co-evolution of the pair's code: every model from one checkpoint against its partner from another.

For checkpoints c_0..c_{n-1} of one run (in training order), each cell (i, j) pairs Quill as of c_i (its LoRA adapter
and its half of the bridges: what it reads with and what it shows) with Ember as of c_j. On fixed never-seen episodes
(payload_ctl.episodes: the payload games and the co-reading games) it measures the payload cross-entropy with the
bridges and with them cut; the gain, cut minus link, is how many nats per payload token the link gives the receiver.
Cut depends on the receiver's own adapter only, so each cell's gain is that pairing's link alone.

If the pair builds a code of its own that keeps changing, the gain is highest on the diagonal (partners of the same
moment) and falls off it; a fixed, text-like code would give the same gain across a row.

Usage: NL_DEVICE=mps python coevolve.py --runs RUN/step_200,RUN/step_400,... --replies R.json --out FILE.json
       [--n 6] [--bs 8] [--games co_qa,co_hop,...]
"""
import argparse
import json
import os

import torch
from peft import PeftModel

import payload_ctl
from link import DEVICE, SIDES, Pair, load

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True, help='comma-separated step directories, in training order')
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=6, help='episodes per game')
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--games', default=None, help='comma-separated games (default: all payload and co-reading games)')
ap.add_argument('--coread_data', default=None)
args = ap.parse_args()

runs = args.runs.split(',')
cks = [torch.load(os.path.join(r, 'bridge.pt'), weights_only=False, map_location='cpu') for r in runs]
toks, models = {}, {}
for x in SIDES:
    toks[x], m = load(x)
    models[x] = PeftModel.from_pretrained(m, os.path.join(runs[0], x), adapter_name='c0')
    for i, r in enumerate(runs[1:], 1):
        models[x].load_adapter(os.path.join(r, x), adapter_name=f'c{i}')
    models[x].eval()
pair = Pair.from_checkpoint(models, toks, cks[0]).eval()
side_state = [{x: {k[len(x) + 1:]: v for k, v in ck['bridge'].items() if k.startswith(x + '.')} for x in SIDES}
              for ck in cks]

eps = payload_ctl.episodes(json.load(open(args.replies)), args.n, coread=True, path=args.coread_data)
if args.games:
    eps = {g: eps[g] for g in args.games.split(',')}
cells = {}
for i in range(len(runs)):
    for j in range(len(runs)):
        at = {'quill': i, 'ember': j}
        for x in SIDES:
            models[x].set_adapter(f'c{at[x]}')
            pair.bridge[x].load_state_dict(side_state[at[x]][x], strict=False)
        rec = []
        for g in eps:
            rec += payload_ctl.table(pair, toks, eps[g], args.bs, conds=(('link', None), ('cut', 'zero')))
        cell = {}
        for x in SIDES + ('all',):
            rs = [r for r in rec if x == 'all' or r['receiver'] == x]
            n = sum(r['n_tokens'] for r in rs)
            cell[x] = dict(gain=round(sum((r['cut_ce'] - r['link_ce']) * r['n_tokens'] for r in rs) / max(n, 1), 4),
                           link=round(sum(r['link_ce'] * r['n_tokens'] for r in rs) / max(n, 1), 4),
                           cut=round(sum(r['cut_ce'] * r['n_tokens'] for r in rs) / max(n, 1), 4), tokens=n)
        cells[f'{i},{j}'] = dict(cell, records=rec)
        print(f'quill c{i} x ember c{j}:', json.dumps({x: cell[x]['gain'] for x in cell}), flush=True)
        json.dump(dict(runs=runs, cells=cells), open(args.out, 'w'))
print('wrote', args.out, flush=True)
