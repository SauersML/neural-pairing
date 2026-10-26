"""The training questions (data/bench.json, train split) that exactly one of the original models answers right, scored
as bench_eval.py scores them (each model's most likely letter after its own 'The answer is', alone, adapters off).
Writes {'quill': [...], 'ember': [...]}, the indices of the questions only that model gets right, with the counts, for
train.py --qa_focus.

Usage: python qa_split.py --run RUN/step_N --out FILE (the step supplies the pair's tokenizers and bridges; its adapters
are not used)
"""
import argparse
import json
import os

import torch

import games as G
from bench_eval import letter_probs
from link import SIDES, Pair, load, partner

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='first N training questions (smoke tests)')
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.run, 'bridge.pt'), weights_only=False)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], models[x] = load(x)
        models[x].eval()
    pair = Pair.from_checkpoint(models, toks, ck).eval()
    items = G.bench('train')
    if args.limit:
        items = items[:args.limit]
    P = letter_probs(pair, items, 'solo', bs=args.bs)
    right = {x: [int(p.argmax()) == it['answer'] for p, it in zip(P[x], items)] for x in SIDES}
    out = {x: [i for i in range(len(items)) if right[x][i] and not right[partner(x)][i]] for x in SIDES}
    out['n'] = len(items)
    out['right'] = {x: sum(right[x]) for x in SIDES}
    print({x: len(out[x]) for x in SIDES}, 'of', len(items), 'right', out['right'], flush=True)
    json.dump(out, open(args.out, 'w'))
