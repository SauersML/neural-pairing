"""Demo transcripts: held-out episodes of chosen games, both models' text with the link and with it cut, and each
model's gate per token (1: over one half, the mean over the exchange blocks).

Usage: python demo.py --run RUN/step_N --replies OUT/replies.json --out OUT/demo.json [--games relay_word,steer] [--n 3]
"""
import argparse
import json

import games as G
from evaluate import load_pair, run_episodes, sent_fraction
from link import SIDES

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--games', default='relay_word,merge_sum,steer,lookup,long_relay,hide_seek,listen_steer')
ap.add_argument('--n', type=int, default=3)
ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()

pair, toks = load_pair(args.run)
games = G.Games(json.load(open(args.replies)), 'test', seed=args.seed)
out = []
for g in args.games.split(','):
    eps = [games.episode(g) for _ in range(args.n)]
    link, cut = run_episodes(pair, eps), run_episodes(pair, eps, mode='zero')
    for ep, a, z in zip(eps, link, cut):
        gates = {x: ''.join('1' if p > 0.5 else '0' for step in a['gates'][x] for p in step) for x in SIDES}
        d = dict(game=g, meta=ep['meta'], task=ep['task'], ok=a['ok'], ok_cut=z['ok'], gates=gates,
                 sent={x: round(sent_fraction([a], x), 3) for x in SIDES})
        if ep['mode'] == 'duplex':
            d.update(text=a['text'], text_cut=z['text'], gate_p=a['gates'])
        else:
            d.update(turns=a['turns'], turns_cut=z['turns'], system=ep['system'], gate_p=a['gates'],
                     gate_turn=a['gate_turn'])
        out.append(d)
        print(json.dumps(d, ensure_ascii=False)[:3000], flush=True)
json.dump(out, open(args.out, 'w'), indent=1, ensure_ascii=False)
