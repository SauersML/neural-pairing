"""Per game, what a sym run's notes carry. On n fresh episodes of each of the run's games: the follower's mean say NLL
with its own notes (on), with the notes cut (cut) and with another episode of the same game's notes (swap); the swap
gap (swap minus on: only notes that tell episodes of a game apart win it) and the cut gap (cut minus on), each with a
95% bootstrap interval over episodes; and the notes the writers wrote: how many, how many different (as decoded
text), and the share of their characters that are letters. Trail is also given as note_stats shows it in training
(the explorer's notes in the first 2 episodes' rooms, by object). Each checkpoint of the run folder is scored
(trained.pt, gate.pt: the weights when privacy began).
Usage: python swarm/sym_games.py --run RUN [--ckpts trained.pt,gate.pt --n 48] --out OUT_JSON"""
import argparse
import json
from pathlib import Path
import random
import sys
import unicodedata

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402


def interval(xs, n=1000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def notes_of(net, device, eps, bs=50):
    """Per episode the decoded hard notes (every writer's, m tokens each, in bank order)."""
    net.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(eps), bs):
            chunk = eps[s0:s0 + bs]
            bank = net.run(chunk, device, need_loss=False)[2]
            for b in range(len(chunk)):
                if bank.get('ids') is None:
                    out.append([])
                    continue
                ids = bank['ids'][0][b][bank['place'][b] >= 0].tolist()
                out.append([net.tok.decode(ids[g:g + net.m]) for g in range(0, len(ids), net.m)])
    net.train()
    return out


def score(net, device, games, n, seed=777, bs=50):
    rng = random.Random(seed)
    eps = [CG.episode(g, rng) for g in games for _ in range(n)]
    with torch.no_grad():
        vals = {m: C.dev_values(net, device, eps, bs, m) for m in ('on', 'off', 'swap')}
    notes = notes_of(net, device, eps, bs)
    out = {}
    for g in list(games) + ['all']:
        rows = [i for i, ep in enumerate(eps) if (g == 'all' or ep['game'] == g) and vals['on'][i] is not None]
        if not rows:
            continue
        swap = [vals['swap'][i] - vals['on'][i] for i in rows]
        cut = [vals['off'][i] - vals['on'][i] for i in rows]
        written = [t for i, ep in enumerate(eps) if g == 'all' or ep['game'] == g for t in notes[i]]
        chars = [c for t in written for c in t if not c.isspace()]
        out[g] = dict(episodes=len(rows), on=round(sum(vals['on'][i] for i in rows) / len(rows), 4),
                      cut=round(sum(vals['off'][i] for i in rows) / len(rows), 4),
                      swap=round(sum(vals['swap'][i] for i in rows) / len(rows), 4),
                      gap_swap=round(sum(swap) / len(swap), 4), gap_swap_ci=interval(swap),
                      gap_cut=round(sum(cut) / len(cut), 4), gap_cut_ci=interval(cut),
                      notes=len(written), distinct=len(set(written)),
                      letters_text=round(sum(unicodedata.category(c).startswith('L') for c in chars)
                                         / max(1, len(chars)), 4))
    trail = [ep for ep in eps if ep['game'] == 'trail']
    if trail:
        ns = C.note_stats(net, device, trail)
        out['trail_probe'] = dict(examples=ns['examples'], distinct=ns['distinct'], of=len(ns['examples']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a crumbs.py sym run folder (args.json and its checkpoints)')
    ap.add_argument('--ckpts', default='trained.pt,gate.pt')
    ap.add_argument('--n', type=int, default=48, help='episodes per game')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    run = Path(args.run)
    games = C.expand_games(json.loads((run / 'args.json').read_text())['games'], {})[0]
    res = dict(run=str(run), n=args.n, games=list(games), ckpts={})
    for ck in args.ckpts.split(','):
        if not (run / ck).exists():
            continue
        net = C.load(run, args.device, ck)
        res['ckpts'][ck] = score(net, args.device, games, args.n)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        show = ('gap_swap', 'gap_swap_ci', 'gap_cut', 'notes', 'distinct', 'letters_text')
        for g, r in res['ckpts'][ck].items():
            if g != 'trail_probe':
                print(run.name, ck, g, json.dumps({k: r[k] for k in show}), flush=True)
        probe = res['ckpts'][ck].get('trail_probe')
        print(run.name, ck, 'trail_probe', json.dumps(probe, ensure_ascii=False), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
