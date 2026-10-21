"""Raw transcripts of the pair: n never-seen episodes of every scored co-reading game (play_tasks.SCORED) and every
payload game, generated greedily three ways on the same episodes: with the bridges, without them ('cut'), and shuffled
(each episode reads the partner of another episode of the same game). Per episode: each side's system and user prompt,
its target, its text under each condition with the scores, and per side one line of gate means per tick (the prompt's
tick first). Writes FILE.txt and FILE.json.

Usage: python transcripts.py --run RUN/step_N --replies REPLIES.json --out FILE.txt [--n 3] [--device mps]
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=3, help='episodes per game (at least 2, so a shuffled partner is another one)')
ap.add_argument('--device', default='mps')
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--coread_data', default=None)
args = ap.parse_args()
assert args.n >= 2, 'shuffling needs two episodes of a game'
os.environ['NL_DEVICE'] = args.device

import torch  # noqa: E402

import payload_ctl  # noqa: E402
import play_tasks  # noqa: E402
from evaluate import load_pair, run_episodes  # noqa: E402
from link import SIDES  # noqa: E402

CONDS = (('link', None), ('cut', 'zero'), ('shuffle', 'shuffle'))


def gate_line(steps):
    """Mean gate per tick (the prompt's tick first), two decimals."""
    return ' '.join(f'{sum(s) / len(s):.2f}' for s in steps if s)


pair, toks = load_pair(args.run)
eps = payload_ctl.episodes(json.load(open(args.replies)), args.n, coread=True, path=args.coread_data)
games = [g for g in payload_ctl.PAYLOAD_GAMES + play_tasks.SCORED if g in eps]
out, lines = [], []
with torch.no_grad():
    for g in games:
        res = {c: run_episodes(pair, eps[g], mode=m, bs=args.bs) for c, m in CONDS}
        for i, ep in enumerate(eps[g]):
            rec = dict(game=g, episode=i, meta=ep['meta'], system=ep['system'], user=ep['task'],
                       target={x: ep['reply'][x][0] for x in SIDES}, conditions={})
            for c, _ in CONDS:
                r = res[c][i]
                rec['conditions'][c] = dict(text=r['text'], scores={k: v for k, v in r.items() if k == 'ok' or
                                                                   k.startswith('score_') or k in ('each', 'holder', 'per_digit')},
                                            gates={x: gate_line(r['gates'][x]) if c != 'cut' else 'no bridge'
                                                   for x in SIDES})
            out.append(rec)
            lines.append(f'{"=" * 100}\n{g}  episode {i}  meta {json.dumps({k: v for k, v in ep["meta"].items() if k != "payload"}, ensure_ascii=False)[:400]}')
            for x in SIDES:
                lines.append(f'--- {x} system:\n{ep["system"][x]}\n--- {x} user:\n{ep["task"][x]}\n--- {x} target:\n{ep["reply"][x][0]}')
            for c, _ in CONDS:
                d = rec['conditions'][c]
                lines.append(f'### {c}  scores {json.dumps(d["scores"])}')
                for x in SIDES:
                    lines.append(f'--- {x} wrote:\n{d["text"][x]}\n    {x} gates (prompt, then each tick): {d["gates"][x]}')
        print(g, {c: round(sum(r['ok'] for r in res[c]) / len(res[c]), 3) for c, _ in CONDS}, flush=True)
open(args.out, 'w').write('\n'.join(lines) + '\n')
json.dump(out, open(os.path.splitext(args.out)[0] + '.json', 'w'), indent=1, ensure_ascii=False)
print('wrote', args.out, 'and', os.path.splitext(args.out)[0] + '.json', len(out), 'episodes', flush=True)
