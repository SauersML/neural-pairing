"""Transcripts of the pair: n held-out episodes of every payload game, every scored co-reading game
(play_tasks.SCORED), co_talk and co_play (when talk.json and play.json sit next to the co-reading data), generated
greedily on the same episodes under three conditions: link, cut, and shuffle (each episode reads the partner of
another episode of the same game). Dialogue games also run 'crossed': each scored receiver in its own conversation
with an independently drawn same-role donor (evalfree.choose_donor), kept separate in receiver_runs.

Per episode: each side's system and user prompt, its target, and under each condition its text, the scores and one
line of mean gates per tick (the prompt's tick first). A co_talk or co_play episode is a whole conversation: its
target is the reference conversation [[speaker, text], ...], and each condition gives the turns written, the scores
(co_talk: the asker's first turn; co_play: each twin's naming turn) and each side's mean gate over the prompt and over
each turn. Writes FILE.txt and FILE.json.

Usage: python transcripts.py --run RUN/step_N --replies REPLIES.json --out FILE.txt [--n 3] [--device mps]
           [--coread_data DATA.json] [--games GAME ...]
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
ap.add_argument('--games', nargs='+', help='optional subset of the transcript games')
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


def turn_gates(steps, turns, n):
    """Mean gate of the prompt and of each of n turns (None where nothing was read), from the per-tick gates and the
    turn under way at each tick (-1: the prompt)."""
    def mean(k):
        g = [v for s, t in zip(steps, turns) if t == k for v in s]
        return round(sum(g) / len(g), 3) if g else None
    return dict(prompt=mean(-1), turns=[mean(k) for k in range(n)])


pair, toks = load_pair(args.run)
replies = json.load(open(args.replies))
eps = payload_ctl.episodes(replies, args.n, coread=True, path=args.coread_data)
games = [g for g in payload_ctl.PAYLOAD_GAMES + play_tasks.SCORED + ('co_talk', 'co_play') if g in eps]
if args.games:
    if set(args.games) - set(games):
        raise ValueError('requested transcript game is unavailable')
    games = [g for g in games if g in args.games]
out, lines = [], []
with torch.no_grad():
    for g in games:
        res = {c: run_episodes(pair, eps[g], mode=m, bs=args.bs) for c, m in CONDS}
        conds = CONDS
        if eps[g][0]['mode'] == 'dialogue':
            import coread
            import evalfree
            source = coread.Coread(replies, 'test', seed=13013, path=args.coread_data)
            donors = [evalfree.choose_donor(ep, source)[0] for ep in eps[g]]
            res['crossed'] = run_episodes(pair, eps[g], mode='crossed', donors=donors)
            conds = CONDS + (('crossed', 'crossed'),)
        for i, ep in enumerate(eps[g]):
            talk = ep['mode'] == 'dialogue'
            assert not ep.get('forced'), 'every turn of a transcript is generated, none is teacher-forced'
            rec = dict(game=g, episode=i, meta=ep['meta'], system=ep['system'], user=ep['task'],
                       target=[[sp, t] for sp, t, _ in ep['turns']] if talk else {x: ep['reply'][x][0] for x in SIDES},
                       conditions={})
            for c, _ in conds:
                r = res[c][i]
                if c == 'crossed':
                    rec['conditions'][c] = dict(scores=dict(ok=r['ok'], each=r['each']),
                                                receiver_runs=r['receiver_runs'], donor=r['donor'],
                                                implementation=r['implementation'])
                    continue
                rec['conditions'][c] = dict(scores={k: v for k, v in r.items() if k == 'ok' or
                                                    k.startswith('score_') or k in ('each', 'holder', 'per_digit')},
                                            gates={x: gate_line(r['gates'][x]) if c != 'cut' else 'no bridge'
                                                   for x in SIDES})
                if talk:
                    rec['conditions'][c].update(turns=[list(t) for t in r['turns']], turn_gates={
                        x: turn_gates(r['gates'][x], r['gate_turn'][x], len(ep['turns'])) if c != 'cut' else None
                        for x in SIDES})
                else:
                    rec['conditions'][c]['text'] = r['text']
            out.append(rec)
            meta = json.dumps({k: v for k, v in ep['meta'].items() if k != 'payload'}, ensure_ascii=False)[:400]
            lines.append(f'{"=" * 100}\n{g}  episode {i}  meta {meta}')
            for x in SIDES:
                lines.append(f'--- {x} system:\n{ep["system"][x]}\n--- {x} user:\n{ep["task"][x]}')
                if not talk:
                    lines.append(f'--- {x} target:\n{ep["reply"][x][0]}')
            if talk:
                lines += [f'--- target turn {k} ({sp}):\n{t}' for k, (sp, t) in enumerate(rec['target'])]
            for c, _ in conds:
                d = rec['conditions'][c]
                lines.append(f'### {c}  scores {json.dumps(d["scores"])}')
                if c == 'crossed':
                    lines.append('Separate isolated conversations below; only the named receiver is scored.')
                    for receiver, r in d['receiver_runs'].items():
                        donor = json.dumps(d['donor'], ensure_ascii=False)
                        lines.append(f'--- focal receiver {receiver}; donor {donor}')
                        lines += [f'--- turn {k} ({sp}) wrote:\n{t}' for k, (sp, t) in enumerate(r['turns'])]
                        for x in SIDES:
                            lines.append(f'    {x} gates (prompt, then each tick): {gate_line(r["gates"][x])}')
                    continue
                if talk:
                    lines += [f'--- turn {k} ({sp}) wrote:\n{t}' for k, (sp, t) in enumerate(d['turns'])]
                for x in SIDES:
                    if not talk:
                        lines.append(f'--- {x} wrote:\n{d["text"][x]}')
                    lines.append(f'    {x} gates (prompt, then each tick): {d["gates"][x]}')
        print(g, {c: round(sum(r['ok'] for r in res[c]) / len(res[c]), 3) for c, _ in conds}, flush=True)
open(args.out, 'w').write('\n'.join(lines) + '\n')
json.dump(out, open(os.path.splitext(args.out)[0] + '.json', 'w'), indent=1, ensure_ascii=False)
print('wrote', args.out, 'and', os.path.splitext(args.out)[0] + '.json', len(out), 'episodes', flush=True)
