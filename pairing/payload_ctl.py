"""Payload loss per never-seen episode with the link, with the link cut, and with shuffled messages: each episode gets
the messages (and, with the soft read, the slot text) of another episode of the same game, so the receiver's input is
perturbed as much as with the link while the information is removed. The episodes are those of train.py
--payload_eval (the test split, seed 4242), every gate open as during the warm-up.

Usage: python payload_ctl.py --run RUN/step_N --replies REPLIES.json --out FILE.json [--n 140]
"""
import argparse
import json

import torch
import torch.nn.functional as F

import games as G
import session as S
from evaluate import load_pair
from link import SIDES

PAYLOAD_GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--replies', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=140)
ap.add_argument('--max_digits', type=int, default=16)
args = ap.parse_args()

pair, toks = load_pair(args.run)
pair.open = True
pads = {x: toks[x].pad_token_id for x in SIDES}


def payload(ep):
    """{x: [(start, end)]}: the characters of x's reply that only the partner can supply (as in train.py)."""
    g, m = ep['game'], ep['meta']
    text = {x: ep['reply'][x][0] for x in SIDES}

    def at(x, sub, after=''):
        i = text[x].find(after + sub)
        return [(i + len(after), i + len(after) + len(sub))] if i >= 0 else []
    if g.startswith('relay'):
        return {m['receiver']: at(m['receiver'], m['value'])}
    if g == 'merge_sum':
        v = m['values']
        return {x: at(x, str(v[S.partner(x)]), "'s is ") + at(x, str(m['sum']), 'sum is ') for x in SIDES}
    if g == 'merge_fact':
        n = m['has_name']
        return {n: at(n, m['city'], 'lives in '), S.partner(n): at(S.partner(n), m['name'])}
    if g == 'lookup':
        return {m['asker']: at(m['asker'], m['value'], 'the word ')}
    return {}


def payload_labels(ep, x):
    spans = payload(ep).get(x, [])
    enc = toks[x](ep['reply'][x][0], add_special_tokens=False, return_offsets_mapping=True)
    return [j for j, (a, b) in enumerate(enc.offset_mapping) if any(a < e and b > s for s, e in spans)]


games = G.Games(json.load(open(args.replies)), 'test', seed=4242, max_digits=args.max_digits)
eps = [games.episode(PAYLOAD_GAMES[i % len(PAYLOAD_GAMES)]) for i in range(args.n)]
out = []
with torch.no_grad():
    for game in PAYLOAD_GAMES:                  # one batch per game, so a shuffled message comes from the same game
        ids = [i for i, ep in enumerate(eps) if ep['game'] == game]
        chunk = [eps[i] for i in ids]
        built = [S.build(ep, toks) for ep in chunk]
        b = S.collate(built, pads, soft=pair.soft)
        ce = {}
        for cond, mode in (('link', None), ('cut', 'zero'), ('shuffle', 'shuffle')):
            last, _ = pair.coupled(b, rounds=3, mode=mode)
            pair.clear()
            ce[cond] = {x: F.cross_entropy(pair.token_logits(x, last[x], b[x]['idx']), b[x]['lab'],
                                           reduction='none').tolist() for x in SIDES}
        for x in SIDES:
            off = 0
            for k, (ep, e) in enumerate(zip(chunk, built)):
                js = payload_labels(ep, x)
                if js:
                    out.append(dict(episode=ids[k], game=game, receiver=x, n_tokens=len(js),
                                    **{f'{c}_ce': sum(ce[c][x][off + j] for j in js) / len(js) for c in ce}))
                off += len(e[x]['labels'])
        print(game, {c: round(sum(r[f'{c}_ce'] for r in out if r['game'] == game) /
                             max(1, sum(r['game'] == game for r in out)), 3) for c in ce}, flush=True)
json.dump(sorted(out, key=lambda r: (r['episode'], r['receiver'])), open(args.out, 'w'))
print('wrote', args.out, len(out), 'records', flush=True)
