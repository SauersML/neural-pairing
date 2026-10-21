"""Held-out evaluation of the linked models: both models generate in lockstep (link.Pair.lockstep), greedy.

Everything is held out: words, numbers in merge_sum (50-99), adjectives, names, cities, concepts, questions, and the
wordings of every instruction. Every game runs with the link and under three channel ablations: 'zero' (no message
arrives), 'shuffle' (each episode receives another episode's messages), and 'random' (random message directions of the
same norm, same gates). Steering also runs with random residual writes of the same norm as the trained writes.

Usage: python evaluate.py --run OUT/run/step_3000 --replies OUT/replies.json --out OUT/eval.json [--n 128]
"""
import argparse
import json
import os
import re

import torch

import games as G
from link import DEVICE, MODELS, SIDES, partner

MAXNEW = {'relay_digits': 80}


def has(text, w):
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(w.lower()) + r'(?![A-Za-z0-9])', text.lower()) is not None


def mentions(text, name, phrase):
    return any(has(text, w) for w in {name, phrase, phrase.removeprefix('the ')}) or has(text, name + 's')


def digits_after(text):
    m = re.search(r'are\s*([\d\s]+)', text)
    return re.findall(r'\d', m.group(1)) if m else re.findall(r'\d', text)


def score(ep, out):
    """Success (0/1) of one episode from both models' texts, and extras."""
    g, m = ep['game'], ep['meta']
    if g.startswith('relay'):
        t = out[m['receiver']]
        if g == 'relay_digits':
            want = m['value'].split()
            got = digits_after(t)
            per = sum(a == b for a, b in zip(want, got)) / len(want)
            return {'ok': float(got[:len(want)] == want), 'per_digit': per, 'n': len(want)}
        if g == 'relay_phrase':
            a, c = m['value'].split(' ', 1)
            return {'ok': float(has(t, a) and has(t, c))}
        return {'ok': float(has(t, m['value']))}
    if g == 'merge_sum':
        ok = [has(out[x], str(m['sum'])) and has(out[x], str(m['values'][partner(x)])) for x in SIDES]
        return {'ok': float(all(ok)), 'each': sum(ok) / 2}
    if g == 'merge_fact':
        ok = [has(out[x], m['name']) and has(out[x], m['city']) for x in SIDES]
        return {'ok': float(all(ok)), 'each': sum(ok) / 2}
    if g == 'lookup':
        return {'ok': float(has(out[m['asker']], m['value'])), 'holder': float(has(out[m['holder']], m['value']))}
    if g in ('steer', 'resist'):
        return {'ok': float(mentions(out[m['receiver']], m['concept'], m['phrase']))}
    if g in ('identity', 'impersonate'):
        r = m['receiver']
        t = out[r]
        me, you = MODELS[r]['name'], MODELS[partner(r)]['name']
        claims_you = re.search(r"\b(i'?m|i am|my name is|this is)\s+" + you.lower(), t.lower()) is not None
        return {'ok': float(has(t, me) and not claims_you), 'claims_partner': float(claims_you)}
    if g == 'plain':
        return {'ok': 1.0}
    raise ValueError(g)


def run_episodes(pair, toks, eps, mode=None, delta_mode=None, bs=32, seed=0, mute=None):
    """Lockstep generation for a list of episodes; returns per-episode texts, gates and scores."""
    gen = torch.Generator().manual_seed(seed)
    res = []
    for b in range(0, len(eps), bs):
        chunk = eps[b:b + bs]
        prompts = {x: [G.encode(toks[x], ep[x])[0] for ep in chunk] for x in SIDES}
        mx = int(os.environ.get('NL_MAXNEW', 0)) or max(MAXNEW.get(ep['game'], 64) for ep in chunk)
        mu = None if mute is None else {x: [x in mute(ep) for ep in chunk] for x in SIDES}
        out = pair.lockstep(prompts, mx, mode=mode, gen=gen, delta_mode=delta_mode, mute=mu)
        for i, ep in enumerate(chunk):
            texts = {x: toks[x].decode(out[x]['tokens'][i], skip_special_tokens=True).strip() for x in SIDES}
            gates = {x: out[x]['gates'][i] for x in SIDES}
            res.append(dict(game=ep['game'], meta=ep['meta'], text=texts, gates=gates, **score(ep, texts)))
    return res


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def quick(pair, toks, games, n=32):
    """Link vs zero-message accuracy on n held-out episodes per game (used during training)."""
    out = {}
    for g in ('relay_word', 'relay_number', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup', 'steer', 'identity'):
        eps = [games.episode(g) for _ in range(n)]
        a = run_episodes(pair, toks, eps)
        z = run_episodes(pair, toks, eps, mode='zero')
        out[g] = [round(mean([r['ok'] for r in a]), 3), round(mean([r['ok'] for r in z]), 3)]
    return out


if __name__ == '__main__':
    from peft import PeftModel
    from link import Pair, load

    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.run, 'link.pt'), weights_only=False)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        models[x] = PeftModel.from_pretrained(m, os.path.join(args.run, x)).eval()
    pair = Pair(models, toks, ck['lr'], ck['ls'], ck['sizes']).to(DEVICE)
    pair.side.load_state_dict(ck['link'])
    pair.eval()
    replies = json.load(open(args.replies))
    games = G.Games(replies, 'test', seed=12345, max_digits=32)

    R = {'games': {}, 'capacity': {}, 'examples': {}}
    for g in G.GAMES + G.COMPETITIVE:
        eps = [games.episode(g) for _ in range(args.n)]
        row = {}
        for mode in (None, 'zero', 'shuffle', 'random'):
            rs = run_episodes(pair, toks, eps, mode=mode, bs=args.bs)
            row[mode or 'link'] = round(mean([r['ok'] for r in rs]), 3)
            if mode is None:
                R['examples'][g] = rs[:6]
                row['sent'] = {x: round(mean([mean([p > 0.5 for p in r['gates'][x]]) for r in rs]), 3) for x in SIDES}
                if g in ('identity', 'impersonate'):
                    row['claims_partner'] = round(mean([r['claims_partner'] for r in rs]), 3)
                if g == 'lookup':
                    row['holder'] = round(mean([r['holder'] for r in rs]), 3)
                if g in ('merge_sum', 'merge_fact'):
                    row['each'] = round(mean([r['each'] for r in rs]), 3)
                if g == 'relay_digits':
                    row['per_digit'] = round(mean([r['per_digit'] for r in rs]), 3)
        if g in ('steer', 'resist'):
            rs = run_episodes(pair, toks, eps, delta_mode='random', bs=args.bs)
            row['random_write'] = round(mean([r['ok'] for r in rs]), 3)
        if g == 'lookup':                   # the holder receives nothing: can the asker still get the word?
            rs = run_episodes(pair, toks, eps, bs=args.bs, mute=lambda ep: {ep['meta']['holder']})
            row['holder_deaf'] = round(mean([r['ok'] for r in rs]), 3)
        R['games'][g] = row
        print(g, json.dumps(row), flush=True)
        json.dump(R, open(args.out, 'w'), indent=1)

    for n in (1, 2, 4, 8, 12, 16, 24, 32):
        eps = []
        while len(eps) < args.n:
            ep = games.episode('relay_digits')
            if len(ep['meta']['value'].split()) == n:
                eps.append(ep)
        rs = run_episodes(pair, toks, eps, bs=args.bs)
        z = run_episodes(pair, toks, eps, mode='zero', bs=args.bs)
        R['capacity'][n] = {'exact': round(mean([r['ok'] for r in rs]), 3),
                            'per_digit': round(mean([r['per_digit'] for r in rs]), 3),
                            'per_digit_zero': round(mean([r['per_digit'] for r in z]), 3)}
        print('digits', n, R['capacity'][n], flush=True)
        json.dump(R, open(args.out, 'w'), indent=1)
