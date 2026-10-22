"""Payload loss per never-seen episode with the bridges, without them ('cut': the two models alone), and shuffled: each
episode reads the partner states of another episode of the same game, so the reader's input is perturbed as much as
with the bridges while the information is removed. The payload is the part of a target reply that only the partner can
supply (session.payload). The episodes are drawn from the test split (seed 4242), n per game: the payload games and,
with --coread, the co-reading games. train.py --payload_eval runs the same table during training.

Usage: python payload_ctl.py --run RUN/step_N --replies REPLIES.json --out FILE.json [--n 20] [--coread]
"""
import argparse
import json

import torch
import torch.nn.functional as F

import games as G
import session as S
from link import SIDES

PAYLOAD_GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')
CONDS = (('link', None), ('cut', 'zero'), ('shuffle', 'shuffle'))


def episodes(replies, n, max_digits=16, coread=False, path=None, pause=0):
    """{game: n episodes of the never-seen split}, the same on every call."""
    games = G.Games(replies, 'test', seed=4242, max_digits=max_digits)
    out = {g: [games.episode(g) for _ in range(n)] for g in PAYLOAD_GAMES}
    if coread:
        import coread as C
        co = C.Coread(replies, 'test', seed=4242, path=path)
        for g in C.GAMES:
            if g != 'co_task' and (g != 'co_distill' or co.distill) and (g != 'co_talk' or co.talk) and \
                    (g != 'co_play' or co.play):
                out[g] = [co.episode(g) for _ in range(n)]
    if not isinstance(pause, int) or isinstance(pause, bool) or pause < 0:
        raise ValueError('pause must be a nonnegative integer')
    if pause:
        for eps in out.values():
            for ep in eps:
                if ep['mode'] == 'duplex':
                    ep['pause'] = pause
                elif ep['mode'] == 'dialogue':
                    ep['pause_turns'] = {j: pause for j in range(len(ep['turns']))}
    return out


@torch.no_grad()
def table(pair, toks, eps, bs=16, conds=CONDS):
    """Per episode (of one game) and receiving model: the mean payload cross-entropy under each condition. The
    episodes are read in nearly equal batches of at most bs, so no batch is a single episode (shuffle)."""
    if len({ep['game'] for ep in eps}) > 1:
        raise ValueError('payload table requires episodes of one game')
    pads = {x: toks[x].pad_token_id for x in SIDES}
    out = []
    for a, z in S.batches(len(eps), bs):
        chunk = eps[a:z]
        built = [S.build(ep, toks) for ep in chunk]
        b = S.collate(built, pads)
        ce = {}
        for name, mode in conds:
            last, _ = pair.joint(b, mode=mode)
            ce[name] = {x: F.cross_entropy(pair.token_logits(x, last[x], b[x]['idx']), b[x]['lab'],
                                           reduction='none').tolist() for x in SIDES}
        for x in SIDES:
            off = 0
            for k, (ep, e) in enumerate(zip(chunk, built)):
                js = S.payload_labels(ep, toks).get(x, [])
                if js:
                    out.append(dict(game=ep['game'], episode=a + k, receiver=x, n_tokens=len(js),
                                    **{f'{c}_ce': sum(ce[c][x][off + j] for j in js) / len(js) for c in ce}))
                off += len(e[x]['labels'])
    return out


def summary(rec):
    """{game/receiving model: [mean payload cross-entropy per condition..., episodes]}."""
    out = {}
    for r in rec:
        out.setdefault(f"{r['game']}/{r['receiver']}", []).append(r)
    conds = [k for k in rec[0] if k.endswith('_ce')] if rec else []
    return {k: [round(sum(r[c] for r in v) / len(v), 3) for c in conds] + [len(v)] for k, v in sorted(out.items())}


if __name__ == '__main__':
    from evaluate import load_pair

    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20, help='episodes per game')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--max_digits', type=int, default=16)
    ap.add_argument('--pause', type=int, default=0, help='silent ticks before replies / dialogue turns')
    ap.add_argument('--coread', action='store_true', help='also the co-reading games')
    args = ap.parse_args()

    pair, toks = load_pair(args.run)
    eps = episodes(json.load(open(args.replies)), args.n, args.max_digits, args.coread, pause=args.pause)
    rec = []
    for g in eps:
        part = table(pair, toks, eps[g], args.bs)
        rec += part
        print(g, json.dumps(summary(part)), flush=True)
    json.dump(rec, open(args.out, 'w'))
    print('wrote', args.out, len(rec), 'records', flush=True)
