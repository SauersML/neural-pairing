"""Free generation of the listen_steer game on test-split episodes. The writer answers a question; its partner never
writes, reads the writer's text as it streams, and is told a concept never seen in training and asked to steer the
writer over the link toward mentioning it. So the concept can reach the writer only through the link.

Each episode runs with the link on ('link'), cut ('cut': mode 'zero', LoRA still on) and with the partner of another
episode ('other': link.Pair's 'shuffle' mode inside a batch of one writer, so the writer reads the listener of the
neighbouring episode, which is told a different concept and reads a different reply). Greedy, evaluate.MAXNEW tokens
(NL_MAXNEW caps them).
Scored with evaluate.mentions on the writer's reply: the episode's concept, and under 'other' also the concept of the
partner the writer read.

Episodes: Games('test', seed=4242) draws, n per writer with distinct questions, ordered so that consecutive episodes
differ in concept (draw). Writes OUT/episodes.json (texts and scores) and OUT/summary.json (counts per writer and exact
sign tests of link against cut and against the other episode's partner); --summarize rewrites summary.json from
episodes.json.

Usage: python pairing/listen_steer_eval.py --run RUN/step_N --replies REPLIES.json --out OUT [--n 64] [--bs 16]
           [--backend mlx|torch] [--summarize]
"""
import argparse
import json
import os
from pathlib import Path

GAME = 'listen_steer'
CONDS = (('link', None), ('cut', 'zero'), ('other', 'shuffle'))


def draw(games, n):
    """{writer: n test episodes with distinct questions}, each writer's list ordered so that consecutive episodes differ
    in concept where the draws allow (the first and last episode of a batch, which meet in the shuffle's roll, may
    not; summary leaves such episodes out of the other partner's counts)."""
    from link import SIDES
    out = {x: [] for x in SIDES}
    for _ in range(100 * n):
        ep = games.episode(GAME)
        w = out[ep['meta']['receiver']]
        if len(w) < n and all(e['meta']['question'] != ep['meta']['question'] for e in w):
            w.append(ep)
        if all(len(v) == n for v in out.values()):
            break
    for x, eps in out.items():
        if len(eps) < n:
            raise ValueError(f'only {len(eps)} distinct-question episodes with {x} writing')
        rest, order = eps[1:], [eps[0]]
        while rest:
            k = next((k for k, e in enumerate(rest) if e['meta']['concept'] != order[-1]['meta']['concept']), 0)
            order.append(rest.pop(k))
        out[x] = order
    return out


def donor(i, batch, writer):
    """Index in the batch of the episode whose listener the writer of episode i reads under 'shuffle' (link.Pair rolls
    the partner's states by one: the first model reads the previous episode, the second the next)."""
    from link import SIDES
    return (i - 1) % batch if writer == SIDES[0] else (i + 1) % batch


def run(pair, eps, bs):
    """Records of eps (one writer) under each condition, in batches of bs."""
    import evaluate as E
    import session as S
    limit = int(os.environ.get('NL_MAXNEW', 0)) or E.MAXNEW.get(GAME, 64)
    rec = []
    for a in range(0, len(eps), bs):
        chunk = eps[a:a + bs]
        if len(chunk) < 2:
            raise ValueError('a batch of one episode has no other partner; choose n and bs accordingly')
        res = {c: S.generate(pair, chunk, max_turn=limit, mode=mode, temperature=0.0) for c, mode in CONDS}
        for i, ep in enumerate(chunk):
            m = ep['meta']
            d = chunk[donor(i, len(chunk), m['receiver'])]['meta']
            r = dict(index=a + i, writer=m['receiver'], listener=m['sender'], question=m['question'],
                     concept=m['concept'], phrase=m['phrase'], listener_task=ep['system'][m['sender']],
                     other_concept=d['concept'], other_phrase=d['phrase'])
            for c, _ in CONDS:
                text = res[c][i]['turns'][-1][1] if res[c][i]['turns'] else ''
                r[c] = dict(text=text, mentions=E.mentions(text, m['concept'], m['phrase']))
            r['other']['mentions_other'] = E.mentions(r['other']['text'], d['concept'], d['phrase'])
            rec.append(r)
        print(json.dumps(dict(writer=chunk[0]['meta']['receiver'], done=a + len(chunk),
                              link=sum(r['link']['mentions'] for r in rec), cut=sum(r['cut']['mentions'] for r in rec))),
              flush=True)
    return rec


def summary(rec):
    """Counts per writer (and all): episodes, mentions of the concept linked and cut, and the exact sign test of link
    against cut (paired by episode); over the n_other episodes whose other partner was told a different concept:
    mentions of the episode's concept linked and with the other partner, mentions of the other partner's concept
    under that control, and the sign tests of link against the other partner on the episode's concept and of the
    other partner's concept against the episode's concept under that control."""
    from stats import sign_test
    out = {}
    for w in sorted({r['writer'] for r in rec}) + ['all']:
        rs = [r for r in rec if w in ('all', r['writer'])]
        ro = [r for r in rs if r['other_concept'] != r['concept']]
        diff = lambda rows, a, b: sign_test([float(a(r)) - float(b(r)) for r in rows])
        out[w] = dict(n=len(rs), link=sum(r['link']['mentions'] for r in rs), cut=sum(r['cut']['mentions'] for r in rs),
                      link_vs_cut=diff(rs, lambda r: r['link']['mentions'], lambda r: r['cut']['mentions']),
                      n_other=len(ro), link_on_other=sum(r['link']['mentions'] for r in ro),
                      other=sum(r['other']['mentions'] for r in ro),
                      other_mentions_other=sum(r['other']['mentions_other'] for r in ro),
                      link_vs_other=diff(ro, lambda r: r['link']['mentions'], lambda r: r['other']['mentions']),
                      other_concept_vs_own=diff(ro, lambda r: r['other']['mentions_other'],
                                                lambda r: r['other']['mentions']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=64, help='episodes per writer')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--summarize', action='store_true', help='rewrite OUT/summary.json from OUT/episodes.json')
    args = ap.parse_args()
    out = Path(args.out)
    s = dict(checkpoint=str(Path(args.run).resolve()).replace(str(Path.home()), '~'), backend=args.backend,
             n_per_writer=args.n, seed=4242)
    if args.summarize:
        s['summary'] = summary(json.loads((out / 'episodes.json').read_text()))
        (out / 'summary.json').write_text(json.dumps(s, indent=1))
        print(json.dumps(s['summary'], indent=1), flush=True)
        return
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
        import mlx.core as mx
        import torch
        from mlx_eval import load_pair
        mx.set_memory_limit(8 * 2**30)
        mx.set_cache_limit(32 * 2**20)
        torch.set_num_threads(2)
    else:
        from evaluate import load_pair
    import games as G
    pair, toks = load_pair(args.run)
    eps = draw(G.Games(json.loads(Path(args.replies).read_text()), 'test', seed=4242), args.n)
    rec = [r for x in eps for r in run(pair, eps[x], args.bs)]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'episodes.json').write_text(json.dumps(rec, indent=1))
    s['summary'] = summary(rec)
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
