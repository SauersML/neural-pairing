"""Free generation of the duplex word game and the shared-pen poem, with the time step of every token, link on and cut,
for figures/interleaved.py.

Duplex (relay_word): both twins write at once, one token per time step from step 0, and neither twin's input contains
the other's text. The sender's prompt holds a word from the test pool (never seen in training), and the receiver is
asked to write the word. Token j of either twin is chosen at step j from hidden states that read the partner's prompt
and the partner's tokens 0..j-1 (session.py). Shared pen (pen_secret): one assistant message written by both twins,
one word per time step, the author changing every word; the holder's prompt holds a private final word from the test
pool, and the receiver writes the last word. Each author reads the whole shared text before writing, so a pen episode
in which the holder writes the word before the receiver does is flagged ('holder_first').

For each game, n test episodes (games.Games, split 'test'), half with each twin as the sender or holder, each generated
greedily with the link on and cut (mode 'zero'), as evalfree.py does. Scores: relay_word, evaluate.score (the
receiver's reply contains the word); pen_secret, whether a word the receiver wrote is the word ('wrote'), whether the
poem's last word is the word ('ends') and 'holder_first'.

Writes OUT (JSON): per episode the prompts, the word, and per condition each twin's tokens as text pieces with their
time steps (duplex) or the words with their authors (pen), and the scores.
Usage: python pairing/interleaved.py --run STEP_DIR --replies REPLIES.json --out OUT.json [--n 40] [--bs 8]
           [--backend mlx|torch] [--memory_gib 10]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

SIDES = ('quill', 'ember')
CONDITIONS = (('link', None), ('cut', 'zero'))
ROLE = dict(relay_word='sender', pen_secret='holder')


def draw(G, replies, game, n, seed):
    """n test episodes of game, alternating the twin that holds the word (n/2 each), in draw order."""
    g = G.Games(replies, 'test', seed)
    eps = {x: [] for x in SIDES}
    while min(map(len, eps.values())) < n // 2:
        ep = g.episode(game)
        x = ep['meta'][ROLE[game]]
        if len(eps[x]) < n // 2:
            eps[x].append(ep)
    return [ep for both in zip(*(eps[x] for x in SIDES)) for ep in both]


def pieces(tok, ids):
    """The text each token of ids adds to the decoded text; a token that ends inside a character adds '' and the
    character goes to the token that completes it."""
    out, before = [], ''
    for j in range(len(ids)):
        now = tok.decode(ids[:j + 1])
        if now.endswith('�') or not now.startswith(before):
            out.append('')
            continue
        out.append(now[len(before):])
        before = now
    rest = tok.decode(ids)
    if out and rest != before:
        out[-1] += rest[len(before):] if rest.startswith(before) else rest
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def first_tick(E, tokens, v):
    """The time step of the token that completes the first occurrence of v in the text of tokens, or None."""
    text = ''
    for t in tokens:
        text += t['text']
        if E.has(text, v):
            return t['tick']
    return None


def word(w):
    return re.sub(r'[^\w]', '', w).lower()


def pen_scores(ep, words):
    v, r = ep['meta']['value'].lower(), ep['meta']['receiver']
    clean = [(x, word(w)) for x, w in words]
    first = next((x for x, w in clean if w == v), None)
    return dict(wrote=any(x == r and w == v for x, w in clean),
                ends=bool(clean) and clean[-1] == (r, v),
                holder_first=first is not None and first != r)


def run(pair, toks, game, eps, bs):
    """One record per episode: prompts, the word, and per condition the generation and its scores."""
    import evaluate as E
    import session as S
    end = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
    recs = []
    for a in range(0, len(eps), bs):
        chunk = eps[a:a + bs]
        res = {c: S.generate(pair, chunk, max_turn=E.MAXNEW.get(game, 64), temperature=0.0, mode=m,
                             keep_tokens=True) for c, m in CONDITIONS}
        for i, ep in enumerate(chunk):
            rec = dict(game=game, index=a + i, meta=ep['meta'], system=ep['system'], task=ep['task'], conditions={})
            for c, _ in CONDITIONS:
                r = res[c][i]
                if game == 'pen_secret':
                    rec['conditions'][c] = dict(text=r['text'], words=[dict(tick=j, author=x, text=w)
                                                                        for j, (x, w) in enumerate(r['words'])],
                                                **pen_scores(ep, r['words']))
                    continue
                tokens = {}
                for x in SIDES:
                    ids = [k for k in r['tokens'][x] if k != end[x]]
                    tokens[x] = [dict(tick=j, id=k, text=p) for j, (k, p) in enumerate(zip(ids, pieces(toks[x], ids)))]
                v, x = ep['meta']['value'], ep['meta']['receiver']
                rec['conditions'][c] = dict(text=r['text'], tokens=tokens, ok=E.score(ep, r)['ok'],
                                            receiver_tick=first_tick(E, tokens[x], v),
                                            sender_wrote=E.has(r['text'][ep['meta']['sender']], v))
            recs.append(rec)
        print(json.dumps({game: a + len(chunk), **{c: sum(r['conditions'][c]['ok' if game == 'relay_word' else 'ends']
                                                          for r in recs) for c, _ in CONDITIONS}}), flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=40, help='episodes per game (even)')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4949)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--memory_gib', type=float, default=10)
    args = ap.parse_args()
    if args.n < 2 or args.n % 2:
        ap.error('--n must be even and positive')
    started = time.monotonic()
    if args.backend == 'mlx':
        import mlx.core as mx
        import mlx_eval
        mx.set_memory_limit(int(args.memory_gib * 2**30))
        mx.set_cache_limit(32 * 2**20)
        pair, toks = mlx_eval.load_pair(args.run)
    else:
        import evaluate
        pair, toks = evaluate.load_pair(args.run)
    import games as G
    replies = json.loads(Path(args.replies).read_text())
    out = dict(checkpoint=Path(args.run).resolve().name, bridge_sha256=sha256(Path(args.run) / 'bridge.pt'),
               backend=args.backend, seed=args.seed, n=args.n, decoding='greedy', episodes={})
    for game in ROLE:
        out['episodes'][game] = run(pair, toks, game, draw(G, replies, game, args.n, args.seed), args.bs)
    out['wall_seconds'] = round(time.monotonic() - started, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print('wrote', args.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
