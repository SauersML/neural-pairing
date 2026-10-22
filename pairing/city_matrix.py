"""Private-city intervention in real first-turn chat prefixes.

Select three shortest distinct, city-free openings per receiver from test chats,
before looking at checkpoint outputs. Use the first twelve alphabetic test cities
that are complete single tokens after every prefix in both tokenizers, excluding
the receivers' private cities. This prevents candidate-length bias: the reported
log probability covers the FULL city, not only its first subtoken. All other
tokens, roles and timing stay fixed within each opening's intervention matrix.

The prediction state is obtained from session.build, truncated at its causal
tick. One vocabulary projection scores all twelve candidates exactly. Shuffle
is an actual Pair.joint(mode='shuffle') between adjacent same-role city pairs.
The exact randomization distribution enumerates all column-label permutations
by integer subset dynamic programming, with rank-one matches as the statistic.
No existing measurement functions are modified.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import coread as C
import games as G
import session as S
from link import SIDES, partner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_design(chats, toks, n=3, k=12):
    templates = {}
    for x in SIDES:
        eligible = []
        for i, chat in enumerate(chats['test']):
            sp, text = chat['turns'][0]
            city = chat['private'][partner(x)]
            if sp != x or not city or city not in text:
                continue
            pre = text[:text.index(city)]
            if not pre.strip() or any(c.lower() in pre.lower() for c in G.POOLS['cities']['test']):
                continue
            eligible.append(dict(source_index=i, receiver=x, prefix=pre,
                                 original_city=city, original_text=text,
                                 own_city=chat['private'][x]))
        unique = {}
        for row in sorted(eligible, key=lambda r: (len(r['prefix']), r['source_index'])):
            unique.setdefault(row['prefix'], row)
        templates[x] = list(unique.values())[:n]
        if len(templates[x]) != n:
            raise ValueError('insufficient distinct test openings')
    own = {r['own_city'] for rows in templates.values() for r in rows}
    cities = []
    for city in sorted(G.POOLS['cities']['test']):
        if city in own:
            continue
        valid = True
        for x, rows in templates.items():
            for row in rows:
                pre = S.ids(toks[x], row['prefix'].rstrip())
                full = S.ids(toks[x], row['prefix'] + city)
                valid &= full[:-1] == pre and len(full) == len(pre) + 1
        if valid:
            cities.append(city)
    if len(cities) < k or k < 12:
        raise ValueError(f'need at least {k} full-token cities; have {cities}')
    cities = cities[:k]
    assert set(cities).isdisjoint(G.POOLS['cities']['train'])
    return dict(cities=cities, templates=templates,
                training_cities=G.POOLS['cities']['train'],
                excluded_test_cities=[c for c in G.POOLS['cities']['test'] if c not in cities],
                selection='shortest distinct city-free test first-turn prefixes; alphabetic full-single-token cities; receiver cities excluded')


def episode(template, true_city, candidate):
    x = template['receiver']
    y = partner(x)
    own = template['own_city']
    systems = {x: C.chat_system(x, own, C.CHAT_TEST if own else C.CHAT[3]),
               y: C.chat_system(y, true_city, C.CHAT_TEST)}
    assert true_city not in systems[x]
    pre = template['prefix']
    return dict(game='co_chat', mode='dialogue', system=systems,
                task={z: None for z in SIDES}, turns=[(x, pre + candidate, 1.)],
                meta=dict(receiver=x, opener=x, private={x: own, y: true_city},
                          payload={x: [(len(pre), len(pre + candidate))]}))


def prediction_input(ep, toks, x):
    built = S.build(ep, toks)
    js = S.payload_labels(ep, toks)[x]
    if len(js) != 1:
        raise ValueError('candidate must be one full token, never truncate a multi-token city')
    p, token, _ = built[x]['labels'][js[0]]
    tick = built[x]['time'][p]
    for z in SIDES:
        b = built[z]
        n = sum(t <= tick for t in b['time'])
        b['ids'], b['time'] = b['ids'][:n], b['time'][:n]
        b['labels'] = [(p, token, 1.)] if z == x else []
        b['policy'] = [True] if z == x else []
        b['spans'] = [(0, 0, 1)] if z == x else []
    assert p == len(built[x]['ids']) - 1
    return built


@torch.no_grad()
def score_pair(pair, toks, template, true_cities, candidates, mode):
    x = template['receiver']
    built = [prediction_input(episode(template, city, candidates[0]), toks, x) for city in true_cities]
    assert built[0][x]['ids'] == built[1][x]['ids']
    b = S.collate(built, {z: toks[z].pad_token_id for z in SIDES})
    h, _ = pair.joint(b, mode=mode)
    lp = pair.token_logits(x, h[x], b[x]['idx']).float().log_softmax(-1)
    tokens = []
    for city in candidates:
        full = S.ids(toks[x], template['prefix'] + city)
        assert full[:-1] == S.ids(toks[x], template['prefix'].rstrip())
        tokens.append(full[-1])
    return lp[:, tokens].cpu().numpy()


def exact_permutation(top):
    """Integer counts for every permutation of candidate labels; no Monte Carlo.

    top[i,j] is 1 iff j is the unique first choice in row i. Each permutation
    pairs each row with one distinct candidate. DP counts every assignment once.
    """
    top = np.asarray(top, dtype=int)
    k = len(top)
    dp = {0: [1] + [0] * k}
    for row in range(k):
        nxt = {}
        for mask, counts in dp.items():
            for col in range(k):
                if mask & (1 << col):
                    continue
                dest = nxt.setdefault(mask | (1 << col), [0] * (k + 1))
                delta = int(top[row, col])
                for score, count in enumerate(counts[:k + 1 - delta]):
                    dest[score + delta] += count
        dp = nxt
    counts = dp[(1 << k) - 1]
    observed = int(np.trace(top))
    assert sum(counts) == math.factorial(k)
    numerator = sum(counts[observed:])
    return dict(observed=observed, total=k, percent=100 * observed / k,
                counts=counts, numerator=numerator, denominator=sum(counts),
                p=numerator / sum(counts))


def summarize(raw):
    matrices, stats, audits = {}, {}, {}
    for x in SIDES:
        matrices[x], stats[x] = {}, {}
        for cond in ('link', 'cut', 'shuffle'):
            a = np.mean(np.asarray(raw[x][cond]), axis=0)
            maximum = a.max(1, keepdims=True)
            top = a == maximum
            top &= top.sum(1, keepdims=True) == 1
            matrices[x][cond] = a.tolist()
            stats[x][cond] = exact_permutation(top)
        link, cut, shuffled = (np.asarray(raw[x][c]) for c in ('link', 'cut', 'shuffle'))
        donor = np.arange(link.shape[1]) ^ 1
        audits[x] = dict(cut_row_max_difference=float(np.max(np.abs(cut - cut[:, :1]))),
                         shuffle_vs_actual_donor_max_difference=float(np.max(np.abs(shuffled - link[:, donor]))))
    return dict(mean_logp=matrices, stats=stats, audits=audits)


def self_test():
    if os.environ.get('NL_TINY') != '1':
        raise ValueError('--self-test requires NL_TINY=1')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    import test_link as T  # executes the existing exactness tests on actual tiny models
    template = dict(receiver='quill', own_city='Accra', prefix="So, you're in ")
    candidates = ['Beijing', 'Cairo']
    for x in SIDES:
        template['receiver'] = x
        for mode in (None, 'zero', 'shuffle'):
            fast = score_pair(T.pair, T.toks, template, candidates, candidates, mode)
            for j, candidate in enumerate(candidates):
                eps = [episode(template, c, candidate) for c in candidates]
                built = [S.build(ep, T.toks) for ep in eps]
                b = S.collate(built, T.pads)
                with torch.no_grad():
                    h, _ = T.pair.joint(b, mode=mode)
                    lp = T.pair.token_logits(x, h[x], b[x]['idx']).log_softmax(-1)
                off, slow = 0, []
                for ep, e in zip(eps, built):
                    js = S.payload_labels(ep, T.toks)[x]
                    slow.append(sum(float(lp[off + z, e[x]['labels'][z][1]]) for z in js))
                    off += len(e[x]['labels'])
                np.testing.assert_allclose(fast[:, j], slow, atol=3e-5, rtol=0)
    top = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    dist = exact_permutation(top)
    brute = [sum(top[i, p[i]] for i in range(4)) for p in itertools.permutations(range(4))]
    assert dist['counts'] == [brute.count(i) for i in range(5)]
    assert exact_permutation(np.eye(12))['numerator'] == 1
    print('PASS: complete-city probabilities match ordinary full-candidate passes in both directions and all controls; exact permutation counts match exhaustive enumeration', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run')
    ap.add_argument('--chat-data', default='/Users/user/mpd-data/neural-pairing/chatdata/chat_clean.json')
    ap.add_argument('--out', default='results/city_matrix.json')
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--design-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        return self_test()
    if args.backend == 'torch' and sys.platform == 'darwin' and not os.environ.get('NL_TINY'):
        ap.error('real torch models are forbidden on the Mac')
    from transformers import AutoTokenizer
    from link import MODELS
    toks = {x: AutoTokenizer.from_pretrained(MODELS[x]['hf'], local_files_only=True) for x in SIDES}
    design = select_design(json.loads(Path(args.chat_data).read_text()), toks)
    if args.design_only:
        print(json.dumps(design, indent=2))
        return
    if not args.run:
        ap.error('--run is required')
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(int(10.5 * 2**30))
        mx.set_cache_limit(32 * 2**20)
    else:
        from evaluate import load_pair
    pair, toks = load_pair(args.run)
    out = dict(schema=1, complete=False, backend=args.backend, precision='4-bit MLX' if args.backend == 'mlx' else 'bf16 torch',
               checkpoint=str(Path(args.run).resolve()), checkpoint_sha256=sha(Path(args.run) / 'bridge.pt'),
               chat_data=args.chat_data, chat_sha256=sha(args.chat_data), design=design,
               code_sha256={p: sha(Path(__file__).with_name(p)) for p in ('city_matrix.py', 'coread.py', 'session.py', 'mlx_eval.py', 'link.py')},
               donor_indices=[i ^ 1 for i in range(len(design['cities']))],
               raw_logp={x: {c: [] for c in ('link', 'cut', 'shuffle')} for x in SIDES})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cities = design['cities']
    for x in SIDES:
        for t, template in enumerate(design['templates'][x]):
            values = {c: [] for c in ('link', 'cut', 'shuffle')}
            for i in range(0, len(cities), 2):
                for cond, mode in (('link', None), ('cut', 'zero'), ('shuffle', 'shuffle')):
                    a = score_pair(pair, toks, template, cities[i:i + 2], cities, mode)
                    values[cond].extend(a.tolist())
                print(json.dumps(dict(receiver=x, opening=t, city_pair=cities[i:i + 2], seconds=round(time.monotonic() - started, 1))), flush=True)
            for cond in values:
                out['raw_logp'][x][cond].append(values[cond])
            target.write_text(json.dumps(out, indent=2) + '\n')
    out.update(summarize(out['raw_logp']))
    out.update(complete=True, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(stats=out['stats'], audits=out['audits'])), flush=True)


if __name__ == '__main__':
    main()
