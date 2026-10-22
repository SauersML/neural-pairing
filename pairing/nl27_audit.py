"""Offline nl27 audit and reanalysis; never loads models or changes bench.json.

python pairing/nl27_audit.py --data data/bench.json --out results/nl27_audit.json
python pairing/nl27_audit.py --data data/bench.json --scores RUN/results/bench.json --out RUN/results/bench_audited.json

Input overlap means the same question and ordered choices, ignoring source/subject and gold label.
Raw input equality is also reported. The exclusion key ignores case and whitespace around punctuation,
but retains every word, number and mathematical symbol (e.g. J >> a stays distinct from J << a).
Near matches use whitespace-collapsed, casefolded question text (math symbols retained), character 5-gram
Jaccard >= .8. This exhaustive lexical screen also flags shared passages/templates with different questions;
it is a conservative sensitivity exclusion, not a count of proven answer leakage or a semantic paraphrase audit.

The original benchmark's conditional McNemar p-values are retained. Its interval scales a binomial interval
by the OBSERVED discordance rate: it is not a population accuracy-difference CI. Here two 97.5% exact marginal
Clopper-Pearson intervals for P(A right, B wrong) and P(B right, A wrong) give, by subtraction and Bonferroni,
a conservative unconditional 95% interval. Independent sampled questions are still an assumption; shared
passages can violate it. No claim about training-seed variability follows from a per-question test.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import beta, binomtest


def normalize(s):
    return ' '.join(s.casefold().split())


def input_key(it):
    tokens = lambda s: tuple(re.findall(r'\w+|[^\w\s]', s.casefold()))
    return tokens(it['q']), tuple(map(tokens, it['choices']))


def matches(left, right, key):
    index = defaultdict(list)
    for i, it in enumerate(left):
        index[key(it)].append(i)
    pairs = [(i, j) for j, it in enumerate(right) for i in index[key(it)]]
    return dict(pairs=len(pairs), left_items=len({i for i, j in pairs}),
                right_items=len({j for i, j in pairs}), indices=pairs)


def near_matches(left, right, threshold=.8):
    """Exhaustive binary character 5-gram intersections, in sparse batches; no search/candidate approximation."""
    text = [normalize(it['q']) for it in left + right]
    vocab, indices, indptr = {}, [], [0]
    for s in text:
        grams = {s[i:i + 5] for i in range(max(1, len(s) - 4))}
        indices.extend(sorted(vocab.setdefault(g, len(vocab)) for g in sorted(grams)))
        indptr.append(len(indices))
    matrix = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr),
                        shape=(len(text), len(vocab)))
    a, b = matrix[:len(left)], matrix[len(left):]
    na, nb = np.diff(a.indptr), np.diff(b.indptr)
    out = []
    for start in range(0, len(right), 64):
        dot = (b[start:start + 64] @ a.T).tocoo()
        sim = dot.data / (nb[start + dot.row] + na[dot.col] - dot.data)
        for k in np.flatnonzero(sim >= threshold):
            i, j = int(dot.col[k]), start + int(dot.row[k])
            if text[i] != text[len(left) + j]:
                out.append(dict(left=i, right=j, jaccard=float(sim[k]),
                                same_choices=left[i]['choices'] == right[j]['choices']))
    return sorted(out, key=lambda r: (r['right'], r['left']))


def paired(a, b):
    """Paired binary accuracy difference, exact McNemar p, conservative population difference CI."""
    if len(a) != len(b) or any(x not in (0, 1) for x in list(a) + list(b)):
        raise ValueError('equal-length binary correctness vectors required')
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n01 = sum(y == 1 and x == 0 for x, y in zip(a, b))
    N, n = len(a), n10 + n01
    if not N:
        return dict(n=0, n10=0, n01=0, diff=None, lo=None, hi=None, p=None)

    def cp(k):
        return (float(beta.ppf(.0125, k, N - k + 1)) if k else 0.,
                float(beta.ppf(.9875, k + 1, N - k)) if k < N else 1.)
    l10, h10 = cp(n10)
    l01, h01 = cp(n01)
    return dict(n=N, n10=n10, n01=n01, diff=(n10 - n01) / N, lo=l10 - h01, hi=h10 - l01,
                p=float(binomtest(n10, n, .5).pvalue) if n else 1.)


def audit(data):
    out = dict(sizes={s: len(v) for s, v in data.items()}, overlap={})
    for left, right in (('train', 'test'), ('train', 'val'), ('val', 'test')):
        out['overlap'][left + '_' + right] = {
            'raw_question': matches(data[left], data[right], lambda it: it['q']),
            'normalized_question': matches(data[left], data[right], lambda it: normalize(it['q'])),
            'raw_input': matches(data[left], data[right], lambda it: (it['q'], tuple(it['choices']))),
            'exact_input': matches(data[left], data[right], input_key),
        }
    out['near_definition'] = 'Casefold, collapse whitespace, retain symbols; character 5-gram Jaccard >= 0.8; exclude normalized exact questions.'
    near = near_matches(data['train'], data['test'])
    out['near_train_test'] = dict(pairs=len(near), test_items=len({r['right'] for r in near}), records=near)
    near_val = near_matches(data['val'], data['test'])
    out['near_val_test'] = dict(pairs=len(near_val), test_items=len({r['right'] for r in near_val}), records=near_val)
    # Formatting-insensitive input duplicates within test: keep the first row, independently of outcomes.
    seen, repeated = set(), []
    for i, it in enumerate(data['test']):
        key = input_key(it)
        if key in seen:
            repeated.append(i)
        seen.add(key)
    out['test_repeated_input_indices'] = repeated
    exact, similar = set(repeated), set(repeated)
    for name in ('train_test', 'val_test'):
        exact.update(j for i, j in out['overlap'][name]['exact_input']['indices'])
        similar.update(j for i, j in out['overlap'][name]['normalized_question']['indices'])
    similar.update(r['right'] for r in near + near_val)
    out['exclude'] = dict(exact_input_overlap=sorted(exact), similar_question_overlap=sorted(similar | exact))
    return out


def rescore(data, scores, exclusions):
    items = data['test']
    expected = [dict(src=it['src'], answer=it['answer']) for it in items]
    if scores.get('questions') != expected:
        raise ValueError('score question order/source/answers do not match data; full test output required')
    c = scores['per_question']
    if any(len(v) != len(items) or any(x not in (0, 1) for x in v) for v in c.values()):
        raise ValueError('score vectors must cover every question with binary outcomes')
    required = ('pair/avg', 'ensemble', 'pair/zero/avg')
    if any(k not in c for k in required):
        raise ValueError('missing pooled probability predictions; cannot average binary correctness to recover them')
    out = {}
    for name, dropped in {'unfiltered': [], **exclusions}.items():
        drop = set(dropped)
        groups = {}
        for src in ('all', 'arc_challenge', 'mmlu'):
            sel = [i for i, it in enumerate(items) if i not in drop and (src == 'all' or it['src'] == src)]
            vectors = {k: [v[i] for i in sel] for k, v in c.items()}
            tests = {k: paired(vectors['pair/avg'], vectors[k]) for k in required[1:]}
            groups[src] = dict(n=len(sel), accuracy={k: sum(v) / len(v) if v else None for k, v in vectors.items()},
                               pooled=tests, pooled_beats_both=bool(sel) and all(
                                   t['diff'] > 0 and t['p'] < .05 for t in tests.values()))
        out[name] = groups
    return out


def self_test():
    """Exact finite-sample coverage checks of the statistical fix, not simulated model measurements."""
    import math
    assert paired([], [])['diff'] is None
    assert paired([1, 0], [1, 0])['lo'] < 0 < paired([1, 0], [1, 0])['hi']
    assert paired([1] * 6, [0] * 6)['p'] == .03125
    for bad in (([0], []), ([2], [0])):
        try:
            paired(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    intervals = {}
    N = 10
    for i in range(N + 1):
        for j in range(N + 1 - i):
            intervals[i, j] = paired([1] * i + [0] * (N - i), [0] * i + [1] * j + [0] * (N - i - j))
    coverage = {}
    for p10, p01 in ((.2, 0), (.2, .1), (.45, .45), (0, 0), (1, 0)):
        covered = 0.
        for (i, j), r in intervals.items():
            prob = (math.comb(N, i) * math.comb(N - i, j) * p10 ** i * p01 ** j *
                    (1 - p10 - p01) ** (N - i - j))
            covered += prob * (r['lo'] <= p10 - p01 <= r['hi'])
        assert covered >= .95 - 1e-12, (p10, p01, covered)
        coverage[str((p10, p01))] = covered
    # Verify the sparse near-match algorithm against direct set arithmetic on real benchmark questions in main().
    return coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scores')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    raw = Path(args.data).read_bytes()
    data = json.loads(raw)
    result = audit(data)
    result['data_sha256'] = hashlib.sha256(raw).hexdigest()
    result['interval_method'] = 'Bonferroni difference of two 97.5% exact marginal Clopper-Pearson intervals; at least 95% unconditional coverage under iid questions'
    result['limitations'] = ['Lexical matches include shared templates/passages, not just repeated questions.',
                             'Base-model pretraining exposure and semantic paraphrases are not audited.',
                             'Original score files identify questions only by source and answer; order is assumed after that validation.',
                             'Per-question tests do not quantify training-seed uncertainty or shared-passage dependence.']
    if args.self_test:
        result['ci_coverage_tests'] = self_test()
        examples = result['near_train_test']['records'][:16]
        a = [data['train'][r['left']] for r in examples] + data['train'][:4]
        b = [data['test'][r['right']] for r in examples] + data['test'][:4]
        direct = []
        for j, y in enumerate(b):
            for i, x in enumerate(a):
                s, t = normalize(x['q']), normalize(y['q'])
                u, v = ({z[k:k + 5] for k in range(max(1, len(z) - 4))} for z in (s, t))
                if s != t and len(u & v) / len(u | v) >= .8:
                    direct.append((i, j))
        assert [(r['left'], r['right']) for r in near_matches(a, b)] == direct
    if args.scores:
        scores_raw = Path(args.scores).read_bytes()
        result['scores_sha256'] = hashlib.sha256(scores_raw).hexdigest()
        result['rescored'] = rescore(data, json.loads(scores_raw), result['exclude'])
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(sizes=result['sizes'], overlap={k: {m: {f: v for f, v in r.items() if f != 'indices'}
                          for m, r in ms.items()} for k, ms in result['overlap'].items()},
                          near_pairs=result['near_train_test']['pairs'], near_test_items=result['near_train_test']['test_items'],
                          excluded={k: len(v) for k, v in result['exclude'].items()}), indent=2))


if __name__ == '__main__':
    main()
