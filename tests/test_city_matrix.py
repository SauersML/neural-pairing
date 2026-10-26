"""city_matrix.py on a tiny pair: complete-city log-probabilities equal ordinary full-candidate passes for both
receivers under link, cut and stranger, and the exact permutation counts equal exhaustive enumeration.
"""
import itertools

import numpy as np
import pytest
import torch

import city_matrix as CM
import link
import session as S

CANDIDATES = ['Beijing', 'Cairo']


@pytest.mark.parametrize('mode', [None, 'zero', 'shuffle'])
@pytest.mark.parametrize('x', link.SIDES)
def test_score_pair_equals_full_candidate_passes(pair, x, mode):
    toks = pair.toks
    pads = {y: toks[y].pad_token_id for y in link.SIDES}
    template = dict(receiver=x, own_city='Accra', prefix="So, you're in ")
    fast = CM.score_pair(pair, toks, template, CANDIDATES, CANDIDATES, mode)
    for j, candidate in enumerate(CANDIDATES):
        eps = [CM.episode(template, c, candidate) for c in CANDIDATES]
        built = [S.build(ep, toks) for ep in eps]
        b = S.collate(built, pads)
        with torch.no_grad():
            h, _ = pair.joint(b, mode=mode)
            lp = pair.token_logits(x, h[x], b[x]['idx']).log_softmax(-1)
        off, slow = 0, []
        for ep, e in zip(eps, built):
            js = S.payload_labels(ep, toks)[x]
            slow.append(sum(float(lp[off + z, e[x]['labels'][z][1]]) for z in js))
            off += len(e[x]['labels'])
        np.testing.assert_allclose(fast[:, j], slow, atol=3e-5, rtol=0)


def test_exact_permutation_counts():
    top = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    brute = [sum(top[i, p[i]] for i in range(4)) for p in itertools.permutations(range(4))]
    assert CM.exact_permutation(top)['counts'] == [brute.count(i) for i in range(5)]
    assert CM.exact_permutation(np.eye(12))['numerator'] == 1
