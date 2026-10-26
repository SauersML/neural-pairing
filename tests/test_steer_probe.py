"""steer_probe.py on a tiny pair: what it reads is the steering vector and the link's write, nothing else. The steered
side's own difference is the steering vector; the partner's states before the steered block are unchanged; at that
block only the link's write differs; a zero steering coefficient reads nothing.
"""
import numpy as np
import pytest

import link
import steer_probe as P

CONCEPTS = ('ocean', 'fire')


@pytest.fixture(scope='module')
def rt(runtime):
    return runtime()


def directions(rt, seed):
    g = np.random.default_rng(seed)
    dim = {x: link.inner(rt.pair.models[x])[0].config.hidden_size for x in link.SIDES}
    vectors = {f'{x}/{c}': g.normal(size=dim[x]).astype(np.float32) for x in link.SIDES for c in CONCEPTS}
    selected = {x: {c: dict(alpha=0.5 + i) for i, c in enumerate(CONCEPTS)} for x in link.SIDES}
    return vectors, selected


def test_steering_write_and_causality(rt):
    at = {d: directions(rt, d) for d in (1, 2, 4)}
    records, means = P.measure(rt, {d: v for d, (v, _) in at.items()}, {d: s for d, (_, s) in at.items()},
                               [(2, 3.), (4, 1.)], bs=2, concepts=CONCEPTS)
    assert len(records) == 2 * 2 * len(CONCEPTS) * 2
    for r in records:
        L = r['layer']
        # The steered side's own difference at L is the steering vector, in its own unit the steering coefficient.
        np.testing.assert_allclose(r['own_check'], r['boost'], rtol=1e-3)
        for d, row in r['depths'].items():
            if d < L:
                # The partner's states before the steered block cannot change.
                assert row['mean'] == row['last'] == row['write'] == row['size'] == 0.
        # At L the partner's own block output is unchanged; only the link's write there differs.
        row = r['depths'][L]
        np.testing.assert_allclose(row['mean'], row['write'], rtol=1e-4, atol=1e-7)
        assert row['size'] > 0
    assert {(m['layer'], m['boost']) for m in means} == {(2, 3.), (4, 1.)}


def test_zero_steering_reads_nothing(rt):
    v, s = directions(rt, 7)
    records, _ = P.measure(rt, {2: v, 4: v}, {2: s, 4: s}, [(2, 0.)], bs=2, concepts=CONCEPTS)
    for r in records:
        assert r['own_check'] == 0.
        for row in r['depths'].values():
            assert row['mean'] == row['write'] == row['size'] == 0.
