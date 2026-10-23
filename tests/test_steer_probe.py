"""steer_probe.py on tiny Qwen/AFM pairs: what it reads is the push and the link's write, nothing else.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python -m pytest -q tests/test_steer_probe.py
"""
import os
os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import numpy as np
import torch
import link
import steered_mind as X
import steer_probe as P

XB = (1, 2, 4)
CONCEPTS = ('ocean', 'fire')


def runtime():
    torch.manual_seed(0)
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], models[x] = link.load(x)
    pair = link.Pair(models, toks, XB, heads=2, head_dim=16).eval()
    for side in pair.bridge.values():
        side.alpha.data.fill_(0.2)                   # alpha starts at 0: no write at all
    rt = X.Runtime.__new__(X.Runtime)
    rt.backend, rt.pair, rt.toks = 'torch', pair, toks
    return rt


def directions(seed):
    g = np.random.default_rng(seed)
    dim = {x: link.inner(runtime_cache.pair.models[x])[0].config.hidden_size for x in link.SIDES}
    vectors = {f'{x}/{c}': g.normal(size=dim[x]).astype(np.float32) for x in link.SIDES for c in CONCEPTS}
    selected = {x: {c: dict(alpha=0.5 + i) for i, c in enumerate(CONCEPTS)} for x in link.SIDES}
    return vectors, selected


runtime_cache = runtime()


def test_push_write_and_causality():
    rt = runtime_cache
    at = {d: directions(d) for d in (1, 2, 4)}
    records, means = P.measure(rt, {d: v for d, (v, _) in at.items()}, {d: s for d, (_, s) in at.items()},
                               [(2, 3.), (4, 1.)], bs=2, concepts=CONCEPTS)
    assert len(records) == 2 * 2 * len(CONCEPTS) * 2
    for r in records:
        L = r['layer']
        # The steered side's own difference at L is the push itself, in its own unit the boost.
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


def test_zero_push_reads_nothing():
    rt = runtime_cache
    v, s = directions(7)
    records, _ = P.measure(rt, {2: v, 4: v}, {2: s, 4: s}, [(2, 0.)], bs=2, concepts=CONCEPTS)
    for r in records:
        assert r['own_check'] == 0.
        for row in r['depths'].values():
            assert row['mean'] == row['write'] == row['size'] == 0.
