"""steering_scale.py: the projection calibration algebra, and state collection and steering on a tiny pair with both
backends.
"""
import numpy as np
import pytest
import torch
import steering_scale as C
import steered_mind as X


def test_projection_gap_and_background_invariance():
    neutral = np.array([[1000., 1., 0.], [1000., -1., 0.]], dtype=np.float32)
    positive = neutral + np.array([0., 3., 4.], dtype=np.float32)
    d, r = C.projection(positive, neutral)
    np.testing.assert_allclose(d, [0., 3., 4.])
    assert r['norm'] == r['gap'] == 5.
    np.testing.assert_allclose((neutral + d) @ (d / r['norm']), r['positive'])
    # A huge shared sink increases the residual norm but not the concept step.
    offset = np.array([1e6, 0., 0.], dtype=np.float32)
    d2, r2 = C.projection(positive + offset, neutral + offset)
    np.testing.assert_array_equal(d, d2)
    assert r2['gap'] == r['gap']
    with pytest.raises(ValueError):
        C.projection(neutral, neutral)


def test_sink_mask_uses_only_neutral_energy_and_keeps_ordinary_geometry():
    ordinary = np.ones((3, 16), dtype=np.float32)
    np.testing.assert_array_equal(C.sink_mask(ordinary), np.ones(16))
    extreme = ordinary.copy()
    extreme[:, [2, 5, 8, 11]] = 1000
    mask = C.sink_mask(extreme)
    assert set(np.flatnonzero(mask == 0)) == {2, 5, 8, 11}
    neutral = np.zeros((3, 16), dtype=np.float32)
    positive = neutral.copy()
    positive[:, 2] = 100
    positive[:, 1] = 3
    positive[:, 7] = 4
    d, r = C.projection(positive, neutral, mask)
    assert d[2] == 0 and r['norm'] == r['gap'] == 5.


def rows():
    return [dict(alpha=a, prompt=p, mention=(a >= 1. and p < 2), fluent_proxy=True)
            for a in C.STRENGTHS for p in range(3)]


def test_selection_preserves_threshold_fluency_and_fallback():
    r = rows()
    assert C.choose(r) == dict(alpha=1., calibrated=True, baseline_mentions=0)
    r[0]['mention'] = True
    assert not C.choose(r)['calibrated']  # two mentions are not baseline + two
    r = rows()
    for row in r:
        if row['alpha'] == 1 and row['prompt'] == 2:
            row['fluent_proxy'] = False
    assert C.choose(r)['alpha'] == 1.5
    for row in r:
        row['mention'] = False
        row['fluent_proxy'] = row['alpha'] <= .5
    assert C.choose(r) == dict(alpha=.5, calibrated=False, baseline_mentions=0)
    for row in r:
        row['fluent_proxy'] = False
    assert C.choose(r)['alpha'] == 0.
    with pytest.raises(ValueError):
        C.choose(r[:-1])
    with pytest.raises(ValueError):
        C.choose(r + [r[0]])


def test_original_preregistration_unchanged():
    assert X.digest(X.protocol()) == '6d62a92c2e5ff5418ef934986b202864cc7a1e2330a68460c65c9f24d770f545'
    assert C.amendment()['layer'] == 15
    # --layer leaves the block-15 amendment (and so its hash) as registered.
    assert C.amendment('sink_free', 15) == C.amendment()
    assert C.amendment('sink_free', 6)['layer'] == 6 and 'layer' not in C.amendment('sink_free', 6)['unchanged']
    assert X.Runtime.layer == 15
    assert C.amendment('sink_free', 15, 1.) == C.amendment() and C.amendment('sink_free', 27, 4.)['boost'] == 4.
    assert X.exact_sign([1] * 12)['p'] == 1 / 4096


def test_real_state_collection_and_steering_both_backends():
    mx = pytest.importorskip('mlx.core')
    import link
    import mlx_pair as M
    import mlx_eval as V
    import session as S
    mx.set_default_device(mx.cpu)
    models, mm, toks = {}, {}, {}
    for x in X.SIDES:
        toks[x], models[x] = link.load(x)
        models[x].eval()
        mm[x] = M.make_model(models[x].config.to_dict())
        mm[x].load_weights([(k, M.from_torch(v)) for k, v in models[x].state_dict().items()
                           if k != 'lm_head.weight' or not models[x].config.tie_word_embeddings])
    tp = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    mp = M.Pair(mm, (1, 2, 4), heads=2, head_dim=16)
    for x in X.SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in tp.bridge[x].state_dict().items()])
    runtimes = []
    for backend, pair in [('torch', tp), ('mlx', V.Pair(mp, toks))]:
        rt = X.Runtime.__new__(X.Runtime)
        rt.backend, rt.pair, rt.toks = backend, pair, toks
        runtimes.append(rt)
    texts = ['A cat sits beside the window.', 'A clerk opens the document.']
    for x in X.SIDES:
        collected = [C.collect(rt, x, texts, layers=(1, 2, 4)) for rt in runtimes]
        for l in (1, 2, 4):
            for field in ('means', 'energy', 'rms'):
                np.testing.assert_allclose(collected[0][l][field], collected[1][l][field], atol=2e-6, rtol=2e-5)
        # Independent full-decoder hook validates the actual zero-based layer.
        got = []
        handle = link.find_layers(models[x])[2].register_forward_hook(
            lambda mod, inp, out: got.append((out[0] if isinstance(out, tuple) else out)[0, 1:].detach()))
        try:
            with torch.inference_mode():
                for text in texts:
                    models[x](input_ids=torch.tensor([S.ids(toks[x], text)]))
        finally:
            handle.remove()
        np.testing.assert_allclose(collected[0][2]['means'], torch.stack([a.mean(0) for a in got]).numpy(), atol=2e-6)
        vec, _ = C.projection(collected[0][2]['means'][:1], collected[0][2]['means'][1:])
        eps = [dict(mode='duplex', system={s: 'You are ' + s for s in X.SIDES},
                    task={s: 'Say hello.' for s in X.SIDES},
                    reply={s: ('Hello friend.', 1.) for s in X.SIDES})]
        batch = S.collate([S.build(e, toks) for e in eps], {s: toks[s].pad_token_id for s in X.SIDES})
        for rt in runtimes:
            with torch.inference_mode():
                base, _ = rt.pair.joint(batch, 'zero')
                with rt.steer({x: vec[None]}, layer_index=2):
                    h, _ = rt.pair.joint(batch, 'zero')
                    generated = S.generate(rt.pair, eps, mode='zero', force=True, keep=True)
                partner = X.SIDES[1 - X.SIDES.index(x)]
                assert torch.equal(h[partner], base[partner])
                assert not torch.equal(h[x], base[x])
                for s in X.SIDES:
                    torch.testing.assert_close(h[s][0], generated[0]['h'][s], atol=3e-5, rtol=3e-5)
