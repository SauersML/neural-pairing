"""Real tiny Qwen/AFM checks, run with MPD_MEM_GIB=8; MLX on CPU.

This is experiment-owned verification, not a replacement model or scorer.
"""
import os
os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1', MLX_ENABLE_TF32='0')
import json
from pathlib import Path
import sys
import numpy as np
import torch
import mlx.core as mx
mx.set_default_device(mx.cpu)
torch.set_num_threads(2)
import link
import mlx_pair as M
import mlx_eval as V
import session as S
import steered_mind as X


def main():
    toks, models = {}, {}
    for x in X.SIDES:
        toks[x], tm = link.load(x)
        models[x] = M.make_model(tm.config.to_dict())
        models[x].load_weights([(k, M.from_torch(v)) for k, v in tm.state_dict().items()
                               if k != 'lm_head.weight' or not tm.config.tie_word_embeddings])
        del tm
    mp = M.Pair(models, (1, 2, 4), heads=2, head_dim=16, checkpoint=False)
    for x in X.SIDES:
        mp.bridge[x].alpha = mx.full((3,), .4)
    rt = X.Runtime.__new__(X.Runtime)
    rt.backend, rt.pair, rt.toks = 'mlx', V.Pair(mp, toks), toks
    rng = np.random.default_rng(21022)
    eps = [dict(mode='duplex', system={x: 'You are ' + x + '.' for x in X.SIDES},
                task={x: ['Say hello.', 'Describe a pleasant afternoon.'][i] for x in X.SIDES},
                reply={x: ('Hello, how are you?', 1.) for x in X.SIDES}) for i in range(2)]
    batch = S.collate([S.build(e, toks) for e in eps], {x: toks[x].pad_token_id for x in X.SIDES})
    vec = rng.normal(0, .03, (2, 128)).astype(np.float32)
    checks = {}
    for sender in X.SIDES:
        receiver = X.SIDES[1 - X.SIDES.index(sender)]
        for mode in (None, 'zero'):
            base, _ = rt.pair.joint(batch, mode)
            with rt.steer({sender: np.zeros_like(vec)}, layer_index=2):
                zero, _ = rt.pair.joint(batch, mode)
            assert all(torch.equal(base[x], zero[x]) for x in X.SIDES)
            with rt.steer({sender: vec}, layer_index=2):
                steered, _ = rt.pair.joint(batch, mode)
                generated = S.generate(rt.pair, eps, mode=mode, force=True, keep=True)
            assert not torch.equal(steered[sender], base[sender])
            if mode == 'zero':
                assert torch.equal(steered[receiver], base[receiver])
            else:
                assert not torch.equal(steered[receiver], base[receiver])
            worst = 0.
            for i, ep in enumerate(eps):
                built = S.build(ep, toks)
                for x in X.SIDES:
                    a, b = steered[x][i, :len(built[x]['ids'])], generated[i]['h'][x]
                    worst = max(worst, float((a - b).abs().max()))
                    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
            checks[f'{sender}/{mode}'] = dict(zero_exact=True, joint_stream_max_error=worst,
                                             receiver_change=float((steered[receiver] - base[receiver]).abs().max()))
    original = V.block
    try:
        with rt.steer({'ember': vec}, layer_index=2):
            raise RuntimeError('restoration check')
    except RuntimeError:
        pass
    assert V.block is original
    # Whole-word, suffix, same-tick and preceding-sender guards with real tokens.
    for c in X.CONCEPTS:
        for r in X.SIDES:
            for structure in ('chat', 'duplex'):
                for p in range(2):
                    e = X.episode(r, structure, p)
                    assert not any(X.matches(str(e[k][x] or ''), c) for k in ('system', 'task') for x in X.SIDES)
    assert not X.matches('category', 'cats') and X.matches('cats', 'cats')
    assert not X.matches('Ember', 'fire') and X.matches('embers', 'fire')
    for x in X.SIDES:
        end = toks[x].convert_tokens_to_ids('<|im_end|>')
        assert X.first_tick(toks[x], S.ids(toks[x], 'A category.') + [end], 'cats') is None
        assert X.first_tick(toks[x], S.ids(toks[x], 'A cat.') + [end], 'cats') is not None
    assert X.exact_sign([1] * 12)['p'] == 1 / 4096
    assert X.exact_sign([0] * 12) == dict(wins=0, losses=0, ties=12, p=1.)
    rows = [dict(alpha=a, mention=(a >= .5 and p < 2), fluent_proxy=True)
            for a in X.STRENGTHS for p in range(3)]
    assert X.choose(rows)['alpha'] == .5 and X.choose(rows)['calibrated']
    assert not X.fluent('cat cat cat cat cat cat cat cat cat cat')
    assert X.fluent('A kitten sleeps beside the window.')
    print(json.dumps(dict(checks=checks, lexical_guards=True, prompt_audit=True,
                          exact_tests=True, calibration_rule=True, restoration=True), indent=2))


if __name__ == '__main__':
    main()
