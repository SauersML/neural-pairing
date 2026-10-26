"""steering_scale.write_gain on a tiny pair: the write after the chosen block, and only that write, scales by the gain,
and a gain that would take the write size to one or more is rejected.
"""
import numpy as np
import pytest
import torch

import steered_mind as X
import steer_probe as P
import steering_scale as S

DEPTHS = (1, 2, 4)


def writes(rt):
    eps = [X.episode('ember', 'chat', p) for p in range(2)]
    steer = {'quill': np.ones((2, rt.pair.models['quill'].config.hidden_size), np.float32)}
    return P.prompt_pass(rt, eps, steer, DEPTHS)[1]


def test_gain_scales_only_its_exchange(runtime):
    rt = runtime()
    rt.layer = 2
    before = writes(rt)
    sizes = S.write_gain(rt.pair, 2, 3.)
    after = writes(rt)
    t = np.tanh(0.2)
    for x in X.SIDES:
        np.testing.assert_allclose(sizes[x], [t, 3 * t, t], rtol=1e-4)
        assert torch.equal(after[(x, 1)], before[(x, 1)])
        torch.testing.assert_close(after[(x, 2)], 3 * before[(x, 2)], rtol=1e-4, atol=1e-7)
        assert before[(x, 2)].abs().sum() > 0


def test_gain_stays_below_one(runtime):
    with pytest.raises(ValueError):
        S.write_gain(runtime().pair, 2, 6.)
