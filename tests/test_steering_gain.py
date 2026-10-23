"""steering_scale.write_gain on a tiny Qwen/AFM pair: the write after the chosen block, and only it, scales by the gain.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python -m pytest -q tests/test_steering_gain.py
"""
import os
os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import numpy as np
import pytest
import torch
import steered_mind as X
import steer_probe as P
import steering_scale as S
from test_steer_probe import runtime

DEPTHS = (1, 2, 4)


def writes(rt):
    eps = [X.episode('ember', 'chat', p) for p in range(2)]
    push = {'quill': np.ones((2, rt.pair.models['quill'].config.hidden_size), np.float32)}
    return P.prompt_pass(rt, eps, push, DEPTHS)[1]


def test_gain_scales_only_its_exchange():
    rt = runtime()
    rt.layer = 2
    before = writes(rt)
    sizes = S.write_gain(rt.pair, 2, 3.)
    after = writes(rt)
    for x in X.SIDES:
        t = np.tanh(0.2)
        np.testing.assert_allclose(sizes[x], [t, 3 * t, t], rtol=1e-4)
        assert torch.equal(after[(x, 1)], before[(x, 1)])
        torch.testing.assert_close(after[(x, 2)], 3 * before[(x, 2)], rtol=1e-4, atol=1e-7)
        assert before[(x, 2)].abs().sum() > 0


def test_gain_stays_below_one():
    rt = runtime()
    with pytest.raises(ValueError):
        S.write_gain(rt.pair, 2, 6.)
