"""The MLX pair (mlx_pair.py, run by mlx_eval.Pair) against the torch pair on tiny random models of both architectures,
with random rank-32 LoRA and open bridges: logits and gates agree under link, cut and shuffle, and an exported
checkpoint reads back into evaluate.load_pair and mlx_pair.load_checkpoint with the same logits.
"""
from argparse import Namespace

import pytest

mx = pytest.importorskip('mlx.core')
import numpy as np
from peft import LoraConfig, get_peft_model
import torch

import evaluate
import link
import mlx_eval as V
import mlx_pair as M
import session as S


@pytest.fixture(scope='module')
def setup():
    torch.manual_seed(17)
    mx.random.seed(17)
    mx.set_cache_limit(32 * 2**20)
    models, toks, mm = {}, {}, {}
    for x in M.SIDES:
        toks[x], m = link.load(x)
        mm[x] = M.make_model(m.config.to_dict())
        mm[x].load_weights([(k, M.from_torch(v)) for k, v in m.state_dict().items()
                            if k != 'lm_head.weight' or not m.config.tie_word_embeddings], strict=True)
        M.add_lora(mm[x], 32)
        m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, target_modules=list(M.TARGETS)))
        for name, mod in m.get_base_model().named_modules():
            if hasattr(mod, 'lora_A'):
                torch.nn.init.normal_(mod.lora_B['default'].weight, std=.006)
                ml = dict(mm[x].named_modules())[name]
                ml.lora_A = M.from_torch(mod.lora_A['default'].weight)
                ml.lora_B = M.from_torch(mod.lora_B['default'].weight)
        models[x] = m.eval()
    xb, sizes = (1, 2, 4), {x: [.3, .6, .9] for x in M.SIDES}
    pair = link.Pair(models, toks, xb, sizes, heads=8, head_dim=128, gate_norm=True)
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.uniform_(-.6, .6)
            side.null.normal_(0, .3)
            side.age.normal_(0, .4)
            for g in side.gate:
                g.weight.normal_(0, .08)
                g.bias.normal_(0, .2)
            for n in list(side.q_norm) + list(side.kv_norm):
                n.weight.uniform_(.8, 1.2)
    mp = M.Pair(mm, xb, sizes, heads=8, head_dim=128, gate_norm=True)
    for x in M.SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in pair.bridge[x].state_dict().items()])
    mx.eval(mp.parameters())
    system = {x: 'You are ' + x + '.' for x in M.SIDES}
    eps = [dict(game='tiny', meta={}, mode='duplex', system=system,
                task={'quill': 'Remember the number 42.', 'ember': 'What is the number?'},
                reply={'quill': ('Yes.', .2), 'ember': ('42.', 1.)}),
           dict(game='tiny', meta={}, mode='dialogue', system=system,
                task={'quill': 'Ask a question.', 'ember': None}, link_off_after=1,
                turns=[('quill', 'Blue?', .4), ('ember', 'Green.', 1.), ('quill', 'Okay.', .6)]),
           dict(game='tiny', meta={}, mode='duplex', system=system,
                task={'quill': 'Say a color.', 'ember': 'Say a name.'},
                reply={'quill': ('Red.', 1.), 'ember': ('Al.', .6)})]
    b = S.collate([S.build(e, toks) for e in eps], {x: toks[x].pad_token_id for x in M.SIDES})
    return Namespace(pair=pair, mp=mp, vp=V.Pair(mp, toks), b=b, mb=M.convert_batch(b))


def mlx_logits(s, h, x):
    return np.array(s.mp.token_logits(x, M.from_torch(h[x]), s.mb[x]['idx']))


@pytest.mark.parametrize('mode', ['zero', None, 'shuffle'])
def test_forward_parity(setup, mode):
    s = setup
    with torch.no_grad():
        h, g = s.pair.joint(s.b, mode)
    mh, mg = s.vp.joint(s.b, mode)
    for x in M.SIDES:
        tl = s.pair.token_logits(x, h[x], s.b[x]['idx']).detach().numpy()
        np.testing.assert_allclose(mlx_logits(s, mh, x), tl, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(mg[x].numpy(), g[x].numpy(), atol=2e-6, rtol=2e-5)


def test_checkpoint_round_trip(setup, tmp_path):
    s = setup
    expected, _ = s.vp.joint(s.b)
    M.export_checkpoint(s.mp, tmp_path, {})
    restored, _ = evaluate.load_pair(str(tmp_path))
    ck = torch.load(tmp_path / 'bridge.pt', weights_only=False)
    assert set(ck['bridge']) == set(s.pair.bridge.state_dict())
    with torch.no_grad():
        h, _ = restored.joint(s.b)
    for x in M.SIDES:
        actual = restored.token_logits(x, h[x], s.b[x]['idx']).detach().numpy()
        np.testing.assert_allclose(actual, mlx_logits(s, expected, x), atol=2e-5, rtol=2e-5)
    for x in M.SIDES:                    # wipe the bridges' alpha and every lora_B, so the load must restore them
        s.mp.bridge[x].alpha = mx.zeros_like(s.mp.bridge[x].alpha)
        for _, mod in s.mp.models[x].named_modules():
            if isinstance(mod, M.LoRA):
                mod.lora_B = mx.zeros_like(mod.lora_B)
    M.load_checkpoint(s.mp, str(tmp_path))
    again, _ = s.vp.joint(s.b)
    for x in M.SIDES:
        np.testing.assert_allclose(again[x].numpy(), expected[x].numpy(), atol=1e-7)
