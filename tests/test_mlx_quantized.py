"""No-mock streamed int4 loading and training on both actual NL_TINY architectures.
Run: MPD_MEM_GIB=2 HF_HUB_OFFLINE=1 python tests/test_mlx_quantized.py
"""
import os
os.environ.update(NL_DEVICE='cpu', NL_TINY='1', HF_HUB_OFFLINE='1', MLX_ENABLE_TF32='0')
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import gc
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from argparse import Namespace
import link
import session as S
import mlx_pair as M
from mlx_train import objective, make_optimizer

torch.set_num_threads(2)
mx.set_cache_limit(32 * 2**20)
mx.random.seed(3)
models, toks = {}, {}
with tempfile.TemporaryDirectory() as tmp:
    for x in M.SIDES:
        toks[x], original = link.load(x)
        original.to(torch.bfloat16).save_pretrained(Path(tmp) / x)
        del original
        gc.collect()
        models[x], _ = M.load_base(str(Path(tmp) / x))
        assert isinstance(models[x].model.embed_tokens, nn.QuantizedEmbedding)
        assert isinstance(models[x].layers[0].self_attn.q_proj, nn.QuantizedLinear)
        if x == 'ember':
            assert isinstance(models[x].lm_head, nn.QuantizedLinear)
            assert not hasattr(models[x].layers[0].mlp, 'gate_proj')
        M.add_lora(models[x], 32)
    pair = M.Pair(models, (1, 2, 4), heads=8, head_dim=128, checkpoint=True)
    for s in pair.bridge.values():
        s.alpha = mx.full(s.alpha.shape, np.arctanh(.1).item())
        for o in s.o:
            o.weight = mx.zeros_like(o.weight)
    sy = {x: 'You are ' + x + '.' for x in M.SIDES}
    eps = [dict(mode='duplex', system=sy, task={'quill': 'Read blue.', 'ember': 'What color?'},
                reply={'quill': ('Okay.', .3), 'ember': ('Blue.', 1.)}),
           dict(mode='duplex', system=sy, task={'quill': 'Read 14.', 'ember': 'What number?'},
                reply={'quill': ('Yes.', .3), 'ember': ('14.', 1.)})]
    built = [S.build(e, toks) for e in eps]
    b = M.convert_batch(S.collate(built, {x: toks[x].pad_token_id for x in M.SIDES}))
    denom = {x: max(1., float(mx.sum(b[x]['w']))) for x in M.SIDES}
    n_show = {x: int(mx.sum(b[x]['att'])) * 3 for x in M.SIDES}
    base = {x: np.array(models[x].layers[0].self_attn.q_proj.base.weight).copy() for x in M.SIDES}
    opt, _ = make_optimizer(Namespace(lr=1e-4, lr_alpha=3e-3, lr_bridge=3e-4,
                                      lr_bridge_quill=None, lr_bridge_ember=None))
    for step in range(3):
        (loss, _), grads = nn.value_and_grad(pair, lambda p: objective(p, b, {}, denom, n_show, .01, chunk=4))(pair)
        mx.eval(loss, grads)
        assert np.isfinite(float(loss))
        assert all(np.all(np.isfinite(np.array(g))) for _, g in tree_flatten(grads))
        assert not any('.base.' in k or k.endswith('.size') for k, _ in tree_flatten(grads))
        opt.update(pair, grads)
        mx.eval(pair.parameters(), opt.state)
        print('int4 step', step + 1, 'loss', float(loss), 'peak GiB', mx.get_peak_memory() / 2**30, flush=True)
    for x in M.SIDES:
        np.testing.assert_array_equal(base[x], np.array(models[x].layers[0].self_attn.q_proj.base.weight))
        assert float(mx.max(mx.abs(pair.bridge[x].o[0].weight))) > 0
        assert float(mx.max(mx.abs(models[x].layers[0].self_attn.q_proj.lora_B))) > 0
print('PASS: streamed BF16 -> int4 bases remain frozen; both LoRAs and bridges train', flush=True)
