"""int4 bases in MLX (mlx_pair.load_base) on tiny random models of both architectures saved in bfloat16: the embeddings,
projections and (untied) LM head load quantized, and LoRA adapters attach on top.
"""
import gc

import pytest

mx = pytest.importorskip('mlx.core')
import mlx.nn as nn
import torch

import link
import mlx_pair as M


def test_int4_bases_load_quantized(tmp_path):
    mx.set_cache_limit(32 * 2**20)
    for x in M.SIDES:
        _, original = link.load(x)
        original.to(torch.bfloat16).save_pretrained(tmp_path / x)
        del original
        gc.collect()
        model, _ = M.load_base(str(tmp_path / x))
        assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
        assert isinstance(model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        if x == 'ember':
            assert isinstance(model.lm_head, nn.QuantizedLinear)
            assert not hasattr(model.layers[0].mlp, 'gate_proj')
        M.add_lora(model, 32)
        assert isinstance(model.layers[0].self_attn.q_proj.base, nn.QuantizedLinear)
