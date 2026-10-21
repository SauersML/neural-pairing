"""Training-only MLX twin of link.Pair and train.py; frozen 4-bit bases, PEFT exports.

Run with HF_HUB_OFFLINE=1 MPD_MEM_GIB=12 python pairing/mlx_pair.py --help.
The data builders are the original Python/torch builders, on CPU. No HF model
is constructed during real-model loading; tensors are quantized one at a time.
"""
import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ['NL_DEVICE'] = 'cpu'
# M5 MLX defaults to TF32 for float32 GEMMs. The bridges promise float32
# math, and TF32 otherwise introduces ~1e-3 errors in the parity test.
os.environ['MLX_ENABLE_TF32'] = '0'

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
import numpy as np
from mlx_lm.models import llama, qwen3

SIDES = ('quill', 'ember')
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
HF = {'quill': os.environ.get('NL_QUILL', 'Qwen/Qwen3-4B'),
      'ember': os.environ.get('NL_EMBER', 'arcee-ai/AFM-4.5B')}
XBLOCKS = (6, 9, 12, 15, 18, 21, 24, 27)


def other(x):
    return SIDES[1 - SIDES.index(x)]


class ArceeMLP(nn.Module):
    """AFM's NON-gated squared-ReLU MLP (there is no gate_proj)."""
    def __init__(self, args):
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, h):
        return self.down_proj(mx.square(mx.maximum(self.up_proj(h), 0)))


def make_model(config):
    config = dict(config)
    if config.get('rope_parameters'):
        config['rope_theta'] = config['rope_parameters']['rope_theta']
        config['rope_scaling'] = config['rope_parameters']
    if config['model_type'] == 'qwen3':
        return qwen3.Model(qwen3.ModelArgs.from_dict(config))
    if config['model_type'] != 'arcee' or config.get('hidden_act', 'relu2') != 'relu2':
        raise ValueError('Only Qwen3 and Arcee relu2 are supported')
    args = llama.ModelArgs.from_dict(config)
    model = llama.Model(args)
    for layer in model.layers:
        layer.mlp = ArceeMLP(args)
    return model


def from_torch(t):
    import torch
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.uint16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


def load_base(hf, bits=4):
    """Stream cached safetensors, avoiding a full bf16 (or float32) base copy."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = Path(hf) if Path(hf).is_dir() else Path(hf_hub_download(hf, 'config.json', local_files_only=True)).parent
    config = json.loads((path / 'config.json').read_text())
    model = make_model(config)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    modules = dict(model.named_modules())
    index = path / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    expected = set(dict(tree_flatten(model.parameters())))
    loaded = set()
    for file in files:
        with safe_open(str(path / file), framework='pt', device='cpu') as sf:
            for key in sf.keys():
                if key == 'lm_head.weight' and config.get('tie_word_embeddings'):
                    continue
                if key not in expected:
                    raise ValueError(f'Unexpected base tensor {key}')
                t = sf.get_tensor(key)
                w = from_torch(t)
                prefix, name = key.rsplit('.', 1)
                mod = modules[prefix]
                if name == 'weight' and isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                    weight, scales, biases = mx.quantize(w, group_size=64, bits=bits)
                    mod.update(dict(weight=weight, scales=scales, biases=biases))
                    mx.eval(weight, scales, biases)
                    loaded.update(prefix + '.' + n for n in ('weight', 'scales', 'biases'))
                else:
                    mod[name] = w
                    mx.eval(w)
                    loaded.add(key)
                del t, w
        gc.collect()
        mx.clear_cache()
    if expected != loaded:
        raise ValueError(f'Missing weights: {expected - loaded}')
    model.freeze()
    return model, config


class LoRA(nn.Module):
    """PEFT orientation, scale alpha/r=2, fp32 trainable matrices."""
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        out, inp = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            inp *= 32 // base.bits
        self.lora_A = mx.random.uniform(-inp ** -0.5, inp ** -0.5, (rank, inp))
        self.lora_B = mx.zeros((out, rank))
        self.enabled = True
        base.freeze()

    def __call__(self, h):
        y = self.base(h)
        if self.enabled:
            z = (h.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
            y = y + (2.0 * z).astype(y.dtype)
        return y


def add_lora(model, rank):
    model.freeze()
    replace = [(n, LoRA(m, rank)) for n, m in model.named_modules()
               if n.split('.')[-1] in TARGETS and isinstance(m, (nn.Linear, nn.QuantizedLinear))]
    model.update_modules(tree_unflatten(replace))
    return model


@contextlib.contextmanager
def adapters_off(model):
    mods = [m for _, m in model.named_modules() if isinstance(m, LoRA)]
    was = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, enabled in zip(mods, was):
            m.enabled = enabled


def linear(inp, out):
    m = nn.Linear(inp, out, bias=False)
    m.weight = mx.random.normal((out, inp)) * inp ** -0.5
    return m


def age_buckets(tq, tk):
    d = mx.maximum(tq[:, :, None] - tk[:, None, :], 0)
    b = mx.where(d == 0, 0, mx.floor(mx.log2(mx.maximum(d, 1).astype(mx.float32))).astype(mx.int32) + 1)
    return mx.where(tk[:, None, :] < 0, 8, mx.minimum(b, 7))


class Side(nn.Module):
    """Names, shapes, initialization and float32 operations mirror link.Side."""
    def __init__(self, d, n, heads, head_dim, sizes, gate_norm=False):
        super().__init__()
        self.heads, self.head_dim, self.gate_norm = heads, head_dim, gate_norm
        w = heads * head_dim
        self.q_norm = [nn.RMSNorm(d, eps=1e-6) for _ in range(n)]
        self.q = [linear(d, w) for _ in range(n)]
        self.o = [linear(w, d) for _ in range(n)]
        self.alpha = mx.zeros((n,))
        self.null = mx.zeros((n, heads, head_dim))
        self.age = mx.zeros((n, heads, 9))
        self.kv_norm = [nn.RMSNorm(d, eps=1e-6) for _ in range(n)]
        self.k = [linear(d, w) for _ in range(n)]
        self.v = [linear(d, w) for _ in range(n)]
        self.gate = [nn.Linear(d, 1) for _ in range(n)]
        for g in self.gate:
            g.weight = mx.zeros_like(g.weight)
            g.bias = mx.full((1,), 2.0)
        self.size = mx.array(sizes, dtype=mx.float32)
        self.freeze(keys=['size'], recurse=False)

    def split(self, h):
        return h.reshape(*h.shape[:-1], self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def show(self, k, h):
        h = h.astype(mx.float32)
        z = self.kv_norm[k](h)
        gh = mx.fast.rms_norm(h, None, 1e-6) if self.gate_norm else h
        s = mx.sigmoid(self.gate[k](gh).squeeze(-1))
        return self.split(self.k[k](z)), self.split(self.v[k](z) * s[..., None]), s

    def read(self, k, h, keys, vals, allow, ages, qmask):
        allow, ages, qmask = (mx.stop_gradient(a) for a in (allow, ages, qmask))
        B, L = h.shape[:2]
        q = self.split(self.q[k](self.q_norm[k](h.astype(mx.float32))))
        nk = mx.broadcast_to(self.null[k][None, :, None, :], (B, self.heads, 1, self.head_dim))
        keys = mx.concatenate([nk, keys], axis=2)
        vals = mx.concatenate([mx.zeros_like(nk), vals], axis=2)
        bias = self.age[k][:, ages].transpose(1, 0, 2, 3)
        bias = mx.concatenate([mx.zeros((B, self.heads, L, 1)), bias], axis=-1)
        allow = mx.concatenate([mx.ones((B, L, 1), dtype=mx.bool_), allow], axis=-1)
        bias = mx.where(allow[:, None], bias, -float('inf'))
        a = mx.fast.scaled_dot_product_attention(q, keys, vals, scale=self.head_dim ** -0.5, mask=bias)
        w = mx.tanh(self.alpha[k]) * self.size[k] * self.o[k](a.transpose(0, 2, 1, 3).reshape(B, L, -1))
        return w * qmask[..., None]


def remat(module, fn, *args):
    def call(params, *inputs):
        module.update(params)
        return fn(*inputs)
    return mx.checkpoint(call)(module.trainable_parameters(), *args)


def causal_mask(b):
    L = b['ids'].shape[1]
    # session.collate always uses right padding and positions arange(L).
    return (mx.arange(L)[:, None] >= mx.arange(L)[None, :])[None, None] & b['att'][:, None, None, :].astype(mx.bool_)


class Pair(nn.Module):
    def __init__(self, models, xblocks=XBLOCKS, sizes=None, heads=8, head_dim=128, gate_norm=False, checkpoint=True):
        super().__init__()
        self.models = models
        self.xblocks = tuple(xblocks)
        self.checkpoint = checkpoint
        assert self.xblocks and max(self.xblocks) < min(len(m.layers) for m in models.values()) - 1
        self.bridge = {x: Side(models[x].args.hidden_size, len(xblocks), heads, head_dim,
                               sizes[x] if sizes else [1.] * len(xblocks), gate_norm) for x in SIDES}

    def exchange(self, k, hq, he, allow, ages, qmask):
        h = dict(quill=hq, ember=he)
        shown = {x: self.bridge[x].show(k, h[x]) for x in SIDES}
        out, writes = {}, {}
        for x in SIDES:
            w = self.bridge[x].read(k, h[x], *shown[other(x)][:2], allow[x], ages[x], qmask[x])
            writes[x] = mx.stop_gradient(mx.sum(mx.mean(mx.square(w), axis=-1) * qmask[x]) /
                                         mx.maximum(mx.sum(qmask[x]), 1) / self.bridge[x].size[k] ** 2)
            out[x] = h[x] + w.astype(h[x].dtype)
        return out['quill'], out['ember'], shown['quill'][2], shown['ember'][2], writes['quill'], writes['ember']

    def checkpoint_exchange(self, k, *inputs):
        # Only this exchange's parameters are checkpoint inputs. Passing the
        # whole pair would create huge zero gradients for unrelated layers.
        params = tree_unflatten([(n, v) for x in SIDES
                                 for name, v in tree_flatten(self.bridge[x].trainable_parameters())
                                 for n in [x + '.' + name]
                                 if name in ('alpha', 'null', 'age') or name.split('.')[1] == str(k)])
        def call(p, *a):
            self.update({'bridge': p})
            return self.exchange(k, *a)
        return mx.checkpoint(call)(params, *inputs)

    def joint(self, b, mode=None):
        if mode not in (None, 'zero', 'shuffle'):
            raise ValueError(mode)
        if mode == 'shuffle' and b['quill']['ids'].shape[0] < 2:
            raise ValueError('shuffle needs at least two episodes')
        h = {x: self.models[x].model.embed_tokens(b[x]['ids']) for x in SIDES}
        masks = {x: causal_mask(b[x]) for x in SIDES}
        allow, ages, qmask = {}, {}, {}
        for x in SIDES:
            ty, vy = b[other(x)]['tick'], b[other(x)]['att'].astype(mx.bool_)
            if mode == 'shuffle':
                shift = 1 if x == SIDES[0] else -1
                ty, vy = mx.roll(ty, shift, axis=0), mx.roll(vy, shift, axis=0)
            allow[x] = (ty[:, None, :] <= b[x]['tick'][:, :, None]) & vy[:, None, :]
            ages[x] = age_buckets(b[x]['tick'], ty)
            qmask[x] = (b[x]['att'].astype(mx.bool_) & (b[x]['tick'] < b[x]['cut'][:, None])).astype(mx.float32)
        gates, writes = {x: [] for x in SIDES}, {x: [] for x in SIDES}
        for l in range(max(len(m.layers) for m in self.models.values())):
            for x in SIDES:
                if l < len(self.models[x].layers):
                    layer = self.models[x].layers[l]
                    h[x] = remat(layer, layer, h[x], masks[x]) if self.checkpoint else layer(h[x], masks[x])
            if l in self.xblocks and mode != 'zero':
                k = self.xblocks.index(l)
                if mode == 'shuffle':
                    # Inverse rolls isolate each mismatched duplex pair at every depth.
                    he = mx.roll(h['ember'], 1, axis=0)
                    aq = dict(allow); ag = dict(ages); qm = dict(qmask)
                    aq['ember'], ag['ember'], qm['ember'] = [mx.roll(v['ember'], 1, axis=0) for v in (allow, ages, qmask)]
                else:
                    he, aq, ag, qm = h['ember'], allow, ages, qmask
                if self.checkpoint:
                    out = self.checkpoint_exchange(k, h['quill'], he, aq, ag, qm)
                else:
                    out = self.exchange(k, h['quill'], he, aq, ag, qm)
                h['quill'], h['ember'] = out[:2]
                gq, ge = out[2:4]
                if mode == 'shuffle':
                    h['ember'], ge = mx.roll(h['ember'], -1, axis=0), mx.roll(ge, -1, axis=0)
                for x, g, w in zip(SIDES, (gq, ge), out[4:]):
                    gates[x].append(g); writes[x].append(w)
        return ({x: self.models[x].model.norm(h[x]) for x in SIDES},
                {x: mx.stack(gates[x], axis=-1) if gates[x] else mx.zeros((*b[x]['ids'].shape, len(self.xblocks))) for x in SIDES},
                {x: mx.stack(writes[x]) if writes[x] else mx.zeros((len(self.xblocks),)) for x in SIDES})

    def token_logits(self, x, h, idx=None):
        if idx is not None:
            h = h.reshape(-1, h.shape[-1])[idx]
        m = self.models[x]
        return (m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)).astype(mx.float32)


def convert_batch(b):
    import torch
    def conv(v):
        if isinstance(v, torch.Tensor):
            a = v.cpu().numpy()
            if a.dtype == np.int64:
                a = a.astype(np.int32)
            return mx.array(a)
        if isinstance(v, dict):
            return {k: conv(w) for k, w in v.items()}
        return v
    return {x: {k: conv(v) for k, v in b[x].items() if k != 'own'} for x in SIDES}


def residual_rms(model, blocks, seqs):
    got = {k: [] for k in blocks}
    for ids in seqs:
        h = model.model.embed_tokens(mx.array([ids]))
        for l, layer in enumerate(model.layers):
            h = layer(h, 'causal')
            if l in got:
                rms = mx.sqrt(mx.mean(mx.square(h[0, 1:].astype(mx.float32)), axis=-1))
                mx.eval(rms)
                got[l].extend(np.array(rms).tolist())
        mx.eval(h)
    # torch.median selects the lower middle value for even counts.
    return [sorted(got[k])[(len(got[k]) - 1) // 2] for k in blocks]


def export_checkpoint(pair, directory, args):
    import torch
    from peft import LoraConfig
    directory = Path(directory)
    for x in SIDES:
        dest = directory / x
        dest.mkdir(parents=True, exist_ok=True)
        adapters = {}
        rank = None
        for name, m in pair.models[x].named_modules():
            if isinstance(m, LoRA):
                rank = m.lora_A.shape[0]
                for key in ('lora_A', 'lora_B'):
                    adapters[f'base_model.model.{name}.{key}.weight'] = m[key]
        if rank is None:
            raise ValueError('Cannot export a model without LoRA')
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0., target_modules=list(TARGETS),
                         bias='none', base_model_name_or_path=HF[x], inference_mode=True)
        cfg.save_pretrained(dest)
        mx.save_safetensors(str(dest / 'adapter_model.safetensors'), adapters, metadata={'format': 'pt'})
    bridge = {f'{x}.{k}': torch.from_numpy(np.array(v).copy())
              for x in SIDES for k, v in tree_flatten(pair.bridge[x].parameters())}
    s = pair.bridge['quill']
    torch.save(dict(bridge=bridge, xblocks=pair.xblocks, heads=s.heads, head_dim=s.head_dim,
                    sizes={x: pair.bridge[x].size.tolist() for x in SIDES}, gate_norm=s.gate_norm,
                    args=args), directory / 'bridge.pt')


def load_checkpoint(pair, directory):
    import torch
    ck = torch.load(Path(directory) / 'bridge.pt', map_location='cpu', weights_only=False)
    s = pair.bridge['quill']
    assert (tuple(ck['xblocks']), ck['heads'], ck['head_dim']) == (pair.xblocks, s.heads, s.head_dim)
    for x in SIDES:
        state = [(k[len(x) + 1:], from_torch(v)) for k, v in ck['bridge'].items() if k.startswith(x + '.')]
        pair.bridge[x].load_weights(state, strict=True)
        pair.bridge[x].gate_norm = ck.get('gate_norm', False)
        weights = mx.load(str(Path(directory) / x / 'adapter_model.safetensors'))
        state = [(k.removeprefix('base_model.model.').removesuffix('.weight'), v) for k, v in weights.items()]
        pair.models[x].load_weights(state, strict=False)
    mx.eval(pair.parameters())


def main():
    from mlx_train import main as train
    train()


if __name__ == '__main__':
    main()
