"""The MLX version of link.Pair, for evaluating a saved step on Apple silicon with 4-bit bases (mlx_eval.py).

load_base builds Qwen3 or AFM in MLX with 4-bit weights, quantizing the checkpoint one tensor at a time (no Hugging
Face model is built) and caching the result (quant_cache). add_lora adds LoRA in PEFT's layout; Pair contains both
models and the bridges, with link.Side's parameter names, shapes and float32 math; load_checkpoint and
export_checkpoint read and write a step in train.py's format (PEFT adapters and bridge.pt).

Usage: import mlx_pair as M; models[x], config = M.load_base(M.HF[x]); M.add_lora(models[x], rank); M.Pair(models, ...)
"""
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ['NL_DEVICE'] = 'cpu'
# MLX on Apple M5 runs float32 matrix products in TF32 by default; the bridges compute in float32, as link.py's do.
os.environ['MLX_ENABLE_TF32'] = '0'

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
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
    """An MLX model, random weights, from a Hugging Face config dict: Qwen3, or Arcee (AFM) with its relu2 MLP."""
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
    """A torch tensor as an MLX array, bf16 kept bit for bit."""
    import torch
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.uint16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())


SAFE_DTYPES = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}


def _uncached_tensors(file):
    """(name, mx array) for every tensor of a safetensors file, read with the page cache off (F_NOCACHE), so that
    reading both bf16 models (17 GB) does not fill macOS's file cache and leave too little free memory."""
    import fcntl
    fd = os.open(file, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, getattr(fcntl, 'F_NOCACHE', 48), 1)
        n = int.from_bytes(os.pread(fd, 8, 0), 'little')
        header = json.loads(os.pread(fd, n, 8))
        header.pop('__metadata__', None)
        for key, info in sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            a, b = info['data_offsets']
            buf, at = bytearray(b - a), 0
            view = memoryview(buf)
            while at < len(buf):
                got = os.preadv(fd, [view[at:at + (1 << 30)]], 8 + n + a + at)
                if got <= 0:
                    raise OSError(f'short read in {file} at {key}')
                at += got
            arr = mx.array(np.frombuffer(buf, SAFE_DTYPES[info['dtype']]).reshape(info['shape']))
            yield key, arr.view(mx.bfloat16) if info['dtype'] == 'BF16' else arr
            del buf, view
    finally:
        os.close(fd)


def quant_cache(hf, bits):
    """The cached quantized weights of model hf: NL_MLX_CACHE (default ~/.cache/neural-pairing/mlx4)/NAME-BITSbit."""
    root = Path(os.environ.get('NL_MLX_CACHE', os.path.expanduser('~/.cache/neural-pairing/mlx4')))
    return root / f"{str(hf).strip('/').replace('/', '--')}-{bits}bit.safetensors"


def load_base(hf, bits=4, cache=True):
    """(model, config) of hf (a local directory or a cached Hugging Face model) with frozen bits-bit weights (bits 0:
    full precision), read one tensor at a time. With cache, the quantized model is saved after the first load
    (quant_cache) and later loads read only that file."""
    from huggingface_hub import hf_hub_download
    path = Path(hf) if Path(hf).is_dir() else Path(hf_hub_download(hf, 'config.json', local_files_only=True)).parent
    config = json.loads((path / 'config.json').read_text())
    model = make_model(config)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    saved = quant_cache(hf, bits) if cache and bits and not Path(hf).is_dir() else None
    if saved is not None and saved.exists():
        model.load_weights(str(saved))
        mx.eval(model.parameters())
        model.freeze()
        return model, config
    modules = dict(model.named_modules())
    index = path / 'model.safetensors.index.json'
    files = (sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists()
             else ['model.safetensors'])
    expected = set(dict(tree_flatten(model.parameters())))
    loaded = set()
    for file in files:
        for key, w in _uncached_tensors(str(path / file)):
            if key == 'lm_head.weight' and config.get('tie_word_embeddings'):
                continue
            if key not in expected:
                raise ValueError(f'Unexpected base tensor {key}')
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
            del w
        gc.collect()
        mx.clear_cache()
    if expected != loaded:
        raise ValueError(f'Missing weights: {expected - loaded}')
    if saved is not None:
        saved.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(saved.with_suffix('.part.safetensors')), dict(tree_flatten(model.parameters())))
        os.replace(saved.with_suffix('.part.safetensors'), saved)
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
        base.freeze()

    def __call__(self, h):
        y = self.base(h)
        z = (h.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
        return y + (2.0 * z).astype(y.dtype)


def add_lora(model, rank):
    """Freezes model and wraps its attention and MLP projections in LoRA of the given rank (PEFT's targets)."""
    model.freeze()
    replace = [(n, LoRA(m, rank)) for n, m in model.named_modules()
               if n.split('.')[-1] in TARGETS and isinstance(m, (nn.Linear, nn.QuantizedLinear))]
    model.update_modules(tree_unflatten(replace))
    return model


def linear(inp, out):
    m = nn.Linear(inp, out, bias=False)
    m.weight = mx.random.normal((out, inp)) * inp ** -0.5
    return m


def age_buckets(tq, tk):
    """link.age_buckets in MLX."""
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


class Pair(nn.Module):
    """Both MLX models and the bridges between them (mlx_eval.Pair runs them for session.py)."""

    def __init__(self, models, xblocks=XBLOCKS, sizes=None, heads=8, head_dim=128, gate_norm=False):
        super().__init__()
        self.models = models
        self.xblocks = tuple(xblocks)
        assert self.xblocks and max(self.xblocks) < min(len(m.layers) for m in models.values()) - 1
        self.bridge = {x: Side(models[x].args.hidden_size, len(xblocks), heads, head_dim,
                               sizes[x] if sizes else [1.] * len(xblocks), gate_norm) for x in SIDES}

    def token_logits(self, x, h, idx=None):
        """Model x's float32 logits from final hidden states h, at the flat positions idx when given."""
        if idx is not None:
            h = h.reshape(-1, h.shape[-1])[idx]
        m = self.models[x]
        return (m.model.embed_tokens.as_linear(h) if m.args.tie_word_embeddings else m.lm_head(h)).astype(mx.float32)


def convert_batch(b):
    """A session.collate batch as MLX arrays (int64 as int32), without the owners lists."""
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


def export_checkpoint(pair, directory, args):
    """Saves pair as a step in train.py's format: PEFT adapters per model and bridge.pt (args stored with it)."""
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


def load_checkpoint(pair, directory, frozen=None):
    """Loads the step in directory into pair; returns its one-way target (the checkpoint's, or frozen= for a two-way
    step read one-way), whose adapter is not loaded and whose alpha is zeroed (link.py), or None."""
    import torch
    ck = torch.load(Path(directory) / 'bridge.pt', map_location='cpu', weights_only=False)
    s = pair.bridge['quill']
    assert (tuple(ck['xblocks']), ck['heads'], ck['head_dim']) == (pair.xblocks, s.heads, s.head_dim)
    if ck.get('frozen') and frozen not in (None, ck['frozen']):
        raise ValueError(f"checkpoint is one-way onto {ck['frozen']}, not {frozen}")
    frozen = ck.get('frozen') or frozen
    for x in SIDES:                      # a one-way checkpoint leaves out the target's read and the reader's show parts
        state = [(k[len(x) + 1:], from_torch(v)) for k, v in ck['bridge'].items() if k.startswith(x + '.')]
        pair.bridge[x].load_weights(state, strict=not ck.get('frozen'))
        pair.bridge[x].gate_norm = ck.get('gate_norm', False)
        if x == frozen:
            pair.bridge[x].alpha = mx.zeros_like(pair.bridge[x].alpha)
            continue
        weights = mx.load(str(Path(directory) / x / 'adapter_model.safetensors'))
        state = [(k.removeprefix('base_model.model.').removesuffix('.weight'), v) for k, v in weights.items()]
        pair.models[x].load_weights(state, strict=False)
    mx.eval(pair.parameters())
    return frozen
