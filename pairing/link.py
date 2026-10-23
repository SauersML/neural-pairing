"""Two language models from different families joined by a corpus callosum: gated cross-attention bridges at several
middle depths, in both directions, in one joint forward pass.

The two models run block by block in lockstep. After each exchange block (by default after blocks 6, 9, ..., 27 of 36;
none in the first six blocks or the last eight), each model reads the other's hidden states at that same depth:

    h_x += tanh(alpha) * size * O(attention(Q(norm(h_x)), K(norm(h_y)), s * V(norm(h_y))))

Q, O and alpha belong to the reader x. K, V and the visibility gate s = sigmoid(w . h_y + b) belong to the sender y:
each of its positions shows itself to the partner as much as its gate lets it, and the K/V space is a code the two
models build together. Both reads at an exchange use the states of that block before the exchange, so the exchange is
simultaneous and symmetric. alpha starts at 0 (as in Flamingo), so at step 0 the pair is exactly the two original
models; O starts random, so alpha gets a gradient. size is the reader's residual RMS after that block, measured once,
so tanh(alpha) is the write's size relative to the residual stream. A learned null key with a zero value lets a reader
attend to nothing. Each reader head also has a learned bias on its attention logits by how long ago the partner
token was read: buckets of tick(i) - tick(j) = 0, 1, 2-3, 4-7, ..., 64 and more, and one bucket for the partner's
prompt. It starts at 0. The bridges compute in float32.

The visibility gate is one per exchange block and position: each gate reads the very state it shows, and a gate shared
by all blocks would have to be computed at one depth before the deeper states exist. The gate multiplies the values
(not the attention logits), so a closed gate hides the position: masking logits would renormalize the attention, and
hiding every position would then hide nothing.

Time: every token has the tick at which its model reads it (session.py). Position i of x reads partner position j iff
tick(j) <= tick(i): the partner's whole prompt (tick -1) and the partner's tokens of the same tick are visible. Each
exchange uses the states of one block, so the dependency graph is acyclic in depth: the joint teacher-forced pass
(Pair.joint) is exact, and generation (session.generate) steps both models together, tick by tick and block by block
(Pair.step), with KV caches and a cache of each bridge's shown keys and values.

One-way link (Pair(frozen=y)): the target y is the original model, bit for bit: no LoRA, no read into it, nothing added
to its residual stream. Only the reader x = partner(y) reads, through its query, output, alpha, null key and recency
bias; y's halves that show its states (key/value norm and projections, visibility gate) still train, as the reader's
channel. y's read parts and x's show parts are unused, untrained and left out of the checkpoint, which records frozen=y.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

DEVICE = os.environ.get('NL_DEVICE') or (f'cuda:{os.environ.get("LOCAL_RANK", 0)}' if torch.cuda.is_available() else 'cpu')
XBLOCKS = (6, 9, 12, 15, 18, 21, 24, 27)
FAR = 10 ** 9                # the tick of padding, and the cut of an episode whose link never closes
AGES = 8                     # recency buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64 and more; then the prompt's

MODELS = {
    'quill': dict(hf=os.environ.get('NL_QUILL', 'Qwen/Qwen3-4B'), name='Quill', family='Qwen',
                  about='a Qwen model made by Alibaba'),
    'ember': dict(hf=os.environ.get('NL_EMBER', 'arcee-ai/AFM-4.5B'), name='Ember', family='AFM',
                  about='an AFM model made by Arcee AI'),
}
SIDES = ('quill', 'ember')
READ = ('q_norm', 'q', 'o', 'alpha', 'null', 'age')       # a Side's parts that read the partner
SHOW = ('kv_norm', 'k', 'v', 'gate')                       # a Side's parts that show its own states


def load(x, dtype=torch.bfloat16, stream=None):
    """Tokenizer and model of side x. NL_TINY=1: a 6-block random model with the same architecture (for smoke tests).
    stream (default: NL_STREAM=1): the blocks' weights stay in host memory, memory-mapped from the checkpoint's files (
    reclaimable file pages, not the process's own memory), and each block copies them to DEVICE only while it runs
    (stream_blocks); the embeddings, final norm and LM head live on DEVICE."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    stream = os.environ.get('NL_STREAM') == '1' if stream is None else stream
    hf = MODELS[x]['hf']
    tok = AutoTokenizer.from_pretrained(hf)
    if os.environ.get('NL_TINY'):
        cfg = AutoConfig.from_pretrained(hf)
        cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 6, 128, 256
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
        if getattr(cfg, 'layer_types', None):
            cfg.layer_types = cfg.layer_types[:6]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    elif stream:
        model = _mapped(hf, dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype)
    if stream:
        dec = inner(model)[0]
        blocks, dec.layers = dec.layers, nn.ModuleList()
        model.to(DEVICE)
        dec.layers = blocks
        return tok, stream_blocks(model)
    return tok, model.to(DEVICE)


_MAPS = []                   # the open memory maps (their tensors read them)


def _safetensors(path):
    """{name: CPU tensor} of a safetensors file, memory-mapped copy-on-write: clean file pages until written."""
    import json
    import mmap
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(n))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    _MAPS.append(mm)
    types = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
    out = {}
    for name, info in header.items():
        if name != '__metadata__':
            a, b = info['data_offsets']
            t = types[info['dtype']]
            out[name] = torch.frombuffer(mm, dtype=t, count=(b - a) // t.itemsize, offset=8 + n + a).view(info['shape'])
    return out


def _mapped(hf, dtype):
    """Model hf on the host with its weights memory-mapped from the cached checkpoint (_safetensors) and its rotary
    embeddings built again (their buffers are not in the checkpoint)."""
    import json
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import cached_file
    try:
        files = sorted(set(json.load(open(cached_file(hf, 'model.safetensors.index.json')))['weight_map'].values()))
    except OSError:
        files = ['model.safetensors']
    cfg = AutoConfig.from_pretrained(hf)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    state = {}
    for name in files:
        state.update(_safetensors(cached_file(hf, name)))
    state = {k: v if v.dtype == dtype else v.to(dtype) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    tied = ['lm_head.weight'] if cfg.tie_word_embeddings else []
    assert not unexpected and missing == tied, (missing, unexpected)
    if tied:
        model.lm_head.weight = inner(model)[0].embed_tokens.weight
    for name, mod in list(model.named_modules()):
        if any(b.is_meta for b in mod.buffers(recurse=False)):           # rotary inverse frequencies
            parent, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent), attr, type(mod)(config=cfg))
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    return model


def stream_blocks(model):
    """Run each block of model on DEVICE copies of its weights, made each time it runs (also when gradient
    checkpointing reruns it for the backward pass), so that only running blocks hold weights on DEVICE: the copies
    live as long as the autograd graph needs them, and the block's own parameters stay where they are. The weights
    streamed are the block's parameters now (frozen); parameters added later (LoRA adapters) move to DEVICE at first
    use. The copies are passed with torch.nn.utils.stateless (not by setting .data, which would change what autograd
    saved)."""
    from torch.nn.utils.stateless import _reparametrize_module
    on = torch.device(DEVICE).type

    def streamed(block, forward, host):
        def run(*args, **kwargs):
            copies = {}
            for name, p in block.named_parameters():
                if id(p) in host:
                    assert not (p.requires_grad and torch.is_grad_enabled()), f'streamed weight {name} must be frozen'
                    copies[name] = p.detach().to(DEVICE, copy=True)
                elif p.device.type != on:
                    p.data = p.data.to(DEVICE)
            with _reparametrize_module(block, copies):
                return forward(*args, **kwargs)
        return run
    for block in find_layers(model):
        block.forward = streamed(block, block.forward, {id(p) for p in block.parameters()})
    return model


def partner(x):
    return SIDES[1 - SIDES.index(x)]


def find_layers(model):
    for name, mod in model.named_modules():
        if name.endswith('layers') and isinstance(mod, nn.ModuleList):
            return mod
    raise ValueError('no decoder blocks found')


def inner(model):
    """The decoder stack (no LM head) and the LM head, under a PEFT wrapper or not."""
    m = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return m.model, m.lm_head


@torch.no_grad()
def residual_rms(model, blocks, seqs):
    """Median per-token RMS of model's residual stream after each of blocks, over the token sequences seqs (the first
    token of each, whose activations are outliers, left out)."""
    layers = find_layers(model)
    got = {l: [] for l in blocks}

    def hook(l):
        return lambda mod, i, o: got[l].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().pow(2).mean(-1).sqrt())
    hooks = [layers[l].register_forward_hook(hook(l)) for l in blocks]
    try:
        for s in seqs:
            model(input_ids=torch.tensor([s], device=DEVICE))
    finally:
        for h in hooks:
            h.remove()
    return [float(torch.cat(got[l]).median()) for l in blocks]


class Run:
    """One model's pass over one chunk of input, block by block: the decoder's own embedding, masks, rotary
    embeddings, blocks and final norm, called as the decoder's forward calls them."""

    def __init__(self, model, ids, att, pos, cache=None):
        self.dec, self.head = inner(model)
        cfg = self.dec.config
        self.h = self.dec.embed_tokens(ids)
        self.pos, self.cache = pos, cache
        self.types = getattr(cfg, 'layer_types', None) or ['full_attention'] * cfg.num_hidden_layers
        kw = dict(config=cfg, inputs_embeds=self.h, attention_mask=att, past_key_values=cache, position_ids=pos)
        self.masks = {'full_attention': create_causal_mask(**kw)}
        if 'sliding_attention' in self.types:
            self.masks['sliding_attention'] = create_sliding_window_causal_mask(**kw)
        self.pe = self.dec.rotary_emb(self.h, pos)

    def block(self, l):
        out = self.dec.layers[l](self.h, attention_mask=self.masks[self.types[l]], position_embeddings=self.pe,
                                 position_ids=self.pos, past_key_values=self.cache, use_cache=self.cache is not None)
        self.h = out[0] if isinstance(out, tuple) else out

    def last(self):
        return self.dec.norm(self.h)


def _linear(i, o):
    lin = nn.Linear(i, o, bias=False)
    nn.init.normal_(lin.weight, std=i ** -0.5)       # unit-RMS outputs from unit-RMS inputs
    return lin


def age_buckets(tq, tk):
    """[B, Lq, Lk] recency bucket of each partner position (ticks tk [B, Lk]) for each reading position (ticks
    tq [B, Lq]): log-spaced buckets of tick(i) - tick(j), and AGES for the partner's prompt."""
    d = (tq[:, :, None] - tk[:, None, :]).clamp(min=0)
    b = torch.where(d == 0, 0, torch.log2(d.clamp(min=1).float()).floor().long() + 1).clamp(max=AGES - 1)
    return torch.where(tk[:, None, :] < 0, AGES, b)


class Side(nn.Module):
    """One model's half of the bridges, at every exchange block: what it reads with (query norm and projection, output
    projection, alpha, null key, recency bias) and what it shows (key/value norm and projections, visibility gate)."""

    def __init__(self, d, n, heads, head_dim, sizes, gate_norm=False):
        super().__init__()
        w = heads * head_dim
        self.heads, self.head_dim = heads, head_dim
        self.gate_norm = gate_norm
        self.q_norm = nn.ModuleList(nn.RMSNorm(d, eps=1e-6) for _ in range(n))
        self.q = nn.ModuleList(_linear(d, w) for _ in range(n))
        self.o = nn.ModuleList(_linear(w, d) for _ in range(n))
        self.alpha = nn.Parameter(torch.zeros(n))
        self.null = nn.Parameter(torch.zeros(n, heads, head_dim))
        self.age = nn.Parameter(torch.zeros(n, heads, AGES + 1))
        self.kv_norm = nn.ModuleList(nn.RMSNorm(d, eps=1e-6) for _ in range(n))
        self.k = nn.ModuleList(_linear(d, w) for _ in range(n))
        self.v = nn.ModuleList(_linear(d, w) for _ in range(n))
        self.gate = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))
        for g in self.gate:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, 2.0)           # every gate starts 88% open
        self.register_buffer('size', torch.tensor([float(s) for s in sizes]))

    def split(self, t):
        return t.view(*t.shape[:-1], self.heads, self.head_dim).transpose(-3, -2)       # [B, H, L, hd]

    def show(self, k, h, src=None, rec=None):
        """Keys, gated values [B, H, L, hd] and gates [B, L] of the states h [B, L, d] at exchange k. src: states to
        show in place of h (the 'random' ablation), with the gates still read from h."""
        z = self.kv_norm[k]((h if src is None else src).float())
        gh = h.float()
        if self.gate_norm:
            # A separate parameter-free norm: random ablations still gate the real h, not src.
            gh = F.rms_norm(gh, (gh.shape[-1],), eps=1e-6)
        s = torch.sigmoid(self.gate[k](gh).squeeze(-1))
        v = self.v[k](z) * s[..., None]
        if rec is not None:
            rec.update(gate=s.detach().cpu(), shown=v.detach().cpu())
        return self.split(self.k[k](z)), self.split(v), s

    def read(self, k, h, keys, vals, allow, ages, qmask, rec=None):
        """The write into the states h [B, Lq, d] from the partner's shown keys and values [B, H, Lk, hd]; allow
        [B, Lq, Lk] says which partner positions each position may read, ages [B, Lq, Lk] gives their recency buckets
        (age_buckets), and qmask [B, Lq] says which positions read."""
        B, Lq = h.shape[:2]
        q = self.split(self.q[k](self.q_norm[k](h.float())))
        nk = self.null[k][None, :, None, :].expand(B, -1, 1, -1)
        keys = torch.cat([nk, keys], 2)
        vals = torch.cat([torch.zeros_like(nk), vals], 2)
        bias = self.age[k][:, ages].transpose(0, 1)                                     # [B, H, Lq, Lk]
        bias = torch.cat([torch.zeros(B, self.heads, Lq, 1, device=h.device), bias], -1)
        allow = torch.cat([torch.ones(B, Lq, 1, dtype=torch.bool, device=h.device), allow], -1)[:, None]
        bias = bias.masked_fill(~allow, float('-inf'))
        if rec is None:
            a = F.scaled_dot_product_attention(q, keys, vals, attn_mask=bias)
        else:
            p = (q @ keys.transpose(-1, -2) / math.sqrt(self.head_dim) + bias).softmax(-1)
            a = p @ vals
        w = torch.tanh(self.alpha[k]) * self.size[k] * self.o[k](a.transpose(1, 2).reshape(B, Lq, -1)) * qmask[..., None]
        if rec is not None:
            rec.update(attn=p.detach().cpu(), write=w.detach().cpu(), alpha=float(torch.tanh(self.alpha[k])))
        return w


def randomize(h, gen):
    """Gaussian vectors with the norm of each of h's positions (the 'random' ablation of what a sender shows)."""
    r = torch.randn(h.shape, generator=gen, device='cpu').to(h.device, h.dtype)
    return F.normalize(r, dim=-1) * h.norm(dim=-1, keepdim=True)


class Pair(nn.Module):
    """Both models and the bridges between them.

    Modes (ablations): None (the bridges as trained); 'zero' (no bridge: exactly the two models alone); 'shuffle' (
    the first side reads the previous episode's partner, and the other side uses the inverse permutation,
    so every duplex mismatched pair is isolated; dialogue still shares the original partner's text;
    batch one game at a time); 'random' (each sender
    shows Gaussian states with the norms of its own, through its own gates). write_mode='random' (generation): each
    write is replaced by a random direction of the same norm.

    record: a dict, or None. When a dict, every exchange appends to record[(side, block)] a dict with the side's gates
    and shown values (as a sender) and its attention weights (key 0 is the null key) and writes (as a reader).
    track: a dict, or None. When a dict, every exchange stores there the mean square of each reader's write per
    reading position, relative to size squared, as a 0-d tensor under (side, exchange index)."""

    def __init__(self, models, toks, xblocks=XBLOCKS, sizes=None, heads=8, head_dim=128, gate_norm=False, frozen=None):
        super().__init__()
        self.models, self.toks, self.xblocks = models, toks, tuple(xblocks)
        self.depth = {x: len(find_layers(models[x])) for x in SIDES}
        assert max(self.xblocks) < min(self.depth.values()) - 1, (self.xblocks, self.depth)
        sizes = sizes or {x: [1.0] * len(self.xblocks) for x in SIDES}
        self.bridge = nn.ModuleDict({x: Side(models[x].config.hidden_size, len(self.xblocks), heads, head_dim, sizes[x], gate_norm)
                                     for x in SIDES})
        self.record = None
        self.track = None
        self.frozen = None
        if frozen is not None:
            self.freeze(frozen)

    def unused(self, frozen=None):
        """The bridge parameter names (state dict keys) a one-way link never uses: the target's read parts and the
        reader's show parts."""
        y = self.frozen if frozen is None else frozen
        if y is None:
            return set()
        parts = {y: READ, partner(y): SHOW}
        return {k for k in self.bridge.state_dict() if k.split('.')[1] in parts[k.split('.')[0]]}

    def freeze(self, y, plain=True):
        """Make the link one-way onto the target y (see the module docstring). y's model must carry no LoRA (plain=False:
        the caller disables y's adapters whenever the link runs, as bench_eval's pair arm does) and no trainable
        parameter. Idempotent; call it again after anything that sets the bridges (--read_start, align)."""
        if y not in SIDES:
            raise ValueError(f'frozen side must be one of {SIDES}, not {y!r}')
        if self.frozen not in (None, y):
            raise ValueError(f'the link is already one-way onto {self.frozen}')
        m = self.models[y]
        if plain and (hasattr(m, 'peft_config') or any('lora_' in n for n, _ in m.named_parameters())):
            raise ValueError(f'the target {y} must be the plain model, without LoRA')
        if any(p.requires_grad for p in m.parameters()):
            raise ValueError(f'the target {y} must have no trainable parameter')
        self.frozen = y
        with torch.no_grad():
            self.bridge[y].alpha.zero_()                       # even an unused read of the target writes exactly 0
        for x, parts in ((y, READ), (partner(y), SHOW)):
            for n, p in self.bridge[x].named_parameters():
                if n.split('.')[0] in parts:
                    p.requires_grad_(False)
        return self

    @classmethod
    def from_checkpoint(cls, models, toks, ck, frozen=None, plain=True):
        """The pair saved in ck. A one-way checkpoint (ck['frozen']) loads one-way; frozen= makes a two-way checkpoint
        one-way onto that side (its saved read parts and the reader's show parts are then unused)."""
        f = ck.get('frozen')
        if f is not None and frozen not in (None, f):
            raise ValueError(f'checkpoint is one-way onto {f}, not {frozen}')
        f = f or frozen
        pair = cls(models, toks, ck['xblocks'], ck['sizes'], ck['heads'], ck['head_dim'], ck.get('gate_norm', False)).to(DEVICE)
        missing, unexpected = pair.bridge.load_state_dict(ck['bridge'], strict=False)
        # a checkpoint from before the recency bias reads as a zero bias, which is exactly that checkpoint's bridges;
        # a one-way checkpoint leaves out the parts the link never uses
        dropped = pair.unused(ck.get('frozen')) if ck.get('frozen') else set()
        assert not unexpected and all(m.endswith('.age') or m in dropped for m in missing), (missing, unexpected)
        if f is not None:
            pair.freeze(f, plain)
        return pair

    def checkpoint(self, **extra):
        s = self.bridge[SIDES[0]]
        drop = self.unused()
        return dict(bridge={k: v for k, v in self.bridge.state_dict().items() if k not in drop}, xblocks=self.xblocks,
                    heads=s.heads, head_dim=s.head_dim, sizes={x: self.bridge[x].size.tolist() for x in SIDES},
                    gate_norm=s.gate_norm, frozen=self.frozen, **extra)

    def _exchange(self, k, h, qmask, mode, gen, write_mode=None, allow=None, ages=None, cache=None, tick=None):
        """The exchange after block xblocks[k] between the sides in h {x: [B, L, d]}. Without a cache, allow and ages
        {x: [B, Lx, Ly]} say what each position reads and how long ago it was read. With a cache (generation, at tick
        tick), each sender's shown keys and values are first appended to cache[y]['blocks'][k], and each reader reads
        all of them that are valid. Returns the new states and the gates. One-way (self.frozen): only the target shows
        and only the reader reads; the reader's gates are returned as zeros (it shows nothing)."""
        rec = {}
        if self.record is not None:
            for x in h:
                rec[x] = {}
                self.record.setdefault((x, self.xblocks[k]), []).append(rec[x])
        shows = [y for y in h if partner(y) != self.frozen]
        shown = {y: self.bridge[y].show(k, h[y], randomize(h[y], gen) if mode == 'random' else None, rec.get(y))
                 for y in shows}
        if cache is not None:
            for y in shows:
                c = cache[y]['blocks'].get(k)
                cache[y]['blocks'][k] = shown[y][:2] if c is None else (torch.cat([c[0], shown[y][0]], 2),
                                                                         torch.cat([c[1], shown[y][1]], 2))
        out = dict(h)
        for x in h:
            if x == self.frozen:
                continue
            y = partner(x)
            shift = 1 if x == SIDES[0] else -1
            if cache is None:
                (keys, vals, _), a, ag = shown[y], allow[x], ages[x]
            elif k in cache[y]['blocks']:
                keys, vals = cache[y]['blocks'][k]
                a, tk = cache[y]['valid'], cache[y]['tick']
                if mode == 'shuffle':
                    a, tk = a.roll(shift, 0), tk.roll(shift, 0)
                B, Lq = h[x].shape[:2]
                a = a[:, None, :].expand(-1, Lq, -1)
                ag = age_buckets(torch.full((B, Lq), tick, device=a.device), tk)
            else:
                continue
            if mode == 'shuffle':
                keys, vals = keys.roll(shift, 0), vals.roll(shift, 0)
            w = self.bridge[x].read(k, h[x], keys, vals, a, ag, qmask[x], rec.get(x))
            if write_mode == 'random':
                r = torch.randn(w.shape, generator=gen, device='cpu').to(w.device)
                w = F.normalize(r, dim=-1) * w.norm(dim=-1, keepdim=True)
            if self.track is not None:
                ms = (w.detach().pow(2).mean(-1) * qmask[x]).sum() / qmask[x].sum().clamp_min(1)
                self.track[(x, k)] = ms / self.bridge[x].size[k] ** 2
            out[x] = h[x] + w.to(h[x].dtype)
        return out, {y: shown[y][2] if y in shown else h[y].new_zeros(h[y].shape[:2], dtype=torch.float32) for y in h}

    def joint(self, b, mode=None, gen=None, push=None):
        """Teacher-forced joint pass of both models over the batch b (session.collate). Returns the last hidden states
        {x: [B, L, d]} (after the final norm) and the gates {x: [B, L, n exchanges]}. push {x: {block: [B, 1, d]}}: added
        to x's states at every position right after that block, before its exchange (steer_read.py)."""
        if mode == 'shuffle' and b[SIDES[0]]['ids'].shape[0] < 2:
            raise ValueError('shuffle needs at least two episodes')
        runs = {x: Run(self.models[x], b[x]['ids'], b[x]['att'], b[x]['pos']) for x in SIDES}
        allow, ages, qmask = {}, {}, {}
        for x in SIDES:
            ty, vy = b[partner(x)]['tick'], b[partner(x)]['att'].bool()
            if mode == 'shuffle':
                shift = 1 if x == SIDES[0] else -1
                ty, vy = ty.roll(shift, 0), vy.roll(shift, 0)
            allow[x] = (ty[:, None, :] <= b[x]['tick'][:, :, None]) & vy[:, None, :]
            ages[x] = age_buckets(b[x]['tick'], ty)
            qmask[x] = (b[x]['att'].bool() & (b[x]['tick'] < b[x]['cut'][:, None])).float()
        gates = {x: [] for x in SIDES}
        for l in range(max(self.depth.values())):
            for x in SIDES:
                if l < self.depth[x]:
                    runs[x].block(l)
                    if push and l in push.get(x, {}):
                        runs[x].h = runs[x].h + push[x][l].to(runs[x].h.dtype)
            if l in self.xblocks and mode != 'zero':
                h, g = self._exchange(self.xblocks.index(l), {x: runs[x].h for x in SIDES}, qmask, mode, gen,
                                      allow=allow, ages=ages)
                for x in SIDES:
                    runs[x].h = h[x]
                    gates[x].append(g[x])
        n = len(self.xblocks)
        return ({x: runs[x].last() for x in SIDES},
                {x: torch.stack(gates[x], -1) if gates[x] else torch.zeros(*b[x]['ids'].shape, n, device=DEVICE)
                 for x in SIDES})

    def new_cache(self):
        """The generation state: each model's KV cache, and per sender the validity and tick of every position it has
        read and its shown keys and values at each exchange."""
        from transformers import DynamicCache
        return {'kv': {x: DynamicCache(config=inner(self.models[x])[0].config) for x in SIDES},
                'shown': {x: {'valid': None, 'tick': None, 'blocks': {}} for x in SIDES}}

    def step(self, chunk, cache, qmask, tick, mode=None, gen=None, write_mode=None):
        """One tick of generation. chunk {x: (ids [B, K], att [B, past + K], pos [B, K])} for the models that read
        tokens at this tick; cache from new_cache(); qmask {x: [B, K]}: the positions that read; tick: this tick (-1:
        the prompts). Returns the last hidden states {x: [B, K, d]} (after the final norm) and the gates
        {x: [B, K, n exchanges]}."""
        runs = {x: Run(self.models[x], ids, att, pos, cache['kv'][x]) for x, (ids, att, pos) in chunk.items()}
        for x, (ids, att, _) in chunk.items():
            v = att[:, -ids.shape[1]:].bool()
            t = torch.full(v.shape, tick, dtype=torch.long, device=v.device)
            sh = cache['shown'][x]
            sh['valid'] = v if sh['valid'] is None else torch.cat([sh['valid'], v], 1)
            sh['tick'] = t if sh['tick'] is None else torch.cat([sh['tick'], t], 1)
        gates = {x: [] for x in runs}
        for l in range(max(self.depth.values())):
            for x in runs:
                if l < self.depth[x]:
                    runs[x].block(l)
            if l in self.xblocks and mode != 'zero':
                h, g = self._exchange(self.xblocks.index(l), {x: runs[x].h for x in runs}, qmask, mode, gen,
                                      write_mode, cache=cache['shown'], tick=tick)
                for x in runs:
                    runs[x].h = h[x]
                    gates[x].append(g[x])
        n = len(self.xblocks)
        return ({x: runs[x].last() for x in runs},
                {x: torch.stack(gates[x], -1) if gates[x] else torch.zeros(*chunk[x][0].shape, n, device=DEVICE)
                 for x in runs})

    def token_logits(self, x, last, idx):
        """Logits of model x at the flat positions idx (into [B * L])."""
        _, head = inner(self.models[x])
        return head(last.reshape(-1, last.shape[-1])[idx]).float()
