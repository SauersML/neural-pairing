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


def load(x, dtype=torch.bfloat16):
    """Tokenizer and model of side x. NL_TINY=1: a 6-block random model with the same architecture (for smoke tests)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
    else:
        model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype)
    return tok, model.to(DEVICE)


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

    def __init__(self, d, n, heads, head_dim, sizes):
        super().__init__()
        w = heads * head_dim
        self.heads, self.head_dim = heads, head_dim
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
        s = torch.sigmoid(self.gate[k](h.float()).squeeze(-1))
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

    Modes (ablations): None (the bridges as trained); 'zero' (no bridge: exactly the two models alone); 'shuffle' (each
    episode reads the partner of the previous episode in the batch, so batch one game at a time); 'random' (each sender
    shows Gaussian states with the norms of its own, through its own gates). write_mode='random' (generation): each
    write is replaced by a random direction of the same norm.

    record: a dict, or None. When a dict, every exchange appends to record[(side, block)] a dict with the side's gates
    and shown values (as a sender) and its attention weights (key 0 is the null key) and writes (as a reader).
    track: a dict, or None. When a dict, every exchange stores there the mean square of each reader's write per
    reading position, relative to size squared, as a 0-d tensor under (side, exchange index)."""

    def __init__(self, models, toks, xblocks=XBLOCKS, sizes=None, heads=8, head_dim=128):
        super().__init__()
        self.models, self.toks, self.xblocks = models, toks, tuple(xblocks)
        self.depth = {x: len(find_layers(models[x])) for x in SIDES}
        assert max(self.xblocks) < min(self.depth.values()) - 1, (self.xblocks, self.depth)
        sizes = sizes or {x: [1.0] * len(self.xblocks) for x in SIDES}
        self.bridge = nn.ModuleDict({x: Side(models[x].config.hidden_size, len(self.xblocks), heads, head_dim, sizes[x])
                                     for x in SIDES})
        self.record = None
        self.track = None

    @classmethod
    def from_checkpoint(cls, models, toks, ck):
        pair = cls(models, toks, ck['xblocks'], ck['sizes'], ck['heads'], ck['head_dim']).to(DEVICE)
        missing, unexpected = pair.bridge.load_state_dict(ck['bridge'], strict=False)
        # a checkpoint from before the recency bias reads as a zero bias, which is exactly that checkpoint's bridges
        assert not unexpected and all(m.endswith('.age') for m in missing), (missing, unexpected)
        return pair

    def checkpoint(self, **extra):
        s = self.bridge[SIDES[0]]
        return dict(bridge=self.bridge.state_dict(), xblocks=self.xblocks, heads=s.heads, head_dim=s.head_dim,
                    sizes={x: self.bridge[x].size.tolist() for x in SIDES}, **extra)

    def _exchange(self, k, h, qmask, mode, gen, write_mode=None, allow=None, ages=None, cache=None, tick=None):
        """The exchange after block xblocks[k] between the sides in h {x: [B, L, d]}. Without a cache, allow and ages
        {x: [B, Lx, Ly]} say what each position reads and how long ago it was read. With a cache (generation, at tick
        tick), each sender's shown keys and values are first appended to cache[y]['blocks'][k], and each reader reads
        all of them that are valid. Returns the new states and the gates."""
        rec = {}
        if self.record is not None:
            for x in h:
                rec[x] = {}
                self.record.setdefault((x, self.xblocks[k]), []).append(rec[x])
        shown = {y: self.bridge[y].show(k, h[y], randomize(h[y], gen) if mode == 'random' else None, rec.get(y))
                 for y in h}
        if cache is not None:
            for y in h:
                c = cache[y]['blocks'].get(k)
                cache[y]['blocks'][k] = shown[y][:2] if c is None else (torch.cat([c[0], shown[y][0]], 2),
                                                                         torch.cat([c[1], shown[y][1]], 2))
        out = dict(h)
        for x in h:
            y = partner(x)
            if cache is None:
                (keys, vals, _), a, ag = shown[y], allow[x], ages[x]
            elif k in cache[y]['blocks']:
                keys, vals = cache[y]['blocks'][k]
                a, tk = cache[y]['valid'], cache[y]['tick']
                if mode == 'shuffle':
                    a, tk = a.roll(1, 0), tk.roll(1, 0)
                B, Lq = h[x].shape[:2]
                a = a[:, None, :].expand(-1, Lq, -1)
                ag = age_buckets(torch.full((B, Lq), tick, device=a.device), tk)
            else:
                continue
            if mode == 'shuffle':
                keys, vals = keys.roll(1, 0), vals.roll(1, 0)
            w = self.bridge[x].read(k, h[x], keys, vals, a, ag, qmask[x], rec.get(x))
            if write_mode == 'random':
                r = torch.randn(w.shape, generator=gen, device='cpu').to(w.device)
                w = F.normalize(r, dim=-1) * w.norm(dim=-1, keepdim=True)
            if self.track is not None:
                ms = (w.detach().pow(2).mean(-1) * qmask[x]).sum() / qmask[x].sum().clamp_min(1)
                self.track[(x, k)] = ms / self.bridge[x].size[k] ** 2
            out[x] = h[x] + w.to(h[x].dtype)
        return out, {y: shown[y][2] for y in h}

    def joint(self, b, mode=None, gen=None):
        """Teacher-forced joint pass of both models over the batch b (session.collate). Returns the last hidden states
        {x: [B, L, d]} (after the final norm) and the gates {x: [B, L, n exchanges]}."""
        runs = {x: Run(self.models[x], b[x]['ids'], b[x]['att'], b[x]['pos']) for x in SIDES}
        allow, ages, qmask = {}, {}, {}
        for x in SIDES:
            ty, vy = b[partner(x)]['tick'], b[partner(x)]['att'].bool()
            if mode == 'shuffle':
                ty, vy = ty.roll(1, 0), vy.roll(1, 0)
            allow[x] = (ty[:, None, :] <= b[x]['tick'][:, :, None]) & vy[:, None, :]
            ages[x] = age_buckets(b[x]['tick'], ty)
            qmask[x] = (b[x]['att'].bool() & (b[x]['tick'] < b[x]['cut'][:, None])).float()
        gates = {x: [] for x in SIDES}
        for l in range(max(self.depth.values())):
            for x in SIDES:
                if l < self.depth[x]:
                    runs[x].block(l)
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
