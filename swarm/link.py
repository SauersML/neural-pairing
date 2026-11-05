"""A swarm: K agents, all one language model, linked through their hidden states. This is the twins' link
(pairing/link.py) for any number of agents sharing one base model, so the whole swarm runs as one batch.

Each agent is K rows of the batch apart from nothing: row b * K + j is agent j of episode b. After each link layer,
every token of every agent attends over the states that the OTHER agents of its episode had after that layer at the
same time step or earlier (prompt tokens at time -1, answer tokens at 0, 1, ...), never over its own:
  the query is the reader's RMS-normalized state mapped to 8 heads of 64, plus a learned vector for the reader's slot;
  keys and values come from the sources' normalized states, the keys plus a learned vector for the sender's slot;
  each value is multiplied by a gate the SENDER sets for each reader: a sigmoid of its own state's general term plus
  a term for the reader's slot, so a sender can choose how loudly each of its tokens reaches each other agent (slots
  follow the order of the names in every agent's prompt; every gate starts 88% open and training charges a small cost
  per open gate, train.py --send_cost); an empty slot contributes zero; each head learns a bias
  for the time gap (prompt, 0, 1, 2-3, 4-7, 8 or more);
  the output is mapped back to the model's width, scaled by a write size tanh(alpha) times the base model's typical
  residual norm at that layer, and added to the reader's residual stream.
alpha starts at 0, so a new swarm is K copies of the base model. All agents answer at the same time (duplex): prompts
are left-padded so that every answer starts at the same column, and no agent sees another's text. Mode 'cut' skips
the link.
"""
import math

import torch
from torch import nn

LAYERS = (6, 10, 14, 18, 22)                  # Qwen3-0.6B has 28 blocks
BUCKETS = 6                                   # time gaps: prompt, 0, 1, 2-3, 4-7, 8+
PROMPT, PAD = -1, -2                          # ticks of prompt and padding positions


def blocks(model):
    """The decoder blocks of a Hugging Face causal LM, PEFT-wrapped or not."""
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and name.endswith('layers'):
            return m
    raise ValueError('no decoder layers found')


def buckets(tq, tk):
    """The time-gap bucket of each (query, key) pair: 0 for a prompt key, else 1 + floor(log2(gap + 1)) up to 5."""
    gap = (tq - tk).clamp(min=0)
    b = (torch.log2(gap.float() + 1).floor().long() + 1).clamp(max=BUCKETS - 1)
    return torch.where(tk < 0, torch.zeros_like(b), b)


class Bridge(nn.Module):
    """The link after one block: every agent reads the other agents (see the module docstring)."""

    def __init__(self, d, heads=8, head_dim=64, slots=12, rms=1.0):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        width = heads * head_dim
        self.nq, self.nk = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, width, bias=False)
        self.k = nn.Linear(d, width, bias=False)
        self.v = nn.Linear(d, width, bias=False)
        self.g = nn.Linear(d, 1 + slots)              # the sender's gate: general, plus one term per reader slot
        nn.init.zeros_(self.g.weight)
        nn.init.zeros_(self.g.bias)
        nn.init.constant_(self.g.bias[:1], 2.0)       # every gate starts 88% open
        self.shown = None
        self.o = nn.Linear(width, d, bias=False)
        self.slot_q = nn.Parameter(torch.zeros(slots, width))
        self.slot_k = nn.Parameter(torch.zeros(slots, width))
        self.null = nn.Parameter(torch.randn(heads, head_dim) * head_dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(heads, BUCKETS))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer('rms', torch.tensor(float(rms)))

    def forward(self, h, k, tick):
        """h [B*K, T, d]: the batch's states after the block; k: agents per episode; tick [B*K, T]. Returns the write
        [B*K, T, d] in h's dtype (zero at padding)."""
        BK, T, d = h.shape
        B, H, D = BK // k, self.heads, self.head_dim
        x = h.float()
        slots = torch.arange(k, device=h.device)
        q = (self.q(self.nq(x)).view(B, k, T, -1) + self.slot_q[slots][None, :, None]).view(B, k, T, H, D)
        z = self.nk(x)
        key = (self.k(z).view(B, k, T, -1) + self.slot_k[slots][None, :, None]).view(B, k * T, H, D)
        g = self.g(z).view(B, k * T, -1)
        gate = torch.sigmoid(g[..., :1] + g[..., 1:1 + k])                   # [B, K*T (source), K (reader)]
        sent = ((tick.view(B, k * T) > PAD)[..., None] & (slots.repeat_interleave(T)[:, None] != slots)).float()
        self.shown = (gate * sent).sum() / sent.sum().clamp(min=1)         # the mean gate over real (source, reader)
        val = self.v(z).view(B, k * T, H, D)
        s = torch.einsum('bjthd,bshd->bjhts', q, key) / math.sqrt(D)          # [B, K, H, T, K*T]
        tq, tk = tick.view(B, k, T), tick.view(B, k * T)
        b = buckets(tq[..., None], tk[:, None, None, :])                        # [B, K, T, K*T]
        s = s + self.bias[:, b].permute(1, 2, 0, 3, 4)
        src = slots.repeat_interleave(T)
        allow = ((tk[:, None, None, :] <= tq[..., None]) & (tk[:, None, None, :] > PAD)
                 & (src[None, None, None, :] != slots[None, :, None, None]))
        s = s.masked_fill(~allow[:, :, None], float('-inf'))
        empty = torch.einsum('bjthd,hd->bjht', q, self.null)[..., None] / math.sqrt(D)
        p = torch.softmax(torch.cat([empty, s], -1), -1)[..., 1:]
        out = torch.einsum('bjhts,bsj,bshd->bjthd', p, gate, val).reshape(BK, T, H * D)
        w = torch.tanh(self.alpha) * self.rms * self.o(out)
        return (w * (tick > PAD)[..., None]).to(h.dtype)


class Swarm(nn.Module):
    """The base model (with its adapter) and one Bridge per link layer, applied by forward hooks on those blocks."""

    def __init__(self, model, layers=LAYERS, heads=8, head_dim=64, slots=12, rms=None):
        super().__init__()
        self.model = model
        self.layers = tuple(layers)
        d = model.config.hidden_size
        self.bridges = nn.ModuleList(Bridge(d, heads, head_dim, slots, 1.0 if rms is None else rms[i])
                                     for i in range(len(self.layers)))
        self.ctx = None
        stack = blocks(model)
        for i, l in enumerate(self.layers):
            stack[l].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(module, args, out):
            c = self.ctx
            if c is None or c['mode'] == 'cut':
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h + self.bridges[i](h, c['k'], c['tick'])
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def forward(self, ids, att, tick, k, mode=None, keep=0):
        """Logits [B*K, T or keep, V] of every agent; keep > 0: only the last keep positions."""
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        self.ctx = dict(k=k, tick=tick, mode=mode)
        try:
            return self.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False,
                              logits_to_keep=keep).logits
        finally:
            self.ctx = None

    @torch.no_grad()
    def calibrate(self, ids, att):
        """Sets each bridge's write unit to the base model's median per-token residual RMS at its layer (on ids)."""
        got = {}
        stack = blocks(self.model)
        handles = [stack[l].register_forward_hook(
            lambda m, a, o, i=i: got.__setitem__(i, (o[0] if isinstance(o, tuple) else o).float()))
            for i, l in enumerate(self.layers)]
        try:
            self.model(input_ids=ids, attention_mask=att, position_ids=(att.cumsum(-1) - 1).clamp(min=0),
                       use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        real = att.bool().clone()
        real[:, 0] = False
        first = att.float().argmax(-1)                     # each row's first real token (massive activations)
        real[torch.arange(len(att)), first] = False
        for i, h in got.items():
            r = h.pow(2).mean(-1).sqrt()[real]
            self.bridges[i].rms.fill_(float(r.median()))

    def link_state(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith('bridges.')}
