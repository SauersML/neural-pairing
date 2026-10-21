"""Two language models from different families, linked by a learned channel of activation vectors.

Each model has a send head and a receive adapter. At every tick each model processes one token of its own text, its
send head reads the residual stream after block LS at that token and writes a message (a vector of K numbers with a
fixed norm) and a gate (send or not). The partner's receive adapter maps the gated message into the partner's residual
stream and adds it after block LR at the partner's next token. That position enters the partner's KV cache, so later
tokens can attend back to it. Neither model ever sees the other's text.

Training reads the coupled models on teacher-forced text with Gauss-Seidel rounds: model A runs with the messages of
the previous round, then model B runs with A's new messages, and so on, every round in one parallel forward pass per
model. After R rounds every chain of up to 2R one-tick hops between the models is exact. Generation runs the two
models tick by tick with KV caches, which is exact for any chain.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 256                      # message width
DEVICE = os.environ.get('NL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

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


class Stop(Exception):
    pass


class Tap:
    """Hooks on one model: adds `delta` to the output of block lr, records the output of block ls, and, when `stop`
    is set, ends the forward pass before block ls + 1.

    Gradient checkpointing reruns each block's forward pass, hooks included, during the backward pass. The add hook
    therefore remembers which delta it added to each input tensor (the rerun gets the same input object) and adds the
    same delta again, whatever `delta` holds by then."""

    def __init__(self, model, lr, ls):
        layers = find_layers(model)
        assert lr < ls < len(layers) - 1
        self.lr, self.ls, self.delta, self.h, self.stop, self.memo = lr, ls, None, None, False, {}
        layers[lr].register_forward_hook(self._add, with_kwargs=True)
        layers[ls].register_forward_hook(self._read, with_kwargs=True)
        layers[ls + 1].register_forward_pre_hook(self._halt, with_kwargs=True)

    @staticmethod
    def _hidden(args, kwargs):
        return args[0] if args else kwargs['hidden_states']

    def _add(self, mod, args, kwargs, out):
        key = id(self._hidden(args, kwargs))
        if torch.is_grad_enabled():
            d = self.memo.setdefault(key, self.delta)
        else:
            d = self.delta
        if d is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + d.to(out[0].dtype),) + out[1:]
        return out + d.to(out.dtype)

    def _read(self, mod, args, kwargs, out):
        self.h = out[0] if isinstance(out, tuple) else out

    def _halt(self, mod, args, kwargs):
        if self.stop:
            raise Stop

    def clear(self):
        self.delta, self.h, self.memo = None, None, {}


class Side(nn.Module):
    """One model's half of the link: send head (residual -> message, gate logit) and receive adapter (message ->
    residual write). The write is an RMS-normalized vector times a learned size, times the gate."""

    def __init__(self, d, size, hid=1024):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.send = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, K))
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)
        self.recv = nn.Sequential(nn.Linear(K, hid), nn.GELU(), nn.Linear(hid, d))
        self.log_size = nn.Parameter(torch.tensor(float(size)).log())

    def message(self, h):
        """h [..., d] -> message [..., K] (norm sqrt(K)), gate probability [...]."""
        x = self.norm(h.float())
        m = F.normalize(self.send(x), dim=-1) * K ** 0.5
        return m, torch.sigmoid(self.gate(x).squeeze(-1))

    def write(self, m):
        """Gated message [..., K] -> residual write [..., d]."""
        r = self.recv(m)
        return F.rms_norm(r, (r.shape[-1],)) * self.log_size.exp()


def gate_value(p, hard=True):
    """Straight-through hard gate: 1 where p > 0.5, with the gradient of p."""
    if not hard:
        return p
    return (p > 0.5).float() + p - p.detach()


def ablate(m, g, mode, gen=None):
    """Replaces messages m [B, T, K] and gates g [B, T] for the channel ablations."""
    if mode is None:
        return m, g
    if mode == 'zero':                        # nothing reaches the partner
        return m, torch.zeros_like(g)
    if mode == 'shuffle':                     # each episode gets the messages of another episode
        return m.roll(1, dims=0), g.roll(1, dims=0)
    if mode == 'random':                      # random directions of the same norm, same gates
        r = torch.randn(m.shape, generator=gen, device='cpu').to(m.device)
        return F.normalize(r, dim=-1) * m.norm(dim=-1, keepdim=True), g
    raise ValueError(mode)


class Pair(nn.Module):
    """Both models, their taps, and both halves of the link."""

    def __init__(self, models, toks, lr, ls, sizes):
        super().__init__()
        self.models, self.toks = models, toks
        self.taps = {x: Tap(models[x], lr[x], ls[x]) for x in SIDES}
        d = {x: models[x].config.hidden_size for x in SIDES}
        self.side = nn.ModuleDict({x: Side(d[x], sizes[x]) for x in SIDES})

    # ----- training: Gauss-Seidel rounds over teacher-forced text -----

    def _run(self, x, ids, att, delta, full):
        """One forward pass of model x. Returns the residual after block ls and, when full, the last hidden state."""
        tap = self.taps[x]
        tap.delta, tap.stop = delta, not full
        dec, _ = inner(self.models[x])
        last = None
        try:
            last = dec(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        except Stop:
            pass
        finally:
            tap.stop = False
        return tap.h, last

    def _send(self, x, h, b, hard=True, mode=None, gen=None):
        """Messages of model x at its ticks: m [B, T, K], gate g [B, T], probability p [B, T], valid [B, T]."""
        T = int(b[x]['n'].max())
        t = torch.arange(T, device=h.device)
        pos = (b[x]['start'][:, None] + t[None]).clamp(max=h.shape[1] - 1)
        valid = t[None] < b[x]['n'][:, None]
        hs = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        m, p = self.side[x].message(hs)
        p = p * valid
        g = gate_value(p) * valid
        m, g = ablate(m, g, mode, gen)
        return m, g, p, valid

    def _deliver(self, y, msg, b, L):
        """Residual writes into model y [B, L, d] from its partner's messages: tick t lands at y's position start + t + 1."""
        m, g, _, valid = msg
        w = self.side[y].write(m) * g[..., None]                    # [B, T, d]
        T = w.shape[1]
        t = torch.arange(T, device=w.device)
        pos = b[y]['start'][:, None] + t[None] + 1
        ok = valid & (pos < b[y]['start'][:, None] + b[y]['n'][:, None]) & (pos < L)
        pos = pos.clamp(max=L - 1)
        out = torch.zeros(w.shape[0], L, w.shape[-1], device=w.device, dtype=w.dtype)
        return out.scatter_add(1, pos[..., None].expand_as(w), w * ok[..., None])

    def coupled(self, b, rounds=2, mode=None, gen=None):
        """Gauss-Seidel forward of both models on the batch b. Returns last hidden states per model and the messages."""
        msgs, last = {}, {}
        order = list(SIDES)
        for r in range(rounds):
            final = r == rounds - 1
            for x in order:
                y = partner(x)
                L = b[x]['ids'].shape[1]
                delta = self._deliver(x, msgs[y], b, L) if y in msgs else None
                h, hl = self._run(x, b[x]['ids'], b[x]['att'], delta, full=final)
                if final:
                    last[x] = hl
                msgs[x] = self._send(x, h, b, mode=mode, gen=gen)
        return last, msgs

    def token_logits(self, x, last, idx):
        """Logits of model x at the flat positions idx (into [B * L])."""
        _, head = inner(self.models[x])
        h = last.reshape(-1, last.shape[-1])[idx]
        return head(h).float()

    def clear(self):
        for t in self.taps.values():
            t.clear()

    # ----- generation: both models tick by tick with KV caches -----

    @torch.no_grad()
    def lockstep(self, prompts, max_new, mode=None, gen=None, force=None, delta_mode=None, mute=None, keep=False):
        """prompts: {x: list of token-id lists}. Greedy text for both models, run in lockstep.

        At tick t each model reads one token (tick 0: the last prompt token), writes a message, and picks its next
        token. Messages written at tick t reach the partner at tick t + 1. A model stops sending after it writes its
        end-of-turn token. force: {x: list of token-id lists} teacher-forces that model's reply. delta_mode 'random'
        replaces each residual write by a random vector of the same norm. mute: {x: list of bools}, episodes in
        which model x receives nothing. keep: also return each tick's messages and logits. Returns
        {x: dict(tokens, gates[, msgs, logits])}."""
        from transformers import DynamicCache
        st = {}
        for x in SIDES:
            tok, model = self.toks[x], self.models[x]
            P = prompts[x]
            B, Lp = len(P), max(len(p) for p in P)
            pad = tok.pad_token_id
            ids = torch.tensor([[pad] * (Lp - len(p)) + p for p in P], device=DEVICE)
            att = torch.tensor([[0] * (Lp - len(p)) + [1] * len(p) for p in P], device=DEVICE)
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            cache = DynamicCache()
            self.taps[x].delta = None
            model(input_ids=ids[:, :-1], attention_mask=att[:, :-1], position_ids=pos[:, :-1], past_key_values=cache,
                  use_cache=True)
            st[x] = dict(cache=cache, att=att[:, :-1], cur=ids[:, -1], pos=pos[:, -1], out=[[] for _ in range(B)],
                         gates=[[] for _ in range(B)], done=torch.zeros(B, dtype=torch.bool, device=DEVICE),
                         end=tok.convert_tokens_to_ids('<|im_end|>'), inbox=None, msgs=[], logits=[])
        for t in range(max_new):
            sent = {}
            for x in SIDES:
                s, model = st[x], self.models[x]
                delta = None
                if s['inbox'] is not None:
                    m, g = s['inbox']
                    delta = (self.side[x].write(m) * g[:, None])
                    if delta_mode == 'random':
                        r = torch.randn(delta.shape, generator=gen).to(delta.device)
                        delta = F.normalize(r, dim=-1) * delta.norm(dim=-1, keepdim=True)
                    delta = delta[:, None]
                self.taps[x].delta = delta
                s['att'] = torch.cat([s['att'], torch.ones_like(s['att'][:, :1])], 1)
                out = model(input_ids=s['cur'][:, None], attention_mask=s['att'], position_ids=s['pos'][:, None],
                            past_key_values=s['cache'], use_cache=True)
                h = self.taps[x].h[:, -1]
                m, p = self.side[x].message(h)
                g = (p > 0.5).float() * (~s['done']).float()
                for i in range(len(g)):
                    if not s['done'][i]:
                        s['gates'][i].append(round(float(p[i]), 3))
                sent[x] = (m, g)
                if keep:
                    s['msgs'].append(m.cpu())
                    s['logits'].append(out.logits[:, -1].float().cpu())
                nxt = out.logits[:, -1].argmax(-1)
                if force is not None and x in force:
                    nxt = torch.tensor([f[t] if t < len(f) else s['end'] for f in force[x]], device=DEVICE)
                for i in range(len(nxt)):
                    if not s['done'][i]:
                        s['out'][i].append(int(nxt[i]))
                s['done'] = s['done'] | (nxt == s['end'])
                s['cur'], s['pos'] = nxt, s['pos'] + 1
            for x in SIDES:
                m, g = sent[partner(x)]
                m, g = ablate(m[:, None], g[:, None], mode, gen)
                g = g[:, 0]
                if mute is not None and x in mute:
                    g = g * (~torch.tensor(mute[x], device=g.device)).float()
                st[x]['inbox'] = (m[:, 0], g)
            if all(bool(st[x]['done'].all()) for x in SIDES):
                break
        for x in SIDES:
            self.taps[x].delta = None
        return {x: dict(tokens=st[x]['out'], gates=st[x]['gates'], msgs=st[x]['msgs'], logits=st[x]['logits'])
                for x in SIDES}
