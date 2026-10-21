"""Two language models from different families, linked by a learned channel of activation vectors.

Each model has a send head and a receive adapter. At every tick each model processes one token of its own text, its
send head reads the residual stream after block LS at that token and writes a message (a vector of K numbers with a
fixed norm) and a gate (send or not). The partner's receive adapter maps the gated message into the partner's residual
stream and adds it after block LR at the partner's next token. That position enters the partner's KV cache, so later
tokens can attend back to it. Neither model ever sees the other's text.

With the soft-token read (Pair.soft) a message instead enters the partner's context as a token of its own: a vector
in the partner's input-embedding space, placed just before the partner token it reaches, so that token and every later
one attend to it. The vector starts as text: the sender's next-token distribution at that tick, carried to the
partner's vocabulary over the tokens whose strings match exactly, as the expected input embedding sum_v p(v) E[v]
plus (1 - sum_v p(v)) null;
the receive adapter's output (zero at the start) is added on top. The probability of unmatched tokens goes to a
learned null vector, which is also the whole slot of a closed gate. Ablation 'zero' removes the slots altogether (they are masked out and the partner's positions are those of
its text alone), so a model whose link is cut reads exactly its own text.

Training reads the coupled models on teacher-forced text with Gauss-Seidel rounds: model A runs with the messages of
the previous round, then model B runs with A's new messages, and so on, every round in one parallel forward pass per
model. After R rounds every chain of up to 2R one-step hops between the models is exact. Generation (session.py) runs
the two models step by step with KV caches, which is exact for any chain. session.py also defines when each token is
read, and so where each message lands.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 256                      # message width
DEVICE = os.environ.get('NL_DEVICE') or (f'cuda:{os.environ.get("LOCAL_RANK", 0)}' if torch.cuda.is_available() else 'cpu')

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


def vocab_map(tx, ty):
    """Token pairs (id in tokenizer tx, id in tokenizer ty) whose decoded strings are identical; special tokens and
    tokens that decode to part of a character are left out."""
    special = set(tx.all_special_ids) | set(tx.added_tokens_decoder)
    strs = [tx.decode([v]) for v in range(len(tx))]
    enc = ty(strs, add_special_tokens=False).input_ids
    pairs = [(v, e[0]) for v, (s, e) in enumerate(zip(strs, enc))
             if v not in special and s and '\ufffd' not in s and len(e) == 1 and ty.decode(e) == s]
    return torch.tensor([a for a, _ in pairs]), torch.tensor([b for _, b in pairs])


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
    residual write). The write is the adapter's output times a learned size, times the gate. The adapter's last layer
    starts at zero (as LoRA's B), so the write starts at exactly zero and grows only along directions that lower the
    receiver's loss; an untrained message is never noise in the receiver's residual stream."""

    def __init__(self, d, size, hid=1024, zero_write=True, soft=False):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.send = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, K))
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)
        self.recv = nn.Sequential(nn.Linear(K, hid), nn.GELU(), nn.Linear(hid, d))
        if zero_write:
            nn.init.zeros_(self.recv[2].weight)
            nn.init.zeros_(self.recv[2].bias)
        self.log_size = nn.Parameter(torch.tensor(float(size)).log())
        if soft:
            self.null = nn.Parameter(torch.zeros(d))         # the slot of a closed gate (soft-token read)

    def message(self, h):
        """h [..., d] -> message [..., K] (norm sqrt(K)), gate probability [...]."""
        x = self.norm(h.float())
        m = F.normalize(self.send(x), dim=-1) * K ** 0.5
        return m, torch.sigmoid(self.gate(x).squeeze(-1))

    def write(self, m):
        """Gated message [..., K] -> residual write [..., d]."""
        return self.recv(m) * self.log_size.exp()


def gate_value(p, hard=True):
    """Straight-through hard gate: 1 where p > 0.5, with the gradient of p."""
    if not hard:
        return p
    return (p > 0.5).float() + p - p.detach()


def ablate_text(t, mode, gen=None):
    """The ablations of ablate() for the text embeddings t [B, T, d + 1] of the soft read (last channel: the matched
    probability, which random keeps)."""
    if mode == 'shuffle':
        return t.roll(1, dims=0)
    if mode == 'random':
        e = t[..., :-1]
        r = torch.randn(e.shape, generator=gen, device='cpu').to(t.device)
        return torch.cat([F.normalize(r, dim=-1) * e.norm(dim=-1, keepdim=True), t[..., -1:]], -1)
    return t


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

    def __init__(self, models, toks, lr, ls, sizes, zero_write=True, soft=False):
        super().__init__()
        self.models, self.toks, self.soft = models, toks, soft
        self.taps = {x: Tap(models[x], lr[x], ls[x]) for x in SIDES}
        d = {x: models[x].config.hidden_size for x in SIDES}
        self.side = nn.ModuleDict({x: Side(d[x], sizes[x], zero_write=zero_write, soft=soft) for x in SIDES})
        self.oracle = False        # soft read, test only: the sender's distribution is its teacher-forced next token
        if soft:                   # x -> partner's vocabulary: matched ids of x, and the partner's input embeddings
            self.vmap, self.vemb, self.vlook = {}, {}, {}
            for x in SIDES:
                y = partner(x)
                ix, iy = vocab_map(toks[x], toks[y])
                E = models[y].get_input_embeddings().weight.detach()
                self.vmap[x], self.vemb[x] = ix.to(E.device), E[iy.to(E.device)].clone()
                self.vlook[x] = torch.full((len(toks[x]),), -1, dtype=torch.long).index_put_((ix,), torch.arange(len(ix))).to(E.device)
        self.open = False          # every gate open (g = 1, no gradient to the gate): the warm-up of training
        self.write_ms = {}         # mean square of the last write into each model, at the positions it reached
        self.inbox = False         # True: every position gets the mean of all open messages received so far

    # ----- training: Gauss-Seidel rounds over teacher-forced text -----

    def _run(self, x, ids, att, delta, full, embeds=None, pos=None):
        """One forward pass of model x. Returns the residual after block ls and, when full, the last hidden state."""
        tap = self.taps[x]
        tap.delta, tap.stop = delta, not full
        dec, _ = inner(self.models[x])
        last = None
        try:
            last = dec(input_ids=ids, inputs_embeds=embeds, attention_mask=att, position_ids=pos,
                       use_cache=False).last_hidden_state
        except Stop:
            pass
        finally:
            tap.stop = False
        return tap.h, last

    def _send(self, x, h, b, mode=None, gen=None, last=None):
        """Messages of model x at its sending positions: m [B, T, K], gate g [B, T], probability p [B, T], valid, and
        with the soft read the sender's distribution in the partner's input space [B, T, d_partner + 1] (text_embedding)."""
        pos, valid = b[x]['send_pos'], b[x]['send_valid']
        hs = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        m, p = self.side[x].message(hs)
        p = p * valid
        g = valid.float() if self.open else gate_value(p) * valid
        m, g = ablate(m, g, mode, gen)
        if not self.soft:
            return m, g, p, valid
        if self.oracle:
            text = self.text_embedding(x, token=b[x]['next_tok'])
        else:
            text = self.text_embedding(x, hidden=last.gather(1, pos[..., None].expand(-1, -1, last.shape[-1])))
        return m, g, p, valid, ablate_text(text, mode, gen)

    @torch.no_grad()
    def text_embedding(self, x, hidden=None, logits=None, token=None):
        """The sender's next-token distribution (from its last hidden states, its logits, or a given token), carried
        to the partner's vocabulary: sum over the exactly matched tokens of p(v) E_partner[v], and as a last channel
        the matched probability sum_v p(v) (the rest goes to the partner's null vector in slot_vectors)
        [..., d_partner + 1]. Rows of 2048 at a time, to bound the memory of the vocabulary-wide distributions."""
        E = self.vemb[x]
        if token is not None:
            j = self.vlook[x][token.clamp(min=0)]
            ok = ((token >= 0) & (j >= 0)).float()[..., None]
            return torch.cat([E[j.clamp(min=0)].float() * ok, ok], -1)
        src = hidden if hidden is not None else logits
        flat = src.reshape(-1, src.shape[-1])
        _, head = inner(self.models[x])
        out = []
        for i in range(0, flat.shape[0], 2048):
            z = flat[i:i + 2048]
            q = torch.softmax((head(z) if hidden is not None else z).float(), -1)[:, self.vmap[x]]
            out.append(torch.cat([(q.to(E.dtype) @ E).float(), q.sum(-1, keepdim=True)], -1))
        return torch.cat(out).reshape(*src.shape[:-1], E.shape[-1] + 1)

    def slot_vectors(self, y, msg):
        """Slot vectors in model y's input from its partner's messages [B, T, d]: where the gate is open the text
        embedding, the null vector times the unmatched probability, and the receive adapter's output; the null vector
        where it is closed."""
        m, g, _, _, text = msg
        side = self.side[y]
        w = side.write(m)
        g = g[..., None]
        return g * (text[..., :-1] + (1 - text[..., -1:]) * side.null + w) + (1 - g) * side.null, w

    def _deliver(self, y, msg, b, L):
        """Residual writes into model y [B, L, d] from its partner's messages, each at its destination position;
        messages that reach the same position are averaged. With the inbox, every position instead gets the mean of
        all open messages that reached it or an earlier position, so a message stays in the stream once it arrives."""
        m, g, _, valid = msg
        dst = b[partner(y)]['dst']
        w = self.side[y].write(m) * g[..., None]                    # [B, T, d]
        ok = (valid & (dst >= 0)).float()
        pos = dst.clamp(min=0)
        out = torch.zeros(w.shape[0], L, w.shape[-1], device=w.device, dtype=w.dtype)
        out = out.scatter_add(1, pos[..., None].expand_as(w), w * ok[..., None])
        if self.inbox:
            c = torch.zeros(w.shape[0], L, device=w.device).scatter_add(1, pos, ok * g.detach()).cumsum(1)
            out = out.cumsum(1) / c.clamp(min=1)[..., None]
            hit = (c > 0).float()
        else:
            n = torch.zeros(w.shape[0], L, device=w.device).scatter_add(1, pos, ok)
            out = out / n.clamp(min=1)[..., None]
            hit = (n > 0).float()
        self.write_ms[y] = (out.detach().pow(2).mean(-1) * hit).sum() / hit.sum().clamp(min=1)   # for logging
        return out

    def _inputs(self, y, b, msg, mode):
        """Input embeddings of model y with its slots filled from the partner's messages (null before the first)."""
        e = self.models[y].get_input_embeddings()(b[y]['ids'])
        slot = b[y]['slot']
        if mode == 'zero':
            return e
        if msg is None:
            v = self.side[y].null.expand(e.shape)
        else:
            vv, w = self.slot_vectors(y, msg)
            v = vv.gather(1, slot.clamp(min=0)[..., None].expand(-1, -1, vv.shape[-1]))
            g = msg[1]
            self.write_ms[y] = (w.detach().pow(2).mean(-1) * g.detach()).sum() / g.detach().sum().clamp(min=1)
        return torch.where((slot >= 0)[..., None], v.to(e.dtype), e)

    def coupled(self, b, rounds=2, mode=None, gen=None):
        """Gauss-Seidel forward of both models on the batch b. Returns last hidden states per model and the messages."""
        msgs, last = {}, {}
        if self.soft:          # every round runs to the last block: the sender's next-token distribution is sent
            for r in range(rounds):
                for x in SIDES:
                    y = partner(x)
                    att, pos = (b[x]['att_cut'], b[x]['pos_cut']) if mode == 'zero' else (b[x]['att'], b[x]['pos'])
                    h, hl = self._run(x, None, att, None, True, self._inputs(x, b, msgs.get(y), mode), pos)
                    last[x] = hl
                    msgs[x] = self._send(x, h, b, mode=mode, gen=gen, last=hl)
            return last, msgs
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
