"""Breadcrumbs between agents: agents act one after another in a world of named places (crumb_games.py); a writer
leaves m crumbs after each of its segments at the segment's place, and every token of a later agent's segment at place
p reads the crumbs that earlier agents left at p. All agents are one language model (Qwen3-0.6B with one rank-16 LoRA
shared by every role), trained end to end on all games mixed: a reader's loss reaches the writers through the crumbs.

Each agent's tokens follow the chat template: its system prompt, then per segment a user turn and, if the segment has a
say, an assistant turn (trained with cross-entropy on the say and <|im_end|>, generated greedily at evaluation with the
true earlier turns in context); a writer's m crumb positions follow each segment that has a place, their inputs m
learned vectors (inputs_embeds).

Arms (--arm):
  cut       no crumbs
  resid     a crumb is its positions' states after each link layer (link.LAYERS); after the same layer, a reader's
            token at place p reads the crumbs at p through a link.Bridge whose time mask is replaced by this
            same-place mask (no crumb to read: the bridge writes zero)
  kv        a crumb is its positions' keys and values at every layer of the model, put in front of the reader's
            sequence as past_key_values with a 4D mask, so that a token at place p attends to the crumbs at p besides
            its own past; no new parameters: the shared model reads them natively. Agent a's positions start at
            a * SPAN, so crumbs always lie in the reader's past at the same distances in training and evaluation
  text      no crumbs; a reader's segment at place p starts with one line per earlier writer segment at p,
            "A note left here: {its user text} {its say}" (the format's ceiling)
  tok       token neuralese: each crumb is one of 64 tokens <z0>..<z63> (the base model's unused vocabulary rows,
            len(tokenizer) on); the writer's final hidden state at a crumb slot times a trainable table A [64, d] (not
            the tied embedding) gives its logits (the state standardized per message position, its cosine with A's
            rows times a learned scale starting at WSCALE: the raw final states share one large direction, which made
            every message the same code under straight-through Gumbel, and unbounded dot products saturated the
            softmax; A starts orthogonal at the embedding rows' norm). Training sends the soft mixture softmax(logits / tau) @ A (full gradient, no noise), tau annealed
            TAU, with a per-slot entropy penalty (BETA, ramped); evaluation sends the argmax token (hard), or the soft
            mixture in mode soft (hard one-hot forward, tau
            from 1.0 to 0.3 over training; argmax at evaluation); a reader's segment at place p gets every earlier
            writer's m tokens at p inside its user turn after the text, as inputs onehot @ A, and sees nothing else of
            the writer. Adversary (--adv lambda): the same base with its adapter disabled reads the same prompt, and
            for each choice group of the reader's says (crumb_games.choices) the loss adds lambda * relu(log p_base
            (gold | the C candidates' first differing token) - max(log(1/C), the same with every message masked)):
            the untrained model may gain nothing from the messages over chance or its own prior, and is never rewarded
            for being confidently wrong
Evaluation at test values, per game: says with crumbs on and with every crumb masked off; on trail also swap (the
follower's tokens in one visited room read the crumbs of a room it did not visit) and transplant (one visited room's
crumbs are replaced by a room's crumbs from another episode). Writes OUT/log.jsonl, OUT/scores.json, OUT/answers.json.
Usage: python swarm/crumbs.py --arm resid --m 4 --out OUT [--steps 1500] [--batch 16] [--games trail,...]
"""
import argparse
import collections
import contextlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import games as G                           # noqa: E402
from swarm.link import LAYERS, Bridge, blocks           # noqa: E402

BASE = 'Qwen/Qwen3-0.6B'
END = '<|im_end|>'
CRUMB_ARMS = ('resid', 'kv', 'tok')
ALPHABET = 64                                           # tok arm: message tokens
TAU = (1.0, 0.1)              # tok arm: softmax temperature of the trained messages, start and end
WSCALE = 10.                  # tok: writer logits = WSCALE x cosine(standardized state, A row): a random A starts
                              # near uniform; unbounded dot products saturated the softmax within 75 steps
A_LR = 0.1                    # tok: A's learning rate as a share of link_lr (A's entries are ~0.02)
BETA = 0.1                    # tok: per-slot entropy penalty, ramped from 0 over the first half of training
ADV_RAMP = 300                # tok: steps over which the adversary ramps in once the hard messages work
WORKING = (1.1, 0.8)          # tok: hard messages work once dev NLL <= 1.1 x soft and <= 0.8 x cut
CHUNK = 48                                              # streams per teacher-forced forward (--chunk)
SPAN = 1024                                             # positions per agent (kv arm)
MODES = ('on', 'off', 'swap', 'transplant')


class Place(Bridge):
    """link.Bridge with a same-place mask instead of the time mask: a reader's token reads only the crumbs that allow
    marks for it; each crumb's gate is the sender's general term."""

    def place(self, h, src, allow):
        """h [B, T, d] the reader's states, src [B, S, d] the crumbs' states, allow [B, T, S]. Returns the write
        [B, T, d] in h's dtype (zero where a token reads no crumb)."""
        B, T, _ = h.shape
        S, H, D = src.shape[1], self.heads, self.head_dim
        q = self.q(self.nq(h.float())).view(B, T, H, D)
        z = self.nk(src.float())
        key, val = self.k(z).view(B, S, H, D), self.v(z).view(B, S, H, D)
        gate = torch.sigmoid(self.g(z)[..., 0])
        s = torch.einsum('bthd,bshd->bhts', q, key) / math.sqrt(D)
        s = s.masked_fill(~allow[:, None], float('-inf'))
        empty = torch.einsum('bthd,hd->bht', q, self.null)[..., None] / math.sqrt(D)
        p = torch.softmax(torch.cat([empty, s], -1), -1)[..., 1:]
        out = torch.einsum('bhts,bs,bshd->bthd', p, gate, val).reshape(B, T, H * D)
        w = torch.tanh(self.alpha) * self.rms * self.o(out)
        return (w * allow.any(-1, keepdim=True)).to(h.dtype)


class Crumbs(nn.Module):
    """The model (with its LoRA), the crumb inputs and, for the resid arm, one Place bridge per link layer.

    A bank holds a batch's crumbs: place and agent [B, S] (place -1: an empty slot), and h (resid: per link layer
    [B, S, d]) or k, v (kv: per layer [B, kv heads, S, head dim])."""

    def __init__(self, model, tok, arm, m=4, layers=LAYERS):
        super().__init__()
        self.model, self.tok, self.arm, self.m, self.layers = model, tok, arm, m, tuple(layers)
        self.end = tok.convert_tokens_to_ids(END)
        d = model.config.hidden_size
        if arm in CRUMB_ARMS:
            std = float(model.get_input_embeddings().weight.float().std())
            self.crumb = nn.Parameter(torch.randn(m, d) * std)
        if arm == 'tok':
            spare = model.config.vocab_size - len(tok)
            if spare < ALPHABET:
                raise ValueError(f'only {spare} unused vocabulary rows; the alphabet needs {ALPHABET}')
            self.alien = list(range(len(tok), len(tok) + ALPHABET))     # <z0>..<z63>
            rows = float(model.get_input_embeddings().weight.float().norm(dim=-1).mean())
            self.A = nn.Parameter(nn.init.orthogonal_(torch.empty(ALPHABET, d)) * rows)
            self.wscale = nn.Parameter(torch.tensor(WSCALE))
            self.register_buffer('wmean', torch.zeros(m, d))      # per message position: each has its own crumb
            self.register_buffer('wvar', torch.ones(m, d))        # input, so pooled statistics left a fixed code
            self.tau, self.adv, self.adv_max, self.adv_sum, self.adv_n = TAU[0], 0., 0., 0., 0
            self.beta, self.ent_sum, self.ent_n, self.soft = 0., 0., 0, False
            self.codes = [collections.deque(maxlen=256) for _ in range(m)]   # recent training codes per position
        if arm == 'resid':
            self.readers = nn.ModuleList(Place(d) for _ in self.layers)
            stack = blocks(model)
            for i, l in enumerate(self.layers):
                stack[l].register_forward_hook(self._hook(i))
        self.ctx = None
        self.grab, self.last = False, None                 # the final norm's output, kept while grab is set
        model.get_base_model().model.norm.register_forward_hook(self._grab)
        self.head = self._ids('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        self.nl = self._ids('\n')

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def _grab(self, module, args, out):
        if self.grab:
            self.last = out

    def link_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith('model.')]

    def _hook(self, i):
        def hook(module, args, out):
            c = self.ctx
            if c is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            if c['src'] is not None:
                h = h + self.readers[i].place(h, c['src'][i], c['allow'])
            if c['got'] is not None:
                c['got'][i] = h
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    def stream(self, ep, a, remap=None):
        """Agent a's tokens: ids, place (the place a token reads, -1: none; remap {place: place it reads instead}),
        target (say tokens), seg (a say token's segment, -1: none), kind (crumb slot j, -1: a token), slots
        [(position, place)], asks [(segment, position where its say starts, place)] and, tok arm, reads [(position,
        writer agent, the writer's slot index)]: the message tokens it reads."""
        agent, remap = ep['agents'][a], remap or {}
        ids = self._ids(f'<|im_start|>system\n{agent["system"]}<|im_end|>\n')
        place, target, seg_of, kind = [-1] * len(ids), [False] * len(ids), [-1] * len(ids), [-1] * len(ids)
        slots, asks, reads = [], [], []

        def add(tokens, p, t=False, k=-1, sg=-1):
            ids.extend(tokens)
            place.extend([p] * len(tokens))
            target.extend([t] * len(tokens))
            seg_of.extend([sg] * len(tokens))
            kind.extend([k] * len(tokens))

        for s, seg in enumerate(agent['segments']):
            p = -1 if seg['place'] is None else seg['place']
            user = seg['user']
            if self.arm == 'text' and p >= 0:
                notes = [f'A note left here: {w["user"]}' + ('' if w['say'] is None else f' {w["say"]}')
                         for b in ep['agents'][:a] if b['write'] for w in b['segments'] if w['place'] == p]
                user = '\n'.join(notes + [user])
            rp = remap.get(p, p)
            if self.arm == 'tok' and rp >= 0:
                add(self._ids(f'<|im_start|>user\n{user}'), rp)
                for w in range(a):
                    writer = ep['agents'][w]
                    k = 0
                    for ws in writer['segments'] if writer['write'] else ():
                        if ws['place'] is None:
                            continue
                        if ws['place'] == rp:
                            for j in range(self.m):
                                reads.append((len(ids), w, k * self.m + j))
                                add([self.tok.pad_token_id], rp)
                        k += 1
                add(self._ids('<|im_end|>\n'), rp)
            else:
                add(self._ids(f'<|im_start|>user\n{user}<|im_end|>\n'), rp)
            if seg['say'] is not None:
                add(self.head, rp)
                asks.append((s, len(ids), rp))
                add(self._ids(seg['say']) + [self.end], rp, True, sg=s)
                add(self.nl, rp)
            if agent['write'] and p >= 0 and self.arm in CRUMB_ARMS:
                for j in range(self.m):
                    slots.append((len(ids), p))
                    add([self.tok.pad_token_id], rp, k=j)
        if len(ids) > SPAN:
            raise ValueError(f'agent stream of {len(ids)} tokens is longer than SPAN')
        return dict(ids=ids, place=place, target=target, seg=seg_of, kind=kind, slots=slots, asks=asks, reads=reads)

    def _pad(self, rows, device, left=False):
        """Tensors ids, att, place, target, kind [N, T] from per-row lists, padded on the right (or left)."""
        T = max(len(r['ids']) for r in rows)
        fill = dict(ids=self.tok.pad_token_id, place=-1, target=False, seg=-1, kind=-1)
        out = {k: torch.full((len(rows), T), v, dtype=torch.bool if k == 'target' else torch.long)
               for k, v in fill.items()}
        out['att'] = torch.zeros((len(rows), T), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r['ids'])
            sl = slice(T - n, T) if left else slice(0, n)
            for k in fill:
                out[k][i, sl] = torch.tensor(r[k], dtype=out[k].dtype)
            out['att'][i, sl] = 1
        R = max(len(r.get('reads', ())) for r in rows)
        rd = [[(-1, -1, -1)] * R for _ in rows]
        for i, r in enumerate(rows):
            off = T - len(r['ids']) if left else 0
            for k, (pos, w, j) in enumerate(r.get('reads', ())):
                rd[i][k] = (pos + off, w, j)
        rd = torch.tensor(rd, dtype=torch.long).reshape(len(rows), R, 3)
        out['rpos'], out['ragent'], out['ridx'] = rd[..., 0], rd[..., 1], rd[..., 2]
        return {k: v.to(device) for k, v in out.items()}

    def empty_bank(self, B, device):
        bank = dict(place=torch.full((B, 0), -1, dtype=torch.long, device=device),
                    agent=torch.zeros((B, 0), dtype=torch.long, device=device))
        if self.arm == 'resid':
            bank['h'] = None
        elif self.arm == 'tok':
            bank['y'] = None
        else:
            bank['k'] = bank['v'] = None
        return bank

    def _read_tokens(self, rows, x, emb, att, bank, on):
        """tok arm: emb with each read position's input onehot @ A of the message token it reads, and att with the
        read positions masked out where on is False or the bank has no such slot (a transplant donor without it)."""
        rpos, ra, ri = x['rpos'], x['ragent'], x['ridx']
        N, T = att.shape
        R, S = rpos.shape[1], bank['place'].shape[1]
        valid = rpos >= 0
        ok = torch.zeros_like(valid)
        if S and on and bank.get('y') is not None:
            first = {}
            for c, w in enumerate(bank['agent'][0].tolist()):
                first.setdefault(w, c)
            look = torch.full((max(int(ra.max()), max(first)) + 1,), -1, dtype=torch.long)
            for w, c in first.items():
                look[w] = c
            start = look.to(rpos.device)[ra.clamp(min=0)]
            col = (start + ri).clamp(0, S - 1)
            ok = valid & (start >= 0) & (start + ri < S) & (bank['place'][rows].gather(1, col) >= 0)
            y = bank['y'][0][rows].gather(1, col[..., None].expand(-1, -1, ALPHABET))
            msg = (y.to(self.A.dtype) @ self.A).to(emb.dtype)
            slot = torch.full((N, T + 1), -1, dtype=torch.long, device=att.device)
            slot.scatter_(1, torch.where(ok, rpos, T), torch.arange(R, device=att.device).expand(N, R).contiguous())
            slot = slot[:, :T]
            pick = msg.gather(1, slot.clamp(min=0)[..., None].expand(-1, -1, emb.shape[-1]))
            emb = torch.where((slot >= 0)[..., None], pick, emb)
        drop = valid & ~ok
        if bool(drop.any()):
            att = torch.cat([att, att[:, :1]], 1).scatter(1, torch.where(drop, rpos, T), 0)[:, :T]
        return emb, att

    def _forward(self, rows, x, a, bank, keep, capture, on=True):
        """One forward of agent a's tokens x (from _pad) for batch rows rows (a LongTensor). Returns the logits at keep
        (an int or index tensor) and, if capture, the states every crumb could be taken from: resid, per link layer
        [N, T, d]; kv, per layer (k, v) [N, heads, S + T, head dim] (S: the bank's slots)."""
        ids, att = x['ids'], x['att']
        emb = self.model.get_input_embeddings()(ids)
        if self.arm in CRUMB_ARMS:
            emb = torch.where((x['kind'] >= 0)[..., None], self.crumb.to(emb.dtype)[x['kind'].clamp(min=0)], emb)
        if self.arm == 'tok' and x.get('rpos') is not None and x['rpos'].shape[1]:
            emb, att = self._read_tokens(rows, x, emb, att, bank, on)
        pos = (att.cumsum(-1) - 1).clamp(min=0) + (a * SPAN if self.arm == 'kv' else 0)
        S = bank['place'].shape[1] if self.arm in CRUMB_ARMS else 0
        allow = None
        if S:
            bp, ba = bank['place'][rows], bank['agent'][rows]
            allow = ((bp[:, None, :] == x['place'][:, :, None]) & (x['place'][:, :, None] >= 0)
                     & (ba[:, None, :] < a) & on)
        if self.arm == 'kv':
            from transformers import DynamicCache
            T = ids.shape[1]
            own = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))[None] & att.bool()[:, None, :]
            own = own | torch.eye(T, dtype=torch.bool, device=ids.device)[None]   # padding rows see themselves
            mask = (torch.cat([allow, own], -1) if S else own)[:, None]
            cache = DynamicCache(ddp_cache_data=[(k[rows], v[rows]) for k, v in zip(bank['k'], bank['v'])]) \
                if S else None
            out = self.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos, past_key_values=cache,
                             use_cache=capture or S > 0, logits_to_keep=keep)
            return out.logits, ([(k, v) for k, v, *_ in out.past_key_values] if capture else None)
        if self.arm == 'resid':
            self.ctx = dict(src=[h[rows] for h in bank['h']] if S else None, allow=allow, got={} if capture else None)
        try:
            logits = self.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False,
                                logits_to_keep=keep).logits
            got = [self.ctx['got'][i] for i in range(len(self.layers))] if capture and self.ctx else None
        finally:
            self.ctx = None
        return logits, got

    def _append(self, bank, rows, streams, got, a):
        """bank with agent a's crumbs (rows' slots, taken from got) appended."""
        B, device = bank['place'].shape[0], bank['place'].device
        S0 = bank['place'].shape[1]
        n = max(len(st['slots']) for st in streams)
        idx = [[0] * n for _ in streams]
        pl = [[-1] * n for _ in range(B)]
        for i, (b, st) in enumerate(zip(rows.tolist(), streams)):
            for j, (t, p) in enumerate(st['slots']):
                idx[i][j], pl[b][j] = t, p
        idx, pl = torch.tensor(idx, device=device), torch.tensor(pl, device=device)

        def put(new):
            return new.new_zeros((B,) + tuple(new.shape[1:])).index_copy(0, rows, new)

        out = dict(place=torch.cat([bank['place'], pl], 1),
                   agent=torch.cat([bank['agent'], torch.full((B, n), a, dtype=torch.long, device=device)], 1))
        if self.arm == 'resid':
            new = [put(h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1]))) for h in got]
            out['h'] = new if bank['h'] is None else [torch.cat([o, x], 1) for o, x in zip(bank['h'], new)]
        elif self.arm == 'tok':
            hs = got.gather(1, idx[..., None].expand(-1, -1, got.shape[-1])).float()
            real = torch.arange(n, device=device)[None] < torch.tensor([len(st['slots']) for st in streams],
                                                                       device=device)[:, None]
            train = torch.is_grad_enabled() and self.training
            pos = torch.arange(n, device=device) % self.m
            if train:
                mu, var = self.wmean.clone(), self.wvar.clone()
                for r in range(self.m):
                    sel = real & (pos == r)[None]
                    if int(sel.sum()) > 1:
                        mu[r], var[r] = hs[sel].mean(0), hs[sel].var(0, unbiased=False)
                        with torch.no_grad():
                            self.wmean[r].lerp_(mu[r].detach(), 0.01)
                            self.wvar[r].lerp_(var[r].detach(), 0.01)
            else:
                mu, var = self.wmean, self.wvar
            z = (hs - mu[pos][None]) / (var[pos][None] + 1e-5).sqrt()
            logits = self.wscale * F.normalize(z, dim=-1) @ F.normalize(self.A.float(), dim=-1).T
            if train or self.soft:
                y = (logits / self.tau).softmax(-1)
                if train:
                    ent = -(y * y.clamp_min(1e-9).log()).sum(-1)[real]
                    self.ent_sum, self.ent_n = self.ent_sum + ent.sum(), self.ent_n + ent.numel()
                    code = y.argmax(-1)
                    for r in range(self.m):
                        self.codes[r].extend(code[real & (pos == r)[None]].tolist())
            else:
                y = F.one_hot(logits.argmax(-1), ALPHABET).float()
            out['y'] = [put(y)] if bank['y'] is None else [torch.cat([bank['y'][0], put(y)], 1)]
        else:
            ks, vs = [], []
            for k, v in got:
                g = (S0 + idx)[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[-1])
                ks.append(put(k.gather(2, g)))
                vs.append(put(v.gather(2, g)))
            out['k'] = ks if bank['k'] is None else [torch.cat([o, x], 2) for o, x in zip(bank['k'], ks)]
            out['v'] = vs if bank['v'] is None else [torch.cat([o, x], 2) for o, x in zip(bank['v'], vs)]
        return out

    def run(self, eps, device, read=None, need_loss=True):
        """Every agent's true tokens, in time order. Returns the says' summed cross-entropy, their token count and
        the bank. read: per episode {agent: {place: place it reads instead}}."""
        bank = self.empty_bank(len(eps), device)
        total, count = 0., 0
        for a in range(max(len(ep['agents']) for ep in eps)):
            def needed(ep):
                ag = ep['agents'][a]
                return (need_loss and any(s['say'] is not None for s in ag['segments'])) or \
                    (self.arm in CRUMB_ARMS and ag['write'])
            rows = [b for b, ep in enumerate(eps) if a < len(ep['agents']) and needed(ep)]
            if not rows:
                continue
            streams = [self.stream(eps[b], a, (read or {}).get(b, {}).get(a)) for b in rows]
            x = self._pad(streams, device)
            writes = self.arm in CRUMB_ARMS and any(st['slots'] for st in streams)
            sel = x['target'][:, 1:].nonzero() if need_loss else None     # (row, position) predicting a say token
            rows_t = torch.tensor(rows, device=device)
            self.grab = (sel is not None and len(sel) > 0) or (writes and self.arm == 'tok')
            try:
                _, got = self._forward(rows_t, x, a, bank, 1, writes)
                h = self.last
            finally:
                self.grab, self.last = False, None
            if sel is not None and len(sel):
                logits = self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float()
                total = total + F.cross_entropy(logits, x['ids'][sel[:, 0], sel[:, 1] + 1], reduction='sum')
                count += len(sel)
                if self.arm == 'tok' and self.adv and torch.is_grad_enabled():
                    self._adversary(rows_t, x, a, bank, [(eps[b], st) for b, st in zip(rows, streams)])
            if writes:
                bank = self._append(bank, rows_t, streams, h if self.arm == 'tok' else got, a)
        return total, count, bank

    def loss(self, eps, device):
        if self.arm == 'tok':
            self.adv_sum, self.adv_n, self.ent_sum, self.ent_n = 0., 0, 0., 0
        total, count, _ = self.run(eps, device)
        out = total / max(1, count)
        if self.arm == 'tok' and self.adv_n:
            out = out + self.adv * self.adv_sum / self.adv_n
        if self.arm == 'tok' and self.ent_n and self.beta:
            out = out + self.beta * self.ent_sum / self.ent_n
        return out

    def adv_targets(self, ep, a, st):
        """Per choice group of agent a's says (crumb_games.choices): (the position whose next token is the first where
        the alternatives part from the gold say, the gold token there, the alternatives' tokens there)."""
        out = []
        for s, cut, _ in st['asks']:
            for group in CG.choices(ep, a, s):
                gold, alts = self._ids(group[0]), [self._ids(c) for c in group[1:]]
                div = [next((i for i, (u, v) in enumerate(zip(gold, c)) if u != v), None) for c in alts]
                if all(d is None for d in div):
                    continue
                first = min(d for d in div if d is not None)
                out.append((cut + first - 1, gold[first], [c[first] for c, d in zip(alts, div) if d == first]))
        return out

    def _adversary(self, rows_t, x, a, bank, items):
        """Adds to adv_sum, adv_n: the base model with its adapter disabled reads the same prompts (the same message
        inputs, so its gradient reaches the writers and A); per choice group relu(log p(gold) - max(log(1/C), log
        p(gold) with the messages masked)) over the C candidates' first differing token: only what the base gains from
        the messages beyond chance and its own prior counts."""
        targets = [(i, t) for i, (ep, st) in enumerate(items) if st['reads'] for t in self.adv_targets(ep, a, st)]
        if not targets:
            return
        def read(on):
            self.grab = True
            try:
                with self.model.disable_adapter():
                    self._forward(rows_t, x, a, bank, 1, False, on)
                return self.last
            finally:
                self.grab, self.last = False, None

        hb = read(True)
        with torch.no_grad():
            h0 = read(False)
        dev = hb.device
        at = (torch.tensor([i for i, _ in targets], device=dev), torch.tensor([t[0] for _, t in targets], device=dev))
        C = max(1 + len(t[2]) for _, t in targets)
        cand = torch.tensor([[t[1]] + t[2] + [-1] * (C - 1 - len(t[2])) for _, t in targets], device=dev)
        W = self.model.get_output_embeddings().weight[cand.clamp(min=0)].float()

        def logp(h):
            lg = torch.einsum('gd,gcd->gc', h[at].float(), W).masked_fill(cand < 0, float('-inf'))
            return lg.log_softmax(-1)[:, 0]

        n = (cand >= 0).sum(-1).float()
        floor = torch.maximum(logp(h0), -n.log())
        self.adv_sum = self.adv_sum + F.relu(logp(hb) - floor).sum()
        self.adv_n += len(targets)

    @torch.no_grad()
    def answer(self, eps, device, on=True, read=None, donor=None, shift=False, base=False, bank=None):
        """{(episode, agent, segment): greedy say}, every earlier turn true. donor: {episode: (place, other episode,
        its place)}, whose crumbs from agent 0 replace this episode's agent-0 crumbs at place; shift: every episode
        reads the next episode's crumbs; base: the readers are the base model with the adapter disabled (the writers
        trained); bank: these crumbs instead of the writers' own."""
        if bank is None:
            _, _, bank = self.run(eps, device, read, need_loss=False)
        if donor:
            bank = transplant(bank, donor)
        if shift:
            bank = shifted(bank)
        out = {}
        for a in range(max(len(ep['agents']) for ep in eps)):
            items, prompts, rows = [], [], []
            for b, ep in enumerate(eps):
                if a >= len(ep['agents']):
                    continue
                st = self.stream(ep, a, (read or {}).get(b, {}).get(a))
                for s, cut, p in st['asks']:
                    say = ep['agents'][a]['segments'][s]['say']
                    items.append((b, s, p, len(self._ids(say)) + 1))
                    prompts.append(dict({k: st[k][:cut] for k in ('ids', 'place', 'target', 'seg', 'kind')},
                                        reads=[r for r in st['reads'] if r[0] < cut]))
                    rows.append(b)
            if not items:
                continue
            x = self._pad(prompts, device, left=True)
            rows_t = torch.tensor(rows, device=device)
            n = len(items)
            done = torch.zeros(n, dtype=torch.bool, device=device)
            gen = [[] for _ in range(n)]
            seg_place = torch.tensor([p for _, _, p, _ in items], device=device)
            for _ in range(max(t for *_, t in items) + 8):
                with self.model.disable_adapter() if base else contextlib.nullcontext():
                    logits, _ = self._forward(rows_t, x, a, bank, 1, False, on)
                nxt = logits[:, -1].argmax(-1)
                for i, (t, d) in enumerate(zip(nxt.tolist(), done.tolist())):
                    if not d:
                        gen[i].append(t)
                live = ~done
                x = dict(x, ids=torch.cat([x['ids'], torch.where(live, nxt, self.tok.pad_token_id)[:, None]], 1),
                         att=torch.cat([x['att'], live.long()[:, None]], 1),
                         place=torch.cat([x['place'], seg_place[:, None]], 1),
                         target=torch.cat([x['target'], torch.zeros_like(x['target'][:, :1])], 1),
                         seg=torch.cat([x['seg'], torch.full_like(x['seg'][:, :1], -1)], 1),
                         kind=torch.cat([x['kind'], torch.full_like(x['kind'][:, :1], -1)], 1))
                done = done | (nxt == self.end)
                if done.all():
                    break
            for (b, s, _, _), g in zip(items, gen):
                out[(b, a, s)] = self.tok.decode(g, skip_special_tokens=True).strip()
        return out

    @torch.no_grad()
    def forced(self, eps, device, mode='on', choose=True, chunk=None, detail=False, bank=None):
        """Every say read teacher-forced with the true earlier turns: per episode the mean NLL (nats per say: its
        tokens and <|im_end|>) and the fraction of its choice groups (crumb_games.choices) whose gold say has the
        highest log-prob (None: no group, or choose False; detail: per episode the list of its groups' results, in
        order). mode: on, off (every crumb masked), transplant (each episode reads the next episode's crumbs), base
        (tok arm: the trained writers' messages read by the base model with its adapter disabled) or base_off (the base
        model with every message masked: its own prior); bank: these crumbs instead of the writers' own."""
        if bank is None:
            self.soft = mode == 'soft'
            try:
                _, _, bank = self.run(eps, device, need_loss=False)
            finally:
                self.soft = False
        if mode == 'transplant':
            bank = shifted(bank)
        chunk = chunk or CHUNK
        nll = [[] for _ in eps]
        gold, alt = {}, {}
        for a in range(max(len(ep['agents']) for ep in eps)):
            jobs = []
            for b, ep in enumerate(eps):
                if a >= len(ep['agents']) or all(sg['say'] is None for sg in ep['agents'][a]['segments']):
                    continue
                jobs.append((b, None, self.stream(ep, a)))
                for s, sg in enumerate(ep['agents'][a]['segments']):
                    if choose and sg['say'] is not None:
                        for g, group in enumerate(CG.choices(ep, a, s)):
                            for c, cand in enumerate(group[1:], 1):
                                jobs.append((b, (s, g, c), self.stream(with_say(ep, a, s, cand), a)))
            for i in range(0, len(jobs), chunk):
                part = jobs[i:i + chunk]
                with self.model.disable_adapter() if mode in ('base', 'base_off') else contextlib.nullcontext():
                    lps = self._say_logprobs(torch.tensor([b for b, _, _ in part], device=device),
                                             [st for _, _, st in part], a, bank, mode not in ('off', 'base_off'),
                                             device)
                for (b, key, _), d in zip(part, lps):
                    if key is None:
                        nll[b] += [-v for v in d.values()]
                        gold.update({(b, a, s): v for s, v in d.items()})
                    else:
                        s, g, c = key
                        alt.setdefault((b, a, s, g), []).append(d[s])
        right = [[] for _ in eps]
        for (b, a, s, g), vals in alt.items():
            right[b].append(float(gold[(b, a, s)] > max(vals)))
        return ([sum(v) / len(v) if v else None for v in nll],
                right if detail else [sum(v) / len(v) if v else None for v in right])

    def _say_logprobs(self, rows, streams, a, bank, on, device):
        """Per stream {segment: summed log-prob of its say tokens}."""
        x = self._pad(streams, device)
        sel = x['target'][:, 1:].nonzero()
        self.grab = True
        try:
            self._forward(rows, x, a, bank, 1, False, on)
            h = self.last
        finally:
            self.grab, self.last = False, None
        logits = self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float()
        lp = logits.log_softmax(-1).gather(1, x['ids'][sel[:, 0], sel[:, 1] + 1][:, None])[:, 0]
        seg = x['seg'][sel[:, 0], sel[:, 1] + 1]
        out = [{} for _ in streams]
        for i, sg, v in zip(sel[:, 0].tolist(), seg.tolist(), lp.tolist()):
            out[i][sg] = out[i].get(sg, 0.) + v
        return out

    @torch.no_grad()
    def calibrate(self, eps, device):
        """Sets each Place bridge's write unit to the base model's median per-token residual RMS at its layer."""
        got = {}
        stack = blocks(self.model)
        handles = [stack[l].register_forward_hook(
            lambda m, a, o, i=i: got.__setitem__(i, (o[0] if isinstance(o, tuple) else o).float()))
            for i, l in enumerate(self.layers)]
        x = self._pad([self.stream(ep, a) for ep in eps for a in range(len(ep['agents']))], device)
        try:
            with self.model.disable_adapter():
                self.model(input_ids=x['ids'], attention_mask=x['att'], use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        real = x['att'].bool().clone()
        real[:, 0] = False                                # each row's first token (massive norms)
        for i, h in got.items():
            self.readers[i].rms.fill_(float(h.pow(2).mean(-1).sqrt()[real].median()))


def with_say(ep, a, s, say):
    """ep with agent a's segment s saying say instead."""
    agents = list(ep['agents'])
    segs = list(agents[a]['segments'])
    segs[s] = dict(segs[s], say=say)
    agents[a] = dict(agents[a], segments=segs)
    return dict(ep, agents=agents)


def shifted(bank):
    """bank with each episode's crumbs replaced by the next episode's (the last episode's by the first's)."""
    B = bank['place'].shape[0]
    idx = (torch.arange(B, device=bank['place'].device) + 1) % B
    return {k: None if v is None else [t[idx] for t in v] if isinstance(v, list) else v[idx] for k, v in bank.items()}


def transplant(bank, donor):
    """A copy of bank where episode b's agent-0 crumbs at place pa are those of episode j's agent-0 crumbs at pc, for
    each donor[b] = (pa, j, pc)."""
    out = dict(bank)
    for name, dim in (('h', 1), ('k', 2), ('v', 2), ('y', 1)):
        if bank.get(name) is None:
            continue
        rows = []
        for old in bank[name]:
            t = old.clone()
            for b, (pa, j, pc) in donor.items():
                dst = ((bank['place'][b] == pa) & (bank['agent'][b] == 0)).nonzero()[:, 0]
                src = ((bank['place'][j] == pc) & (bank['agent'][j] == 0)).nonzero()[:, 0]
                if dim == 1:
                    t[b, dst] = old[j, src]
                else:
                    t[b, :, dst] = old[j, :, src]
            rows.append(t)
        out[name] = rows
    return out


def trail_summary(eps, texts):
    """Trail's parts: objects named right, key named right, and the key by case (crumb_games.case)."""
    parts = [(CG.trail_parts(ep, t), CG.case(ep)) for ep, t in zip(eps, texts)]
    out = dict(objects=sum(sum(o) for (o, _), _ in parts) / (CG.N_VISIT * len(parts)),
               key=sum(k for (_, k), _ in parts) / len(parts),
               reachable=sum(c != 'none' for _, c in parts) / len(parts))
    for c in CG.CASES:
        got = [k for (_, k), cc in parts if cc == c]
        out[f'key_{c}'] = sum(got) / max(1, len(got))
        out[f'n_{c}'] = len(got)
    return out


def generated(net, device, eps, game, bs=50, keep_rows=40):
    """Greedy says of eps (true earlier turns) with crumbs on and, for the crumb arms, off; on trail also swap (the
    follower's tokens in one visited room read the crumbs of a room it did not visit) and room (one visited room's
    crumbs replaced by a room's crumbs of the next episode). Returns {mode: summary} and the first keep_rows
    episodes' says per mode."""
    n, crumbs = len(eps), net.arm in CRUMB_ARMS
    modes = ['on'] + (['off'] if crumbs else []) + (['swap', 'room'] if crumbs and game == 'trail' else []) + \
        (['base'] if net.arm == 'tok' else [])
    scores, rows = {}, []
    for mode in modes:
        prng = random.Random(f'{game}-{mode}')
        texts, plans = {}, {}
        for s0 in range(0, n, bs):
            chunk = eps[s0:s0 + bs]
            read, donor = None, None
            if mode == 'swap':
                read = {}
                for b, ep in enumerate(chunk):
                    t = ep['trail']
                    pa = prng.choice(t['visit'])
                    pb = prng.choice([r for r in range(CG.N_ROOMS) if r not in t['visit']])
                    read[b] = {1: {pa: pb}}
                    plans[s0 + b] = (pa, pb)
            if mode == 'room':
                donor = {}
                for b, ep in enumerate(chunk):
                    t, j = ep['trail'], (b + 1) % len(chunk)
                    pa = prng.choice(t['visit'])
                    pc = prng.choice([r for r in range(CG.N_ROOMS)
                                      if chunk[j]['trail']['objects'][r] != t['objects'][pa]])
                    donor[b] = (pa, j, pc)
                    plans[s0 + b] = (pa, s0 + j, pc)
            got = net.answer(chunk, device, on=mode != 'off', read=read, donor=donor, base=mode == 'base')
            texts.update({(s0 + b, a, sg): t for (b, a, sg), t in got.items()})
        vals = [CG.score(eps[b], a, sg, t) for (b, a, sg), t in texts.items()]
        summary = dict(score=sum(vals) / len(vals), n=len(vals))
        if game == 'trail':
            final = [texts[(b, 1, len(eps[b]['agents'][1]['segments']) - 1)] for b in range(n)]
            summary.update(trail_summary(eps, final))
            obj = lambda b: eps[b]['trail']['objects']                                      # noqa: E731
            if mode == 'swap':
                summary['follows_crumb'] = sum(CG.said(eps[b], final[b], pa) == obj(b)[pb]
                                               for b, (pa, pb) in plans.items()) / n
                summary['keeps_own'] = sum(CG.said(eps[b], final[b], pa) == obj(b)[pa] for b, (pa, _) in plans.items()) / n
            if mode == 'room':
                summary['follows_crumb'] = sum(CG.said(eps[b], final[b], pa) == obj(j)[pc]
                                               for b, (pa, j, pc) in plans.items()) / n
                summary['keeps_own'] = sum(CG.said(eps[b], final[b], pa) == obj(b)[pa]
                                           for b, (pa, _, _) in plans.items()) / n
        scores[mode] = summary
        for b in range(min(n, keep_rows)):
            says = {f'{a}/{sg}': t for (bb, a, sg), t in texts.items() if bb == b}
            rows.append(dict(game=game, mode=mode, episode=eps[b], says=says, plan=plans.get(b)))
    return scores, rows


def evaluate(net, device, n=300, n_choice=150, n_gen=100, seed=12345, games=None, bs=50, keep_rows=40):
    """Per game, on the same n fixed-seed test episodes for every arm: per mode (on; for the crumb arms also off and
    transplant, the within-model controls) each episode's mean say NLL and, on the first n_choice episodes, its
    forced-choice accuracy; greedy says of the first n_gen episodes (generated()). Returns {game: dict(tf={mode:
    means}, episodes={mode: dict(nll, choice)}, gen={mode: summary})} and the kept generated rows."""
    net.eval()
    out, rows = {}, []
    for game in games or CG.GAMES:
        rng = random.Random(f'{seed}-{game}')
        eps = [CG.episode(game, rng, True) for _ in range(n)]
        modes = ['on'] + (['off', 'transplant'] if net.arm in CRUMB_ARMS else []) + \
            (['soft', 'base', 'base_off'] if net.arm == 'tok' else [])
        tf, per = {}, {}
        for mode in modes:
            nll, choice = [], []
            for s0 in range(0, n, bs):
                a_, b_ = net.forced(eps[s0:s0 + bs], device, mode, choose=s0 < n_choice)
                nll += a_
                choice += b_
            got = [c for c in choice if c is not None]
            tf[mode] = dict(nll=sum(nll) / len(nll), choice=sum(got) / len(got) if got else None, n=len(nll),
                            n_choice=len(got))
            per[mode] = dict(nll=nll, choice=choice)
        gen, kept = generated(net, device, eps[:n_gen], game, bs, keep_rows) if n_gen else ({}, [])
        out[game] = dict(tf=tf, episodes=per, gen=gen)
        rows += kept
    net.train()
    return out, rows


@torch.no_grad()
def dev_nll(net, device, eps, bs=50, mode='on'):
    """Mean say NLL over eps (crumbs on, or forced's mode)."""
    net.eval()
    got = []
    for s0 in range(0, len(eps), bs):
        got += [v for v in net.forced(eps[s0:s0 + bs], device, mode=mode, choose=False)[0] if v is not None]
    net.train()
    return sum(got) / len(got)


@torch.no_grad()
def probe(net, device, n_fit=480, n_test=300, bs=48, every=3):
    """How much an object's crumbs hold, apart from any reader: ridge maps (fit on training-value trail episodes)
    from the crumbs at a room (its m slots, one layer: resid the state after a link layer, kv the keys and values of a
    layer, every every-th layer and the last) to its object's input embedding (the mean over the word's tokens, with
    a leading space); at test values each map's output picks the nearest (cosine) of the test words. The ridge
    strength is chosen on training words held out of the fit. Returns per-layer accuracy, the best, and chance."""
    emb = net.model.get_input_embeddings().weight.float()

    def word_vec(w):
        return emb[net._ids(' ' + w)].mean(0)

    def collect(test, n, seed):
        rng = random.Random(seed)
        feats, words = {}, []
        for s0 in range(0, n, bs):
            eps = [CG.episode('trail', rng, test) for _ in range(min(bs, n - s0))]
            _, _, bank = net.run(eps, device, need_loss=False)
            if 'y' in bank:
                layers = {'tokens': bank['y'][0]}
            elif 'h' in bank:
                layers = {net.layers[i]: h for i, h in enumerate(bank['h'])}
            else:
                L = len(bank['k'])
                keep = sorted(set(range(0, L, every)) | {L - 1})
                layers = {l: torch.cat([bank['k'][l], bank['v'][l]], 1).transpose(1, 2).flatten(2) for l in keep}
            for b, ep in enumerate(eps):
                for r in range(CG.N_ROOMS):
                    sl = ((bank['place'][b] == r) & (bank['agent'][b] == 0)).nonzero()[:, 0]
                    for l, f in layers.items():
                        feats.setdefault(l, []).append(f[b, sl].reshape(-1).float().cpu())
                    words.append(ep['trail']['objects'][r])
        return {l: torch.stack(v) for l, v in feats.items()}, words

    fit, fit_words = collect(False, n_fit, 4242)
    test, test_words = collect(True, n_test, 4343)
    pool = sorted(G.WORD_POOLS[1])
    vocab = sorted(set(fit_words))
    held = set(vocab[::5])                                   # training words held out to choose the ridge strength
    Y = torch.stack([word_vec(w) for w in fit_words]).cpu()
    cand = torch.stack([word_vec(w) for w in pool]).cpu()

    def ridge(X, Y, lam):
        mx, my = X.mean(0), Y.mean(0)
        Xc = X - mx
        K = Xc @ Xc.T
        alpha = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K)), Y - my)
        return lambda Z: (Z - mx) @ (Xc.T @ alpha) + my

    def nearest(pred, cands, names, truth):
        sim = F.normalize(pred, dim=-1) @ F.normalize(cands, dim=-1).T
        return sum(names[i] == t for i, t in zip(sim.argmax(-1).tolist(), truth)) / len(truth)

    out = {}
    for l, X in fit.items():
        tr = torch.tensor([w not in held for w in fit_words])
        hw = sorted(held)
        hc = torch.stack([word_vec(w) for w in hw]).cpu()
        best = max((nearest(ridge(X[tr], Y[tr], lam)(X[~tr]), hc, hw, [w for w, t in zip(fit_words, tr) if not t]),
                    lam) for lam in (1e-3, 1e-2, 1e-1, 1.0))
        out[l] = nearest(ridge(X, Y, best[1])(test[l]), cand, pool, test_words)
    top = max(out, key=out.get)
    return dict(layers={str(l): v for l, v in out.items()}, best_layer=top, best=out[top], chance=1 / len(pool),
                n_test=len(test_words))


def evaluate_oldgrowth(net, device, sizes=None, n=None, n_gen=100, bs=25, log=None):
    """oldgrowth's test sets (crumb_games.oldgrowth_eval) per number of walkers G, crumbs on and off: the asker's say
    NLL, its forced choice per asked room (its object against the other 7), and the greedy answers' per-room accuracy
    on the first n_gen episodes, each over all asked rooms and over the rooms some walker saw (the ceiling: the share
    of rooms seen)."""
    net.eval()
    out = {}
    mean = lambda v: sum(v) / len(v) if v else None                                       # noqa: E731
    for g in sizes or CG.OG_SIZES:
        eps = CG.oldgrowth_eval(g)[:n]
        seen = [[p['seen'] for p in CG.oldgrowth_parts(ep, '')] for ep in eps]
        rec = dict(ceiling=mean([x for v in seen for x in v]))
        for mode in ('on', 'off'):
            nll, right = [], []
            for s0 in range(0, len(eps), bs):
                a_, r_ = net.forced(eps[s0:s0 + bs], device, mode, detail=True)
                nll += a_
                right += r_
            pairs = [(r, x) for rr, ss in zip(right, seen) for r, x in zip(rr, ss)]
            texts = {}
            for s0 in range(0, n_gen, bs):
                got = net.answer(eps[s0:s0 + bs], device, on=mode == 'on')
                texts.update({s0 + b: t for (b, a, sg), t in got.items()})
            parts = [p for b, t in texts.items() for p in CG.oldgrowth_parts(eps[b], t)]
            rec[mode] = dict(nll=mean(nll), choice=mean([r for r, _ in pairs]),
                             choice_seen=mean([r for r, x in pairs if x]),
                             choice_unseen=mean([r for r, x in pairs if not x]),
                             gen=mean([p['right'] for p in parts]), gen_seen=mean([p['right'] for p in parts if p['seen']]))
        if torch.cuda.is_available():
            rec['gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
        out[g] = rec
        if log:
            log(dict(oldgrowth=g, **{k: v for k, v in rec.items()}))
    net.train()
    return out


def intercept(net, device, n=2000, others=200, bs=50):
    """tok arm: every writer's messages (argmax tokens) on n training-value and n test-value trail episodes and on
    others of each other game, with what the writer saw: dict(alphabet, m, messages=[dict(split, game, episode,
    writer, segment, place, room, tokens (alphabet indices), vocab_ids, note (the text arm's English note for the same
    segment) and, on trail, object, key_here, key_room, key_before (the writer passed the key's room earlier), rank
    (the writer's k-th room), follower_visits)], trail_episodes={split: [episode]}). Episodes of game g, split s:
    random.Random(f'intercept-{g}-{s}'), crumb_games.episode(g, rng, s == 'test') in turn."""
    net.eval()
    out = dict(alphabet=dict(vocab_ids=net.alien, names=[f'<z{k}>' for k in range(ALPHABET)]), m=net.m, messages=[],
               trail_episodes={})
    for split in ('train', 'test'):
        for game in CG.GAMES:
            count = n if game == 'trail' else others
            rng = random.Random(f'intercept-{game}-{split}')
            eps = [CG.episode(game, rng, split == 'test') for _ in range(count)]
            if game == 'trail':
                out['trail_episodes'][split] = eps
            for s0 in range(0, count, bs):
                chunk = eps[s0:s0 + bs]
                with torch.no_grad():
                    _, _, bank = net.run(chunk, device, need_loss=False)
                if bank.get('y') is None:
                    continue
                toks, place, agent = bank['y'][0].argmax(-1).tolist(), bank['place'].tolist(), bank['agent'].tolist()
                for b, ep in enumerate(chunk):
                    for a, ag in enumerate(ep['agents']):
                        if not ag['write']:
                            continue
                        cols = [c for c in range(len(place[b])) if agent[b][c] == a and place[b][c] >= 0]
                        k = 0
                        for sg_i, sg in enumerate(ag['segments']):
                            if sg['place'] is None:
                                continue
                            ids = [toks[b][c] for c in cols[k * net.m:(k + 1) * net.m]]
                            k += 1
                            r = sg['place']
                            rec = dict(split=split, game=game, episode=s0 + b, writer=a, segment=sg_i, place=r,
                                       room=ep['places'][r], tokens=ids, vocab_ids=[net.alien[i] for i in ids],
                                       note=f'A note left here: {sg["user"]}' + ('' if sg['say'] is None
                                                                                else f' {sg["say"]}'))
                            if game == 'trail':
                                t = ep['trail']
                                rec.update(object=t['objects'][r], key_here=r == t['key'],
                                           key_room=ep['places'][t['key']],
                                           key_before=t['order'].index(t['key']) < t['order'].index(r),
                                           rank=t['order'].index(r), follower_visits=r in t['visit'])
                            out['messages'].append(rec)
    net.train()
    return out


def load(out, device):
    """The trained net of a run folder (its args.json and trained.pt)."""
    out = Path(out)
    args = json.loads((out / 'args.json').read_text())
    net = build(device, args['arm'], args['m'], base=args.get('base', BASE))
    net.load_state_dict(torch.load(out / 'trained.pt', map_location=device)['weights'], strict=False)
    if args['arm'] == 'tok':
        net.adv = net.adv_max = args.get('adv', 0.)
        net.tau = TAU[1]
    return net


def build(device, arm, m, base=BASE, rank=16, model=None, tok=None, layers=LAYERS):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = model or AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True,
                                                          attn_implementation='sdpa')
    model = get_peft_model(model.to(device), LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    return Crumbs(model, tok, arm, m, layers).to(device)


def train(net, device, steps, batch, seed, games=None, out=None, lr=2e-4, link_lr=1e-3, every=100, patience=300,
          min_steps=600, micro=None, weights=None, curriculum=None):
    """steps of batch episodes of one game each (similar lengths, little padding), the game drawn at random for each
    step, in backward passes of micro episodes (the mean of their losses). Every every steps, the mean say NLL of 8 fixed training-value episodes per game; training stops once that
    has not improved by 1% for patience steps (after min_steps). curriculum [(until_step, G_max)]: oldgrowth episodes
    draw G from 1 to G_max up to until_step (afterwards the game's own 1 to 4). tok: each dev evaluation also takes
    the soft messages and the cut; once the hard messages work (WORKING) the adversary ramps in over ADV_RAMP steps,
    and training runs at least ADV_RAMP + 300 steps past that. The dev NLL that stops training is the hard one.
    Returns the steps run and the NLL record."""
    games = games or CG.GAMES
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev_rng = random.Random(777)
    dev = [CG.episode(g, dev_rng) for g in games for _ in range(8)]
    if net.arm == 'resid':
        cal = random.Random(99)
        net.calibrate([CG.episode(cal.choice(games), cal) for _ in range(8)], device)
    lora = [p for p in net.model.parameters() if p.requires_grad]
    link = net.link_params()
    alpha = [p for p in link if net.arm == 'tok' and p is net.A]
    rest = [p for p in link if not any(p is q for q in alpha)]
    opt = torch.optim.AdamW([dict(params=lora, lr=lr)] + ([dict(params=rest, lr=link_lr)] if rest else []) +
                            ([dict(params=alpha, lr=link_lr * A_LR)] if alpha else []),
                            weight_decay=0.0)
    warm = min(50, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1, s / steps))))
    log = open(out / 'log.jsonl', 'a') if out else None
    started = time.monotonic()
    best, best_step, record, work = float('inf'), 0, [], None
    for step in range(1, steps + 1):
        game = rng.choices(games, [(weights or {}).get(g, 1.) for g in games])[0]
        gmax = next((g for until, g in curriculum or () if step <= until), None) if game == 'oldgrowth' else None
        eps = [CG.episode(game, rng, **(dict(G=rng.randint(1, gmax)) if gmax else {})) for _ in range(batch)]
        if net.arm == 'tok':
            net.adv = net.adv_max * min(1., max(0., (step - work) / ADV_RAMP)) if work else 0.
            net.beta = BETA * min(1., step / max(1, steps / 2))
        parts = [eps[i:i + (micro or batch)] for i in range(0, batch, micro or batch)]
        loss = 0.
        for part in parts:
            part_loss = net.loss(part, device) / len(parts)
            part_loss.backward()
            loss += part_loss.detach()
        torch.nn.utils.clip_grad_norm_(lora + link, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = None
        if step % 25 == 0 or step == steps:
            rec = dict(step=step, game=game, loss=round(loss.item(), 4), s=round(time.monotonic() - started))
            if torch.cuda.is_available():
                rec['gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            if net.arm == 'resid':
                rec['write'] = [round(torch.tanh(r.alpha).item(), 3) for r in net.readers]
        if net.arm == 'tok':
            net.tau = TAU[0] * (TAU[1] / TAU[0]) ** (step / steps)
            if rec:
                rec['tau'] = round(net.tau, 3)
                rec['beta'] = round(net.beta, 3)
                rec['codes'] = [len(set(q)) for q in net.codes]
                rec['ent'] = round(float(torch.as_tensor(net.ent_sum).detach()) / max(1, net.ent_n), 3)
                if net.adv_n:
                    rec['adv_w'] = round(net.adv, 3)
                    rec['adv'] = round(float(torch.as_tensor(net.adv_sum).detach()) / net.adv_n, 4)
        if every and step % every == 0:
            v = dev_nll(net, device, dev)
            record.append((step, v))
            rec = dict(rec or dict(step=step), dev_nll=round(v, 4))
            if net.arm == 'tok':
                soft, cut = dev_nll(net, device, dev, mode='soft'), dev_nll(net, device, dev, mode='off')
                rec.update(dev_soft=round(soft, 4), dev_cut=round(cut, 4))
                if work is None and v <= WORKING[0] * soft and v <= WORKING[1] * cut:
                    work = rec['working'] = step
            if v < best * 0.99:
                best, best_step = v, step
            hold = net.arm == 'tok' and net.adv_max and work and step < work + ADV_RAMP + 300
            if step >= min_steps and step - best_step >= patience and not hold:
                rec['stop'] = f'flat since step {best_step}'
        if rec:
            print(json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + '\n')
                log.flush()
            if 'stop' in rec:
                return step, record
    return steps, record


def report(root, boots=2000):
    """Paired comparisons over the arms' scores.json under root (the same test episodes in every arm): per game and
    crumb or text arm, its mean say NLL minus cut's, and its forced-choice accuracy minus cut's; per crumb arm, crumbs
    on minus off and on minus transplant (its own weights); each a mean of per-episode differences with a bootstrap
    95% interval. Writes root/report.json."""
    import numpy as np
    arms = {}
    for d in sorted(Path(root).iterdir()):
        if (d / 'scores.json').exists():
            arms[d.name] = json.loads((d / 'scores.json').read_text())
    rs = np.random.default_rng(0)

    def paired(x, y):
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if not pairs:
            return None
        d = np.array([a - b for a, b in pairs])
        means = d[rs.integers(0, len(d), (boots, len(d)))].mean(1)
        return dict(mean=round(float(d.mean()), 4), lo=round(float(np.percentile(means, 2.5)), 4),
                    hi=round(float(np.percentile(means, 97.5)), 4), n=len(d))

    out = dict(arms={}, vs_cut={}, within={})
    cut = arms.get('cut4')
    for name, sc in arms.items():
        out['arms'][name] = dict(steps_run=sc['steps_run'], probe=sc.get('probe'),
                                 tf={g: v['tf'] for g, v in sc['test'].items()},
                                 gen={g: {m: {k: round(x, 3) if isinstance(x, float) else x for k, x in s.items()}
                                          for m, s in v['gen'].items()} for g, v in sc['test'].items()})
        for g, v in sc['test'].items():
            ep = v['episodes']
            if cut and name != 'cut4' and g in cut['test']:
                ce = cut['test'][g]['episodes']['on']
                out['vs_cut'].setdefault(name, {})[g] = dict(nll=paired(ep['on']['nll'], ce['nll']),
                                                             choice=paired(ep['on']['choice'], ce['choice']))
            if 'off' in ep:
                out['within'].setdefault(name, {})[g] = {
                    f'on-{m}': dict(nll=paired(ep['on']['nll'], ep[m]['nll']),
                                    choice=paired(ep['on']['choice'], ep[m]['choice']))
                    for m in ('off', 'transplant', 'soft', 'base', 'base_off') if m in ep}
    (Path(root) / 'report.json').write_text(json.dumps(out, indent=1))
    return out


def main():
    global CHUNK
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', choices=('cut', 'resid', 'kv', 'text', 'tok'))
    ap.add_argument('--base', default=BASE, help='the base model (Hugging Face id)')
    ap.add_argument('--adv', type=float, default=0., help='tok arm: the adversary term\'s weight lambda')
    ap.add_argument('--n_intercept', type=int, default=2000, help='tok arm: trail episodes per split in messages.json')
    ap.add_argument('--m', type=int, default=4)
    ap.add_argument('--out', help='output folder (absolute)')
    ap.add_argument('--report', help='write ROOT/report.json from the arms under ROOT and stop')
    ap.add_argument('--steps', type=int, default=2400)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--micro', type=int, help='episodes per backward pass (default: the whole batch)')
    ap.add_argument('--min_steps', type=int, default=600, help='no early stop before this step')
    ap.add_argument('--weight', default='', help='game=w,...: how often a game is drawn (default 1 each)')
    ap.add_argument('--link_lr', type=float, default=1e-3, help='learning rate of the crumb, bridge and alphabet')
    ap.add_argument('--curriculum', default='', help='until:G_max,...: oldgrowth walkers up to G_max until a step')
    ap.add_argument('--init', help='a run folder whose trained.pt LoRA (and crumbs, tiled to m) to start from')
    ap.add_argument('--alpha0', type=float, default=0., help='resid arm: the readers\' starting write size tanh(alpha)')
    ap.add_argument('--n_test', type=int, default=300)
    ap.add_argument('--chunk', type=int, default=CHUNK, help='streams per teacher-forced forward at evaluation')
    ap.add_argument('--games', default=','.join(CG.GAMES))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='load OUT/trained.pt and only evaluate')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.report:
        print(json.dumps(report(args.report)['vs_cut'], indent=1))
        return
    CHUNK = args.chunk
    out = Path(args.out)
    if not out.is_absolute():
        raise SystemExit('--out must be absolute')
    out.mkdir(parents=True, exist_ok=True)
    games = tuple(args.games.split(','))
    net = build(args.device, args.arm, args.m, base=args.base)
    if args.arm == 'tok':
        net.adv = net.adv_max = args.adv
    curriculum = [tuple(int(x) for x in c.split(':')) for c in args.curriculum.split(',') if c]
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))
    weights = {g: float(w) for g, w in (kv.split('=') for kv in args.weight.split(',') if kv)}
    if args.init and not args.resume:
        w = torch.load(Path(args.init) / 'trained.pt', map_location='cpu')['weights']
        own = net.state_dict()
        keep = {k: v for k, v in w.items() if 'lora_' in k and k in own}
        if 'crumb' in w and 'crumb' in own:
            keep['crumb'] = w['crumb'].repeat(-(-args.m // len(w['crumb'])), 1)[:args.m]
        if not keep:
            raise SystemExit(f'no LoRA weights in {args.init}')
        net.load_state_dict(keep, strict=False)
        print(json.dumps(dict(init=args.init, tensors=len(keep))), flush=True)
    if args.arm == 'resid' and args.alpha0:
        for r in net.readers:
            r.alpha.data.fill_(math.atanh(args.alpha0))
    if 'oldgrowth' in games and args.arm in CRUMB_ARMS and not args.resume:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        check = evaluate_oldgrowth(net, args.device, sizes=(32,), n=25, n_gen=25)
        print(json.dumps(dict(memory_check_G32=check[32].get('gib'))), flush=True)
    if args.resume:
        saved = torch.load(out / 'trained.pt', map_location=args.device)
        net.load_state_dict(saved['weights'], strict=False)
        ran, record = saved['steps_run'], saved['dev_nll']
    else:
        ran, record = train(net, args.device, args.steps, args.batch, args.seed, games, out, micro=args.micro,
                            min_steps=args.min_steps, weights=weights, link_lr=args.link_lr, curriculum=curriculum)
        torch.save(dict(steps_run=ran, dev_nll=record, weights={k: v for k, v in net.state_dict().items()
                                                                if 'lora_' in k or not k.startswith('model.')}),
                   out / 'trained.pt')
    if args.arm == 'tok':
        torch.save(net.A.detach().float().cpu(), out / 'A.pt')
        net.model.save_pretrained(out / 'adapter')
        (out / 'messages.json').write_text(json.dumps(intercept(net, args.device, args.n_intercept)))
    test, rows = evaluate(net, args.device, args.n_test, n_choice=args.n_test // 2, games=games)
    scores = dict(arm=args.arm, m=args.m, base=args.base, adv=args.adv, steps=args.steps, steps_run=ran, batch=args.batch, micro=args.micro,
                  weight=weights, alpha0=args.alpha0, link_lr=args.link_lr, curriculum=curriculum, init=args.init,
                  games=games,
                  dev_nll=record, test=test)
    if args.arm in CRUMB_ARMS and 'trail' in games:
        scores['probe'] = probe(net, args.device)
    if 'oldgrowth' in games:
        scores['oldgrowth'] = evaluate_oldgrowth(net, args.device, log=lambda r: print(json.dumps(r), flush=True))
    (out / 'scores.json').write_text(json.dumps(scores, indent=1))
    (out / 'answers.json').write_text(json.dumps(rows))
    print(json.dumps({g: v['tf'] for g, v in test.items()}), flush=True)


if __name__ == '__main__':
    main()
