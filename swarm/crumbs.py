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
            own keys and values in each attention layer (crumb_attention) with a 4D mask, so that a token at place p
            attends to the crumbs at p besides its own past; no new parameters: the shared model reads them natively.
            Agent a's positions start at a * SPAN, so crumbs always lie in the reader's past at the same distances in
            training and evaluation. build(ckpt=True) recomputes each layer in the backward pass (gradient
            checkpointing): the crumbs go in as layer arguments, not a cache, so the recompute reads the same ones
  text      no crumbs; a reader's segment at place p starts with one line per earlier writer segment at p,
            "A note left here: {its user text} {its say}" (the format's ceiling)
  sym       crumbs are real vocabulary tokens: after each segment with a place the writer writes a note of m tokens
            with its own head, autoregressively, after a cue turn (CUE): each token fed back (and sent) is
            softmax(logits / tau) @ the embedding matrix in training (full gradient, no noise; tau annealed SYM_TAU)
            and the argmax token at evaluation (or the soft mixture in mode soft); special tokens never. A note sees
            the writer's tokens up to its place, not the writer's earlier notes. A reader's segment at place p starts
            with one line per earlier note at p, "A note left here:" and its m tokens as plain input embeddings.
            Besides the says, the loss adds lam times the softmax mass on letter tokens (letter_table: a character of
            Unicode category L*, or bytes that are not complete UTF-8), BETA (ramped over the first half) times each
            token's entropy minus the entropy of its batch-mean distribution, and --contrast times a contrastive
            term: each reader's says read again with the next episode's notes, softplus of its says' summed NLL with
            its own notes minus that with the other's (-log the share of p(says | own) + p(says | other) its own notes
            take). lam is 0 up to --letters_from, then 0.01, and after each dev times 1.5 while the dev notes' gap
            (cut minus hard NLL) holds 0.8 of its last value before the penalty, else times 0.5 (LETTERS, LAM_RULE).
            Privacy (--adv lambda), ramped in over ADV_RAMP steps once the dev notes are under PRIV_LETTERS letters
            with the gap held: the base model with
            its adapter disabled, and an eavesdropper (a second LoRA, 'eaves', trained in alternation on the hard notes
            to answer the reader's choice groups), each read the reader's prompt with the soft notes; per choice group
            of the reader's says (crumb_games.choices) the loss adds lambda * relu(log p(gold | the C candidates'
            first differing token) - max(log(1/C), the same with every note masked)) for each: neither may gain from
            the notes over chance or its own prior, and neither is rewarded for being confidently wrong
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
CRUMB_ARMS = ('resid', 'kv', 'sym')
SYM_TAU = (1.0, 0.3)          # sym arm: softmax temperature of the trained notes, start and end
CUE = 'Leave a short note here for whoever comes next.'   # sym: the writer's user turn before each note
NOTE = 'A note left here:'    # sym: what a reader's line starts with, before the note's tokens
LETTERS = (300, 0.01)         # sym: letter penalty lam 0 up to step 300 (--letters_from), then 0.01, and after each
LAM_RULE = (0.8, 1.5, 0.5)    # dev times 1.5 while the dev gap (cut minus hard NLL) holds 0.8 of its last value
                              # before the penalty, else times 0.5
PRIV_LETTERS = 0.05           # sym: privacy starts once the dev notes are under 5% letters with the gap held
CONTRAST = 1.0                # sym: the contrastive term's weight (own notes against the next episode's)
BETA = 0.1                    # sym: entropy penalty and batch-diversity bonus, ramped from 0 over the first half
ADV_RAMP = 300                # sym: steps over which the privacy terms ramp in once the notes are letterless
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
        if arm in ('resid', 'kv'):
            std = float(model.get_input_embeddings().weight.float().std())
            self.crumb = nn.Parameter(torch.randn(m, d) * std)
        # kv arm: the crumbs are the K/V of m slot tokens after each segment (slots), or of the writer's own user text
        # at that place (sentence; lastN: its last N tokens); rerope re-rotates them to sit just before the reader's
        # first token at the place (else they keep the writer's positions); remember lets each reader token see the
        # crumbs of every place it has been at (else only the place it is at)
        self.source, self.rerope, self.remember = 'slots', False, False
        if arm == 'sym':
            self.letter, self.banned = letter_table(tok, model.config.vocab_size)
            self.cue = self._ids(f'<|im_start|>user\n{CUE}{END}\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            self.note = self._ids(NOTE)
            self.tau, self.lam, self.beta, self.adv, self.adv_max = SYM_TAU[0], 0., 0., 0., 0.
            self.contrast = 0.
            self.soft, self.eaves_on, self.stash = False, False, []
            self.codes = [collections.deque(maxlen=256) for _ in range(m)]   # recent training argmax ids per position
            self.sums = collections.defaultdict(float)
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
        [(position, place)] (sym: m per note, at the position its note branches off, with no tokens of its own), asks
        [(segment, position where its say starts, place)] and, sym arm, reads [(position, writer agent, the writer's
        slot index)]: the note tokens it reads. A segment may give its say as token ids (say_ids) and a teacher's
        top-k for them (teacher: (ids [T, k], log-probs [T, k]) per say token, and one more row for the closing
        <|im_end|> when the teacher ended there): then tk holds, per token, its row in the stream's tk_ids and tk_lp
        (-1: none, cross-entropy)."""
        agent, remap = ep['agents'][a], remap or {}
        ids = self._ids(f'<|im_start|>system\n{agent["system"]}<|im_end|>\n')
        place, target, seg_of, kind = [-1] * len(ids), [False] * len(ids), [-1] * len(ids), [-1] * len(ids)
        slots, asks, reads, tk, tk_ids, tk_lp = [], [], [], [-1] * len(ids), [], []

        def add(tokens, p, t=False, k=-1, sg=-1, rows=None):
            ids.extend(tokens)
            place.extend([p] * len(tokens))
            target.extend([t] * len(tokens))
            seg_of.extend([sg] * len(tokens))
            kind.extend([k] * len(tokens))
            tk.extend(rows or [-1] * len(tokens))

        for s, seg in enumerate(agent['segments']):
            p = -1 if seg['place'] is None else seg['place']
            user = seg['user']
            if self.arm == 'text' and p >= 0:
                notes = [f'A note left here: {w["user"]}' + ('' if w['say'] is None else f' {w["say"]}')
                         for b in ep['agents'][:a] if b['write'] for w in b['segments'] if w['place'] == p]
                user = '\n'.join(notes + [user])
            rp = remap.get(p, p)
            if self.arm == 'sym' and rp >= 0:
                add(self._ids('<|im_start|>user\n'), rp)
                for w in range(a):
                    writer = ep['agents'][w]
                    k = 0
                    for ws in writer['segments'] if writer['write'] else ():
                        if ws['place'] is None:
                            continue
                        if ws['place'] == rp:
                            add(self.note, rp)
                            for j in range(self.m):
                                reads.append((len(ids), w, k * self.m + j))
                                add([self.tok.pad_token_id], rp)
                            add(self.nl, rp)
                        k += 1
                add(self._ids(f'{user}<|im_end|>\n'), rp)
            else:
                start = len(ids)
                add(self._ids(f'<|im_start|>user\n{user}<|im_end|>\n'), rp)
                head = len(self._ids('<|im_start|>user\n'))
                span = range(start + head, start + len(self._ids(f'<|im_start|>user\n{user}')))
            if seg['say'] is not None:
                add(self.head, rp)
                asks.append((s, len(ids), rp))
                say = list(seg.get('say_ids') or self._ids(seg['say']))
                rows = None
                if seg.get('teacher') is not None:
                    n0 = sum(len(t) for t in tk_ids)
                    tk_ids.append(seg['teacher'][0])
                    tk_lp.append(seg['teacher'][1])
                    end = n0 + len(say) if len(seg['teacher'][0]) > len(say) else -1
                    rows = [n0 + i for i in range(len(say))] + [end]
                add(say + [self.end], rp, True, sg=s, rows=rows)
                add(self.nl, rp)
            if agent['write'] and p >= 0 and self.arm == 'kv' and self.source != 'slots':
                take = list(span) if self.source == 'sentence' else list(span)[-int(self.source[4:]):]
                slots.extend((t, p) for t in take)
            elif agent['write'] and p >= 0 and self.arm in CRUMB_ARMS:
                for j in range(self.m):
                    slots.append((len(ids), p))
                    if self.arm != 'sym':
                        add([self.tok.pad_token_id], rp, k=j)
        if len(ids) > SPAN:
            raise ValueError(f'agent stream of {len(ids)} tokens is longer than SPAN')
        out = dict(ids=ids, place=place, target=target, seg=seg_of, kind=kind, slots=slots, asks=asks, reads=reads,
                   tk=tk)
        if tk_ids:
            out['tk_ids'], out['tk_lp'] = torch.cat(tk_ids), torch.cat(tk_lp)
        return out

    def _pad(self, rows, device, left=False):
        """Tensors ids, att, place, target, kind [N, T] from per-row lists, padded on the right (or left)."""
        T = max(len(r['ids']) for r in rows)
        fill = dict(ids=self.tok.pad_token_id, place=-1, target=False, seg=-1, kind=-1, tk=-1)
        out = {k: torch.full((len(rows), T), v, dtype=torch.bool if k == 'target' else torch.long)
               for k, v in fill.items()}
        out['att'] = torch.zeros((len(rows), T), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r['ids'])
            sl = slice(T - n, T) if left else slice(0, n)
            for k in fill:
                out[k][i, sl] = torch.tensor(r.get(k, [-1] * n), dtype=out[k].dtype)
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
        elif self.arm == 'sym':
            bank['e'] = bank['ids'] = None
        else:
            bank['k'] = bank['v'] = bank['pos'] = None
        return bank

    def _read_tokens(self, rows, x, emb, att, bank, on):
        """sym arm: emb with each read position's input the note token's embedding it reads (bank e), and att with
        the read positions masked out where on is False or the bank has no such slot (a transplant donor without it)."""
        rpos, ra, ri = x['rpos'], x['ragent'], x['ridx']
        N, T = att.shape
        R, S = rpos.shape[1], bank['place'].shape[1]
        valid = rpos >= 0
        ok = torch.zeros_like(valid)
        if S and on and bank.get('e') is not None:
            first = {}
            for c, w in enumerate(bank['agent'][0].tolist()):
                first.setdefault(w, c)
            look = torch.full((max(int(ra.max()), max(first)) + 1,), -1, dtype=torch.long)
            for w, c in first.items():
                look[w] = c
            start = look.to(rpos.device)[ra.clamp(min=0)]
            col = (start + ri).clamp(0, S - 1)
            ok = valid & (start >= 0) & (start + ri < S) & (bank['place'][rows].gather(1, col) >= 0)
            msg = bank['e'][0][rows].gather(1, col[..., None].expand(-1, -1, emb.shape[-1])).to(emb.dtype)
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
        [N, T, d]; kv, per layer (k, v) [N, heads, S + T, head dim] (S: the bank's slots); sym, per layer (k, v)
        [N, heads, T, head dim], the writer's past its notes branch off."""
        ids, att = x['ids'], x['att']
        emb = self.model.get_input_embeddings()(ids)
        if self.arm in ('resid', 'kv'):
            emb = torch.where((x['kind'] >= 0)[..., None], self.crumb.to(emb.dtype)[x['kind'].clamp(min=0)], emb)
        if self.arm == 'sym' and x.get('rpos') is not None and x['rpos'].shape[1]:
            emb, att = self._read_tokens(rows, x, emb, att, bank, on)
        pos = (att.cumsum(-1) - 1).clamp(min=0) + (a * SPAN if self.arm == 'kv' else 0)
        S = bank['place'].shape[1] if self.arm in ('resid', 'kv') else 0
        allow = None
        if S:
            bp, ba = bank['place'][rows], bank['agent'][rows]
            allow = ((bp[:, None, :] == x['place'][:, :, None]) & (x['place'][:, :, None] >= 0)
                     & (ba[:, None, :] < a) & on)
        if self.arm == 'kv':
            T = ids.shape[1]
            own = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))[None] & att.bool()[:, None, :]
            own = own | torch.eye(T, dtype=torch.bool, device=ids.device)[None]   # padding rows see themselves
            if S and self.remember:
                P = int(max(int(bp.max()), int(x['place'].max()))) + 1
                seen = (x['place'][..., None] == torch.arange(P, device=ids.device)).long().cummax(1).values.bool()
                allow = (seen.gather(2, bp.clamp(min=0)[:, None, :].expand(-1, T, -1)) & (bp >= 0)[:, None, :]
                         & (ba[:, None, :] < a) & on)
            mask = (torch.cat([allow, own], -1) if S else own)[:, None]
            ks = [k[rows] for k in bank['k']] if S else None
            if S and self.rerope:
                delta = self._reroped(x, pos, bp, bank['pos'][0][rows])
                ks = [self._rotate(k, delta) for k in ks]
            got = {} if capture else None
            out = self.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=False,
                             logits_to_keep=keep, crumb_k=ks, crumb_v=[v[rows] for v in bank['v']] if S else None,
                             crumb_got=got)
            return out.logits, ([got[i] for i in range(len(got))] if capture else None)
        if self.arm == 'resid':
            self.ctx = dict(src=[h[rows] for h in bank['h']] if S else None, allow=allow, got={} if capture else None)
        try:
            out = self.model(inputs_embeds=emb, attention_mask=att, position_ids=pos,
                             use_cache=capture and self.arm == 'sym', logits_to_keep=keep)
            logits = out.logits
            if self.arm == 'sym':
                got = [(k, v) for k, v, *_ in out.past_key_values] if capture else None
            else:
                got = [self.ctx['got'][i] for i in range(len(self.layers))] if capture and self.ctx else None
        finally:
            self.ctx = None
        return logits, got

    def _append(self, bank, rows, streams, got, a, x=None):
        """bank with agent a's crumbs (rows' slots, taken from got; sym: the notes written after x) appended."""
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
        elif self.arm == 'sym':
            e, ids = self._notes(x, got, streams)
            out['e'] = [put(e)] if bank['e'] is None else [torch.cat([bank['e'][0], put(e)], 1)]
            out['ids'] = [put(ids)] if bank['ids'] is None else [torch.cat([bank['ids'][0], put(ids)], 1)]
        else:
            ks, vs = [], []
            for k, v in got:
                g = (S0 + idx)[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[-1])
                ks.append(k.gather(2, g))
                vs.append(v.gather(2, g))
            ks, vs = [put(k) for k in ks], [put(v) for v in vs]
            out['k'] = ks if bank['k'] is None else [torch.cat([o, x], 2) for o, x in zip(bank['k'], ks)]
            out['v'] = vs if bank['v'] is None else [torch.cat([o, x], 2) for o, x in zip(bank['v'], vs)]
            wpos = put(((x['att'].cumsum(-1) - 1).clamp(min=0) + a * SPAN).gather(1, idx))
            out['pos'] = [wpos] if bank.get('pos') is None else [torch.cat([bank['pos'][0], wpos], 1)]
        return out

    def _reroped(self, x, pos, bp, wpos):
        """Per bank slot the position shift that puts the crumbs at a place just before the reader's first token
        there, in their order (0 where the reader never is)."""
        delta = torch.zeros_like(wpos)
        for p in range(int(bp.max()) + 1):
            at = x['place'] == p
            first = torch.where(at, pos, torch.full_like(pos, 1 << 30)).min(1).values
            col = bp == p
            target = first[:, None] - col.long().sum(1, keepdim=True) + col.long().cumsum(1) - 1
            delta = torch.where(col & at.any(1, keepdim=True), target - wpos, delta)
        return delta

    def _rotate(self, k, delta):
        """Keys k [N, heads, S, head dim] (already rotated at their positions) rotated on by delta [N, S]."""
        inv = self.model.get_base_model().model.rotary_emb.inv_freq.float()
        ang = delta.float()[..., None] * inv
        cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None], torch.cat([ang.sin()] * 2, -1)[:, None]
        kf = k.float()
        h = kf.shape[-1] // 2
        return (kf * cos + torch.cat([-kf[..., h:], kf[..., :h]], -1) * sin).to(k.dtype)

    def _notes(self, x, got, streams):
        """sym arm: every note of the writer streams x (from _pad; got: their per-layer (k, v), the past each note
        branches off), m passes, each over every note at once: the note's tokens see the writer's tokens before its
        place, then the cue turn (CUE) and the note's tokens so far (inputs_embeds, 4D mask), and the last one's
        final state through the model's own head gives the next token's logits (special tokens masked). The token
        fed back and sent is softmax(logits / tau) @ the embedding matrix in training (and mode soft), else the argmax
        token's row. Returns the note embeddings [N, n, d] and argmax ids [N, n] in slot order (note k, token j at
        k * m + j)."""
        from transformers import DynamicCache
        device, att = x['ids'].device, x['att'].bool()
        N, T = att.shape
        m = self.m
        K = [len(st['slots']) // m for st in streams]
        Kn = max(K)
        pe = torch.tensor([[st['slots'][k * m][0] for k in range(c)] + [0] * (Kn - c) for st, c in zip(streams, K)],
                          device=device)
        valid = torch.arange(Kn, device=device)[None] < torch.tensor(K, device=device)[:, None]
        start = torch.cat([torch.zeros_like(pe[:, :1]), att.long().cumsum(-1)], 1).gather(1, pe)  # tokens before
        E = self.model.get_input_embeddings().weight
        cue = E[torch.tensor(self.cue, device=device)]
        train = torch.is_grad_enabled() and self.training
        banned = self.banned.to(device)
        embs, ids = [], []
        for j in range(m):
            L = len(self.cue) + j
            new = cue[None, None].expand(N, Kn, -1, -1)
            if embs:
                new = torch.cat([new, torch.stack(embs, 2)], 2)
            new = new.reshape(N, Kn * L, -1)
            pos = (start[:, :, None] + torch.arange(L, device=device)).reshape(N, Kn * L)
            past = (torch.arange(T, device=device)[None, None] < pe[:, :, None]) & att[:, None] & valid[..., None]
            past = past[:, :, None].expand(-1, -1, L, -1).reshape(N, Kn * L, T)
            blk, off = torch.arange(Kn * L, device=device) // L, torch.arange(Kn * L, device=device) % L
            own = (blk[:, None] == blk[None]) & (off[:, None] >= off[None])
            mask = torch.cat([past, own[None].expand(N, -1, -1)], -1)[:, None]
            cache = DynamicCache(ddp_cache_data=[(k, v) for k, v in got])
            self.grab = True
            try:
                self.model(inputs_embeds=new, attention_mask=mask, position_ids=pos, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
                h = self.last.view(N, Kn, L, -1)[:, :, -1]
            finally:
                self.grab, self.last = False, None
            logits = self.model.get_output_embeddings()(h).float().masked_fill(banned, float('-inf'))
            hard = logits.argmax(-1)
            if train or self.soft:
                p = (logits / self.tau).softmax(-1)
                e = p.to(E.dtype) @ E
                if train:
                    self._track(p, hard, valid, j)
            else:
                e = E[hard]
            embs.append(e.to(cue.dtype))
            ids.append(hard)
        return torch.stack(embs, 2).reshape(N, Kn * m, -1), torch.stack(ids, 2).reshape(N, Kn * m)

    def _track(self, p, hard, valid, j):
        """Training, note token j: the sums of the letter mass, the entropy and the batch-mean distribution's entropy
        (loss terms), the argmax ids' letters, and the argmax ids per position."""
        pv = p[valid]
        if not len(pv):
            return
        letter = self.letter.to(p.device)
        mean = pv.mean(0)
        s = self.sums
        s['let'] = s['let'] + pv[:, letter].sum()
        s['ent'] = s['ent'] + -(pv * pv.clamp_min(1e-12).log()).sum()
        s['div'] = s['div'] + -(mean * mean.clamp_min(1e-12).log()).sum()
        s['hard_let'] += float(letter[hard[valid]].sum())
        s['n_tok'] += len(pv)
        s['n_div'] += 1
        self.codes[j].extend(hard[valid].tolist())

    def eaves_params(self):
        return [p for n, p in self.model.named_parameters() if '.eaves.' in n]

    @contextlib.contextmanager
    def _eaves(self, train=False):
        """The eavesdropper adapter in place of the shared one; its weights frozen unless train (backward inside)."""
        self.model.set_adapter('eaves')
        if not train:
            for q in self.eaves_params():
                q.requires_grad_(False)
        try:
            yield
        finally:
            self.model.set_adapter('default')

    def eaves_step(self, scale=1.):
        """The eavesdropper's update on the stashed reader prompts (the hard notes): backward of scale times the mean
        -log p(gold) among each choice group's candidates. Empties the stash and returns the loss."""
        items, self.stash = self.stash, []
        total, n = 0., 0
        with self._eaves(train=True):
            for rows_t, x, a, bank, targets in items:
                lp, _ = self._gold_logp(rows_t, x, a, bank, targets, True)
                total, n = total - lp.sum(), n + len(lp)
            if n:
                (scale * total / n).backward()
        return float(total) / max(1, n)

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
            self.grab = sel is not None and len(sel) > 0
            try:
                _, got = self._forward(rows_t, x, a, bank, 1, writes)
                h = self.last
            finally:
                self.grab, self.last = False, None
            if sel is not None and len(sel):
                logp = self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float().log_softmax(-1)
                ce = -logp.gather(1, x['ids'][sel[:, 0], sel[:, 1] + 1][:, None])[:, 0]
                ce = self._with_kl(ce, logp, streams, sel, x)
                total = total + ce.sum()
                count += len(sel)
                if self.arm == 'sym' and self.adv and torch.is_grad_enabled():
                    self._adversary(rows_t, x, a, bank, [(eps[b], st) for b, st in zip(rows, streams)])
                if self.arm == 'sym' and self.contrast and torch.is_grad_enabled() and len(eps) > 1:
                    self._contrast(rows_t, x, a, bank, sel, ce)
            if writes:
                bank = self._append(bank, rows_t, streams, got, a, x)
        return total, count, bank

    def loss(self, eps, device):
        """The says' mean cross-entropy; sym also lam * letter mass + beta * (entropy - batch-mean entropy) per note
        token, contrast * the contrastive term per reader row and adv * the two privacy terms' mean per choice group
        (sums: the terms, kept for the log)."""
        if self.arm == 'sym':
            self.sums = collections.defaultdict(float)
        total, count, _ = self.run(eps, device)
        out = total / max(1, count)
        if self.arm == 'sym':
            s = self.sums
            if s['n_tok']:
                out = out + self.lam * s['let'] / s['n_tok'] + self.beta * (s['ent'] / s['n_tok'] - s['div'] / s['n_div'])
            for k in ('adv', 'eav'):
                if s[f'n_{k}']:
                    out = out + self.adv * s[k] / s[f'n_{k}']
            if s['n_con']:
                out = out + self.contrast * s['con'] / s['n_con']
        return out

    def _contrast(self, rows_t, x, a, bank, sel, own):
        """sym, training: the contrastive term of the rows that have notes to read. Their says read again with the
        next episode's notes (shifted); per row softplus(own - other), each its says' summed NLL (own: per say token,
        the pass with its own notes): -log the share of p(says | own) + p(says | other) its own notes take."""
        has = (bank['place'][rows_t] >= 0).any(-1)
        if not bool(has.any()):
            return
        self.grab = True
        try:
            self._forward(rows_t, x, a, shifted(bank), 1, False)
            h = self.last
        finally:
            self.grab, self.last = False, None
        gold = x['ids'][sel[:, 0], sel[:, 1] + 1]
        other = F.cross_entropy(self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float(), gold,
                                reduction='none')
        d = torch.zeros(len(rows_t), device=own.device).index_add(0, sel[:, 0], other - own)
        keep = has & (torch.zeros_like(d).index_add(0, sel[:, 0], torch.ones_like(own)) > 0)
        self.sums['con'] = self.sums['con'] + F.softplus(-d[keep]).sum()
        self.sums['con_gap'] += float(d[keep].detach().sum())
        self.sums['n_con'] += int(keep.sum())

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

    def _gold_logp(self, rows_t, x, a, bank, targets, on):
        """log p(gold) among each target's candidates (adv_targets: their first differing token) as the model with its
        current adapters reads x, and the candidates' count."""
        self.grab = True
        try:
            self._forward(rows_t, x, a, bank, 1, False, on)
            h = self.last
        finally:
            self.grab, self.last = False, None
        dev = h.device
        at = (torch.tensor([i for i, _ in targets], device=dev), torch.tensor([t[0] for _, t in targets], device=dev))
        C = max(1 + len(t[2]) for _, t in targets)
        cand = torch.tensor([[t[1]] + t[2] + [-1] * (C - 1 - len(t[2])) for _, t in targets], device=dev)
        W = self.model.get_output_embeddings().weight[cand.clamp(min=0)].float()
        lg = torch.einsum('gd,gcd->gc', h[at].float(), W).masked_fill(cand < 0, float('-inf'))
        return lg.log_softmax(-1)[:, 0], (cand >= 0).sum(-1).float()

    def _adversary(self, rows_t, x, a, bank, items):
        """sym privacy terms: the base model with its adapter disabled (adv) and, once eaves_on, the eavesdropper adapter
        (eav, its weights frozen here) read the reader's prompts with the same soft notes, so their gradient reaches
        the writers; per choice group relu(log p(gold) - max(log(1/C), log p(gold) with the notes masked)) over the C
        candidates' first differing token: only what a reader gains from the notes beyond chance and its own prior
        counts. Once eaves_on, the prompts with the hard notes go to the stash for the eavesdropper's own update."""
        targets = [(i, t) for i, (ep, st) in enumerate(items) if st['reads'] for t in self.adv_targets(ep, a, st)]
        if not targets:
            return
        readers = [('adv', self.model.disable_adapter)] + ([('eav', self._eaves)] if self.eaves_on else [])
        for key, ctx in readers:
            with ctx():
                lp, n = self._gold_logp(rows_t, x, a, bank, targets, True)
                with torch.no_grad():
                    lp0, _ = self._gold_logp(rows_t, x, a, bank, targets, False)
            self.sums[key] = self.sums[key] + F.relu(lp - torch.maximum(lp0, -n.log())).sum()
            self.sums[f'n_{key}'] += len(targets)
        if self.eaves_on:
            E = self.model.get_input_embeddings().weight
            hard = dict(bank, e=[E[bank['ids'][0]].detach()])
            self.stash.append((rows_t, dict(x), a, hard, targets))

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
        order). mode: on, off (every crumb masked), transplant (each episode reads the next episode's crumbs), soft
        (sym arm: the soft notes at the current tau), base (sym: the trained writers' notes read by the base model
        with its adapter disabled) or base_off (the base model with every note masked: its own prior); bank: these
        crumbs instead of the writers' own."""
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

    def _with_kl(self, ce, logp, streams, sel, x):
        """ce with each say token that has a teacher top-k (stream tk) replaced by KL(teacher || student) there (logp:
        the student's log-probabilities at sel)."""
        tk = x['tk'][sel[:, 0], sel[:, 1] + 1]
        has = tk >= 0
        if not bool(has.any()):
            return ce
        n = [len(st.get('tk_ids', ())) for st in streams]
        off = torch.tensor([sum(n[:i]) for i in range(len(n))], device=ce.device)
        big_ids = torch.cat([st['tk_ids'] for st in streams if 'tk_ids' in st]).to(ce.device)
        big_lp = torch.cat([st['tk_lp'] for st in streams if 'tk_lp' in st]).to(ce.device).float()
        g = torch.where(has, off[sel[:, 0]] + tk.clamp(min=0), 0)        # row 0 stands in where there is no teacher
        return torch.where(has, kl_topk(logp, big_ids[g], big_lp[g], normalized=True), ce)

    def _say_kl(self, rows, streams, a, bank, on, device):
        """Per stream {segment: (summed KL(teacher top-k || student) over its say tokens with a teacher, their count)}."""
        x = self._pad(streams, device)
        sel = x['target'][:, 1:].nonzero()
        self.grab = True
        try:
            self._forward(rows, x, a, bank, 1, False, on)
            h = self.last
        finally:
            self.grab, self.last = False, None
        tk = x['tk'][sel[:, 0], sel[:, 1] + 1]
        keep = tk >= 0
        sel, tk = sel[keep], tk[keep]
        out = [{} for _ in streams]
        if not len(sel):
            return out
        logp = self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float().log_softmax(-1)
        kl = self._with_kl(torch.zeros(len(sel), device=logp.device), logp, streams, sel, x)
        seg = x['seg'][sel[:, 0], sel[:, 1] + 1]
        for i, sg, v in zip(sel[:, 0].tolist(), seg.tolist(), kl.tolist()):
            t, c = out[i].get(sg, (0., 0))
            out[i][sg] = (t + v, c + 1)
        return out

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


def kl_topk(logits, ids, lp, normalized=False):
    """Per row KL(teacher || student) on the partition {each of the teacher's top-k tokens, every other token}: ids
    and lp [N, k] (the teacher's log-probs, not renormalized; swarm/opendata.kl_topk): 0 when the student matches the
    teacher, and student mass outside the top-k costs. A one-hot teacher (k 1, lp 0) gives the cross-entropy.
    normalized: logits are already log-probabilities."""
    t = lp.float()
    s = (logits if normalized else logits.float().log_softmax(-1)).gather(1, ids)
    t_rest = (1 - t.exp().sum(-1)).clamp_min(1e-12)
    s_rest = (1 - s.exp().sum(-1)).clamp_min(1e-12)
    return (t.exp() * (t - s)).sum(-1) + t_rest * (t_rest.log() - s_rest.log())


def shifted(bank):
    """bank with each episode's crumbs replaced by the next episode's (the last episode's by the first's)."""
    B = bank['place'].shape[0]
    idx = (torch.arange(B, device=bank['place'].device) + 1) % B
    return {k: None if v is None else [t[idx] for t in v] if isinstance(v, list) else v[idx] for k, v in bank.items()}


def transplant(bank, donor):
    """A copy of bank where episode b's agent-0 crumbs at place pa are those of episode j's agent-0 crumbs at pc, for
    each donor[b] = (pa, j, pc)."""
    out = dict(bank)
    pairs = {b: (((bank['place'][b] == pa) & (bank['agent'][b] == 0)).nonzero()[:, 0],
                 ((bank['place'][j] == pc) & (bank['agent'][j] == 0)).nonzero()[:, 0])
             for b, (pa, j, pc) in donor.items()}
    if any(len(d) != len(s) for d, s in pairs.values()):
        # crumbs of different lengths (kv arm, the writer's own text): hide b's crumbs at pa and append j's as new slots
        S, n = bank['place'].shape[1], max(len(s) for _, s in pairs.values())
        B = bank['place'].shape[0]
        out['place'] = torch.cat([bank['place'], bank['place'].new_full((B, n), -1)], 1)
        out['agent'] = torch.cat([bank['agent'], bank['agent'].new_zeros((B, n))], 1)
        for b, (pa, j, pc) in donor.items():
            d, s = pairs[b]
            out['place'][b, d] = -1
            out['place'][b, S:S + len(s)] = pa
        for name, dim in (('h', 1), ('k', 2), ('v', 2), ('pos', 1)):
            if bank.get(name) is None:
                continue
            rows = []
            for old in bank[name]:
                shape = list(old.shape)
                shape[dim] = n
                t = torch.cat([old, old.new_zeros(shape)], dim)
                for b, (pa, j, pc) in donor.items():
                    s = pairs[b][1]
                    if dim == 1:
                        t[b, S:S + len(s)] = old[j, s]
                    else:
                        t[b, :, S:S + len(s)] = old[j, :, s]
                rows.append(t)
            out[name] = rows
        return out
    for name, dim in (('h', 1), ('k', 2), ('v', 2), ('pos', 1), ('e', 1), ('ids', 1)):
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
        (['base'] if net.arm == 'sym' else [])
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


def expand_games(spec, weights):
    """The games of a comma list and their training weights: "corpus" becomes every corpus task whose rows are cached
    (c_TASK), weighted by corpus.mix_of() times their number (so 1 on average beside the other games' default 1)."""
    games, weights = [], dict(weights)
    for g in spec.split(','):
        if g == 'corpus':
            from swarm import corpus as CP
            mix = CP.mix_of()
            for t in sorted(mix):
                games.append(f'c_{t}')
                weights.setdefault(f'c_{t}', mix[t] * len(mix))
        elif g:
            games.append(g)
    return tuple(games), weights


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
            (['soft', 'base', 'base_off'] if net.arm == 'sym' else [])
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
            if 'e' in bank:
                layers = {'tokens': bank['e'][0]}
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
    """sym arm: every writer's hard notes on n training-value and n test-value trail episodes and on others of each
    other game, with what the writer saw: dict(m, messages=[dict(split, game, episode, writer, segment, place, room,
    vocab_ids (the note's m Qwen token ids), text (their decoded text), letters (how many are letter tokens), note
    (the text arm's English note for the same segment) and, on trail, object, key_here, key_room, key_before (the
    writer passed the key's room earlier), rank (the writer's k-th room), follower_visits)], trail_episodes={split:
    [episode]}). Episodes of game g, split s: random.Random(f'intercept-{g}-{s}'), crumb_games.episode(g, rng, s ==
    'test') in turn."""
    net.eval()
    out = dict(m=net.m, messages=[], trail_episodes={})
    letter = net.letter.tolist()
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
                if bank.get('ids') is None:
                    continue
                toks, place, agent = bank['ids'][0].tolist(), bank['place'].tolist(), bank['agent'].tolist()
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
                                       room=ep['places'][r], vocab_ids=ids, text=net.tok.decode(ids),
                                       letters=sum(letter[i] for i in ids),
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


@torch.no_grad()
def note_stats(net, device, eps, show=2, bs=50):
    """sym arm: the hard notes on eps: the share of their tokens that are letters, and the first show trail episodes'
    explorer notes as 'object: text'."""
    net.eval()
    n_tok, n_let, examples, shown = 0, 0, [], 0
    letter = net.letter.to(device)
    for s0 in range(0, len(eps), bs):
        chunk = eps[s0:s0 + bs]
        _, _, bank = net.run(chunk, device, need_loss=False)
        if bank.get('ids') is None:
            continue
        ids, real = bank['ids'][0], bank['place'] >= 0
        n_tok, n_let = n_tok + int(real.sum()), n_let + int(letter[ids][real].sum())
        for b, ep in enumerate(chunk):
            if 'trail' not in ep or shown >= show:
                continue
            shown += 1
            cols = ((bank['place'][b] >= 0) & (bank['agent'][b] == 0)).nonzero()[:, 0].tolist()
            for k, r in enumerate(ep['trail']['order']):
                note = net.tok.decode(ids[b, cols[k * net.m:(k + 1) * net.m]].tolist())
                examples.append(f'{ep["trail"]["objects"][r]}: {note}')
    net.train()
    return dict(letters=n_let / max(1, n_tok), examples=examples)


def letter_table(tok, V, _cache={}):
    """[V] bool tensors (letter, banned). A token is a letter if its text has a character of Unicode category L*, or
    its bytes are not complete UTF-8 (byte-level BPE: tokens are strings over the GPT-2 byte alphabet); banned: the
    special and added tokens and the rows past the tokenizer's."""
    key = (tok.name_or_path, V)
    if key not in _cache:
        import unicodedata
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + \
            list(range(ord('\xae'), ord('\xff') + 1))
        cs, extra = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + extra)
                extra += 1
        byte = {chr(c): b for b, c in zip(bs, cs)}
        letter, banned = torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.bool)
        banned[len(tok):] = True
        for i in list(tok.added_tokens_decoder) + list(tok.all_special_ids):
            if i < V:
                banned[i] = True
        for i, t in enumerate(tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))):
            if banned[i]:
                continue
            try:
                text = bytes(byte[c] for c in t).decode('utf-8')
            except (KeyError, UnicodeDecodeError):
                letter[i] = True
                continue
            letter[i] = any(unicodedata.category(c).startswith('L') for c in text)
        _cache[key] = (letter, banned)
    return _cache[key]


def load(out, device):
    """The trained net of a run folder (its args.json and trained.pt)."""
    out = Path(out)
    args = json.loads((out / 'args.json').read_text())
    net = build(device, args['arm'], args['m'], base=args.get('base', BASE))
    net.load_state_dict(torch.load(out / 'trained.pt', map_location=device)['weights'], strict=False)
    if args['arm'] == 'sym':
        net.adv = net.adv_max = args.get('adv', 0.)
        net.tau = SYM_TAU[1]
    return net


def crumb_attention(module, query, key, value, attention_mask, crumb_k=None, crumb_v=None, crumb_got=None, **kw):
    """kv arm: sdpa attention with the bank's crumb keys and values (crumb_k, crumb_v: per layer [N, kv heads, S, head
    dim]) in front of the stream's own; crumb_got collects each layer's keys and values, crumbs first (the first
    pass's: a gradient-checkpointing recompute leaves them)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    i = module.layer_idx
    if crumb_k is not None:
        key, value = torch.cat([crumb_k[i].to(key.dtype), key], 2), torch.cat([crumb_v[i].to(value.dtype), value], 2)
    if crumb_got is not None:
        crumb_got.setdefault(i, (key, value))
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


def use_crumb_attention(model):
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
    AttentionInterface.register('crumbs_sdpa', crumb_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register('crumbs_sdpa', sdpa_mask)
    model.config._attn_implementation = 'crumbs_sdpa'


def build(device, arm, m, base=BASE, rank=16, model=None, tok=None, layers=LAYERS, ckpt=False):
    """The model with its LoRA (sym: and the eavesdropper's) and the arm's crumb modules; ckpt (kv arm): gradient
    checkpointing, every layer recomputed in the backward pass."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = model or AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True,
                                                          attn_implementation='sdpa')
    if arm == 'kv':
        use_crumb_attention(model)
    if ckpt:
        if arm != 'kv':
            raise ValueError('gradient checkpointing is for the kv arm (resid hooks and sym caches would replay)')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    def config():
        return LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model.to(device), config())
    if arm == 'sym':                                     # the eavesdropper (privacy term), trained in alternation
        model.add_adapter('eaves', config())
        model.set_adapter('default')
    return Crumbs(model, tok, arm, m, layers).to(device)


def train(net, device, steps, batch, seed, games=None, out=None, lr=2e-4, link_lr=1e-3, every=100, patience=300,
          min_steps=600, micro=None, weights=None, curriculum=None, letters_from=LETTERS[0], lam0=LETTERS[1],
          gap0=None, lam_rule=LAM_RULE[1:], lost=0, private_steps=0, tau0=SYM_TAU[0], beta_full=False, dev=None,
          dev_fn=None):
    """steps of batch episodes of one game each (similar lengths, little padding), the game drawn at random for each
    step, in backward passes of micro episodes (the mean of their losses). Every every steps, the mean say NLL of 8
    fixed training-value episodes per game; training stops once that has not improved by 1% for patience steps
    (after min_steps). curriculum [(until_step, G_max)]: oldgrowth episodes draw G from 1 to G_max up to until_step
    (afterwards the game's own 1 to 4). sym: tau annealed SYM_TAU, beta ramped to BETA over the first half; each dev
    evaluation also takes the soft notes, the cut, the transplant, and the hard notes' letter share with example
    notes; lam 0 up to letters_from (OUT/pre.pt: the weights at its last dev), then LETTERS[1], and after each dev
    times LAM_RULE[1] while the gap (cut minus hard dev NLL) is at least LAM_RULE[0] of its value at the last dev
    before the penalty, else times LAM_RULE[2]; once the letter share is under PRIV_LETTERS with the gap held
    (OUT/gate.pt: the weights then) the privacy terms ramp in. A continuation (sym, from a run's trained.pt) passes
    letters_from 0, the weight to start from (lam0), the reference gap (gap0: the first run's last dev before the
    penalty), lam_rule (times up on a held dev, down otherwise; once private, no rise while the notes stay under
    PRIV_LETTERS), tau0 (the annealed end) and beta_full; lost stops training after that many devs in a row without
    the gap held, private_steps that many steps after privacy starts. Then
    over ADV_RAMP steps, the eavesdropper trains on every step after each backward pass (its own AdamW at lr), and
    training runs at least ADV_RAMP + 300 steps past it. The dev NLL that stops training is the hard one. Returns
    the steps run and the NLL record. dev: these dev episodes instead of 8 per game; dev_fn(net, dev) -> (the dev
    value, more numbers to log) instead of the mean say NLL."""
    games = games or CG.GAMES
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev_rng = random.Random(777)
    dev = dev if dev is not None else [CG.episode(g, dev_rng) for g in games for _ in range(8)]
    if net.arm == 'resid':
        cal = random.Random(99)
        net.calibrate([CG.episode(cal.choice(games), cal) for _ in range(8)], device)
    sym = net.arm == 'sym'
    lora = [p for p in net.model.parameters() if p.requires_grad]
    link = net.link_params()
    opt = torch.optim.AdamW([dict(params=lora, lr=lr)] + ([dict(params=link, lr=link_lr)] if link else []),
                            weight_decay=0.0)
    opt_e = torch.optim.AdamW(net.eaves_params(), lr=lr, weight_decay=0.0) if sym else None
    warm = min(50, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1, s / steps))))
    log = open(out / 'log.jsonl', 'a') if out else None
    started = time.monotonic()
    best, best_step, record, work, letters, misses = float('inf'), 0, [], None, None, 0
    for step in range(1, steps + 1):
        game = rng.choices(games, [(weights or {}).get(g, 1.) for g in games])[0]
        gmax = next((g for until, g in curriculum or () if step <= until), None) if game == 'oldgrowth' else None
        eps = [CG.episode(game, rng, **(dict(G=rng.randint(1, gmax)) if gmax else {})) for _ in range(batch)]
        if sym:
            t1 = SYM_TAU[1]
            net.tau = tau0 * (t1 / tau0) ** (step / steps)
            net.beta = BETA if beta_full else BETA * min(1., step / max(1, steps / 2))
            if step == letters_from + 1:
                net.lam = lam0
            net.adv = net.adv_max * min(1., (step - work) / ADV_RAMP) if work else 0.
            net.eaves_on = bool(work and net.adv_max)
        parts = [eps[i:i + (micro or batch)] for i in range(0, batch, micro or batch)]
        loss, stat, eloss = 0., collections.Counter(), []
        for part in parts:
            part_loss = net.loss(part, device) / len(parts)
            part_loss.backward()
            loss += part_loss.detach()
            if sym:
                stat.update({k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in net.sums.items()})
                if net.stash:
                    eloss.append(net.eaves_step(1 / len(parts)))
        torch.nn.utils.clip_grad_norm_(lora + link, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if eloss:
            torch.nn.utils.clip_grad_norm_(net.eaves_params(), 1.0)
            opt_e.step()
            opt_e.zero_grad(set_to_none=True)
        rec = None
        if step % 25 == 0 or step == steps:
            rec = dict(step=step, game=game, loss=round(loss.item(), 4), s=round(time.monotonic() - started))
            if torch.cuda.is_available():
                rec['gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            if net.arm == 'resid':
                rec['write'] = [round(torch.tanh(r.alpha).item(), 3) for r in net.readers]
            if sym:
                n = max(1, stat['n_tok'])
                rec.update(tau=round(net.tau, 3), beta=round(net.beta, 3), lam=net.lam,
                           let=round(stat['let'] / n, 4), hard_let=round(stat['hard_let'] / n, 4),
                           ent=round(stat['ent'] / n, 3), div=round(stat['div'] / max(1, stat['n_div']), 3),
                           codes=[len(set(q)) for q in net.codes])
                for k in ('adv', 'eav', 'con'):
                    if stat[f'n_{k}']:
                        rec[k] = round(stat[k] / stat[f'n_{k}'], 4)
                if stat['n_con']:
                    rec['con_gap'] = round(stat['con_gap'] / stat['n_con'], 4)
                if stat['n_adv']:
                    rec['adv_w'] = round(net.adv, 3)
                if eloss:
                    rec['eaves_loss'] = round(sum(eloss) / len(eloss), 4)
        if every and step % every == 0:
            v, more = dev_fn(net, dev) if dev_fn else (dev_nll(net, device, dev), {})
            record.append((step, v))
            rec = dict(rec or dict(step=step), dev_nll=round(v, 4), **more)
            if sym:
                rec.update(dev_soft=round(dev_nll(net, device, dev, mode='soft'), 4),
                           dev_cut=round(dev_nll(net, device, dev, mode='off'), 4),
                           dev_transplant=round(dev_nll(net, device, dev, mode='transplant'), 4))
                ns = note_stats(net, device, dev)
                letters = ns['letters']
                gap = rec['dev_cut'] - v
                rec.update(letters=round(letters, 4), gap=round(gap, 4), examples=ns['examples'])
                keep, (up, down) = LAM_RULE[0], lam_rule
                held = gap0 is not None and gap >= keep * gap0
                if gap0 is None or step <= letters_from:
                    gap0 = gap
                    if out and step <= letters_from:
                        torch.save(dict(steps_run=step, dev_nll=record, weights=kept(net)), out / 'pre.pt')
                else:
                    misses = 0 if held else misses + 1
                    if not held:
                        net.lam *= down
                    elif work is None or letters >= PRIV_LETTERS:
                        net.lam *= up
                    rec.update(held=held, lam_next=net.lam)
                if work is None and held and step > letters_from and letters < PRIV_LETTERS:
                    work = rec['private'] = step
                    if out:
                        torch.save(dict(steps_run=step, dev_nll=record, weights=kept(net)), out / 'gate.pt')
            if v < best * 0.99:
                best, best_step = v, step
            hold = sym and net.adv_max and work and step < work + ADV_RAMP + 300
            if step >= min_steps and step - best_step >= patience and not hold:
                rec['stop'] = f'flat since step {best_step}'
            if sym and lost and misses >= lost:
                rec['stop'] = f'gap not held for {misses} devs in a row'
            if sym and private_steps and work and step >= work + private_steps:
                rec['stop'] = f'{step - work} steps since privacy started'
        if rec:
            print(json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + '\n')
                log.flush()
            if 'stop' in rec:
                return step, record
    return steps, record


def kept(net):
    """The weights a run saves: every LoRA (the eavesdropper's too) and the crumb modules."""
    return {k: v for k, v in net.state_dict().items() if 'lora_' in k or not k.startswith('model.')}


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
    ap.add_argument('--arm', choices=('cut', 'resid', 'kv', 'text', 'sym'))
    ap.add_argument('--base', default=BASE, help='the base model (Hugging Face id)')
    ap.add_argument('--adv', type=float, default=0., help='sym arm: the privacy terms\' weight lambda')
    ap.add_argument('--n_intercept', type=int, default=2000, help='sym arm: trail episodes per split in messages.json')
    ap.add_argument('--contrast', type=float, default=CONTRAST, help='sym arm: the contrastive term\'s weight')
    ap.add_argument('--letters_from', type=int, default=LETTERS[0], help='sym arm: the step the letter penalty starts')
    ap.add_argument('--every', type=int, default=100, help='steps between dev evaluations')
    ap.add_argument('--lam0', type=float, default=LETTERS[1], help='sym arm: the letter weight it starts at')
    ap.add_argument('--gap0', type=float, help='sym arm: the reference gap (default: the last dev before the penalty)')
    ap.add_argument('--lam_rule', default=','.join(map(str, LAM_RULE[1:])),
                    help='sym arm: up,down: the letter weight\'s factor after a dev with the gap held, and without')
    ap.add_argument('--lost', type=int, default=0, help='sym arm: stop after this many devs in a row without the gap')
    ap.add_argument('--private_steps', type=int, default=0, help='sym arm: stop this many steps after privacy starts')
    ap.add_argument('--tau0', type=float, default=SYM_TAU[0], help='sym arm: the notes\' starting temperature')
    ap.add_argument('--beta_full', action='store_true', help='sym arm: the entropy terms at full weight from step 1')
    ap.add_argument('--m', type=int, default=4)
    ap.add_argument('--out', help='output folder (absolute)')
    ap.add_argument('--report', help='write ROOT/report.json from the arms under ROOT and stop')
    ap.add_argument('--steps', type=int, default=2400)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--micro', type=int, help='episodes per backward pass (default: the whole batch)')
    ap.add_argument('--min_steps', type=int, default=600, help='no early stop before this step')
    ap.add_argument('--weight', default='', help='game=w,...: how often a game is drawn (default 1 each)')
    ap.add_argument('--link_lr', type=float, default=1e-3, help='learning rate of the crumb inputs and bridge')
    ap.add_argument('--curriculum', default='', help='until:G_max,...: oldgrowth walkers up to G_max until a step')
    ap.add_argument('--init', help='a run folder whose trained.pt LoRA (and crumbs, tiled to m) to start from')
    ap.add_argument('--alpha0', type=float, default=0., help='resid arm: the readers\' starting write size tanh(alpha)')
    ap.add_argument('--n_test', type=int, default=300)
    ap.add_argument('--chunk', type=int, default=CHUNK, help='streams per teacher-forced forward at evaluation')
    ap.add_argument('--games', default=','.join(CG.GAMES),
                    help='training games; "corpus" stands for every cached corpus task (c_TASK) at corpus.mix_of()')
    ap.add_argument('--eval_games', default='',
                    help='games to evaluate (default: the training games); "corpus" as above')
    ap.add_argument('--n_gen', type=int, default=100, help='greedy-answered test episodes per game')
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
    weights = {g: float(w) for g, w in (kv.split('=') for kv in args.weight.split(',') if kv)}
    games, weights = expand_games(args.games, weights)
    eval_games = expand_games(args.eval_games, {})[0] if args.eval_games else games
    net = build(args.device, args.arm, args.m, base=args.base)
    if args.arm == 'sym':
        net.adv = net.adv_max = args.adv
        net.contrast = args.contrast
    curriculum = [tuple(int(x) for x in c.split(':')) for c in args.curriculum.split(',') if c]
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))
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
                            min_steps=args.min_steps, weights=weights, link_lr=args.link_lr, curriculum=curriculum,
                            every=args.every, letters_from=args.letters_from, lam0=args.lam0, gap0=args.gap0,
                            lam_rule=tuple(float(x) for x in args.lam_rule.split(',')), lost=args.lost,
                            private_steps=args.private_steps, tau0=args.tau0, beta_full=args.beta_full)
        torch.save(dict(steps_run=ran, dev_nll=record, weights=kept(net)), out / 'trained.pt')
    if args.arm == 'sym':
        (out / 'messages.json').write_text(json.dumps(intercept(net, args.device, args.n_intercept)))
    test, rows = evaluate(net, args.device, args.n_test, n_choice=args.n_test // 2, n_gen=args.n_gen, games=eval_games)
    scores = dict(arm=args.arm, m=args.m, base=args.base, adv=args.adv, steps=args.steps, steps_run=ran, batch=args.batch, micro=args.micro,
                  weight=weights, alpha0=args.alpha0, link_lr=args.link_lr, curriculum=curriculum, init=args.init,
                  games=games, eval_games=eval_games, dev_nll=record, test=test)
    if args.arm in CRUMB_ARMS and 'trail' in eval_games:
        scores['probe'] = probe(net, args.device)
    if 'oldgrowth' in eval_games:
        scores['oldgrowth'] = evaluate_oldgrowth(net, args.device, log=lambda r: print(json.dumps(r), flush=True))
    (out / 'scores.json').write_text(json.dumps(scores, indent=1))
    (out / 'answers.json').write_text(json.dumps(rows))
    print(json.dumps({g: v['tf'] for g, v in test.items()}), flush=True)


if __name__ == '__main__':
    main()
