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
            Besides the says, the loss adds lam times the softmax mass on letter tokens (swarm/letters.py, by the
            bytes pending before the token: the unfinished character the note's argmax tokens so far end in; a token is
            a letter if it completes a character of category L* there, breaks or cannot continue what is pending, or
            leaves unfinished bytes that occur inside some letter's UTF-8 encoding; so the lone continuation byte B1
            ends the emoji U+1F9F1 after F0 9F A7 and is no letter there, but ends U+00F1 after C3), BETA (ramped over the first half) times each
            token's entropy minus the entropy of its batch-mean distribution, and --contrast times a contrastive
            term: each reader's says read again with other episodes' notes (--con_negatives: the next episode's, or
            with 0 every other episode in the backward pass), -log the share of p(says | own) + the p(says | other)s
            its own notes take, each p from its says' summed NLL (one other: softplus of own minus other). lam is 0 up to --letters_from, then 0.01, and after each dev times 1.5 while the dev notes' gap
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
import re
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import games as G                           # noqa: E402
from swarm.link import LAYERS, Bridge, blocks           # noqa: E402
from swarm import letters as LT                        # noqa: E402
from swarm import one_note as ON                        # noqa: E402

BASE = 'Qwen/Qwen3-0.6B'
END = '<|im_end|>'
CRUMB_ARMS = ('resid', 'kv', 'sym')
SYM_TAU = (1.0, 0.3)          # sym arm: softmax temperature of the trained notes, start and end
CUE = 'Leave a short note here for whoever comes next.'   # sym: the writer's user turn before each note
NOTE = ON.LINE                 # sym: what a reader's line starts with, before the note's tokens
LETTERS = (300, 0.01)         # sym: letter penalty lam 0 up to step 300 (--letters_from), then 0.01, and after each
LAM_RULE = (0.8, 1.5, 0.5)    # dev times 1.5 while the dev gap (cut minus hard NLL) holds 0.8 of its last value
                              # before the penalty, else times 0.5
PRIV_LETTERS = 0.05           # sym: privacy starts once the dev notes are under 5% letters with the gap held
CONTRAST = 1.0                # sym: the contrastive term's weight (own notes against other episodes')
BETA = 0.1                    # sym: entropy penalty and batch-diversity bonus, ramped from 0 over the first half
ADV_RAMP = 300                # sym: steps over which the privacy terms ramp in once the notes are letterless
# sym, --protect, at a dev once past the gate: the code is lost when its distinct notes fall under this share of the best
# weights' or its swap gap under this share of theirs; then back to the best weights at this times the learning rate
PROTECT = (.5, .25, .5)
RIGHT_EPS = 24                # sym, --protect: test-value episodes per game in a dev's right-answer numbers
ONE_NOTES = 96                 # sym, --one_note: notes per game and dev of the untrained reader's one-note read
ONE_CHARGED = 12               # sym, --one_note: notes per backward pass the frozen-reader penalty charges
ONE_DRAWN = .5                 # sym, --one_note: the least share of a game's weight it is drawn at: a game drawn rarely drifts
ONE_SURE = 5.                  # sym, --one_note: the nats by which a token of the notes the penalty reads stands over its next
ONE_ADV_MOST = 8.              # sym, --one_note: the most a game's penalty weight grows to, by half each dev it stays read
PEAK_MIN = .2                  # sym, --one_note: a game whose follower never got this far over swapped notes cannot collapse
RIGHT_NOISE = .05             # sym, --frozen_adv: an untrained reader's own-minus-swapped share with its interval over this
CHUNK = 48                                              # streams per teacher-forced forward (--chunk)
SPAN = 1024                                             # positions per agent (kv arm)
MODES = ('on', 'off', 'swap', 'transplant')
KV_SOURCES = ('slots', 'sentence', 'lastN', 'evenN', 'mixN')


def kv_source(text):
    """text if it names a kv crumb source (KV_SOURCES, N a positive whole number), else ValueError."""
    if text in ('slots', 'sentence') or re.fullmatch(r'(last|even|mix)[1-9][0-9]*', text):
        return text
    raise ValueError(f'kv source {text!r} is none of {KV_SOURCES}')


def taken(span, source):
    """The positions of the writer's user text at a visit (span) that a kv source keeps as crumbs: all of them
    (sentence), the last N (lastN), N evenly spaced over the text, its first and last token among them (evenN; mixN
    adds the m slot tokens after the visit), all of a text of N or fewer, none (slots)."""
    span = list(span)
    if source == 'slots':
        return []
    if source == 'sentence':
        return span
    n = int(source[3:] if source.startswith('mix') else source[4:])
    if source.startswith('last'):
        return span[-n:]
    if len(span) <= n:
        return span
    return [span[round(j * (len(span) - 1) / max(1, n - 1))] for j in range(n)]


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
        # at that place (sentence; lastN: its last N tokens; evenN: N evenly spaced; mixN: those N and the m slot
        # tokens; taken); rerope re-rotates them to sit just before the reader's
        # first token at the place (else they keep the writer's positions); remember lets each reader token see the
        # crumbs of every place it has been at (else only the place it is at)
        self.source, self.rerope, self.remember = 'slots', False, False
        self.split = False                             # crumb_open: the chat split into two agents' own real chats
        self.one_note = False                          # sym: the frozen-reader penalty reads one note at a time
        self.warm = None                               # sym, one_note: the writer's last notes as the penalty reads them
        self.charged = 0                               # sym, one_note: which notes the next penalty charges
        if arm == 'sym':
            self.banned = letter_table(tok, model.config.vocab_size)[1]
            self.masks = LT.LetterMasks(tok, model.config.vocab_size)   # letters by the bytes pending before a token
            self.letter = self.masks.mask(b'')                           # with nothing pending
            self.on_device = {}                                          # (pending, device): its mask there
            self.letter_first = letter_table(tok, model.config.vocab_size, exact=False)[0]   # the letters metric's
            self.hint = ''                     # a line added to every writer's system prompt (--note_hint)
            self.cue = self._ids(f'<|im_start|>user\n{CUE}{END}\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            self.note = self._ids(NOTE)
            self.tau, self.lam, self.beta, self.adv, self.adv_max = SYM_TAU[0], 0., 0., 0., 0.
            self.contrast = 0.
            self.choice = 0.                   # training: weight of -log p(gold among a say's candidates) (--choice)
            self.con_negatives = 1                 # other episodes per contrastive row (0: all in the pass)
            self.soft, self.eaves_on, self.stash = False, False, []
            self.note_pass = 'soft'            # what a note token sees of the note so far (_notes; --note_pass)
            self.detach_notes = False          # training: send the soft notes without gradient (the warm-up)
            self.writer_base = False           # the warm-up: writers write with the adapter off (the base model's
                                               # notes, no gradient), so only the readers learn
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
        (-1: none, cross-entropy). An agent's system may be None: no system line. A segment with context: a writer's own
        past turn, its say no target and all its tokens, chat markers too, the crumbs. A segment with at: its first
        token's position in the whole chat; the stream then has pos, every token's position."""
        agent, remap = ep['agents'][a], remap or {}
        system = agent['system']
        if system is not None and self.arm == 'sym' and agent['write'] and self.hint:
            system = f'{system}\n{self.hint}'
        ids = [] if system is None else self._ids(f'<|im_start|>system\n{system}<|im_end|>\n')
        true = list(range(-len(ids), 0))
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
            seg0 = len(ids)
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
                add(say + [self.end], rp, not seg.get('context'), sg=s, rows=rows)
                add(self.nl, rp)
            if agent['write'] and p >= 0 and self.arm == 'kv':
                slots.extend((t, p) for t in taken(range(seg0, len(ids)) if seg.get('context') else span, self.source))
            if (agent['write'] and p >= 0 and self.arm in CRUMB_ARMS
                    and (self.arm != 'kv' or self.source == 'slots' or self.source.startswith('mix'))):
                for j in range(self.m):
                    slots.append((len(ids), p))
                    if self.arm != 'sym':
                        add([self.tok.pad_token_id], rp, k=j)
            if 'at' in seg:
                true.extend(range(seg['at'], seg['at'] + len(ids) - seg0))
        if len(ids) > SPAN:
            raise ValueError(f'agent stream of {len(ids)} tokens is longer than SPAN')
        out = dict(ids=ids, place=place, target=target, seg=seg_of, kind=kind, slots=slots, asks=asks, reads=reads,
                   tk=tk)
        if len(true) == len(ids) and any('at' in seg for seg in agent['segments']):
            out['pos'] = true
        if tk_ids:
            out['tk_ids'], out['tk_lp'] = torch.cat(tk_ids), torch.cat(tk_lp)
        return out

    def _pad(self, rows, device, left=False):
        """Tensors ids, att, place, target, kind [N, T] from per-row lists, padded on the right (or left)."""
        T = max(len(r['ids']) for r in rows)
        fill = dict(ids=self.tok.pad_token_id, place=-1, target=False, seg=-1, kind=-1, tk=-1)
        out = {k: torch.full((len(rows), T), v, dtype=torch.bool if k == 'target' else torch.long)
               for k, v in fill.items()}
        if all('pos' in r for r in rows):
            fill['pos'] = 0
            out['pos'] = torch.zeros((len(rows), T), dtype=torch.long)
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
        pos = x['pos'] if 'pos' in x else (att.cumsum(-1) - 1).clamp(min=0) + (a * SPAN if self.arm == 'kv' else 0)
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
            if self.warm is not None:
                out['warm'] = [put(self.warm)] if bank.get('warm') is None else [torch.cat([bank['warm'][0], put(self.warm)], 1)]
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
            wpos = put((x['pos'] if 'pos' in x else (x['att'].cumsum(-1) - 1).clamp(min=0) + a * SPAN).gather(1, idx))
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
        branches off), each over every note at once: the note's tokens see the writer's tokens before its place, then
        the cue turn (CUE) and the note's tokens so far (inputs_embeds, 4D mask), and each token's logits come from
        the final state before it through the model's own head (special tokens masked). Sent: softmax(logits / tau)
        @ the embedding matrix in training (and mode soft), else the argmax token's row. What each token sees of the
        note so far (note_pass): soft, m passes each feeding back the soft tokens so far; greedy, m passes feeding back
        the argmax tokens (the contexts of greedy decoding at evaluation); one, the argmax tokens decoded without
        gradient in m passes, then one pass with gradient over the cue and those m - 1 tokens, whose logits at each
        note position make the soft tokens (greedy's numbers in two passes' worth of backward). Evaluation (no soft)
        decodes greedily in m passes. Returns the note embeddings [N, n, d] and argmax ids [N, n] in slot order (note
        k, token j at k * m + j)."""
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
        track = self.training                  # the letter and entropy sums (the base writer's warm-up notes too)
        banned = self.banned.to(device)
        head = self.model.get_output_embeddings()
        c = len(self.cue)

        def states(fed):
            """The final states [N, Kn, c + len(fed), d] of one pass over each note's cue and the tokens fed."""
            L = c + len(fed)
            new = cue[None, None].expand(N, Kn, -1, -1)
            if fed:
                new = torch.cat([new, torch.stack(fed, 2)], 2)
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
                return self.last.view(N, Kn, L, -1)
            finally:
                self.grab, self.last = False, None

        def logits_of(h):
            return head(h).float().masked_fill(banned, float('-inf'))

        soft = train or self.soft
        mode = self.note_pass if soft else 'greedy'
        embs, ids, sent, warm = [], [], [], []
        pend = [[b''] * Kn for _ in range(N)]                  # training: each note's pending bytes (argmax tokens)
        if mode == 'one':
            with torch.no_grad():
                for j in range(m):
                    hard = logits_of(states(embs)[:, :, -1]).argmax(-1)
                    ids.append(hard)
                    embs.append(E[hard].to(cue.dtype))
            every = logits_of(states(embs[:m - 1])[:, :, c - 1:c - 1 + m])     # [N, Kn, m, V]
        for j in range(m):
            if mode == 'one':
                logits, hard = every[:, :, j], ids[j]
            else:
                logits = logits_of(states(embs)[:, :, -1])
                hard = logits.argmax(-1)
                ids.append(hard)
            if soft or track:
                p = (logits / self.tau).softmax(-1)
            if track:
                self._track(p, hard, valid, j, pend)
                pend = [[self.masks.pending(P, t) for P, t in zip(row, h)] for row, h in zip(pend, hard.tolist())]
            e = (p.to(E.dtype) @ E).to(cue.dtype) if soft else E[hard].to(cue.dtype)
            sent.append(e.detach() if train and self.detach_notes else e)
            if train and self.one_note and self.adv:
                warm.append((self._warm(logits).to(E.dtype) @ E).to(cue.dtype))
            if mode == 'soft':
                embs.append(e)
            elif mode == 'greedy':
                embs.append(E[hard].to(cue.dtype))
        self.warm = torch.stack(warm, 2).reshape(N, Kn * m, -1) if warm else None
        return torch.stack(sent, 2).reshape(N, Kn * m, -1), torch.stack(ids, 2).reshape(N, Kn * m)

    def _warm(self, logits):
        """The note token's distribution for the penalty: softmax at the temperature, token by token, at which the writer's
        first choice stands ONE_SURE nats over its second; never colder than the run's."""
        two = logits.topk(2, -1).values.detach()
        scale = (ONE_SURE / (two[..., 0] - two[..., 1]).clamp_min(1e-6)).clamp_max(1 / self.tau)
        return ((logits - two[..., :1]) * scale[..., None]).softmax(-1)

    def _track(self, p, hard, valid, j, pend):
        """Training, note token j: the sums of the letter mass (each note's letters after its pending bytes pend
        [N][Kn]), the entropy and the batch-mean distribution's entropy (loss terms), the argmax ids' letters, and the
        argmax ids per position."""
        pv = p[valid]
        if not len(pv):
            return
        mean = pv.mean(0)
        s = self.sums
        for P in sorted({P for row in pend for P in row}):
            sel = valid & torch.tensor([[q == P for q in row] for row in pend], device=p.device)
            if sel.any():
                if (P, p.device) not in self.on_device:
                    self.on_device[P, p.device] = self.masks.mask(P).to(p.device)
                letter = self.on_device[P, p.device]
                s['let'] = s['let'] + p[sel][:, letter].sum()
                s['hard_let'] += float(letter[hard[sel]].sum())
        s['ent'] = s['ent'] + -(pv * pv.clamp_min(1e-12).log()).sum()
        s['div'] = s['div'] + -(mean * mean.clamp_min(1e-12).log()).sum()
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
            base = writes and self.arm == 'sym' and self.writer_base
            self.grab = sel is not None and len(sel) > 0
            try:
                _, got = self._forward(rows_t, x, a, bank, 1, writes and not base)
                h = self.last
            finally:
                self.grab, self.last = False, None
            if sel is not None and len(sel):
                logp = self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float().log_softmax(-1)
                ce = -logp.gather(1, x['ids'][sel[:, 0], sel[:, 1] + 1][:, None])[:, 0]
                ce = self._with_kl(ce, logp, streams, sel, x)
                total = total + ce.sum()
                count += len(sel)
                if self.arm == 'sym' and self.adv and torch.is_grad_enabled() and not self.one_note:
                    self._adversary(rows_t, x, a, bank, [(eps[b], st) for b, st in zip(rows, streams)])
                if self.arm == 'sym' and self.contrast and torch.is_grad_enabled() and len(eps) > 1:
                    self._contrast(rows_t, x, a, bank, sel, ce)
                if self.arm == 'sym' and self.choice and torch.is_grad_enabled():
                    targets = [(i, t) for i, (b, st) in enumerate(zip(rows, streams)) if st['reads']
                               for t in self.adv_targets(eps[b], a, st)]
                    if targets:                        # the says' own forward pass: no extra one
                        self.sums['choice'] = self.sums['choice'] - self._among(h, targets)[0].sum()
                        self.sums['n_choice'] += len(targets)
            if writes and base:                       # the warm-up: the base model's notes, without gradient
                with torch.no_grad(), self.model.disable_adapter():
                    _, got = self._forward(rows_t, x, a, bank, 1, True)
                    bank = self._append(bank, rows_t, streams, got, a, x)
            elif writes:
                bank = self._append(bank, rows_t, streams, got, a, x)
        if self.arm == 'sym' and self.one_note and self.adv and need_loss and torch.is_grad_enabled():
            self.one_note_penalty(eps, bank, device)
        return total, count, bank

    def loss(self, eps, device):
        """The says' mean cross-entropy; sym also lam * letter mass + beta * (entropy - batch-mean entropy) per note
        token, contrast * the contrastive term per reader row, adv * the privacy terms' mean per choice group and
        choice * the mean per choice group of -log p(gold among its candidates' first differing token), what the notes
        are for (sums: the terms, kept for the log)."""
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
            if s['n_choice']:
                out = out + self.choice * s['choice'] / s['n_choice']
        return out

    def _contrast(self, rows_t, x, a, bank, sel, own):
        """sym, training: the contrastive term of the rows that have notes to read. Their says read again with other
        episodes' notes: the k-th next episode's for k = 1 .. K (shifted; K con_negatives, 0: every other episode in
        the pass); per row -log the share of p(says | own) + sum_k p(says | other k) its own notes take, each p from
        its says' summed NLL (own: per say token, the pass with its own notes); with K = 1, softplus(own - other).
        con_gap: the mean over the others of their summed NLL minus own."""
        has = (bank['place'][rows_t] >= 0).any(-1)
        if not bool(has.any()):
            return
        B = bank['place'].shape[0]
        K = B - 1 if self.con_negatives == 0 else min(self.con_negatives, B - 1)
        gold = x['ids'][sel[:, 0], sel[:, 1] + 1]
        zero = torch.zeros(len(rows_t), device=own.device)
        mine = zero.index_add(0, sel[:, 0], own)
        others = []
        for k in range(1, K + 1):
            self.grab = True
            try:
                self._forward(rows_t, x, a, shifted(bank, k), 1, False)
                h = self.last
            finally:
                self.grab, self.last = False, None
            other = F.cross_entropy(self.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float(), gold,
                                    reduction='none')
            others.append(zero.index_add(0, sel[:, 0], other))
        o = torch.stack(others)                                            # [K, rows]
        keep = has & (zero.index_add(0, sel[:, 0], torch.ones_like(own)) > 0)
        con = torch.logsumexp(torch.cat([-mine[None], -o]), 0) + mine
        self.sums['con'] = self.sums['con'] + con[keep].sum()
        self.sums['con_gap'] += float((o - mine).mean(0)[keep].detach().sum())
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
        return self._among(h, targets)

    def _among(self, h, targets):
        """log p(gold) among each target's candidates, from the final norm's output h; and the candidates' count."""
        dev = h.device
        at = (torch.tensor([i for i, _ in targets], device=dev), torch.tensor([t[0] for _, t in targets], device=dev))
        C = max(1 + len(t[2]) for _, t in targets)
        cand = torch.tensor([[t[1]] + t[2] + [-1] * (C - 1 - len(t[2])) for _, t in targets], device=dev)
        W = self.model.get_output_embeddings().weight[cand.clamp(min=0)].float()
        lg = torch.einsum('gd,gcd->gc', h[at].float(), W).masked_fill(cand < 0, float('-inf'))
        return lg.log_softmax(-1)[:, 0], (cand >= 0).sum(-1).float()

    def _one_stream(self, ask, gold):
        """The untrained reader's prompt for one note (one_note.chat, the note's tokens read from the bank) with the
        gold answer after it: a stream for _pad, and where the answer starts."""
        ids = self._ids(f'<|im_start|>system\n{ask["system"]}<|im_end|>\n<|im_start|>user\n') + list(self.note)
        reads = []
        for j in range(self.m):
            reads.append((len(ids), ask['w'], ask['k'] * self.m + j))
            ids.append(self.tok.pad_token_id)
        ids = ids + list(self.nl) + self._ids(f'{ask["question"]}<|im_end|>\n') + list(self.head)
        cut = len(ids)
        ids = ids + gold
        n = len(ids)
        return dict(ids=ids, place=[-1] * n, target=[False] * n, seg=[-1] * n, kind=[-1] * n, tk=[-1] * n, reads=reads), cut

    def one_note_penalty(self, eps, bank, device):
        """sym: the frozen reader's one-note read, charged: the base model with its adapter disabled reads up to
        ONE_CHARGED notes of these episodes, one at a time with the soft notes, so the gradient reaches their writers; per
        target relu(log p(gold) - log p(gold) with the note masked), among its candidates: what the note gives the reader
        over its own prior. No floor at chance: a reader whose prior says Safe stays under it on every note that says Trap,
        which is what one_note.moved reads. One way only: any two fixed strings move a reader by some amount, so charging
        moves either way asks trap's two-note code for the impossible (a run's follower fell to .73 with its read unchanged
        at -.51). The notes it reads are _warm's: a code the writer is sure of has softmax(logits / tau) at one token and no
        gradient at all. A run's count penalty of 2.5 nats at weight 3 had a median gradient norm of .011 (trail's whole
        step had 6.3) and never moved those notes; on those weights its gradient was 0 at temperature 1. ONE_SURE 3 is too
        warm: count's notes keep 4% of their mass on their token, the reader reads nothing and nothing is charged (.08
        nats); at 5 they keep 51%, it is charged 2.4 nats with a gradient norm of 30; at 8, 99%, with .1."""
        items = [(b, ask) for b, ep in enumerate(eps) for ask in ON.asks(ep)]
        if not items:
            return
        self.charged += 1
        items = random.Random(self.charged).sample(items, min(ONE_CHARGED, len(items)))
        streams, rows, aims = [], [], []
        for i, (b, ask) in enumerate(items):
            gold = self._ids(ask['gold'])
            st, cut = self._one_stream(ask, gold)
            streams.append(st)
            rows.append(b)
            aims += [(i, t) for t in ON.targets(cut, gold, [self._ids(x) for x in ask['alts']])]
        if not aims:
            return
        x = self._pad(streams, device)
        rows_t = torch.tensor(rows, device=device)
        with self.model.disable_adapter():
            lp, _ = self._gold_logp(rows_t, x, 0, dict(bank, e=bank['warm']), aims, True)
            with torch.no_grad():
                lp0, _ = self._gold_logp(rows_t, x, 0, bank, aims, False)
        self.sums['adv'] = self.sums['adv'] + F.relu(lp - lp0).sum()
        self.sums['n_adv'] += len(aims)

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


def shifted(bank, k=1):
    """bank with each episode's crumbs replaced by those of the k-th episode after it (wrapping round: with k = 1 the
    last episode's by the first's)."""
    B = bank['place'].shape[0]
    idx = (torch.arange(B, device=bank['place'].device) + k) % B
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
def sym_dev(net, device, dev):
    """sym arm: a dev's NLLs with the soft notes, cut, transplanted (the next dev episode's notes, of any game) and
    swapped (another episode of the same game's notes), and note_stats' letter shares and examples."""
    out = dict(dev_soft=round(dev_nll(net, device, dev, mode='soft'), 4),
               dev_cut=round(dev_nll(net, device, dev, mode='off'), 4),
               dev_transplant=round(dev_nll(net, device, dev, mode='transplant'), 4),
               dev_swap=round(dev_nll(net, device, dev, mode='swap'), 4))
    ns = note_stats(net, device, dev)
    out.update({k: round(ns[k], 4) for k in ('letters', 'letters_exact', 'letters_text', 'letters_max')},
               letters_by_game={g: round(v, 4) for g, v in ns['letters_by_game'].items()}, examples=ns['examples'],
               distinct=ns['distinct'])
    return out


def swap_chunks(eps, bs=50):
    """eps in forced chunks for the within-game swap: each game's episodes together (a last chunk of one joins the
    chunk before it), so the transplant's next episode is always one of the same game; a game with one episode is a
    chunk of one, which reads its own notes (no swap there, so it adds no gap rather than a cross-game one)."""
    return [[eps[i] for i in c] for c in swap_index(eps, bs)]


def swap_index(eps, bs=50):
    """swap_chunks as lists of indices into eps."""
    groups = {}
    for i, ep in enumerate(eps):
        groups.setdefault(ep['game'], []).append(i)
    chunks = []
    for g in groups.values():
        parts = [g[s0:s0 + bs] for s0 in range(0, len(g), bs)]
        if len(parts) > 1 and len(parts[-1]) == 1:
            tail = parts.pop()
            parts[-1] = parts[-1] + tail
        chunks += parts
    return chunks


def dev_values(net, device, eps, bs=50, mode='on'):
    """Each episode's mean say NLL in eps' order (None: no says), crumbs on or forced's mode; swap: each episode
    reads the notes of another episode of its own game (the transplant within swap_chunks)."""
    net.eval()
    got = [None] * len(eps)
    idx = swap_index(eps, bs) if mode == 'swap' else [list(range(s0, min(len(eps), s0 + bs)))
                                                       for s0 in range(0, len(eps), bs)]
    for c in idx:
        vals = net.forced([eps[i] for i in c], device, mode='transplant' if mode == 'swap' else mode, choose=False)[0]
        for i, v in zip(c, vals):
            got[i] = v
    net.train()
    return got


def dev_nll(net, device, eps, bs=50, mode='on'):
    """Mean say NLL over eps (dev_values' modes). swap: a note that is the same for every episode, or that carries
    only the game, scores no gap against it."""
    got = [v for v in dev_values(net, device, eps, bs, mode) if v is not None]
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
                                       letters=sum(net.masks.walk(ids)[0]),
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
    """sym arm: the hard notes on eps: letters, the share of their tokens that are letters by the first rule (every
    token that is not complete UTF-8 a letter); letters_exact, by the penalty's rule (swarm/letters.py, each note
    read from its start with the bytes pending before each token); letters_text,
    the share of the decoded notes' characters other than spaces that are of category L*, pooled over every note,
    and per game (letters_by_game) with the highest game's (letters_max); and the first show trail episodes' explorer
    notes as 'object: text', and distinct, how many different notes those are."""
    import unicodedata
    net.eval()
    n_tok, n_let, n_exact, n_chr, n_ltr, examples, shown, notes = 0, 0, 0, 0, 0, [], 0, []
    per = collections.defaultdict(lambda: [0, 0])          # game: [characters, letters]
    first = net.letter_first.to(device)
    for s0 in range(0, len(eps), bs):
        chunk = eps[s0:s0 + bs]
        _, _, bank = net.run(chunk, device, need_loss=False)
        if bank.get('ids') is None:
            continue
        ids, real = bank['ids'][0], bank['place'] >= 0
        n_tok, n_let = n_tok + int(real.sum()), n_let + int(first[ids][real].sum())
        for b in range(len(chunk)):
            got = ids[b][real[b]].tolist()
            n_exact += sum(sum(net.masks.walk(got[g:g + net.m])[0]) for g in range(0, len(got), net.m))
            got = [c for c in net.tok.decode(ids[b][real[b]].tolist()) if not c.isspace()]
            k = sum(unicodedata.category(c).startswith('L') for c in got)
            n_chr, n_ltr = n_chr + len(got), n_ltr + k
            g = per[chunk[b].get('game', '?')]
            g[0], g[1] = g[0] + len(got), g[1] + k
        for b, ep in enumerate(chunk):
            if 'trail' not in ep or shown >= show:
                continue
            shown += 1
            cols = ((bank['place'][b] >= 0) & (bank['agent'][b] == 0)).nonzero()[:, 0].tolist()
            for k, r in enumerate(ep['trail']['order']):
                note = net.tok.decode(ids[b, cols[k * net.m:(k + 1) * net.m]].tolist())
                examples.append(f'{ep["trail"]["objects"][r]}: {note}')
                notes.append(note)
    net.train()
    by_game = {g: l / c for g, (c, l) in sorted(per.items()) if c}
    return dict(letters=n_let / max(1, n_tok), letters_exact=n_exact / max(1, n_tok), letters_text=n_ltr / max(1, n_chr),
                letters_by_game=by_game, letters_max=max(by_game.values(), default=0.), examples=examples,
                distinct=len(set(notes)))


def letter_bytes(_cache=[]):
    """Every byte string that occurs inside the UTF-8 encoding of a character of Unicode category L* (each proper
    substring of each letter's encoding), over all code points; computed once."""
    if not _cache:
        import unicodedata
        got = set()
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF or not unicodedata.category(chr(cp)).startswith('L'):
                continue
            b = chr(cp).encode()
            got.update(b[i:j] for i in range(len(b)) for j in range(i + 1, len(b) + 1) if (i, j) != (0, len(b)))
        _cache.append(frozenset(got))
    return _cache[0]


def fragment_letter(b):
    """For a byte-level BPE token whose bytes b are not complete UTF-8: True if it can be part of a letter. Its bytes
    split into a head of continuation bytes, a complete middle and a tail (an unfinished sequence); it is a letter if
    the middle has an L* character, or the head or the tail occurs inside some letter's UTF-8 encoding."""
    import unicodedata
    k = 0
    while k < len(b) and (b[k] & 0xC0) == 0x80:
        k += 1
    head, rest = b[:k], b[k:]
    m = len(rest)
    for q in range(len(rest) - 1, -1, -1):
        c = rest[q]
        if (c & 0xC0) != 0x80:
            need = 1 if c < 0x80 else 2 if c >> 5 == 6 else 3 if c >> 4 == 14 else 4
            if len(rest) - q < need:
                m = q
            break
    mid, tail = rest[:m], rest[m:]
    try:
        text = mid.decode('utf-8')
    except UnicodeDecodeError:
        return True
    sub = letter_bytes()
    return any(unicodedata.category(c).startswith('L') for c in text) or head in sub or tail in sub


def letter_table(tok, V, exact=True, _cache={}):
    """[V] bool tensors (letter, banned). A token is a letter if its text has a character of Unicode category L*; a
    token whose bytes are not complete UTF-8 (byte-level BPE: tokens are strings over the GPT-2 byte alphabet) is a
    letter if it can be part of one (fragment_letter; exact False: every such token, the first rule, kept for the
    letters metric); banned: the special and added tokens and the rows past the tokenizer's."""
    key = (tok.name_or_path, V, exact)
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
                raw = bytes(byte[c] for c in t)
            except KeyError:
                letter[i] = True
                continue
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                letter[i] = fragment_letter(raw) if exact else True
                continue
            letter[i] = any(unicodedata.category(c).startswith('L') for c in text)
        _cache[key] = (letter, banned)
    return _cache[key]


def load(out, device, ckpt='trained.pt'):
    """The trained net of a run folder (its args.json and trained.pt, or another checkpoint there: gate.pt, pre.pt)."""
    out = Path(out)
    args = json.loads((out / 'args.json').read_text())
    net = build(device, args['arm'], args['m'], base=args.get('base', BASE))
    net.load_state_dict(torch.load(out / ckpt, map_location=device)['weights'], strict=False)
    if args['arm'] == 'sym':
        net.adv = net.adv_max = args.get('adv', 0.)
        net.tau = SYM_TAU[1]
        net.hint = args.get('note_hint', '')
        net.note_pass = args.get('note_pass', 'soft')
    return net


CRUMB_PROBE = []    # eval only (crumb_select.py): f(layer, query, key, mask, S, scaling) sees each layer's attention
CRUMB_KEEP = {}     # eval only: layer -> [N, kv heads, S] bool, the crumbs each key-value head may read


def keep_heads(mask, keep, heads, S):
    """mask [N, 1 or heads, T, S + T] (bool, or additive) with crumb s hidden from the query heads of each key-value
    head whose keep [N, kv heads, S] is False."""
    k = keep.repeat_interleave(heads // keep.shape[1], 1)[:, :, None, :]
    m = mask.expand(-1, heads, -1, -1)
    crumbs = m[..., :S] & k if m.dtype == torch.bool else torch.where(k, m[..., :S], torch.finfo(m.dtype).min)
    return torch.cat([crumbs, m[..., S:]], -1)


def crumb_attention(module, query, key, value, attention_mask, crumb_k=None, crumb_v=None, crumb_got=None, **kw):
    """kv arm: sdpa attention with the bank's crumb keys and values (crumb_k, crumb_v: per layer [N, kv heads, S, head
    dim]) in front of the stream's own; crumb_got collects each layer's keys and values, crumbs first (the first
    pass's: a gradient-checkpointing recompute leaves them)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    i = module.layer_idx
    if crumb_k is not None:
        S = crumb_k[i].shape[2]
        key, value = torch.cat([crumb_k[i].to(key.dtype), key], 2), torch.cat([crumb_v[i].to(value.dtype), value], 2)
        for f in CRUMB_PROBE:
            f(i, query, key, attention_mask, S, module.scaling)
        if i in CRUMB_KEEP:
            attention_mask = keep_heads(attention_mask, CRUMB_KEEP[i], query.shape[1], S)
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
          detach_until=0, gate_held=1, gate_distinct=0, gate_letters=PRIV_LETTERS, lam_min=0., ref_letters=None,
          dev_fn=None, cut0=None, gap_ref='first', dev0=False, gap_by='cut', resume=None, letters_by='max',
          protect=False, frozen_adv=0.):
    """steps of batch episodes of one game each (similar lengths, little padding), the game drawn at random for each
    step, in backward passes of micro episodes (the mean of their losses). Every every steps, the mean say NLL of 8
    fixed training-value episodes per game; training stops once that has not improved by 1% for patience steps
    (after min_steps). curriculum [(until_step, G_max)]: oldgrowth episodes draw G from 1 to G_max up to until_step
    (afterwards the game's own 1 to 4). sym: tau annealed SYM_TAU, beta ramped to BETA over the first half; each dev
    evaluation also takes the soft notes, the cut, the transplant, and the hard notes' letter share with example
    notes; lam 0 up to letters_from (OUT/pre.pt: the weights at its last dev), then LETTERS[1], and after each dev
    times LAM_RULE[1] while the gap (cut minus hard dev NLL, or swap minus hard: gap_by) is at least LAM_RULE[0] of its value at the last dev
    before the penalty, else times LAM_RULE[2]; once the letter share is under PRIV_LETTERS with the gap held
    (OUT/gate.pt: the weights then) the privacy terms ramp in. A continuation (sym, from a run's trained.pt) passes
    letters_from 0, the weight to start from (lam0), the reference gap (gap0: the first run's last dev before the
    penalty), lam_rule (times up on a held dev, down otherwise; once private, no rise while the notes stay under
    PRIV_LETTERS), tau0 (the annealed end) and beta_full; lost stops training after that many devs in a row without
    the gap held, private_steps that many steps after privacy starts. Then
    over ADV_RAMP steps, the eavesdropper trains on every step after each backward pass (its own AdamW at lr), and
    training runs at least ADV_RAMP + 300 steps past it. The dev NLL that stops training is the hard one. Returns
    the steps run and the NLL record. dev: these dev episodes instead of 8 per game; dev_fn(net, dev) -> (the dev
    value, more numbers to log) instead of the mean say NLL. cut0 (sym continuation): the gap is cut0 minus the hard
    dev NLL, a fixed reference cut (the first run's at its gap0 dev), so a reader whose own no-note NLL drifts up cannot
    hold the gap while its hard NLL gets worse; the gap against the dev's own cut is logged as gap_now. gap_ref max:
    the reference gap is the largest dev gap so far (for a letter weight from step 0, with no dev before it); dev0:
    one sym dev before the first step (the untrained notes). Until the reference gap is positive (the notes help
    the reader) no dev holds it, the weight stays, no dev counts as missed and each dev's gap is the next reference.
    The letter share that gates privacy and the weight's
    rise is letters_text (the share of the decoded notes' characters that are letters: one token " pillow" is six),
    of the dev's game with the most (letters_by max: one game writing words cannot hide among six writing digits) or
    pooled over every note (all); both are logged, letters_max and letters_text, and each game's, letters_by_game.
    The privacy gate can be stricter: gate_held devs in a row with the gap held, at least gate_distinct different
    notes among the dev's example notes (note_stats' distinct) and letters_text under gate_letters (default
    PRIV_LETTERS); until then only the follower, the letter weight, the entropy terms and the contrastive term train.
    lam_min: the weight never falls below it (a weight halved after missed devs lets words back into the notes);
    ref_letters (gap_ref max): only a dev whose decoded notes are under this share of letters can raise the reference
    gap (a dev of words would set a bar no letterless code holds); gap_by: the gap the controller and the privacy gate
    use, cut (the cut, or cut0, minus the hard dev NLL: a note that is the same for every episode can win it, as the
    reader may do better with any note than with none) or swap (the same reader's NLL with another episode of the same
    game's notes minus with its own: only notes that tell episodes apart win it); both are logged, gap_cut and
    gap_swap;
    letters (the first rule) and letters_exact (the penalty's, per token) are logged beside it. detach_until: up to
    that step (the warm-up) writers write with the adapter off, so the notes are the base model's and only the readers
    learn them, the entropy terms are off, and the letter controller only takes the reference gap; from the next
    step the writers use the adapter with the full gradient. resume (resumed(), a sym run's checkpoint): go on from
    its step with its letter weight, reference gap, privacy start and the devs held and missed in a row, the episode
    stream replayed to that step and a fresh optimizer whose learning rate warms up again over 50 steps toward the
    cosine at the true step; a dev at that step first (resumed: True) checks the weights, and changes nothing.
    protect (sym, past the gate): every dev logs right (right_by_game on RIGHT_EPS fresh test-value episodes per game:
    the follower's and the untrained base model's right-answer share with own and swapped notes) and right_train (the
    follower's on the dev's training-value episodes). A dev with letters under gate_letters, at least gate_distinct
    distinct notes and the best score so far (own minus swapped for the follower, over the games, by their --weight)
    saves OUT/best.pt. A dev that has lost the code (PROTECT) goes back to those weights, with fresh optimizers and the
    learning rate halved and warming up again: a writer whose soft notes saturate on one token has no gradient left, so
    a lost code does not come back by itself. Training ends on the best weights. The letter weight goes per game: from
    the weight at the gate, times 1.5 at a dev where that game's notes are over PRIV_LETTERS letters, back down by 0.75
    where under 1%, so one game's words squeeze no other game's code. Games are drawn by --weight times what their
    follower still lacks (1 minus own minus swapped as a share of the room above chance, at least 1/4). OUT/steps.jsonl
    logs every step's terms and gradient norm. frozen_adv: if at two devs in a row the untrained base model's own minus
    swapped share has its 95% interval over RIGHT_NOISE in some game, adv (with no eavesdropper, which never trains)
    ramps to this over 100 steps: a penalty on that frozen reader alone."""
    games = games or CG.GAMES
    if net.arm == 'sym' and net.contrast and (micro or batch) < 2:
        raise ValueError('the contrastive term reads the next episode\'s notes: it needs 2 or more episodes per backward '
                         'pass (--micro)')
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
    warm, start = min(50, steps), resume['step'] if resume else 0
    warm_from, lr_scale = 0, 1.                        # going back to the best weights: warm up again, at a lower rate
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s - warm_from + 1) / warm) * lr_scale * 0.5 *
                                              (1 + math.cos(math.pi * min(1, (s + start) / steps))))
    log = open(out / 'log.jsonl', 'a') if out else None
    fine = open(out / 'steps.jsonl', 'a') if out and protect else None
    top, lam_base, lam_game, adv_from, hot_devs = None, None, {}, None, 0   # protect: the best weights' dev and the rest
    hot_games, peak, adv_game = set(), {}, {}          # --one_note: the games the frozen reader has read above noise, the
    #                                                    most each game's follower has got over swapped notes, each penalty
    weights0 = dict(weights or {})
    right_eps = {g: [CG.episode(g, r, True) for _ in range(RIGHT_EPS)]
                 for g, r in ((g, random.Random(f'right-{g}')) for g in games if g in CG.CHOICES)} if protect else {}
    one_eps = {g: [CG.episode(g, r, True) for _ in range(RIGHT_EPS)]
               for g, r in ((g, random.Random(f'one-{g}')) for g in games)} if protect and net.one_note else {}
    train_eps = {g: [e for e in dev if e['game'] == g] for g in right_eps} if protect else {}

    def best_weights():
        got = net.load_state_dict(torch.load(out / 'best.pt', map_location=device)['weights'], strict=False)
        if got.unexpected_keys:
            raise RuntimeError(f'best.pt: weights not in the net: {got.unexpected_keys[:5]}')

    def guard(step, rec):
        """protect's dev: the right-answer numbers, then the best weights kept or gone back to, the letter weights and
        the game weights (train's docstring). Adds to rec."""
        nonlocal top, weights, warm_from, lr_scale, misses, run_held, adv_from, hot_devs
        right = rec['right'] = right_by_game(net, device, right_eps)
        rec['right_train'] = right_by_game(net, device, train_eps, ('follower',))
        total = sum(weights0.get(g, 1.) for g in right)
        one = rec['one_note'] = one_note_by_game(net, device, one_eps) if net.one_note else {}
        score = rec['score'] = round(sum(weights0.get(g, 1.) * (r['follower_own'] - r['follower_swap'] - leak(one.get(g)))
                                         for g, r in right.items()) / total, 4)
        gap = rec['gap_swap']

        def fell(share):                               # --one_note: the games whose follower is under this share of its peak
            return [g for g, r in right.items() if net.one_note and peak[g] >= PEAK_MIN and
                    r['follower_own'] - r['follower_swap'] < share * peak[g]]

        for g, r in right.items():
            peak[g] = max(peak.get(g, 0.), r['follower_own'] - r['follower_swap'])
        rec['fell'] = fell(PROTECT[0])
        lost = top is not None and (rec['distinct'] < PROTECT[0] * top['distinct'] or gap < PROTECT[1] * top['gap'] or
                                    bool(fell(PROTECT[1])))
        if lost:
            best_weights()
            lr_scale *= PROTECT[2]
            opt.state.clear()
            opt_e.state.clear()
            warm_from = sched.last_epoch
            for g, base in zip(opt.param_groups, sched.base_lrs):
                g['lr'] = base * sched.lr_lambdas[0](warm_from)
            misses = run_held = 0
            rec.update(back_to=top['step'], lr_scale=lr_scale)
        elif rec['letters_max'] < gate_letters and rec['distinct'] >= gate_distinct and not rec['fell'] and \
                (top is None or score > top['score']):
            top = dict(step=step, score=score, distinct=rec['distinct'], gap=gap)
            rec['best'] = True
            torch.save(dict(steps_run=step, dev_nll=record, weights=kept(net), lam=lam_game, lam_base=lam_base),
                       out / 'best.pt')
        for g, v in rec['letters_by_game'].items():    # the letter weight of each game
            lam = lam_game.get(g, lam_base)
            lam_game[g] = lam * 1.5 if v >= PRIV_LETTERS else max(lam_base, lam * .75) if v < .01 else lam
        rec['lam_games'] = {g: round(v, 3) for g, v in lam_game.items()}
        weights = dict(weights0)                       # the games drawn: more of what the follower still lacks
        for g, r in right.items():                     # --one_note: and of what the untrained model still reads
            lack = 1 - max(0., r['follower_own'] - r['follower_swap']) / max(1e-6, 1 - r['chance'])
            weights[g] = weights0.get(g, 1.) * (max(ONE_DRAWN, lack, leak(one.get(g))) if net.one_note else max(.25, lack))
        rec['weights'] = {g: round(v, 3) for g, v in weights.items()}
        hot = [g for g, r in one.items() if max(r['ci'][0], r['moved_ci'][0]) > RIGHT_NOISE] if net.one_note else \
            [g for g, r in right.items() if r['base_ci'][0] > RIGHT_NOISE]
        hot_devs = hot_devs + 1 if hot else 0
        hot_games.update(hot)
        if hot:
            rec['base_hot'] = hot
        if frozen_adv and adv_from is None and hot_devs >= (1 if net.one_note else 2):
            adv_from, net.adv_max = step, frozen_adv
            rec['frozen_adv'] = frozen_adv
        if frozen_adv and net.one_note:                # a penalty that is not biting grows, unless its follower is falling
            for g in hot:
                if g not in rec['fell']:
                    adv_game[g] = min(ONE_ADV_MOST, adv_game[g] * 1.5) if g in adv_game else frozen_adv
            for g in rec['fell'] if lost else ():
                adv_game[g] = adv_game.get(g, frozen_adv) * .5
            rec['adv_games'] = {g: round(v, 3) for g, v in adv_game.items()}

    for done in range(1, start + 1):                   # resume: the episode stream as it was at that step
        game = rng.choices(games, [(weights or {}).get(g, 1.) for g in games])[0]
        gmax = next((g for until, g in curriculum or () if done <= until), None) if game == 'oldgrowth' else None
        for _ in range(batch):
            CG.episode(game, rng, **(dict(G=rng.randint(1, gmax)) if gmax else {}))
    if resume:
        net.tau = tau0 * (SYM_TAU[1] / tau0) ** (start / steps)
        rec = dict(step=start, resumed=True, dev_nll=round(dev_nll(net, device, dev), 4), **sym_dev(net, device, dev))
        rec.update(gap_cut=round(rec['dev_cut'] - rec['dev_nll'], 4),
                   gap_swap=round(rec['dev_swap'] - rec['dev_nll'], 4))
        if protect:
            lam_base, record = resume['lam'], list(resume['record'])
            guard(start, rec)
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + '\n')
            log.flush()
    elif sym and dev0:                                 # the notes before any training
        rec = dict(step=0, dev_nll=round(dev_nll(net, device, dev), 4), **sym_dev(net, device, dev))
        rec.update(gap_cut=round(rec['dev_cut'] - rec['dev_nll'], 4), gap_swap=round(rec['dev_swap'] - rec['dev_nll'], 4))
        rec['gap'] = rec['gap_swap' if gap_by == 'swap' else 'gap_cut']
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + '\n')
            log.flush()
    started = time.monotonic()
    best, best_step, record, work, letters, misses, run_held = float('inf'), 0, [], None, None, 0, 0
    if resume:
        record, work, misses, run_held = list(resume['record']), resume['work'], resume['misses'], resume['run_held']
        gap0, net.lam = resume['gap0'], resume['lam']
        best_step, best = min(record, key=lambda r: r[1]) if record else (0, float('inf'))
    for step in range(start + 1, steps + 1):
        game = rng.choices(games, [(weights or {}).get(g, 1.) for g in games])[0]
        gmax = next((g for until, g in curriculum or () if step <= until), None) if game == 'oldgrowth' else None
        eps = [CG.episode(game, rng, **(dict(G=rng.randint(1, gmax)) if gmax else {})) for _ in range(batch)]
        if sym:
            t1 = SYM_TAU[1]
            net.tau = tau0 * (t1 / tau0) ** (step / steps)
            net.beta = BETA if beta_full else BETA * min(1., step / max(1, steps / 2))
            net.detach_notes = net.writer_base = step <= detach_until   # the warm-up: the base model's notes,
            if net.detach_notes:                                         # which only the readers learn
                net.beta = 0.
            if step == letters_from + 1 and not resume:
                net.lam = lam0
            net.adv = net.adv_max * min(1., (step - work) / ADV_RAMP) if work else 0.
            net.eaves_on = bool(work and net.adv_max)
            if protect and lam_base is not None:
                net.lam = lam_game.get(game, lam_base)
            if frozen_adv:                             # the frozen reader's penalty alone, no eavesdropper
                charged = adv_from and (not net.one_note or game in hot_games)
                most = adv_game.get(game, net.adv_max) if net.one_note else net.adv_max
                net.adv = most * min(1., (step - adv_from) / 100) if charged else 0.
                net.eaves_on = False
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
        grad = torch.nn.utils.clip_grad_norm_(lora + link, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if eloss:
            torch.nn.utils.clip_grad_norm_(net.eaves_params(), 1.0)
            opt_e.step()
            opt_e.zero_grad(set_to_none=True)
        if fine:
            n = max(1, stat['n_tok'])
            fine.write(json.dumps(dict(
                step=step, game=game, loss=round(loss.item(), 4), grad=round(float(grad), 4), adv_w=round(net.adv, 3),
                ent=round(stat['ent'] / n, 4), div=round(stat['div'] / max(1, stat['n_div']), 4),
                codes=[len(set(q)) for q in net.codes],
                **{k: round(stat[k] / stat[f'n_{k}'], 4) for k in ('adv', 'eav', 'con', 'choice') if stat[f'n_{k}']})) + '\n')
            fine.flush()
        rec = None
        if step % 25 == 0 or step == steps:
            rec = dict(step=step, game=game, loss=round(loss.item(), 4), s=round(time.monotonic() - started))
            if torch.cuda.is_available():
                rec['gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            if net.arm == 'resid':
                rec['write'] = [round(torch.tanh(r.alpha).item(), 3) for r in net.readers]
            if sym:
                n = max(1, stat['n_tok'])
                rec.update(tau=round(net.tau, 3), beta=round(net.beta, 3), lam=net.lam, detached=net.detach_notes,
                           let=round(stat['let'] / n, 4), hard_let=round(stat['hard_let'] / n, 4),
                           ent=round(stat['ent'] / n, 3), div=round(stat['div'] / max(1, stat['n_div']), 3),
                           codes=[len(set(q)) for q in net.codes])
                for k in ('adv', 'eav', 'con', 'choice'):
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
                rec.update(sym_dev(net, device, dev))
                letters = rec['letters_max' if letters_by == 'max' else 'letters_text']
                gap_cut, gap_swap = (rec['dev_cut'] if cut0 is None else cut0) - v, rec['dev_swap'] - v
                gap = gap_swap if gap_by == 'swap' else gap_cut
                rec.update(gap=round(gap, 4), gap_cut=round(gap_cut, 4), gap_swap=round(gap_swap, 4))
                if cut0 is not None:
                    rec['gap_now'] = round(rec['dev_cut'] - v, 4)
                keep, (up, down) = LAM_RULE[0], lam_rule
                held = gap0 is not None and gap0 > 0 and gap >= keep * gap0
                run_held = 0 if gap0 is None or step <= max(letters_from, detach_until) or gap0 <= 0 or not held \
                    else run_held + 1                  # devs in a row with the gap held, the controller's
                if gap0 is None or step <= max(letters_from, detach_until):
                    gap0 = gap
                    if out and step <= letters_from:
                        torch.save(dict(steps_run=step, dev_nll=record, weights=kept(net)), out / 'pre.pt')
                elif gap0 <= 0:                        # the notes have not helped yet: no gap to hold, this one is
                    gap0 = gap                         # the next reference, the weight stays and no dev is missed
                    rec.update(held=False, lam_next=net.lam)
                else:
                    misses = 0 if held else misses + 1
                    if protect and work is not None:   # guard's letter weights per game
                        pass
                    elif not held:
                        net.lam = max(lam_min, net.lam * down)
                    elif work is None or letters >= PRIV_LETTERS:
                        net.lam *= up
                    rec.update(held=held, lam_next=net.lam)
                if gap_ref == 'max':
                    if ref_letters is None or letters < ref_letters:
                        gap0 = max(gap0, gap)
                    rec['gap_ref'] = round(gap0, 4)
                if work is None and held and run_held >= gate_held and rec['distinct'] >= gate_distinct and \
                        step > max(letters_from, detach_until) and letters < gate_letters:
                    work = rec['private'] = step
                    if out:
                        torch.save(dict(steps_run=step, dev_nll=record, weights=kept(net)), out / 'gate.pt')
                if protect and work is not None:
                    lam_base = net.lam if lam_base is None else lam_base
                    guard(step, rec)
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
                break
    else:
        step = steps
    net.detach_notes = net.writer_base = False
    if protect and top and top['step'] != step:
        best_weights()
        rec = dict(step=step, final_weights=top['step'])
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
    return step, record


def resumed(path):
    """train's resume state from a sym run's checkpoint (gate.pt, trained.pt) and the log.jsonl beside it: the
    checkpoint's step, the letter weight and reference gap its dev left, the step privacy started, the devs in a row
    held (after the warm-up) and missed, and the dev record up to it."""
    path = Path(path)
    step = torch.load(path, map_location='cpu')['steps_run']
    devs = [r for r in map(json.loads, (path.parent / 'log.jsonl').read_text().splitlines())
            if 'dev_nll' in r and 'examples' in r and not r.get('resumed') and r['step'] <= step]
    last = devs[-1]
    if last['step'] != step or 'lam_next' not in last or 'gap_ref' not in last:
        raise SystemExit(f'{path}: no controller dev at step {step} in its log')
    work = next((r['private'] for r in devs if 'private' in r), None)
    misses = run_held = 0
    for r in reversed(devs):
        if 'held' not in r or r['held']:
            break
        misses += 1
    for r in reversed(devs):
        if not r.get('held'):
            break
        run_held += 1
    return dict(step=step, lam=last['lam_next'], gap0=last['gap_ref'], work=work, misses=misses, run_held=run_held,
                record=[(r['step'], r['dev_nll']) for r in devs if r['step'] > 0])


def boot_interval(xs, n=1000, seed=0):
    """The 95% bootstrap interval of the mean of xs."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return [round(means[int(.025 * n)], 4), round(means[int(.975 * n) - 1], 4)]


@torch.no_grad()
def right_by_game(net, device, by_game, readers=('follower', 'base'), bs=48):
    """Per game of by_game {game: episodes}: the right-answer share (forced's choice groups, per episode then the mean)
    of the trained writers' notes read by the follower and the untrained base model with the adapter off, with own
    notes and another episode's of the game (the next in its chunk); the base model's own minus swapped with its 95%
    interval over episodes; chance."""
    was = net.training
    net.eval()
    out = {}
    for g, eps in by_game.items():
        got = collections.defaultdict(list)
        for c in swap_index(eps, bs):
            part = [eps[i] for i in c]
            bank = net.run(part, device, need_loss=False)[2]
            for reader in readers:
                mode = 'on' if reader == 'follower' else 'base'
                got[f'{reader}_own'] += net.forced(part, device, mode, bank=bank)[1]
                got[f'{reader}_swap'] += net.forced(part, device, mode, bank=shifted(bank))[1]
        r = {k: round(sum(v) / len(v), 4) for k, v in got.items()}
        odds = [1 / len(group) for ep in eps for a, ag in enumerate(ep['agents']) for s in range(len(ag['segments']))
                for group in CG.choices(ep, a, s)]
        r.update(chance=round(sum(odds) / len(odds), 4), episodes=len(eps))
        if 'base' in readers:
            r['base_ci'] = boot_interval([x - y for x, y in zip(got['base_own'], got['base_swap'])])
        out[g] = r
    if was:
        net.train()
    return out


def leak(one):
    """What one game's one-note read (one_note_by_game) counts against the best weights' score."""
    if not one:
        return 0.
    return max(0., one['own'] - one['swap'], one['moved'] if one['moved_ci'][0] > 0 else 0.)


@torch.no_grad()
def one_note_by_game(net, device, by_game, bs=48, most=None):
    """Per game of by_game {game: episodes}: the untrained base model (the adapter off) reading the trained writers'
    notes one at a time as text (one_note.py), up to most notes: the share right on own notes and on another episode's
    of the game (the next in its chunk), own minus swapped with its 95% interval over episodes (ci), one_note.moved with
    its own (moved_ci), the true value's mean margin gain on own notes in nats, chance, the notes read."""
    was = net.training
    net.eval()
    most = most or ONE_NOTES
    out = {}
    for g, eps in by_game.items():
        per, n = [], 0
        for c in swap_index(eps, bs):
            part = [eps[i] for i in c]
            bank = net.run(part, device, need_loss=False)[2]
            notes = [ON.written(net, bank, b, ep) for b, ep in enumerate(part)]
            for b, ep in enumerate(part):
                right, margin = [[], []], [[], []]
                for ask in ON.asks(ep):
                    if n >= most:
                        break
                    n += 1
                    with net.model.disable_adapter():
                        for i, note in enumerate((notes[b], notes[(b + 1) % len(part)])):
                            pick, m = ON.read(net.model, net.tok, ask, ON.note_of(note, ask), device)
                            right[i].append(float(pick == ask['gold']))
                            margin[i].append(m)
                if right[0]:
                    per.append((sum(right[0]) / len(right[0]), sum(right[1]) / len(right[1]), ON.moved(*margin),
                                sum(o - s for o, s in zip(*margin)) / len(margin[0])))
        if per:
            own, swap, shift, nats = zip(*per)
            out[g] = dict(own=round(sum(own) / len(per), 4), swap=round(sum(swap) / len(per), 4),
                          ci=boot_interval([x - y for x, y in zip(own, swap)]), moved=round(sum(shift) / len(per), 4),
                          moved_ci=boot_interval(list(shift)), nats=round(sum(nats) / len(per), 3), chance=ON.chance(eps),
                          notes=n)
    if was:
        net.train()
    return out


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
    ap.add_argument('--con_negatives', type=int, default=1,
                    help='sym arm: other episodes per contrastive row, the next ones in the pass (0: all of them)')
    ap.add_argument('--letters_from', type=int, default=LETTERS[0], help='sym arm: the step the letter penalty starts')
    ap.add_argument('--every', type=int, default=100, help='steps between dev evaluations')
    ap.add_argument('--lam0', type=float, default=LETTERS[1], help='sym arm: the letter weight it starts at')
    ap.add_argument('--gap0', type=float, help='sym arm: the reference gap (default: the last dev before the penalty)')
    ap.add_argument('--lr', type=float, default=2e-4, help='LoRA learning rate (peak; warmup then cosine)')
    ap.add_argument('--cut0', type=float, help='sym arm: a fixed reference cut NLL for the gap (see train)')
    ap.add_argument('--gap_ref', default='first', choices=('first', 'max'),
                    help='sym arm: the reference gap: the last dev before the penalty, or the largest so far')
    ap.add_argument('--dev0', action='store_true', help='sym arm: a dev before training (the untrained notes)')
    ap.add_argument('--gap_by', default='swap', choices=('swap', 'cut'),
                    help='sym arm: the gap the controller and the privacy gate use (see train); both are logged')
    ap.add_argument('--letters_by', default='max', choices=('max', 'all'),
                    help='sym arm: the letter share the controller, the reference rule and the privacy gate use: the '
                         'dev game with the most letters (max) or pooled over every note (all); both are logged')
    ap.add_argument('--note_hint', default='', help='sym arm: a line added to every writer\'s system prompt')
    ap.add_argument('--note_pass', default='soft', choices=('soft', 'greedy', 'one'),
                    help='sym arm: what a note token sees of the note so far in training (soft tokens, or the argmax '
                         'tokens: greedy in m passes, one: decoded without gradient, then one pass with it)')
    ap.add_argument('--lam_rule', default=','.join(map(str, LAM_RULE[1:])),
                    help='sym arm: up,down: the letter weight\'s factor after a dev with the gap held, and without')
    ap.add_argument('--lost', type=int, default=0, help='sym arm: stop after this many devs in a row without the gap')
    ap.add_argument('--private_steps', type=int, default=0, help='sym arm: stop this many steps after privacy starts')
    ap.add_argument('--protect', action='store_true', help='sym arm: past the gate, right-answer numbers at every dev, '
                    'the best weights kept and gone back to when the code is lost, letter weights per game (train)')
    ap.add_argument('--frozen_adv', type=float, default=0., help='sym arm, --protect: the penalty on the untrained base '
                    'model alone, only once it reads the notes above noise (train)')
    ap.add_argument('--one_note', action='store_true', help='sym arm, --protect: the privacy bar is the untrained base model '
                    'reading one note at a time in every game, logged at every dev, counted against the best weights\' '
                    'score, and what --frozen_adv charges, per game once it reads above noise; no game\'s follower may '
                    'fall under half its peak in the best weights, or under a quarter at all (train, one_note.py)')
    ap.add_argument('--choice', type=float, default=0., help='sym arm: weight of -log p(gold among a say\'s candidates)')
    ap.add_argument('--gate_held', type=int, default=1, help='sym arm: devs in a row with the gap held before privacy')
    ap.add_argument('--lam_min', type=float, default=0., help='sym arm: the letter weight never falls below this')
    ap.add_argument('--ref_letters', type=float, default=None,
                    help='sym arm, gap_ref max: only devs under this share of letters raise the reference gap')
    ap.add_argument('--gate_distinct', type=int, default=0,
                    help="sym arm: privacy waits for at least this many different notes among a dev's example notes")
    ap.add_argument('--gate_letters', type=float, default=PRIV_LETTERS,
                    help='sym arm: privacy waits for the decoded notes to be under this share of letters')
    ap.add_argument('--tau0', type=float, default=SYM_TAU[0], help='sym arm: the notes\' starting temperature')
    ap.add_argument('--beta_full', action='store_true', help='sym arm: the entropy terms at full weight from step 1')
    ap.add_argument('--detach_until', type=int, default=0,
                    help='sym arm: up to this step (the warm-up) writers write with the adapter off, so the notes '
                         'are the base model\'s and only the readers learn them; the entropy terms are off and the '
                         'letter controller starts after it')
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
    ap.add_argument('--continue_from', help='sym arm: a checkpoint (gate.pt, trained.pt) whose log.jsonl sits beside '
                                            'it: train on from its step and controller state (see train, resume)')
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
        net.contrast, net.con_negatives = args.contrast, args.con_negatives
        net.choice = args.choice
        net.one_note = args.one_note
        net.hint = args.note_hint
        net.note_pass = args.note_pass
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
    state = None
    if args.continue_from and not args.resume:
        got = net.load_state_dict(torch.load(args.continue_from, map_location=args.device)['weights'], strict=False)
        if got.unexpected_keys:
            raise SystemExit(f'{args.continue_from}: weights not in the net: {got.unexpected_keys[:5]}')
        state = resumed(args.continue_from)
        print(json.dumps(dict(continue_from=args.continue_from, **{k: v for k, v in state.items() if k != 'record'})),
              flush=True)
    if args.resume:
        saved = torch.load(out / 'trained.pt', map_location=args.device)
        net.load_state_dict(saved['weights'], strict=False)
        ran, record = saved['steps_run'], saved['dev_nll']
    else:
        ran, record = train(net, args.device, args.steps, args.batch, args.seed, games, out, micro=args.micro,
                            min_steps=args.min_steps, weights=weights, link_lr=args.link_lr, curriculum=curriculum,
                            every=args.every, letters_from=args.letters_from, lam0=args.lam0, gap0=args.gap0,
                            lam_rule=tuple(float(x) for x in args.lam_rule.split(',')), lost=args.lost,
                            private_steps=args.private_steps, tau0=args.tau0, beta_full=args.beta_full, lr=args.lr,
                            cut0=args.cut0, gap_ref=args.gap_ref, dev0=args.dev0, gap_by=args.gap_by,
                            detach_until=args.detach_until, gate_held=args.gate_held,
                            gate_distinct=args.gate_distinct, gate_letters=args.gate_letters, lam_min=args.lam_min,
                            ref_letters=args.ref_letters, resume=state, letters_by=args.letters_by,
                            protect=args.protect, frozen_adv=args.frozen_adv)
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
