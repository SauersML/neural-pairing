"""Two agents that hold the two halves of an open-ended, multi-turn task (swarm/opendata.py episodes) and talk in real
vocabulary tokens before each answer, trained to answer as a frozen teacher that saw everything (its stored top-32
log-probabilities), token by token, while their messages are pushed off letters.

The agents are one Qwen3 (--base, Qwen3-1.7B for development) with one LoRA; the teacher is the same base, frozen,
whose answers and top-32 opendata.py stored, so training never runs it. An episode is up to 6 task turns. At task
turn k the user's request goes to one agent (who); the two exchange talk_k messages of m tokens (who first, then
turn about); then who answers. The other agent sees neither the request nor the answer. Each agent's transcript (the
Qwen3 chat template, thinking off):
  system: the two-agent rules            user: its private material (opendata's view)
  per task turn: user: the request (or that the partner has one), then the messages "\\nA: ..." "\\nB: ..."
                 assistant (who only): the answer, teacher-forced
Message tokens: any vocabulary token but special and control ones. A letter is a token whose text has a character of
Unicode category L*, or whose bytes are not complete UTF-8 and can be part of a letter's encoding (the first half of
an emoji cannot, so emoji stay open). When the letter controller starts, both system prompts gain SYMBOLS ("write your
messages ... in emoji, symbols and digits only"), which moves the prior onto the symbol vocabulary as letters start
to cost.

One training step, per micro-batch:
  decode   the messages greedily with the key-value cache and no gradient, the letter bias (bias) taken off every
           letter's logit: the hard tokens, which are Jacobi decoding's fixed point, so no refinement passes run
  writer   one gradient pass over each agent's transcript with those tokens as inputs; at each of its message positions
           the soft token softmax((logits - bias * letter) / tau) over the allowed vocabulary times the embedding
           matrix (its argmax is the decoded token)
  reader   one gradient pass over each agent's transcript with every received message's tokens replaced by the
           sender's soft tokens, scored at its answer tokens: the KL from the teacher's top-32 (exact on {top-32,
           rest}); fill-in turns (real text) by cross-entropy. The answer's gradient reaches the writer through the
           soft tokens
Plus: lam times the mean softmax mass on letters per message position (the letter controller sets lam and the bias,
bias = --bias_k * lam); a per-token contrast hinge contrast * relu(target - (KL_swap - KL_on)), KL_swap the same
reader with the received messages of the next batch episode (a third pass); once letters are under --priv_letters with
the gap held, privacy: lam_p * relu(KL_none - KL_on) of the untrained model (adapter off) in the reader's seat, so the
messages must not help a reader that was not trained with them.

Dev (every --dev_every steps; --dev_n episodes per family, carved from the training records and never trained on),
hard messages only, per family: KL per answer token with the messages (on), with the received messages of another
dev episode of the family (swap) and with no conversation at all (none); the share the messages close,
(none - on) / none, overall and by task-turn index; letters, distinct tokens, 3 example messages and the 20 most used
message tokens. The controller's
gap is the mean over families of swap - on. The final evaluation runs the same on eval_seen and eval_heldout (unseen
sources) and adds the untrained model as a reader (base_on, base_none: its gain from reading the messages).

Usage: python swarm/opentalk.py --data DIR --out OUT --ckpt CKPT [--base Qwen/Qwen3-1.7B --m 32 --steps 1500]
Writes OUT/log.jsonl (also printed), OUT/scores.json and CKPT/adapter.
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import opendata as O                        # noqa: E402

BASE = 'Qwen/Qwen3-1.7B'
ASSIST = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
SYSTEM = ('You are agent {x}. You and agent {y} are two assistants who serve one user together. Each of you holds '
          'private material that the other cannot see, and each request from the user comes to one of you. Before '
          'each answer you and agent {y} exchange short messages: write to agent {y} what it needs to know. When a '
          'request comes to you, answer the user directly.')
SYMBOLS = 'From now on, write your messages to your partner in emoji, symbols and digits only, with no letters.'
CHUNK = 512                                            # positions per checkpointed vocabulary chunk
_letters = {}
_fragments = []


def other(x):
    return 'A' if x == 'B' else 'B'


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


# --------------------------------------------------------------------------------------------------------- letters

def byte_chars():
    """Byte-level BPE's map from each byte to the character that stands for it in the vocabulary (GPT-2's)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + \
        list(range(ord('\xae'), ord('\xff') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def letter_fragments():
    """Every byte string that occurs inside the UTF-8 encoding of a letter (a character of category L*) without being
    the whole encoding, over all code points."""
    if not _fragments:
        subs = set()
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF or not unicodedata.category(chr(cp)).startswith('L'):
                continue
            b = chr(cp).encode()
            subs.update(b[i:j] for i in range(len(b)) for j in range(i + 1, len(b) + 1) if j - i < len(b))
        _fragments.append(subs)
    return _fragments[0]


def fragment_is_letter(b, subs):
    """Whether bytes b that are not complete UTF-8 can be part of a letter: their leading continuation bytes (the end
    of a character the previous token began) or their unfinished last character (which the next token ends) occur
    inside some letter's encoding, or their complete middle holds a letter, or the middle is not UTF-8 at all. The
    prefix F0 9F of every 4-byte emoji, for one, is no letter's part."""
    k = 0
    while k < len(b) and (b[k] & 0xC0) == 0x80:
        k += 1
    head, rest, m = b[:k], b[k:], len(b) - k
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
    return any(unicodedata.category(c).startswith('L') for c in text) or head in subs or tail in subs


def letter_sets(tok, V):
    """(allowed, letter): bool masks over V vocabulary rows. Not allowed: special and added tokens, rows past the
    tokenizer, tokens with a control character other than tab, newline or carriage return. Letter: a character of
    category L*, or bytes that are not complete UTF-8 and can be part of a letter (fragment_is_letter)."""
    key = (getattr(tok, 'name_or_path', ''), len(tok), V)
    if key in _letters:
        return _letters[key]
    dec = {c: b for b, c in byte_chars().items()}
    subs = letter_fragments()
    special = set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}))
    allowed, letter = torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.bool)
    for s, tid in tok.get_vocab().items():
        if tid >= V or tid in special:
            continue
        try:
            b = bytes(dec[c] for c in s)
        except KeyError:
            allowed[tid] = letter[tid] = True
            continue
        try:
            text = b.decode('utf-8')
        except UnicodeDecodeError:
            allowed[tid], letter[tid] = True, fragment_is_letter(b, subs)
            continue
        if not text or any(unicodedata.category(c) == 'Cc' and c not in '\t\n\r' for c in text):
            continue
        allowed[tid] = True
        letter[tid] = any(unicodedata.category(c).startswith('L') for c in text)
    _letters[key] = (allowed, letter)
    return allowed, letter


def token_label(tok, i):
    """A vocabulary token as text, or as its bytes in hex when they are not complete UTF-8 (an emoji's first half)."""
    dec = {c: b for b, c in byte_chars().items()}
    s = tok.convert_ids_to_tokens(int(i)) or ''
    try:
        return bytes(dec[c] for c in s).decode('utf-8')
    except (KeyError, UnicodeDecodeError):
        return 'x' + bytes(dec.get(c, 63) for c in s).hex()


# ------------------------------------------------------------------------------------------------------- the model

class Cache:
    """A batch of rows run chunk by chunk with one key-value cache: flush(tails) appends each row's own ids (right
    padded, the padding masked and not counted in positions) and returns each row's hidden state at its last new
    token [R, d] (rows with no new ids: their previous one's, unused)."""

    def __init__(self, net, R, lm=None):
        self.net, self.R, self.lm = net, R, lm or net.lm
        self.cache = self.att = None
        self.n = torch.zeros(R, dtype=torch.long, device=net.device)
        self.tokens = 0

    def flush(self, tails):
        L = max(1, max(len(t) for t in tails))
        ids = torch.full((self.R, L), self.net.pad, dtype=torch.long)
        a = torch.zeros(self.R, L, dtype=torch.long)
        for r, t in enumerate(tails):
            if t:
                ids[r, :len(t)] = torch.tensor(t)
                a[r, :len(t)] = 1
        ids, a = ids.to(self.net.device), a.to(self.net.device)
        full = a if self.att is None else torch.cat([self.att, a], 1)
        pos = self.n[:, None] + (a.cumsum(-1) - 1).clamp(min=0)
        out = self.lm.model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=self.cache,
                            use_cache=True)
        self.cache, self.att = out.past_key_values, full
        self.n = self.n + a.sum(1)
        self.tokens += int(a.sum())
        last = (a.sum(1) - 1).clamp(min=0)
        return out.last_hidden_state[torch.arange(self.R, device=ids.device), last]


class Talk:
    """The LoRA model, its tokenizer, the vocabulary masks and the message rules (m tokens, bias, tau)."""

    def __init__(self, model, tok, m=32):
        self.model, self.tok, self.m = model, tok, m
        V = self.lm.lm_head.weight.shape[0]
        allowed, letter = letter_sets(tok, V)
        self.allowed, self.letter = allowed.to(self.device), letter.to(self.device)
        self.letter_f = self.letter.float()
        self.bias, self.tau = 0.0, 1.0
        self.symbols = False                           # SYMBOLS joins both system prompts (the controller sets it)
        self.pad = tok.convert_tokens_to_ids('<|endoftext|>')
        self._ids = {}
        self.stats = dict(decode_tokens=0, grad_tokens=0)
        self.base_lm = None                            # the untrained model as a separate frozen copy (privacy)
        self.dec = None                                # a plain copy holding base + LoRA merged, for decoding

    def sync_dec(self):
        """Writes base weight + scaling * B @ A of every LoRA layer into the decoding copy (after each optimizer
        step): greedy decoding then runs plain linear layers, without the adapter's extra kernels."""
        if self.dec is None:
            return
        mods = dict(self.dec.named_modules())
        with torch.no_grad():
            for name, mod in self.model.named_modules():
                if hasattr(mod, 'base_layer') and hasattr(mod, 'lora_A') and 'default' in mod.lora_A:
                    tgt = mods[name.removeprefix('base_model.model.')]
                    A, B = mod.lora_A['default'].weight.float(), mod.lora_B['default'].weight.float()
                    w = mod.base_layer.weight.float() + mod.scaling['default'] * (B @ A)
                    tgt.weight.copy_(w.to(tgt.weight.dtype))

    @property
    def lm(self):
        return self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model

    @property
    def device(self):
        return self.lm.model.embed_tokens.weight.device

    @property
    def E(self):
        return self.lm.model.embed_tokens.weight

    def ids(self, text, cache=True):
        if not cache:
            return self.tok(text, add_special_tokens=False).input_ids
        if text not in self._ids:
            self._ids[text] = self.tok(text, add_special_tokens=False).input_ids
        return self._ids[text]

    def base(self):
        """The untrained model (the adapter off), as a context."""
        return self.model.disable_adapter() if hasattr(self.model, 'disable_adapter') else O._Null()

    # ---------------------------------------------------------------------------------------------- transcripts

    def prefix(self, ep, x):
        view = ep['views'][x]
        mat = f'Your private material:\n{view}' if view else 'You hold no private material.'
        rules = SYSTEM.format(x=x, y=other(x)) + (' ' + SYMBOLS if self.symbols else '')
        return self.ids(f'<|im_start|>system\n{rules}<|im_end|>\n') + \
            self.ids(f'<|im_start|>user\n{mat}<|im_end|>\n', cache=False)

    def request(self, ep, k, x, talk=True):
        t, y = ep['turns'][k], other(x)
        if x == t['who']:
            ask = (f'Write part {t["user"]} of the text, exactly as it reads, and nothing else.'
                   if ep['family'] == 'fill' else t['user'])
            head = f'<|im_start|>user\nRequest from the user:\n{ask}'
            tail = f'\n\nYour messages with agent {y} before you answer:'
        else:
            head = f'<|im_start|>user\nAgent {y} has a request from the user that you cannot see.'
            tail = f'\n\nYour messages with agent {y} before it answers:'
        return self.ids(head, cache=x != t['who']) + (self.ids(tail) if talk else [])

    def close(self, ep, k, x):
        """After turn k's messages: the answerer's answer header, answer and end; the other's end of turn."""
        t = ep['turns'][k]
        if x != t['who']:
            return self.ids('<|im_end|>\n')
        r = list(t['response_ids'])
        return self.ids(ASSIST) + r + (self.ids('\n') if r and r[-1] == O.IM_END else self.ids('<|im_end|>\n'))

    def mark(self, s):
        return self.ids(f'\n{s}:')

    @staticmethod
    def speaker(ep, k, j):
        w = ep['turns'][k]['who']
        return w if j % 2 == 0 else other(w)

    def build(self, ep, x, msgs, mode='on', recv=None):
        """Agent x's transcript: dict(ids, own=[(pos, g)], recv=[(pos, g)], ans=[(k, pos, n)]). msgs: the episode's
        messages per turn, lists of (speaker, ids); g counts the episode's messages in order. mode none: no
        conversation. recv: ids to put in place of the received messages, in order (cycled; hard swaps)."""
        ids, own, got, ans, g, r = list(self.prefix(ep, x)), [], [], [], 0, 0
        for k, t in enumerate(ep['turns']):
            ids += self.request(ep, k, x, talk=mode != 'none')
            for s, m in (msgs[k] if mode != 'none' else []):
                ids += self.mark(s)
                if s == x:
                    own.append((len(ids), g))
                    ids += m
                else:
                    got.append((len(ids), g))
                    ids += recv[r % len(recv)] if recv else m
                    r += 1
                g += 1
            c = self.close(ep, k, x)
            if x == t['who']:
                ans.append((k, len(ids) + len(self.ids(ASSIST)), len(t['response_ids'])))
            ids += c
        return dict(ids=ids, own=own, recv=got, ans=ans)

    # ------------------------------------------------------------------------------------------------ messages

    def hard_logits(self, h, lm=None):
        z = (lm or self.lm).lm_head(h).float().masked_fill(~self.allowed, float('-inf'))
        return z - self.bias * self.letter_f if self.bias else z

    @torch.no_grad()
    def decode(self, eps):
        """Greedy messages for a batch, both agents' rows in one key-value cache (rows 0..N-1 agent A, N..2N-1 B).
        At each message the speaker's row writes and both rows read each new token. Returns per episode, per turn,
        the messages as (speaker, ids)."""
        was = self.model.training
        self.model.eval()
        N = len(eps)
        row = {('A', i): i for i in range(N)} | {('B', i): N + i for i in range(N)}
        D = self.dec                                   # the merged copy (sync_dec), else the adapter model
        C = Cache(self, 2 * N, D)
        pending = [self.prefix(eps[i], x) for x in 'AB' for i in range(N)]
        out = [[[] for _ in ep['turns']] for ep in eps]
        for k in range(max(len(ep['turns']) for ep in eps)):
            live = [i for i in range(N) if k < len(eps[i]['turns'])]
            for i in live:
                for x in 'AB':
                    pending[row[(x, i)]] += self.request(eps[i], k, x)
            for j in range(max(eps[i]['turns'][k]['talk'] for i in live)):
                spk = {i: self.speaker(eps[i], k, j) for i in live if j < eps[i]['turns'][k]['talk']}
                for i, s in spk.items():
                    for x in 'AB':
                        pending[row[(x, i)]] += self.mark(s)
                h = C.flush(pending)
                pending = [[] for _ in range(2 * N)]
                ei = list(spk)
                sr = torch.tensor([row[(spk[i], i)] for i in ei], device=self.device)
                toks = []
                for t in range(self.m):
                    nxt = self.hard_logits(h[sr], D).argmax(-1).tolist()
                    toks.append(nxt)
                    if t < self.m - 1:
                        tails = [[] for _ in range(2 * N)]
                        for i, v in zip(ei, nxt):
                            tails[row[('A', i)]] = tails[row[('B', i)]] = [v]
                        h = C.flush(tails)
                for q, i in enumerate(ei):
                    msg = [toks[t][q] for t in range(self.m)]
                    out[i][k].append((spk[i], msg))
                    for x in 'AB':                     # tokens 0..m-2 are in the cache already
                        pending[row[(x, i)]].append(msg[-1])
            for i in live:
                for x in 'AB':
                    pending[row[(x, i)]] += self.close(eps[i], k, x)
        self.stats['decode_tokens'] += C.tokens
        self.model.train(was)
        return out

    # ---------------------------------------------------------------------------------------------------- passes

    def _soft(self, h):
        z = self.hard_logits(h)
        p = (z / self.tau).softmax(-1)
        return p.to(self.E.dtype) @ self.E, p @ self.letter_f, z.argmax(-1)

    def chunked(self, fn, h, *rest):
        """fn over h (and rest) in CHUNK-row pieces, each under activation checkpointing when grad is on: the
        [positions, vocabulary] logits are never all kept."""
        outs = []
        for s in range(0, h.shape[0], CHUNK):
            part = [h[s:s + CHUNK]] + [r[s:s + CHUNK] for r in rest]
            if torch.is_grad_enabled():
                outs.append(checkpoint(fn, *part, use_reentrant=False))
            else:
                outs.append(fn(*part))
        if isinstance(outs[0], tuple):
            return tuple(torch.cat(z, 0) for z in zip(*outs))
        return torch.cat(outs, 0)

    def forward(self, seqs, put=None, lm=None):
        """Hidden states [R, T, d] of right-padded id sequences (lm: another model's); put: (rows, positions, vectors
        [n, d]) written over those input embeddings (the soft received messages)."""
        L = lm or self.lm
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        a = torch.zeros(len(seqs), T, dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            a[r, :len(s)] = 1
        ids, a = ids.to(self.device), a.to(self.device)
        x = L.model.embed_tokens(ids)
        if put is not None:
            rows, pos, vec = put
            x = x.index_put((rows, pos), vec.to(x.dtype))
        self.stats['grad_tokens'] += int(a.sum()) if torch.is_grad_enabled() else 0
        return L.model(inputs_embeds=x, attention_mask=a, position_ids=(a.cumsum(-1) - 1).clamp(min=0),
                       use_cache=False).last_hidden_state

    def writer(self, eps, msgs):
        """The writer pass: per episode a dict {g: soft vectors [m, d]} of its messages, the mean letter mass per
        message position and the share of positions whose argmax is the decoded token (1 at the fixed point)."""
        rows, seqs = [], []
        for i, ep in enumerate(eps):
            for x in 'AB':
                b = self.build(ep, x, msgs[i])
                if b['own']:
                    end = b['own'][-1][0] + self.m
                    rows.append((i, x, b))
                    seqs.append(b['ids'][:end])
        h = self.forward(seqs)
        R, P, G, toks = [], [], [], []
        for r, (i, x, b) in enumerate(rows):
            for pos, g in b['own']:
                R += [r] * self.m
                P += list(range(pos - 1, pos - 1 + self.m))
                G.append((i, g))
                toks += b['ids'][pos:pos + self.m]
        hs = h[torch.tensor(R, device=self.device), torch.tensor(P, device=self.device)]
        vec, mass, arg = self.chunked(self._soft, hs)
        fixed = float((arg == torch.tensor(toks, device=self.device)).float().mean())
        soft = [dict() for _ in eps]
        for q, (i, g) in enumerate(G):
            soft[i][g] = vec[q * self.m:(q + 1) * self.m]
        return soft, mass.mean(), fixed, torch.tensor(toks)

    def reader(self, eps, msgs, soft=None, perm=None, mode='on', lm=None):
        """The reader pass: per (episode, turn) answered, the summed per-token loss and the token count (KL from the
        teacher's top-32, or cross-entropy on real text), as tensors [Q] and a list of (episode, turn, agent).
        soft: the writer's vectors (received messages read as the sender's soft tokens); without it the hard ids.
        perm: received messages from episode perm[i] (its messages from the same seat, cycled). mode none: no
        conversation. lm: another model as the reader (the frozen base_lm)."""
        seqs, puts, keys, tgt = [], ([], [], []), [], []
        L = lm or self.lm
        for i, ep in enumerate(eps):
            j = perm[i] if perm is not None else i
            for x in 'AB':
                sent = [(g, m) for g, (s, m) in enumerate_msgs(msgs[j]) if s != x]
                b = self.build(ep, x, msgs[i], mode=mode,
                               recv=[m for _, m in sent] if perm is not None and soft is None else None)
                if not b['ans']:
                    continue
                r = len(seqs)
                seqs.append(b['ids'])
                if soft is not None and mode != 'none':
                    for q, (pos, g) in enumerate(b['recv']):
                        gs = g if perm is None else sent[q % len(sent)][0]
                        puts[0].extend([r] * self.m)
                        puts[1].extend(range(pos, pos + self.m))
                        puts[2].append(soft[j][gs])
                for k, pos, n in b['ans']:
                    keys.append((i, k, x))
                    tgt.append((r, pos, n, ep, k))
        put = None
        if puts[2]:
            put = (torch.tensor(puts[0], device=self.device), torch.tensor(puts[1], device=self.device),
                   torch.cat(puts[2], 0))
        h = self.forward(seqs, put, L)
        kl = lambda h_, ids_, lp_: O.kl_topk(L.lm_head(h_), ids_, lp_)                       # noqa: E731
        ce = lambda h_, t_: F.cross_entropy(L.lm_head(h_).float(), t_, reduction='none')     # noqa: E731
        kl_pos, kl_ids, kl_lp, kl_q, ce_pos, ce_t, ce_q = [], [], [], [], [], [], []
        for q, (r, pos, n, ep, k) in enumerate(tgt):
            span = [(r, p) for p in range(pos - 1, pos - 1 + n)]
            if ep['teacher']:
                ids, lp = ep['topk'][k]
                kl_pos += span
                kl_ids.append(ids.long())
                kl_lp.append(lp.float())
                kl_q += [q] * n
            else:
                ce_pos += span
                ce_t += list(ep['turns'][k]['response_ids'])
                ce_q += [q] * n
        tot = torch.zeros(len(tgt), device=self.device)
        cnt = torch.zeros(len(tgt), device=self.device)
        for pos, q, fn, extra in ((kl_pos, kl_q, kl, (kl_ids, kl_lp)), (ce_pos, ce_q, ce, (ce_t,))):
            if not pos:
                continue
            rr = torch.tensor([p[0] for p in pos], device=self.device)
            pp = torch.tensor([p[1] for p in pos], device=self.device)
            if fn is kl:
                args = (torch.cat(extra[0]).to(self.device), torch.cat(extra[1]).to(self.device))
            else:
                args = (torch.tensor(extra[0], device=self.device),)
            per = self.chunked(fn, h[rr, pp], *args)
            qq = torch.tensor(q, device=self.device)
            tot = tot.index_add(0, qq, per)
            cnt = cnt.index_add(0, qq, torch.ones_like(per))
        return tot, cnt, keys

    def loss(self, eps, lam=0.0, contrast=0.0, target=0.2, lam_p=0.0):
        """One micro-batch's loss and its parts (see the module docstring)."""
        msgs = self.decode(eps)
        soft, mass, fixed, toks = self.writer(eps, msgs)
        tot, cnt, keys = self.reader(eps, msgs, soft)
        on = tot / cnt.clamp(min=1)
        loss = on.mean() + lam * mass
        parts = dict(kl=float(on.detach().mean()), letters_soft=float(mass.detach()), fixed=fixed,
                     letters=float(self.letter[toks.to(self.device)].float().mean()),
                     distinct=len(torch.unique(toks)))
        if contrast and len(eps) > 1:
            perm = list(range(1, len(eps))) + [0]
            ts, cs, _ = self.reader(eps, msgs, soft, perm=perm)
            margin = ts / cs.clamp(min=1) - on
            loss = loss + contrast * (target - margin).clamp(min=0).mean()
            parts['margin'] = float(margin.detach().mean())
        if lam_p:                       # a frozen copy: checkpoint recomputation never sees an adapter switch
            tb, cb, _ = self.reader(eps, msgs, soft, lm=self.base_lm)
            with torch.no_grad():
                tn, cn, _ = self.reader(eps, msgs, mode='none', lm=self.base_lm)
            gain = tn / cn.clamp(min=1) - tb / cb.clamp(min=1)
            loss = loss + lam_p * gain.clamp(min=0).mean()
            parts['gain_base'] = float(gain.detach().mean())
        return loss, parts


def enumerate_msgs(msgs):
    """An episode's messages in order: (g, (speaker, ids))."""
    return list(enumerate(m for turn in msgs for m in turn))


def build(device, base=BASE, rank=16, ckpt=None, m=32, gc=True, base_copy=False, merged=True):
    """The Talk net: base in bf16 (sdpa) with a fresh LoRA on every linear layer (or the one saved at ckpt);
    base_copy: also a frozen copy of the base as base_lm (the privacy reader); merged: a decoding copy (sync_dec)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)

    def load():
        return AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model = load()
    if ckpt is not None and (Path(ckpt) / 'adapter').exists():
        model = PeftModel.from_pretrained(model, str(Path(ckpt) / 'adapter'), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    if gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    net = Talk(model, tok, m=m)
    if base_copy:
        net.base_lm = frozen(load(), gc)
    if merged:
        net.dec = frozen(load(), gc=False).eval()
        net.sync_dec()
    return net


def frozen(lm, gc=True):
    """A model with no trainable parameters, in training mode so that gradient checkpointing runs (no dropout)."""
    for p in lm.parameters():
        p.requires_grad_(False)
    if gc:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    return lm.train()


def save(net, ckpt, extra=None):
    ckpt = Path(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    net.model.save_pretrained(str(ckpt / 'adapter'))
    (ckpt / 'talk.json').write_text(json.dumps(dict(m=net.m, bias=net.bias, tau=net.tau, symbols=net.symbols,
                                                    **(extra or {}))))


# ------------------------------------------------------------------------------------------------------ evaluation

def fit(ep, net, max_len):
    """The episode cut to the turns whose transcripts stay under max_len tokens (estimated); None under one turn."""
    view = max(len(net.ids(ep['views'][x], cache=False)) for x in 'AB')
    n, out = view + 120, []
    for t in ep['turns']:
        n += len(net.ids(t['user'], cache=False)) + 40 + t['talk'] * (net.m + 3) + len(t['response_ids']) + 12
        if n > max_len:
            break
        out.append(t)
    if not out:
        return None
    k = len(out)
    return dict(ep, turns=out, topk={j: v for j, v in ep.get('topk', {}).items() if j < k}, length=n)


@torch.no_grad()
def evaluate(net, eps, bs=8, base=False, examples=3):
    """Hard-message numbers per family (see the module docstring): KL per answer token on, swap and none, the share
    closed overall and by task-turn index, letters, distinct tokens, example messages; base adds the untrained
    model's on and none."""
    fams = {}
    for ep in eps:
        fams.setdefault(ep['family'], []).append(ep)
    res, all_toks = {}, []
    was = net.model.training
    net.model.eval()
    for f, xs in sorted(fams.items()):
        xs = sorted(xs, key=lambda e: e['length'])
        acc = {mode: {} for mode in ('on', 'swap', 'none', 'base_on', 'base_none')}
        toks, ex = [], []
        for s in range(0, len(xs), bs):
            b = xs[s:s + bs]
            msgs = net.decode(b)
            perm = list(range(1, len(b))) + [0]
            runs = [('on', dict()), ('none', dict(mode='none'))]
            if len(b) > 1:
                runs.append(('swap', dict(perm=perm)))
            for mode, kw in runs:
                tot, cnt, keys = net.reader(b, msgs, **kw)
                for (i, k, x), t_, c_ in zip(keys, tot.tolist(), cnt.tolist()):
                    acc[mode][(s + i, k)] = (t_, c_)
            if base:
                with net.base():
                    for mode, kw in (('base_on', dict()), ('base_none', dict(mode='none'))):
                        tot, cnt, keys = net.reader(b, msgs, **kw)
                        for (i, k, x), t_, c_ in zip(keys, tot.tolist(), cnt.tolist()):
                            acc[mode][(s + i, k)] = (t_, c_)
            for m_ in msgs:
                for turn in m_:
                    for spk, ids in turn:
                        toks.append(ids)
                        if len(ex) < examples:
                            ex.append(f'{spk}: ' + net.tok.decode(ids))
        all_toks += toks
        d = {}
        for mode, v in acc.items():
            if v:
                d[mode] = round(sum(t for t, _ in v.values()) / max(1, sum(c for _, c in v.values())), 4)
        if 'on' in d and 'none' in d:
            d['closed'] = round(O.closed_share(d['none'], d['on']), 4)
        if 'swap' in d:
            d['gap'] = round(d['swap'] - d['on'], 4)
        by = {}
        for (q, k), (t, c) in acc['on'].items():
            tn, cn = acc['none'].get((q, k), (0, 0))
            by.setdefault(k, [[0, 0], [0, 0]])
            by[k][0][0] += t
            by[k][0][1] += c
            by[k][1][0] += tn
            by[k][1][1] += cn
        d['closed_by_turn'] = [round(O.closed_share(by[k][1][0] / max(1, by[k][1][1]), by[k][0][0] / max(1, by[k][0][1])),
                                     3) for k in sorted(by)]
        flat = torch.tensor([t for m_ in toks for t in m_])
        d['letters'] = round(float(net.letter.cpu()[flat].float().mean()), 4) if len(flat) else None
        d['distinct'] = int(len(torch.unique(flat)))
        d['n'] = len(xs)
        d['examples'] = ex
        res[f] = d
    net.model.train(was)
    flat = torch.tensor([t for m_ in all_toks for t in m_])
    fam = [v for v in res.values() if 'gap' in v]
    res['_all'] = dict(letters=round(float(net.letter.cpu()[flat].float().mean()), 4), distinct=int(len(torch.unique(flat))),
                       gap=round(mean(v['gap'] for v in fam), 4), closed=round(mean(v['closed'] for v in res.values()
                                                                                    if 'closed' in v), 4))
    if len(flat):                                      # the 20 most used message tokens, with their share
        u, c = torch.unique(flat, return_counts=True)
        top = c.argsort(descending=True)[:20]
        res['_all']['top'] = [[token_label(net.tok, u[j]), round(float(c[j]) / len(flat), 4)] for j in top.tolist()]
    return res


# -------------------------------------------------------------------------------------------------------- training

def gpu_util():
    try:
        return float(torch.cuda.utilization())
    except Exception:
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10).stdout
            return float(out.split()[0])
        except Exception:
            return None


class Util:
    """The GPU's utilization sampled every second in a thread while on (training steps, not devs)."""

    def __init__(self):
        import threading
        self.on, self.xs = False, []
        if torch.cuda.is_available():
            threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            if self.on:
                u = gpu_util()
                if u is not None:
                    self.xs.append(u)
            time.sleep(1.0)

    def mean(self):
        return round(mean(self.xs), 1) if self.xs else None


def batches(stream, net, args, rng):
    """Micro-batches of similar length (padding cut): a window of --bucket steps' episodes sorted by length, cut into
    micro-batches, shuffled."""
    while True:
        win = []
        for ep in stream:
            ep = fit(ep, net, args.max_len)
            if ep:
                win.append(ep)
            if len(win) >= args.batch * args.bucket:
                break
        if not win:
            return
        win.sort(key=lambda e: e['length'])
        chunks = [win[i:i + args.micro] for i in range(0, len(win), args.micro)]
        rng.shuffle(chunks)
        yield from chunks


def letter_ctrl(lam, gap, ref, args):
    """After a dev: the letter weight times --ctrl_up when the gap holds --ctrl_keep of its value at the controller's
    start (ref), else times --ctrl_down; and whether it held."""
    ok = gap >= args.ctrl_keep * ref if ref > 0 else gap >= ref
    return (min(lam * args.ctrl_up, args.lam_max) if ok else lam * args.ctrl_down), ok


def ctrl_ready(gap, gap0, args):
    """The controller starts once the gap is positive and at least --ctrl_start of its step-0 value."""
    return gap > 0 and gap >= args.ctrl_start * gap0


def train(net, args, log):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    data = O.Episodes(args.data, 'train', seed=args.seed, teacher=True, wait_minutes=args.wait_data)
    dev, seen, rest = [], {}, []
    for ep in data.recs:
        f = ep['family']
        if seen.get(f, 0) < args.dev_n and (e := fit(ep, net, args.max_len)):
            dev.append(e)
            seen[f] = seen.get(f, 0) + 1
        else:
            rest.append(ep)
    data.recs, data.pos = rest, 0
    lora = [p for p in net.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50))
    log(dict(train=len(rest), dev=len(dev), dev_families=seen, allowed=int(net.allowed.sum()),
             letters=int(net.letter.sum()), letter_free=int((net.allowed & ~net.letter).sum()),
             trainable=sum(p.numel() for p in lora), cut_at_degenerate=data.cuts))
    feed = batches(data, net, args, rng)
    micros = max(1, args.batch // args.micro)
    lam, ref, ok, gap0, priv, base_gap, lam_p_max = 0.0, None, False, None, None, None, args.lam_p
    started, step, util, t_train = time.monotonic(), 0, Util(), 0.0
    net.model.train()
    while True:
        frac = min(1.0, step / args.steps)
        net.tau = args.tau0 * (args.tau1 / args.tau0) ** frac
        if step % args.dev_every == 0 and args.dev_every > 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.monotonic()
            d = evaluate(net, dev, bs=args.dev_bs)
            gap = d['_all']['gap']
            gap0 = gap if gap0 is None else gap0
            letters = d['_all']['letters']
            if args.ctrl and step >= args.free:
                if ref is None and not ctrl_ready(gap, gap0, args):
                    log(dict(step=step, ctrl_wait=True, gap=gap, at_step0=gap0))
                else:
                    if ref is None:
                        ref, lam, ok = gap, args.lam0, True
                        net.symbols = bool(args.symbols_line)            # the prior moves onto the symbols
                    else:
                        lam, ok = letter_ctrl(lam, gap, ref, args)
                    net.bias = args.bias_k * lam
                    log(dict(step=step, ctrl=True, gap=gap, ref=ref, held=ok, lam=round(lam, 5),
                             bias=round(net.bias, 4), letters=letters, symbols_line=net.symbols))
            if priv is None and args.lam_p and ref is not None and letters < args.priv_letters and ok:
                priv, base_gap = step, gap
                log(dict(step=step, privacy_starts=True, letters=letters, gap=gap))
            elif priv is not None and base_gap > 0 and gap < args.ctrl_keep * base_gap:
                lam_p_max /= 2
                log(dict(step=step, lam_p_halved=round(lam_p_max, 5), gap=gap))
            log(dict(step=step, tau=round(net.tau, 3), lam=lam, dev=d, dev_s=round(time.monotonic() - t0),
                     s=round(time.monotonic() - started)))
            save(net, args.ckpt, dict(step=step, lam=lam))
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if step >= args.steps or (time.monotonic() - started) / 60 > args.max_minutes:
            save(net, args.ckpt, dict(step=step, lam=lam))
            log(dict(step=step, done=True, s=round(time.monotonic() - started)))
            return dict(step=step, lam=lam)
        lam_p = lam_p_max * min(1.0, (step - priv) / args.priv_ramp) if priv is not None else 0.0
        t_step = time.monotonic()
        parts_all, n_eps = [], 0
        util.on = True
        for _ in range(micros):
            eps = next(feed, None)
            if eps is None:
                log(dict(step=step, data_exhausted=True))
                save(net, args.ckpt, dict(step=step, lam=lam))
                return dict(step=step, lam=lam)
            try:
                loss, parts = net.loss(eps, lam, args.contrast, args.contrast_target, lam_p)
                (loss * len(eps) / args.batch).backward()
                parts_all.append(parts)
                n_eps += len(eps)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                loss = None
                torch.cuda.empty_cache()
                args.max_len = int(args.max_len * 0.9)                 # shorter episodes from here on
                log(dict(step=step, oom=True, length=max(e['length'] for e in eps), micro=len(eps),
                         max_len=args.max_len))
                parts_all = []
                break
        util.on = False
        if not parts_all:
            continue
        torch.nn.utils.clip_grad_norm_(lora, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        net.sync_dec()
        step += 1
        t_train += time.monotonic() - t_step
        if step % args.log_every == 0 or step == 20:
            rec = dict(step=step, tau=round(net.tau, 3), lam=lam, lam_p=round(lam_p, 4), eps=n_eps,
                       s_step=round(time.monotonic() - t_step, 2), s=round(time.monotonic() - started))
            for k in parts_all[0]:
                rec[k] = round(mean(p[k] for p in parts_all), 4)
            if torch.cuda.is_available():
                rec['gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            if step == 20:                         # throughput over the training steps so far (devs left out)
                rec.update(tok_s_grad=round(net.stats['grad_tokens'] / t_train),
                           tok_s_decode=round(net.stats['decode_tokens'] / t_train), gpu_util=util.mean(),
                           h_per_1000_steps=round(t_train / step * 1000 / 3600, 2))
            log(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--bucket', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=6144)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--tau0', type=float, default=1.0)
    ap.add_argument('--tau1', type=float, default=0.5)
    ap.add_argument('--free', type=int, default=200)
    ap.add_argument('--ctrl', action='store_true')
    ap.add_argument('--lam0', type=float, default=0.01)
    ap.add_argument('--lam_max', type=float, default=4.0)
    ap.add_argument('--ctrl_up', type=float, default=2.0)
    ap.add_argument('--ctrl_down', type=float, default=0.5)
    ap.add_argument('--ctrl_keep', type=float, default=0.8)
    ap.add_argument('--ctrl_start', type=float, default=0.5)
    ap.add_argument('--bias_k', type=float, default=40.0)
    ap.add_argument('--contrast', type=float, default=1.0)
    ap.add_argument('--contrast_target', type=float, default=0.2)
    ap.add_argument('--lam_p', type=float, default=0.5)
    ap.add_argument('--priv_letters', type=float, default=0.05)
    ap.add_argument('--priv_ramp', type=int, default=200)
    ap.add_argument('--dev_every', type=int, default=200)
    ap.add_argument('--dev_n', type=int, default=32)
    ap.add_argument('--dev_bs', type=int, default=8)
    ap.add_argument('--eval_n', type=int, default=0, help='episodes per eval split (0: all)')
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--max_minutes', type=float, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--parts', default='train,eval')
    ap.add_argument('--symbols_line', type=int, default=1, help='add SYMBOLS to the system prompts at the controller start')
    ap.add_argument('--wait_data', type=float, default=0, help='minutes to wait for new train shards when out of them')
    ap.add_argument('--eval_data', default='', help='the final evaluation\'s data folder (default --data)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.jsonl', 'a')

    def log(rec):
        line = json.dumps(rec, ensure_ascii=False)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    log(dict(args=vars(args)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build(device, args.base, args.rank, args.ckpt if 'train' not in args.parts else None, args.m,
                base_copy='train' in args.parts and args.lam_p > 0)
    state = {}
    if 'train' in args.parts:
        state = train(net, args, log)
    if 'eval' in args.parts:
        meta = Path(args.ckpt) / 'talk.json'
        if meta.exists():
            j = json.loads(meta.read_text())
            net.bias, net.tau = j.get('bias', net.bias), j.get('tau', net.tau)
            net.symbols = j.get('symbols', net.symbols)
        scores = dict(state=state)
        for split in ('eval_seen', 'eval_heldout'):
            eps = [e for r in O.Episodes(args.eval_data or args.data, split, seed=1, teacher=True)
                   if (e := fit(r, net, args.max_len))]
            if args.eval_n:
                eps = eps[:args.eval_n]
            scores[split] = evaluate(net, eps, bs=args.dev_bs, base=True)
            log(dict(eval=split, res=scores[split]))
        (out / 'scores.json').write_text(json.dumps(scores, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
