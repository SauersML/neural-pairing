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
Message tokens: any vocabulary token but special and control ones. Whether a token is a letter depends on the bytes
pending before it in its message (swarm/letters.py): it is one when those bytes and its own complete a character of
Unicode category L*, cannot be read as UTF-8, or leave an unfinished character that could still become a letter. So
both halves of an emoji such as U+1F9F1 (F0 9F A7, then B1) are open, while B1 after C3 is the letter n with tilde.
The decode bias, the soft letter mass and the letters metric all use the mask of each position's pending bytes (in
the writer pass, those of the decoded tokens before it). When the letter controller starts, both system prompts gain SYMBOLS ("write your
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

--letter_charge target: a message pays only for its letter share above a target, so a message at the target owes nothing
and filling it earns nothing (under the other charges a message of filler costs less than one that carries little). The
target falls only while messages really help: outside fill, the last dev's closed share and the training steps' (a
no-message pass under no_grad gives each step's none) are at least --value_bar, and the messages are not constant. The
weight follows the letters (target_step); the bias stays where it is. --family_norm 1: every family's KL counts in shares
of its own none (its none KL at the last dev), so fill's cross-entropy does not swamp small-KL families.
--contrast_ref none: the contrast hinge compares the reader with no messages, not with wrong ones: it only ever lowers
KL_on, so wrong messages earn nothing by hurting; the swap pass stays as a no_grad check.

--format chat: each agent's transcript is a plain chat with no system line and no frame, as close to the teacher's as the
split allows: its view opens its first user turn, a request is a user turn, its own messages are assistant turns and its
partner's are user turns. mode none is then just the answerer's own turns in the real format. --dev_copy 1: each dev adds
the same reader on literal copies of the partner's text (Talk.copy_messages) as the bar the writer has to beat; the
target stays shut until it does, outside fill.

--warm 1: the writer's warm start, before any of the above. It learns by plain cross-entropy to write literal copies
(Talk.warm_loss: policy target), teacher-forced in one pass, with no decode and no reader; the teacher's answer only picks
the targets and the writer never sees it. --warm_reader 1 also trains the reader on the same copies. When a dev finds the
writer within --warm_margin of the head copy's closed share outside fill, the run goes on with the full loss. A damaged
adapter's own none flatters what it closes, so every gate measures against the better of its none and the untrained
model's (closed_fair, copy_POLICY_fair).

--two 1: separate LoRAs for the writer and the reader. In the chat format a message and an answer are both assistant turns
with the same opening, so one LoRA doing both jobs loses whichever has no loss on it: copying alone doubled the answers' none
(dev 500), and KL alone turned messages into answers (dev 1000). Writer: message positions and the decoding copy. Reader:
answer positions. In the full stage the writer runs at --writer_lr of the learning rate, because Adam scales its weak
gradient through the soft tokens up to full-size steps.

--protect with a target run keeps one best checkpoint (<ckpt>_best), by the writer's closed share outside fill, written
only at a new best, in any stage. A dev under --value_keep of it goes back to it. A dev whose share fell by more than
--value_noise since the last dev is not good: the letter target stays shut until a dev that did not fall.

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
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import opendata as O                        # noqa: E402
from swarm import letters as LT                        # noqa: E402

BASE = 'Qwen/Qwen3-1.7B'
ASSIST = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
SYSTEM = ('You are agent {x}. You and agent {y} are two assistants who serve one user together. Each of you holds '
          'private material that the other cannot see, and each request from the user comes to one of you. Before '
          'each answer you and agent {y} exchange short messages: write to agent {y} what it needs to know. When a '
          'request comes to you, answer the user directly.')
USER_HEAD, TURN_END = '<|im_start|>user\n', '<|im_end|>\n'
ANSWER_HEAD = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
TARGET_SLACK = 0.02                                    # target: while open it stays at most this far above the letters
SYMBOLS = 'From now on, write your messages to your partner in emoji, symbols and digits only, with no letters.'
CHUNK = 512                                            # positions per checkpointed vocabulary chunk
SCALE_FLOOR = 0.05                                     # family_norm: the smallest none KL a family is divided by
CHARGE_EPS = 1e-3                                      # letter_charge sqrt: 3% of one token's share of a 32-token message
LEAKY = ('doc', 'pair', 'code', 'chat_doc')            # families whose simulated requests can name hidden details
CLEAN = ('chat_alt', 'fill', 'debug', 'plan')
_letters = {}


def other(x):
    return 'A' if x == 'B' else 'B'


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


# --------------------------------------------------------------------------------------------------------- letters

byte_chars, letter_fragments = LT.byte_chars, LT.letter_fragments


def letter_sets(tok, V):
    """(allowed, letter): bool masks over V vocabulary rows, letter with nothing pending (swarm/letters.py)."""
    key = (getattr(tok, 'name_or_path', ''), len(tok), V)
    if key not in _letters:
        lm = LT.LetterMasks(tok, V)
        _letters[key] = (lm.allowed, lm.mask(b''))
    return _letters[key]


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


def run_layers(lm, x, pos, write, mask, ks, vs):
    """The decoder stack of a Qwen3 lm on input embeddings x [R, L, d] at positions pos [R, L], with a key-value
    cache held per row: layer i's keys and values go into ks[i], vs[i] [R, kv heads, T, head dim] at slots write
    [R, L], and each query attends the slots mask [R, 1, L, Tu] allows among the first Tu. The final norm's output
    [R, L, d]. The lm's own modules (norms, projections, rotary embedding, MLP) do the work; only the cache and the
    attention call are here. The query heads that share a key-value head ride in its query axis (head j * G + g
    reads key-value head j, as in the model's grouped attention), so the attention kernel needs no repeated keys."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    cos, sin = lm.model.rotary_emb(x, pos)
    idx = None
    for i, layer in enumerate(lm.model.layers):
        a = layer.self_attn
        h = layer.input_layernorm(x)
        shp = (*h.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(h).view(shp)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(shp)).transpose(1, 2)
        v = a.v_proj(h).view(shp).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if idx is None:
            B, H, L, D = q.shape
            K, Tu = k.shape[1], mask.shape[-1]
            G = H // K
            idx = write[:, None, :, None].expand(-1, K, -1, D)
            mg = mask.repeat(1, 1, G, 1)                                  # row g * L + l: query l's mask
        ks[i].scatter_(2, idx, k)
        vs[i].scatter_(2, idx, v)
        o = F.scaled_dot_product_attention(q.reshape(B, K, G * L, D), ks[i][:, :, :Tu], vs[i][:, :, :Tu],
                                           attn_mask=mg, scale=a.scaling).reshape(B, H, L, D)
        x = x + a.o_proj(o.transpose(1, 2).reshape(*x.shape[:-1], -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return lm.model.norm(x)


class FastCache:
    """Cache's interface (flush(tails) -> each row's hidden state at its last new token [R, d]) on a preallocated
    key-value cache with one write position per row (row r's tokens go to slots n_r, n_r + 1, ... with no padding
    between chunks), so nothing is copied as it grows. Slot T - 1 takes the padding's writes and no real token.
    Attention reads only the first Tu slots, Tu the least of T, T / 2, T / 4, ... (down to least) that holds every
    row: a short batch does not read the whole cache at every token. A flush of one token per row (every message
    step) replays a CUDA graph captured once per Tu (graph=True, on CUDA): the whole decoder stack as one launch.
    Build one per row count and reuse it: reset() starts a batch."""

    def __init__(self, net, R, T, lm=None, graph=True, least=64):
        self.net, self.R, self.T, self.lm = net, R, T, lm or net.lm
        self.sizes = sorted({T >> k for k in range(32) if (T >> k) >= least} | {T})
        c = self.lm.config
        hd = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
        dt, dev = self.lm.model.embed_tokens.weight.dtype, net.device
        self.ks = [torch.zeros(R, c.num_key_value_heads, T, hd, dtype=dt, device=dev) for _ in range(c.num_hidden_layers)]
        self.vs = [torch.zeros_like(k) for k in self.ks]
        self.ar = torch.arange(T, device=dev)
        self.graph = graph and dev.type == 'cuda'
        self.g = {}                                              # Tu -> (graph, its output)
        self.reset()

    def reset(self):
        self.nl = [0] * self.R                                   # tokens held per row
        self.tokens = 0

    def _run(self, ids, pos, write, Tu=None):
        x = self.lm.model.embed_tokens(ids)
        mask = self.ar[None, None, None, :Tu or self.T] <= write[:, None, :, None]
        return run_layers(self.lm, x, pos, write, mask, self.ks, self.vs)

    def _step(self, ids, pos, write, Tu):
        if not self.graph:
            return self._run(ids, pos, write, Tu)
        if not self.g:
            self.sid, self.spos, self.swr = ids.clone(), pos.clone(), write.clone()
        if Tu not in self.g:                                     # warm up on a side stream, then capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    self._run(self.sid, self.spos, self.swr, Tu)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = self._run(self.sid, self.spos, self.swr, Tu)
            self.g[Tu] = g, out
        g, out = self.g[Tu]
        self.sid.copy_(ids)
        self.spos.copy_(pos)
        self.swr.copy_(write)
        g.replay()
        return out

    def flush(self, tails):
        tails = list(tails) + [[] for _ in range(self.R - len(tails))]
        L = max(1, max(len(t) for t in tails))
        lens = [len(t) for t in tails]
        if max(n + k for n, k in zip(self.nl, lens)) >= self.T - 1:
            raise OverflowError('a row outgrew the fast cache')
        ids = torch.full((self.R, L), self.net.pad, dtype=torch.long)
        write = torch.full((self.R, L), self.T - 1, dtype=torch.long)
        pos = torch.zeros(self.R, L, dtype=torch.long)
        for r, t in enumerate(tails):
            if t:
                ids[r, :len(t)] = torch.tensor(t)
                write[r, :len(t)] = pos[r, :len(t)] = torch.arange(self.nl[r], self.nl[r] + len(t))
        dev = self.net.device
        ids, pos, write = ids.to(dev), pos.to(dev), write.to(dev)
        need = max(n + k for n, k in zip(self.nl, lens))
        h = (self._step(ids, pos, write, next(t for t in self.sizes if t >= need)) if L == 1 else
             self._run(ids, pos, write, need))
        self.nl = [n + k for n, k in zip(self.nl, lens)]
        self.tokens += sum(lens)
        last = torch.tensor([max(0, k - 1) for k in lens], device=dev)
        return h[torch.arange(self.R, device=dev), last]


def is_filler(text):
    """A token's text that is only whitespace, or an ellipsis ('…' or two or more dots) with whitespace around it."""
    s = text.strip()
    return not s or (set(s) <= set('.…') and (len(s) > 1 or s == '…'))


class Talk:
    """The LoRA model, its tokenizer, the vocabulary masks and the message rules (m tokens, bias, tau)."""

    def __init__(self, model, tok, m=32):
        self.model, self.tok, self.m = model, tok, m
        V = self.lm.lm_head.weight.shape[0]
        self.masks = LT.LetterMasks(tok, V, self.device)      # letters by the bytes pending before a token
        self.allowed, self.letter = self.masks.allowed, self.masks.mask(b'')
        self.letter_f = self.letter.float()
        self._table, self._row = None, {}                  # float letter masks [K, V] per pending prefix
        self.bias, self.tau = 0.0, 1.0
        self.symbols = False                           # SYMBOLS joins both system prompts (the controller sets it)
        self.pad = tok.convert_tokens_to_ids('<|endoftext|>')
        self._ids = {}
        self.stats = dict(decode_tokens=0, grad_tokens=0)
        self._filler = {}                              # token id -> is_filler of its text
        self.charge = 'linear'                         # the letter penalty's shape (letter_charge)
        self.letter_target = 0.0                       # charge target: the letter share a message owes nothing for
        self.family_scale = {}                         # family_norm: family -> the none KL its KL is divided by
        self.contrast_ref = 'swap'                     # what the contrast hinge compares KL_on with
        self.chat = False                              # --format chat: plain chat transcripts (no system line, no frame)
        self.talk_max = 0                              # message turns per task turn at most (fit; 0: as the data says)
        self.warm_reader = False                       # warm_loss also trains the reader on the literal copies
        self.read_model = None                         # --two: the reader's own LoRA model (self.model then only writes)
        self.base_lm = None                            # the untrained model as a separate frozen copy (privacy)
        self.dec = None                                # a plain copy holding base + LoRA merged, for decoding
        self.fast_eps, self.fast_T, self._fast = 0, 0, {}   # the fast decode cache (off: 0 episodes)

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
    def read_lm(self):
        return self.lm if self.read_model is None else self.read_model.get_base_model()

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
        """The untrained model in the reader's seat (the reader's adapter off), as a context."""
        m = self.model if self.read_model is None else self.read_model
        return m.disable_adapter() if hasattr(m, 'disable_adapter') else O._Null()

    # ---------------------------------------------------------------------------------------------- transcripts

    def opens(self, ep, x):
        """Chat format: whether x's view has opened its first user turn (so the next user turn has no header)."""
        return dict(open=self.chat and bool(ep['views'][x]))

    def user_turn(self, state):
        if state and state['open']:
            state['open'] = False
            return []
        return self.ids(USER_HEAD)

    def prefix(self, ep, x):
        view = ep['views'][x]
        if self.chat:
            return self.ids(f'{USER_HEAD}{view}\n\n', cache=False) if view else []
        mat = f'Your private material:\n{view}' if view else 'You hold no private material.'
        rules = SYSTEM.format(x=x, y=other(x)) + (' ' + SYMBOLS if self.symbols else '')
        return self.ids(f'<|im_start|>system\n{rules}<|im_end|>\n') + \
            self.ids(f'<|im_start|>user\n{mat}<|im_end|>\n', cache=False)

    def request(self, ep, k, x, talk=True, state=None):
        t, y = ep['turns'][k], other(x)
        if self.chat and x != t['who']:
            return []
        if x == t['who']:
            ask = (f'Write part {t["user"]} of the text, exactly as it reads, and nothing else.'
                   if ep['family'] == 'fill' else t['user'])
            if self.chat:
                return self.user_turn(state) + self.ids(ask + TURN_END, cache=False)
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
            return [] if self.chat else self.ids('<|im_end|>\n')
        r = list(t['response_ids'])
        return self.ids(self.answer_head()) + r + (self.ids('\n') if r and r[-1] == O.IM_END
                                                   else self.ids('<|im_end|>\n'))

    def answer_head(self):
        return ANSWER_HEAD if self.chat else ASSIST

    def mark(self, s, x=None, state=None):
        """Before a message from s in x's transcript."""
        if self.chat:
            return self.ids(ANSWER_HEAD) if s == x else self.user_turn(state)
        return self.ids(f'\n{s}:')

    def unmark(self):
        """After a message."""
        return self.ids(TURN_END) if self.chat else []

    def held(self, ep, s, k):
        """The ids s holds when it writes at task turn k, oldest first: its view, its earlier requests and the teacher's
        answers to them, and this turn's request if it is s's. Nothing from a later turn."""
        ids = list(self.ids(ep['views'][s], cache=False)) if ep['views'][s] else []
        for j, t in enumerate(ep['turns'][:k + 1]):
            if t['who'] == s:
                ids += self.ids(t['user'], cache=False)
                if j < k:
                    r = list(t['response_ids'])
                    ids += r[:-1] if r and r[-1] == O.IM_END else r
        return ids

    def copy_messages(self, ep, policy='head', L=None):
        """Literal messages in the episode's talk slots: the asker's first slot is its request's first L tokens (its later
        slots empty), and each partner slot takes from what the partner holds by then: head walks L tokens at a time from
        the start, never repeating; best takes the windows covering most of this turn's answer (an oracle: the teacher's
        answer picks them); whole sends all that is new, with no length limit (later slots empty); target is best made
        writable: every message m tokens, padded with newlines, anything outside the allowed vocabulary a newline."""
        L = L or self.m
        res, sent = [], dict(A=0, B=0)
        for k, t in enumerate(ep['turns']):
            who, y = t['who'], other(t['who'])
            src = self.held(ep, y, k)
            slots = [j for j in range(t['talk']) if self.speaker(ep, k, j) == y]
            windows = best_windows(src, t['response_ids'], L, len(slots)) if policy in ('best', 'target') else None
            msgs = []
            for j in range(t['talk']):
                if j % 2 == 0:
                    m = self.ids(t['user'], cache=False)[:L] if j == 0 else []
                    msgs.append((who, list(m)))
                    continue
                if policy == 'head':
                    m = src[sent[y]:sent[y] + L]
                    sent[y] += L
                elif policy in ('best', 'target'):
                    m = windows[slots.index(j)]
                else:
                    m = src[sent[y]:]
                    sent[y] = len(src)
                msgs.append((y, list(m)))
            res.append([(s_, self.padded(m)) for s_, m in msgs] if policy == 'target' else msgs)
        return res

    def padded(self, m):
        """A literal message as exactly m allowed tokens."""
        nl = self.ids('\n')[0]
        keep = [v if self.allowed[v] else nl for v in m[:self.m]]
        return keep + [nl] * (self.m - len(keep))

    @staticmethod
    def speaker(ep, k, j):
        w = ep['turns'][k]['who']
        return w if j % 2 == 0 else other(w)

    def build(self, ep, x, msgs, mode='on', recv=None):
        """Agent x's transcript: dict(ids, own=[(pos, g)], recv=[(pos, g)], ans=[(k, pos, n)]). msgs: the episode's
        messages per turn, lists of (speaker, ids); g counts the episode's messages in order. mode none: no
        conversation. recv: ids to put in place of the received messages, in order (cycled; hard swaps)."""
        ids, own, got, ans, g, r = list(self.prefix(ep, x)), [], [], [], 0, 0
        state = self.opens(ep, x)
        for k, t in enumerate(ep['turns']):
            ids += self.request(ep, k, x, talk=mode != 'none', state=state)
            for s, m in (msgs[k] if mode != 'none' else []):
                ids += self.mark(s, x, state)
                if s == x:
                    own.append((len(ids), g))
                    ids += m
                else:
                    got.append((len(ids), g))
                    ids += recv[r % len(recv)] if recv else m
                    r += 1
                ids += self.unmark()
                g += 1
            c = self.close(ep, k, x)
            if x == t['who']:
                ans.append((k, len(ids) + len(self.ids(self.answer_head())), len(t['response_ids'])))
            ids += c
        return dict(ids=ids, own=own, recv=got, ans=ans)

    # ------------------------------------------------------------------------------------------------ messages

    def rows(self, pends):
        """Row indices into table() for pending prefixes (each prefix's mask is added once)."""
        new = [P for P in dict.fromkeys(pends) if P not in self._row]
        for P in new:
            self._row[P] = len(self._row)
        if new or self._table is None:
            self._table = torch.stack([self.masks.mask(P).float() for P in self._row])
        return torch.tensor([self._row[P] for P in pends], device=self.device)

    def letters_of(self, msgs):
        """Letter flags of message tokens, each message read from its start (pending bytes carried within it)."""
        return [f for ids in msgs for f in self.masks.walk(ids)[0]]

    def message_stats(self, msgs):
        """Over messages (token id lists): letter tokens per message, the share of messages with no letter token, and
        the share of tokens that are filler (is_filler: whitespace and ellipsis only)."""
        counts = [sum(self.masks.walk(ids)[0]) for ids in msgs]
        toks = [t for ids in msgs for t in ids]
        for t in set(toks) - self._filler.keys():
            self._filler[t] = is_filler(self.tok.decode([t]))
        shares = [c / len(ids) for c, ids in zip(counts, msgs)]
        charge = mean(math.sqrt(x + CHARGE_EPS) for x in shares) if self.charge == 'sqrt' else mean(shares)
        n = len(toks)
        entropy = -sum(v / n * math.log(v / n) for v in Counter(toks).values())
        return dict(letters_per_msg=round(mean(counts), 3), no_letter_msgs=round(mean(c == 0 for c in counts), 4),
                    filler=round(mean(self._filler[t] for t in toks), 4), charge=round(charge, 4),
                    eff_tokens=round(math.exp(entropy), 1),
                    distinct_msgs=round(len({tuple(ids) for ids in msgs}) / len(msgs), 4))

    def hard_logits(self, h, lm=None, rows=None):
        """Logits over the allowed vocabulary, the letter bias taken off the letters after each row's pending bytes
        (rows: indices into the table; none: nothing pending)."""
        z = (lm or self.lm).lm_head(h).float().masked_fill(~self.allowed, float('-inf'))
        if not self.bias:
            return z
        return z - self.bias * (self._table[rows] if rows is not None else self.letter_f)

    def decode(self, eps):
        """Greedy messages for a batch (see _decode). With the fast cache on (fast_eps episodes per cache, set by
        --fast_decode), a batch of more episodes runs in pieces of fast_eps, each in the one preallocated cache: an
        episode's messages do not depend on the others in its batch."""
        if not self.fast_eps or len(eps) <= self.fast_eps:
            return self._decode(eps)
        return [m for s in range(0, len(eps), self.fast_eps) for m in self._decode(eps[s:s + self.fast_eps])]

    def cache(self, N, D):
        """The key-value cache for N episodes: the one FastCache of 2 * fast_eps rows (fast_T slots per row) kept
        across batches (rows past 2N stay empty), or a Cache."""
        if not self.fast_eps:
            return Cache(self, 2 * N, D)
        key = (id(D), self.fast_eps, self.fast_T)
        if key not in self._fast:
            self._fast[key] = FastCache(self, 2 * self.fast_eps, self.fast_T, D)
        c = self._fast[key]
        c.reset()
        return c

    def _decode(self, eps):
        """Greedy messages for a batch (_messages) in its key-value cache; a batch that outgrows the fast cache runs
        again in a Cache."""
        D = self.dec                                   # the merged copy (sync_dec), else the adapter model
        C = self.cache(len(eps), D)
        if isinstance(C, FastCache):
            try:
                return self._messages(eps, C, D)
            except OverflowError:
                C = Cache(self, 2 * len(eps), D)
        return self._messages(eps, C, D)

    @torch.no_grad()
    def _messages(self, eps, C, D):
        """Greedy messages for a batch, both agents' rows in one key-value cache C (rows 0..N-1 agent A, N..2N-1 B)
        run by model D. At each message the speaker's row writes and both rows read each new token. Returns per
        episode, per turn, the messages as (speaker, ids)."""
        was = self.model.training
        self.model.eval()
        try:
            return self._messages_in(eps, C, D)
        finally:
            self.model.train(was)

    def _messages_in(self, eps, C, D):
        N = len(eps)
        row = {('A', i): i for i in range(N)} | {('B', i): N + i for i in range(N)}
        pending = [self.prefix(eps[i], x) for x in 'AB' for i in range(N)]
        state = {(x, i): self.opens(eps[i], x) for x in 'AB' for i in range(N)}
        out = [[[] for _ in ep['turns']] for ep in eps]
        for k in range(max(len(ep['turns']) for ep in eps)):
            live = [i for i in range(N) if k < len(eps[i]['turns'])]
            for i in live:
                for x in 'AB':
                    pending[row[(x, i)]] += self.request(eps[i], k, x, state=state[(x, i)])
            for j in range(max(eps[i]['turns'][k]['talk'] for i in live)):
                spk = {i: self.speaker(eps[i], k, j) for i in live if j < eps[i]['turns'][k]['talk']}
                for i, s in spk.items():
                    for x in 'AB':
                        pending[row[(x, i)]] += self.mark(s, x, state[(x, i)])
                h = C.flush(pending)
                pending = [[] for _ in range(2 * N)]
                ei = list(spk)
                sr = torch.tensor([row[(spk[i], i)] for i in ei], device=self.device)
                toks, utf = [], [b''] * len(ei)             # each message starts on a character boundary
                for t in range(self.m):
                    nxt = self.hard_logits(h[sr], D, self.rows(utf)).argmax(-1).tolist()
                    utf = [self.masks.pending(P, v) for P, v in zip(utf, nxt)]
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
                        pending[row[(x, i)]] += [msg[-1]] + self.unmark()
            for i in live:
                for x in 'AB':
                    pending[row[(x, i)]] += self.close(eps[i], k, x)
        self.stats['decode_tokens'] += C.tokens
        return out

    # ---------------------------------------------------------------------------------------------------- passes

    def _soft(self, h, rows):
        z = self.hard_logits(h, rows=rows)
        p = (z / self.tau).softmax(-1)
        return p.to(self.E.dtype) @ self.E, (p * self._table[rows]).sum(-1), z.argmax(-1)

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
        """The writer pass: per episode a dict {g: soft vectors [m, d]} of its messages, the letter mass at each message
        position [messages * m] and the share of positions whose argmax is the decoded token (1 at the fixed point)."""
        hs, G, toks = self.own_hidden(eps, msgs)
        pends = [q for g in range(0, len(toks), self.m) for q in self.masks.walk(toks[g:g + self.m])[1]]
        vec, mass, arg = self.chunked(self._soft, hs, self.rows(pends))
        fixed = float((arg == torch.tensor(toks, device=self.device)).float().mean())
        soft = [dict() for _ in eps]
        for q, (i, g) in enumerate(G):
            soft[i][g] = vec[q * self.m:(q + 1) * self.m]
        return soft, mass, fixed, torch.tensor(toks)

    def own_hidden(self, eps, msgs):
        """One gradient pass over each agent's transcript up to its last message: the hidden states before each of its
        message tokens [messages * m, d], per message (episode, g), and those tokens."""
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
        return h[torch.tensor(R, device=self.device), torch.tensor(P, device=self.device)], G, toks

    def warm_loss(self, eps):
        """The writer's warm start: the cross-entropy of its message positions against literal copies, teacher-forced."""
        msgs = [self.copy_messages(e, 'target') for e in eps]
        hs, _, toks = self.own_hidden(eps, msgs)
        t = torch.tensor(toks, device=self.device)

        def fn(h_, t_):
            z = self.hard_logits(h_)
            return F.cross_entropy(z, t_, reduction='none'), (z.argmax(-1) == t_).float()
        per, hit = self.chunked(fn, hs, t)
        loss = per.mean()
        parts = dict(ce=float(per.detach().mean()), acc=float(hit.mean()), distinct=len(torch.unique(t)),
                     letters=mean(self.letters_of(t.view(-1, self.m).tolist())),
                     acc_first=float(hit.view(-1, self.m)[:, 0].mean()))    # choosing what to send, not continuing it
        if self.warm_reader:
            tot, cnt, keys = self.reader(eps, msgs)
            on = tot / cnt.clamp(min=1)
            loss = loss + (on * self.answer_weights(eps, keys)).mean()
            parts['kl'] = float(on.detach().mean())
        return loss, parts

    def letter_charge(self, mass):
        """The letter penalty from the soft letter mass per message position [messages * m]. linear: the mean, each
        letter token priced the same however few remain, so a message's first words keep paying long after the average
        one stopped. sqrt: the mean over messages of sqrt(letter share + CHARGE_EPS), so the last letters cost the most
        (8 of 32 cost 0.5, not 0.25; 1 costs 0.18, not 0.03); the epsilon bounds the gradient at no letters. target:
        the mean over messages of the share above letter_target, so going below it earns nothing."""
        if self.charge == 'linear':
            return mass.mean()
        if self.charge == 'target':
            return (mass.view(-1, self.m).mean(1) - self.letter_target).clamp(min=0).mean()
        return (mass.view(-1, self.m).mean(1) + CHARGE_EPS).sqrt().mean()

    def reader(self, eps, msgs, soft=None, perm=None, mode='on', lm=None):
        """The reader pass: per (episode, turn) answered, the summed per-token loss and the token count (KL from the
        teacher's top-32, or cross-entropy on real text), as tensors [Q] and a list of (episode, turn, agent).
        soft: the writer's vectors (received messages read as the sender's soft tokens); without it the hard ids.
        perm: received messages from episode perm[i] (its messages from the same seat, cycled). mode none: no
        conversation. lm: another model as the reader (the frozen base_lm)."""
        seqs, puts, keys, tgt = [], ([], [], []), [], []
        L = lm or self.read_lm
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

    def answer_weights(self, eps, keys):
        """Per (episode, turn, agent): 1 over its family's none KL (family_scale; 1 when unset)."""
        return torch.tensor([1 / max(SCALE_FLOOR, self.family_scale.get(eps[i]['family'], 1.0)) for i, _, _ in keys],
                            device=self.device)

    def loss(self, eps, lam=0.0, contrast=0.0, target=0.2, lam_p=0.0):
        """One micro-batch's loss and its parts (see the module docstring)."""
        msgs = self.decode(eps)
        soft, mass, fixed, toks = self.writer(eps, msgs)
        tot, cnt, keys = self.reader(eps, msgs, soft)
        on = tot / cnt.clamp(min=1)
        w = self.answer_weights(eps, keys)
        loss = (on * w).mean() + lam * self.letter_charge(mass)
        rows = toks.view(-1, self.m)
        parts = dict(kl=float(on.detach().mean()), letters_soft=float(mass.detach().mean()), fixed=fixed,
                     letters=mean(self.letters_of(rows.tolist())), distinct=len(torch.unique(toks)),
                     distinct_msgs=len(torch.unique(rows, dim=0)) / max(1, rows.shape[0]))
        perm = list(range(1, len(eps))) + [0]
        if contrast and self.contrast_ref == 'none':
            with torch.no_grad():                    # its own none: sending nothing must earn nothing, and the untrained
                tn, cn, _ = self.reader(eps, msgs, mode='none')  # none is far higher once the reader has trained
                none = tn / cn.clamp(min=1)
                if len(eps) > 1:
                    ts, cs, _ = self.reader(eps, msgs, soft, perm=perm)
                    margin = ts / cs.clamp(min=1) - on
                    parts.update(margin=float(margin.mean()), margin_n=float((margin * w).mean()))
            value = (none - on) * w
            loss = loss + contrast * (target - value).clamp(min=0).mean()
            parts['value'] = float(value.detach().mean())
            keep = torch.tensor([eps[i]['family'] != 'fill' for i, _, _ in keys], device=self.device)
            if keep.any():
                parts['value_nf'] = float(value.detach()[keep].mean())
        elif contrast and len(eps) > 1:
            ts, cs, _ = self.reader(eps, msgs, soft, perm=perm)
            margin = ts / cs.clamp(min=1) - on
            loss = loss + contrast * (target - margin * w).clamp(min=0).mean()
            parts['margin'] = float(margin.detach().mean())
        if lam_p:                       # a frozen copy: checkpoint recomputation never sees an adapter switch
            tb, cb, _ = self.reader(eps, msgs, soft, lm=self.base_lm)
            with torch.no_grad():
                tn, cn, _ = self.reader(eps, msgs, mode='none', lm=self.base_lm)
            gain = tn / cn.clamp(min=1) - tb / cb.clamp(min=1)
            loss = loss + lam_p * (gain * w).clamp(min=0).mean()
            parts['gain_base'] = float(gain.detach().mean())
        return loss, parts


def best_windows(src, answer, L, n):
    """n non-overlapping windows of L tokens of src covering most of the answer's distinct tokens, those rare in src
    counting more."""
    if len(src) <= L:
        return [list(src)] + [[] for _ in range(n - 1)]
    count, want = Counter(src), set(answer)
    weight = {t: 1 / count[t] for t in want if t in count}
    stride = max(1, L // 4)
    scored = sorted(((-sum(weight.get(t, 0.0) for t in set(src[s:s + L])), s)
                     for s in range(0, len(src) - L + stride, stride)))
    picks = []
    for _, s in scored:
        if all(abs(s - q) >= L for q in picks):
            picks.append(s)
        if len(picks) == n:
            break
    return [src[s:s + L] for s in picks] + [[] for _ in range(n - len(picks))]


def enumerate_msgs(msgs):
    """An episode's messages in order: (g, (speaker, ids))."""
    return list(enumerate(m for turn in msgs for m in turn))


def build(device, base=BASE, rank=16, ckpt=None, m=32, gc=True, base_copy=False, merged=True, two=False):
    """The Talk net: base in bf16 (sdpa) with a fresh LoRA on every linear layer (or the one saved at ckpt);
    base_copy: also a frozen copy of the base as base_lm (the privacy reader); merged: a decoding copy (sync_dec);
    two: a separate reader model with its own LoRA (always when ckpt saved one)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)

    def load():
        return AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation='sdpa').to(device)

    def adapted(sub):
        if ckpt is not None and (Path(ckpt) / sub).exists():
            lm = PeftModel.from_pretrained(load(), str(Path(ckpt) / sub), is_trainable=True)
        else:
            lm = get_peft_model(load(), LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        if gc:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
        return lm
    net = Talk(adapted('adapter'), tok, m=m)
    if two or (ckpt is not None and (Path(ckpt) / 'adapter_reader').exists()):
        net.read_model = adapted('adapter_reader')
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


def save(net, ckpt, extra=None, opt=None, state=None):
    """The adapter and talk.json (m, bias, tau, symbols, extra); with opt and state also optim.pt and state.json
    (the controller and the data read so far), so that --resume continues from here."""
    ckpt = Path(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    net.model.save_pretrained(str(ckpt / 'adapter'))
    if net.read_model is not None:
        net.read_model.save_pretrained(str(ckpt / 'adapter_reader'))
    (ckpt / 'talk.json').write_text(json.dumps({**dict(m=net.m, bias=net.bias, tau=net.tau, symbols=net.symbols),
                                                **(extra or {})}))
    if opt is not None:
        torch.save(opt.state_dict(), ckpt / 'optim.pt')
    if state is not None:
        (ckpt / 'state.json').write_text(json.dumps(state))


# ------------------------------------------------------------------------------------------------------ evaluation

def fit(ep, net, max_len):
    """The episode cut to the turns whose transcripts stay under max_len tokens (estimated); None under one turn."""
    view = max(len(net.ids(ep['views'][x], cache=False)) for x in 'AB')
    n, out = view + 120, []
    for t in ep['turns']:
        if net.talk_max:
            t = dict(t, talk=min(t['talk'], net.talk_max))
        n += len(net.ids(t['user'], cache=False)) + 40 + t['talk'] * (net.m + (12 if net.chat else 3)) + \
            len(t['response_ids']) + 12
        if n > max_len:
            break
        out.append(t)
    if not out:
        return None
    k = len(out)
    return dict(ep, turns=out, topk={j: v for j, v in ep.get('topk', {}).items() if j < k}, length=n)


def paired_se(a, b):
    """The standard error of the mean per-token difference a - b over the (episode, turn) keys both hold, each a
    (summed KL, tokens)."""
    diffs = [a[k][0] / a[k][1] - b[k][0] / b[k][1] for k in a.keys() & b.keys() if a[k][1] and b[k][1]]
    if len(diffs) < 2:
        return 0.0
    m = sum(diffs) / len(diffs)
    return round(math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)), 4)


@torch.no_grad()
def evaluate(net, eps, bs=8, base=False, examples=3, readers=(), copies=()):
    """Hard-message numbers per family (see the module docstring): KL per answer token on, swap and none, the share
    closed overall and by task-turn index, letters, distinct tokens, example messages. Untrained readers, never
    trained on anything: base True adds the untrained model's on, swap and none, base 'read' its on and swap; readers
    [(name, lm)] adds other frozen models' on and swap. Each gap (swap - on) comes with its standard error over
    (episode, turn) pairs. copies: policies of Talk.copy_messages whose literal messages the same reader also reads
    (copy_POLICY, copy_POLICY_closed)."""
    fams = {}
    for ep in eps:
        fams.setdefault(ep['family'], []).append(ep)
    res, all_toks = {}, []
    was = net.model.training
    net.model.eval()
    if net.read_model is not None:
        net.read_model.eval()
    for f, xs in sorted(fams.items()):
        xs = sorted(xs, key=lambda e: e['length'])
        acc = defaultdict(dict)
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
                modes = [('base_on', dict())] + ([('base_swap', dict(perm=perm))] if len(b) > 1 else [])
                modes += [('base_none', dict(mode='none'))] if base is True else []
                with net.base():
                    for mode, kw in modes:
                        tot, cnt, keys = net.reader(b, msgs, **kw)
                        for (i, k, x), t_, c_ in zip(keys, tot.tolist(), cnt.tolist()):
                            acc[mode][(s + i, k)] = (t_, c_)
            for pol in copies:
                tot, cnt, keys = net.reader(b, [net.copy_messages(e, pol) for e in b])
                for (i, k, x), t_, c_ in zip(keys, tot.tolist(), cnt.tolist()):
                    acc[f'copy_{pol}'][(s + i, k)] = (t_, c_)
            for name, lm in readers:
                for mode, kw in [(f'{name}_on', dict())] + ([(f'{name}_swap', dict(perm=perm))] if len(b) > 1 else []):
                    tot, cnt, keys = net.reader(b, msgs, lm=lm, **kw)
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
            d['none_on'] = round(d['none'] - d['on'], 4)               # what the right messages buy over none
            for pol in copies:
                d[f'copy_{pol}_closed'] = round(O.closed_share(d['none'], d[f'copy_{pol}']), 4)
            if 'base_none' in d:                                       # against the better none, the honest measure
                ref = min(d['none'], d['base_none'])
                d['closed_fair'] = round(O.closed_share(ref, d['on']), 4)
                for pol in copies:
                    d[f'copy_{pol}_fair'] = round(O.closed_share(ref, d[f'copy_{pol}']), 4)
        for pre in ('', 'base_', *(f'{name}_' for name, _ in readers)):
            if f'{pre}swap' in d and f'{pre}on' in d:
                d[f'{pre}gap'] = round(d[f'{pre}swap'] - d[f'{pre}on'], 4)
                d[f'{pre}gap_se'] = paired_se(acc[f'{pre}swap'], acc[f'{pre}on'])
        if 'base_none' in d and 'base_on' in d:
            d['base_none_on'] = round(d['base_none'] - d['base_on'], 4)
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
        d['letters'] = round(mean(net.letters_of(toks)), 4) if len(flat) else None
        d.update(net.message_stats(toks) if len(flat) else {})
        d['distinct'] = int(len(torch.unique(flat)))
        d['n'] = len(xs)
        d['examples'] = ex
        res[f] = d
    net.model.train(was)
    if net.read_model is not None:
        net.read_model.train(was)
    flat = torch.tensor([t for m_ in all_toks for t in m_])
    fam = [v for v in res.values() if 'gap' in v]
    res['_all'] = dict(letters=round(mean(net.letters_of(all_toks)), 4), distinct=int(len(torch.unique(flat))),
                       gap=round(mean(v['gap'] for v in fam), 4), closed=round(mean(v['closed'] for v in res.values()
                                                                                    if 'closed' in v), 4))
    res['_all'].update(net.message_stats(all_toks) if len(flat) else {})
    for k in ('on', 'swap', 'none', 'none_on'):                 # the means over families the gap is one difference of
        vals = [v[k] for v in res.values() if k in v]
        if vals:
            res['_all'][k] = round(mean(vals), 4)
    for pre in ('', 'base_', *(f'{name}_' for name, _ in readers)):  # each reader's gap and its standard error
        vs = [v for k, v in res.items() if not k.startswith('_') and f'{pre}gap' in v]
        if vs:
            if pre:
                res['_all'][f'{pre}gap'] = round(mean(v[f'{pre}gap'] for v in vs), 4)
            res['_all'][f'{pre}gap_se'] = round(math.sqrt(sum(v[f'{pre}gap_se'] ** 2 for v in vs)) / len(vs), 4)
    for name, group in (('_leaky', LEAKY), ('_clean', CLEAN)):
        vs = {f: res[f] for f in group if f in res and 'closed' in res[f]}
        if vs:
            res[name] = dict(families=sorted(vs), closed=round(mean(v['closed'] for v in vs.values()), 4),
                             gap=round(mean(v['gap'] for v in vs.values() if 'gap' in v), 4)
                             if any('gap' in v for v in vs.values()) else None)
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
    """After a dev: the letter weight times --ctrl_up when the gap holds --ctrl_keep of the reference gap (ref), else
    times --ctrl_down but never under --lam_min (a weight halved after missed devs lets words back into the
    messages); and whether it held."""
    ok = gap >= args.ctrl_keep * ref if ref > 0 else gap >= ref
    return (min(lam * args.ctrl_up, args.lam_max) if ok else max(args.lam_min, lam * args.ctrl_down)), ok


def next_ref(ref, gap, letters, args):
    """The reference gap after a controller dev: --gap_ref first keeps the one at the controller's start; max takes
    the larger, but only from a dev whose messages are under --ref_letters letters (when set), so a dev of words
    cannot set a bar no letterless code holds."""
    if args.gap_ref == 'max' and (args.ref_letters is None or letters < args.ref_letters):
        return max(ref, gap)
    return ref


def ctrl_update(lam, gap, ref, letters, anchored, step, args):
    """One controller dev after its start: (lam, held, ref, anchored). letter_ctrl sets the weight, next_ref the
    reference. --gap_ref anchor: the first dev under --ref_letters letters that misses the reference (set by a dev of
    words) with a positive gap becomes the reference, once (anchored: its step); it stays a miss, so privacy waits
    for a dev that holds against the new reference."""
    lam, ok = letter_ctrl(lam, gap, ref, args)
    if (args.gap_ref == 'anchor' and anchored is None and not ok and gap > getattr(args, 'gap_floor', 0.0)
            and gap > 0 and args.ref_letters is not None and letters < args.ref_letters):
        return lam, ok, gap, step
    return lam, ok, next_ref(ref, gap, letters, args), anchored


def ramped(lam, lam_from, t0, step, ramp):
    """The letter weight at step: from lam_from at step t0 linearly to the controller's lam over ramp steps (a jump
    in the weight and the bias together locks messages into filler within about 30 steps)."""
    if t0 is None or ramp <= 0:
        return lam
    return lam_from + (lam - lam_from) * min(1.0, (step - t0) / ramp)


def target_step(target, lam, letters, opened, half, args):
    """One training step's letter target and weight: while open the target halves every half steps down to
    --target_end and the weight moves by --lam_rate times the letters above (or under) the target, within --lam_min and
    --lam_max; while shut both stay."""
    if not opened:
        return target, lam
    lam = min(args.lam_max, max(args.lam_min, lam + args.lam_rate * (letters - target)))
    return max(args.target_end, target * 0.5 ** (1 / half)), lam


def target_open(value, margin, distinct, dev_value, args, copy_value=None, fell=False):
    """Whether the letter target may fall. Messages help outside fill: the last dev's closed share and the mean over
    the last --target_window steps both reach --value_bar, and the dev's beats a literal copy's (copy_value). Messages
    are not constant: the steps' mean share of distinct messages reaches --min_msgs, and at least half of the help goes
    when another episode's messages are read. fell: the last dev's share fell, so it was not a good dev."""
    if fell or len(value) < value.maxlen or dev_value is None or dev_value < args.value_bar:
        return False
    if copy_value is not None and dev_value <= copy_value:
        return False
    help_ = mean(value)
    return help_ >= args.value_bar and mean(distinct) >= args.min_msgs and mean(margin) >= 0.5 * help_


def value_outside_fill(d, key='closed'):
    """A dev's mean closed share (or key) over its families other than fill (None without any)."""
    vals = [v[key] for f, v in d.items() if not f.startswith('_') and f != 'fill' and key in v]
    return round(mean(vals), 4) if vals else None


def families_alive(d, z):
    """How many of a dev's families have a gap above z standard errors."""
    return sum(v['gap'] > z * v['gap_se'] for f, v in d.items()
               if not f.startswith('_') and 'gap' in v and v.get('gap_se') is not None)


def dev_trouble(gap, ref, eff_tokens, args, alive=None, value=None, peak=None):
    """Why a dev should go back to the best weights ('' if it should not): the messages collapsed onto under
    --collapse_tokens effective tokens; with alive (a target run), fewer than --min_alive families read above noise,
    or the closed share outside fill (value) fell under --value_keep of its best so far (peak); otherwise the gap fell
    under --gap_floor or --lost_keep of the reference."""
    if eff_tokens is not None and eff_tokens < args.collapse_tokens:
        return 'collapsed'
    if alive is not None:
        if alive < args.min_alive:
            return 'dead'
        return 'value' if value is not None and peak and value < args.value_keep * peak else ''
    if ref is not None and gap < max(args.gap_floor, args.lost_keep * ref):
        return 'gap'
    return ''


def ctrl_ready(gap, gap0, args):
    """The controller starts once the gap is positive and at least --ctrl_start of its step-0 value."""
    return gap > 0 and gap >= args.ctrl_start * gap0


def resume_state(path):
    """A checkpoint's talk.json and state.json (state.json: ref, ok, gap0, priv, base_gap, lam_p_max and files, the
    shards the run had read at its start; {} when the checkpoint has none)."""
    R = Path(path)
    st = R / 'state.json'
    return json.loads((R / 'talk.json').read_text()), (json.loads(st.read_text()) if st.exists() else {})


def train(net, args, log):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    talk, st = resume_state(args.resume) if args.resume else ({}, {})
    have = {p.name for p in (Path(args.data) / 'shards').glob('train-*.jsonl.gz')}
    hold = sorted(have - set(st['files'])) if st.get('files') else []     # read later, as the first part did
    data = O.Episodes(args.data, 'train', seed=args.seed, teacher=True, wait_minutes=args.wait_data, hold=hold)
    files = sorted(data.files)
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
    groups = [dict(params=lora)]
    if net.read_model is not None:
        groups.append(dict(params=[p for p in net.read_model.parameters() if p.requires_grad]))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    stage = 'warm' if args.warm else 'full'          # warm: the writer learns to copy; full: the whole loss
    warm = [0]                                       # steps the warmup counts as done (a resumed optimizer)
    lr_scale = [1.0]                                 # halved by each go-back to the best weights
    lr_at = lambda s: min(1, (s + 1 + warm[0]) / 50) * lr_scale[0]                      # noqa: E731

    def schedule():
        if net.read_model is None:
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        return torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: lr_at(s) * (args.writer_lr if stage == 'full' else 1.0),
                                                       lr_at])
    sched = schedule()
    log(dict(train=len(rest), dev=len(dev), dev_families=seen, allowed=int(net.allowed.sum()),
             letters=int(net.letter.sum()), letter_free=int((net.allowed & ~net.letter).sum()),
             trainable=sum(p.numel() for g in groups for p in g['params']), cut_at_degenerate=data.cuts))
    feed = batches(data, net, args, rng)
    micros = max(1, args.batch // args.micro)
    lam, ref, ok, gap0, priv, base_gap, lam_p_max = 0.0, None, False, None, None, None, args.lam_p
    anchored = None
    target_mode = args.letter_charge == 'target'
    target, half, dev_value, copy_value = args.target0, args.target_half, None, None   # target and its halving time
    prev_value, fell = None, False                   # the last dev's share outside fill, and whether it fell to there
    gate_v, gate_m, gate_d, gate_l = (deque(maxlen=args.target_window) for _ in range(4))
    copies = ('head', 'best', 'target') if args.dev_copy else ()
    lam_from, ramp_t0, lam_now = 0.0, None, 0.0      # the weight in force moves from lam_from to lam from ramp_t0
    best, rollbacks, last_dev, force_dev, watch_on = None, 0, None, False, True
    roll_d, roll_m = deque(maxlen=args.watch_steps), deque(maxlen=args.watch_steps)
    out_of_data = False
    started, step, util, t_train = time.monotonic(), 0, Util(), 0.0
    first = 0                                        # the step this part of the run started at
    dec_train = 0                                    # message tokens decoded in training steps (not in devs)
    skip_dev = None

    def load_weights(B):
        """The writer's adapter and the reader's from a checkpoint (the reader starts from the writer's if it saved one)."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        B = Path(B)
        set_peft_model_state_dict(net.model, load_file(str(B / 'adapter' / 'adapter_model.safetensors')))
        if net.read_model is not None:
            sub = 'adapter_reader' if (B / 'adapter_reader').exists() else 'adapter'
            set_peft_model_state_dict(net.read_model, load_file(str(B / sub / 'adapter_model.safetensors')))
        net.sync_dec()

    if args.resume:                                  # the adapter, controller and data position of a checkpoint
        R = Path(args.resume)
        load_weights(R)
        step, lam = int(talk['step']), float(talk.get('lam', 0.0))
        net.bias, net.symbols = float(talk.get('bias', 0.0)), bool(talk.get('symbols', False))
        ref, ok, gap0 = st.get('ref'), bool(st.get('ok', False)), st.get('gap0')
        priv, base_gap, lam_p_max = st.get('priv'), st.get('base_gap'), st.get('lam_p_max', args.lam_p)
        anchored = st.get('anchored')
        lam_from, ramp_t0 = st.get('lam_from', lam), st.get('ramp_t0')
        best, rollbacks, lr_scale[0] = st.get('best'), st.get('rollbacks', 0), st.get('lr_scale', 1.0)
        target, half, dev_value = st.get('target', target), st.get('half', half), st.get('dev_value')
        copy_value, stage = st.get('copy_value'), st.get('stage', stage)
        prev_value, fell = st.get('prev_value', dev_value), st.get('fell', False)
        net.family_scale = st.get('family_scale', net.family_scale)
        optimizer = (R / 'optim.pt').exists() and (R / 'adapter_reader').exists() == (net.read_model is not None)
        if optimizer:
            opt.load_state_dict(torch.load(R / 'optim.pt', map_location=net.device))
        if (R / 'optim.pt').exists():                # a fresh optimizer on trained weights takes no warmup either
            warm[0] = step
        sched = schedule()
        for _ in range(step * micros):               # the micro-batches the first part trained on
            next(feed)
        first = step                                 # its dev at this step has run and set the controller
        skip_dev = None if args.dev_first else step  # (--dev_first: measure again here, say in a new format)
        log(dict(resumed=str(R), step=step, lam=lam, bias=net.bias, symbols=net.symbols, ref=ref, gap0=gap0,
                 optimizer=optimizer, held_shards=hold, data_pos=data.pos))

    def ckpt_state():
        return dict(ref=ref, ok=ok, gap0=gap0, priv=priv, base_gap=base_gap, lam_p_max=lam_p_max, files=files,
                    anchored=anchored, lam_from=lam_from, ramp_t0=ramp_t0, best=best, rollbacks=rollbacks,
                    lr_scale=lr_scale[0], target=target, half=half, dev_value=dev_value, copy_value=copy_value,
                    family_scale=net.family_scale, stage=stage, prev_value=prev_value, fell=fell)

    def go_back(slower=True):
        """The best weights and optimizer state, and the weight, bias and tau they were measured at."""
        nonlocal lam, lam_from, ramp_t0, lam_now, target, half, dev_value, copy_value, prev_value, fell
        B = Path(best['path'])
        load_weights(B)
        opt.load_state_dict(torch.load(B / 'optim.pt', map_location=net.device))
        lam = lam_now = lam_from = best['lam']
        ramp_t0 = None
        net.bias = best['bias']
        if target_mode:                              # letters fell too fast for the messages: half the pace from there
            target, half, dev_value = best['target'], half * 2 if slower else half, best.get('dev_value')
            copy_value, prev_value, fell = best.get('copy_value'), best.get('dev_value'), False
            for q in (gate_v, gate_m, gate_d, gate_l):
                q.clear()

    def save_best(gap, charge, lam_b, bias_b):
        """The one best checkpoint, swapped in whole: a crash while saving leaves the old one."""
        nonlocal best
        path = str(Path(args.ckpt).parent / (Path(args.ckpt).name + '_best'))
        best = dict(step=step, gap=gap, charge=charge, lam=lam_b, bias=bias_b, path=path, target=target,
                    dev_value=dev_value, copy_value=copy_value)
        save(net, path + '.tmp', dict(step=step, lam=lam_b, bias=bias_b), opt, ckpt_state())
        shutil.rmtree(path, ignore_errors=True)
        os.replace(path + '.tmp', path)
        log(dict(step=step, new_best=True, gap=gap, charge=charge, lam=lam_b, value_nf=dev_value))

    if args.resume and target_mode and args.protect and best is None and dev_value is not None:
        save_best(st.get('gap0'), None, lam, net.bias)   # a rollback point from the first step on
    net.model.train()
    if net.read_model is not None:
        net.read_model.train()
    while True:
        frac = min(1.0, step / args.steps)
        net.tau = args.tau0 * (args.tau1 / args.tau0) ** frac
        lam_now = ramped(lam, lam_from, ramp_t0, step, args.ramp)
        net.letter_target = target
        if ref is not None and not target_mode:
            net.bias = args.bias_k * lam_now
        stop = step >= args.steps or (time.monotonic() - started) / 60 > args.max_minutes or out_of_data \
            or rollbacks > args.max_rollbacks
        scheduled = args.dev_every > 0 and step % args.dev_every == 0 and step != skip_dev and step != last_dev
        final = stop and args.protect and args.dev_every > 0 and step != last_dev and step != skip_dev
        if scheduled or force_dev or final:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.monotonic()
            watch_on = scheduled or not force_dev            # a forced dev that finds nothing quiets the watch until
            force_dev, last_dev = False, step                # the next scheduled one
            roll_d.clear()
            roll_m.clear()
            d = evaluate(net, dev, bs=args.dev_bs, base=(not target_mode and 'read') or bool(args.dev_base)
                         if args.dev_base else False, copies=copies)
            gap = d['_all']['gap']
            gap0 = gap if gap0 is None else gap0
            fair = any('closed_fair' in v for f, v in d.items() if not f.startswith('_'))
            dev_value = value_outside_fill(d, 'closed_fair' if fair else 'closed')
            fell = bool(target_mode and prev_value is not None and dev_value is not None
                        and dev_value < prev_value - args.value_noise)
            prev_value = dev_value
            copy_value = value_outside_fill(d, 'copy_head_fair' if fair else 'copy_head_closed')
            alive = families_alive(d, args.read_z)
            d['_all'].update(value_nf=dev_value, copy_head_nf=copy_value, alive=alive,
                             copy_best_nf=value_outside_fill(d, 'copy_best_fair' if fair else 'copy_best_closed'),
                             copy_target_nf=value_outside_fill(d, 'copy_target_fair' if fair else 'copy_target_closed'),
                             none_nf=value_outside_fill(d, 'none'), base_none_nf=value_outside_fill(d, 'base_none'),
                             on_nf=value_outside_fill(d, 'on'))
            switched = (stage == 'warm' and dev_value is not None and copy_value is not None
                        and copy_value >= args.value_bar and dev_value >= copy_value - args.warm_margin)
            if switched:
                stage = 'full'
                log(dict(step=step, warm_done=True, value_nf=dev_value, copy_head_nf=copy_value))
            if args.family_norm:
                net.family_scale = {f: min(v['none'], v.get('base_none', v['none'])) for f, v in d.items()
                                    if not f.startswith('_') and 'none' in v}
            letters = d['_all']['letters']
            lam_dev, bias_dev = lam_now, net.bias
            eff = d['_all'].get('eff_tokens')
            collapsed = bool(args.protect and eff is not None and eff < args.collapse_tokens)
            if args.ctrl and step >= args.free and not collapsed:
                lam_old = lam
                if ref is None and not ctrl_ready(gap, gap0, args):
                    log(dict(step=step, ctrl_wait=True, gap=gap, at_step0=gap0))
                else:
                    if ref is None:
                        ref, lam, ok = gap, args.lam0, True
                        net.symbols = bool(args.symbols_line)            # the prior moves onto the symbols
                        if target_mode:                                  # the bias stays here from now on
                            net.bias = args.bias_k * lam
                    else:
                        lam, ok, ref, anchored = ctrl_update(lam, gap, ref, letters, anchored, step, args)
                        if target_mode:                                  # the weight follows the letters (target_step)
                            lam = lam_old
                    if lam != lam_old:
                        lam_from, ramp_t0 = lam_now, step
                    lam_now = ramped(lam, lam_from, ramp_t0, step, args.ramp)
                    if not target_mode:
                        net.bias = args.bias_k * lam_now
                    log(dict(step=step, ctrl=True, gap=gap, ref=ref, held=ok, lam=round(lam, 5),
                             bias=round(args.bias_k * lam, 4), bias_now=round(net.bias, 4), letters=letters,
                             symbols_line=net.symbols, anchored=anchored))
            read, read_se = d['_all'].get('base_gap'), d['_all'].get('base_gap_se')
            reading = read is None or read > args.read_z * read_se    # the frozen reader reads above noise
            if priv is None and args.lam_p and ref is not None and letters < args.priv_letters and (ok or target_mode):
                if reading:
                    priv, base_gap = step, gap
                    log(dict(step=step, privacy_starts=True, letters=letters, gap=gap, base_gap=read, base_se=read_se))
                else:
                    log(dict(step=step, privacy_not_needed=True, letters=letters, base_gap=read, base_se=read_se))
            elif priv is not None and base_gap > 0 and gap < args.ctrl_keep * base_gap:
                lam_p_max /= 2
                log(dict(step=step, lam_p_halved=round(lam_p_max, 5), gap=gap))
            log(dict(step=step, tau=round(net.tau, 3), lam=lam, lam_now=round(lam_now, 5), dev=d,
                     dev_s=round(time.monotonic() - t0), s=round(time.monotonic() - started)))
            if args.protect and (stage == 'full' or target_mode):
                why = '' if switched or stage != 'full' else dev_trouble(
                    gap, ref, eff, args, alive if target_mode else None, dev_value, best and best.get('dev_value'))
                if why and best is not None and step != first:   # a part's first dev has trained nothing to undo
                    go_back()
                    rollbacks += 1
                    ok, watch_on = False, True
                    lr_scale[0] *= args.rollback_lr
                    for g in opt.param_groups:
                        g['lr'] = args.lr * lr_scale[0]
                    log(dict(step=step, rollback=True, why=why, to=best['step'], gap=gap, eff_tokens=eff, lam=lam,
                             bias=net.bias, lr=args.lr * lr_scale[0], rollbacks=rollbacks, target=target, half=half,
                             alive=alive))
                elif not why:
                    charge = d['_all'].get('charge', letters)
                    if target_mode:
                        better = dev_value is not None and (best is None or best.get('dev_value') is None
                                                            or dev_value > best['dev_value'])
                    else:
                        better = best is None or gap - lam_dev * charge > best['gap'] - lam_dev * best['charge']
                    if better:
                        save_best(gap, charge, lam_dev, bias_dev)
            save(net, args.ckpt, dict(step=step, lam=lam), opt, ckpt_state())
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if stop or step >= args.steps or (time.monotonic() - started) / 60 > args.max_minutes:
            if args.protect and (stage == 'full' or target_mode) and best is not None and best['step'] != step:
                if (target_mode and target < best.get('target', target) and not fell
                        and (Path(args.ckpt) / 'adapter').exists()):
                    load_weights(args.ckpt)                 # letters fell since the best: the last dev's, at its lower letters
                    log(dict(step=step, ends_on_dev=True, best=best['step'], target=target))
                else:
                    go_back(slower=False)                   # ends on its best weights
                    log(dict(step=step, ends_on_best=best['step'], gap=best['gap']))
            save(net, args.ckpt, dict(step=step, lam=lam), opt, ckpt_state())
            log(dict(step=step, done=True, s=round(time.monotonic() - started), rollbacks=rollbacks,
                     best=best and best['step']))
            return dict(step=step, lam=lam)
        lam_p = lam_p_max * min(1.0, (step - priv) / args.priv_ramp) if priv is not None else 0.0
        t_step = time.monotonic()
        parts_all, n_eps = [], 0
        dec0 = net.stats['decode_tokens']
        util.on = True
        for _ in range(micros):
            eps = next(feed, None)
            if eps is None:
                log(dict(step=step, data_exhausted=True))
                out_of_data, parts_all = True, []
                break
            try:
                if stage == 'warm':
                    loss, parts = net.warm_loss(eps)
                else:
                    loss, parts = net.loss(eps, lam_now, args.contrast, args.contrast_target, lam_p)
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
        for g in opt.param_groups:
            torch.nn.utils.clip_grad_norm_(g['params'], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        net.sync_dec()
        step += 1
        t_train += time.monotonic() - t_step
        dec_train += net.stats['decode_tokens'] - dec0
        if args.protect and watch_on and stage == 'full':   # a collapse takes ~30 steps: the batches can call a dev early
            roll_d.append(mean(p['distinct'] for p in parts_all))
            roll_m.append(mean(p.get('margin', 1e9) for p in parts_all))
            if len(roll_d) == args.watch_steps and (mean(roll_d) < args.watch_distinct or (
                    ref is not None and not target_mode and mean(roll_m) < args.watch_margin * ref)):
                force_dev = True
                log(dict(step=step, forced_dev=True, distinct=round(mean(roll_d), 1), margin=round(mean(roll_m), 4)))
        opened = False
        if target_mode and ref is not None and stage == 'full':   # letters fall only while messages help and differ
            for q, k in ((gate_v, 'value_nf'), (gate_m, 'margin_n'), (gate_d, 'distinct_msgs'), (gate_l, 'letters_soft')):
                vals = [p[k] for p in parts_all if k in p]
                if vals:
                    q.append(mean(vals))
            opened = target_open(gate_v, gate_m, gate_d, dev_value, args, copy_value, fell)
            target, lam = target_step(target, lam, mean(p['letters_soft'] for p in parts_all), opened, half, args)
            if opened:
                target = min(target, mean(gate_l) + TARGET_SLACK)
        if step % args.log_every == 0 or step == first + 20:
            rec = dict(step=step, tau=round(net.tau, 3), lam=round(lam_now, 5), lam_p=round(lam_p, 4), eps=n_eps,
                       s_step=round(time.monotonic() - t_step, 2), s=round(time.monotonic() - started))
            if target_mode:
                rec.update(target=round(target, 4), open=opened, help=round(mean(gate_v), 4) if gate_v else None, stage=stage)
            for k in {k for p in parts_all for k in p}:
                rec[k] = round(mean(p[k] for p in parts_all if k in p), 4)
            if torch.cuda.is_available():
                rec['gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            if step == first + 20:                 # throughput over this part's training steps (devs left out)
                rec.update(tok_s_grad=round(net.stats['grad_tokens'] / t_train),
                           tok_s_decode=round(dec_train / t_train), gpu_util=util.mean(),
                           h_per_1000_steps=round(t_train / 20 * 1000 / 3600, 2))
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
    ap.add_argument('--lam_min', type=float, default=0.0, help='the letter weight never falls below this')
    ap.add_argument('--ramp', type=int, default=0, help='steps over which a controller change of the weight and bias '
                    'takes effect, linearly (0: at once)')
    ap.add_argument('--dev_base', type=int, default=1, help='each dev reports the untrained model as a reader, on and '
                    'swapped')
    ap.add_argument('--read_z', type=float, default=3.0, help='privacy starts only if that reader\'s gap is over this '
                    'many standard errors')
    ap.add_argument('--readers', default='', help='comma-separated other untrained models read the messages at the '
                    'final evaluation, on and swapped')
    ap.add_argument('--protect', type=int, default=0, help='save the best weights at each dev; a dev that loses the gap '
                    'or collapses goes back to them at a lower learning rate; the run ends on them')
    ap.add_argument('--lost_keep', type=float, default=0.5, help='protect: a dev under this share of the reference gap '
                    'has lost it')
    ap.add_argument('--gap_floor', type=float, default=0.05, help='protect: a dev under this gap has lost it, and does '
                    'not re-anchor the reference')
    ap.add_argument('--collapse_tokens', type=float, default=8.0, help='protect: a dev whose messages use under this '
                    'many effective tokens has collapsed')
    ap.add_argument('--rollback_lr', type=float, default=0.5, help='protect: the learning rate is multiplied by this at '
                    'each go-back')
    ap.add_argument('--max_rollbacks', type=int, default=3, help='protect: after more go-backs than this the run ends')
    ap.add_argument('--watch_steps', type=int, default=20, help='protect: the training steps the early check averages')
    ap.add_argument('--watch_distinct', type=float, default=40.0, help='protect: a dev comes early if the mean distinct '
                    'tokens per step fall under this')
    ap.add_argument('--watch_margin', type=float, default=0.25, help='protect: or if the mean training margin falls '
                    'under this share of the reference gap')
    ap.add_argument('--letter_charge', default='linear', choices=('linear', 'sqrt', 'target'),
                    help='the letter penalty: mean letter mass, the mean over messages of its square root, or of its '
                    'share above a falling target')
    ap.add_argument('--target0', type=float, default=0.44, help='target: the first letter target')
    ap.add_argument('--target_half', type=float, default=112.0, help='target: steps per halving while open; doubled by '
                    'each go-back')
    ap.add_argument('--target_end', type=float, default=0.01, help='target: the lowest target')
    ap.add_argument('--target_window', type=int, default=30, help='target: the steps the training help is averaged over')
    ap.add_argument('--value_bar', type=float, default=0.15, help='target: the closed share outside fill that opens it')
    ap.add_argument('--value_keep', type=float, default=0.8, help='target: a dev under this share of the best closed share '
                    'outside fill goes back to it')
    ap.add_argument('--value_noise', type=float, default=0.01, help='target: a fall in the closed share outside fill '
                    'since the last dev, past this, means that dev was not good and the target stays shut')
    ap.add_argument('--two', type=int, default=0, help='1: separate LoRAs for the writer and the reader')
    ap.add_argument('--writer_lr', type=float, default=0.1, help='two: the writer\'s share of the learning rate in the '
                    'full stage')
    ap.add_argument('--min_msgs', type=float, default=0.5, help='target: the share of distinct messages per step needed')
    ap.add_argument('--min_alive', type=int, default=4, help='target: fewer families reading above noise goes back')
    ap.add_argument('--lam_rate', type=float, default=0.2, help='target: the weight moves by this times letters - target')
    ap.add_argument('--format', default='agent', choices=('agent', 'chat'),
                    help='chat: plain chat transcripts with no system line (see the module docstring)')
    ap.add_argument('--talk_max', type=int, default=0, help='message turns per task turn at most (0: as the data says)')
    ap.add_argument('--warm', type=int, default=0, help='1: start with the writer\'s warm start (see the module docstring)')
    ap.add_argument('--warm_reader', type=int, default=0, help='warm: 1 also trains the reader on the literal copies')
    ap.add_argument('--warm_margin', type=float, default=0.02, help='warm: how near the head copy the writer must score')
    ap.add_argument('--dev_first', type=int, default=0, help='1: a resumed run starts with a dev')
    ap.add_argument('--dev_copy', type=int, default=0, help='1: each dev also reads literal copies, the bar for the target')
    ap.add_argument('--family_norm', type=int, default=0, help='1: each family\'s KL in shares of its none KL')
    ap.add_argument('--contrast_ref', default='swap', choices=('swap', 'none'),
                    help='what the contrast hinge compares KL_on with (none: wrong messages earn nothing by hurting)')
    ap.add_argument('--gap_ref', default='first', choices=('first', 'max', 'anchor'),
                    help="the reference gap: the controller start's, the largest since (see next_ref), or re-anchored "
                    "once on the first letterless dev that misses it (see ctrl_update)")
    ap.add_argument('--ref_letters', type=float, default=None,
                    help='gap_ref max: only devs under this share of letters raise the reference gap')
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
    ap.add_argument('--resume', default='', help='a checkpoint folder to continue from (adapter, talk.json, state.json, '
                    'optim.pt)')
    ap.add_argument('--fast_decode', type=int, default=1, help='decode in a preallocated cache with a CUDA graph per '
                    'message step (CUDA only)')
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
                base_copy='train' in args.parts and args.lam_p > 0, two=bool(args.two))
    net.charge = args.letter_charge
    net.contrast_ref = args.contrast_ref
    net.chat, net.talk_max, net.warm_reader = args.format == 'chat', args.talk_max, bool(args.warm_reader)
    if args.fast_decode and device == 'cuda':
        net.fast_eps, net.fast_T = args.micro, 64 * ((args.max_len + 1024) // 64)
    state = {}
    if 'train' in args.parts:
        state = train(net, args, log)
    if 'eval' in args.parts:
        meta = Path(args.ckpt) / 'talk.json'
        if meta.exists():
            j = json.loads(meta.read_text())
            net.bias, net.tau = j.get('bias', net.bias), j.get('tau', net.tau)
            net.symbols = j.get('symbols', net.symbols)
        scores, sets = dict(state=state), {}
        for split in ('eval_seen', 'eval_heldout'):
            eps = [e for r in O.Episodes(args.eval_data or args.data, split, seed=1, teacher=True)
                   if (e := fit(r, net, args.max_len))]
            if args.eval_n:
                eps = eps[:args.eval_n]
            sets[split] = eps
            scores[split] = evaluate(net, eps, bs=args.dev_bs, base=True,
                                     copies=('head', 'best', 'target') if args.dev_copy else ())
            log(dict(eval=split, res=scores[split]))
            (out / 'scores.json').write_text(json.dumps(scores, indent=1, ensure_ascii=False))
        for spec in filter(None, args.readers.split(',')):        # written after each, so a failure here loses no score
            from transformers import AutoModelForCausalLM
            name = spec.split('/')[-1].replace('-', '_').replace('.', '_').lower()
            lm = AutoModelForCausalLM.from_pretrained(spec, dtype=torch.bfloat16, attn_implementation='sdpa')
            lm = lm.to(device).eval()
            for split, eps in sets.items():
                scores[f'{split}_{name}'] = evaluate(net, eps, bs=args.dev_bs, readers=[(name, lm)])
                log(dict(eval=f'{split}_{name}', res=scores[f'{split}_{name}']))
                (out / 'scores.json').write_text(json.dumps(scores, indent=1, ensure_ascii=False))
            del lm
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    main()
