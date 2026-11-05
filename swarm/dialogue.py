"""Two agents, one language model, talking in real tokens that other readers cannot interpret: any token but letters.

Agents A and B (Qwen3-4B with one rank-16 LoRA; roles set by the system prompt) each hold private information and
must solve a task together. They speak in turns, B first, two rounds (B, A, B, A). A message is m=16 real vocabulary
tokens that the speaker writes with its own LM head; special and control tokens are masked. During training each
message token is a soft token, softmax(logits / tau) over the vocabulary times the embedding matrix. The listener's
answer loss therefore reaches every message token with full gradient: there is no sampling and no straight-through. At
evaluation the message is the greedy argmax, real token ids readable as text. After the conversation each agent
writes its answer, trained by cross-entropy. --channel picks how a speaker writes:
  auto    one token at a time, each soft token fed back as its next input (exact, but m sequential passes whose
          saved caches fill the card)
  jacobi  the first token from the turn's mark, then all the others refined together over --iters passes, each pass
          reading the previous pass's tokens as its inputs (Jacobi decoding: after k passes the first k + 1 tokens are
          what auto writes, so the default m - 1 passes give exactly auto's message). Only the last pass carries
          gradient (--grad_all: every pass): each token's gradient runs through the speaker's reading of the context
          and of the earlier tokens as inputs, not back through how those earlier tokens were chosen; one saved pass
  slots   in parallel, reading its m tokens out at m learned placeholder inputs (one forward pass per turn)
  kv     reference only, with no tokens: the listener's attention reads the speaker's slot keys and values at every
         layer

Letters: a token counts as a letter if its text has a character of Unicode category L* (any script), or if its bytes
are not complete UTF-8 (it may be part of a letter). Digits, punctuation, symbols, emoji and whitespace are not
letters. The loss adds lam times the mean softmax mass on letters per message position:
  - lam is 0 for the first --free steps, so normal communication (English included) forms first;
  - then lam starts at --lam0 and doubles every --lam_every steps while the hard dev messages are at least
    --letter_goal letters;
  - once they are under it, lam holds; it rises again if letters come back.
Over --steps, tau anneals from --tau0 to --tau1 and an entropy penalty ramps from 0 to --beta.

Privacy starts once the hard dev messages are under --letter_goal letters, or at step --priv_by if that comes first.
The loss then adds
  lam_p * relu(lp_base - floor),
with lam_p ramped from 0 to --lam_p over --priv_ramp steps.
  - lp_base: the untrained model's (adapter off) mean log-probability per token of B's answer, read in B's seat with
    the soft messages. The vocabulary is shared, so its gain trains the speakers.
  - floor: the larger of its closed-book log-probability and chance among the candidates (when the task has
    candidates). The speakers are never rewarded for pushing the reader below its own prior.
If the speakers' own on-minus-swap falls below --keep times its value at the start of privacy, lam_p halves.

Every --dev_every steps, on fixed dev episodes (training values), each task reports:
  - B's answer log-probability with the hard messages (on), and with each received message taken from another dev
    episode of the task (swap);
  - the same with the soft messages;
  - both agents' on minus swap, hard (xgap_A, xgap_B) and soft (soft_xgap_A, soft_xgap_B);
  - the letter fraction of the hard messages and 3 decoded examples;
  - the untrained model's gain over closed-book on the hard messages.
Corpus fill-in is scored in nats per answer token; everything else in nats per answer. The checkpoint is saved at
every dev.
--mix corpus: training samples swarm/corpus.py's tasks with weight 1 - --old_share (fill-in 60%, split-input 40%) and
the games below with --old_share; the games then serve as evaluations.

Games (all need both directions; training never sees the test values):
  squad      A has a SQuAD v1.1 paragraph, B the question; both answer (A at half weight, since it must get the
             question from B). Test: the validation split, whose paragraphs training never saw. Score: token F1 (and
             exact match)
  hotpot     each agent has one of a HotpotQA question's two gold pages, and B has the question; both answer (A at
             half weight). Test: the validation split. Score: token F1 (and exact match)
  schedule   each agent has a 12-hour calendar, each hour free with probability 1/2 (at least one hour is free for
             both); both name the earliest hour free for both. Score: the named hour is that hour; the episode also
             counts whether both named the same hour and it is free for both
  intersect  each agent has 8 words, exactly 2 on both lists (test words from games.WORD_POOLS' held-out pool); both
             name the 2 shared words. Score: shared words named over max(2, own words named)
  twenty     A knows a target object by its 4 attributes; B has 16 distinct random numbered objects, one of them the
             target (chosen at random); both name the target's number. Score: the number is right

Evaluation of the trained agents uses the hard messages. Every reader sees the same test episodes (--n per game) and
gives B's answers:
  trained       the agents with their conversation (on), with every received message zeroed (cut), and with every
                received message taken from another test episode of the same game (transplant)
  base_reads    the untrained model (adapter off) in B's seat, reading the conversation's tokens
  base_closed   the untrained model with B's view alone (closed-book)
  oracle        the untrained model with B's view and A's private information in plain text
  ic_MODEL      Qwen3-8B and AFM-4.5B, each given 8 training-episode conversations as text (with B's view and the
                answer), then a test one (--n_ic per game); ic_MODEL_closed is the same without any conversation
Every eavesdropper is an untrained reader.
Each reader is scored on its greedy answer. trained, base_reads, base_closed and oracle are also scored by forced
choice among the candidate answers (schedule: 12 hours, intersect: the 28 pairs of B's words, twenty: 16 numbers), and
by the teacher-forced log-probability of the true answer. In scores.json:
  - paired: the trained agents minus each reader, per game, with a bootstrap 95% interval over episodes;
  - gains: each reader minus its own closed-book.
Writes OUT/log.jsonl, OUT/scores.json, OUT/examples.json, OUT/transcripts.json (the hard message ids of every test
conversation and of the intercepted training ones, for swarm/eavesdrop.py) and CKPT (adapter/, talk.pt).
Usage: python swarm/dialogue.py --out OUT --ckpt CKPT [--channel jacobi] [--mix corpus] [--parts train,eval,incontext]
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import games as G                           # noqa: E402
try:
    from swarm import corpus as CP                     # noqa: E402  the corpus tasks (--mix corpus)
except ImportError:
    CP = None

BASE = 'Qwen/Qwen3-4B'
OTHERS = ('Qwen/Qwen3-8B', 'arcee-ai/AFM-4.5B')
M = 16                                                 # tokens per message
SPEAKERS = ('B', 'A', 'B', 'A')                        # the turns, in order
GAMES = ('squad', 'hotpot', 'schedule', 'intersect', 'twenty')
QA = ('squad', 'hotpot')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
END = '<|im_end|>'
SHOTS = 8


def other(x):
    return 'A' if x == 'B' else 'B'


# ----------------------------------------------------------------------------------------------------------- games

SQUAD = 'rajpurkar/squad'
SQUAD_FILES = ('plain_text/train-00000-of-00001.parquet', 'plain_text/validation-00000-of-00001.parquet')
SQUAD_WORDS, ANSWER_WORDS = 150, 6
_squad = {}


def squad_rows(test):
    """SQuAD v1.1 questions (training or validation split) whose paragraph has at most SQUAD_WORDS words and whose first
    answer at most ANSWER_WORDS: dict(context, question, answers). Downloaded when missing (about 35 MB)."""
    if test not in _squad:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(SQUAD, SQUAD_FILES[test], repo_type='dataset', local_files_only=True)
        except Exception:
            path = hf_hub_download(SQUAD, SQUAD_FILES[test], repo_type='dataset')
        rows = []
        for r in pq.read_table(path, columns=['context', 'question', 'answers']).to_pylist():
            golds = list(dict.fromkeys(' '.join(a.split()) for a in r['answers']['text'] if a.strip()))
            if golds and len(r['context'].split()) <= SQUAD_WORDS and len(golds[0].split()) <= ANSWER_WORDS:
                rows.append(dict(context=' '.join(r['context'].split()), question=r['question'].strip(),
                                 answers=golds))
        _squad[test] = rows
    return _squad[test]


HOURS = [f'{h}:00' for h in range(8, 20)]
SIZES = ('tiny', 'small', 'large', 'huge')
COLOURS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'white', 'black')
MATERIALS = ('wooden', 'metal', 'glass', 'stone', 'paper', 'plastic', 'woollen', 'leather')
NOUNS = ('ball', 'cube', 'rod', 'disc')
ATTRS = (SIZES, COLOURS, MATERIALS, NOUNS)


def describe(obj):
    return 'a ' + ' '.join(obj)


def squad(rng, test):
    r = rng.choice(squad_rows(test))
    views = dict(A=('Task: agent B has a question about the paragraph below. Help B answer it, then answer it '
                    f'yourself.\n\nParagraph: {r["context"]}'),
                 B=('Task: agent A has a paragraph that answers your question. Answer the question.\n\n'
                    f'Question: {r["question"]}'))
    a = r['answers'][0]
    return dict(game='squad', views=views, answers=dict(A=a, B=a), weights=dict(A=0.5, B=1.0), golds=r['answers'])


def hotpot(rng, test):
    r = rng.choice(CG.hotpot_rows(test))
    i = rng.randrange(2)
    page = {x: f'Your page, "{r["titles"][j]}": {r["pages"][j]}' for x, j in (('A', i), ('B', 1 - i))}
    views = dict(A=('Task: agent B has a question; you and B each hold one page needed to answer it. Help B answer it, '
                    f'then answer it yourself.\n\n{page["A"]}'),
                 B=('Task: agent A holds another page needed to answer your question. Answer the question.\n\n'
                    f'Question: {r["question"]}\n\n{page["B"]}'))
    a = r['answer']
    return dict(game='hotpot', views=views, answers=dict(A=a, B=a), weights=dict(A=0.5, B=1.0), golds=[a])


def schedule(rng, test):
    while True:
        free = {x: [rng.random() < 0.5 for _ in HOURS] for x in 'AB'}
        both = [h for h, a, b in zip(HOURS, free['A'], free['B']) if a and b]
        if both:
            break
    views = {x: (f'Task: agree with agent {other(x)} on a meeting time: the earliest hour free for both of you. Name '
                 'it.\n\nYour calendar: ' + ', '.join(f'{h} {"free" if f else "busy"}' for h, f in zip(HOURS, free[x])))
             for x in 'AB'}
    return dict(game='schedule', views=views, answers=dict(A=both[0], B=both[0]), weights=dict(A=1.0, B=1.0),
                golds=dict(free=free), cands=dict(A=list(HOURS), B=list(HOURS)))


def pair(a, b):
    return ' and '.join(sorted((a, b)))


def intersect(rng, test):
    words = rng.sample(G.WORD_POOLS[test], 14)
    shared, own = words[:2], dict(A=words[2:8], B=words[8:])
    lists = {x: rng.sample(shared + own[x], 8) for x in 'AB'}
    views = {x: (f'Task: you and agent {other(x)} each have 8 words; exactly 2 of them are on both lists. Name the 2 '
                 f'shared words.\n\nYour words: {", ".join(lists[x])}') for x in 'AB'}
    a = pair(*shared)
    return dict(game='intersect', views=views, answers=dict(A=a, B=a), weights=dict(A=1.0, B=1.0),
                golds=dict(shared=shared, lists=lists),
                cands={x: [pair(u, v) for u, v in itertools.combinations(lists[x], 2)] for x in 'AB'})


def twenty(rng, test):
    objs = set()
    while len(objs) < 16:
        objs.add(tuple(rng.choice(v) for v in ATTRS))
    objs = rng.sample(sorted(objs), 16)
    k = rng.randrange(16) + 1
    target = objs[k - 1]
    views = dict(A=('Task: agent B has a numbered list of 16 objects, and the target is one of them. Both of you must '
                    f'name the target\'s number.\n\nThe target is {describe(target)}.'),
                 B=('Task: agent A knows which of your 16 objects is the target. Both of you must name the target\'s '
                    'number.\n\nYour objects:\n' + '\n'.join(f'{i + 1}. {describe(o)}' for i, o in enumerate(objs))))
    a = f'number {k}'
    cands = [f'number {i}' for i in range(1, 17)]
    return dict(game='twenty', views=views, answers=dict(A=a, B=a), weights=dict(A=1.0, B=1.0),
                golds=dict(k=k, target=list(target)), cands=dict(A=cands, B=cands))


MAKERS = dict(squad=squad, hotpot=hotpot, schedule=schedule, intersect=intersect, twenty=twenty)


def episode(game, rng, test=False):
    """One episode: dict(game, views {A, B}, answers {A, B}, weights {A, B}, golds, cands {A, B} or absent). game:
    one of the games here or a corpus task."""
    return MAKERS[game](rng, test) if game in MAKERS else CP.TASKS[game](rng, test)


def family(game):
    """0: the games here (evaluations once the corpus is in the mix); 1: corpus fill-in; 2: corpus split-input."""
    return 0 if game in MAKERS else CP.FAMILY[game]


def mixture(args):
    """Sampling weight per training task. --mix games: --games, equally. --mix corpus: the corpus tasks in its cache
    at 1 - args.old_share (CP.mix_of(): fill-in 60%, split-input 40%) and --games at args.old_share, equally."""
    games = args.games.split(',')
    if args.mix != 'corpus':
        return {g: 1.0 / len(games) for g in games}
    w = {g: args.old_share / len(games) for g in games}
    w.update({t: (1 - args.old_share) * v for t, v in CP.mix_of().items()})
    return w


def test_set(game, n, split='test'):
    """n fixed episodes of a game: test values ('test') or training values with their own seeds ('dev', 'shots')."""
    return [episode(game, random.Random(f'{split}-{game}-{i}'), test=split == 'test') for i in range(n)]


def first_line(text):
    return text.strip().split('\n')[0].strip()


def score(ep, x, text):
    """dict of agent x's answer's scores; 'main' is the one every reader is compared by."""
    g, t = ep['game'], first_line(text)
    if g not in MAKERS:
        return CP.score(ep, x, text)
    if g in QA:
        f1 = max(CG.f1(t, a) for a in ep['golds'])
        return dict(main=f1, f1=f1, em=max(float(CG.words(t) == CG.words(a)) for a in ep['golds']))
    if g == 'schedule':
        m = re.search(r'\b(\d{1,2}):00\b', t)
        h = f'{int(m.group(1))}:00' if m else None
        free = ep['golds']['free']
        ok = h in HOURS and free['A'][HOURS.index(h)] and free['B'][HOURS.index(h)]
        return dict(main=float(h == ep['answers'][x]), free=float(bool(ok)), hour=h)
    if g == 'intersect':
        own = ep['golds']['lists'][x] if x in ep['golds']['lists'] else []
        named = {w for w in own if re.search(rf'\b{re.escape(w)}\b', t.lower())}
        got = len(named & set(ep['golds']['shared']))
        return dict(main=got / max(2, len(named)))
    if g == 'twenty':
        m = re.search(r'\d+', t)
        return dict(main=float(bool(m) and int(m.group()) == ep['golds']['k']))
    raise ValueError(g)


def agree(ep, sa, sb):
    """schedule: both agents named the same hour and it is free for both."""
    return float(sa.get('hour') is not None and sa.get('hour') == sb.get('hour') and sa['free'] == 1.0)


# --------------------------------------------------------------------------------------------------------- letters

_letters = {}


def byte_chars():
    """Byte-level BPE's map from each byte to the character that stands for it in the vocabulary (GPT-2's, as Qwen
    uses)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + \
        list(range(ord('\xae'), ord('\xff') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def letter_sets(tok, V):
    """(allowed, letter): bool masks over the V vocabulary rows. Not allowed: special and added tokens, rows past the
    tokenizer, and tokens with a control character other than tab, newline or carriage return. Letter: a token whose
    text has a character of Unicode category L* (any script), or whose bytes are not complete UTF-8 (a fragment may be
    part of a letter)."""
    key = (getattr(tok, 'name_or_path', ''), len(tok), V)
    if key in _letters:
        return _letters[key]
    dec = {c: b for b, c in byte_chars().items()}
    special = set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}))
    allowed, letter = torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.bool)
    for s, tid in tok.get_vocab().items():
        if tid >= V or tid in special:
            continue
        try:
            text = bytes(dec[c] for c in s).decode('utf-8')
        except (KeyError, UnicodeDecodeError):         # not byte-level, or a fragment of a character
            allowed[tid] = letter[tid] = True
            continue
        if not text or any(unicodedata.category(c) == 'Cc' and c not in '\t\n\r' for c in text):
            continue
        allowed[tid] = True
        letter[tid] = any(unicodedata.category(c).startswith('L') for c in text)
    _letters[key] = (allowed, letter)
    return allowed, letter


def letter_frac(ids, letter):
    """The fraction of the token ids (any shape) that are letters."""
    return float(letter.to(ids.device)[ids].float().mean())


# ----------------------------------------------------------------------------------------------------------- model

class Stream:
    """One agent's (or reader's) stream over a batch of rows, run chunk by chunk with its key-value cache. The first
    chunk starts with each row's own prefix pieces (left-padded); push() adds pieces shared by the chunk's rows (id
    lists, [n, d] tensors, per-row [N, n, d] tensors or [N, n] id tensors); flush(tails) runs the pending pieces plus
    per-row id tails (right-padded) and returns the chunk's hidden states [N, T, d] (after the final norm). n: each
    row's next position (the kv channel sets it to the conversation's shared clock); inject() appends another
    stream's key-value entries."""

    def __init__(self, net, prefixes):
        self.net, self.prefixes, self.pending = net, prefixes, []
        self.cache = self.att = self.n = None

    def inject(self, kv, on=True):
        """Appends key-value entries (per layer (k, v) [N, kv heads, m, head dim], positions already applied) to the
        cache, attended by every later position of this stream (on) or masked off (on=False)."""
        from transformers import DynamicCache
        assert self.cache is not None and not self.pending
        self.cache = DynamicCache(ddp_cache_data=[(torch.cat([k0, k], 2), torch.cat([v0, v], 2))
                                                  for (k0, v0, *_), (k, v) in zip(self.cache, kv)])
        m = kv[0][0].shape[2]
        self.att = torch.cat([self.att, torch.full((self.att.shape[0], m), int(on), dtype=self.att.dtype,
                                                   device=self.att.device)], 1)

    def push(self, *pieces):
        self.pending.extend(pieces)

    def peek(self, x):
        """Hidden states [N, T, d] of input vectors x [N, T, d] at this stream's next positions, run on a fork of its
        cache: the stream itself is unchanged (the cache's update concatenates into new tensors)."""
        from transformers import DynamicCache
        assert self.cache is not None and not self.pending
        fork = DynamicCache(ddp_cache_data=[(k, v) for k, v, *_ in self.cache])
        N, T = x.shape[:2]
        a = torch.ones(N, T, dtype=self.att.dtype, device=self.att.device)
        pos = self.n[:, None] + torch.arange(T, device=self.n.device)
        out = self.net.lm.model(inputs_embeds=x.to(self.net.dtype), attention_mask=torch.cat([self.att, a], 1),
                                position_ids=pos, past_key_values=fork, use_cache=True)
        return out.last_hidden_state

    def flush(self, tails=None):
        net, N = self.net, len(self.prefixes)
        parts, atts = [], []
        if self.cache is None:
            x, a = net.pack(self.prefixes)
            parts.append(x)
            atts.append(a)
        if self.pending:
            u = torch.cat([net.piece(p, N) for p in self.pending], 1)
            parts.append(u)
            atts.append(torch.ones(N, u.shape[1], dtype=torch.long, device=u.device))
            self.pending = []
        if tails is not None:
            x, a = net.pack([[t] for t in tails], left=False)
            parts.append(x)
            atts.append(a)
        x, a = torch.cat(parts, 1), torch.cat(atts, 1)
        if self.cache is None:
            full, pos = a, (a.cumsum(-1) - 1).clamp(min=0)
        else:
            full, pos = torch.cat([self.att, a], 1), self.n[:, None] + (a.cumsum(-1) - 1).clamp(min=0)
        out = net.lm.model(inputs_embeds=x, attention_mask=full, position_ids=pos, past_key_values=self.cache,
                           use_cache=True)
        self.cache, self.att = out.past_key_values, full
        self.n = a.sum(1) if self.n is None else self.n + a.sum(1)
        return out.last_hidden_state


class Talk(nn.Module):
    """The model (with its LoRA), the vocabulary masks and the message slots [turns, m, d] (slots and kv channels)."""

    def __init__(self, model, tok, m=M, turns=len(SPEAKERS)):
        super().__init__()
        self.model, self.tok, self.m = model, tok, m
        emb = self.lm.model.embed_tokens.weight
        self.slots = nn.Parameter(torch.randn(turns, m, emb.shape[1]) * float(emb.float().std()))
        allowed, letter = letter_sets(tok, self.lm.lm_head.weight.shape[0])
        self.register_buffer('allowed', allowed)
        self.register_buffer('letter', letter)
        self.channel = 'auto'                          # auto, jacobi, slots or kv (see the module docstring)
        self.iters = 4                                 # jacobi: refinement passes per message
        self.grad_last = True                          # jacobi: only the last pass carries gradient
        self.bias = 0.0                                # nats taken off every letter token's logit when writing
        self.tau = 1.0
        self.track = False                             # keep the messages' gradients (msg_grad)
        self.msgs = []
        self._ids = {}
        self.end = tok.convert_tokens_to_ids(END)

    @property
    def lm(self):
        return self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model

    @property
    def dtype(self):
        return self.lm.model.embed_tokens.weight.dtype

    @property
    def device(self):
        return self.lm.model.embed_tokens.weight.device

    def ids(self, text):
        if text not in self._ids:
            self._ids[text] = self.tok(text, add_special_tokens=False).input_ids
        return self._ids[text]

    def piece(self, p, N):
        emb = self.lm.model.embed_tokens
        if isinstance(p, list):
            return emb(torch.tensor(p, device=self.device))[None].expand(N, -1, -1)
        if p.dtype == torch.long:
            return emb(p.to(self.device))
        p = p.to(self.device, self.dtype)
        return p[None].expand(N, -1, -1) if p.dim() == 2 else p

    def pack(self, rows, left=True):
        """rows: per row a list of pieces (id lists, [n] id tensors or [n, d] tensors). Returns embeddings [N, T, d] and
        the mask."""
        emb = self.lm.model.embed_tokens
        seqs = []
        for pieces in rows:
            parts = []
            for p in pieces:
                if not len(p):
                    continue
                if isinstance(p, list) or p.dtype == torch.long:
                    parts.append(emb(torch.as_tensor(p, device=self.device, dtype=torch.long)))
                else:
                    parts.append(p.to(self.device, self.dtype))
            seqs.append(torch.cat(parts, 0))
        T = max(len(s) for s in seqs)
        x = torch.stack([F.pad(s, (0, 0, T - len(s), 0) if left else (0, 0, 0, T - len(s))) for s in seqs])
        att = torch.zeros(len(seqs), T, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if left:
                att[i, T - len(s):] = 1
            else:
                att[i, :len(s)] = 1
        return x, att

    def target_lp(self, h, starts, targets):
        """Per-row sum and count of the log-probabilities of targets[i] at chunk positions starts[i].. (each token
        predicted by the position before it)."""
        L, N = max(len(t) for t in targets), len(targets)
        pos = torch.tensor([[min(starts[i] - 1 + j, h.shape[1] - 1) for j in range(L)] for i in range(N)],
                           device=h.device)
        hs = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        lp = self.lm.lm_head(hs).float().log_softmax(-1)
        tgt = torch.tensor([t + [0] * (L - len(t)) for t in targets], device=h.device)
        mask = torch.tensor([[1.0] * len(t) + [0.0] * (L - len(t)) for t in targets], device=h.device)
        got = lp.gather(-1, tgt[..., None])[..., 0] * mask
        return got.sum(1), mask.sum(1)

    def write(self, h, hard=None):
        """Message tokens from the speaker's final hidden states h [N, n, d]: (input vectors [N, n, d] for the listener,
        argmax ids [N, n], probabilities [N, n, V] or None). soft (training, or hard=False): softmax(logits / tau)
        over the allowed vocabulary times the embedding matrix; hard: the argmax token's embedding. Letter tokens' logits
        are lowered by self.bias first (a decoding rule, the same in training and evaluation)."""
        z = self.lm.lm_head(h).float().masked_fill(~self.allowed, float('-inf'))
        if self.bias:
            z = z - self.bias * self.letter.float()
        idx = z.argmax(-1)
        hard = not self.training if hard is None else hard
        E = self.lm.model.embed_tokens.weight
        if hard:
            return E[idx], idx, None
        p = (z / self.tau).softmax(-1)
        return p.to(E.dtype) @ E, idx, p if torch.is_grad_enabled() else None

    def prefix(self, ep, x, mode='talk'):
        """Agent x's opening (system prompt and its user turn up to the conversation): talk, closed (no conversation) or
        oracle (the partner's private information in plain text, no conversation)."""
        sys_ = (f'You are agent {x}. You and agent {other(x)} each hold private information and must solve a task '
                f'together; neither of you can see the other\'s information. When it is your turn, write to agent '
                f'{other(x)} and tell it what it needs to know.')
        user = ep['views'][x]
        if mode == 'talk':
            user += f'\n\nYour conversation with agent {other(x)}:'
        elif mode == 'oracle':
            user += f'\n\nAgent {other(x)}\'s private information, in plain text:\n{ep["views"][other(x)]}'
        return self.ids(f'<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{user}')

    def converse(self, eps, mode='on', donor=None, hard=None):
        """The conversation of a batch of episodes. mode on: the listener receives the speaker's message; cut: zeros;
        transplant: donor[t] (input vectors [N, m, d], another episode's message at turn t). Returns both agents'
        streams (pending the last message heard), the message vectors sent per turn, their argmax ids [N, turns, m]
        (None for kv) and their probabilities per turn ([N, m, V], or Nones)."""
        if self.channel == 'kv':
            return self.converse_kv(eps, mode, donor)
        S = {x: Stream(self, [[self.prefix(ep, x)] for ep in eps]) for x in 'AB'}
        sent, idx, probs = [], [], []
        for t, spk in enumerate(SPEAKERS):
            mark = self.ids(f'\n{spk}:')
            if self.channel == 'auto':               # token by token: each soft token is the speaker's next input
                S[spk].push(mark)
                h, words = S[spk].flush()[:, -1:], []
                for j in range(self.m):
                    words.append(self.write(h, hard))
                    S[spk].push(words[-1][0])
                    if j < self.m - 1:
                        h = S[spk].flush()[:, -1:]
                v = torch.cat([w[0] for w in words], 1)
                i = torch.cat([w[1] for w in words], 1)
                p = torch.cat([w[2] for w in words], 1) if words[0][2] is not None else None
            elif self.channel == 'jacobi':             # all m tokens refined together, each from the previous pass
                S[spk].push(mark)
                v0, i0, p0 = self.write(S[spk].flush()[:, -1:], hard)
                v, i, p = v0.expand(-1, self.m - 1, -1), i0.expand(-1, self.m - 1), None
                for k in range(self.iters):              # m - 1 passes: exactly the autoregressive message
                    last = k == self.iters - 1
                    with torch.set_grad_enabled(torch.is_grad_enabled() and (last or not self.grad_last)):
                        before = i
                        v, i, p = self.write(S[spk].peek(torch.cat([v0, v[:, :-1]], 1)), hard)
                    if hard and not torch.is_grad_enabled() and torch.equal(i, before):
                        break                                 # hard tokens at their fixed point
                v, i = torch.cat([v0, v], 1), torch.cat([i0, i], 1)
                p = torch.cat([p0, p], 1) if p0 is not None else None
                S[spk].push(v)
            else:                                     # slots: the m tokens at once
                S[spk].push(mark, self.slots[t])
                v, i, p = self.write(S[spk].flush()[:, -self.m:], hard)
            sent.append(v)
            idx.append(i)
            probs.append(p)
            got = v if mode == 'on' else torch.zeros_like(v) if mode == 'cut' else donor[t]
            S[other(spk)].push(mark, got)
        if self.track:
            self.msgs = [v for v in sent if v.requires_grad]
            for v in self.msgs:
                v.retain_grad()
        return S, sent, torch.stack(idx, 1), probs

    def converse_kv(self, eps, mode='on', donor=None):
        """The kv channel (reference, no tokens): a message is the speaker's m slot positions; the listener's attention
        at every layer reads their keys and values, appended to its own cache (cut: masked off; transplant: donor[t]'s).
        One clock: both openings first, then each turn at the next positions of the conversation, so a message lies in
        the listener's past at its true distance. Returns the streams, per turn dict(kv=[(k, v)] per layer, h=the
        slots' final states [N, m, d]), None ids and None probabilities."""
        S = {x: Stream(self, [[self.prefix(ep, x)] for ep in eps]) for x in 'AB'}
        for x in 'AB':
            S[x].flush()
        clock = torch.maximum(S['A'].n, S['B'].n)
        sent = []
        for t, spk in enumerate(SPEAKERS):
            S[spk].n = clock.clone()
            S[spk].push(self.ids(f'\n{spk}:'), self.slots[t])
            h = S[spk].flush()
            kv = [(k[:, :, -self.m:].clone(), v[:, :, -self.m:].clone()) for k, v, *_ in S[spk].cache]
            clock = S[spk].n.clone()
            S[other(spk)].inject(kv if mode != 'transplant' else donor[t]['kv'], on=mode != 'cut')
            S[other(spk)].n = clock.clone()
            sent.append(dict(kv=kv, h=h[:, -self.m:]))
        if self.track:
            self.msgs = [k for d in sent for kv in d['kv'] for k in kv if k.requires_grad]
            for k in self.msgs:
                k.retain_grad()
        return S, sent, None, [None] * len(SPEAKERS)

    def msg_grad(self):
        """The gradient norm that reached the messages (their vectors, or the kv channel's keys and values) in the last
        tracked backward."""
        g = [m.grad.float().pow(2).sum() for m in self.msgs if m.grad is not None]
        self.msgs = []
        return float(torch.stack(g).sum().sqrt()) if g else 0.0

    def rows(self, eps, msgs=None, mode='talk', tails=None, own=None):
        """A reader in B's seat: B's opening and every turn's mark and message (msgs: the turns' token ids
        [N, turns, m], or per turn input vectors [N, m, d]); closed or oracle: no conversation. own='B': B's own turns
        as it wrote them (the slots channel: its slots). tails: per-row id lists appended."""
        out = []
        for i, ep in enumerate(eps):
            pieces = [self.prefix(ep, 'B', mode)]
            if mode == 'talk':
                for t, spk in enumerate(SPEAKERS):
                    if own == spk and self.channel == 'slots':
                        msg = self.slots[t]
                    elif isinstance(msgs, list):
                        msg = msgs[t][i]
                    else:
                        msg = msgs[i][t].to(self.device)
                    pieces += [self.ids(f'\n{spk}:'), msg]
            out.append(pieces + ([tails[i]] if tails is not None else []))
        return out

    def answer_tail(self, ep, x, text=None):
        return self.ids(SUFFIX) + self.ids(ep['answers'][x] if text is None else text) + [self.end]

    def loss(self, eps, lam=0.0, beta=0.0, lam_p=0.0, info=0.0, contrast=0.0, target=0.2):
        """The batch's loss: each agent's answer cross-entropy (mean per token, weighted); lam times the mean softmax
        mass on letters per message position; beta times the mean entropy per position; minus info times the
        messages' information about the episode per position (the entropy of the batch's mean distribution minus the
        mean entropy: an anti-collapse term); lam_p times the untrained model's gain over its floor (privacy); contrast
        times a hinge on the margin per answer token between B's answer log-probability with the messages it received
        and with the next batch episode's (B's own turns kept), relu(target - margin): a message that does not depend
        on the episode pays target, and the hinge stops pushing once the code beats the swap by target nats per token.
        Per token, like the cross-entropy: the summed version's gradient was the answer length times larger.
        Returns the loss, its parts and the argmax ids [N, turns, m]."""
        S, sent, ids, probs = self.converse(eps)
        total, parts, s = 0.0, {}, len(self.ids(SUFFIX))
        for x in 'AB':
            tails = [self.answer_tail(ep, x) for ep in eps]
            h = S[x].flush(tails)
            U = h.shape[1] - max(len(t) for t in tails)
            got, n = self.target_lp(h, [U + s] * len(eps), [t[s:] for t in tails])
            ce = -got / n
            if x == 'B':
                own = got
            w = torch.tensor([ep['weights'][x] for ep in eps], device=ce.device)
            total = total + (w * ce).mean()
            parts[f'ce_{x}'] = round(float(ce.detach().mean()), 4)
        if probs[0] is not None:
            p = torch.stack(probs, 1)                                       # [N, turns, m, V]
            mass = (p @ self.letter.float()).mean()
            with torch.set_grad_enabled(torch.is_grad_enabled() and (beta > 0 or info > 0)):
                ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean()
                pm = p.mean(0)
                mi = -(pm * pm.clamp_min(1e-30).log()).sum(-1).mean() - ent
            total = total + lam * mass + (beta * ent if beta > 0 else 0.0) - (info * mi if info > 0 else 0.0)
            parts['info'] = round(float(mi.detach()), 4)
            parts.update(letters_soft=round(float(mass.detach()), 4), ent=round(float(ent.detach()), 4),
                         letters_hard=round(letter_frac(ids, self.letter), 4), distinct=len(torch.unique(ids)))
        if contrast and len(eps) > 1 and self.channel != 'kv':
            perm = list(range(1, len(eps))) + [0]
            tails = [self.answer_tail(ep, 'B') for ep in eps]
            sw, n = self.read_lp(self.rows(eps, swap_received(sent, perm), tails=tails), [t[s:] for t in tails])
            margin = (own - sw) / n
            total = total + contrast * (target - margin).clamp(min=0).mean()
            parts['contrast'] = round(float(margin.detach().mean()), 4)
        if lam_p:
            gain = self.reader_gain(eps, sent)
            total = total + lam_p * gain.clamp(min=0).mean()
            parts['gain_base'] = round(float(gain.detach().mean()), 4)
        return total, parts, ids

    def reader_gain(self, eps, sent):
        """Per episode: the untrained model's (adapter off) mean log-probability per token of B's answer in B's seat
        reading the messages sent (vectors with their graph, so the gain trains the speakers), minus its floor: the
        larger of the same closed-book and chance among the candidates."""
        tails = [self.answer_tail(ep, 'B') for ep in eps]
        s = len(self.ids(SUFFIX))
        targets = [t[s:] for t in tails]
        with self.reader('base'):
            lp, n = self.read_lp(self.rows(eps, sent, tails=tails), targets)
            with torch.no_grad():
                lp0, n0 = self.read_lp(self.rows(eps, mode='closed', tails=tails), targets)
        chance = torch.tensor([-math.log(len(ep['cands']['B'])) / len(t) if 'cands' in ep else -math.inf
                               for ep, t in zip(eps, targets)], device=lp.device)
        return lp / n - torch.maximum(lp0 / n0, chance)

    def reader(self, who):
        """A context: the untrained model (base: the adapter off) or the agents (None)."""
        return self.model.disable_adapter() if who == 'base' else _null()

    def read_lp(self, rows, targets):
        """Per-row sum and count of the log-probabilities of targets, the last tokens of each row (one full pass)."""
        x, att = self.pack(rows)
        h = self.lm.model(inputs_embeds=x, attention_mask=att, position_ids=(att.cumsum(-1) - 1).clamp(min=0),
                          use_cache=False).last_hidden_state
        T = h.shape[1]
        return self.target_lp(h, [T - len(t) for t in targets], targets)

    @torch.no_grad()
    def greedy(self, stream, new=24):
        """Greedy continuation of a stream after the answer prompt (SUFFIX), up to <|im_end|>; returns texts."""
        stream.push(self.ids(SUFFIX))
        h = stream.flush()
        N = h.shape[0]
        out = torch.zeros(N, 0, dtype=torch.long, device=h.device)
        done = torch.zeros(N, dtype=torch.bool, device=h.device)
        for _ in range(new):
            nxt = self.lm.lm_head(h[:, -1]).argmax(-1)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == self.end
            if done.all():
                break
            stream.push(nxt[:, None])
            h = stream.flush()
        texts = []
        for row in out.tolist():
            row = row[:row.index(self.end)] if self.end in row else row
            texts.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return texts

    @torch.no_grad()
    def choose(self, rows, cands, answers, bs=64):
        """Forced choice: per row, the candidate whose text (plus <|im_end|>) after rows[i] + SUFFIX is most likely,
        and the true answer's mean log-probability per token."""
        flat = []
        for i, (r, cs) in enumerate(zip(rows, cands)):
            for c in cs + [answers[i]]:
                flat.append((r, self.answer_tail(None, None, c)))
        s = len(self.ids(SUFFIX))
        scores = []
        for j in range(0, len(flat), bs):
            part = flat[j:j + bs]
            lp, n = self.read_lp([r + [t] for r, t in part], [t[s:] for _, t in part])
            scores += list(zip(lp.tolist(), n.tolist()))
        picks, truth, j = [], [], 0
        for i, cs in enumerate(cands):
            got = scores[j:j + len(cs)]
            picks.append(cs[max(range(len(cs)), key=lambda c: got[c][0])] if cs else None)
            lp, n = scores[j + len(cs)]
            truth.append(lp / n)
            j += len(cs) + 1
        return picks, truth


def build(device, base=BASE, rank=16, ckpt=None, tok_name=None, model=None, m=M):
    """The Talk net: the base model with a fresh rank-16 LoRA (or the one saved at ckpt) in bf16 on device."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name or base, local_files_only=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True)
    model = model.to(device)
    state = None
    if ckpt is not None and (Path(ckpt) / 'talk.pt').exists():
        state = torch.load(Path(ckpt) / 'talk.pt', map_location=device)
    if ckpt is not None and (Path(ckpt) / 'adapter').exists():
        model = PeftModel.from_pretrained(model, str(Path(ckpt) / 'adapter'), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    net = Talk(model, tok, m=state['m'] if state else m).to(device)
    if state:
        net.slots.data.copy_(state['slots'])
        net.channel, net.tau, net.iters = state['channel'], state['tau'], state.get('iters', 4)
        net.grad_last = state.get('grad_last', False)
        net.bias = state.get('bias', 0.0)
    return net


def save(net, ckpt):
    ckpt = Path(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    net.model.save_pretrained(str(ckpt / 'adapter'), selected_adapters=['default'])
    torch.save(dict(slots=net.slots.detach().cpu(), channel=net.channel, tau=net.tau, m=net.m, iters=net.iters,
                    grad_last=net.grad_last, bias=net.bias),
               ckpt / 'talk.pt')


# ---------------------------------------------------------------------------------------------------- dev numbers

def mean(xs):
    return sum(xs) / len(xs)


def msg_map(msg, f):
    """f applied to each tensor of a message (vectors [N, m, d], or the kv channel's dict)."""
    if isinstance(msg, dict):
        return dict(kv=[(f(k), f(v)) for k, v in msg['kv']], h=f(msg['h']))
    return f(msg)


def msg_cat(parts):
    """One message from its batches' parts."""
    if isinstance(parts[0], dict):
        return dict(kv=[(torch.cat([p['kv'][j][0] for p in parts]), torch.cat([p['kv'][j][1] for p in parts]))
                        for j in range(len(parts[0]['kv']))], h=torch.cat([p['h'] for p in parts]))
    return torch.cat(parts)


@torch.no_grad()
def answer_nats(net, eps, mode='on', donor=None, hard=None, bs=32):
    """Each agent's teacher-forced log-probability of its answer after the conversation, per episode (nats per answer,
    or for corpus fill-in, family 1, nats per answer token); the message vectors per turn; the ids [N, turns, m]."""
    nats, sent, ids, s = dict(A=[], B=[]), [], [], len(net.ids(SUFFIX))
    for j in range(0, len(eps), bs):
        part = eps[j:j + bs]
        S, v, i, _ = net.converse(part, mode, None if donor is None else
                                  [msg_map(u, lambda z: z[j:j + bs]) for u in donor], hard)
        sent.append(v)
        ids.append(i)
        per_tok = torch.tensor([family(ep['game']) == 1 for ep in part], device=net.device)
        for x in 'AB':
            tails = [net.answer_tail(ep, x) for ep in part]
            h = S[x].flush(tails)
            U = h.shape[1] - max(len(t) for t in tails)
            lp, n = net.target_lp(h, [U + s] * len(part), [t[s:] for t in tails])
            nats[x] += torch.where(per_tok, lp / n, lp).tolist()
    return (nats, [msg_cat([v[t] for v in sent]) for t in range(len(SPEAKERS))],
            torch.cat(ids) if ids[0] is not None else None)


def swap_order(eps):
    """Per episode, the index of the next episode of the same task (cyclic): its messages are the swap's."""
    by = {}
    for i, ep in enumerate(eps):
        by.setdefault(ep['game'], []).append(i)
    out = [0] * len(eps)
    for idx in by.values():
        for a, b in zip(idx, idx[1:] + idx[:1]):
            out[a] = b
    return out


@torch.no_grad()
def gains(net, eps, ids, who, bs=32):
    """Per episode: a reader's (base, or None for the trained agents) mean log-probability per token of B's answer
    reading the hard messages ids [N, turns, m], minus the same closed-book."""
    s, out = len(net.ids(SUFFIX)), []
    with net.reader(who):
        for j in range(0, len(eps), bs):
            part = eps[j:j + bs]
            tails = [net.answer_tail(ep, 'B') for ep in part]
            tg = [t[s:] for t in tails]
            a, n = net.read_lp(net.rows(part, ids[j:j + bs], tails=tails), tg)
            b, n0 = net.read_lp(net.rows(part, mode='closed', tails=tails), tg)
            out += (a / n - b / n0).tolist()
    return out


@torch.no_grad()
def dev_eval(net, dev, readers=(), bs=32, soft=True):
    """Per task on the fixed dev episodes (dev: a list, any tasks): B's answer log-probability with the hard messages
    (on) and with each received message taken from the next dev episode of the task (swap); the same with soft
    messages (soft_on, soft_swap); both agents' on minus swap (xgap_A, xgap_B; soft_xgap_A, soft_xgap_B); the hard
    messages' letter fraction and distinct tokens, 3 decoded examples (each the list of turns), and each reader's
    gain over closed-book (gain_base); soft=False: the hard numbers only. 'fill': the fill-in sources pooled (each swaps within its own source). '_all': the
    letter fraction over every task."""
    was = net.training
    net.eval()
    perm = swap_order(dev)
    modes = [('', True)] + ([('soft_', False)] if soft and net.channel != 'kv' else [])
    got, ids = {}, None
    for name, hard in modes:
        on, sent, i = answer_nats(net, dev, hard=hard, bs=bs)
        sw, _, _ = answer_nats(net, dev, 'transplant', [msg_map(v, lambda z: z[perm]) for v in sent], hard, bs)
        got[name] = (on, sw)
        ids = i if hard else ids
    extra = {who: gains(net, dev, ids, who, bs) for who in readers} if ids is not None else {}
    out, groups = {}, {}
    for k, ep in enumerate(dev):
        groups.setdefault(ep['game'], []).append(k)
    fills = [k for g, rows in groups.items() if g.startswith('fill_') for k in rows]
    if len([g for g in groups if g.startswith('fill_')]) > 1:
        groups['fill'] = fills                                          # every fill-in source pooled
    for g, rows in groups.items():
        r = {}
        for name, (on, sw) in got.items():
            r[f'{name}on'] = mean([on['B'][k] for k in rows])
            r[f'{name}swap'] = mean([sw['B'][k] for k in rows])
            for x in 'AB':
                r[f'{name}xgap_{x}'] = mean([on[x][k] - sw[x][k] for k in rows])
        if ids is not None:
            r['letters'] = letter_frac(ids[rows], net.letter)
            r['distinct'] = len(torch.unique(ids[rows]))
            r['examples'] = [[net.tok.decode(ids[k, t].tolist()) for t in range(len(SPEAKERS))] for k in rows[:3]]
        for who, v in extra.items():
            r[f'gain_{who}'] = mean([v[k] for k in rows])
        out[g] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
    if ids is not None:
        out['_all'] = dict(letters=round(letter_frac(ids, net.letter), 4))
    net.train(was)
    return out


BIG_DEV = ('squad', 'qa_squad', 'translate', 'summarize')


def dev_size(g, games, args):
    """Dev episodes for a task: --dev_big for squad, qa_squad, translate and summarize, and --dev_big shared over the
    fill-in sources (pooled in dev_eval); --dev_n_corpus for the other corpus tasks, --dev_n for the other games."""
    if g.startswith('fill_'):
        return -(-args.dev_big // len([x for x in games if x.startswith('fill_')]))
    if g in BIG_DEV:
        return args.dev_big
    return args.dev_n_corpus if family(g) else args.dev_n


def held(d, tasks, key='xgap_B'):
    """The mean of a dev number over tasks (the speakers' on minus swap, by default)."""
    return mean([d[g][key] for g in tasks if g in d])


# -------------------------------------------------------------------------------------------------------- training

def swap_received(sent, perm):
    """The per-turn messages with those B receives (A's turns) taken from episode perm[i] and B's own turns kept."""
    return [v[perm] if spk == 'A' else v for v, spk in zip(sent, SPEAKERS)]


POOL = ('squad', 'qa_squad', 'translate', 'summarize', 'fill')


def pooled_gap(d):
    """Hard on minus swap (B) averaged over the 48-episode dev tasks present (fill: the sources pooled)."""
    return mean([d[g]['xgap_B'] for g in POOL if g in d])


def letter_ctrl(lam, pooled, ref, args):
    """The letter weight's feedback step after a dev: times --ctrl_up when the pooled gap holds at least --ctrl_keep of
    its value before the penalty (ref), else times --ctrl_down. Returns the new weight and whether the gap held."""
    ok = pooled >= args.ctrl_keep * ref if ref > 0 else pooled >= ref
    return (min(lam * args.ctrl_up, args.lam_max) if ok else lam * args.ctrl_down), ok


def ctrl_ready(pooled, pooled0, args):
    """Whether the letter controller may start: the pooled gap is positive and at least --ctrl_start of its step-0
    value (28's English phase collapsed to one token by step 200, and a reference of 0 then always counted as held)."""
    return pooled > 0 and pooled >= args.ctrl_start * pooled0


def letter_lam(step, lam, raised, letters, args):
    """The letter penalty's weight (and the step it last rose) after the dev at this step: 0 before --free; then
    --lam0; doubled (up to --lam_max) once --lam_every steps have passed since it last rose while the hard dev messages
    hold at least --letter_goal letters; otherwise held."""
    if step < args.free:
        return 0.0, None
    if not lam:
        return args.lam0, step
    if step - raised >= args.lam_every and letters is not None and letters >= args.letter_goal:
        return min(2 * lam, args.lam_max), step
    return lam, raised


def train(net, args, out, log):
    """Trains for --steps steps (or --max_minutes) on the task mixture: the answers' cross-entropy, the letter penalty
    on its schedule (letter_lam), the entropy penalty ramp, and privacy (the untrained model's gain) from the first dev
    under --letter_goal letters or step --priv_by. Dev numbers and a checkpoint every --dev_every steps."""
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    lora = [p for name, p in net.model.named_parameters() if p.requires_grad and '.default.' in name]
    every = lora + [net.slots]
    opt = torch.optim.AdamW([dict(params=lora, lr=args.lr), dict(params=[net.slots], lr=args.own_lr)],
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50))
    mix = mixture(args)
    games = list(mix)
    gated = [g for g in games if family(g)] or games
    dev = [ep for g in games for ep in test_set(g, dev_size(g, games, args), 'dev')]
    log(dict(mix={g: round(w, 4) for g, w in mix.items()}, allowed=int(net.allowed.sum()),
             letters=int(net.letter.sum()), vocab=len(net.allowed), dev=len(dev)))
    micros = {}                                                         # per task: halved on that task's OOMs
    started, step = time.monotonic(), 0
    lam, raised, letters = 0.0, None, None
    priv, base_x, lam_p_max = None, None, args.lam_p
    d0 = None                                                           # the step-0 dev, beside every later one
    ref, ok, pooled0 = None, False, None                                # --ctrl: the pooled gap before the penalty
    dev_bs = 16
    while True:
        frac = min(1.0, step / args.steps)
        net.tau = args.tau0 * (args.tau1 / args.tau0) ** frac
        beta = args.beta * frac
        if step % args.dev_every == 0:
            while True:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    full = step % args.full_every == 0
                    d = dev_eval(net, dev, ('base',) if full else (), bs=dev_bs, soft=full)
                    break
                except torch.cuda.OutOfMemoryError:
                    if dev_bs == 1:
                        raise
                    dev_bs //= 2
                    log(dict(step=step, dev_oom=True, bs=dev_bs))
            d0 = d0 or {g: {k: v[k] for k in ('xgap_B', 'soft_xgap_B') if k in v} for g, v in d.items() if g != '_all'}
            for g, v in d0.items():
                d[g].update({f'{k}_0': x for k, x in v.items()})
            letters = d.get('_all', {}).get('letters')
            x = held(d, gated)
            if args.ctrl:
                pooled = pooled_gap(d)
                pooled0 = pooled if pooled0 is None else pooled0
            if args.ctrl and step >= args.free and ref is None and not ctrl_ready(pooled, pooled0, args):
                log(dict(step=step, ctrl_wait=True, pooled=round(pooled, 4), at_step0=round(pooled0, 4)))
            elif args.ctrl and step >= args.free:
                if ref is None:
                    ref, lam, ok = pooled, args.lam0, True
                else:
                    lam, ok = letter_ctrl(lam, pooled, ref, args)
                net.bias = args.bias_k * lam
                log(dict(step=step, ctrl=True, pooled=round(pooled, 4), ref=round(ref, 4), held=ok, lam=round(lam, 5),
                         bias=round(net.bias, 4), letters=letters))
            start = (letters is not None and letters < args.priv_letters and ok) if args.ctrl else (
                (letters is not None and letters < args.letter_goal) or step >= args.priv_by)
            if priv is None and net.channel != 'kv' and step >= args.free and start:
                priv, base_x = step, x
                log(dict(step=step, privacy_starts=True, letters=letters, xgap_B=round(x, 4)))
            elif priv is not None and base_x > 0 and x < args.keep * base_x:
                lam_p_max /= 2
                log(dict(step=step, lam_p_halved=round(lam_p_max, 5), xgap_B=round(x, 4), at_start=round(base_x, 4)))
            log(dict(step=step, tau=round(net.tau, 3), lam=lam, dev=d, full=full, s=round(time.monotonic() - started)))
            save(net, args.ckpt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                                # the dev's blocks back before training
        if not args.ctrl:
            lam, raised = letter_lam(step, lam, raised, letters, args)
        lam_p = lam_p_max * min(1.0, (step - priv) / args.priv_ramp) if priv is not None else 0.0
        if step >= args.steps or (time.monotonic() - started) / 60 > args.max_minutes:
            save(net, args.ckpt)
            log(dict(step=step, done=True, s=round(time.monotonic() - started)))
            return
        g = rng.choices(games, weights=[mix[x] for x in games])[0]
        micro = micros.get(g, args.micro)
        accum = max(1, args.batch // micro)
        try:
            parts_all, gmsg = [], 0.0
            net.track = True
            for _ in range(accum):
                eps = [episode(g, rng) for _ in range(micro)]
                loss, parts, ids = net.loss(eps, lam, beta, lam_p, args.info, args.contrast, args.contrast_target)
                (loss / accum).backward()
                gmsg += net.msg_grad() ** 2
                parts_all.append(parts)
            net.track = False
            gslots = float(net.slots.grad.norm()) if net.slots.grad is not None else 0.0
            torch.nn.utils.clip_grad_norm_(every, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            net.track, net.msgs, loss, parts, ids = False, [], None, None, None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            micros[g] = micro // 2
            log(dict(oom=True, game=g, micro=micro // 2))
            continue
        step += 1
        if step % args.log_every == 0:
            rec = dict(step=step, game=g, tau=round(net.tau, 3), lam=lam, beta=round(beta, 4), lam_p=round(lam_p, 4),
                       s=round(time.monotonic() - started), g_msg=float(f'{gmsg ** 0.5:.3g}'),
                       g_slots=float(f'{gslots:.3g}'))
            for k in parts_all[0]:
                rec[k] = round(mean([p[k] for p in parts_all]), 4)
            if torch.cuda.is_available():
                rec['gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            log(rec)


# ------------------------------------------------------------------------------------------------------ evaluation

def batches(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def bootstrap(a, b, boots=2000, seed=0):
    """Mean of a - b (paired by episode) and its bootstrap 95% interval."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(boots))
    return dict(diff=round(sum(d) / len(d), 4), lo=round(means[int(0.025 * boots)], 4),
                hi=round(means[int(0.975 * boots) - 1], 4), n=len(d))


def budget(eps):
    """Greedy answer tokens for a batch: the corpus's max_new (long for fill-in, code, translation and summaries),
    else 24."""
    return max(ep.get('max_new', 24) for ep in eps)


@torch.no_grad()
def conversations(net, eps, bs=32, mode='on', donor=None):
    """The hard messages [n, turns, m] and both agents' greedy answers for a list of episodes. donor: the 'on'
    messages of the same episodes, each episode receiving those of the next one of its task (transplant)."""
    msgs, texts = [], dict(A=[], B=[])
    perm = swap_order(eps)
    emb = net.lm.model.embed_tokens
    for j in range(0, len(eps), bs):
        part = eps[j:j + bs]
        d = None
        if mode == 'transplant':
            ids = donor[perm[j:j + bs]].to(net.device)
            d = [emb(ids[:, t]) for t in range(len(SPEAKERS))]
        S, _, idx, _ = net.converse(part, mode, d, hard=True)
        msgs.append(idx.cpu())
        for x in 'AB':
            texts[x] += net.greedy(S[x], budget(part))
    return torch.cat(msgs), texts


def read_greedy(net, rows, bs=32, new=24):
    texts = []
    for part in batches(rows, bs):
        texts += net.greedy(Stream(net, part), new)
    return texts


def evaluate(net, args, out, log):
    """Trained agents (on, cut, transplant), the untrained model reading the conversation, closed-book and oracle, on
    --n test episodes per game. Returns the per-game numbers and records."""
    net.eval()
    games = args.games.split(',')
    res, recs = {}, {}
    for g in games:
        t0 = time.monotonic()
        eps = test_set(g, args.n)
        per = {}
        msgs, on = conversations(net, eps)
        recs[g] = dict(msgs=msgs, texts={'trained': on['B']})
        for mode in ('on', 'cut', 'transplant'):
            texts = on if mode == 'on' else conversations(net, eps, mode=mode, donor=msgs)[1]
            sb = [score(ep, 'B', t) for ep, t in zip(eps, texts['B'])]
            sa = [score(ep, 'A', t) for ep, t in zip(eps, texts['A'])]
            name = 'trained' if mode == 'on' else f'trained_{mode}'
            per[name] = [s['main'] for s in sb]
            per[name + '_A'] = [s['main'] for s in sa]
            if g in QA:
                per[name + '_em'] = [s['em'] for s in sb]
            if g == 'schedule':
                per[name + '_agree'] = [agree(ep, a, b) for ep, a, b in zip(eps, sa, sb)]
            recs[g]['texts'][name] = texts['B']
            recs[g]['texts'][name + '_A'] = texts['A']
        with net.reader('base'):
            for name, mode in (('base_reads', 'talk'), ('base_closed', 'closed'), ('oracle', 'oracle')):
                texts = read_greedy(net, net.rows(eps, msgs, mode=mode), new=budget(eps))
                sb = [score(ep, 'B', t) for ep, t in zip(eps, texts)]
                per[name] = [s['main'] for s in sb]
                if g in QA:
                    per[name + '_em'] = [s['em'] for s in sb]
                recs[g]['texts'][name] = texts
        # forced choice and the true answer's log-probability: trained B (its own stream) and the untrained readers
        fc = {}
        cands = [ep['cands']['B'] for ep in eps] if 'cands' in eps[0] else None
        answers = [ep['answers']['B'] for ep in eps]
        for name, rows, who in (('trained', net.rows(eps, msgs, own='B'), None),
                                ('trained_closed', net.rows(eps, mode='closed'), None),
                                ('base_reads', net.rows(eps, msgs), 'base'),
                                ('base_closed', net.rows(eps, mode='closed'), 'base'),
                                ('oracle', net.rows(eps, mode='oracle'), 'base')):
            with net.reader(who):
                picks, truth = net.choose(rows, cands or [[] for _ in eps], answers)
            fc[name] = dict(true_lp=round(mean(truth), 4))
            if cands:
                fc[name]['choice'] = round(mean([score(ep, 'B', p)['main'] for ep, p in zip(eps, picks)]), 4)
        res[g] = dict(per=per, forced=fc)
        log(dict(part='eval', game=g, s=round(time.monotonic() - t0), letters=round(letter_frac(msgs, net.letter), 4),
                 means={k: round(mean(v), 4) for k, v in per.items()}, forced=fc))
    return res, recs


@torch.no_grad()
def intercept(net, games, n, bs=32):
    """n training-value episodes (spread over games) and their hard messages, for the in-context shots and the crack
    pairs: list of (episode, ids [turns, m])."""
    net.eval()
    out = []
    for g in games:
        eps = test_set(g, max(SHOTS, n // len(games)), 'shots')
        msgs = []
        for part in batches(eps, bs):
            _, _, idx, _ = net.converse(part, hard=True)
            msgs.append(idx.cpu())
        out += list(zip(eps, torch.cat(msgs)))
    return out


def ic_block(tok, ep, ids, answer=None, closed=False):
    """One in-context example: B's view, the conversation (each message decoded by tok, quoted) and the answer."""
    lines = [f'What B knows:\n{ep["views"]["B"]}']
    if not closed:
        lines.append('Conversation:\n' + '\n'.join(
            f'{spk}: {json.dumps(tok.decode(ids[t].tolist()), ensure_ascii=False)}' for t, spk in enumerate(SPEAKERS)))
    lines.append('Answer:' + (f' {answer}\n\n' if answer is not None else ''))
    return '\n'.join(lines)


IC_HEAD = ('Two agents, A and B, solve tasks together by exchanging short messages. Each example below shows what B '
           'knows, their conversation, and the answer to B\'s task.\n\n')
IC_HEAD_CLOSED = 'Each example below shows what agent B knows and the answer to B\'s task.\n\n'


@torch.no_grad()
def incontext(args, games, shots, test, msgs, device, log, qtok, loader=None, bs=8):
    """Each of args.others (a model and tokenizer from loader(name)) given SHOTS training conversations of the game as
    text (the messages decoded by qtok, Qwen3-4B's tokenizer), then a test one, greedy until a newline; and the same
    without conversations."""
    loader = loader or _load_lm
    res = {}
    for name in args.others.split(','):
        model, tok = loader(name, device)
        short = name.split('/')[-1]
        for g in games:
            t0 = time.monotonic()
            pool = [(ep, ids) for ep, ids in shots if ep['game'] == g][:SHOTS]
            eps = test[g][:args.n_ic]
            for closed in (False, True):
                head = (IC_HEAD_CLOSED if closed else IC_HEAD) + ''.join(
                    ic_block(qtok, ep, ids, ep['answers']['B'], closed) for ep, ids in pool)
                prompts = [head + ic_block(qtok, ep, msgs[g][i], None, closed) for i, ep in enumerate(eps)]
                texts = []
                for part in batches(prompts, bs):
                    enc = tok(part, return_tensors='pt', padding=True).to(device)
                    gen = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
                    texts += [tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True) for o in gen]
                key = f'ic_{short}' + ('_closed' if closed else '')
                res.setdefault(g, {})[key] = dict(scores=[score(ep, 'B', t)['main'] for ep, t in zip(eps, texts)],
                                                  texts=[first_line(t) for t in texts])
            log(dict(part='incontext', model=short, game=g, s=round(time.monotonic() - t0),
                     means={k: round(mean(v['scores']), 4) for k, v in res[g].items() if k.startswith(f'ic_{short}')}))
        del model
        _free()
    return res


def _load_lm(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    return model, tok


def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


CLOSED = dict(trained='trained_cut', base_reads='base_closed')


def report(res, extra, games):
    """scores.json per game: every reader's mean main score (and the extras), the forced-choice numbers, the trained
    agents minus every other reader paired by episode with bootstrap 95% intervals, and each reader's gain over its
    own closed-book (trained: over cut; forced: in mean log-probability per answer token)."""
    out = {}
    for g in games:
        per = dict(res[g]['per'])
        for part in extra:
            for k, v in part.get(g, {}).items():
                per[k] = v['scores']
        means = {k: round(mean(v), 4) for k, v in per.items()}
        n = len(per['trained'])
        paired = {k: bootstrap(per['trained'][:len(v)], v) for k, v in per.items()
                  if k != 'trained' and not k.endswith(('_A', '_em', '_agree')) and len(v) <= n}
        gain = {k: round(means[k] - means[c], 4) for k, c in CLOSED.items() if c in means}
        gain.update({k: round(means[k] - means[k + '_closed'], 4) for k in means if k + '_closed' in means})
        fc = res[g]['forced']
        gain['forced_trained'] = round(fc['trained']['true_lp'] - fc['trained_closed']['true_lp'], 4)
        gain['forced_base_reads'] = round(fc['base_reads']['true_lp'] - fc['base_closed']['true_lp'], 4)
        out[g] = dict(n=n, means=means, forced=fc, paired=paired, gains=gain)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--parts', default='train,eval,incontext')
    ap.add_argument('--games', default=','.join(GAMES))
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--tok', default=None)
    ap.add_argument('--init', default=None, help='a checkpoint to start training from')
    ap.add_argument('--channel', default='jacobi', choices=('auto', 'jacobi', 'slots', 'kv'))
    ap.add_argument('--iters', type=int, default=0, help='jacobi: refinement passes per message (0: m - 1, exact)')
    ap.add_argument('--grad_all', action='store_true', help='jacobi: gradient through every pass, not only the last')
    ap.add_argument('--m', type=int, default=M, help='tokens per message')
    ap.add_argument('--mix', default='games', help='games: --games only; corpus: swarm/corpus.py tasks too')
    ap.add_argument('--old_share', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--micro', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--own_lr', type=float, default=3e-4, help="the slots' learning rate")
    ap.add_argument('--tau0', type=float, default=1.0)
    ap.add_argument('--tau1', type=float, default=0.3)
    ap.add_argument('--beta', type=float, default=0.0, help='the entropy penalty at the end of its ramp')
    ap.add_argument('--info', type=float, default=0.0, help="the anti-collapse term's weight (from the start)")
    ap.add_argument('--free', type=int, default=300, help='steps before the letter penalty starts')
    ap.add_argument('--lam0', type=float, default=0.1)
    ap.add_argument('--lam_every', type=int, default=200)
    ap.add_argument('--lam_max', type=float, default=50.0)
    ap.add_argument('--letter_goal', type=float, default=0.01)
    ap.add_argument('--ctrl', action='store_true', help='the letter weight as a feedback controller on the pooled gap '
                    '(from --lam0 at --free; see letter_ctrl) instead of the doubling schedule')
    ap.add_argument('--ctrl_up', type=float, default=1.5)
    ap.add_argument('--ctrl_down', type=float, default=0.5)
    ap.add_argument('--ctrl_keep', type=float, default=0.8)
    ap.add_argument('--ctrl_start', type=float, default=0.5, help='--ctrl starts only once the pooled gap is at least '
                    'this share of its step-0 value (until then the phase stays letter-free)')
    ap.add_argument('--bias_k', type=float, default=0.0, help='--ctrl: letter logits lowered by bias_k times the weight')
    ap.add_argument('--priv_letters', type=float, default=0.05, help='--ctrl: privacy starts under this letter share '
                    'with the gap held')
    ap.add_argument('--contrast', type=float, default=0.0, help="the contrastive term's weight (see Talk.loss)")
    ap.add_argument('--contrast_target', type=float, default=0.2, help='its hinge target, nats per answer token')
    ap.add_argument('--lam_p', type=float, default=0.3, help='the privacy weight at the end of its ramp')
    ap.add_argument('--priv_ramp', type=int, default=500)
    ap.add_argument('--priv_by', type=int, default=2000, help='privacy starts here at the latest')
    ap.add_argument('--keep', type=float, default=0.5, help='lam_p halves when on minus swap falls under this share')
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--dev_big', type=int, default=48)
    ap.add_argument('--dev_every', type=int, default=100)
    ap.add_argument('--full_every', type=int, default=100, help='devs at multiples of this add the soft numbers and '
                    "the base reader's gain (the others: hard on and swap only)")
    ap.add_argument('--dev_n', type=int, default=12)
    ap.add_argument('--dev_n_corpus', type=int, default=12)
    ap.add_argument('--max_minutes', type=float, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--n_ic', type=int, default=200)
    ap.add_argument('--shots', type=int, default=1000, help='intercepted training conversations (in-context shots '
                    'and transcripts.json)')
    ap.add_argument('--others', default=','.join(OTHERS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.jsonl', 'a')

    def log(rec):
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print(json.dumps(rec), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    parts = args.parts.split(',')
    games = args.games.split(',')
    ckpt = args.init if 'train' in parts else args.ckpt
    net = build(args.device, args.base, ckpt=ckpt, tok_name=args.tok, m=args.m)
    if ckpt is None:
        net.channel, net.iters, net.grad_last = args.channel, args.iters or args.m - 1, not args.grad_all
    log(dict(start=True, args=vars(args), channel=net.channel, m=net.m, allowed=int(net.allowed.sum()),
             letters=int(net.letter.sum())))
    if 'train' in parts:
        train(net, args, out, log)
    if not {'eval', 'incontext'} & set(parts):
        return
    if net.channel == 'kv':
        log(dict(error='the kv channel sends no tokens: nothing to evaluate'))
        sys.exit(3)
    res, recs = evaluate(net, args, out, log)
    msgs = {g: recs[g]['msgs'] for g in games}
    test = {g: test_set(g, args.n) for g in games}
    extra = []
    (out / 'scores.json').write_text(json.dumps(report(res, extra, games), indent=1))
    shots = intercept(net, games, args.shots)
    tr = dict(test={g: [dict(i=i, ids=msgs[g][i].tolist()) for i in range(len(msgs[g]))] for g in games}, shots={})
    for ep, ids in shots:
        rows = tr['shots'].setdefault(ep['game'], [])
        rows.append(dict(i=len(rows), ids=ids.tolist()))
    (out / 'transcripts.json').write_text(json.dumps(tr))
    if 'incontext' in parts:
        extra.append(incontext(args, games, shots, test, msgs, args.device, log, net.tok))
        (out / 'scores.json').write_text(json.dumps(report(res, extra, games), indent=1))
    examples = {}
    for g in games:
        rows = []
        for i in range(min(40, args.n)):
            ep = test[g][i]
            r = dict(views=ep['views'], answer=ep['answers']['B'], golds=ep['golds'] if g in QA else None,
                     msgs=[net.tok.decode(t) for t in msgs[g][i].tolist()], ids=msgs[g][i].tolist(),
                     readers={k: v[i] for k, v in recs[g]['texts'].items()})
            for part in extra:
                for k, v in part.get(g, {}).items():
                    if i < len(v['texts']):
                        r['readers'][k] = first_line(v['texts'][i])
            rows.append(r)
        examples[g] = rows
    (out / 'examples.json').write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    log(dict(done=True))


if __name__ == '__main__':
    main()
