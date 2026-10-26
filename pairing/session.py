"""Sessions: how the two models' token streams line up in time, for teacher-forced training and for generation.

Time runs in ticks. Both models read their prompts at tick -1; then at each tick each model reads a chunk of tokens
(possibly none). At every exchange block a token read at tick t reads every partner token read at a tick <= t
(link.py), so the teacher-forced joint pass over whole episodes (collate, link.Pair.joint) equals generation tick by
tick (generate, link.Pair.step).

Modes:
- pen: one shared assistant text, one tick per word; the author reads all shared text before writing.
  The inactive model catches up when it next takes the pen; both flush the final text at the end.
- duplex: both models write at once, one token per tick each, and neither model's input contains the other's text
  (the scripted games).
- dialogue: the models take turns. The speaker writes one token per tick. The listener reads the speaker's text as a
  user message in its own tokenizer while it streams: at each tick it reads the tokens of the decoded text so far that
  can no longer change (all but the last), and the rest when the turn ends. A single writer whose partner only listens
  is a dialogue with one turn.

Optional pause=k (duplex) or pause_turns={turn: k} (dialogue) inserts k silent ticks before the assistant header: at
each, both models read one unlabelled space token (one token ID per tick, never a re-tokenized run of spaces). The
header opens the reply at the following tick. Without pauses a duplex prompt's last token is read at tick 0, where
the reply starts.

An episode may close the link after its first k turns (link_off_after = k): from the tick that opens turn k on, no
token reads its partner.

Both models use the ChatML format. Qwen3's assistant header ends with an empty thinking block.
"""
import re

import torch

from link import DEVICE, FAR, SIDES, inner, partner

THINK = {'quill': '<think>\n\n</think>\n\n', 'ember': ''}


def ids(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def pause_count(ep, turn=None):
    k = (ep.get('pause', 0) if turn is None
         else ep.get('pause_turns', {}).get(turn, ep.get('pause_turns', {}).get(str(turn), 0)))
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('pause counts must be nonnegative integers')
    return k


def pause_token(tok):
    """One ordinary space, forced as an ID (never re-tokenize repeated spaces). No chat control semantics."""
    token = ids(tok, ' ')
    if len(token) != 1 or token[0] in tok.all_special_ids:
        raise ValueError('pause requires a single ordinary space token')
    return token


def prefix(tok, x, system, task=None):
    s = f'<|im_start|>system\n{system}<|im_end|>\n'
    if task:
        s += f'<|im_start|>user\n{task}<|im_end|>\n'
    return ids(tok, s)


def assistant_header(tok, x):
    return ids(tok, '<|im_start|>assistant\n' + THINK[x])


def stable(ltok, text, final=False):
    """The listener's tokens of the text so far that later text cannot change (all but the last)."""
    if not final and text.endswith('�'):
        return None
    t = ids(ltok, text)
    return t if final else t[:-1]


def content(ep, toks, x, turn=None):
    """Reply token IDs and which were sampled; replay keeps IDs instead of re-tokenizing decoded text."""
    if turn is None:
        tokens = ep.get('reply_tokens', {}).get(x)
        sampled = ep.get('reply_sampled', {}).get(x)
        text = ep['reply'][x][0]
    else:
        tokens = ep.get('turn_tokens', [None] * len(ep['turns']))[turn]
        sampled = ep.get('turn_sampled', [None] * len(ep['turns']))[turn]
        text = ep['turns'][turn][1]
    end = toks[x].convert_tokens_to_ids('<|im_end|>')
    tokens = ids(toks[x], text) + [end] if tokens is None else tokens
    sampled = [True] * len(tokens) if sampled is None else sampled
    if not tokens or tokens[-1] != end or len(tokens) != len(sampled):
        raise ValueError('replay tokens must end in im_end and have one sampling flag per token')
    return tokens, sampled


class Timeline:
    """Builds one model pair's token streams, with the tick at which each token is read."""

    def __init__(self, toks):
        self.toks = toks
        self.seq = {x: [] for x in SIDES}
        self.time = {x: [] for x in SIDES}
        self.labels = {x: [] for x in SIDES}         # (position, token, weight)
        self.policy = {x: [] for x in SIDES}         # whether each label was actually sampled
        self.turn_t = []                             # the tick that opens each turn
        self.spans = {x: [] for x in SIDES}          # (turn, first label, end label) of each turn a model writes

    def add(self, x, toks, t):
        self.seq[x] += toks
        self.time[x] += [t] * len(toks)

    def label(self, x, token, w, sampled=True):
        self.labels[x].append((len(self.seq[x]) - 1, token, w))
        self.policy[x].append(sampled)


def build(ep, toks):
    """Teacher-forced timeline of an episode.

    ep: mode ('duplex', 'dialogue' or 'pen'), system {x}, task {x or None}, and
      pen: text, optional pen_start (default quill), pen_chunk (positive words per author), pen_weight.
      duplex: reply {x: (text, weight)}; a model's reply text is written one token per tick from tick 0.
      dialogue: turns [(speaker, text, weight)], optionally link_off_after.
    replay() supplies exact reply/turn token IDs and sampling flags for policy updates.
    Returns {x: dict(ids, time, labels, cut, spans)}: cut is the first tick whose tokens do not read the partner (or
    None); spans lists (turn, start, end) of the labels of each turn the model writes."""
    if ep['mode'] == 'pen':
        return build_pen(ep, toks)
    tl = Timeline(toks)
    if ep['mode'] == 'duplex':
        pause = pause_count(ep)
        for x in SIDES:
            p = prefix(toks[x], x, ep['system'][x], ep['task'][x]) + assistant_header(toks[x], x)
            _, w = ep['reply'][x]
            r, sampled = content(ep, toks, x)
            header = assistant_header(toks[x], x) if pause else p[-1:]
            tl.add(x, p[:-len(header)], -1)
            for t in range(pause):
                tl.add(x, pause_token(toks[x]), t)
            tl.add(x, header, pause)
            tl.label(x, r[0], w, sampled[0])
            for j in range(len(r) - 1):
                tl.add(x, [r[j]], pause + j + 1)
                tl.label(x, r[j + 1], w, sampled[j + 1])
            tl.spans[x].append((0, 0, len(tl.labels[x])))
        return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], policy=tl.policy[x], cut=None,
                        spans=tl.spans[x]) for x in SIDES}
    for x in SIDES:
        tl.add(x, prefix(toks[x], x, ep['system'][x], ep['task'][x]), -1)
    t = 0
    for k, (s, text, w) in enumerate(ep['turns']):
        lis = partner(s)
        tokens, sampled = content(ep, toks, s, k)
        # the tick that opens a turn also reads the end of the previous turn (added at the same tick below)
        tl.turn_t.append(t)
        for _ in range(pause_count(ep, k)):
            for x in SIDES:
                tl.add(x, pause_token(toks[x]), t)
            t += 1
        start = len(tl.labels[s])
        tl.add(s, assistant_header(toks[s], s), t)
        tl.label(s, tokens[0], w, sampled[0])
        tl.add(lis, ids(toks[lis], '<|im_start|>user\n'), t)
        t += 1
        got = []
        for j in range(len(tokens) - 1):
            tl.add(s, [tokens[j]], t)
            tl.label(s, tokens[j + 1], w, sampled[j + 1])
            st = stable(toks[lis], toks[s].decode(tokens[:j + 1]))
            if st is not None and len(st) > len(got):
                tl.add(lis, st[len(got):], t)
                got = st
            t += 1
        tl.spans[s].append((k, start, len(tl.labels[s])))
        if k == len(ep['turns']) - 1:
            break
        full = ids(toks[lis], toks[s].decode(tokens[:-1]))
        tl.add(s, [tokens[-1]] + ids(toks[s], '\n'), t)
        tl.add(lis, full[len(got):] + ids(toks[lis], '<|im_end|>\n'), t)
    k = ep.get('link_off_after')
    cut = None if k is None else (tl.turn_t[k] if k < len(tl.turn_t) else FAR)
    return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], policy=tl.policy[x], cut=cut,
                    spans=tl.spans[x]) for x in SIDES}


def payload(ep):
    """{x: [(start, end)]}: the character spans of x's target reply that only the partner can supply (the relayed
    value, the partner's number and the sum, the partner's half of the fact, the looked-up word; co-reading games:
    meta['payload']). Duplex games, and dialogue co-reading games (co_talk, co_play), whose payload spans are in the
    text of the first turn the model writes."""
    if ep['mode'] == 'pen':
        return pen_payload(ep)
    g, m = ep['game'], ep['meta']
    if ep['mode'] != 'duplex':
        if g.startswith('co_') and m.get('payload'):
            return {x: [tuple(sp) for sp in spans] for x, spans in m['payload'].items()
                    if any(sp == x for sp, *_ in ep['turns'])}
        return {}
    if g.startswith('co_'):
        return {x: [tuple(sp) for sp in spans] for x, spans in m['payload'].items()}
    text = {x: ep['reply'][x][0] for x in SIDES}

    def at(x, sub, after=''):
        i = text[x].find(after + sub)
        return [(i + len(after), i + len(after) + len(sub))] if i >= 0 else []
    if g.startswith('relay'):
        leads = {'relay_word': ('The secret word is ', 'The word is '), 'relay_number': ('The number is ',),
                 'relay_phrase': ('The phrase is "',), 'relay_digits': ('The digits are ',)}[g]
        r = m['receiver']
        return {r: next((s for s in (at(r, m['value'], lead) for lead in leads) if s), [])}
    if g == 'merge_sum':
        v = m['values']
        return {x: at(x, str(v[partner(x)]), "'s is ") + at(x, str(m['sum']), 'sum is ') for x in SIDES}
    if g == 'merge_fact':
        n = m['has_name']
        return {n: at(n, m['city'], 'lives in '), partner(n): at(partner(n), m['name'])}
    if g == 'lookup':
        return {m['asker']: at(m['asker'], m['value'], 'the word ')}
    return {}


def payload_labels(ep, toks):
    """{x: indices into x's labels (build()) of the payload tokens}: in a duplex episode label j predicts reply token j,
    and in a dialogue episode x's spans index x's own first turn, whose token j label j predicts (build() lays out a
    model's labels turn by turn, over its own turns only)."""
    if ep['mode'] == 'pen':
        return pen_payload_labels(ep, toks)
    out = {}
    for x, spans in payload(ep).items():
        if not spans:
            continue
        text = ep['reply'][x][0] if ep['mode'] == 'duplex' else next(t for sp, t, *_ in ep['turns'] if sp == x)
        enc = toks[x](text, add_special_tokens=False, return_offsets_mapping=True)
        js = [j for j, (a, b) in enumerate(enc.offset_mapping) if any(a < e and b > s0 for s0, e in spans)]
        if js:
            out[x] = js
    return out


def collate(built, pads, owners=None):
    """built: list of build() outputs. Returns per side the batch link.Pair.joint reads: ids, att, pos, tick (FAR at
    padding) [B, L] and cut [B] (FAR: the link never closes), right-padded; and the flat labels: idx (into [B * L]),
    lab, w, the sides each label's loss trains (own, own_mask) and the batch index of each label's episode (row)."""
    B = len(built)
    owners = owners or [{x: frozenset(SIDES) for x in SIDES} for _ in range(B)]
    b = {}
    for x in SIDES:
        L = max(len(e[x]['ids']) for e in built)
        ids_ = torch.full((B, L), pads[x], dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        tick = torch.full((B, L), FAR, dtype=torch.long)
        idx, lab, w, own, row, pg = [], [], [], [], [], []
        for i, e in enumerate(built):
            s = e[x]
            n = len(s['ids'])
            ids_[i, :n] = torch.tensor(s['ids'])
            att[i, :n] = 1
            tick[i, :n] = torch.tensor(s['time'])
            pg.extend(s.get('policy', [True] * len(s['labels'])))
            for p, tokid, wt in s['labels']:
                idx.append(i * L + p)
                lab.append(tokid)
                w.append(wt)
                own.append(owners[i][x])
                row.append(i)
        cut = torch.tensor([FAR if e[x]['cut'] is None else e[x]['cut'] for e in built])
        b[x] = dict(ids=ids_.to(DEVICE), att=att.to(DEVICE),
                    pos=torch.arange(L, device=DEVICE)[None].expand(B, L).contiguous(),
                    tick=tick.to(DEVICE), cut=cut.to(DEVICE),
                    idx=torch.tensor(idx, device=DEVICE, dtype=torch.long),
                    lab=torch.tensor(lab, device=DEVICE, dtype=torch.long),
                    w=torch.tensor(w, device=DEVICE, dtype=torch.float), own=own,
                    pg=torch.tensor(pg, device=DEVICE, dtype=torch.float),
                    own_mask={o: torch.tensor([o in s for s in own], device=DEVICE, dtype=torch.float) for o in SIDES},
                    row=torch.tensor(row, device=DEVICE, dtype=torch.long))
    return b


def batches(n, bs):
    """(start, end) of nearly equal batches of at most bs of n episodes: no batch is a single episode unless n is 1, so
    a shuffled partner is always another episode."""
    if n < 0 or bs < 1:
        raise ValueError('n must be nonnegative and bs positive')
    if n == 0:
        return []
    parts = -(-n // bs)
    if n > 1 and parts > n // 2:
        raise ValueError('batch size cannot partition these episodes without a singleton; increase bs')
    cuts = [round(k * n / parts) for k in range(parts + 1)]
    return list(zip(cuts, cuts[1:]))


# ----- generation -----

class Stream:
    """One episode's generation state for one model."""

    def __init__(self):
        self.pending = []          # tokens to read at the next tick
        self.speaking = False      # whether the last token read predicts this model's next token
        self.out = []              # tokens this model wrote in the current turn
        self.sampled = []          # false for forced tokens and the artificial length-limit EOS
        self.turns = []            # (speaker, text) of finished turns
        self.turn_tokens = []
        self.turn_sampled = []
        self.gates = []            # per tick: the gate of each token read (mean over the exchanges)
        self.gate_turn = []        # per tick: the turn under way (-1: the prompt)
        self.h = []                # (keep) per tick: the last hidden states of the tokens read
        self.p = []                # (keep) per tick: their gates at every exchange


def replay(ep, result):
    """The generated replies as an episode for training, retaining any later quiz turns."""
    if ep['mode'] == 'pen':
        return dict(ep, text=result['text'], pen_tokens=result['tokens'], pen_sampled=result['sampled'])
    if ep['mode'] == 'duplex':
        return dict(ep, reply={x: (result['text'][x], 1.0) for x in SIDES},
                    reply_tokens=result['tokens'], reply_sampled=result['sampled'])
    n = len(result['turns'])
    return dict(ep, turns=[(x, text, ep['turns'][j][2]) for j, (x, text) in enumerate(result['turns'])]
                + ep['turns'][n:], turn_tokens=result['tokens'] + [None] * (len(ep['turns']) - n),
                turn_sampled=result['sampled'] + [None] * (len(ep['turns']) - n))


@torch.no_grad()
def generate(pair, eps, max_turn=64, temperature=0.0, gen=None, mode=None, mute=None, force=None,
             link_off_after=None, keep=False, write_mode=None, keep_tokens=False):
    """Runs both models through the episodes (all of the same mode) together, tick by tick, block by block, with KV
    caches (link.Pair.step).

    eps: episodes as for build(), with 'turns' giving the speaker order (and, if force, the texts); duplex episodes
    give 'reply' only for force. force: teacher-force the episodes' texts; a dialogue episode's 'forced' lists the
    turns that are always teacher-forced (for example a fixed question after free turns). mode, write_mode: ablations
    (link.Pair). mute: {x: [bool per episode]}: x reads nothing from its partner. link_off_after (dialogue): number of
    finished turns after which neither model reads the other (an episode's own link_off_after comes first). Returns
    per episode the texts and, per tick, the gate of each token read, prompt first (keep: also every token's last
    hidden state, 'h' {x: [n, d]}, and its gates at every exchange, 'p' {x: [n, n exchanges]}, in reading order).
    keep_tokens (also enabled by keep) retains exact emitted IDs and sampling flags for replay()."""
    if eps and eps[0]['mode'] == 'pen':
        return generate_pen(pair, eps, max_turn, temperature, gen, mode, mute, force, keep, write_mode, keep_tokens)
    toks = pair.toks
    head = {x: inner(pair.models[x])[1] for x in SIDES}
    B = len(eps)
    if mode == 'shuffle' and B < 2:
        raise ValueError('shuffle needs at least two episodes')
    dmode = eps[0]['mode']
    end = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
    st = {x: [Stream() for _ in range(B)] for x in SIDES}
    turn = [0] * B
    done = [False] * B
    waiting = [pause_count(ep) if dmode == 'duplex' else 0 for ep in eps]
    resume = {x: [[] for _ in eps] for x in SIDES}
    got = [[] for _ in range(B)]                  # listener tokens read so far in the current turn
    off = [ep.get('link_off_after', link_off_after) if dmode == 'dialogue' else None for ep in eps]
    cache = pair.new_cache()
    att, pos = {}, {}

    def reads(x, i):
        """Whether x's tokens read the partner at this tick (the prompt's tick comes before any cut)."""
        if mute is not None and x in mute and mute[x][i]:
            return False
        return not (step >= 0 and off[i] is not None and turn[i] >= off[i])

    def collect(x, i, mask, hid, g):
        v = mask.bool()
        st[x][i].gates.append([round(float(p), 3) for p in g[v].mean(-1)])
        st[x][i].gate_turn.append(turn[i] if step >= 0 else -1)
        if keep:
            st[x][i].h.append(hid[v].float().cpu())
            st[x][i].p.append(g[v].float().cpu())

    # the prompts, left-padded, at tick -1
    step = -1
    chunk, qm, P = {}, {}, {}
    for x in SIDES:
        P[x] = []
        for i, ep in enumerate(eps):
            p = prefix(toks[x], x, ep['system'][x], ep['task'][x])
            if dmode == 'duplex':
                p = p + assistant_header(toks[x], x)
                header = assistant_header(toks[x], x) if waiting[i] else p[-1:]
                st[x][i].pending, st[x][i].speaking = header, True
                p = p[:-len(header)]
            P[x].append(p)
        Lp = max(map(len, P[x]))
        pad = toks[x].pad_token_id
        ids_ = torch.tensor([[pad] * (Lp - len(p)) + p for p in P[x]], device=DEVICE)
        att[x] = torch.tensor([[0] * (Lp - len(p)) + [1] * len(p) for p in P[x]], device=DEVICE)
        pos[x] = att[x].sum(-1)
        chunk[x] = (ids_, att[x], (att[x].cumsum(-1) - 1).clamp(min=0))
        qm[x] = att[x].float() * torch.tensor([float(reads(x, i)) for i in range(B)], device=DEVICE)[:, None]
    hid, gates = pair.step(chunk, cache, qm, -1, mode, gen, write_mode)
    for x in SIDES:
        for i in range(B):
            collect(x, i, att[x][i], hid[x][i], gates[x][i])

    def start_turn(i):
        if dmode != 'dialogue':
            return
        if turn[i] >= len(eps[i]['turns']):
            done[i] = True
            return
        waiting[i] = pause_count(eps[i], turn[i])
        s = eps[i]['turns'][turn[i]][0]
        lis = partner(s)
        resume[s][i] = assistant_header(toks[s], s)
        st[s][i].speaking, st[lis][i].speaking = True, False
        st[s][i].out = []
        st[s][i].sampled = []
        resume[lis][i] = ids(toks[lis], '<|im_start|>user\n')
        if not waiting[i]:
            for x in SIDES:
                st[x][i].pending += resume[x][i]
                resume[x][i] = []
        got[i] = []

    for i in range(B):
        start_turn(i)
        if dmode == 'duplex' and waiting[i]:
            for x in SIDES:
                resume[x][i], st[x][i].pending = st[x][i].pending, []
    for step in range(10 ** 6):
        if all(done):
            break
        paused = [bool(k) for k in waiting]
        for i in range(B):
            if paused[i]:
                for x in SIDES:
                    st[x][i].pending += pause_token(toks[x])
        chunk, qm, masks, chunks = {}, {}, {}, {}
        for x in SIDES:
            chunks[x] = [st[x][i].pending if not done[i] else [] for i in range(B)]
            K = max(map(len, chunks[x]))
            if K == 0:
                continue
            pad = toks[x].pad_token_id
            inp = torch.tensor([c + [pad] * (K - len(c)) for c in chunks[x]], device=DEVICE)
            masks[x] = torch.tensor([[1] * len(c) + [0] * (K - len(c)) for c in chunks[x]], device=DEVICE)
            att[x] = torch.cat([att[x], masks[x]], 1)
            chunk[x] = (inp, att[x], pos[x][:, None] + torch.arange(K, device=DEVICE)[None])
            qm[x] = masks[x].float() * torch.tensor([float(reads(x, i)) for i in range(B)], device=DEVICE)[:, None]
        hid, gates = pair.step(chunk, cache, qm, step, mode, gen, write_mode)
        for x in chunk:
            last = (masks[x].sum(-1) - 1).clamp(min=0)
            logits = head[x](hid[x][torch.arange(B, device=DEVICE), last]).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            for i in range(B):
                if not chunks[x][i]:
                    continue
                collect(x, i, masks[x][i], hid[x][i], gates[x][i])
                pos[x][i] += len(chunks[x][i])
                st[x][i].pending = []
                if st[x][i].speaking and not paused[i]:
                    k = int(nxt[i])
                    forced = force or (dmode == 'dialogue' and turn[i] in eps[i].get('forced', ()))
                    sampled = not forced and len(st[x][i].out) < max_turn
                    if forced:
                        want, _ = content(eps[i], toks, x, None if dmode == 'duplex' else turn[i])
                        k = want[len(st[x][i].out)] if len(st[x][i].out) < len(want) else end[x]
                    if not forced and len(st[x][i].out) >= max_turn:
                        k = end[x]
                    st[x][i].out.append(k)
                    st[x][i].sampled.append(sampled)
                    st[x][i].pending = [k]
        # turn bookkeeping: listeners read stable text, finished turns close
        for i in range(B):
            if done[i]:
                continue
            if paused[i]:
                waiting[i] -= 1
                if not waiting[i]:
                    for x in SIDES:
                        st[x][i].pending += resume[x][i]
                        resume[x][i] = []
                continue
            if dmode == 'duplex':
                if all(st[x][i].out and st[x][i].out[-1] == end[x] for x in SIDES):
                    done[i] = True
                for x in SIDES:
                    if st[x][i].out and st[x][i].out[-1] == end[x]:
                        st[x][i].speaking, st[x][i].pending = False, []
                continue
            s = eps[i]['turns'][turn[i]][0]
            lis = partner(s)
            out = st[s][i].out
            if out and out[-1] == end[s]:
                text = toks[s].decode(out[:-1])
                full = ids(toks[lis], text)
                st[s][i].pending = [end[s]] + ids(toks[s], '\n')
                st[lis][i].pending += full[len(got[i]):] + ids(toks[lis], '<|im_end|>\n')
                for x in SIDES:
                    st[x][i].turns.append((s, text))
                    st[x][i].turn_tokens.append(list(out))
                    st[x][i].turn_sampled.append(list(st[s][i].sampled))
                turn[i] += 1
                start_turn(i)
            else:
                stb = stable(toks[lis], toks[s].decode(out))
                if stb is not None and len(stb) > len(got[i]):
                    st[lis][i].pending += stb[len(got[i]):]
                    got[i] = stb
    res = []
    for i in range(B):
        if dmode == 'duplex':
            texts = {x: toks[x].decode([k for k in st[x][i].out if k != end[x]]) for x in SIDES}
            res.append(dict(text=texts, gates={x: st[x][i].gates for x in SIDES}))
        else:
            res.append(dict(turns=st[SIDES[0]][i].turns, gates={x: st[x][i].gates for x in SIDES},
                            gate_turn={x: st[x][i].gate_turn for x in SIDES}))
        if keep_tokens or keep:
            res[-1]['tokens'] = ({x: st[x][i].out for x in SIDES} if dmode == 'duplex'
                                 else st[SIDES[0]][i].turn_tokens)
            res[-1]['sampled'] = ({x: st[x][i].sampled for x in SIDES} if dmode == 'duplex'
                                  else st[SIDES[0]][i].turn_sampled)
        if keep:
            res[-1]['h'] = {x: torch.cat(st[x][i].h) for x in SIDES}
            res[-1]['p'] = {x: torch.cat(st[x][i].p) for x in SIDES}
    return res


def pen_author(ep, word):
    """The model that writes word number word of a shared-pen episode (authors alternate every pen_chunk words)."""
    chunk = ep.get('pen_chunk', 1)
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError('pen_chunk must be a positive integer')
    start = ep.get('pen_start', SIDES[0])
    if start not in SIDES:
        raise ValueError('pen_start must name a model')
    return SIDES[(SIDES.index(start) + word // chunk) % 2]


def pen_words(text):
    """The end offset of each word of text. Whitespace belongs to the following word; trailing whitespace is an
    unscored final flush."""
    return [m.end() for m in re.finditer(r'\S+', text)]


def pen_prefix(tok, text, committed):
    """tok's final tokens of text; an error when they do not start with the tokens already committed."""
    full = stable(tok, text, final=True)
    if full[:len(committed)] != committed:
        raise ValueError('pen word boundary revises committed tokenizer tokens')
    return full


def build_pen(ep, toks):
    """One tick per word, only its author advances. The next author catches up in its own tokenizer.

    The inactive model does not consume the unfinished word: otherwise its same-tick states would leak future
    subword tokens back to the author in joint(). Both streams are the complete shared assistant text, without
    extra headers. The listener waits for a complete word, then uses dialogue's final stable-prefix flush.
    """
    text = ep['text']
    pen_author(ep, 0)  # validate even for an empty text
    tl = Timeline(toks)
    got, header = {}, {}
    for x in SIDES:
        p = prefix(toks[x], x, ep['system'][x], ep['task'][x]) + assistant_header(toks[x], x)
        tl.add(x, p[:-1], -1)
        got[x], header[x] = [], p[-1:]
    before = 0
    ends = pen_words(text)
    for j, after in enumerate(ends):
        s = pen_author(ep, j)
        prior = pen_prefix(toks[s], text[:before], got[s])
        full = pen_prefix(toks[s], text[:after], prior)
        target = full[len(prior):]
        if not target:
            raise ValueError('a pen word must produce at least one token')
        if 'pen_tokens' in ep and ep['pen_tokens'][j] != target:
            raise ValueError('pen replay IDs must match the shared text in the author tokenizer')
        sampled = ep.get('pen_sampled', [None] * len(ends))[j]
        sampled = [True] * len(target) if sampled is None else sampled
        if len(sampled) != len(target):
            raise ValueError('one pen sampling flag is required per word token')
        tl.add(s, header[s] + prior[len(got[s]):], j)
        header[s] = []
        first = len(tl.labels[s])
        for z, token in enumerate(target):
            tl.label(s, token, ep.get('pen_weight', 1.0), sampled[z])
            tl.add(s, [token], j)
        tl.spans[s].append((j, first, len(tl.labels[s])))
        got[s] = full
        before = after
    for x in SIDES:
        full = pen_prefix(toks[x], text, got[x])
        tl.add(x, header[x] + full[len(got[x]):] + [toks[x].convert_tokens_to_ids('<|im_end|>')], len(ends))
    return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], policy=tl.policy[x], cut=None,
                    spans=tl.spans[x]) for x in SIDES}


def pen_payload(ep):
    """Secret character spans authored by the non-holder (including repeated occurrences)."""
    if ep.get('game') != 'pen_secret':
        return {}
    receiver = partner(ep['meta']['holder'])
    text, value = ep['text'], ep['meta']['value']
    spans = []
    for j, word in enumerate(re.finditer(r'\S+', text)):
        if pen_author(ep, j) == receiver:
            spans += [(word.start() + m.start(), word.start() + m.end())
                      for m in re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)', word.group())]
    return {receiver: spans} if spans else {}


def pen_payload_labels(ep, toks):
    built, out = build_pen(ep, toks), {}
    for x, spans in pen_payload(ep).items():
        enc = toks[x](ep['text'], add_special_tokens=False, return_offsets_mapping=True)
        end = pen_words(ep['text'])
        # A model's labels include its own words only, while offsets index the whole shared text.
        label = 0
        chosen = []
        for j, stop in enumerate(end):
            if pen_author(ep, j) != x:
                continue
            before = 0 if j == 0 else end[j - 1]
            a, b = len(ids(toks[x], ep['text'][:before])), len(ids(toks[x], ep['text'][:stop]))
            for ti in range(a, b):
                lo, hi = enc.offset_mapping[ti]
                if any(lo < z and hi > s for s, z in spans):
                    chosen.append(label)
                label += 1
        assert label == len(built[x]['labels'])
        if chosen:
            out[x] = chosen
    return out


@torch.no_grad()
def generate_pen(pair, eps, max_turn=64, temperature=0.0, gen=None, mode=None, mute=None, force=None,
                 keep=False, write_mode=None, keep_tokens=False):
    """Autoregressive shared pen using Pair.step. max_turn limits words; pen_max_tokens bounds each word.

    Only the author runs during a word, so its partner cache is fixed while subword tokens are sampled at the same
    tick. At a handoff the next author reads all accumulated shared text. Free sampling rejects special tokens,
    incomplete Unicode and noncanonical encodings; a proposed next-word token hands over the pen without being
    consumed. Boundary proposals and the final chat terminator have no word labels.
    """
    if mode == 'shuffle':
        raise ValueError('pen generation cannot shuffle asynchronous authors; use joint() for shuffled payload loss')
    results = []
    for row, ep in enumerate(eps):
        toks, cache = pair.toks, pair.new_cache()
        got, header = {x: [] for x in SIDES}, {}
        states, gate_states, gates = ({x: [] for x in SIDES} for _ in range(3))
        count = {x: 0 for x in SIDES}
        last = {}

        def read(chunks, tick):
            chunk, qm = {}, {}
            for x, token in chunks.items():
                if not token:
                    continue
                n = len(token)
                inp = torch.tensor([token], device=DEVICE)
                att = torch.ones(1, count[x] + n, dtype=torch.long, device=DEVICE)
                pos = torch.arange(count[x], count[x] + n, device=DEVICE)[None]
                chunk[x] = inp, att, pos
                qm[x] = torch.full((1, n), float(not (mute and x in mute and mute[x][row])), device=DEVICE)
                count[x] += n
            h, g = pair.step(chunk, cache, qm, tick, mode, gen, write_mode)
            for x in h:
                last[x] = h[x][:, -1:]
                gates[x].append([round(float(v), 3) for v in g[x][0].mean(-1)])
                if keep:
                    states[x].append(h[x][0].float().cpu())
                    gate_states[x].append(g[x][0].float().cpu())
            return h

        prompts = {}
        for x in SIDES:
            p = prefix(toks[x], x, ep['system'][x], ep['task'][x]) + assistant_header(toks[x], x)
            prompts[x], header[x] = p[:-1], p[-1:]
        read(prompts, -1)
        text, words, tokens, sampled = '', [], [], []
        ends = pen_words(ep.get('text', '')) if force else None
        limit = len(ends) if force else min(max_turn, ep.get('pen_words', max_turn))
        for j in range(limit):
            s = pen_author(ep, j)
            prior = pen_prefix(toks[s], text, got[s])
            pending = header[s] + prior[len(got[s]):]
            if pending:
                read({s: pending}, j)
            h = last[s]
            header[s] = []
            got[s] = prior[:]
            out = []
            if force:
                wanted = pen_prefix(toks[s], ep['text'][:ends[j]], prior)[len(prior):]
                if 'pen_tokens' in ep and ep['pen_tokens'][j] != wanted:
                    raise ValueError('pen replay IDs do not match text')
                cap = len(wanted)
            else:
                cap = ep.get('pen_max_tokens', 32)
                if cap < 1:
                    raise ValueError('pen_max_tokens must be positive')
            for z in range(cap):
                if force:
                    token = wanted[z]
                else:
                    logits = inner(pair.models[s])[1](h[0, -1]).float().clone()
                    # Constrain to a canonical, complete-Unicode word in this tokenizer, not a fixed vocabulary.
                    while True:
                        if not torch.isfinite(logits).any():
                            raise ValueError('no canonical pen continuation')
                        token = int(torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)) \
                            if temperature > 0 else int(logits.argmax())
                        if out and token == toks[s].convert_tokens_to_ids('<|im_end|>'):
                            break
                        candidate = toks[s].decode(prior + out + [token], skip_special_tokens=False)
                        suffix = candidate[len(text):] if candidate.startswith(text) else ''
                        boundary = bool(out) and re.search(r'\S\s', suffix) is not None
                        if boundary and token not in toks[s].all_special_ids:
                            token = toks[s].convert_tokens_to_ids('<|im_end|>')
                            break
                        valid = (token not in toks[s].all_special_ids and suffix and '�' not in candidate
                                 and len(pen_words(suffix)) == 1 and not suffix[-1].isspace()
                                 and (not text or suffix[0].isspace())
                                 and ids(toks[s], candidate) == prior + out + [token])
                        if valid:
                            break
                        logits[token] = -float('inf')
                    if token == toks[s].convert_tokens_to_ids('<|im_end|>'):
                        break
                out.append(token)
                h = read({s: [token]}, j)[s]
                got[s].append(token)
            if not out:
                raise ValueError('pen word must have at least one token')
            new_text = toks[s].decode(prior + out)
            if force:
                new_text = ep['text'][:ends[j]]
            words.append((s, new_text[len(text):]))
            text = new_text
            tokens.append(out)
            sampled.append([not force] * len(out))
        if force:
            text = ep['text']
        flush = {}
        for x in SIDES:
            full = pen_prefix(toks[x], text, got[x])
            flush[x] = header[x] + full[len(got[x]):] + [toks[x].convert_tokens_to_ids('<|im_end|>')]
        read(flush, len(words))
        result = dict(text=text, words=words, gates=gates)
        if keep or keep_tokens:
            result.update(tokens=tokens, sampled=sampled)
        if keep:
            result.update(h={x: torch.cat(states[x]) for x in SIDES}, p={x: torch.cat(gate_states[x]) for x in SIDES})
        results.append(result)
    return results
