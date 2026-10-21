"""Sessions: how the two models' token streams line up in time, for teacher-forced training and for generation.

Time runs in ticks. Both models read their prompts at tick -1; then at each tick each model reads a chunk of tokens
(possibly none). At every exchange block a token read at tick t reads every partner token read at a tick <= t
(link.py), so the teacher-forced joint pass over whole episodes (collate, link.Pair.joint) equals generation tick by
tick (generate, link.Pair.step).

Modes:
- duplex: both models write at once, one token per tick each, and neither sees the other's text (the scripted games).
- dialogue: the models take turns. The speaker writes one token per tick. The listener reads the speaker's text as a
  user message in its own tokenizer while it streams: at each tick it reads the tokens of the decoded text so far that
  can no longer change (all but the last), and the rest when the turn ends. A single writer whose partner only listens
  is a dialogue with one turn.

An episode may close the link after its first k turns (link_off_after = k): from the tick that opens turn k on, no
token reads its partner.

Both models use the ChatML format. Qwen3's assistant header carries an empty thinking block.
"""
import torch

from link import DEVICE, FAR, SIDES, inner, partner

THINK = {'quill': '<think>\n\n</think>\n\n', 'ember': ''}


def ids(tok, s):
    return tok(s, add_special_tokens=False).input_ids


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


class Timeline:
    """Builds one model pair's token streams, with the tick at which each token is read."""

    def __init__(self, toks):
        self.toks = toks
        self.seq = {x: [] for x in SIDES}
        self.time = {x: [] for x in SIDES}
        self.labels = {x: [] for x in SIDES}         # (position, token, weight)
        self.turn_t = []                             # the tick that opens each turn
        self.spans = {x: [] for x in SIDES}          # (turn, first label, end label) of each turn a model writes

    def add(self, x, toks, t):
        self.seq[x] += toks
        self.time[x] += [t] * len(toks)

    def label(self, x, token, w):
        self.labels[x].append((len(self.seq[x]) - 1, token, w))


def build(ep, toks):
    """Teacher-forced timeline of an episode.

    ep: mode ('duplex' or 'dialogue'), system {x}, task {x or None}, and
      duplex: reply {x: (text, weight)}; a model's reply text is written one token per tick from tick 0.
      dialogue: turns [(speaker, text, weight)], optionally link_off_after.
    Returns {x: dict(ids, time, labels, cut, spans)}: cut is the first tick whose tokens do not read the partner (or None);
    spans lists (turn, start, end) of the labels of each turn the model writes."""
    end = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
    tl = Timeline(toks)
    if ep['mode'] == 'duplex':
        for x in SIDES:
            p = prefix(toks[x], x, ep['system'][x], ep['task'][x]) + assistant_header(toks[x], x)
            text, w = ep['reply'][x]
            r = ids(toks[x], text) + [end[x]]
            tl.add(x, p[:-1], -1)
            tl.add(x, p[-1:], 0)
            tl.label(x, r[0], w)
            for j in range(len(r) - 1):
                tl.add(x, [r[j]], j + 1)
                tl.label(x, r[j + 1], w)
            tl.spans[x].append((0, 0, len(tl.labels[x])))
        return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], cut=None, spans=tl.spans[x])
                for x in SIDES}
    for x in SIDES:
        tl.add(x, prefix(toks[x], x, ep['system'][x], ep['task'][x]), -1)
    t = 0
    for k, (s, text, w) in enumerate(ep['turns']):
        lis = partner(s)
        content = ids(toks[s], text) + [end[s]]
        # the tick that opens a turn also reads the end of the previous turn (added at the same tick below)
        tl.turn_t.append(t)
        start = len(tl.labels[s])
        tl.add(s, assistant_header(toks[s], s), t)
        tl.label(s, content[0], w)
        tl.add(lis, ids(toks[lis], '<|im_start|>user\n'), t)
        t += 1
        got = []
        for j in range(len(content) - 1):
            tl.add(s, [content[j]], t)
            tl.label(s, content[j + 1], w)
            st = stable(toks[lis], toks[s].decode(content[:j + 1]))
            if st is not None and len(st) > len(got):
                tl.add(lis, st[len(got):], t)
                got = st
            t += 1
        tl.spans[s].append((k, start, len(tl.labels[s])))
        if k == len(ep['turns']) - 1:
            break
        full = ids(toks[lis], toks[s].decode(content[:-1]))
        tl.add(s, [content[-1]] + ids(toks[s], '\n'), t)
        tl.add(lis, full[len(got):] + ids(toks[lis], '<|im_end|>\n'), t)
    k = ep.get('link_off_after')
    cut = None if k is None else (tl.turn_t[k] if k < len(tl.turn_t) else FAR)
    return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], cut=cut, spans=tl.spans[x]) for x in SIDES}





def payload(ep):
    """{x: [(start, end)]}: the character spans of x's target reply that only the partner can supply (the relayed
    value, the partner's number and the sum, the partner's half of the fact, the looked-up word; co-reading games:
    meta['payload']). Duplex games, and dialogue co-reading games (co_talk), whose payload spans are in the first turn's
    text of the model that writes it."""
    g, m = ep['game'], ep['meta']
    if ep['mode'] != 'duplex':
        if g.startswith('co_') and m.get('payload'):
            return {x: [tuple(sp) for sp in spans] for x, spans in m['payload'].items() if ep['turns'][0][0] == x}
        return {}
    if g.startswith('co_'):
        return {x: [tuple(sp) for sp in spans] for x, spans in m['payload'].items()}
    text = {x: ep['reply'][x][0] for x in SIDES}

    def at(x, sub, after=''):
        i = text[x].find(after + sub)
        return [(i + len(after), i + len(after) + len(sub))] if i >= 0 else []
    if g.startswith('relay'):
        return {m['receiver']: at(m['receiver'], m['value'])}
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
    and in a dialogue episode the first speaker's label j predicts token j of the first turn."""
    out = {}
    for x, spans in payload(ep).items():
        if not spans:
            continue
        text = ep['reply'][x][0] if ep['mode'] == 'duplex' else ep['turns'][0][1]
        enc = toks[x](text, add_special_tokens=False, return_offsets_mapping=True)
        js = [j for j, (a, b) in enumerate(enc.offset_mapping) if any(a < e and b > s0 for s0, e in spans)]
        if js:
            out[x] = js
    return out

def collate(built, pads, owners=None):
    """built: list of build() outputs. Returns per side the batch link.Pair.joint reads: ids, att, pos, tick (FAR at
    padding) [B, L] and cut [B] (FAR: the link never closes), right-padded; and the flat labels: idx (into [B * L]),
    lab, w, the sides each label's loss trains (own, own_mask) and the episode of each label (row)."""
    B = len(built)
    owners = owners or [{x: frozenset(SIDES) for x in SIDES} for _ in range(B)]
    b = {}
    for x in SIDES:
        L = max(len(e[x]['ids']) for e in built)
        ids_ = torch.full((B, L), pads[x], dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        tick = torch.full((B, L), FAR, dtype=torch.long)
        idx, lab, w, own, row = [], [], [], [], []
        for i, e in enumerate(built):
            s = e[x]
            n = len(s['ids'])
            ids_[i, :n] = torch.tensor(s['ids'])
            att[i, :n] = 1
            tick[i, :n] = torch.tensor(s['time'])
            for p, tokid, wt in s['labels']:
                idx.append(i * L + p)
                lab.append(tokid)
                w.append(wt)
                own.append(owners[i][x])
                row.append(i)
        cut = torch.tensor([FAR if e[x]['cut'] is None else e[x]['cut'] for e in built])
        b[x] = dict(ids=ids_.to(DEVICE), att=att.to(DEVICE), pos=torch.arange(L, device=DEVICE)[None].expand(B, L).contiguous(),
                    tick=tick.to(DEVICE), cut=cut.to(DEVICE),
                    idx=torch.tensor(idx, device=DEVICE, dtype=torch.long), lab=torch.tensor(lab, device=DEVICE, dtype=torch.long),
                    w=torch.tensor(w, device=DEVICE, dtype=torch.float), own=own,
                    own_mask={o: torch.tensor([o in s for s in own], device=DEVICE, dtype=torch.float) for o in SIDES},
                    row=torch.tensor(row, device=DEVICE, dtype=torch.long))
    return b


def batches(n, bs):
    """(start, end) of nearly equal batches of at most bs of n episodes: no batch is a single episode unless n is 1, so
    a shuffled partner is always another episode."""
    parts = max(1, -(-n // bs))
    cuts = [round(k * n / parts) for k in range(parts + 1)]
    return list(zip(cuts, cuts[1:]))


# ----- generation -----

class Stream:
    """One episode's generation state for one model."""

    def __init__(self):
        self.pending = []          # tokens to read at the next tick
        self.speaking = False      # whether the last token read predicts this model's next token
        self.out = []              # tokens this model wrote in the current turn
        self.turns = []            # (speaker, text) of finished turns
        self.gates = []            # per tick: the gate of each token read (mean over the exchanges)
        self.gate_turn = []        # per tick: the turn under way (-1: the prompt)
        self.h = []                # (keep) per tick: the last hidden states of the tokens read
        self.p = []                # (keep) per tick: their gates at every exchange


@torch.no_grad()
def generate(pair, eps, max_turn=64, temperature=0.0, gen=None, mode=None, mute=None, force=None,
             link_off_after=None, keep=False, write_mode=None):
    """Runs both models through the episodes (all of the same mode) together, tick by tick, block by block, with KV
    caches (link.Pair.step).

    eps: episodes as for build(), with 'turns' giving the speaker order (and, if force, the texts); duplex episodes
    give 'reply' only for force. force: teacher-force the episodes' texts; a dialogue episode's 'forced' lists the
    turns that are always teacher-forced (for example a fixed question after free turns). mode, write_mode: ablations
    (link.Pair). mute: {x: [bool per episode]}: x reads nothing from its partner. link_off_after (dialogue): number of
    finished turns after which neither model reads the other (an episode's own link_off_after comes first). Returns
    per episode the texts and, per tick, the gate of each token read, prompt first (keep: also every token's last
    hidden state, 'h' {x: [n, d]}, and its gates at every exchange, 'p' {x: [n, n exchanges]}, in reading order)."""
    toks = pair.toks
    head = {x: inner(pair.models[x])[1] for x in SIDES}
    B = len(eps)
    dmode = eps[0]['mode']
    end = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
    st = {x: [Stream() for _ in range(B)] for x in SIDES}
    turn = [0] * B
    done = [False] * B
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
                st[x][i].pending, st[x][i].speaking = p[-1:], True
                p = p[:-1]
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
        s = eps[i]['turns'][turn[i]][0]
        lis = partner(s)
        st[s][i].pending += assistant_header(toks[s], s)
        st[s][i].speaking, st[lis][i].speaking = True, False
        st[s][i].out = []
        st[lis][i].pending += ids(toks[lis], '<|im_start|>user\n')
        got[i] = []

    for i in range(B):
        start_turn(i)
    for step in range(10 ** 6):
        if all(done):
            break
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
                if st[x][i].speaking:
                    k = int(nxt[i])
                    if force or (dmode == 'dialogue' and turn[i] in eps[i].get('forced', ())):
                        if dmode == 'duplex':
                            want = ids(toks[x], eps[i]['reply'][x][0]) + [end[x]]
                        else:
                            want = ids(toks[x], eps[i]['turns'][turn[i]][1]) + [end[x]]
                        k = want[len(st[x][i].out)] if len(st[x][i].out) < len(want) else end[x]
                    if len(st[x][i].out) >= max_turn:
                        k = end[x]
                    st[x][i].out.append(k)
                    st[x][i].pending = [k]
        # turn bookkeeping: listeners read stable text, finished turns close
        for i in range(B):
            if done[i]:
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
        if keep:
            res[-1]['h'] = {x: torch.cat(st[x][i].h) for x in SIDES}
            res[-1]['p'] = {x: torch.cat(st[x][i].p) for x in SIDES}
    return res
