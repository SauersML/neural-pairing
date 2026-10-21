"""Sessions: how the two models' token streams line up in time, for teacher-forced training and for generation.

Time runs in steps. At each step each model reads a chunk of tokens (possibly none). Every token a model reads after
its prefilled prompt sends a message (and a gate); a message read at step t reaches the partner at the first token the
partner reads at a later step, where it is added after block LR. Messages that reach the same token are averaged.

Modes:
- duplex: both models write at once, one token per step each, and neither sees the other's text (the scripted games).
- dialogue: the models take turns. The speaker writes one token per step. The listener reads the speaker's text as a
  user message in its own tokenizer while it streams: at each step it reads the tokens of the decoded text so far that
  can no longer change (all but the last), and the rest when the turn ends. A single writer whose partner only listens
  is a dialogue with one turn.

An episode may close the link after its first k turns (link_off_after = k): from the step that opens turn k on, no
message is delivered.

Both models use the ChatML format. Qwen3's assistant header carries an empty thinking block.
"""
import torch

from link import DEVICE, SIDES, partner

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
    """Builds one model pair's token streams, with the step at which each token is read."""

    def __init__(self, toks):
        self.toks = toks
        self.seq = {x: [] for x in SIDES}
        self.time = {x: [] for x in SIDES}
        self.labels = {x: [] for x in SIDES}         # (position, token, weight)
        self.turn_t = []                             # the step that opens each turn
        self.spans = {x: [] for x in SIDES}          # (turn, first label, end label) of each turn a model writes

    def add(self, x, toks, t):
        self.seq[x] += toks
        self.time[x] += [t] * len(toks)

    def label(self, x, token, w):
        self.labels[x].append((len(self.seq[x]) - 1, token, w))


def build(ep, toks):
    """Teacher-forced timeline of an episode.

    ep: mode ('duplex' or 'dialogue'), system {x}, task {x or None}, and
      duplex: reply {x: (text, weight)}; a model's reply text is written one token per step from step 0.
      dialogue: turns [(speaker, text, weight)], optionally link_off_after.
    Returns {x: dict(ids, time, labels, cut, spans)}: cut is the first step whose messages are not delivered (or None);
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
        # the step that opens a turn also reads the end of the previous turn (added at the same step below)
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
    cut = None if k is None else (tl.turn_t[k] if k < len(tl.turn_t) else 10 ** 9)
    return {x: dict(ids=tl.seq[x], time=tl.time[x], labels=tl.labels[x], cut=cut, spans=tl.spans[x]) for x in SIDES}


def destinations(e, x):
    """For each token model x reads after its prompt (each message it sends), the partner position the message reaches:
    the first partner token read at a later step, or -1 (none, or the link closed)."""
    y = partner(x)
    tx = torch.tensor([t for t in e[x]['time'] if t >= 0])
    if not len(tx):
        return torch.zeros(0, dtype=torch.long)
    ty = torch.tensor(e[y]['time'])
    later = ty[None, :] > tx[:, None]                            # [sends, partner positions]
    first = torch.where(later.any(1), later.float().argmax(1), torch.full_like(tx, -1))
    if e[x]['cut'] is not None:
        first[tx >= e[x]['cut']] = -1
    return first


def layout(e, soft):
    """{y: (position of each of y's tokens in its input, [(slot position, partner message index)], input length)}.
    With the soft read every partner message that reaches token p takes a slot just before p, in sending order."""
    out = {}
    for y in SIDES:
        n = len(e[y]['ids'])
        at = [[] for _ in range(n)]
        if soft:
            for j, d in enumerate(destinations(e, partner(y)).tolist()):
                if d >= 0:
                    at[d].append(j)
        pos, slots, k = [], [], 0
        for p in range(n):
            for j in at[p]:
                slots.append((k, j))
                k += 1
            pos.append(k)
            k += 1
        out[y] = (pos, slots, k)
    return out


def collate(built, pads, owners=None, soft=False):
    """built: list of build() outputs. Returns the batch Pair.coupled reads: per side ids, att, send positions, their
    destinations in the partner's sequence (first partner position read at a later step), and flat labels. With the
    soft read (soft=True) the inputs also hold the slots: slot gives the partner message index at each slot (-1 at
    tokens), att_cut and pos_cut mask the slots out and number the tokens as if the slots were absent, and next_tok
    gives the token after each sending position."""
    B = len(built)
    owners = owners or [{x: frozenset(SIDES) for x in SIDES} for _ in range(B)]
    lay = [layout(e, soft) for e in built]
    b = {}
    for x in SIDES:
        L = max(lay[i][x][2] for i in range(B))
        ids_ = torch.full((B, L), pads[x], dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        slot = torch.full((B, L), -1, dtype=torch.long)
        T = max(sum(t >= 0 for t in e[x]['time']) for e in built)
        send = torch.zeros((B, T), dtype=torch.long)
        valid = torch.zeros((B, T), dtype=torch.bool)
        nxt = torch.full((B, T), -1, dtype=torch.long)
        idx, lab, w, own, row = [], [], [], [], []
        for i, e in enumerate(built):
            s = e[x]
            pos, slots, n = lay[i][x]
            pos = torch.tensor(pos, dtype=torch.long)
            ids_[i, pos] = torch.tensor(s['ids'])
            att[i, :n] = 1
            for k, j in slots:
                slot[i, k] = j
            act = [p for p, t in enumerate(s['time']) if t >= 0]
            send[i, :len(act)] = pos[act]
            valid[i, :len(act)] = True
            nxt[i, :len(act)] = torch.tensor([s['ids'][p + 1] if p + 1 < len(s['ids']) else -1 for p in act],
                                             dtype=torch.long)
            for p, tokid, wt in s['labels']:
                idx.append(i * L + int(pos[p]))
                lab.append(tokid)
                w.append(wt)
                own.append(owners[i][x])
                row.append(i)
        cut = att * (slot < 0)
        b[x] = dict(ids=ids_.to(DEVICE), att=att.to(DEVICE), send_pos=send.to(DEVICE), send_valid=valid.to(DEVICE),
                    idx=torch.tensor(idx, device=DEVICE), lab=torch.tensor(lab, device=DEVICE),
                    w=torch.tensor(w, device=DEVICE, dtype=torch.float), own=own,
                    own_mask={o: torch.tensor([o in s for s in own], device=DEVICE, dtype=torch.float) for o in SIDES},
                    row=torch.tensor(row, device=DEVICE), slot=slot.to(DEVICE), att_cut=cut.to(DEVICE),
                    pos=torch.arange(L, device=DEVICE)[None].expand(B, L).contiguous(),
                    pos_cut=(cut.cumsum(1) - 1).clamp(min=0).to(DEVICE), next_tok=nxt.to(DEVICE))
    for x in SIDES:
        T = b[x]['send_pos'].shape[1]
        dst = torch.full((B, T), -1, dtype=torch.long)
        for i, e in enumerate(built):
            d = destinations(e, x)
            dst[i, :len(d)] = d
        b[x]['dst'] = dst.to(DEVICE)
    return b


# ----- generation -----

class Stream:
    """One episode's generation state for one model."""

    def __init__(self):
        self.pending = []          # tokens to read at the next step
        self.speaking = False      # whether the last token read predicts this model's next token
        self.out = []              # tokens this model wrote in the current turn
        self.turns = []            # (speaker, text) of finished turns
        self.gates = []            # per step: list of gate probabilities of the tokens read
        self.gate_turn = []        # per step: the turn under way (dialogue)
        self.p = []                # (keep) per step: the unrounded gate probabilities


@torch.no_grad()
def generate(pair, eps, max_turn=64, temperature=0.0, gen=None, mode=None, mute=None, force=None,
             link_off_after=None, keep=False, write_mode=None):
    """Runs both models through the episodes (all of the same mode), tick by tick with KV caches.

    eps: episodes as for build(), with 'turns' giving the speaker order (and, if force, the texts); duplex episodes
    give 'reply' only for force. force: teacher-force the episodes' texts; a dialogue episode's 'forced' lists the
    turns that are always teacher-forced (for example a fixed question after free turns). mode: a channel ablation (link.ablate).
    mute: {x: [bool per episode]} cuts what x receives. link_off_after: number of finished turns after which no
    message is delivered (an episode's own link_off_after comes first). write_mode='random': each residual write is
    replaced by a random direction of the same norm (a control for steering). Returns per episode the texts and per-step gate probabilities of both models (keep: also the
    unrounded probabilities, concatenated over steps)."""
    from transformers import DynamicCache

    from link import ablate
    toks, models = pair.toks, pair.models
    B = len(eps)
    dmode = eps[0]['mode']
    end = {x: toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
    st = {x: [Stream() for _ in range(B)] for x in SIDES}
    cache, att, pos = {}, {}, {}
    for x in SIDES:
        P = []
        for i, ep in enumerate(eps):
            p = prefix(toks[x], x, ep['system'][x], ep['task'][x])
            if dmode == 'duplex':
                p = p + assistant_header(toks[x], x)
                st[x][i].pending, st[x][i].speaking = p[-1:], True
                p = p[:-1]
            P.append(p)
        Lp = max(map(len, P))
        pad = toks[x].pad_token_id
        ids_ = torch.tensor([[pad] * (Lp - len(p)) + p for p in P], device=DEVICE)
        att[x] = torch.tensor([[0] * (Lp - len(p)) + [1] * len(p) for p in P], device=DEVICE)
        pos[x] = att[x].sum(-1)
        cache[x] = DynamicCache()
        pair.taps[x].delta = None
        models[x](input_ids=ids_, attention_mask=att[x], position_ids=(att[x].cumsum(-1) - 1).clamp(min=0),
                  past_key_values=cache[x], use_cache=True)
    turn = [0] * B
    tstep = [0] * B
    done = [False] * B
    mail = {x: [None] * B for x in SIDES}         # (sum of writes, count, open count) waiting for x; soft: slot list
    box = {x: [None] * B for x in SIDES}          # with pair.inbox: (sum, open count) of every write received
    got = [[] for _ in range(B)]                  # listener tokens read so far in the current turn
    off = [ep.get('link_off_after', link_off_after) for ep in eps]

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
        tstep[i] = 0

    for i in range(B):
        start_turn(i)
    for step in range(10 ** 6):
        if all(done):
            break
        sent = {}
        for x in SIDES:
            chunks = [st[x][i].pending if not done[i] else [] for i in range(B)]
            K = max(map(len, chunks))
            if K == 0:
                sent[x] = None
                continue
            if pair.soft:
                h, logits, m_, ns, text = soft_step(pair, x, chunks, mail, att, pos, cache)
                for i in range(B):
                    pos[x][i] += ns[i]
            else:
                pad = toks[x].pad_token_id
                inp = torch.tensor([c + [pad] * (K - len(c)) for c in chunks], device=DEVICE)
                m_ = torch.tensor([[1] * len(c) + [0] * (K - len(c)) for c in chunks], device=DEVICE)
                pid = (pos[x][:, None] + torch.arange(K, device=DEVICE)[None]).clamp(min=0)
                d = models[x].config.hidden_size
                delta = torch.zeros(B, K, d, device=DEVICE)
                for i in range(B):
                    if chunks[i] and mail[x][i] is not None:
                        w, n, c = mail[x][i]
                        if pair.inbox:
                            old = box[x][i]
                            box[x][i] = (w, c) if old is None else (old[0] + w, old[1] + c)
                        else:
                            delta[i, 0] = w / n
                        mail[x][i] = None
                    if chunks[i] and pair.inbox and box[x][i] is not None:
                        delta[i, :len(chunks[i])] = box[x][i][0] / max(box[x][i][1], 1.0)
                pair.taps[x].delta = delta
                att[x] = torch.cat([att[x], m_], 1)
                out = models[x](input_ids=inp, attention_mask=att[x], position_ids=pid, past_key_values=cache[x],
                                use_cache=True)
                pair.taps[x].delta = None
                h = pair.taps[x].h
                last = (m_.sum(-1) - 1).clamp(min=0)
                logits = out.logits[torch.arange(B), last].float()
            m, p = pair.side[x].message(h)                                   # [B, K, Kmsg], [B, K]
            g = (p > 0.5).float() * m_.float()
            sent[x] = (m, g, m_, text) if pair.soft else (m, g, m_)
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            for i in range(B):
                if not chunks[i]:
                    continue
                st[x][i].gates.append([round(float(v), 3) for v in p[i, :len(chunks[i])]])
                st[x][i].gate_turn.append(turn[i])
                pos[x][i] += len(chunks[i])
                st[x][i].pending = []
                if keep:
                    st[x][i].p.append(p[i, :len(chunks[i])].cpu())
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
        # deliver this step's messages
        for x in SIDES:
            if sent[x] is None:
                continue
            y = partner(x)
            if pair.soft:
                if mode != 'zero':
                    soft_deliver(pair, x, sent[x], mail, mode, gen, write_mode, done, off, turn, mute)
                continue
            m, g, m_ = sent[x]
            m, g = ablate(m, g, mode, gen)
            w = pair.side[y].write(m) * g[..., None]                       # [B, K, d]
            if write_mode == 'random':
                r = torch.randn(w.shape, generator=gen, device='cpu').to(w.device)
                w = torch.nn.functional.normalize(r, dim=-1) * w.norm(dim=-1, keepdim=True)
            for i in range(B):
                n = int(m_[i].sum())
                if n == 0 or done[i]:
                    continue
                if off[i] is not None and turn[i] >= off[i]:
                    continue
                if mute is not None and y in mute and mute[y][i]:
                    continue
                s, c = w[i, :n].sum(0), float(g[i, :n].sum())
                old = mail[y][i]
                mail[y][i] = (s, n, c) if old is None else (old[0] + s, old[1] + n, old[2] + c)
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
            res[-1]['p'] = {x: torch.cat(st[x][i].p) if st[x][i].p else torch.zeros(0) for x in SIDES}
    return res


def soft_step(pair, x, chunks, mail, att, pos, cache):
    """One step of model x with the soft read: each episode's slots (the vectors in its mail) come first, then the
    tokens it reads. Returns the residual after block ls at the tokens [B, Kr, d], the logits after the last token,
    the token mask [B, Kr], the slot counts, and the text embeddings of x's next-token distributions at the tokens."""
    model, tap = pair.models[x], pair.taps[x]
    B = len(chunks)
    slots = [mail[x][i] if chunks[i] and mail[x][i] is not None else [] for i in range(B)]
    for i in range(B):
        if chunks[i]:
            mail[x][i] = None
    ns = [len(v) for v in slots]
    K = max(ns[i] + len(chunks[i]) for i in range(B))
    Kr = max(map(len, chunks))
    pad = pair.toks[x].pad_token_id
    inp = torch.tensor([[pad] * ns[i] + chunks[i] + [pad] * (K - ns[i] - len(chunks[i])) for i in range(B)],
                       device=DEVICE)
    e = model.get_input_embeddings()(inp)
    for i in range(B):
        if ns[i]:
            e[i, :ns[i]] = torch.stack(slots[i]).to(e.dtype)
    full = torch.tensor([[1] * (ns[i] + len(chunks[i])) + [0] * (K - ns[i] - len(chunks[i])) for i in range(B)],
                        device=DEVICE)
    pid = (pos[x][:, None] + torch.arange(K, device=DEVICE)[None]).clamp(min=0)
    att[x] = torch.cat([att[x], full], 1)
    tap.delta = None
    out = model(inputs_embeds=e, attention_mask=att[x], position_ids=pid, past_key_values=cache[x], use_cache=True)
    # the tokens' places in the chunk (padding repeats the last token; its mask is 0)
    ri = torch.tensor([[ns[i] + min(k, max(len(chunks[i]) - 1, 0)) for k in range(Kr)] for i in range(B)],
                      device=DEVICE)
    m_ = torch.tensor([[1] * len(c) + [0] * (Kr - len(c)) for c in chunks], device=DEVICE)
    h = tap.h.gather(1, ri[..., None].expand(-1, -1, tap.h.shape[-1]))
    lg = out.logits.gather(1, ri[..., None].expand(-1, -1, out.logits.shape[-1]))
    logits = lg[torch.arange(B), (m_.sum(-1) - 1).clamp(min=0)].float()
    return h, logits, m_, ns, pair.text_embedding(x, logits=lg)


def soft_deliver(pair, x, sent, mail, mode, gen, write_mode, done, off, turn, mute):
    """Slot vectors from the tokens x read this step, appended to its partner's mail in reading order."""
    from link import ablate, ablate_text
    y = partner(x)
    m, g, m_, text = sent
    m, g = ablate(m, g, mode, gen)
    text = ablate_text(text, mode, gen)
    v, w = pair.slot_vectors(y, (m, g, None, None, text))
    if write_mode == 'random':
        c = text[..., :-1] + (1 - text[..., -1:]) * pair.side[y].null + w
        r = torch.randn(c.shape, generator=gen, device='cpu').to(c.device)
        v = g[..., None] * torch.nn.functional.normalize(r, dim=-1) * c.norm(dim=-1, keepdim=True) + \
            (1 - g[..., None]) * pair.side[y].null
    for i in range(len(done)):
        n = int(m_[i].sum())
        if n == 0 or done[i] or (off[i] is not None and turn[i] >= off[i]) or (mute is not None and y in mute and mute[y][i]):
            continue
        mail[y][i] = (mail[y][i] or []) + [v[i, k] for k in range(n)]
