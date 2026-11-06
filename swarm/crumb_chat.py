"""Where a real chat's information is lost on the way through full KV, stage by stage, with no training: the adapter off
throughout. A real chat (family chat_alt) has turns in order, A's and B's alternating, and the teacher read all of them;
its top-32 at every answer token is cached. B's answer tokens are scored by KL to it, as in swarm/crumb_open.py. From the
reader as it is now to the teacher, one change at a time:
  parts       the setup now (crumb_open's zero_on): A, under its system line, reads "Part 1: A user asked you ... You
              answered ...", at most --visits parts; B reads all of it, under its own system line, its tokens SPAN
              positions after A's
  framed      the same text, one part per A turn with no cap
  ordered     ... and each of B's turns reads only what A read before it in the chat
  no_system   ... and B has no system line, as the teacher has none
  true_pos    ... and A's and B's tokens sit about where they sit in the chat (A's text is longer than the chat's, so
              each part ends at its turn's end)
  alone       ... and A reads its own turns in the chat's format with no framing, at their true positions, and without
              B's turns (only A's own earlier turns)
  with_b      ... and A's keys and values come from one pass over the whole chat, B's turns included: the teacher's own
              keys and values, which must give a KL of about 0
and to one side, each one change from alone: alone_future (B reads A's later turns too), alone_span (B's tokens SPAN
positions after A's, A's at their own count) and alone_system (B has its system line back). From ordered, why the framed
text gets worse without B's system line: marked, marked_no_system and marked_true_pos are ordered, no_system and true_pos
with each part's chat markers among the crumbs. The split as it could run: chat_walker (A's own turns in the chat's format
under its system line, B under its own, SPAN after), chat_walker_no_b_system, and alone_both_system (both system lines,
true positions). cut_agent and cut_plain are B alone with and without its system line, whole_chat the whole chat in one
pass (the check on the cached teacher).
Writes one JSON: per stage the KL per answer token, the cut it is measured against, the closed share with a 95% interval
over episodes, and the same per B turn index.
Usage: python swarm/crumb_chat.py --data DIR --out OUT.json [--family chat_alt --split eval_seen --n_eval 600]"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_open as CO                      # noqa: E402
from swarm import crumbs as C                           # noqa: E402

AGENT_STAGES = ('parts', 'framed', 'ordered', 'alone_system', 'marked', 'chat_walker', 'alone_both_system')   # against cut_agent
PLAIN_STAGES = ('no_system', 'true_pos', 'alone', 'with_b', 'alone_future', 'alone_span', 'whole_chat', 'marked_no_system',
                'marked_true_pos', 'chat_walker_no_b_system')          # against cut_plain


def chat_tokens(rec):
    """The teacher's stream of the record (opendata.teacher_ids) with each token's turn, and each turn's answer start."""
    od = CO.od()
    ids, turn, starts = [], [], {}
    for k, t in enumerate(rec['turns']):
        n0 = len(ids)
        ids += od.piece(od.USER_HEAD + t['teacher_user'] + od.END + od.ASSIST_HEAD)
        starts[k] = len(ids)
        r = list(t['response_ids'])
        ids += r + (od.piece('\n') if r and r[-1] == od.IM_END else od.piece(od.END))
        turn += [k] * (len(ids) - n0)
    return ids, turn, starts


@torch.no_grad()
def forward(net, device, ids, pos, bank=None, allow=None, capture=False):
    """One stream at positions pos, the adapter off, reading bank [(k, v) per layer, [1, heads, S, head dim]] where allow
    [T, S]. Returns the final norm's output [T, d] and, if capture, per layer (k, v) of its own tokens."""
    x = torch.tensor([ids], device=device)
    T = len(ids)
    own = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    mask = (torch.cat([allow, own], -1) if bank else own)[None, None]
    got = {} if capture else None
    S = bank[0][0].shape[2] if bank else 0
    net.grab = True
    try:
        with net.model.disable_adapter():
            net.model(inputs_embeds=net.model.get_input_embeddings()(x), attention_mask=mask,
                      position_ids=torch.tensor([pos], device=device), use_cache=False, logits_to_keep=1,
                      crumb_k=[k for k, _ in bank] if bank else None, crumb_v=[v for _, v in bank] if bank else None,
                      crumb_got=got)
        h = net.last[0]
    finally:
        net.grab, net.last = False, None
    return h, ([(got[i][0][:, :, S:], got[i][1][:, :, S:]) for i in range(len(got))] if capture else None)


def taken(kv, idx, device):
    at = torch.tensor(idx, device=device)
    return [(k[:, :, at], v[:, :, at]) for k, v in kv]


def score(net, h, rec, starts, device):
    """Per turn with an answer in this stream {turn: (summed KL over its answer tokens, their count)}."""
    out = {}
    for k, start in starts.items():
        ids, lp = CO.targets(rec, k)
        n = min(len(rec['turns'][k]['response_ids']), len(ids))
        logp = net.model.get_output_embeddings()(h[start - 1:start - 1 + n]).float().log_softmax(-1)
        out[k] = (float(C.kl_topk(logp, ids[:n].to(device), lp[:n].to(device).float(), normalized=True).sum()), n)
    return out


def framed_stream(net, rec, walker):
    """A's framed text, one part per A turn as crumb_open's material: its ids, and per part (turn, first and last + 1
    index of its text in ids), and the same with its chat markers."""
    ids = net._ids(f'<|im_start|>system\n{CO.WALKER}<|im_end|>\n')
    spans, marks, i = [], [], 0
    head = net._ids('<|im_start|>user\n')
    for k, t in enumerate(rec['turns']):
        if t['who'] != walker:
            continue
        i += 1
        user = f'Part {i}:\nA user asked you:\n{t["user"]}\n\nYou answered:\n{t["response"]}'
        start = len(ids)
        ids = ids + net._ids(f'<|im_start|>user\n{user}<|im_end|>\n')
        spans.append((k, start + len(head), start + len(net._ids(f'<|im_start|>user\n{user}'))))
        marks.append((k, start, len(ids)))
    return ids, spans, marks


def reader_stream(net, chat, turn, starts, reader, system):
    """B's tokens of the chat, under its system line if system: ids, per token its index in the chat (-1: the system
    line) and its turn, and per B turn the answer start in this stream."""
    head = net._ids(f'<|im_start|>system\n{CO.READER}<|im_end|>\n') if system else []
    keep = [i for i, k in enumerate(turn) if k in reader]
    where = {i: len(head) + j for j, i in enumerate(keep)}
    return head + [chat[i] for i in keep], [-1] * len(head) + keep, [-1] * len(head) + [turn[i] for i in keep], \
        {k: where[starts[k]] for k in reader}


def stages(net, rec, device, span):
    """{stage: per turn (summed KL, count)} for one chat, parts and cut_agent aside (crumb_open's own scorer gives them)."""
    chat, turn, starts = chat_tokens(rec)
    walker, reader_side = CO.sides(rec)
    reader = {k for k, t in enumerate(rec['turns']) if t['who'] == reader_side}
    out = {}
    full_h, full_kv = forward(net, device, chat, list(range(len(chat))), capture=True)
    out['whole_chat'] = score(net, full_h, rec, {k: starts[k] for k in reader}, device)

    def reading(system, bank, entry_turn, future=False, positions=None):
        ids, index, turns, at = reader_stream(net, chat, turn, starts, reader, system)
        if positions is None:                          # the chat's positions, a system line just before them
            n = sum(i < 0 for i in index)
            positions = [j - n if i < 0 else i for j, i in enumerate(index)]
        et, rt = torch.tensor(entry_turn, device=device), torch.tensor(turns, device=device)
        allow = ((et[None, :] < rt[:, None]) | future) & (rt[:, None] >= 0)   # a system line reads nothing
        h, _ = forward(net, device, ids, positions, bank, allow)
        return score(net, h, rec, at, device)

    out['cut_plain'] = forward_cut(net, device, chat, turn, starts, reader, rec, False)
    # A's framed text
    fids, spans, marks = framed_stream(net, rec, walker)
    idx = [i for _, a, b in spans for i in range(a, b)]
    ent = [k for k, a, b in spans for _ in range(a, b)]
    _, fkv = forward(net, device, fids, list(range(len(fids))), capture=True)
    fbank = taken(fkv, idx, device)

    def counted(system):                               # B's tokens at their own count, SPAN after A's
        ids, *_ = reader_stream(net, chat, turn, starts, reader, system)
        return [span + j for j in range(len(ids))]

    out['framed'] = reading(True, fbank, ent, future=True, positions=counted(True))
    out['ordered'] = reading(True, fbank, ent, positions=counted(True))
    out['no_system'] = reading(False, fbank, ent, positions=counted(False))
    # each part ends where its turn ends in the chat (a position may fall below 0: only differences count)
    ends = {k: max(i for i, x in enumerate(turn) if x == k) + 1 for k, _, _ in spans}
    placed = [None] * len(fids)
    for k, a, b in spans:
        placed[a:b] = [ends[k] - (b - i) for i in range(a, b)]
    for i in range(len(fids) - 2, -1, -1):             # the framing around them, next to them
        placed[i] = placed[i] if placed[i] is not None else placed[i + 1] - 1 if placed[i + 1] is not None else None
    for i in range(1, len(fids)):
        placed[i] = placed[i] if placed[i] is not None else placed[i - 1] + 1
    _, tkv = forward(net, device, fids, placed, capture=True)
    out['true_pos'] = reading(False, taken(tkv, idx, device), ent)
    # the same three with the parts' chat markers among the crumbs
    midx = [i for _, a, b in marks for i in range(a, b)]
    ment = [k for k, a, b in marks for _ in range(a, b)]
    out['marked'] = reading(True, taken(fkv, midx, device), ment, positions=counted(True))
    out['marked_no_system'] = reading(False, taken(fkv, midx, device), ment, positions=counted(False))
    out['marked_true_pos'] = reading(False, taken(tkv, midx, device), ment)
    # A's own turns in the chat's format, without B's
    mine = [i for i, k in enumerate(turn) if k not in reader]
    a_ids, a_turn = [chat[i] for i in mine], [turn[i] for i in mine]
    _, akv = forward(net, device, a_ids, mine, capture=True)
    every = list(range(len(mine)))
    abank = taken(akv, every, device)
    out['alone'] = reading(False, abank, a_turn)
    out['alone_future'] = reading(False, abank, a_turn, future=True)
    out['alone_system'] = reading(True, abank, a_turn)
    _, ckv = forward(net, device, a_ids, every, capture=True)
    out['alone_span'] = reading(False, taken(ckv, every, device), a_turn, positions=counted(False))
    # the split as it could run: A's turns in the chat's format under its system line, not itself among the crumbs
    line = net._ids(f'<|im_start|>system\n{CO.WALKER}<|im_end|>\n')
    n = len(line)
    _, wkv = forward(net, device, line + a_ids, list(range(n + len(a_ids))), capture=True)
    wbank = taken(wkv, [n + i for i in every], device)
    out['chat_walker'] = reading(True, wbank, a_turn, positions=counted(True))
    out['chat_walker_no_b_system'] = reading(False, wbank, a_turn, positions=counted(False))
    _, bkv = forward(net, device, line + a_ids, list(range(-n, 0)) + mine, capture=True)
    out['alone_both_system'] = reading(True, taken(bkv, [n + i for i in every], device), a_turn)
    # A's keys and values from the whole chat: B reading them is the teacher again
    out['with_b'] = reading(False, taken(full_kv, mine, device), a_turn)
    return out


def row(net, rec, ep, device, span):
    """One chat's {stage: {B turn index: (summed KL, count)}}."""
    got = CO.kl_pass(net, [ep], device, ('on', 'cut'), zero=True)
    mine = [k for k, t in enumerate(rec['turns']) if t['who'] == CO.sides(rec)[1]]
    r = dict(parts={k: got['on'][0].get(j, (0., 0)) for j, k in enumerate(mine)},
             cut_agent={k: got['cut'][0].get(j, (0., 0)) for j, k in enumerate(mine)})
    r.update(stages(net, rec, device, span))
    order = {k: j for j, k in enumerate(mine)}         # per B turn index, not the chat's turn
    return {st: {order[k]: v for k, v in d.items()} for st, d in r.items()}


def forward_cut(net, device, chat, turn, starts, reader, rec, system):
    ids, _, _, at = reader_stream(net, chat, turn, starts, reader, system)
    h, _ = forward(net, device, ids, list(range(len(ids))))
    return score(net, h, rec, at, device)


def pool(rows, stage, turn=None):
    """Summed KL and count over rows' turns; rows: per episode {stage: {B turn index: (KL, n)}}."""
    s = n = 0.
    for r in rows:
        for t, (a, b) in r[stage].items():
            if turn is None or t == turn:
                s, n = s + a, n + b
    return s, n


def summary(rows, seed=0):
    """Per stage the KL per token, its cut, the closed share with a 95% interval over episodes, and per B turn index."""
    out = {}
    rng = random.Random(seed)
    resamples = [[rng.choice(rows) for _ in rows] for _ in range(1000)]
    stages_ = list(rows[0])
    for st in stages_:
        s, n = pool(rows, st)
        out[st] = dict(kl=round(s / max(1, n), 4), tokens=int(n))
    turns = sorted({t for r in rows for t in r['cut_plain']})
    for st, cut in [(x, 'cut_agent') for x in AGENT_STAGES] + [(x, 'cut_plain') for x in PLAIN_STAGES]:
        if st not in out:
            continue

        def share(part, turn=None):
            c, _ = pool(part, cut, turn)
            return (c - pool(part, st, turn)[0]) / c if c else None

        got = sorted(share(p) for p in resamples)
        out[st].update(against=cut, closed=round(share(rows), 4), closed_ci=[round(got[25], 4), round(got[974], 4)],
                       by_turn={t: dict(closed=None if share(rows, t) is None else round(share(rows, t), 4),
                                        kl=round(pool(rows, st, t)[0] / max(1, pool(rows, st, t)[1]), 4),
                                        cut=round(pool(rows, cut, t)[0] / max(1, pool(rows, cut, t)[1]), 4),
                                        tokens=int(pool(rows, st, t)[1])) for t in turns})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help="opendata's folder for the teacher model (shards/ and teacher/)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--family', default='chat_alt')
    ap.add_argument('--split', default='eval_seen')
    ap.add_argument('--span', type=int, default=6144)
    ap.add_argument('--part_tokens', type=int, default=384)
    ap.add_argument('--visits', type=int, default=8)
    ap.add_argument('--n_eval', type=int, default=600, help="crumb_open's: the first n records that fit, of any family")
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    C.SPAN, C.CHUNK = args.span, 2
    net = C.build(args.device, 'kv', 16, base=args.base, ckpt=False)
    net.source = 'sentence'
    net.eval()
    picked, seen = [], 0
    for rec in CO.records(args.data, args.split, args.seed):
        ep = CO.episode(net, rec, args.part_tokens, args.visits)
        if ep is None:
            continue
        seen += 1
        if ep['open']['family'] == args.family:
            picked.append((rec, ep))
        if seen >= args.n_eval:
            break
    rows, capped, total, lost, started = [], 0, 0, 0, time.monotonic()

    def save():
        res = dict(vars(args), episodes=len(rows), of=len(picked), capped=capped, material_tokens=total,
                   past_the_cap=lost, s=round(time.monotonic() - started), stages=summary(rows) if rows else None)
        Path(args.out).write_text(json.dumps(res, indent=1))
        return res

    for i, (rec, ep) in enumerate(picked):
        w, _ = CO.sides(rec)
        every = CO.parts(net, CO.material(rec, w), args.part_tokens, 10 ** 6)
        sizes = [len(net._ids(p)) for p in every]
        capped, total, lost = capped + (len(every) > args.visits), total + sum(sizes), lost + sum(sizes[args.visits:])
        rows.append(row(net, rec, ep, args.device, args.span))
        if i % 10 == 0:
            res = save()
            print(json.dumps(dict(episodes=len(rows), of=len(picked), s=res['s'],
                                  closed={k: v.get('closed') for k, v in res['stages'].items()})), flush=True)
    res = save()
    print(json.dumps({k: v for k, v in res.items() if k != 'stages'}), flush=True)
    for st, v in res['stages'].items():
        print(st, json.dumps({k: x for k, x in v.items() if k != 'by_turn'}), flush=True)


if __name__ == '__main__':
    main()
