"""Breadcrumbs on open-ended multi-turn data (swarm/opendata.py records). One place, the desk. Agent A, the walker,
holds one side of a record: its private view and the task turns it answered there (the user's request and the
teacher's answer). It comes to the desk again and again; on each visit it reads the next part of that material (parts
of about --part_tokens tokens, at most --visits of them) and leaves crumbs, on top of the ones from its earlier visits.
Agent B, the reader, then holds the other side's whole conversation at the desk: its own view in its first user turn,
each of its task turns' requests, and the teacher's answer to each. Every answer token is trained toward the teacher
(the frozen base model, which saw both sides; its top-32 cached with the records, opendata.Episodes(teacher=True), so
the teacher never runs here) by KL(teacher || reader) on the teacher's top-32 plus one bucket for every other token.
Fill-in turns (no teacher, real text) use the text itself as a one-hot teacher, so the KL there is the cross-entropy.
The reader is the side named B unless B has no task turn. Crumbs come from the kv arm of swarm/crumbs.py: m slot tokens
per visit (--source slots), or the keys and values of the visit's own part (sentence). Training: LoRA with gradient
checkpointing, batches of episodes of similar length (sorted pools of POOL batches; when a pass over the training
records ends before --steps, the next starts in a new order), a dev of 32 training episodes (KL with the crumbs on and
cut) every 200 steps, and the GPU's utilization logged at step 20 and every 100 steps (OUT/gpu.jsonl).

Evaluation, per split (eval_seen, eval_heldout): the reader's KL per answer token with the crumbs on, cut and
transplanted (each record reads the next record's crumbs), and the share of the cut KL that the crumbs close, (cut - on)
/ cut. These are given overall, per source, family and group of families (GROUPS), by the reader's turn index and by
visit count (only the walker's first v visits, v in VISITS_AT, and all of them). Beside them are the same with no
training: the adapter off and the crumbs the parts' own keys and values. Writes OUT/log.jsonl, OUT/gpu.jsonl,
OUT/trained.pt, OUT/scores.json.
Usage: python swarm/crumb_open.py --data DIR --out OUT [--base Qwen/Qwen3-1.7B --source slots --m 16 --steps 1500]
(DIR: opendata's folder for the teacher model, holding shards/ and teacher/)"""
import argparse
import contextlib
import importlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402

OD = None                                               # swarm.opendata, imported on first use (tests swap in a stand-in)
WALKER = ('You are agent A. You come back to this desk again and again; each time you read the next part of some '
          'material here, and you leave marks for agent B, who comes after you.')
READER = 'You are agent B. Agent A read material at this desk before you and left marks here. Help the user.'
K = 32                                                  # the teacher rows' top-k
SPLITS = ('eval_seen', 'eval_heldout')
VISITS_AT = (1, 2, 4)                                   # by_visit: the walker's first v visits (and all of them)
VSAME = 4                                               # by_visit_same: records with at least this many visits
POOL = 16                                               # batches per length-sorted training pool
N_DEV, DEV_EVERY = 32, 200
# by_group: families pooled. real: no simulated request at all (every user turn is real text), the headline group;
# simulated: every family with a simulated user, who is shown the hidden half (opendata.sim_prompt), so its requests
# can name details only that half holds (a hand check found them in doc, pair, code and debug).
GROUPS = dict(real=('chat_alt', 'fill'), simulated=('chat_doc', 'doc', 'pair', 'code', 'debug', 'plan'))


def od():
    global OD
    if OD is None:
        OD = importlib.import_module('swarm.opendata')
    return OD


def parts(net, text, part_tokens, most):
    """text cut into at most most parts of up to part_tokens tokens, at line ends where a line fits."""
    out, cur, n = [], [], 0
    for line in text.split('\n'):
        ids = net._ids(line + '\n')
        while len(ids) > part_tokens:
            if cur:
                out.append(''.join(cur))
                cur, n = [], 0
            out.append(net.tok.decode(ids[:part_tokens]))
            ids = ids[part_tokens:]
        if n + len(ids) > part_tokens and cur:
            out.append(''.join(cur))
            cur, n = [], 0
        cur.append(net.tok.decode(ids))
        n += len(ids)
    if cur:
        out.append(''.join(cur))
    return [p for p in out if p.strip()][:most]


def sides(rec):
    """(walker, reader): the reader is B unless B has no task turn."""
    return ('A', 'B') if any(t['who'] == 'B' for t in rec['turns']) else ('B', 'A')


def material(rec, w):
    """The walker's side: its view, then each task turn it answered."""
    text = [rec['views'].get(w, '').strip()]
    for t in rec['turns']:
        if t['who'] == w:
            text.append(f'A user asked you:\n{t["user"]}\n\nYou answered:\n{t["response"]}')
    return '\n\n'.join(x for x in text if x)


def targets(rec, k):
    """Turn k's teacher (ids [T, K], lp [T, K]): the cached top-K (rec['topk']), or for a turn without a teacher the
    real text one-hot (its token at log-prob 0, the other columns at -1e4: probability 0); None when a teacher turn's
    row is missing."""
    if rec.get('teacher', True):
        got = (rec.get('topk') or {}).get(k)
        return None if got is None else (got[0].long(), got[1].float())
    ids = torch.tensor(rec['turns'][k]['response_ids'], dtype=torch.long)
    lp = torch.full((len(ids), K), -1e4)
    lp[:, 0] = 0.
    return ids[:, None].expand(-1, K).contiguous(), lp


def episode(net, rec, part_tokens, most, teacher=True):
    """A crumbs.py episode of one opendata record: the walker's visits to the desk, then the reader's conversation
    there (the teacher's top-k on every answer token when teacher). None when a side is empty or a stream is longer
    than SPAN."""
    w, r = sides(rec)
    ps = parts(net, material(rec, w), part_tokens, most)
    mine = [k for k, t in enumerate(rec['turns']) if t['who'] == r]
    if not ps or not mine:
        return None
    walker = dict(name='agent A', system=WALKER, write=True,
                  segments=[dict(place=0, user=f'Part {k + 1}:\n{p}', say=None, values=None) for k, p in enumerate(ps)])
    view, segs = rec['views'].get(r, '').strip(), []
    for i, k in enumerate(mine):
        turn = rec['turns'][k]
        say = list(turn['response_ids'])
        ended = bool(say) and say[-1] == net.end                   # the teacher's own <|im_end|>: its row is kept
        say = say[:-1] if ended else say
        if not say:
            return None
        tg = targets(rec, k) if teacher else None
        n = len(say) + ended
        if teacher and (tg is None or len(tg[0]) < n):
            return None
        user = f'{view}\n\n{turn["user"]}' if i == 0 and view else turn['user']
        segs.append(dict(place=0, user=user, say=turn['response'], say_ids=say, values=None,
                         teacher=None if tg is None else (tg[0][:n], tg[1][:n])))
    ep = dict(game='open', places=['the desk'],
              agents=[walker, dict(name='agent B', system=READER, write=False, segments=segs)],
              open=dict(id=rec['id'], source=rec['source'], family=rec.get('family'), split=rec['split'],
                        visits=len(ps)))
    try:
        ep['open']['tokens'] = len(net.stream(ep, 0)['ids']) + len(net.stream(ep, 1)['ids'])
    except ValueError:
        return None
    return ep


def first_visits(ep, v):
    """ep with the walker's first v visits only."""
    w = dict(ep['agents'][0], segments=ep['agents'][0]['segments'][:v])
    return dict(ep, agents=[w] + ep['agents'][1:], open=dict(ep['open'], visits=v))


@torch.no_grad()
def kl_pass(net, eps, device, modes=('on',), zero=False):
    """{mode: per episode {reader segment: (summed KL over its answer tokens, their count)}}, the walker run once.
    modes: on, cut, transplant (each episode reads the next one's crumbs); zero: the adapter off and the crumbs the
    parts' own keys and values."""
    keep = net.source
    out = {}
    try:
        net.source = 'sentence' if zero else keep
        with net.model.disable_adapter() if zero else contextlib.nullcontext():
            _, _, bank = net.run(eps, device, need_loss=False)
            streams = [net.stream(ep, 1) for ep in eps]
            for mode in modes:
                b = C.shifted(bank) if mode == 'transplant' else bank
                got = []
                for s0 in range(0, len(eps), C.CHUNK):
                    rows = torch.arange(s0, min(len(eps), s0 + C.CHUNK), device=device)
                    got += net._say_kl(rows, streams[s0:s0 + C.CHUNK], 1, b, mode != 'cut', device)
                out[mode] = got
    finally:
        net.source = keep
    return out


def summarize(per):
    """per: list of dict(source, family, visits, turns={mode: [(sum, n) per reader turn]}, by_visit={v: [(sum, n) per
    turn]}) -> token-weighted KL per mode and the closed shares, overall, by turn index, by visit count and per
    source."""
    def tot(rows, mode, turn=None):
        s = n = 0.
        for r in rows:
            for t, (a, b) in enumerate(r['turns'].get(mode, ())):
                if turn is None or t == turn:
                    s, n = s + a, n + b
        return s, n

    def share(cut, on):
        return (cut - on) / cut if cut else None

    def block(rows):
        out = {mode: tot(rows, mode)[0] / max(1, tot(rows, mode)[1]) for mode in rows[0]['turns']} if rows else {}
        out['tokens'] = tot(rows, 'cut')[1]
        for mode in ('on', 'transplant'):
            if mode in out:
                out[f'closed_{mode}'] = share(out['cut'], out[mode])
        if 'zero_cut' in out:
            out['closed_zero'] = share(out['zero_cut'], out['zero_on'])
        return out

    def visit(rows, v):
        """KL with the walker's first v visits (all: every visit) on, over rows."""
        on = [r['by_visit'][v] if v != 'all' and v < r['visits'] else r['turns']['on'] for r in rows]
        s_on, n = sum(a for x in on for a, _ in x), sum(b for x in on for _, b in x)
        s_cut = sum(a for r in rows for a, _ in r['turns']['cut'])
        return dict(on=s_on / max(1, n), cut=s_cut / max(1, n), closed=share(s_cut, s_on), episodes=len(rows))

    res = block(per)
    res['by_turn'] = {}
    for t in range(max((len(r['turns']['cut']) for r in per), default=0)):
        rows = [r for r in per if len(r['turns']['cut']) > t]
        on, cut = tot(rows, 'on', t), tot(rows, 'cut', t)
        res['by_turn'][t + 1] = dict(on=on[0] / max(1, on[1]), cut=cut[0] / max(1, cut[1]), closed=share(cut[0], on[0]),
                                     episodes=len(rows))
    res['by_visit'] = {v: visit([r for r in per if r['visits'] >= v], v) for v in VISITS_AT
                       if any(r['visits'] >= v for r in per)}
    res['by_visit']['all'] = visit(per, 'all')
    same = [r for r in per if r['visits'] >= VSAME]
    res[f'by_visit_same{VSAME}'] = {v: visit(same, v) for v in VISITS_AT + ('all',) if v == 'all' or v <= VSAME} \
        if same else {}
    res['by_source'] = {s: block([r for r in per if r['source'] == s]) for s in sorted({r['source'] for r in per})}
    return res


def score_batch(net, eps, device, zero=True):
    """The summarize() rows of a batch of episodes."""
    modes = kl_pass(net, eps, device, ('on', 'cut', 'transplant') if len(eps) > 1 else ('on', 'cut'))
    if zero:
        modes.update({f'zero_{k}': v for k, v in kl_pass(net, eps, device, ('on', 'cut'), zero=True).items()})
    seen = {}
    for v in VISITS_AT:
        sub = [(i, first_visits(ep, v)) for i, ep in enumerate(eps) if ep['open']['visits'] > v]
        if sub:
            seen.update({(i, v): g for (i, _), g in zip(sub, kl_pass(net, [e for _, e in sub], device)['on'])})
    rows = []
    for i, ep in enumerate(eps):
        nt = len(ep['agents'][1]['segments'])
        rows.append(dict(source=ep['open']['source'], family=ep['open']['family'], visits=ep['open']['visits'],
                         turns={m: [g[i].get(t, (0., 0)) for t in range(nt)] for m, g in modes.items()},
                         by_visit={v: [seen[(i, v)].get(t, (0., 0)) for t in range(nt)]
                                   for v in VISITS_AT if (i, v) in seen}))
    return rows


def records(data, split, seed):
    return od().Episodes(data, split, seed=seed, teacher=True)


def passes(first, more, skip, fits, on_pass=None):
    """Training episodes, pass after pass over the training records: first (their iterator, maybe started), then
    more(n) for pass n (the records in a new order). Records whose id is in skip are left out, and so are those
    fits(rec) turns down (None); a whole pass that gives none ends it. on_pass(n) is called as pass n starts."""
    it, n = first, 1
    while True:
        used = 0
        for rec in it:
            if rec['id'] not in skip and (ep := fits(rec)) is not None:
                used += 1
                yield ep
        if not used:
            return
        n += 1
        if on_pass is not None:
            on_pass(n)
        it = more(n)


def evaluate_open(net, device, data, n, part_tokens, most, bs=4, seed=7, zero=True):
    """summarize() per split over its first n records that fit (also per family and per group of families)."""
    net.eval()
    out = {}
    for split in SPLITS:
        eps = []
        for rec in records(data, split, seed):
            ep = episode(net, rec, part_tokens, most)
            if ep is not None:
                eps.append(ep)
            if len(eps) >= n:
                break
        eps.sort(key=lambda e: e['open']['tokens'])
        per = [r for s0 in range(0, len(eps), bs) for r in score_batch(net, eps[s0:s0 + bs], device, zero)]
        out[split] = summarize(per) if per else None
        if per:
            out[split]['episodes'] = len(per)
            out[split]['by_family'] = {f: {k: v for k, v in summarize([r for r in per if r['family'] == f]).items()
                                           if not isinstance(v, dict)}
                                       for f in sorted({r['family'] for r in per})}
            out[split]['by_group'] = {}
            for g, fams in GROUPS.items():
                rows = [r for r in per if r['family'] in fams]
                if rows:
                    got = summarize(rows)
                    got.pop('by_source')
                    out[split]['by_group'][g] = dict(got, episodes=len(rows))
    net.train()
    return out


def dev_kl(net, eps, device, bs=4):
    """(KL per answer token with the crumbs on, dict(dev_cut, dev_closed)) over eps."""
    net.eval()
    s = dict(on=0., cut=0.)
    n = 0
    for s0 in range(0, len(eps), bs):
        got = kl_pass(net, eps[s0:s0 + bs], device, ('on', 'cut'))
        for mode in s:
            s[mode] += sum(a for d in got[mode] for a, _ in d.values())
        n += sum(b for d in got['on'] for _, b in d.values())
    net.train()
    on, cut = s['on'] / max(1, n), s['cut'] / max(1, n)
    return on, dict(dev_cut=round(cut, 4), dev_closed=round((cut - on) / cut, 4) if cut else None)


class GpuLog:
    """The GPU's utilization (%) and memory (MiB), sampled every second by nvidia-smi in a thread."""
    def __init__(self):
        self.samples = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                got = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                u, m = got.stdout.strip().split('\n')[0].split(',')
                self.samples.append((time.monotonic(), float(u), float(m)))
            except Exception:
                return
            time.sleep(1)

    def since(self, t0):
        got = [(u, m) for t, u, m in self.samples if t >= t0]
        return (round(sum(u for u, _ in got) / len(got), 1), max(m for _, m in got)) if got else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help="opendata's folder for the teacher model (shards/ and teacher/)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--source', default='slots', choices=('slots', 'sentence'))
    ap.add_argument('--m', type=int, default=16, help='slot crumbs per visit (--source slots)')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--micro', type=int, default=2)
    ap.add_argument('--every', type=int, default=DEV_EVERY)
    ap.add_argument('--span', type=int, default=6144, help='positions per agent stream (longer records are skipped)')
    ap.add_argument('--part_tokens', type=int, default=384)
    ap.add_argument('--visits', type=int, default=8, help='most walker visits (parts of its material)')
    ap.add_argument('--n_eval', type=int, default=100, help='records per evaluation split')
    ap.add_argument('--no_ckpt', action='store_true', help='no gradient checkpointing')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_only', action='store_true', help='load OUT/trained.pt if there, then only evaluate')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))
    C.SPAN, C.CHUNK = args.span, 2
    net = C.build(args.device, 'kv', args.m, base=args.base, ckpt=not args.no_ckpt)
    net.source = args.source
    train_recs = records(args.data, 'train', args.seed)
    dev = []
    for rec in train_recs:
        ep = episode(net, rec, args.part_tokens, args.visits)
        if ep is not None:
            dev.append(ep)
        if len(dev) >= N_DEV:
            break
    epochs = [1]

    def new_pass(n):
        epochs.append(n)
        print(json.dumps(dict(epoch=n, step=drawn[0] // args.batch + 1)), flush=True)

    train_eps = passes(train_recs, lambda n: records(args.data, 'train', args.seed + 1000 * (n - 1)),
                       {ep['open']['id'] for ep in dev}, lambda r: episode(net, r, args.part_tokens, args.visits),
                       new_pass)
    gpu, marks, pool, drawn = GpuLog(), {}, [], [0]
    shuffle = random.Random(args.seed + 1)
    log = open(out / 'gpu.jsonl', 'a')

    def at_step(s):
        """At the start of step s: the mean GPU utilization over steps 11 to 20 and over every 100 steps."""
        marks[s] = time.monotonic()
        if s == 21 or (s > 1 and (s - 1) % 100 == 0):
            lo = 11 if s == 21 else s - 100
            u, m = gpu.since(marks.get(lo, 0.))
            rec = dict(steps=[lo, s - 1], gpu_util=u, gpu_mib=m)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()

    def fill():
        got = list(itertools.islice(train_eps, POOL * args.batch))
        got.sort(key=lambda e: e['open']['tokens'])
        chunks = [got[i:i + args.batch] for i in range(0, len(got), args.batch)]
        shuffle.shuffle(chunks)
        pool.extend(e for c in chunks for e in c)

    def game(rng, test=False):
        if drawn[0] % args.batch == 0:
            at_step(drawn[0] // args.batch + 1)
        drawn[0] += 1
        if not pool:
            fill()
        if not pool:
            raise SystemExit('opendata training records ran out')
        return pool.pop(0)

    CG.EPISODES['open'] = game
    ran, record = 0, []
    if args.eval_only:
        if (out / 'trained.pt').exists():
            saved = torch.load(out / 'trained.pt', map_location=args.device)
            net.load_state_dict(saved['weights'], strict=False)
            ran, record = saved['steps_run'], saved['dev_nll']
    else:
        ran, record = C.train(net, args.device, args.steps, args.batch, args.seed, games=('open',), out=out,
                              micro=args.micro, min_steps=args.steps, every=args.every, dev=dev,
                              dev_fn=lambda n, d: dev_kl(n, d, args.device))
        torch.save(dict(steps_run=ran, dev_nll=record, weights=C.kept(net)), out / 'trained.pt')
    scores = dict(vars(args), steps_run=ran, epochs=epochs[-1], dev_kl=record,
                  test=evaluate_open(net, args.device, args.data, args.n_eval, args.part_tokens, args.visits))
    (out / 'scores.json').write_text(json.dumps(scores, indent=1))
    print(json.dumps({s: {k: v for k, v in (r or {}).items() if not isinstance(v, dict)}
                      for s, r in scores['test'].items()}), flush=True)


if __name__ == '__main__':
    main()
