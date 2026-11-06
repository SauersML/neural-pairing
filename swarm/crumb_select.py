"""KV selection at the trained slots' budget, evaluation only. The reader gets the full copy (the keys and values of the
walker's own text, crumb_open's sentence source) with only m positions kept per walker visit, chosen three ways:
evenly spaced over the visit; by the base reader's attention (SnapKV: the attention its tokens before its first answer
give each position, summed over every layer and head and max-pooled over 7 positions, the same positions at every
layer and head as a slot is; snaphead: each layer's key-value heads keep their own m); and those shared positions with
a trained crumbs adapter as walker and reader. Modes zero_* run with the adapter off and close a share of the base
reader's cut KL (as crumb_open's closed_zero); the others run with the adapter on and close a share of its own cut (as
the slots' closed_on). Each is the KL per answer token to the teacher overall, per group of families and per turn, on
crumb_open's evaluation episodes (the same records, fit and order as its evaluate_open, the fit taken with the
adapter's slots). Usage:
python swarm/crumb_select.py --data DIR --ckpt TRAINED_PT --out OUT [--m 16 --budgets 16,64 --n_eval 600]"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_open as CO                      # noqa: E402
from swarm import crumbs as C                           # noqa: E402

POOL = 7


def runs_of(net, ep):
    """Per walker visit its bank columns [start, end): the runs of consecutive positions in the walker's slots."""
    t = [p for p, _ in net.stream(ep, 0)['slots']]
    runs, start = [], 0
    for j in range(1, len(t) + 1):
        if j == len(t) or t[j] != t[j - 1] + 1:
            runs.append((start, j))
            start = j
    return runs


def even_keep(runs, S, m):
    keep = torch.zeros(S, dtype=torch.bool)
    for a, b in runs:
        n = b - a
        if n <= m:
            keep[a:b] = True
        else:
            keep[[a + round(j * (n - 1) / max(1, m - 1)) for j in range(m)]] = True
    return keep


def top_keep(score, runs, m):
    """score [..., S] -> the m highest columns of each run (all of a shorter run)."""
    keep = torch.zeros(score.shape, dtype=torch.bool, device=score.device)
    for a, b in runs:
        if b - a <= m:
            keep[..., a:b] = True
        else:
            keep.scatter_(-1, score[..., a:b].topk(m, -1).indices + a, True)
    return keep


def pooled(v):
    return F.max_pool1d(v.reshape(-1, 1, v.shape[-1]), POOL, 1, POOL // 2).reshape(v.shape)


def masked(bank, keep):
    """bank with the columns keep [B, S] leaves out read by no one."""
    return dict(bank, place=torch.where(keep, bank['place'], torch.full_like(bank['place'], -1)))


def votes_of(net, streams, bank, device, block=256):
    """Per layer [B, kv heads, S]: the attention the reader's tokens before its first answer token give each crumb,
    summed over those tokens and each key-value head's query heads (the reader as the net is now)."""
    state, votes = {}, {}

    def probe(i, query, key, mask, S, scaling):
        N, H, T, _ = query.shape
        kv = key.shape[1]
        k = key.float().repeat_interleave(H // kv, 1)
        out = torch.zeros(N, kv, S, device=query.device)
        for n in range(N):
            idx = state['w'][n].nonzero()[:, 0]
            for s0 in range(0, len(idx), block):
                r = idx[s0:s0 + block]
                z = query[n][:, r].float() @ k[n].transpose(-1, -2) * scaling
                mk = mask[n][:, r]
                z = z.masked_fill(~mk, float('-inf')) if mk.dtype == torch.bool else z + mk.float()
                out[n] += z.softmax(-1)[..., :S].sum(1).view(kv, H // kv, S).sum(1)
        state['out'][i] = out

    C.CRUMB_PROBE.append(probe)
    try:
        for s0 in range(0, len(streams), C.CHUNK):
            rows = torch.arange(s0, min(len(streams), s0 + C.CHUNK), device=device)
            x = net._pad(streams[s0:s0 + C.CHUNK], device)
            tgt = x['target'].bool()
            T = tgt.shape[1]
            first = torch.where(tgt.any(1), tgt.float().argmax(1), torch.full_like(tgt[:, 0], T, dtype=torch.long))
            state.update(w=(torch.arange(T, device=device)[None] < first[:, None]) & (x['place'] >= 0)
                         & x['att'].bool(), out={})
            net._forward(rows, x, 1, bank, 1, False, True)
            for i, v in state['out'].items():
                votes.setdefault(i, []).append(v)
    finally:
        C.CRUMB_PROBE.remove(probe)
    return {i: torch.cat(v, 0) for i, v in sorted(votes.items())}


def kl(net, streams, bank, device, on=True, keep=None):
    """crumb_open's per-stream {segment: (summed KL, tokens)} with bank (keep: per layer [B, kv heads, S])."""
    got = []
    for s0 in range(0, len(streams), C.CHUNK):
        rows = torch.arange(s0, min(len(streams), s0 + C.CHUNK), device=device)
        C.CRUMB_KEEP.update({i: k[rows] for i, k in (keep or {}).items()})
        try:
            got += net._say_kl(rows, streams[s0:s0 + C.CHUNK], 1, bank, on, device)
        finally:
            C.CRUMB_KEEP.clear()
    return got


def score_batch(net, eps, device, budgets, adapter=True):
    """Per episode dict(family, turns={mode: [(summed KL, tokens) per reader turn]}) and per budget the columns kept."""
    streams = [net.stream(ep, 1) for ep in eps]
    runs = [runs_of(net, ep) for ep in eps]
    modes, kept = {}, {}
    with torch.no_grad():
        with net.model.disable_adapter():
            bank = net.run(eps, device, need_loss=False)[2]
            S = bank['place'].shape[1]
            modes['zero_cut'] = kl(net, streams, bank, device, on=False)
            modes['zero_full'] = kl(net, streams, bank, device)
            votes = votes_of(net, streams, bank, device)
            total = pooled(torch.stack(list(votes.values())).sum((0, 2)))
            heads = {i: pooled(v) for i, v in votes.items()}
            for m in budgets:
                ev = torch.stack([even_keep(r, S, m) for r in runs]).to(device)
                sh = torch.stack([top_keep(total[b], runs[b], m) for b in range(len(eps))])
                ph = {i: torch.stack([top_keep(v[b], runs[b], m) for b in range(len(eps))]) for i, v in heads.items()}
                modes[f'zero_even{m}'] = kl(net, streams, masked(bank, ev), device)
                modes[f'zero_snap{m}'] = kl(net, streams, masked(bank, sh), device)
                modes[f'zero_snaphead{m}'] = kl(net, streams, bank, device, keep=ph)
                kept[m] = (ev, sh)
        if adapter:
            bank = net.run(eps, device, need_loss=False)[2]
            modes['cut'] = kl(net, streams, bank, device, on=False)
            modes['full'] = kl(net, streams, bank, device)
            for m in budgets:
                ev, sh = kept[m]
                modes[f'even{m}'] = kl(net, streams, masked(bank, ev), device)
                modes[f'snap{m}'] = kl(net, streams, masked(bank, sh), device)
    rows = []
    for i, ep in enumerate(eps):
        nt = len(ep['agents'][1]['segments'])
        rows.append(dict(family=ep['open']['family'], positions=sum(b - a for a, b in runs[i]),
                         kept={m: int(kept[m][1][i].sum()) for m in budgets},
                         turns={k: [g[i].get(t, (0., 0)) for t in range(nt)] for k, g in modes.items()}))
    return rows


def summarize(rows):
    """Token-weighted KL per mode, the share of its cut each closes (zero_* against zero_cut, the rest against cut),
    and the same per turn."""
    def tot(mode, turn=None):
        s = n = 0.
        for r in rows:
            for t, (a, b) in enumerate(r['turns'].get(mode, ())):
                if turn is None or t == turn:
                    s, n = s + a, n + b
        return s / n if n else None

    def block(turn=None):
        kls = {k: tot(k, turn) for k in rows[0]['turns']}
        closed = {}
        for k, v in kls.items():
            ref = kls.get('zero_cut' if k.startswith('zero_') else 'cut')
            if k not in ('cut', 'zero_cut') and ref and v is not None:
                closed[k] = (ref - v) / ref
        return dict(kl=kls, closed=closed)

    out = dict(block(), episodes=len(rows), positions=sum(r['positions'] for r in rows) / len(rows),
               kept={m: sum(r['kept'][m] for r in rows) / len(rows) for m in rows[0]['kept']})
    out['by_turn'] = {t + 1: dict(block(t), episodes=sum(len(r['turns']['zero_cut']) > t for r in rows))
                      for t in range(max(len(r['turns']['zero_cut']) for r in rows))}
    return out


def evaluate_select(net, device, data, n, part_tokens, most, budgets, bs=4, seed=7, adapter=True, log=None):
    """summarize() per split, overall and per group of families, on crumb_open's evaluation episodes."""
    net.eval()
    out = {}
    for split in CO.SPLITS:
        eps = []
        keep = net.source
        net.source = 'slots'                       # the fit as crumb_open's evaluation took it
        try:
            for rec in CO.records(data, split, seed):
                ep = CO.episode(net, rec, part_tokens, most)
                if ep is not None:
                    eps.append(ep)
                if len(eps) >= n:
                    break
        finally:
            net.source = keep
        eps.sort(key=lambda e: e['open']['tokens'])
        net.source = 'sentence'
        try:
            per = []
            for s0 in range(0, len(eps), bs):
                per += score_batch(net, eps[s0:s0 + bs], device, budgets, adapter)
                if log and (s0 // bs) % 25 == 0:
                    print(json.dumps(dict(split=split, episodes=len(per), of=len(eps))), file=log, flush=True)
        finally:
            net.source = keep
        if per:
            out[split] = dict(all=summarize(per))
            for g, fams in CO.GROUPS.items():
                rows = [r for r in per if r['family'] in fams]
                if rows:
                    out[split][g] = summarize(rows)
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help="opendata's folder for the teacher model (shards/ and teacher/)")
    ap.add_argument('--ckpt', required=True, help="a crumb_open slots run's trained.pt (the adapter and its slots)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--m', type=int, default=16, help="the checkpoint's slots per visit (its evaluation fit)")
    ap.add_argument('--budgets', default='16,64', help='positions kept per walker visit')
    ap.add_argument('--span', type=int, default=6144)
    ap.add_argument('--part_tokens', type=int, default=384)
    ap.add_argument('--visits', type=int, default=8)
    ap.add_argument('--n_eval', type=int, default=600)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))
    C.SPAN, C.CHUNK = args.span, 2
    net = C.build(args.device, 'kv', args.m, base=args.base, ckpt=False)
    saved = torch.load(args.ckpt, map_location=args.device)
    missing, unexpected = net.load_state_dict(saved['weights'], strict=False)
    if unexpected or not saved['weights']:
        raise SystemExit(f'checkpoint keys not in the net: {unexpected[:5]}')
    budgets = [int(b) for b in args.budgets.split(',')]
    res = evaluate_select(net, args.device, args.data, args.n_eval, args.part_tokens, args.visits, budgets,
                          log=sys.stdout)
    (out / 'scores.json').write_text(json.dumps(dict(vars(args), steps_run=saved.get('steps_run'), test=res),
                                                indent=1))
    for split, groups in res.items():
        for g, r in groups.items():
            print(split, g, json.dumps({k: round(v, 4) for k, v in r['closed'].items()}), flush=True)


if __name__ == '__main__':
    main()
