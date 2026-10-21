"""What crosses the bridges, at the states where a model predicts what only its partner has (the payload).

Held-out episodes (payload_ctl.episodes: the payload games and the co-reading games), one teacher-forced joint pass
per batch of one game with link.Pair.record on. For every payload label of the receiving model x (the state that
predicts a payload token) and every exchange block:
  null      attention mass on the null key (reading nothing)
  prompt    mass on the partner's prompt (tick < 0): reading what the partner was given
  written   mass on the partner's own written tokens (tick >= 0): reading the partner's writing stream, where a code
            the pair builds can live (the partner's reply often never mentions the value)
  copy      mass on partner positions whose token is the target token (same string, at least 2 characters): plain
            copying of text
  lens_top10, lens_rank   the logit lens of the write alone (x's final norm and LM head on the bridge's write at that
            state): the fraction of payload labels whose target token is in the top 10, and the median rank. A
            token-like write decodes to the target; a write in the pair's own code does not
  rand_top10   the same for random directions in place of the write (the chance level)
  gate_prompt, gate_written   the partner's mean visibility gate over its prompt and over its written tokens
  alpha     tanh(alpha) of the reader at that block (the write's size relative to the residual stream)
Heads are averaged. Means are over payload labels; per game and receiving model, and pooled.

Usage: python analyze_bridge.py --run RUN/step_N --replies REPLIES.json --out FILE.json [--n 16] [--games a,b]
"""
import argparse
import json
import statistics

import torch

import session as S
from evaluate import load_pair
from link import DEVICE, FAR, SIDES, inner, partner
from payload_ctl import episodes


def position_masks(ticks, device='cpu'):
    """Prompt/written masks on the recorded attention's device (records live on CPU)."""
    ticks = ticks.to(device)
    prompt = (ticks < 0).float()
    written = ((ticks >= 0) & (ticks < FAR)).float()
    return prompt, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=16, help='episodes per game')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--games', default=None)
    ap.add_argument('--coread_data', default=None)
    args = ap.parse_args()

    pair, toks = load_pair(args.run)
    pair.to(DEVICE)
    pads = {x: toks[x].pad_token_id for x in SIDES}
    replies = json.load(open(args.replies))
    E = episodes(replies, args.n, coread=True, path=args.coread_data)
    if args.games:
        E = {g: E[g] for g in args.games.split(',')}
    norm_head = {x: inner(pair.models[x]) for x in SIDES}
    gen = torch.Generator().manual_seed(0)


    def piece(tok, i):
        return tok.decode([i]).strip().lower()


    @torch.no_grad()
    def lens(x, w, target):
        """Rank of each target token under x's final norm and LM head applied to the vectors w [n, d]."""
        dec, head = norm_head[x]
        logits = head(dec.norm(w.to(DEVICE, dtype=head.weight.dtype))).float()
        return (logits > logits.gather(1, target[:, None].to(DEVICE))).sum(1).cpu()


    acc = {}
    for g, eps in E.items():
        for a, z in S.batches(len(eps), args.bs):
            chunk = eps[a:z]
            built = [S.build(ep, toks) for ep in chunk]
            b = S.collate(built, pads)
            pair.record = {}
            with torch.no_grad():
                pair.joint(b)
            rec, pair.record = pair.record, None
            for x in SIDES:
                y = partner(x)
                L = b[x]['ids'].shape[1]
                ty = b[y]['tick']
                prompt, writ = position_masks(ty)
                ystr = [[piece(toks[y], int(i)) for i in row] for row in b[y]['ids']]
                off, sel = 0, []                                       # (episode in batch, position, target id)
                for k, (ep, e) in enumerate(zip(chunk, built)):
                    for j in S.payload_labels(ep, toks).get(x, []):
                        flat = int(b[x]['idx'][off + j])
                        sel.append((flat // L, flat % L, int(b[x]['lab'][off + j])))
                    off += len(e[x]['labels'])
                if not sel:
                    continue
                rows = torch.tensor([s[0] for s in sel])
                cols = torch.tensor([s[1] for s in sel])
                tgt = torch.tensor([s[2] for s in sel])
                copy = torch.zeros(len(sel), ty.shape[1])
                for n, (r, _, t) in enumerate(sel):
                    ts = piece(toks[x], t)
                    if len(ts) >= 2:
                        copy[n] = torch.tensor([float(s == ts) for s in ystr[r]])
                for blk in pair.xblocks:
                    rx, ry = rec[(x, blk)][0], rec[(y, blk)][0]
                    p = rx['attn'].mean(1)[rows, cols]                 # [n, 1 + Lk]: null key first
                    w = rx['write'][rows, cols]
                    rank = lens(x, w, tgt)
                    rnd = lens(x, torch.randn(w.shape, generator=gen), tgt)
                    gate = ry['gate']
                    row = acc.setdefault((g, x, blk), {k: [] for k in ('null', 'prompt', 'written', 'copy', 'rank',
                                                                        'rand_rank', 'gate_prompt', 'gate_written')})
                    row['null'] += p[:, 0].tolist()
                    row['prompt'] += (p[:, 1:] * prompt[rows]).sum(1).tolist()
                    row['written'] += (p[:, 1:] * writ[rows]).sum(1).tolist()
                    row['copy'] += (p[:, 1:] * copy).sum(1).tolist()
                    row['rank'] += rank.tolist()
                    row['rand_rank'] += rnd.tolist()
                    row['gate_prompt'] += ((gate * prompt).sum(1) / prompt.sum(1).clamp_min(1))[rows].tolist()
                    row['gate_written'] += ((gate * writ).sum(1) / writ.sum(1).clamp_min(1))[rows].tolist()
                    row['alpha'] = rx['alpha']
        print(f'{g}: done', flush=True)


    def summarize(row):
        m = lambda v: round(sum(v) / max(len(v), 1), 4)
        return dict(n=len(row['rank']), null=m(row['null']), prompt=m(row['prompt']), written=m(row['written']),
                    copy=m(row['copy']), lens_top10=m([float(r < 10) for r in row['rank']]),
                    lens_rank=statistics.median(row['rank']), rand_top10=m([float(r < 10) for r in row['rand_rank']]),
                    gate_prompt=m(row['gate_prompt']), gate_written=m(row['gate_written']), alpha=round(row['alpha'], 4))


    out = {'run': args.run, 'per': {}, 'pooled': {}}
    for (g, x, blk), row in sorted(acc.items()):
        out['per'].setdefault(f'{g}/{x}', {})[blk] = summarize(row)
    for blk in pair.xblocks:
        rows = [r for (g, x, b_), r in acc.items() if b_ == blk]
        pooled = {k: [v for r in rows for v in r[k]] for k in rows[0] if k != 'alpha'}
        pooled['alpha'] = statistics.mean(r['alpha'] for r in rows)
        out['pooled'][blk] = summarize(pooled)
    json.dump(out, open(args.out, 'w'), indent=1)
    for blk, r in out['pooled'].items():
        print(blk, json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
