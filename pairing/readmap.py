"""Where a reader looks in its partner while it writes: for never-seen episodes of one game, one teacher-forced joint
pass with link.Pair.record on, and for every token the receiving model writes, its attention over the partner's
positions at every exchange block (heads averaged; the null key's mass kept separately).

Per episode, writes the partner's tokens (decoded pieces, with their ticks), the receiver's written tokens (with the
payload labels marked: the part only the partner can supply, session.payload_labels), each partner position's gate,
and attn[block][i][j]: the mass that receiver label i puts on partner position j at that block. Positions the
causal tick rule hides get exactly zero.

Usage: python readmap.py --run RUN/step_N --replies REPLIES.json --game relay_word --out FILE.json [--n 4]
"""
import argparse
import json

import torch

import session as S
from evaluate import load_pair
from link import SIDES, partner
from payload_ctl import episodes

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--replies', required=True)
ap.add_argument('--game', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=4, help='episodes')
ap.add_argument('--coread_data', default=None)
args = ap.parse_args()

pair, toks = load_pair(args.run)
eps = episodes(json.load(open(args.replies)), args.n, coread=True, path=args.coread_data)[args.game]
pads = {x: toks[x].pad_token_id for x in SIDES}
built = [S.build(ep, toks) for ep in eps]
b = S.collate(built, pads)
pair.record = {}
with torch.no_grad():
    pair.joint(b)
rec, pair.record = pair.record, None
out = []
for i, (ep, e) in enumerate(zip(eps, built)):
    labs = S.payload_labels(ep, toks)
    for x in SIDES:
        js = labs.get(x, [])
        if not js:
            continue
        y = partner(x)
        Ly = b[y]['ids'].shape[1]
        lab = [e[x]['labels'][j] for j in js]               # (position that predicts it, token id, weight)
        pos = [p for p, _, _ in lab]
        ids_y = b[y]['ids'][i].tolist()
        valid = b[y]['att'][i].bool().tolist()
        keep = [j for j in range(Ly) if valid[j]]
        rec_x = {blk: r for (side, blk), r in rec.items() if side == x}
        attn, null, gates = {}, {}, {}
        for blk, parts in rec_x.items():
            p = parts[0]['attn'][i].float().mean(0)          # [Lq, 1 + Lk]: heads averaged; column 0 is the null key
            attn[blk] = [[round(float(p[q, 1 + j]), 5) for j in keep] for q in pos]
            null[blk] = [round(float(p[q, 0]), 5) for q in pos]
        for (side, blk), parts in rec.items():
            if side == y:
                gates[blk] = [round(float(parts[0]['gate'][i][j]), 4) for j in keep]
        written = [toks[x].decode([t]) for _, t, _ in lab]
        out.append(dict(game=args.game, episode=i, receiver=x, payload_label_index=js,
                        partner_pieces=[toks[y].decode([ids_y[j]]) for j in keep],
                        partner_ticks=[int(b[y]['tick'][i][j]) for j in keep],
                        receiver_written=written, receiver_positions=pos,
                        receiver_pieces=[toks[x].decode([t]) for t in b[x]['ids'][i][:int(b[x]['att'][i].sum())].tolist()], partner_gate=gates,
                        attn=attn, null=null, target=ep['reply'][x][0] if 'reply' in ep else None))
json.dump(out, open(args.out, 'w'))
print('wrote', args.out, len(out), 'receivers', flush=True)
