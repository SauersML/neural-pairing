"""How far the link moves the partner along its own concept direction when one twin is steered (chains/nl31.sh).

steering_scale.py (nl29, nl30) steers one twin after block L and counts the partner's chat messages that name the
concept: 12 of 96 after block 6, 1 after block 15, 26 after block 15 at four times the strength, none after block 27 at
either strength. Here the same chat episodes (steered_mind.episode, both prompts) run linked, once with an arm's
steering vector and once with a zero vector, in the same batch shapes; only the prompt pass (tick -1, the same tokens in
both runs) is read. At each probe depth d the partner's residual stream right after the exchange there (its block-d
output plus the link's write, link.Pair.record) is differenced, steered minus unsteered, and projected on the partner's
own concept direction at d (fit by steering_scale.py at d: the arm directory given for d). The projection is divided by
the partner's own calibrated strength at d (its selected alpha times the direction's norm): 1 is as far along its
concept as the push that makes the partner itself say the concept. Reported as the mean over the partner's prompt
positions and at its last prompt position (where its first message starts); 'write' is the part the link writes at d
itself (the rest arrives through earlier exchanges and the partner's own blocks), and 'size' is the whole
difference's length, all in the same unit. The steered twin's own difference at L, in its own unit, is the arm's boost
(nothing it reads at L has changed yet): a check of the measurement.

Writes OUT/probe.json: one record per arm, steered side, concept and prompt, and the means per arm and depth.
Usage: python pairing/steer_probe.py --run STEP --depth 6:DIR 15:DIR 27:DIR --arm 6:1 15:1 15:4 27:1 27:4 --out OUT
  (--depth D:DIR: the directions.npz and selected.json fit at depth D; an arm L:B steers with the depth-L directions
  at B times their selected strengths)
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steered_mind as X  # noqa: E402

SIDES = X.SIDES


def load_depth(path):
    with np.load(Path(path) / 'directions.npz') as archive:
        vectors = dict(archive)
    return vectors, json.loads((Path(path) / 'selected.json').read_text())


@contextlib.contextmanager
def capture(pair, depths):
    """{(side, d): [block-d output per call]} while inside; the first call is the prompt pass."""
    import link
    out, handles = {}, []
    for x in SIDES:
        layers = link.find_layers(pair.models[x])
        for d in depths:
            def hook(module, inputs, output, key=(x, d)):
                h = output[0] if isinstance(output, tuple) else output
                out.setdefault(key, []).append(h.detach().float().cpu())
            handles.append(layers[d].register_forward_hook(hook))
    try:
        yield out
    finally:
        for handle in handles:
            handle.remove()


def prompt_pass(rt, eps, vectors, depths):
    """Over the prompt pass: each side's states right after the exchange at each depth d {(x, d): [B, L, dim]}, the
    link's write there {(x, d): [B, L, dim]} (zero at a depth with no exchange), and the valid prompt positions
    {x: [B, L] bool}. The steering hook (rt.steer) is registered before the capture hooks, so the captured block
    output carries the push."""
    import session as S
    import torch
    pair = rt.pair
    pair.record = {}
    try:
        with torch.inference_mode(), rt.steer(vectors), capture(pair, depths) as blocks:
            S.generate(pair, eps, max_turn=1, temperature=0.)
        states, writes, valid = {}, {}, {}
        for x in SIDES:
            valid[x] = pair.record[(x, pair.xblocks[0])][0]['write'].abs().sum(-1) > 0
            for d in depths:
                h = blocks[(x, d)][0]
                w = pair.record[(x, d)][0]['write'].float() if d in pair.xblocks else torch.zeros_like(h)
                states[(x, d)], writes[(x, d)] = h + w, w
    finally:
        pair.record = None
    return states, writes, valid


def measure(rt, vectors_at, selected_at, arms, bs=6, concepts=None):
    """Records and per-arm means; vectors_at / selected_at: {depth: directions / selected strengths fit there}."""
    depths = sorted(vectors_at)
    records = []
    for layer, boost in arms:
        rt.layer = layer
        for steered in SIDES:
            partner = SIDES[1 - SIDES.index(steered)]
            jobs = [(c, p) for c in (concepts or X.CONCEPTS) for p in range(2)]
            for start in range(0, len(jobs), bs):
                batch = jobs[start:start + bs]
                eps = [X.episode(partner, 'chat', p) for _, p in batch]
                push = np.stack([vectors_at[layer][f'{steered}/{c}'] * selected_at[layer][steered][c]['alpha'] * boost
                                 for c, _ in batch]).astype(np.float32)
                on, w_on, valid = prompt_pass(rt, eps, {steered: push}, depths)
                off, w_off, valid0 = prompt_pass(rt, eps, {steered: np.zeros_like(push)}, depths)
                for b, (c, p) in enumerate(batch):
                    v = valid0[partner][b].numpy()
                    assert (v == valid[partner][b].numpy()).all()
                    rec = dict(layer=layer, boost=boost, steered=steered, partner=partner, concept=c, prompt=p,
                               depths={})
                    for d in depths:
                        u = vectors_at[d][f'{partner}/{c}']
                        unit = float(selected_at[d][partner][c]['alpha'] * np.linalg.norm(u))
                        u = u / np.linalg.norm(u)
                        delta = (on[(partner, d)][b] - off[(partner, d)][b]).numpy()[v]
                        dw = (w_on[(partner, d)][b] - w_off[(partner, d)][b]).numpy()[v]
                        proj = delta @ u / unit
                        rec['depths'][d] = dict(mean=float(proj.mean()), last=float(proj[-1]),
                                                write=float((dw @ u / unit).mean()),
                                                size=float(np.linalg.norm(delta, axis=-1).mean() / unit))
                    if layer in depths:
                        # Nothing the steered side reads at L has changed yet, so its difference there is the push.
                        own = vectors_at[layer][f'{steered}/{c}']
                        vs = valid0[steered][b].numpy()
                        delta = (on[(steered, layer)][b] - off[(steered, layer)][b]).numpy()[vs]
                        rec['own_check'] = float(np.median(delta @ (own / np.linalg.norm(own))) /
                                                 (selected_at[layer][steered][c]['alpha'] * np.linalg.norm(own)))
                    records.append(rec)
                print('probe', layer, boost, steered, len(records), flush=True)
    means = []
    for layer, boost in arms:
        rows = [r for r in records if r['layer'] == layer and r['boost'] == boost]
        means.append(dict(layer=layer, boost=boost, n=len(rows), depths={
            d: {k: float(np.mean([r['depths'][d][k] for r in rows])) for k in ('mean', 'last', 'write', 'size')}
            for d in depths}, own_check=float(np.median([r['own_check'] for r in rows if 'own_check' in r]))
            if any('own_check' in r for r in rows) else None))
    return records, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--depth', nargs='+', required=True, help='D:DIR, the directions fit at depth D')
    ap.add_argument('--arm', nargs='+', required=True, help='L:B, steer after block L at B times')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=6)
    args = ap.parse_args()
    vectors_at, selected_at = {}, {}
    for spec in args.depth:
        d, path = spec.split(':', 1)
        vectors_at[int(d)], selected_at[int(d)] = load_depth(path)
    arms = [(int(a.split(':')[0]), float(a.split(':')[1])) for a in args.arm]
    assert all(layer in vectors_at for layer, _ in arms)
    rt = X.Runtime('torch', args.run)
    records, means = measure(rt, vectors_at, selected_at, arms, args.bs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X.save(out / 'probe.json', dict(run=str(Path(args.run).resolve()), depth=args.depth, arms=args.arm,
                                    means=means, records=records))
    for m in means:
        print(m['layer'], m['boost'], {d: round(v['last'], 3) for d, v in m['depths'].items()}, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
