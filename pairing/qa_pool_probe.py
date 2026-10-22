"""Exploratory confidence decoding at the QA letter position, with real shuffled partners.

Use --mlx on the Mac under macgpu.lock (4-bit; not bf16 evidence). Default torch
backend is for the lead's GPU confirmation. Fits fixed-penalty ridge probes with
three held-out folds on validation questions only. No test questions or gold
answers are used in the confidence target: it is the partner's max A-D probability
with the link cut. Separate probes use link writes, shuffled-partner writes, and
the receiver's own cut probabilities. A write probe can exploit question features
in the receiver's queries; decodability alone is NOT proof of confidence transfer.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

import games as G
import session as S
from link import SIDES, partner
from qa_pool import letter_layout


@contextmanager
def capture_writes(pair, positions, mlx=False):
    """Observe actual read() returns without changing the computation or recording attention."""
    bridge = pair.mlx.bridge if mlx else pair.bridge
    saved, got = {}, {x: {} for x in SIDES}
    for x in SIDES:
        side = bridge[x]
        saved[x] = side.read

        def read(k, h, *args, x=x, **kwargs):
            w = saved[x](k, h, *args, **kwargs)
            if mlx:
                import mlx.core as mx
                selected = mx.stack([w[i, p] for i, p in enumerate(positions[x])])
                mx.eval(selected)
                got[x][k] = np.array(selected.astype(mx.float32))
            else:
                got[x][k] = torch.stack([w[i, p] for i, p in enumerate(positions[x])]).float().detach().cpu().numpy()
            return w
        side.read = read
    try:
        yield got
    finally:
        for x in SIDES:
            bridge[x].read = saved[x]


def crossfit(features, target, folds):
    """Fixed ridge penalty: 10 times mean training kernel diagonal, with train-only scaling."""
    predicted = np.empty_like(target)
    for f in sorted(set(folds)):
        fit, test = folds != f, folds == f
        A, B = features[fit].astype(np.float64), features[test].astype(np.float64)
        mu, sd = A.mean(0), A.std(0)
        sd = np.maximum(sd, 1e-6)
        A, B = (A - mu) / sd, (B - mu) / sd
        K = A @ A.T / A.shape[1]
        penalty = 10 * max(float(np.trace(K) / len(K)), 1e-8)
        base = target[fit].mean()
        weights = np.linalg.solve(K + penalty * np.eye(len(K)), target[fit] - base)
        predicted[test] = base + (B @ A.T / A.shape[1]) @ weights
    return predicted


def summarize(features, target, folds):
    pred = crossfit(features, target, folds)
    baseline = np.empty_like(target)
    for f in sorted(set(folds)):
        baseline[folds == f] = target[folds != f].mean()
    mse = float(np.mean((target - pred) ** 2))
    null_mse = float(np.mean((target - baseline) ** 2))
    return dict(mse=mse, baseline_mse=null_mse, heldout_r2=1 - mse / null_mse,
                prediction=pred.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mlx', action='store_true')
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--seed', type=int, default=28)
    ap.add_argument('--features', help='optional local .npz of measured writes for probe audits')
    args = ap.parse_args()
    if args.bs < 2 or args.n < 12:
        ap.error('need --bs >= 2 and --n >= 12')
    if args.mlx:
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(256 * 2**20)
    else:
        from evaluate import load_pair
    pair, toks = load_pair(args.run)
    pool = G.bench('val')
    ids = random.Random(args.seed).sample(range(len(pool)), min(args.n, len(pool)))
    items = [pool[i] for i in ids]
    result = dict(checkpoint=args.run, numerics='4-bit MLX' if args.mlx else 'torch bf16 bases / float32 bridge',
                  split='val', seed=args.seed, indices=ids, n=len(ids), bs=args.bs,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  caveat='Exploratory cross-fitted decoding, not a causal proof or a test-set accuracy estimate.')
    probs = {c: {x: [] for x in SIDES} for c in ('link', 'zero', 'shuffle')}
    features = {c: {x: [] for x in SIDES} for c in ('link', 'shuffle')}
    pads = {x: toks[x].pad_token_id for x in SIDES}
    with torch.inference_mode():
        for a, z in S.batches(len(items), args.bs):
            rows = [(G.qa_episode(it, 'pair'), None) for it in items[a:z]]
            built = [S.build(ep, toks) for ep, _ in rows]
            batch = S.collate(built, pads)
            _, at, lid = letter_layout(toks, rows, built)
            for c in probs:
                if c == 'zero':
                    h, _ = pair.joint(batch, mode='zero')
                else:
                    with capture_writes(pair, at, args.mlx) as writes:
                        h, _ = pair.joint(batch, mode=None if c == 'link' else 'shuffle')
                    for x in SIDES:
                        features[c][x].append(np.concatenate([writes[x][k] for k in sorted(writes[x])], axis=1))
                for x in SIDES:
                    idx = torch.tensor([i * batch[x]['ids'].shape[1] + p for i, p in enumerate(at[x])],
                                       device=h[x].device)
                    lg = pair.token_logits(x, h[x], idx)[:, lid[x]].float()
                    probs[c][x].append(lg.softmax(-1).cpu().numpy())
                del h
                if args.mlx:
                    mx.clear_cache()
            print(f'{z}/{len(items)} validation questions', flush=True)
    probs = {c: {x: np.concatenate(v) for x, v in q.items()} for c, q in probs.items()}
    # Keep each shuffled batch wholly in one fold: held-out questions must not
    # appear as donors in the fit set. Use the same folds for every control.
    folds = np.empty(len(items), dtype=int)
    for k, (a, z) in enumerate(S.batches(len(items), args.bs)):
        folds[a:z] = k % 3
    if args.features:
        np.savez_compressed(args.features, **{c + '_' + x: np.concatenate(features[c][x])
                                             for c in features for x in SIDES})
    result['folds'] = folds.tolist()
    result['probes'] = {}
    result['probabilities'] = {c: {x: p.tolist() for x, p in q.items()} for c, q in probs.items()}
    for x in SIDES:
        y = partner(x)
        target = probs['zero'][y].max(1).astype(np.float64)
        letter = np.eye(4)[probs['zero'][y].argmax(1)]
        tests = {'receiver_cut_probs': probs['zero'][x], 'partner_letter': letter,
                 'receiver_probs_and_partner_letter': np.concatenate([probs['zero'][x], letter], axis=1)}
        tests.update({c + '_writes': np.concatenate(features[c][x]) for c in features})
        result['probes'][x] = dict(target='partner cut max A-D probability', target_values=target.tolist(),
                                   **{name: summarize(X, target, folds) for name, X in tests.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({x: {k: v['heldout_r2'] for k, v in t.items() if isinstance(v, dict)}
                      for x, t in result['probes'].items()}), flush=True)


if __name__ == '__main__':
    main()
