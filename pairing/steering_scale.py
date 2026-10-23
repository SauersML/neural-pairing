"""Natural-projection calibration for steered_mind; its evaluation stays unchanged.

Use a fresh output directory. Mac: lockf -k ~/mpd-data/locks/macgpu.lock
env MPD_MEM_GIB=12 python pairing/steering_scale.py --backend mlx --run CKPT --out OUT.
The original protocol is retained verbatim alongside a separate calibration amendment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import steered_mind as X

STRENGTHS = [0., .125, .25, .5, 1., 1.5, 2., 3., 4., 6., 8.]
LAYERS = (6, 9, 12, 15, 18, 21, 24, 27)


def amendment(geometry='sink_free', layer=15, boost=1.):
    extra = {} if boost == 1 else dict(boost=boost, boost_rule='cut check and evaluation steer at boost x the selected '
                                       'strength; calibration and selection unchanged')
    return dict(**extra, schema=2, original_protocol_sha256=X.digest(X.protocol()),
                strengths=STRENGTHS, layer=layer,
                geometry=geometry,
                direction='difference of means with neutral-only sink mask' if geometry == 'sink_free'
                else 'unmodified difference of means',
                sink_rule='exclude the four largest neutral-energy coordinates only if together they carry >=95% of energy',
                scale='positive minus neutral mean projection on the fitted unit direction',
                unit='one natural mean projection gap along the unit difference direction',
                selection='original mention/fluency rule, smallest passing candidate; original fallback',
                unchanged=('texts, layer, ' if layer == 15 else 'texts, ') +
                'all-token injection, scoring, controls, counts and statistical tests')


def collect(rt, side, texts, layers=LAYERS):
    """Actual link-cut content states; same token averaging as Runtime.states.

    Retain per-text means and second moments, allowing projection and residual
    concentration diagnostics without storing every token's full activation.
    """
    import session as S
    means = {l: [] for l in layers}
    energy = {l: [] for l in layers}
    rms = {l: [] for l in layers}
    for text in texts:
        ids = S.ids(rt.toks[side], text)
        if len(ids) < 2:
            raise ValueError('At least two tokens required after excluding the first token')
        if rt.backend == 'mlx':
            import mlx.core as mx
            model = rt.pair.mlx.models[side]
            h = model.model.embed_tokens(mx.array([ids]))
            for l in range(max(layers) + 1):
                h = model.layers[l](h, 'causal')
                mx.eval(h)
                if l in means:
                    a = np.array(h[0, 1:].astype(mx.float32))
                    means[l].append(a.mean(0))
                    energy[l].append((a * a).mean(0))
                    rms[l].append(float(np.sqrt((a * a).mean(-1)).mean()))
        else:
            import torch
            import link
            with torch.inference_mode():
                t = torch.tensor([ids], device=link.DEVICE)
                run = link.Run(rt.pair.models[side], t, torch.ones_like(t),
                               torch.arange(len(ids), device=link.DEVICE)[None])
                for l in range(max(layers) + 1):
                    run.block(l)
                    if l in means:
                        a = run.h[0, 1:].float().cpu().numpy()
                        means[l].append(a.mean(0))
                        energy[l].append((a * a).mean(0))
                        rms[l].append(float(np.sqrt((a * a).mean(-1)).mean()))
    return {l: dict(means=np.stack(means[l]), energy=np.stack(energy[l]), rms=np.array(rms[l])) for l in layers}


def sink_mask(energy):
    """Detect an extreme coordinate sink using only neutral residuals.

    Four of 2560 coordinates holding >=95% of energy is a low-dimensional
    background, not a useful total-norm scale. Ordinary concentration (Quill)
    is left alone. The same rule applies independently to both models/layers.
    """
    e = energy.mean(0).astype(np.float64)
    order = np.argsort(e)[::-1]
    mask = np.ones(e.shape, dtype=np.float32)
    if len(e) > 4 and e[order[:4]].sum() >= .95 * e.sum():
        mask[order[:4]] = 0
    return mask


def projection(positive, neutral, mask=None):
    """One vector moves the neutral mean projection to the positive mean."""
    d = positive.mean(0) - neutral.mean(0)
    if mask is not None:
        d = d * mask
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Concept direction must have a positive finite norm')
    u = d / norm
    pp, pn = positive @ u, neutral @ u
    return d.astype(np.float32), dict(norm=norm, positive=pp.tolist(), neutral=pn.tolist(),
                                     positive_mean=float(pp.mean()), neutral_mean=float(pn.mean()),
                                     gap=float(pp.mean() - pn.mean()))


def concentration(energy):
    e = energy.mean(0).astype(np.float64)
    order = np.argsort(e)[::-1]
    return dict(top_dimensions=order[:10].tolist(),
                top_fractions=(e[order[:10]] / e.sum()).tolist(),
                top1=float(e[order[:1]].sum() / e.sum()),
                top10=float(e[order[:10]].sum() / e.sum()))


def fit(rt, dest, geometry='sink_free', layer=15):
    vectors, report = {}, {}
    for side in X.SIDES:
        ntexts = [f.format(s) for f in X.FRAMES for s in X.NEUTRAL]
        neutral = collect(rt, side, ntexts)
        report[side] = dict(neutral={str(l): dict(rms=float(n['rms'].mean()),
                                                 **concentration(n['energy'])) for l, n in neutral.items()},
                            concepts={})
        for ci, (c, (_, sentences)) in enumerate(X.CONCEPTS.items()):
            pos = collect(rt, side, [f.format(s) for f in X.FRAMES for s in sentences])
            np.savez(dest / ('states_' + side + '_' + c + '.npz'), **pos[layer])
            layers = {}
            for l in LAYERS:
                p, n = pos[l]['means'], neutral[l]['means']
                mask = sink_mask(neutral[l]['energy']) if geometry == 'sink_free' else None
                d, row = projection(p, n, mask)
                row['excluded_dimensions'] = [] if mask is None else np.flatnonzero(mask == 0).tolist()
                row['raw_norm'] = float(np.linalg.norm(p.mean(0) - n.mean(0)))
                residual_norm = float(neutral[l]['rms'].mean() * np.sqrt(d.size))
                row.update(residual_norm=residual_norm, old_unit_over_gap=residual_norm / row['norm'],
                           direction_concentration=concentration(d[None] ** 2))
                # Cross-fit by content sentence, keeping all five frames of a
                # sentence in the same half. No fitted/test prompt is used here.
                held = np.arange(len(p)) % len(sentences) >= len(sentences) // 2
                dc, _ = projection(p[~held], n[~held], mask)
                uc = dc / np.linalg.norm(dc)
                row['heldout_gap'] = float((p[held] @ uc).mean() - (n[held] @ uc).mean())
                layers[str(l)] = row
                if l == layer:
                    key = side + '/' + c
                    vectors[key] = d
                    rng = np.random.default_rng(21022 + X.SIDES.index(side) * 100 + ci)
                    noise = rng.normal(size=d.shape).astype(np.float32)
                    vectors[key + '/random'] = noise / np.linalg.norm(noise) * np.linalg.norm(d)
            report[side]['concepts'][c] = layers
            X.save(dest / 'fit.json', report)
            print('fit', side, c, 'gap', layers[str(layer)]['norm'], 'old/gap', layers[str(layer)]['old_unit_over_gap'],
                  flush=True)
        np.savez(dest / ('states_' + side + '_neutral.npz'), **{
            str(l) + '_' + k: a for l, fields in neutral.items() for k, a in fields.items()})
    np.savez(dest / 'directions.npz', **vectors)
    return vectors


def choose(rows):
    """Original calibration decision, with only the candidate grid changed."""
    for a in STRENGTHS:
        group = [r for r in rows if r['alpha'] == a]
        if len(group) != len(X.CALIBRATION) or {r['prompt'] for r in group} != set(range(len(X.CALIBRATION))):
            raise ValueError('Calibration requires exactly one row per strength and prompt')
    baseline = sum(r['mention'] for r in rows if r['alpha'] == 0)
    good = []
    for a in STRENGTHS[1:]:
        group = [r for r in rows if r['alpha'] == a]
        if all(r['fluent_proxy'] for r in group):
            good.append(a)
            count = sum(r['mention'] for r in group)
            if count >= 2 and count >= baseline + 2:
                return dict(alpha=a, calibrated=True, baseline_mentions=baseline)
    return dict(alpha=max(good, default=0.), calibrated=False, baseline_mentions=baseline)


def calibration(rt, vectors, dest, bs):
    import games as G
    path = dest / 'calibration.json'
    records = json.loads(path.read_text()) if path.exists() else []
    finished = {(r['concept'], r['alpha'], r['prompt'], r['side']) for r in records}
    jobs = [(c, a, p) for c in X.CONCEPTS for a in STRENGTHS for p in range(len(X.CALIBRATION))
            if any((c, a, p, x) not in finished for x in X.SIDES)]
    for start in range(0, len(jobs), bs):
        batch = jobs[start:start + bs]
        eps = [dict(game='steered_mind', mode='duplex', system={x: G.system(x) for x in X.SIDES},
                    task={x: X.CALIBRATION[p] for x in X.SIDES}, reply={x: ('', 1.) for x in X.SIDES}, meta={})
               for c, a, p in batch]
        vv = {x: np.stack([vectors[x + '/' + c] * a for c, a, p in batch]) for x in X.SIDES}
        outputs = rt.generate(eps, vv, 'zero', 32)
        for (c, a, p), output in zip(batch, outputs):
            for x in X.SIDES:
                if (c, a, p, x) in finished:
                    continue
                text = output['text'][x]
                records.append(dict(concept=c, alpha=a, prompt=p, side=x, text=text,
                                    mention=bool(X.matches(text, c)), fluent_proxy=X.fluent(text),
                                    tokens=output['tokens'][x], sampled=output['sampled'][x]))
        X.save(path, records)
        if rt.backend == 'mlx':
            import mlx.core as mx
            import resource
            print('memory', dict(active_gib=mx.get_active_memory() / 2**30,
                                 peak_gib=mx.get_peak_memory() / 2**30,
                                 peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30), flush=True)
        print('calibration', len(records), '/', len(X.CONCEPTS) * len(STRENGTHS) * 6, flush=True)
    selected = {x: {c: choose([r for r in records if r['side'] == x and r['concept'] == c])
                    for c in X.CONCEPTS} for x in X.SIDES}
    X.save(dest / 'selected.json', selected)
    return selected


def cut_check(rt, vectors, selected, dest, bs):
    """Steer only one solo; record its unsteered partner and zero baseline.

    The original calibration steers both isolated sides simultaneously. This
    extra diagnostic verifies no intervention reaches the cut partner.
    """
    import games as G
    records = []
    for sender in X.SIDES:
        receiver = X.SIDES[1 - X.SIDES.index(sender)]
        jobs = [(c, p) for c in X.CONCEPTS for p in range(3)]
        for start in range(0, len(jobs), bs):
            batch = jobs[start:start + bs]
            eps = [dict(game='steered_mind', mode='duplex', system={x: G.system(x) for x in X.SIDES},
                        task={x: X.CALIBRATION[p] for x in X.SIDES}, reply={x: ('', 1.) for x in X.SIDES}, meta={})
                   for c, p in batch]
            vv = {sender: np.stack([vectors[sender + '/' + c] * selected[sender][c]['alpha'] for c, p in batch])}
            steered = rt.generate(eps, vv, 'zero', 32)
            base = rt.generate(eps, {sender: np.zeros_like(vv[sender])}, 'zero', 32)
            for (c, p), s, b in zip(batch, steered, base):
                identical = s['tokens'][receiver] == b['tokens'][receiver]
                if not identical:
                    raise AssertionError('Cut partner changed under sender-only steering')
                records.append(dict(sender=sender, receiver=receiver, concept=c, prompt=p,
                                    alpha=selected[sender][c]['alpha'], partner_token_identical=identical,
                                    sender_mention=bool(X.matches(s['text'][sender], c)),
                                    sender_fluent=X.fluent(s['text'][sender]),
                                    partner_mention=bool(X.matches(s['text'][receiver], c)),
                                    text=s['text'], baseline_text=b['text']))
            X.save(dest / 'cut_check.json', records)
            print('cut check', len(records), '/ 72', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', choices=('fit', 'calibrate', 'cut', 'evaluate', 'validate', 'all'), default='all',
                    help='validate fits/calibrates/checks cut isolation without running transfer tests')
    ap.add_argument('--bs', type=int, default=6)
    ap.add_argument('--layer', type=int, choices=LAYERS, default=15,
                    help='steer after this block instead of the preregistered 15 (recorded in the amendment)')
    ap.add_argument('--boost', type=float, default=1.,
                    help='cut check and evaluation at this multiple of the calibrated strength (recorded in the amendment)')
    ap.add_argument('--geometry', choices=('sink_free', 'raw'), default='sink_free',
                    help='raw reproduces the diagnostic that only changes strength; sink_free also removes neutral sinks')
    args = ap.parse_args()
    if args.bs < 1:
        ap.error('--bs must be positive')
    if not args.boost > 0:
        ap.error('--boost must be positive')
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    fixed = amendment(args.geometry, args.layer, args.boost)
    registration = dict(amendment=fixed, sha256=X.digest(fixed))
    path = dest / 'amendment.json'
    if path.exists():
        if json.loads(path.read_text()) != registration:
            raise ValueError('Calibration amendment changed; use a new output directory')
    elif any(dest.glob('*.json')) or (dest / 'directions.npz').exists():
        raise ValueError('Use a fresh output directory; never overwrite an original experiment')
    X.save(path, registration)
    X.save(dest / 'protocol.json', dict(protocol=X.protocol(), sha256=X.digest(X.protocol())))
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    manifest = dict(checkpoint=str(Path(args.run).resolve()), backend=args.backend, batch_size=args.bs,
                    amendment_sha256=registration['sha256'], started_unix=time.time(),
                    numerical_note='4-bit MLX differs from bf16; requires GPU confirmation' if args.backend == 'mlx'
                    else 'bf16 torch CUDA',
                    code_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ('steering_scale.py', 'steered_mind.py', 'mlx_eval.py', 'mlx_pair.py',
                                           'link.py', 'session.py', 'games.py')})
    if (dest / 'manifest.json').exists():
        old = json.loads((dest / 'manifest.json').read_text())
        for k in ('checkpoint', 'backend', 'batch_size', 'amendment_sha256'):
            if old[k] != manifest[k]:
                raise ValueError('Cannot mix runs: ' + k)
    X.save(dest / 'manifest.json', manifest)
    rt = X.Runtime(args.backend, args.run)
    rt.layer = args.layer
    if args.phase in ('fit', 'validate', 'all') and not (dest / 'directions.npz').exists():
        vectors = fit(rt, dest, args.geometry, args.layer)
    else:
        with np.load(dest / 'directions.npz') as archive:
            vectors = dict(archive)
    if args.phase in ('calibrate', 'validate', 'all'):
        calibration(rt, vectors, dest, args.bs)
    if args.phase in ('cut', 'evaluate', 'validate', 'all'):
        selected = json.loads((dest / 'selected.json').read_text())
        selected = {x: {c: dict(v, alpha=v['alpha'] * args.boost) for c, v in s.items()} for x, s in selected.items()}
        if args.phase in ('cut', 'validate', 'all'):
            cut_check(rt, vectors, selected, dest, args.bs)
        if args.phase in ('evaluate', 'all'):
            X.evaluate(rt, vectors, selected, dest, args.bs)
    manifest.update(elapsed_seconds=time.time() - manifest['started_unix'], completed_phase=args.phase)
    X.save(dest / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
