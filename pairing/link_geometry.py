"""Observe real link writes in opening, no-objective chats; fit on separate wordings.

Mac collection MUST run under lockf macgpu.lock, MPD_MEM_GIB=12. MLX uses
mlx_eval.load_pair (4-bit bases). The torch backend refuses real models unless
CUDA is selected; it is for the lead's bf16 confirmation, never a Mac fallback.

The primary measurement is the position predicting the FIRST reply token, before
any sampled text. Two later prediction positions are also saved, with their
actual emitted IDs. The receiver opens the conversation, so the partner has not
written text. The receiver's own city is fixed, independently of the target.
Each wording is crossed with every test city, in each receiver direction.

Instrumentation calls the original _exchange and observes output minus input,
including residual dtype rounding, without switching away from fused attention.
No existing score or forward path is replaced. Feature vectors are saved in an
NPZ beside the raw manifest, outside the repository by default.

PCA, regularized LDA, and the ridge linear classifier fit on wordings 0..2 only.
Plots contain wordings 3..5 only. City labels in probe fitting are test-pool
values never used as private values in model training, but necessarily known to
the supervised probe. Projection and probe generalization concern WORDINGS,
not new city classes. LDA separation is not an independent transfer test.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('quill', 'ember')
WORDINGS = (
    'You are spending today in {city}. There is no game and nothing you have to do: just chat with {P} about whatever you like.',
    'You have a free afternoon in {city}. Enjoy an ordinary conversation with {P} about whatever interests you.',
    'For today, your location is {city}. You and {P} can simply talk; there is no task to complete.',
    'Your day is unfolding in {city}. Chat casually with {P}, with no particular goal in mind.',
    'At the moment you are visiting {city}. Take a little time to talk with {P} about anything you wish.',
    'You are relaxing in {city} today. Have a spontaneous conversation with {P}; nothing needs to be solved.',
)
SCRATCH = Path('/Users/user/mpd-data/scratch/codex-swarm/wave20/link_geometry')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def design(city_count=None):
    """Balanced factorial: no city is encoded in row order or receiver input."""
    pools = json.loads((ROOT / 'data/pools.json').read_text())['cities']
    own = pools['test'][-1]
    cities = [c for c in pools['test'] if c != own]
    if city_count is not None:
        cities = cities[:city_count]
    assert set(cities).isdisjoint(pools['train']) and own not in cities
    return [dict(receiver=x, city=c, own_city=own, wording=w, split='fit' if w < 3 else 'test')
            for x in SIDES for w in range(len(WORDINGS)) for c in cities]


def episode(row):
    import coread as C
    y = SIDES[1 - SIDES.index(row['receiver'])]
    private = {row['receiver']: row['own_city'], y: row['city']}
    systems = {x: C.chat_system(x, private[x], WORDINGS[row['wording']]) for x in SIDES}
    assert row['city'].casefold() not in systems[row['receiver']].casefold()
    return dict(game='co_chat', mode='dialogue', system=systems, task={x: None for x in SIDES},
                turns=[(row['receiver'], '', 1.)], meta=dict(private=private, opener=row['receiver']))


def cpu(a):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy().copy()
    import mlx.core as mx
    return np.array(a.astype(mx.float32))


class Capture:
    """Read-only observer: capture the actual applied delta at each prediction."""
    def __init__(self, pair, receiver, n_tokens):
        self.pair, self.receiver, self.n_tokens = pair, receiver, n_tokens
        self.values, self.residual, self.tick = {}, {}, -1

    def step(self, chunk, cache, qmask, tick, *args, **kwargs):
        self.tick = tick
        x = self.receiver
        self.cols = (chunk[x][1][:, -chunk[x][0].shape[1]:].sum(-1) - 1).tolist() if x in chunk else []
        return self.original_step(chunk, cache, qmask, tick, *args, **kwargs)

    def exchange(self, k, h, *args, **kwargs):
        out, gates = self.original_exchange(k, h, *args, **kwargs)
        x = self.receiver
        if 0 <= self.tick < self.n_tokens and x in h:
            import numpy as np
            # Slice on device before materializing; never save full prompts.
            before = np.stack([cpu(h[x][i, p]) for i, p in enumerate(self.cols)])
            after = np.stack([cpu(out[x][i, p]) for i, p in enumerate(self.cols)])
            self.values[self.tick, k] = after - before
            self.residual[self.tick, k] = np.linalg.norm(before, axis=-1)
        return out, gates

    @contextlib.contextmanager
    def installed(self):
        p = self.pair
        self.original_step, self.original_exchange = p.step, p._exchange
        old = {k: p.__dict__.get(k) for k in ('step', '_exchange')}
        present = {k: k in p.__dict__ for k in old}
        p.step, p._exchange = self.step, self.exchange
        try:
            yield self
        finally:
            for k, v in old.items():
                if present[k]:
                    setattr(p, k, v)
                else:
                    delattr(p, k)


def collect(args):
    import numpy as np
    import torch
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(int(10.5 * 2**30))
        mx.set_cache_limit(32 * 2**20)
    else:
        if not os.environ.get('NL_TINY') and not os.environ.get('NL_DEVICE', '').startswith('cuda'):
            raise ValueError('Real torch models require NL_DEVICE=cuda; Mac uses locked 4-bit MLX only')
        from evaluate import load_pair
    import session as S
    torch.set_num_threads(2)
    rows = design(args.cities)
    code_hashes = {f: sha(ROOT / 'pairing' / f) for f in
                   ('link_geometry.py', 'mlx_eval.py', 'mlx_pair.py', 'session.py', 'coread.py')}
    args.raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pair, toks = load_pair(args.run)
    for offset in range(0, len(rows), args.bs):
        part = rows[offset:offset + args.bs]
        # Batches never cross receiver boundaries, even with an unusual --bs.
        if len({r['receiver'] for r in part}) > 1:
            raise ValueError('--bs must divide the prompts per receiver')
        dest = args.raw / f'batch_{offset:04d}.npz'
        meta = dest.with_suffix('.json')
        if dest.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if (saved['design'] != part or saved['run'] != str(Path(args.run).resolve())
                    or saved['backend'] != args.backend or saved['tokens_per_prompt'] != args.tokens):
                raise ValueError('resume manifest differs')
            continue
        x = part[0]['receiver']
        eps = [episode(r) for r in part]
        obs = Capture(pair, x, args.tokens)
        with torch.inference_mode(), obs.installed():
            raw = S.generate(pair, eps, max_turn=args.tokens, keep_tokens=True)
        vectors = np.stack([np.stack([obs.values[t, k] for k in range(len(pair.xblocks))], 1)
                            for t in range(args.tokens)], 1)
        residual = np.stack([np.stack([obs.residual[t, k] for k in range(len(pair.xblocks))], 1)
                             for t in range(args.tokens)], 1)
        ratios = np.linalg.norm(vectors, axis=-1) / np.maximum(residual, 1e-30)
        samples = []
        for r, ep, o in zip(part, eps, raw):
            emitted = o['tokens'][0]
            if len(emitted) < args.tokens:
                raise ValueError('an early EOS needs an explicitly missing-token analysis')
            samples.append(dict(r, system=ep['system'], tokens=emitted[:args.tokens],
                                prompt_tokens={s: len(S.prefix(toks[s], s, ep['system'][s], None)) for s in SIDES},
                                pieces=[toks[x].decode([t]) for t in emitted[:args.tokens]], text=o['turns'][0][1]))
        np.savez_compressed(dest, vectors=vectors, residual_norm=residual, ratios=ratios)
        meta.write_text(json.dumps(dict(design=part, samples=samples, run=str(Path(args.run).resolve()),
                                        backend=args.backend, blocks=pair.xblocks,
                                        tokens_per_prompt=args.tokens, code_sha256=code_hashes), indent=2))
        print(json.dumps(dict(done=offset + len(part), total=len(rows), seconds=round(time.monotonic()-start, 1))), flush=True)
        if args.backend == 'mlx':
            mx.clear_cache()
    files = ['bridge.pt', 'quill/adapter_model.safetensors', 'ember/adapter_model.safetensors']
    (args.raw / 'manifest.json').write_text(json.dumps(dict(
        run=str(Path(args.run).resolve()), backend=args.backend, quantization='4-bit' if args.backend == 'mlx' else 'bf16',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256={f: sha(Path(args.run) / f) for f in files},
        code_sha256=code_hashes,
        blocks=pair.xblocks, samples=len(rows), seconds=time.monotonic()-start), indent=2))


def read_raw(path):
    import numpy as np
    rows, vectors, ratios = [], {x: [] for x in SIDES}, {x: [] for x in SIDES}
    manifest = json.loads((path / 'manifest.json').read_text())
    for f in sorted(path.glob('batch_*.json')):
        m = json.loads(f.read_text())
        a = np.load(f.with_suffix('.npz'))
        x = m['samples'][0]['receiver']
        rows += m['samples']
        vectors[x].append(a['vectors'])
        ratios[x].append(a['ratios'])
    assert len(rows) == manifest['samples']
    return rows, {x: np.concatenate(vectors[x]) for x in SIDES}, {x: np.concatenate(ratios[x]) for x in SIDES}, manifest


def linear_probe(a, y, fit, ridge=.01):
    """Ridge classifier with a fixed trace-scaled penalty, solved in the dual."""
    import numpy as np
    z = a.astype(np.float64) - a[fit].mean(0)
    train = z[fit]
    target = np.eye(int(y.max()) + 1)[y[fit]]
    target -= target.mean(0)
    gram = train @ train.T
    penalty = ridge * np.trace(gram) / len(train)
    dual = np.linalg.solve(gram + np.eye(len(train)) * max(penalty, 1e-20), target)
    return (z[~fit] @ train.T @ dual).argmax(1)


def project(a, y, fit):
    """Fit-only PCA and ridge-within-covariance Fisher LDA, same held-out points."""
    import numpy as np
    from scipy.linalg import eigh
    mean = a[fit].mean(0)
    z = a.astype(np.float64) - mean
    _, sing, vt = np.linalg.svd(z[fit], full_matrices=False)
    rank = int((sing > sing[0] * 1e-7).sum())
    basis = vt[:rank].T
    train = z[fit] @ basis
    yy = y[fit]
    centres = np.stack([train[yy == c].mean(0) for c in np.unique(y)])
    within = train - centres[yy]
    sw = within.T @ within / len(train)
    sb = centres.T @ centres / len(centres)
    # Fixed shrinkage; no tuning against the plotted data.
    reg = .1 * np.trace(sw) / max(rank, 1)
    vals, axes = eigh(sb, sw + np.eye(rank) * max(reg, 1e-20))
    lda = z @ basis @ axes[:, np.argsort(vals)[-2:][::-1]]
    pca = z @ vt[:2].T
    return dict(pca=pca[~fit].tolist(), lda=lda[~fit].tolist(),
                pca_variance=(sing[:2] ** 2 / (sing ** 2).sum()).tolist(),
                lda_eigenvalues=sorted(vals.tolist(), reverse=True)[:2], fit_rank=rank)


def analyze(args):
    import numpy as np
    before = None if args.after_only else read_raw(args.before)
    after = read_raw(args.after)
    keys = ('receiver', 'city', 'own_city', 'wording', 'split')
    if before is not None:
        assert [[r[k] for k in keys] for r in before[0]] == [[r[k] for k in keys] for r in after[0]]
    result = dict(schema=1, complete=before is not None, measurement=__doc__,
                  before=before[3] if before is not None else None, after=after[3],
                  wordings=list(WORDINGS), primary_prediction_index=0,
                  projection_fit_wordings=[0, 1, 2], projection_test_wordings=[3, 4, 5],
                  ridge_penalty=.01, lda_shrinkage=.1, map_block=15, per_receiver={})
    chat_path = Path('/Users/user/mpd-data/neural-pairing/chatdata/chat_clean.json')
    if chat_path.exists():
        chats = json.loads(chat_path.read_text())
        training_cities = {c for r in chats['train'] for c in r['private'].values() if c}
        measured_cities = {r['city'] for r in after[0]}
        assert not training_cities & measured_cities
        result['training_value_audit'] = dict(path=str(chat_path), sha256=sha(chat_path),
                                             training_chats=len(chats['train']),
                                             training_private_cities=sorted(training_cities),
                                             overlap=[])
    for x in SIDES:
        rows = [r for r in after[0] if r['receiver'] == x]
        cities = sorted({r['city'] for r in rows})
        y = np.array([cities.index(r['city']) for r in rows])
        fit = np.array([r['split'] == 'fit' for r in rows])
        wording = np.array([r['wording'] for r in rows])
        blocks = after[3]['blocks']
        records = []
        for k, block in enumerate(blocks):
            features = after[1][x][:, 0, k]
            pred = linear_probe(features, y, fit)
            p = project(features, y, fit)
            wording_means = np.stack([features[wording == w].mean(0) for w in range(len(WORDINGS))])
            grand = features.mean(0)
            wording_variance = float(sum(np.square(mu-grand).sum() * (wording == w).sum()
                                         for w, mu in enumerate(wording_means)) /
                                     np.square(features-grand).sum())
            diagnostic = dict(
                wording_variance_fraction=wording_variance,
                ridge_correct={str(lam): int((linear_probe(features, y, fit, lam) == y[~fit]).sum())
                               for lam in (.001, .01, .1, 1.)},
                leave_one_wording_out_correct=[int((linear_probe(features, y, wording != w) == y[wording == w]).sum())
                                              for w in range(len(WORDINGS))],
                centered_correct=int((linear_probe(features-wording_means[wording], y, fit) == y[~fit]).sum()),
                centered_note='Diagnostic only: subtracts each wording mean, including unlabelled balanced test '
                              'sets. This is a transductive nuisance check, not the primary probe score.')
            later = []
            for token in range(1, after[1][x].shape[1]):
                q = linear_probe(after[1][x][:, token, k], y, fit)
                later.append(dict(prediction_index=token, correct=int((q == y[~fit]).sum()),
                                  total=int((~fit).sum()), prediction=q.tolist()))
            permutation_hits = []
            # Permute city labels independently within each fit wording.
            # Preserve balanced classes; apply no permutation to test labels.
            for seed in range(32):
                rng = np.random.default_rng(seed)
                yp = y.copy()
                for w in range(3):
                    ix = np.array([i for i, r in enumerate(rows) if r['wording'] == w])
                    yp[ix] = rng.permutation(yp[ix])
                permutation_hits.append(int((linear_probe(features, yp, fit) == y[~fit]).sum()))
            rate = {}
            for name, data in [('before', before), ('after', after)]:
                if data is None:
                    continue
                values = data[2][x][:, 0, k]
                rate[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                  min=float(values.min()), max=float(values.max()), values=values.tolist())
            records.append(dict(block=block, correct=int((pred == y[~fit]).sum()), total=int((~fit).sum()),
                                accuracy=float((pred == y[~fit]).mean()), prediction=pred.tolist(),
                                expected=y[~fit].tolist(), permutation_correct=permutation_hits,
                                diagnostics=diagnostic, later_prediction_probes=later,
                                write_preview={str(w): features[next(i for i, r in enumerate(rows)
                                                                    if r['wording'] == w and r['city'] == cities[0]), :24].tolist()
                                               for w in (0, 3)}, ratio=rate, **p))
        result['per_receiver'][x] = dict(cities=cities, chance=1/len(cities), samples=rows,
                                        test_indices=np.flatnonzero(~fit).tolist(), layers=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for x, r in result['per_receiver'].items():
        print(x, [(v['block'], v['correct'], v['total'], round(v['ratio']['after']['mean'], 4)) for v in r['layers']])


def selftest():
    """Real tiny models: recorder does not alter generation; delta matches joint."""
    import numpy as np
    import torch
    import link
    import session as S
    from peft import LoraConfig, get_peft_model
    assert os.environ.get('NL_TINY') and link.DEVICE == 'cpu'
    torch.set_num_threads(2)
    models, toks = {}, {}
    for x in SIDES:
        toks[x], m = link.load(x)
        models[x] = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'])).eval()
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    with torch.no_grad():
        for side in p.bridge.values():
            side.alpha.fill_(.4)
    rows = design(2)[:2]
    eps = [episode(r) for r in rows]
    baseline = S.generate(p, eps, max_turn=3, keep_tokens=True)
    cap = Capture(p, 'quill', 3)
    with cap.installed():
        measured = S.generate(p, eps, max_turn=3, keep_tokens=True)
    assert [r['tokens'] for r in baseline] == [r['tokens'] for r in measured]
    assert 'step' not in p.__dict__ and '_exchange' not in p.__dict__
    replay = [S.replay(ep, o) for ep, o in zip(eps, measured)]
    built = [S.build(ep, toks) for ep in replay]
    b = S.collate(built, {x: toks[x].pad_token_id for x in SIDES})
    original = p._exchange
    joint = {}
    def observe(k, h, *a, **kw):
        out, gate = original(k, h, *a, **kw)
        joint[k] = cpu(out['quill']) - cpu(h['quill'])
        return out, gate
    p._exchange = observe
    try:
        with torch.no_grad():
            p.joint(b)
    finally:
        del p._exchange
    error = 0.
    for t in range(3):
        for k in range(3):
            want = np.stack([joint[k][i, e['quill']['labels'][t][0]] for i, e in enumerate(built)])
            error = max(error, float(np.max(np.abs(want - cap.values[t, k]))))
            np.testing.assert_allclose(want, cap.values[t, k], atol=2e-6, rtol=2e-5)
    # Real tiny MLX counterpart, including exact observer/no-observer identity.
    import mlx_pair as M
    import mlx_eval as V
    import mlx.core as mx
    mm = {}
    for x in SIDES:
        base = models[x].get_base_model()
        mm[x] = M.make_model(base.config.to_dict())
        weights = []
        for name, value in base.state_dict().items():
            if 'lora_' in name:
                continue
            name = name.replace('.base_layer.', '.')
            if name == 'lm_head.weight' and base.config.tie_word_embeddings:
                continue
            weights.append((name, M.from_torch(value)))
        mm[x].load_weights(weights)
    mp = M.Pair(mm, (1, 2, 4), heads=2, head_dim=16, checkpoint=False)
    for x in SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in p.bridge[x].state_dict().items()])
    mx.eval(mp.parameters())
    vp = V.Pair(mp, toks)
    plain = S.generate(vp, eps, max_turn=3, keep_tokens=True)
    cm = Capture(vp, 'quill', 3)
    with cm.installed():
        recorded = S.generate(vp, eps, max_turn=3, keep_tokens=True)
    assert [r['tokens'] for r in plain] == [r['tokens'] for r in recorded] == [r['tokens'] for r in baseline]
    for key in cap.values:
        np.testing.assert_allclose(cap.values[key], cm.values[key], atol=2e-6, rtol=2e-5)
    # Leakage audit: all receiver input strings match across target cities.
    for x in SIDES:
        for w in range(6):
            group = [r for r in design() if r['receiver'] == x and r['wording'] == w]
            assert len({episode(r)['system'][x] for r in group}) == 1
    print(json.dumps(dict(observer_tokens='identical', torch_mlx='passed', joint_max_error=error,
                          design_leakage='passed')), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--run', required=True)
    c.add_argument('--raw', type=Path, required=True)
    c.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    c.add_argument('--bs', type=int, default=4)
    c.add_argument('--tokens', type=int, default=3)
    c.add_argument('--cities', type=int)
    a = sub.add_parser('analyze')
    a.add_argument('--before', type=Path, default=SCRATCH / 'before')
    a.add_argument('--after', type=Path, default=SCRATCH / 'after')
    a.add_argument('--out', type=Path, default=ROOT / 'results/link_geometry.json')
    a.add_argument('--after-only', action='store_true', help='explicit partial analysis; no before comparison')
    sub.add_parser('selftest')
    args = ap.parse_args()
    if args.command == 'collect':
        if args.bs < 1 or args.tokens < 1 or (args.cities is not None and not 2 <= args.cities <= 20):
            ap.error('positive batch size and token count, and 2..20 cities are required')
        collect(args)
    elif args.command == 'analyze':
        analyze(args)
    else:
        selftest()


if __name__ == '__main__':
    main()
