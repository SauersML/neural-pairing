"""Per-head link reads and local write interventions at a generated city's first token.

No model operator is substituted: the observer calls the original Side.read and
computes attention separately. Interventions zero only the selected receiver's
write at the prediction position, with all earlier states/cache fixed. 'only'
means only this depth writes at that position, not an entire-chat depth ablation.
The null key is column zero. Token positions, ticks and exact emitted IDs survive.

Mac: MPD_MEM_GIB=12 /usr/bin/lockf -k ~/mpd-data/locks/macgpu.lock 
     /Users/user/mpd-data/venv/bin/python pairing/link_reads.py --run RUN 
     --inputs CHAT7.json CHAT11.json --regenerate --out results/link_reads.json
GPU confirmation: same CLI with --backend torch (CUDA required for real models).
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

SIDES = ('quill', 'ember')


def other(x):
    return SIDES[1 - SIDES.index(x)]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class ReadProbe:
    """Scoped read observer/intervention, for either genuine torch or MLX pairs."""
    def __init__(self, pair, receiver, position, condition='all', depth=None, capture=False):
        self.pair, self.receiver, self.position = pair, receiver, position
        self.condition, self.depth, self.capture = condition, depth, capture
        self.attention = {}
        self.write_samples = {}
        self.is_mlx = hasattr(pair, 'mlx')
        self.side = (pair.mlx.bridge if self.is_mlx else pair.bridge)[receiver]

    def weights(self, k, h, keys, allow, ages):
        s, i = self.side, self.position
        if self.is_mlx:
            import mlx.core as mx
            q = s.split(s.q[k](s.q_norm[k](h[:, i:i+1].astype(mx.float32))))
            nk = mx.broadcast_to(s.null[k][None, :, None, :], (h.shape[0], s.heads, 1, s.head_dim))
            kk = mx.concatenate([nk, keys], 2)
            bias = s.age[k][:, ages[:, i:i+1]].transpose(1, 0, 2, 3)
            bias = mx.concatenate([mx.zeros((h.shape[0], s.heads, 1, 1)), bias], -1)
            mask = mx.concatenate([mx.ones((h.shape[0], 1, 1), mx.bool_), allow[:, i:i+1]], -1)
            p = mx.softmax(q @ kk.transpose(0, 1, 3, 2) * s.head_dim**-.5 +
                           mx.where(mask[:, None], bias, -float('inf')), axis=-1)
            return p[0, :, 0].tolist()
        import torch
        q = s.split(s.q[k](s.q_norm[k](h[:, i:i+1].float())))
        nk = s.null[k][None, :, None, :].expand(h.shape[0], -1, 1, -1)
        kk = torch.cat([nk, keys], 2)
        bias = s.age[k][:, ages[:, i:i+1]].transpose(0, 1)
        bias = torch.cat([torch.zeros(h.shape[0], s.heads, 1, 1, device=h.device), bias], -1)
        mask = torch.cat([torch.ones(h.shape[0], 1, 1, dtype=torch.bool, device=h.device), allow[:, i:i+1]], -1)
        p = (q @ kk.transpose(-1, -2) * s.head_dim**-.5 + bias.masked_fill(~mask[:, None], -float('inf'))).softmax(-1)
        return p[0, :, 0].detach().cpu().tolist()

    def read(self, k, h, keys, vals, allow, ages, qmask, *args, **kwargs):
        w = self.original(k, h, keys, vals, allow, ages, qmask, *args, **kwargs)
        if self.capture:
            self.attention[str(self.pair.xblocks[k])] = self.weights(k, h, keys, allow, ages)
            v = w[0, self.position, :32]
            self.write_samples[str(self.pair.xblocks[k])] = v.tolist() if self.is_mlx else v.detach().cpu().tolist()
        drop = (self.condition == 'none' or
                self.condition == 'remove' and self.pair.xblocks[k] == self.depth or
                self.condition == 'only' and self.pair.xblocks[k] != self.depth)
        if drop:
            if self.is_mlx:
                import mlx.core as mx
                w = w * (mx.arange(w.shape[1]) != self.position)[None, :, None]
            else:
                w = w.clone()
                w[:, self.position] = 0
        return w

    @contextlib.contextmanager
    def installed(self):
        self.original = self.side.read
        present = 'read' in self.side.__dict__
        previous = self.side.__dict__.get('read')
        # MLX's Module dictionaries contain parameter arrays; this is a method,
        # never a parameter or a persistent model mutation.
        object.__setattr__(self.side, 'read', self.read)
        try:
            yield self
        finally:
            if present:
                object.__setattr__(self.side, 'read', previous)
            else:
                object.__delattr__(self.side, 'read')


def mentions(text, city):
    return list(re.finditer(r'(?<!\w)' + re.escape(city) + r'(?!\w)', text, re.I))


def targets(ep, result, toks):
    """One first mention per receiver, including text-exposed cases (marked)."""
    import session as S
    from play_live import token_pieces
    built = S.build(S.replay(ep, result), toks)
    found = []
    for x in SIDES:
        city = ep['meta']['secrets'][other(x)]['city']
        if mentions(ep['system'][x], city):
            raise ValueError('partner city is present in receiver prompt')
        for turn, (speaker, text) in enumerate(result['turns']):
            if speaker != x or not mentions(text, city):
                continue
            lo, hi = mentions(text, city)[0].span()
            pieces = token_pieces(toks[x], result['tokens'][turn][:-1], text)
            at = 0
            for j, piece in enumerate(pieces):
                if at < hi and at + len(piece) > lo:
                    break
                at += len(piece)
            else:
                raise ValueError('city does not overlap any emitted token')
            _, start, end = next(v for v in built[x]['spans'] if v[0] == turn)
            position, target, _ = built[x]['labels'][start + j]
            assert target == result['tokens'][turn][j]
            found.append(dict(receiver=x, sender=other(x), city=city, turn=turn,
                              text=text, prefix=text[:lo], token_index=j,
                              target_id=target, target_piece=pieces[j], position=position,
                              tick=built[x]['time'][position],
                              text_exposed=any(mentions(t, city) for _, t in result['turns'][:turn])))
            break
    return built, found


def stream_to_target(pair, built, target):
    """Replay exact input chunks; return cache before the prediction tick."""
    import torch
    from link import DEVICE
    cache, count = pair.new_cache(), {x: 0 for x in SIDES}
    tick_end = target['tick']
    for tick in sorted({t for x in SIDES for t in built[x]['time'] if t <= tick_end}):
        chunk, qm = {}, {}
        for x in SIDES:
            ix = [i for i, t in enumerate(built[x]['time']) if t == tick]
            if not ix:
                continue
            assert ix == list(range(count[x], count[x] + len(ix)))
            ids = torch.tensor([[built[x]['ids'][i] for i in ix]], device=DEVICE)
            att = torch.ones(1, count[x] + len(ix), dtype=torch.long, device=DEVICE)
            pos = torch.arange(count[x], count[x] + len(ix), device=DEVICE)[None]
            chunk[x], qm[x] = (ids, att, pos), torch.ones_like(ids, dtype=torch.float)
            count[x] += len(ix)
        if tick == tick_end:
            local = target['position'] - (count[target['receiver']] - chunk[target['receiver']][0].shape[1])
            return cache, chunk, qm, local, count
        pair.step(chunk, cache, qm, tick)
    raise ValueError('target tick absent')


def logprob(pair, h, x, position, token):
    import torch
    idx = torch.tensor([position], device=h[x].device)
    return float(pair.token_logits(x, h[x], idx).float().log_softmax(-1)[0, token])


def measure(pair, built, target):
    import torch
    x, y = target['receiver'], target['sender']
    with torch.inference_mode():
        cache, chunk, qm, local, count = stream_to_target(pair, built, target)
        def forward(condition='all', depth=None, capture=False):
            with ReadProbe(pair, x, local, condition, depth, capture).installed() as obs:
                h, _ = pair.step(chunk, copy.deepcopy(cache), qm, target['tick'])
                lp = logprob(pair, h, x, local, target['target_id'])
            return lp, obs.attention, obs.write_samples
        # An untouched original pass proves observer neutrality on each real case.
        h, _ = pair.step(chunk, copy.deepcopy(cache), qm, target['tick'])
        untouched = logprob(pair, h, x, local, target['target_id'])
        logits = pair.token_logits(x, h[x], torch.tensor([local], device=h[x].device))[0].float()
        argmax = int(logits.argmax())
        full, attention, write_samples = forward(capture=True)
        assert full == untouched, ('observer changed output', full, untouched)
        none, _, _ = forward('none')
        depth_results = []
        for depth in pair.xblocks:
            removed, _, _ = forward('remove', depth)
            only, _, _ = forward('only', depth)
            depth_results.append(dict(depth=depth, remove_logprob=removed, only_logprob=only,
                                      removal_effect=full-removed, only_effect=only-none))
            print(json.dumps(dict(city=target['city'], receiver=x, **depth_results[-1])), flush=True)
        tokens = built[y]['ids'][:count[y]]
        decoded = pair.toks[y].decode(tokens)
        from play_live import token_pieces
        pieces = token_pieces(pair.toks[y], tokens, decoded)
        spans = [m.span() for m in mentions(decoded, target['city'])]
        positions, at = [], 0
        for j, piece in enumerate(pieces):
            if built[y]['time'][j] < 0 and any(at < hi and at+len(piece) > lo for lo, hi in spans):
                positions.append(j)
            at += len(piece)
        if not positions:
            raise ValueError('private city tokens not found in sender prompt')
        for depth, heads in attention.items():
            assert all(abs(sum(h)-1) < 1e-5 and len(h) == count[y]+1 for h in heads)
        return dict(target, logprob=full, no_local_writes_logprob=none,
                    greedy_id=argmax, greedy_piece=pair.toks[x].decode([argmax]),
                    observer_logprob_error=full-untouched, depths=depth_results,
                    write_samples=write_samples,
                    partner_ids=tokens, partner_pieces=pieces,
                    partner_ticks=built[y]['time'][:count[y]],
                    private_city_positions=positions, attention=attention,
                    city_attention={d: [sum(h[j+1] for j in positions) for h in heads]
                                    for d, heads in attention.items()})


def self_test():
    """Actual tiny Qwen/Arcee, torch/MLX weights and writes; no fake operators."""
    os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1')
    import torch
    import mlx.core as mx
    import mlx_pair as M
    import mlx_eval as V
    import link
    import session as S
    torch.set_num_threads(2)
    torch.manual_seed(19)
    models, mm, toks = {}, {}, {}
    for x in SIDES:
        toks[x], models[x] = link.load(x)
        models[x].eval()
        mm[x] = M.make_model(models[x].config.to_dict())
        mm[x].load_weights([(k, M.from_torch(v)) for k, v in models[x].state_dict().items()
                            if k != 'lm_head.weight' or not mm[x].args.tie_word_embeddings])
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    with torch.no_grad():
        for s in p.bridge.values():
            s.alpha.fill_(.4)
            s.age.normal_(0, .3)
            s.null.normal_(0, .2)
    mp = M.Pair(mm, (1, 2, 4), heads=2, head_dim=16, checkpoint=False)
    for x in SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in p.bridge[x].state_dict().items()])
    vp = V.Pair(mp, toks)
    ep = dict(mode='dialogue', system={'quill': 'Today I am in Dallas.', 'ember': 'Today I am in Beijing.'},
              task={x: None for x in SIDES}, turns=[('quill', 'So, Beijing?', 1.)],
              meta={'secrets': {'quill': {'city': 'Dallas'}, 'ember': {'city': 'Beijing'}}})
    out = S.generate(p, [ep], force=True, keep_tokens=True)[0]
    built, ts = targets(ep, out, toks)
    t = ts[0]
    batch = S.collate([built], {x: toks[x].pad_token_id for x in SIDES})
    with torch.inference_mode():
        h, _ = p.joint(batch)
        with ReadProbe(p, 'quill', t['position'], capture=True).installed() as obs:
            observed, _ = p.joint(batch)
        assert torch.equal(h['quill'], observed['quill'])
        p.record = {}
        p.joint(batch)
        for d, heads in obs.attention.items():
            torch.testing.assert_close(torch.tensor(heads), p.record['quill', int(d)][0]['attn'][0, :, t['position']])
        p.record = None
        a, b = measure(p, built, t), measure(vp, built, t)
        assert abs(a['logprob']-b['logprob']) < 2e-5
        assert abs(a['logprob']-logprob(p, h, 'quill', t['position'], t['target_id'])) < 2e-5
        for da, db in zip(a['depths'], b['depths']):
            for key in ('remove_logprob', 'only_logprob'):
                assert abs(da[key]-db[key]) < 2e-5
        for d in a['attention']:
            torch.testing.assert_close(torch.tensor(a['attention'][d]), torch.tensor(b['attention'][d]), atol=2e-6, rtol=2e-5)
        # Independently mute only that query using qmask, rather than the observer.
        cache, chunk, qm, local, _ = stream_to_target(p, built, t)
        qm['quill'][0, local] = 0
        h, _ = p.step(chunk, cache, qm, t['tick'])
        assert abs(logprob(p, h, 'quill', local, t['target_id'])-a['no_local_writes_logprob']) < 1e-6
    assert 'read' not in p.bridge['quill'].__dict__ and 'read' not in mp.bridge['quill'].__dict__
    print('PASS: observer bitwise neutrality; exact head weights; cached/joint prediction; torch/MLX interventions; independent query mute', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run')
    ap.add_argument('--inputs', nargs='+')
    ap.add_argument('--from-results', type=Path, help='replay exact saved chats from a previous link_reads result')
    ap.add_argument('--out', default='results/link_reads.json')
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--regenerate', action='store_true')
    ap.add_argument('--max-turn', type=int, default=64)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run or bool(args.inputs) == bool(args.from_results):
        ap.error('--run and exactly one of --inputs / --from-results are required')
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(64 * 2**20)
    else:
        import torch
        if not os.environ.get('NL_TINY') and not torch.cuda.is_available():
            ap.error('real torch models require a CUDA GPU; Mac uses 4-bit MLX only')
        from evaluate import load_pair
    import torch
    import session as S
    torch.set_num_threads(2)
    pair, toks = load_pair(args.run)
    result = dict(checkpoint=str(Path(args.run).resolve()), backend=args.backend,
                  numerics='4-bit MLX bases, float32 link' if args.backend == 'mlx' else 'bf16 torch bases, float32 link',
                  method='Prediction-position writes only; preceding cached history fixed; null at column zero; heads separate.',
                  source_hashes={f: file_hash(Path(__file__).with_name(f)) for f in
                                 ('link_reads.py', 'link.py', 'mlx_pair.py', 'mlx_eval.py', 'session.py')},
                  chats=[], records=[])
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = outpath.with_suffix('.part.json')
        temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temp.replace(outpath)
    if args.from_results:
        prior = json.loads(args.from_results.read_text())
        sources = [(str(args.from_results), dict(episode=c['episode'], raw=[c['raw']],
                                                checkpoint=c['generation_checkpoint'])) for c in prior['chats']]
    else:
        sources = [(path, json.loads(Path(path).read_text())) for path in args.inputs]
    for path, source in sources:
        ep = source['episode']
        if args.regenerate:
            with torch.inference_mode():
                raw = S.generate(pair, [ep], max_turn=args.max_turn, temperature=0., keep_tokens=True)[0]
            generation_checkpoint = str(Path(args.run).resolve())
        else:
            raw = source['raw'][0]
            generation_checkpoint = source['checkpoint']
        result['chats'].append(dict(source=path, source_sha256=file_hash(path), episode=ep, raw=raw,
                                    generation_checkpoint=generation_checkpoint))
        print(json.dumps(dict(source=path, turns=raw['turns'])), flush=True)
        save()
        built, ts = targets(ep, raw, toks)
        for t in ts:
            record = measure(pair, built, t)
            if args.regenerate and record['greedy_id'] != record['target_id']:
                raise AssertionError('replayed city prediction differs from actual greedy generation')
            record['chat_index'] = len(result['chats'])-1
            result['records'].append(record)
            save()
    result['complete'] = True
    save()
    print('Wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
