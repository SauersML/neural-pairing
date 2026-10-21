"""4-bit MLX evaluation with the original session, scoring, and CLI schemas.

MPD_MEM_GIB=12 python pairing/mlx_eval.py evalfree --run RUN --replies FILE --out_dir OUT
The CPU tensors below are an interface to session.py, never base-model weights.
Self-attention and sender K/V are cached independently at every layer/exchange.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('MPD_MEM_GIB', '12')
os.environ['NL_DEVICE'] = 'cpu'
import mlx_pair as M
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
import numpy as np
import torch

SIDES = M.SIDES


def tensor(a):
    """Preserve bf16 across the small CPU interface (numpy has no bf16)."""
    mx.eval(a)
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.view(mx.uint16)).copy()).view(torch.bfloat16)
    return torch.from_numpy(np.array(a).copy())


def array(t):
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.int64:
            t = t.to(torch.int32)
        return M.from_torch(t)
    return t


def block(layer, h, att, pos, cache=None):
    """MLX decoder block with per-row RoPE positions, including padding gaps.

    mlx_lm's ordinary cache offset counts padding slots. session.generate uses
    actual token counts instead, so apply the supplied positions before caching.
    Positions within each chunk are consecutive at all valid tokens; left-pad
    positions may be negative since their keys are masked and never read.
    """
    a = layer.self_attn
    z = layer.input_layernorm(h)
    B, L, _ = z.shape
    q = a.q_proj(z).reshape(B, L, a.n_heads, -1)
    k = a.k_proj(z).reshape(B, L, a.n_kv_heads, -1)
    v = a.v_proj(z).reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if hasattr(a, 'q_norm'):
        q, k = a.q_norm(q), a.k_norm(k)
    offset = pos[:, -1] - (L - 1)
    q = a.rope(q.transpose(0, 2, 1, 3), offset=offset)
    k = a.rope(k.transpose(0, 2, 1, 3), offset=offset)
    past = 0 if cache is None else cache.offset
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    query = past + mx.arange(L)
    key = mx.arange(k.shape[2])
    mask = (key[None, :] <= query[:, None])[None, None] & att[:, None, None, :].astype(mx.bool_)
    if getattr(layer, 'use_sliding', False):
        mask = mask & (key[None, :] > query[:, None] - layer.args.sliding_window)[None, None]
    z = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
    h = h + a.o_proj(z.transpose(0, 2, 1, 3).reshape(B, L, -1))
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    mx.eval(h, k, v)
    return h


class Pair:
    """Torch evaluator protocol over an MLX Pair; only small results cross CPU.

    This lets session.generate, evaluate.run_episodes, evalfree's donor matching
    and statistics, and the transcript/readmap serializers run unchanged.
    """
    def __init__(self, pair, toks):
        self.mlx = pair
        self.toks = toks
        self.xblocks = pair.xblocks
        self.record = None
        self.models = {x: SimpleNamespace(model=None, lm_head=lambda h, x=x: self.token_logits(x, h))
                       for x in SIDES}

    def token_logits(self, x, h, idx=None):
        return tensor(self.mlx.token_logits(x, array(h), None if idx is None else array(idx)))

    def new_cache(self):
        return dict(kv={x: [KVCache() for _ in self.mlx.models[x].layers] for x in SIDES},
                    shown={x: dict(valid=None, tick=None, blocks={}) for x in SIDES})

    def _exchange(self, k, h, qmask, ticks, valid, mode, cache=None):
        shown = {x: self.mlx.bridge[x].show(k, h[x]) for x in h}
        if cache is not None:
            for x, (key, val, _) in shown.items():
                c = cache[x]['blocks'].setdefault(k, KVCache())
                c.update_and_fetch(key, val)
        out = dict(h)
        for x in h:
            y = M.other(x)
            s = self.mlx.bridge[x]
            if cache is None:
                keys, vals = shown[y][:2]
            else:
                if k not in cache[y]['blocks']:
                    continue
                keys, vals = cache[y]['blocks'][k].keys_and_values()
            ty, vy = ticks[y], valid[y]
            if mode == 'shuffle':
                shift = 1 if x == SIDES[0] else -1
                keys, vals, ty, vy = [mx.roll(a, shift, axis=0) for a in (keys, vals, ty, vy)]
            tq = ticks[x] if cache is None else ticks[x][:, -h[x].shape[1]:]
            allow = (ty[:, None, :] <= tq[:, :, None]) & vy[:, None, :].astype(mx.bool_)
            ages = M.age_buckets(tq, ty)
            if self.record is None:
                w = s.read(k, h[x], keys, vals, allow, ages, qmask[x])
            else:
                B, L = h[x].shape[:2]
                q = s.split(s.q[k](s.q_norm[k](h[x].astype(mx.float32))))
                nk = mx.broadcast_to(s.null[k][None, :, None, :], (B, s.heads, 1, s.head_dim))
                kk, vv = mx.concatenate([nk, keys], 2), mx.concatenate([mx.zeros_like(nk), vals], 2)
                bias = s.age[k][:, ages].transpose(1, 0, 2, 3)
                bias = mx.concatenate([mx.zeros((B, s.heads, L, 1)), bias], -1)
                mask = mx.concatenate([mx.ones((B, L, 1), mx.bool_), allow], -1)
                p = mx.softmax((q @ kk.transpose(0, 1, 3, 2)) * s.head_dim ** -.5 +
                               mx.where(mask[:, None], bias, -float('inf')), axis=-1)
                z = (p @ vv).transpose(0, 2, 1, 3).reshape(B, L, -1)
                w = mx.tanh(s.alpha[k]) * s.size[k] * s.o[k](z) * qmask[x][..., None]
                # readmap consumes head means. Store only that mean on CPU so
                # recording long prompts does not retain H copies of the map.
                self.record.setdefault((x, self.xblocks[k]), []).append(
                    dict(attn=tensor(mx.mean(p, axis=1, keepdims=True)), gate=tensor(shown[x][2])))
            out[x] = h[x] + w.astype(h[x].dtype)
        mx.eval(out, shown)
        return out, {x: shown[x][2] for x in h}

    def _forward(self, ids, att, pos, ticks, qmask, mode, cache=None):
        if mode not in (None, 'zero', 'shuffle'):
            raise ValueError(f'unsupported MLX evaluation mode: {mode}')
        if mode == 'shuffle' and next(iter(ids.values())).shape[0] < 2:
            raise ValueError('shuffle needs at least two episodes')
        h = {x: self.mlx.models[x].model.embed_tokens(ids[x]) for x in ids}
        gates = {x: [] for x in ids}
        valid = att
        if cache is not None:
            for x in ids:
                sh = cache['shown'][x]
                v = att[x][:, -ids[x].shape[1]:]
                sh['valid'] = v if sh['valid'] is None else mx.concatenate([sh['valid'], v], 1)
                sh['tick'] = ticks[x] if sh['tick'] is None else mx.concatenate([sh['tick'], ticks[x]], 1)
            ticks = {x: cache['shown'][x]['tick'] for x in SIDES}
            valid = {x: cache['shown'][x]['valid'] for x in SIDES}
        for l in range(max(len(m.layers) for m in self.mlx.models.values())):
            for x in ids:
                if l < len(self.mlx.models[x].layers):
                    h[x] = block(self.mlx.models[x].layers[l], h[x], att[x], pos[x],
                                 None if cache is None else cache['kv'][x][l])
            if l in self.xblocks and mode != 'zero':
                h, g = self._exchange(self.xblocks.index(l), h, qmask, ticks, valid, mode,
                                      None if cache is None else cache['shown'])
                for x in ids:
                    gates[x].append(g[x])
        h = {x: self.mlx.models[x].model.norm(h[x]) for x in ids}
        gates = {x: mx.stack(gates[x], -1) if gates[x] else mx.zeros((*ids[x].shape, len(self.xblocks))) for x in ids}
        return {x: tensor(v) for x, v in h.items()}, {x: tensor(v) for x, v in gates.items()}

    def step(self, chunk, cache, qmask, tick, mode=None, gen=None, write_mode=None):
        if write_mode is not None:
            raise ValueError('random writes are not part of MLX link/cut/shuffle evaluation')
        ids = {x: array(v[0]) for x, v in chunk.items()}
        return self._forward(ids, {x: array(v[1]) for x, v in chunk.items()},
                             {x: array(v[2]) for x, v in chunk.items()},
                             {x: mx.full(v.shape, tick, mx.int32) for x, v in ids.items()},
                             {x: array(v) for x, v in qmask.items()}, mode, cache)

    def joint(self, batch, mode=None, gen=None):
        b = M.convert_batch(batch)
        return self._forward({x: b[x]['ids'] for x in SIDES}, {x: b[x]['att'] for x in SIDES},
                             {x: b[x]['pos'] for x in SIDES}, {x: b[x]['tick'] for x in SIDES},
                             {x: (b[x]['att'].astype(mx.bool_) & (b[x]['tick'] < b[x]['cut'][:, None])).astype(mx.float32)
                              for x in SIDES}, mode)


def load_pair(run):
    """Only mlx_pair.load_base may load real bases; torch reads bridge PEFT only."""
    from transformers import AutoTokenizer
    ck = torch.load(Path(run) / 'bridge.pt', map_location='cpu', weights_only=False)
    models, toks = {}, {}
    for x in SIDES:
        cfg = json.loads((Path(run) / x / 'adapter_config.json').read_text())
        hf = cfg.get('base_model_name_or_path') or M.HF[x]
        if os.environ.get('NL_TINY'):
            # Reproduce link.load's random NL_TINY bases, never HF base weights.
            # This is also the no-mock checkpoint/CLI integration test path.
            from link import load
            toks[x], tiny = load(x)
            models[x] = M.make_model(tiny.config.to_dict())
            models[x].load_weights([(k, M.from_torch(v)) for k, v in tiny.state_dict().items()
                                    if k != 'lm_head.weight' or not tiny.config.tie_word_embeddings])
            del tiny
        else:
            models[x], _ = M.load_base(hf)
            toks[x] = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        M.add_lora(models[x], cfg['r'])
        mx.eval(models[x].parameters())
        print(json.dumps(dict(loaded=x, active_gib=mx.get_active_memory() / 2**30)), flush=True)
    mp = M.Pair(models, ck['xblocks'], ck['sizes'], ck['heads'], ck['head_dim'],
                ck.get('gate_norm', False), checkpoint=False)
    del ck
    M.load_checkpoint(mp, run)
    return Pair(mp, toks), toks


def payload_table(pair, toks, eps, bs=16, conds=None):
    """Same payload records, projecting only payload positions in small chunks."""
    import payload_ctl as P
    import session as S
    conds = P.CONDS if conds is None else conds
    out = []
    for a, z in S.batches(len(eps), bs):
        built = [S.build(ep, toks) for ep in eps[a:z]]
        b = S.collate(built, {x: toks[x].pad_token_id for x in SIDES})
        records = []
        for x in SIDES:
            off = 0
            for i, (ep, e) in enumerate(zip(eps[a:z], built)):
                js = S.payload_labels(ep, toks).get(x, [])
                if js:
                    records.append((x, [off + j for j in js], dict(game=ep['game'], episode=a + i,
                                                                receiver=x, n_tokens=len(js))))
                off += len(e[x]['labels'])
        for name, mode in conds:
            h, _ = pair.joint(b, mode)
            for x, js, rec in records:
                ce = []
                for start in range(0, len(js), 16):
                    ix = torch.tensor(js[start:start + 16])
                    logits = pair.mlx.token_logits(x, array(h[x]), array(b[x]['idx'][ix]))
                    ce.extend(nn.losses.cross_entropy(logits, array(b[x]['lab'][ix]), reduction='none').tolist())
                rec[name + '_ce'] = sum(ce) / len(ce)
        out.extend(rec for _, _, rec in records)
    return out


@contextlib.contextmanager
def backend():
    """Scoped dependency injection: original CLI and scoring source is unchanged."""
    import evaluate
    import payload_ctl
    import play_tasks
    old_load, old_table = evaluate.load_pair, payload_ctl.table
    processes = play_tasks.subprocess

    def code_process(*args, **kwargs):
        # These short assertion programs must not inherit a model-sized lease
        # and time out in Python startup before executing the actual test.
        env = dict(os.environ if kwargs.get('env') is None else kwargs['env'])
        env['MPD_MEM_GIB'] = '1'
        env.pop('MEM_LEASE_GIB', None)
        return processes.run(*args, **dict(kwargs, env=env))

    play_tasks.subprocess = SimpleNamespace(run=code_process, TimeoutExpired=processes.TimeoutExpired)
    evaluate.load_pair, payload_ctl.table = load_pair, payload_table
    try:
        yield
    finally:
        evaluate.load_pair, payload_ctl.table = old_load, old_table
        play_tasks.subprocess = processes



def payload_main():
    import payload_ctl as P
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--max_digits', type=int, default=16)
    ap.add_argument('--coread', action='store_true')
    args = ap.parse_args()
    pair, toks = load_pair(args.run)
    eps = P.episodes(json.loads(Path(args.replies).read_text()), args.n, args.max_digits, args.coread)
    rec = []
    for g in eps:
        part = payload_table(pair, toks, eps[g], args.bs)
        rec += part
        print(g, json.dumps(P.summary(part)), flush=True)
    Path(args.out).write_text(json.dumps(rec))
    print('wrote', args.out, len(rec), 'records', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument('command', choices=('evalfree', 'transcripts', 'readmap', 'payload'))
    ap.add_argument('--memory_gib', type=float, default=10.5)
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        ap.print_help()
        return
    args, rest = ap.parse_known_args()
    if not 0 < args.memory_gib <= 10.5 or float(os.environ['MPD_MEM_GIB']) > 12:
        ap.error('memory_gib must be in (0,10.5], MPD_MEM_GIB <= 12')
    mx.set_memory_limit(int(args.memory_gib * 2**30))
    mx.set_cache_limit(32 * 2**20)
    torch.set_num_threads(2)
    locations = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    locations.add_argument('--out_dir')
    locations.add_argument('--coread_data')
    loc, _ = locations.parse_known_args(rest)
    if args.command == 'transcripts':
        path = Path(loc.coread_data) if loc.coread_data else Path(__file__).resolve().parents[1] / 'data/coread.json'
        if not path.with_name('talk.json').exists():
            print('Skipping co_talk: talk.json is absent next to the co-reading data.', flush=True)
        if '--device' not in rest:
            rest += ['--device', 'cpu']
    module = 'payload_ctl' if args.command == 'payload' else args.command
    sys.argv = [module + '.py', *rest]
    started = time.monotonic()
    with backend():
        if args.command == 'payload':
            payload_main()
        else:
            runpy.run_module(module, run_name='__main__')
    if args.command == 'evalfree' and '--summarize' not in rest:
        out = Path(loc.out_dir)
        manifest = json.loads((out / 'manifest.json').read_text())
        import evalfree
        manifest['device'] = 'mlx'
        for name in ('mlx_eval.py', 'mlx_pair.py'):
            manifest['code_hashes'][name] = evalfree.file_hash(Path(__file__).with_name(name))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import resource
    print(json.dumps(dict(wall_seconds=time.monotonic() - started,
                          mlx_peak_gib=mx.get_peak_memory() / 2**30,
                          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)), flush=True)


if __name__ == '__main__':
    main()
