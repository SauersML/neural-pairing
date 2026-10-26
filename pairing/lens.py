"""Per-token effects of the link in teacher-forced episodes, at every position of both models.

For each position: the tokens whose log probability the link raises and lowers most (nats), KL(p_control || p_link),
at a labelled position the target's change and ranks, and the partner positions it attends to most at each exchange
block and over all of them. Three contrasts of the linked pass against a control:
  total   the link cut (Pair.joint mode 'zero').
  tick    the incoming writes at one receiving tick removed at every exchange block (--direction: for one receiver
          or both); later reads and both models' states are recomputed, and the sender's keys stay. By default the
          ticks are the silent ticks and the first writing tick of each turn.
  block   at each exchange block, the receiver's residual stream without that block's write, run through its remaining
          blocks with no later exchanges (earlier exchanges stay in it); the run with the write from the last block
          must equal the full linked pass.
At the payload positions the write's own logit lens (as analyze_bridge) is compared with the block contrast's ranks.
Attention weights describe what is read; they are not a causal attribution to source tokens. Each tick contrast runs
in one batch with an unablated control sequence, so batch shape cannot change the numbers.

Writes OUT_DIR/GAME_NNN.json and .txt per episode, and OUT_DIR/manifest.json.
Usage (from pairing/):
  python lens.py --run RUN/step_N --replies REPLIES.json --games relay_word,co_play \
      --coread_data ../data/coread.json --n 1 --out_dir RUN/lens --ablation_bs 4
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch

import session as S
from link import SIDES, Run, inner, partner


@contextmanager
def intervention(pair, batch, ticks=None, direction='both', capture=None):
    """While inside, Pair.joint on batch removes incoming writes: ticks is one tick per sequence (None: none removed)
    or 'all' (every write), for the receiver direction ('both', 'quill' or 'ember'). capture, a dict, receives per
    (side, block) the residual stream before and after the exchange, the write and the attention weights. The Pair is
    restored on exit; pair.record must be None."""
    if direction not in ('both', *SIDES):
        raise ValueError('direction names the receiver: both, quill or ember')
    if pair.record is not None:
        raise ValueError('lens needs pair.record=None to preserve fused inference')
    n = batch[SIDES[0]]['ids'].shape[0]
    if ticks is not None and ticks != 'all' and len(ticks) != n:
        raise ValueError('ticks needs one ablation tick per sequence')
    exchange = pair._exchange
    saved_exchange = pair.__dict__.get('_exchange')
    saved_reads = {x: pair.bridge[x].__dict__.get('read') for x in SIDES}
    reads = {x: pair.bridge[x].read for x in SIDES}

    def read_for(x):
        def read(k, h, keys, vals, allow, ages, qmask, rec=None):
            w = reads[x](k, h, keys, vals, allow, ages, qmask, rec)
            # The write stays the fused read's; a second read with a record gives the attention weights.
            diagnostic = {}
            reads[x](k, h, keys, vals, allow, ages, qmask, diagnostic)
            capture[(x, pair.xblocks[k])] = dict(
                before=h.detach().cpu().clone(), write=w.detach().cpu(),
                attn=diagnostic['attn'], alpha=diagnostic['alpha'])
            return w
        return read

    def wrapped(k, h, qmask, mode, gen, *args, **kwargs):
        masks = dict(qmask)
        if ticks is not None:
            for x in SIDES:
                if direction not in ('both', x):
                    continue
                if ticks == 'all':
                    masks[x] = torch.zeros_like(qmask[x])
                else:
                    remove = torch.zeros_like(qmask[x], dtype=torch.bool)
                    for row, tick in enumerate(ticks):
                        if tick is not None:
                            remove[row] = batch[x]['tick'][row] == tick
                    masks[x] = qmask[x].masked_fill(remove, 0)
        result, gates = exchange(k, h, masks, mode, gen, *args, **kwargs)
        if capture is not None:
            for x in SIDES:
                capture[(x, pair.xblocks[k])]['after'] = result[x].detach().cpu().clone()
        return result, gates

    try:
        pair._exchange = wrapped
        if capture is not None:
            for x in SIDES:
                pair.bridge[x].read = read_for(x)
        yield
    finally:
        if saved_exchange is None:
            del pair._exchange
        else:
            pair._exchange = saved_exchange
        if capture is not None:
            for x in SIDES:
                if saved_reads[x] is None:
                    del pair.bridge[x].read
                else:
                    pair.bridge[x].read = saved_reads[x]


@torch.no_grad()
def joint(pair, batch, ticks=None, direction='both', capture=None):
    """The linked final states of batch under intervention(pair, batch, ticks, direction, capture)."""
    with intervention(pair, batch, ticks, direction, capture):
        return pair.joint(batch)[0]


@torch.no_grad()
def continue_block(pair, batch, side, block, residual):
    """side's final normed states after running residual, its state after block, through the remaining blocks with
    no exchanges."""
    b = batch[side]
    run = Run(pair.models[side], b['ids'], b['att'], b['pos'])
    run.h = residual.to(run.h.device, run.h.dtype)
    for depth in range(block + 1, pair.depth[side]):
        run.block(depth)
    return run.last()


def top_changes(delta, tok, k=5, positive=True):
    """Per position, the k tokens with the largest positive (or, positive False, negative) delta."""
    values, ids = (delta if positive else -delta).topk(min(k, delta.shape[-1]), dim=-1)
    return [[dict(id=int(i), text=tok.decode([int(i)]), nats=float(v if positive else -v))
             for v, i in zip(vs, ix) if v > 0] for vs, ix in zip(values.cpu(), ids.cpu())]


@torch.no_grad()
def contrast(pair, side, on, off, head_chunk=32, targets=None):
    """Per position of the final states on and off [positions, d]: the tokens most raised and lowered (log p on minus
    log p off), KL(p_off || p_on) and, at positions in targets ({position: token id}), the target's change and ranks.
    Float64 keeps KL nonnegative for tiny writes."""
    if head_chunk < 1:
        raise ValueError('head_chunk must be positive')
    _, head = inner(pair.models[side])
    targets = targets or {}
    out = []
    for a in range(0, len(on), head_chunk):
        z = min(len(on), a + head_chunk)
        lp = head(on[a:z]).double().log_softmax(-1)
        lq = head(off[a:z]).double().log_softmax(-1)
        d = lp - lq
        up, down = top_changes(d, pair.toks[side]), top_changes(d, pair.toks[side], positive=False)
        kl = (lq.exp() * (-d)).sum(-1).clamp_min(0).cpu().tolist()
        for j in range(z - a):
            row = dict(promoted=up[j], suppressed=down[j], kl_cut_to_link=kl[j])
            if a + j in targets:
                target = targets[a + j]
                row.update(target_delta_nats=float(d[j, target]),
                           target_promoted_rank=int((d[j] > d[j, target]).sum()),
                           target_decode_rank=int((lp[j] > lp[j, target]).sum()))
            out.append(row)
    return out


def tick_metadata(ep, built):
    """({turn: its first writing tick}, the set of silent ticks before each turn as session.build adds them). In the
    pen game every word span is a turn."""
    turns, silent = {}, set()
    for x in SIDES:
        b = built[x]
        for turn, start, end in b['spans']:
            if start == end:
                continue
            first = b['time'][b['labels'][start][0]]
            turns[turn] = first
            count = S.pause_count(ep, turn) if ep['mode'] == 'dialogue' else \
                S.pause_count(ep) if ep['mode'] == 'duplex' else 0
            silent.update(range(first - count, first))
    return turns, silent


def sender_view(pair, side, built, records, topk=3):
    """Per position of side: the null key's mass and the top partner positions by attention, averaged over heads, at
    each block (also per head) and averaged over blocks."""
    y = partner(side)
    by_block, means = {}, []

    def items(p):
        values, indices = p[1:].topk(min(topk, len(p) - 1))
        return [dict(position=int(i), tick=built[y]['time'][i], token_id=built[y]['ids'][i],
                     text=pair.toks[y].decode([built[y]['ids'][i]]), mass=float(v))
                for v, i in zip(values, indices) if v > 0]

    for block in pair.xblocks:
        p = records[(side, block)].pop('attn')[0]  # frees the full attention once summarized
        m = p.mean(0)
        means.append(m)
        by_block[str(block)] = [dict(alpha=records[(side, block)]['alpha'],
                                      null_mass=float(m[pos, 0]), top=items(m[pos]),
                                      heads=[dict(null_mass=float(h[pos, 0]), top=items(h[pos])) for h in p])
                                 for pos in range(len(built[side]['ids']))]
    mean = torch.stack(means).mean(0)
    return [dict(partner=y, enabled=(built[side]['cut'] is None or
                                    built[side]['time'][p] < built[side]['cut']),
                 null_mass=float(mean[p, 0]), top=items(mean[p]),
                 blocks={str(b): by_block[str(b)][p] for b in pair.xblocks}) for p in range(len(mean))]


@torch.no_grad()
def analyze_episode(pair, ep, ticks=None, direction='both', ablation_bs=4, head_chunk=32,
                    top_reads=3, max_positions=2048):
    """The report of one episode (the module docstring's contrasts); ticks None selects the silent ticks and turn
    starts."""
    if ablation_bs < 1 or top_reads < 1:
        raise ValueError('ablation_bs and top_reads must be positive')
    pair.eval()
    # Pair.models is a plain dict, not a registered ModuleDict.
    for model in pair.models.values():
        model.eval()
    start = time.perf_counter()
    built = S.build(ep, pair.toks)
    longest = max(len(built[x]['ids']) for x in SIDES)
    if max_positions and longest > max_positions:
        raise ValueError(f'episode has {longest} positions; raise --max_positions explicitly (no truncation)')
    pads = {x: pair.toks[x].pad_token_id for x in SIDES}
    batch = S.collate([built], pads)
    device = batch[SIDES[0]]['ids'].device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    turns, silent = tick_metadata(ep, built)
    boundaries = {turn: tick - (S.pause_count(ep, turn) if ep['mode'] == 'dialogue' else
                                S.pause_count(ep) if ep['mode'] == 'duplex' else 0)
                  for turn, tick in turns.items()}
    selected = sorted(silent | set(turns.values())) if ticks is None else sorted(set(ticks))
    records = {}
    linked = joint(pair, batch, capture=records)
    cut = pair.joint(batch, mode='zero')[0]
    payload = S.payload_labels(ep, pair.toks)
    out = dict(schema_version=1, game=ep['game'], episode=ep, direction=direction,
               selected_ticks=selected, turn_start_ticks=turns, silent_ticks=sorted(silent),
               semantics=dict(delta='log p(link) - log p(control), nats',
                              kl='KL(control || link); control=cut or tick-ablated or pre-write continuation',
                              tick='incoming writes at receiver tick, all exchange blocks; later states recomputed',
                              block='pre/post exchange continuation, earlier writes retained, later exchanges disabled',
                              text='teacher forced, including dialogue listener text; no free-running text effect',
                              ranks='zero based; direct=norm(write)+head as analyze_bridge; downstream=with-write logits',
                              attention='descriptive softmax weights; not source-token causal attribution'),
               positions={}, blocks={}, tick_effects=[], comparison={})
    for x in SIDES:
        labels = {p: (j, tid) for j, (p, tid, _) in enumerate(built[x]['labels'])}
        targets = {p: tid for p, (_, tid) in labels.items()}
        effects = contrast(pair, x, linked[x][0], cut[x][0], head_chunk, targets)
        view = sender_view(pair, x, built, records, top_reads)
        rows = []
        for p, (tid, tick) in enumerate(zip(built[x]['ids'], built[x]['time'])):
            label = labels.get(p)
            turn = next((turn for turn, a, z in built[x]['spans'] if label and a <= label[0] < z),
                        max((turn for turn, start in boundaries.items() if tick >= start), default=None))
            rows.append(dict(position=p, tick=tick, input_id=tid, input_text=pair.toks[x].decode([tid]),
                             target_id=None if label is None else label[1],
                             target_text=None if label is None else pair.toks[x].decode([label[1]]),
                             turn=turn, silent=tick in silent, payload=bool(label and label[0] in payload.get(x, [])),
                             reads=view[p], **effects[p]))
        out['positions'][x] = rows
        out['blocks'][x], out['comparison'][x] = {}, {}
        dec, head = inner(pair.models[x])
        for block in pair.xblocks:
            rec = records.pop((x, block))
            on = continue_block(pair, batch, x, block, rec['after'])
            off = continue_block(pair, batch, x, block, rec['before'])
            decoded = contrast(pair, x, on[0], off[0], head_chunk, targets)
            out['blocks'][x][str(block)] = decoded
            js = payload.get(x, [])
            ps = [built[x]['labels'][j][0] for j in js]
            ts = [built[x]['labels'][j][1] for j in js]
            ranks = []
            for a in range(0, len(ps), head_chunk):
                positions = ps[a:a + head_chunk]
                w = rec['write'][0, positions].to(head.weight.device, head.weight.dtype)
                logits = head(dec.norm(w)).float()
                tgt = torch.tensor(ts[a:a + head_chunk], device=logits.device)
                ranks.extend((logits > logits.gather(1, tgt[:, None])).sum(1).cpu().tolist())
            n = len(ps)
            out['comparison'][x][str(block)] = dict(
                n_payload=n, direct_top10=None if not n else sum(r < 10 for r in ranks) / n,
                downstream_top10=None if not n else sum(decoded[p]['target_decode_rank'] < 10 for p in ps) / n,
                promoted_top10=None if not n else sum(decoded[p]['target_promoted_rank'] < 10 and
                                                      decoded[p]['target_delta_nats'] > 0 for p in ps) / n,
                direct_ranks=ranks, downstream_ranks=[decoded[p]['target_decode_rank'] for p in ps])
            if block == max(pair.xblocks):
                if not torch.equal(on, linked[x]):
                    raise AssertionError('last exchange continuation differs from full joint pass')
    del cut, linked
    # Each batch's first sequence is the unablated control, so control and ablation share a batch shape.
    for a in range(0, len(selected), ablation_bs):
        chunk = selected[a:a + ablation_bs]
        b = S.collate([built] * (len(chunk) + 1), pads)
        h = joint(pair, b, [None] + chunk, direction)
        for row, tick in enumerate(chunk, 1):
            item = dict(tick=tick, direction=direction, receivers={}, pre_tick_max_abs=0.0)
            for x in SIDES:
                early = b[x]['tick'][0] < tick
                if not torch.equal(h[x][0, early], h[x][row, early]):
                    raise AssertionError(f'noncausal or numerical pre-tick change: {x}, tick {tick}')
                positions = (b[x]['tick'][0] >= tick).nonzero().flatten()
                effects = contrast(pair, x, h[x][0, positions], h[x][row, positions], head_chunk,
                                   {i: out['positions'][x][p]['target_id'] for i, p in enumerate(positions.tolist())
                                    if out['positions'][x][p]['target_id'] is not None})
                item['receivers'][x] = [dict(position=p, tick=built[x]['time'][p], **effect)
                                        for p, effect in zip(positions.tolist(), effects)]
            out['tick_effects'].append(item)
    out['seconds'] = time.perf_counter() - start
    out['execution'] = dict(ablation_bs=ablation_bs, control_rows=1, head_chunk=head_chunk,
                            joint_passes=2 + (len(selected) + ablation_bs - 1) // ablation_bs,
                            positions={x: len(built[x]['ids']) for x in SIDES},
                            model_dtype={x: str(inner(pair.models[x])[1].weight.dtype) for x in SIDES},
                            cuda_peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                                     if device.type == 'cuda' else None))
    return out


def render(result):
    """The text report: every position in tick order with its reads and total contrast, then the tick and block
    contrasts."""
    quote = lambda s: json.dumps(s, ensure_ascii=False)
    changes = lambda items: ', '.join(f'{quote(v["text"])} {v["nats"]:+.6g}' for v in items) or '(none)'
    lines = [f'{result["game"]}: teacher-forced next-token effects, nats; KL(control || link)']
    ordered = sorted((r['tick'], x, r['position'], r) for x in SIDES for r in result['positions'][x])
    for tick, x, pos, r in ordered:
        action = 'writes ' + quote(r['target_text']) if r['target_text'] is not None else \
            ('silent; reads ' if r['silent'] else 'reads ') + quote(r['input_text'])
        reads = ' '.join(f'{quote(v["text"])}({v["mass"]:.3f})' for v in r['reads']['top']) or '(none)'
        if not r['reads']['enabled']:
            reads = '(disabled at this tick)'
        lines.append(f'tick {tick}  {x} {action} [pos {pos}, turn {r["turn"]}] | '
                     f'read from {partner(x)}: {reads} | link raises: {changes(r["promoted"])} | '
                     f'KL {r["kl_cut_to_link"]:.6g}')
    for item in result['tick_effects']:
        for x, rows in item['receivers'].items():
            for r in rows:
                lines.append(f'ablate tick {item["tick"]} ({item["direction"]}) -> {x} tick {r["tick"]} '
                             f'pos {r["position"]} | tick raises: {changes(r["promoted"])} | '
                             f'tick lowers: {changes(r["suppressed"])} | KL {r["kl_cut_to_link"]:.6g}')
    for x, blocks in result['blocks'].items():
        for block, rows in blocks.items():
            for p, r in enumerate(rows):
                lines.append(f'block {block} -> {x} pos {p} | write raises: {changes(r["promoted"])} | '
                             f'KL {r["kl_cut_to_link"]:.6g}')
    return '\n'.join(lines) + '\n'


def episode_sets(replies, n, games=None, coread_data=None, pause=0):
    """{game: n test episodes}: payload_ctl.episodes' draws (the episodes analyze_bridge reads), and any other
    requested game drawn from Games or Coread with seed 4242; each with pause silent ticks."""
    import games as G
    from payload_ctl import episodes
    out = episodes(replies, n, coread=bool(games and any(g.startswith('co_') for g in games)),
                   path=coread_data, pause=pause)
    names = list(out) if games is None else games
    generator = G.Games(replies, 'test', seed=4242)
    co = None
    for name in names:
        if name in out:
            continue
        if name.startswith('co_'):
            import coread as C
            if name not in C.GAMES:
                raise ValueError(f'unknown game: {name}')
            if co is None:
                co = C.Coread(replies, 'test', seed=4242, path=coread_data)
            if name in ('co_play', 'co_talk', 'co_distill') and not getattr(co, name[3:]):
                raise ValueError(f'{name} data unavailable')
            out[name] = [co.episode(name) for _ in range(n)]
        else:
            out[name] = [generator.episode(name) for _ in range(n)]
        for ep in out[name]:
            if ep['mode'] == 'duplex':
                ep['pause'] = pause
            elif ep['mode'] == 'dialogue':
                ep['pause_turns'] = {j: pause for j in range(len(ep['turns']))}
    return {name: out[name] for name in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--games', default=None, help='comma-separated names; default seven value games')
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--pause', type=int, default=0)
    ap.add_argument('--coread_data')
    ap.add_argument('--ticks', default=None, help='comma-separated ticks, or all (default silent ticks + turn starts)')
    ap.add_argument('--direction', choices=('both', *SIDES), default='both', help='receiver of removed writes')
    ap.add_argument('--ablation_bs', type=int, default=4, help='ablated sequences per pass, plus one control sequence')
    ap.add_argument('--head_chunk', type=int, default=32)
    ap.add_argument('--top_reads', type=int, default=3)
    ap.add_argument('--max_positions', type=int, default=2048, help='fail before oversized episode; 0 disables guard')
    args = ap.parse_args()
    if (args.n < 1 or args.pause < 0 or args.ablation_bs < 1 or args.head_chunk < 1 or
            args.top_reads < 1 or args.max_positions < 0):
        ap.error('n, batch/chunk sizes, top_reads must be positive; pause/max_positions must be nonnegative')
    from evaluate import load_pair
    pair, _ = load_pair(args.run)
    eps = episode_sets(json.loads(Path(args.replies).read_text()), args.n,
                       None if args.games is None else args.games.split(','), args.coread_data, args.pause)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(run=args.run, arguments=vars(args), complete=False, episodes=[])
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for game, examples in eps.items():
        for i, ep in enumerate(examples):
            ticks = None
            if args.ticks == 'all':
                built = S.build(ep, pair.toks)
                ticks = sorted({t for x in SIDES for t in built[x]['time'] if t >= 0})
            elif args.ticks is not None:
                ticks = [int(t) for t in args.ticks.split(',')]
            result = analyze_episode(pair, ep, ticks, args.direction, args.ablation_bs, args.head_chunk,
                                     args.top_reads, args.max_positions)
            result.update(run=args.run, episode_index=i)
            stem = f'{game}_{i:03d}'
            (dest / f'{stem}.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
            (dest / f'{stem}.txt').write_text(render(result))
            entry = dict(game=game, index=i, json=f'{stem}.json', text=f'{stem}.txt', seconds=result['seconds'],
                         comparison=result['comparison'], execution=result['execution'])
            manifest['episodes'].append(entry)
            (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(json.dumps(entry), flush=True)
    manifest['complete'] = True
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
