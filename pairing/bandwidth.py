"""Payload code-length savings in bits, over lengths and silent-round budgets.

This is a predictive-information proxy, NOT Shannon capacity: a confident wrong
donor can make savings exceed source entropy. No rising/flattening fit is imposed.
Raw means have no distribution-free finite mean CI without a known bound. We
therefore report exact median intervals and a separate betting CI for the mean
of gains clipped to a predeclared +/- clip_bits. Clipping never alters raw scores.

Each scored item has its own independent same-role donor (draws with replacement,
including chance same-value donors). Donors are not scored. Batching composes
the focal receiver and donor sender, exactly the isolated pair produced by
payload_ctl.table on [item, donor] in shuffle mode, without batch-size dependence.
Inference is stratified by receiver; pause comparisons reuse the same pairs.

Games now holds out digit sum 3 modulo 7; older checkpoints used full support.
The one-digit test cell has ONLY '3' (zero entropy): it cannot show capacity.
--seen_values accepts a
complete training-value manifest {relay_digits: ["1 2", ...], relay_number: [...]}.
Exclusion certifies only that manifest, not pretraining. Without it numeric rows
are explicitly unverified unless --numeric_split_verified asserts the checkpoint
and all its ancestors used the current split. --require_unseen rejects unverified
rows. Exhausted cells fail.

python pairing/bandwidth.py --run RUN/step_N --replies REPLIES.json --out bandwidth.json
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

import games as G
import payload_ctl
import session as S
from link import DEVICE, SIDES, partner
from stats import betting_ci, median_ci

LENGTHS = (1, 2, 4, 8, 12, 16)
PAUSES = (0, 2, 4, 8)


def canonical(game, value):
    if game == 'relay_digits':
        digits = ''.join(str(value).split())
        if not digits or any(c not in '0123456789' for c in digits):
            raise ValueError('digit exclusions must contain decimal digits')
        return ' '.join(digits)
    if game == 'relay_number':
        return str(int(value))
    return str(value).casefold()


def numeric_support(length):
    """Exact cardinality of Games' digit-sum == 3 (mod 7) support, including zeros.

    relay_number 0..9999 has the same sums as four zero-padded digits.
    """
    counts = [1] + [0] * 6
    for _ in range(length or 4):
        counts = [sum(counts[(r - d) % 7] for d in range(10)) for r in range(7)]
    return counts[3]


def plans(replies, n=512, lengths=LENGTHS, seed=4242, seen_values=None, require_unseen=False,
          numeric_split_verified=False):
    """n focal/donor pairs per cell, exactly half per receiver, ordinary Games episodes.

    Rejection conditions on length, receiver and exclusion only; donors are NOT
    conditioned to differ in value (one-digit matches are legitimate null draws).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 4 or n % 2:
        raise ValueError('n must be even and at least 4')
    if not lengths or len(set(lengths)) != len(lengths) or any(type(k) is not int or k < 1 for k in lengths):
        raise ValueError('lengths must be distinct positive integers')
    if seen_values is not None and (not isinstance(seen_values, dict) or
                                   any(not isinstance(v, list) for v in seen_values.values())):
        raise ValueError('seen_values must map game names to complete lists of training values')
    out = {}
    for cell, (game, length) in enumerate([('relay_digits', k) for k in lengths] +
                                         [('relay_number', None), ('relay_word', None)]):
        supplied = seen_values is not None and game in seen_values
        excluded = {canonical(game, v) for v in (seen_values or {}).get(game, [])}
        if game == 'relay_word':
            excluded.update(canonical(game, v) for v in G.POOLS['words']['train'])
            status = 'pools_test_not_train'
            support = len({v.casefold() for v in G.POOLS['words']['test']} - excluded)
        else:
            status = ('excluded_from_supplied_training_values' if supplied else
                      'current_numeric_split_asserted_for_checkpoint' if numeric_split_verified else
                      'unverified_training_overlap')
            if require_unseen and not (supplied or numeric_split_verified):
                raise ValueError(f'{game} needs a complete --seen_values manifest for --require_unseen')
            support = numeric_support(length)
            blocked = sum(G.held_out_digits(v) and (len(v.split()) == length if length else
                          0 <= int(v) < 10000) for v in excluded)
            support -= blocked
            if support <= 0:
                raise ValueError(f'no unseen values remain for {game}/{length}')
        games = G.Games(replies, 'test', seed=seed + cell * 1009, max_digits=length or 16)

        def draw(receiver):
            for _ in range(100000):
                ep = games.episode(game)
                if ep['meta']['receiver'] != receiver:
                    continue
                value = ep['meta']['value']
                if length is not None and len(value.split()) != length:
                    continue
                if canonical(game, value) in excluded:
                    continue
                return ep
            raise ValueError(f'could not draw {game}/{length}; exclusion set may exhaust the pool')

        rows = []
        for i in range(n):
            r = SIDES[i % 2]
            ep, donor = draw(r), draw(r)
            # Use the established role definition, disallowing singleton fallback.
            if payload_ctl.donor_groups([ep, donor]) != [[0, 1]] or any(
                    ep['meta'].get(k) != donor['meta'].get(k) for k in payload_ctl.ROLES):
                raise ValueError('donor must have the same role')
            rows.append(dict(item=i, episode=ep, donor=donor, length=length, holdout=status,
                             source_support=support, source_entropy_bits=math.log2(support)))
        out[f'{game}/{length}' if length else game] = rows
    return out


def exposure(ep, built, toks, exchanges):
    """Logical sender exposure through the LAST payload prediction, inclusive.

    Both prompts occupy tick -1; it counts once, despite many token positions.
    Cached sender states stay exposed after it stops writing. Active sender ticks
    and actual reader position x depth exchanges are reported separately.
    """
    r, s = ep['meta']['receiver'], ep['meta']['sender']
    js = S.payload_labels(ep, toks)[r]
    positions = [built[r]['labels'][j][0] for j in js]
    last = max(built[r]['time'][p] for p in positions)
    ticks = last + 2  # -1, 0, ..., last
    return dict(sender_exposure_ticks=ticks, post_prompt_ticks=last + 1, prompt_ticks=1,
                sender_active_ticks=len({t for t in built[s]['time'] if t <= last}),
                depth_exchanges=exchanges, tick_exchanges=ticks * exchanges,
                reader_position_exchanges=(max(positions) + 1) * exchanges,
                payload_prediction_ticks=[built[r]['time'][p] for p in positions])


@torch.no_grad()
def measure(pair, toks, items, pause=0, bs=8, logit_chunk=128):
    """Per-token natural log p from the same CE kernel/labels as payload_ctl.table.

    Only payload logits are materialized; chunking bounds the full-vocabulary
    projection allocation. The LM backbones remain in the loaded dtype (bf16 on
    GPU, float32 tiny); CE and bridge math are float32.
    """
    if type(pause) is not int or pause < 0 or bs < 1 or logit_chunk < 1:
        raise ValueError('pause must be nonnegative; batch and logit chunk sizes positive')
    if not items:
        return []
    for x in SIDES:
        if pair.models[x].training:
            raise ValueError('benchmark requires eval models')
    eps = [it['episode'] for it in items]
    if len({ep['game'] for ep in eps}) != 1:
        raise ValueError('measure requires one game')
    records = []
    pads = {x: toks[x].pad_token_id for x in SIDES}
    for group in payload_ctl.donor_groups(eps):
        for start in range(0, len(group), bs):
            indices = group[start:start + bs]
            chunk = [items[i] for i in indices]
            episodes = [dict(it['episode'], pause=pause) for it in chunk]
            donors = [dict(it['donor'], pause=pause) for it in chunk]
            built = [S.build(ep, toks) for ep in episodes]
            donor_built = [S.build(ep, toks) for ep in donors]
            mixed = []
            rows = []
            for it, ep, donor, a, d in zip(chunk, episodes, donors, built, donor_built):
                r, s = ep['meta']['receiver'], ep['meta']['sender']
                if ep['game'] != donor['game'] or any(ep['meta'].get(k) != donor['meta'].get(k)
                                                      for k in payload_ctl.ROLES):
                    raise ValueError('donor must have the same game and role')
                js = S.payload_labels(ep, toks).get(r, [])
                if not js:
                    raise ValueError('missing receiver payload')
                mixed.append({r: a[r], s: d[s]})
                rows.append(dict(item=it['item'], game=ep['game'], length=it['length'], pause=pause,
                                 receiver=r, value=ep['meta']['value'], donor_value=donor['meta']['value'],
                                 holdout=it['holdout'], n_tokens=len(js), payload_label_indices=js,
                                 source_support=it['source_support'], source_entropy_bits=it['source_entropy_bits'],
                                 token_ids=[a[r]['labels'][j][1] for j in js],
                                 **exposure(ep, a, toks, len(pair.xblocks))))
            for cond, sequences, mode in (('link', built, None), ('shuffle', mixed, None), ('cut', built, 'zero')):
                batch = S.collate(sequences, pads)
                hidden, _ = pair.joint(batch, mode=mode)
                for x in SIDES:
                    chosen, owners = [], []
                    for i, row in enumerate(rows):
                        if row['receiver'] == x:
                            L = batch[x]['ids'].shape[1]
                            chosen.extend(i * L + sequences[i][x]['labels'][j][0]
                                          for j in row['payload_label_indices'])
                            owners.extend([i] * row['n_tokens'])
                            row[cond + '_logp'] = []
                            row[cond + '_argmax'] = []
                    targets = [token for row in rows if row['receiver'] == x for token in row['token_ids']]
                    for lo in range(0, len(chosen), logit_chunk):
                        ix = torch.tensor(chosen[lo:lo + logit_chunk], device=DEVICE)
                        target = torch.tensor(targets[lo:lo + logit_chunk], device=DEVICE)
                        logits = pair.token_logits(x, hidden[x], ix)
                        lp = -F.cross_entropy(logits, target, reduction='none')
                        for i, p, pred in zip(owners[lo:lo + logit_chunk], lp.tolist(), logits.argmax(-1).tolist()):
                            rows[i][cond + '_logp'].append(p)
                            rows[i][cond + '_argmax'].append(pred)
                        del logits, lp
                del hidden, batch
            for row in rows:
                for cond in ('link', 'shuffle', 'cut'):
                    row[cond + '_ce'] = -math.fsum(row[cond + '_logp']) / row['n_tokens']
                    row[cond + '_code_bits'] = -math.fsum(row[cond + '_logp']) / math.log(2)
                    row[cond + '_tf_token_accuracy'] = sum(a == b for a, b in zip(
                        row['token_ids'], row[cond + '_argmax'])) / row['n_tokens']
                # Difference first: alpha=0 is EXACTLY zero, no rounded CE subtraction.
                row['bits'] = math.fsum(a - b for a, b in zip(row['link_logp'], row['shuffle_logp'])) / math.log(2)
                row['cut_bits'] = math.fsum(a - b for a, b in zip(row['link_logp'], row['cut_logp'])) / math.log(2)
                row['bits_per_tick'] = row['bits'] / row['sender_exposure_ticks']
                row['bits_per_exchange'] = row['bits'] / row['tick_exchanges']
            records.extend(rows)
    return sorted(records, key=lambda r: r['item'])


def interval(values, alpha=.05, clip_bits=128.):
    """Raw exact median CI; separate bounded-mean betting CI. None = infinity."""
    if not math.isfinite(clip_bits) or clip_bits <= 0:
        raise ValueError('clip_bits must be finite and positive')
    xs = list(values)
    if not xs:
        raise ValueError('empty sample')
    clipped = [max(-clip_bits, min(clip_bits, x)) for x in xs]
    ci = median_ci(xs, alpha)
    return dict(n=len(xs), mean=statistics.mean(xs), median=statistics.median(xs),
                median_ci=[v if math.isfinite(v) else None for v in ci],
                clipped_mean=statistics.mean(clipped), clipped_mean_ci=list(betting_ci(
                    clipped, alpha, bounds=(-clip_bits, clip_bits))), clipped_items=sum(a != b for a, b in zip(xs, clipped)))


def summarize(rows, alpha=.05, clip_bits=128.):
    """95% per-cell inference; no simultaneous claim across lengths/pauses.

    Fixed receiver quotas are not iid draws from a mixture. Mean CIs combine
    two per-role intervals at alpha/2 by Bonferroni; raw median intervals are
    per role only. Pooled raw mean/median are descriptive, without a fake CI.
    """
    sides = {x: [r for r in rows if r['receiver'] == x] for x in SIDES}
    if any(not rs for rs in sides.values()):
        raise ValueError('summary needs both receivers')
    per = {x: interval([r['bits'] for r in rs], alpha / 2, clip_bits) for x, rs in sides.items()}
    for x, rs in sides.items():
        for metric in ('bits_per_tick', 'bits_per_exchange'):
            ci = median_ci([r[metric] for r in rs], alpha / 2)
            per[x][metric] = dict(mean=statistics.mean(r[metric] for r in rs),
                                  median=statistics.median(r[metric] for r in rs),
                                  median_ci=[v if math.isfinite(v) else None for v in ci])
    n = len(rows)
    weights = {x: len(rs) / n for x, rs in sides.items()}
    out = dict(n=n, mean_bits=statistics.mean(r['bits'] for r in rows),
               source_support=rows[0]['source_support'], source_entropy_bits=rows[0]['source_entropy_bits'],
               median_bits=statistics.median(r['bits'] for r in rows),
               clipped_mean_bits=sum(weights[x] * per[x]['clipped_mean'] for x in SIDES),
               clipped_mean_ci=[sum(weights[x] * per[x]['clipped_mean_ci'][j] for x in SIDES) for j in (0, 1)],
               clipped_items=sum(p['clipped_items'] for p in per.values()), per_receiver=per,
               mean_cut_bits=statistics.mean(r['cut_bits'] for r in rows),
               mean_bits_per_tick=statistics.mean(r['bits_per_tick'] for r in rows),
               mean_bits_per_exchange=statistics.mean(r['bits_per_exchange'] for r in rows),
               total_bits_per_total_ticks=sum(r['bits'] for r in rows) / sum(r['sender_exposure_ticks'] for r in rows),
               same_value_donors=sum(r['value'] == r['donor_value'] for r in rows),
               mean_exposure_ticks=statistics.mean(r['sender_exposure_ticks'] for r in rows))
    return out


@torch.no_grad()
def greedy(pair, items, pause, n=4, bs=4, max_turn=64):
    """Free autoregressive decoding, NOT teacher-forced argmax accuracy.

    The balanced first n focal items are decoded under all three conditions.
    Shuffle composes the same receiver/donor streams in an ordinary duplex ep.
    """
    from evaluate import digits_after
    out = []
    for start in range(0, min(n, len(items)), bs):
        chunk = items[start:min(start + bs, n)]
        for cond, mode in (('link', None), ('shuffle', None), ('cut', 'zero')):
            eps = []
            for it in chunk:
                ep = copy.deepcopy(it['episode'])
                ep['pause'] = pause
                if cond == 'shuffle':
                    s = ep['meta']['sender']
                    for key in ('system', 'task', 'reply'):
                        ep[key][s] = it['donor'][key][s]
                eps.append(ep)
            results = S.generate(pair, eps, max_turn=max_turn, mode=mode, temperature=0.)
            for it, ep, result in zip(chunk, eps, results):
                text = result['text'][ep['meta']['receiver']]
                want = ep['meta']['value'].split()
                got = digits_after(text)
                out.append(dict(item=it['item'], condition=cond, text=text, value=ep['meta']['value'],
                                per_digit=sum(a == b for a, b in zip(want, got)) / len(want),
                                exact=float(got == want)))
    return out


def text_table(results):
    lines = ['Predictive code-length savings; not Shannon capacity. Numeric holdout status is in JSON.',
             'Current Games split: digit sum = 3 mod 7. Length 1 has only value 3: zero source entropy.',
             'CI is for the CAPPED mean (not the raw mean); raw exact median CIs per receiver are in JSON.',
             'cell                 pause    n   raw bits   capped mean [CI]           bits/tick bits/exchange cut bits']
    for cell, cells in results.items():
        for k, data in cells.items():
            s = data['summary']
            lo, hi = s['clipped_mean_ci']
            lines.append(f'{cell:22s} {str(k):>3s} {s["n"]:5d} {s["mean_bits"]:10.5g} '
                         f'{s["clipped_mean_bits"]:9.4f} [{lo:8.3f},{hi:8.3f}] '
                         f'{s["mean_bits_per_tick"]:10.4g} {s["mean_bits_per_exchange"]:13.5g} {s["mean_cut_bits"]:8.4g}')
            if data.get('greedy'):
                acc = {c: statistics.mean(r['per_digit'] for r in data['greedy'] if r['condition'] == c)
                       for c in ('link', 'shuffle', 'cut')}
                lines.append('  free greedy per-digit accuracy: ' + ' '.join(f'{c}={v:.3f}' for c, v in acc.items()))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=512, help='focal items per length, half per receiver; donors extra')
    ap.add_argument('--bs', type=int, default=8, help='focal pairs per forward batch')
    ap.add_argument('--lengths', default=','.join(map(str, LENGTHS)))
    ap.add_argument('--pauses', default=','.join(map(str, PAUSES)))
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--seen_values', help='complete per-game training-value exclusion JSON')
    ap.add_argument('--require_unseen', action='store_true')
    ap.add_argument('--numeric_split_verified', action='store_true',
                    help='assert checkpoint AND all ancestors trained with the current numeric split')
    ap.add_argument('--clip_bits', type=float, default=128.)
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--logit_chunk', type=int, default=128)
    ap.add_argument('--greedy_n', type=int, default=4, help='balanced focal subset per digit cell; 0 disables')
    ap.add_argument('--greedy_bs', type=int, default=4)
    ap.add_argument('--max_turn', type=int, default=64)
    args = ap.parse_args()
    pauses = sorted(int(k) for k in args.pauses.split(','))
    if not pauses or len(set(pauses)) != len(pauses) or any(k < 0 for k in pauses):
        ap.error('pauses must be distinct nonnegative integers')
    if args.greedy_n < 0 or args.greedy_n % 2 or args.greedy_n > args.n or args.greedy_bs < 1 or args.max_turn < 1:
        ap.error('greedy_n must be even, between 0 and n; greedy_bs and max_turn positive')
    interval([0.], args.alpha, args.clip_bits)  # validate before loading models
    if args.bs < 1 or args.logit_chunk < 1:
        ap.error('bs and logit_chunk must be positive')
    if not os.environ.get('NL_TINY') and not DEVICE.startswith('cuda'):
        ap.error('full models require CUDA; use NL_TINY=1 NL_DEVICE=cpu on the Mac')
    start = time.perf_counter()
    R = json.loads(Path(args.replies).read_text())
    seen = json.loads(Path(args.seen_values).read_text()) if args.seen_values else None
    items = plans(R, args.n, tuple(int(k) for k in args.lengths.split(',')), args.seed, seen, args.require_unseen,
                  args.numeric_split_verified)
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    out = dict(config=vars(args), tiny=bool(os.environ.get('NL_TINY')), device=DEVICE,
               dtype={x: str(next(pair.models[x].parameters()).dtype) for x in SIDES},
               replies_sha256=hashlib.sha256(Path(args.replies).read_bytes()).hexdigest(),
               seen_values_sha256=hashlib.sha256(Path(args.seen_values).read_bytes()).hexdigest() if args.seen_values else None,
               code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('bandwidth.py', 'games.py', 'link.py', 'session.py', 'payload_ctl.py', 'stats.py')},
               definitions=dict(bits='sum(logp_link - logp_shuffle)/ln(2); raw unbounded predictive contrast',
                   interval='per-cell stratified betting CI of clipped gain mean; exact raw median CI per receiver',
                   exposure='ticks -1 through final payload prediction inclusive; cached sender remains exposed',
                   exchange='one depth exchange per logical exposure tick; prompt tick bundles many positions',
                   rates='mean per-item gain/denominator; total gain/total ticks also provided',
                   holdout='current Games numeric test split; old checkpoints unverified without supplied exclusions',
                   source_entropy='log2 current test-support size after exclusions; length 1 is degenerate'),
               cells={}, plans=items)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for cell, cell_items in items.items():
        out['cells'][cell] = {}
        for k in pauses:
            t = time.perf_counter()
            rows = measure(pair, toks, cell_items, k, args.bs, args.logit_chunk)
            data = dict(rows=rows, summary=summarize(rows, args.alpha, args.clip_bits))
            if cell_items[0]['episode']['game'] == 'relay_digits' and args.greedy_n:
                data['greedy'] = greedy(pair, cell_items, k, args.greedy_n, args.greedy_bs, args.max_turn)
            if 0 in out['cells'][cell] and k:
                base = out['cells'][cell][0]['rows']
                changes = [dict(r, bits=r['bits'] - b['bits'], cut_bits=r['cut_bits'] - b['cut_bits'],
                                bits_per_tick=r['bits_per_tick'] - b['bits_per_tick'],
                                bits_per_exchange=r['bits_per_exchange'] - b['bits_per_exchange'])
                           for r, b in zip(rows, base)]
                data['pause_minus_zero'] = summarize(changes, args.alpha, 2 * args.clip_bits)
            data['seconds'] = time.perf_counter() - t
            out['cells'][cell][k] = data
            out['seconds'] = time.perf_counter() - start
            out['complete'] = False
            output.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
            print(f'{cell} pause={k} mean_bits={data["summary"]["mean_bits"]:.5f} ({data["seconds"]:.1f}s)', flush=True)
    out['complete'] = True
    if DEVICE.startswith('cuda'):
        out['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30
    output.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    table = text_table(out['cells'])
    output.with_suffix('.txt').write_text(table + '\n')
    print(table, flush=True)


if __name__ == '__main__':
    main()
