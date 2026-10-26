"""The same secrets under every turn structure, scored on payload cross-entropy.

Items (secrets): n secrets balanced over KINDS and receivers (words, names and cities: single-word test values that
occur in no fixed prompt text; numbers: 1000 to 9999), each with an independently drawn donor secret of the same kind
that differs from it. Structures: duplex, dialogue (the holder first says NEUTRAL, then the receiver writes the answer)
and the shared pen; duplex and dialogue also with each of DEFAULT_PAUSES silent ticks. The receiver's target is
'The {noun} is {value}.' and its payload the value. assert_no_text_leak checks that the secret is in no text the
receiver reads before its first payload token. Per item: payload CE linked and cut (payload_ctl.table), and shuffled:
the receiver's own stream joined with the donor episode's holder stream in one Pair.joint (the isolated pair of mode
'shuffle' on a two-episode batch; donors are not scored). Reported per structure, and per kind and receiver: shuffle
minus link and cut minus link (exact sign test and median interval), ticks, forward tokens and payload nats per tick.

Numbers have no test split in pools.json and are marked unverified unless --seen_numbers lists every training secret
to exclude. Ticks are session ticks (the pen game has one per word); forward tokens count both models' input
positions, prompts included.

Writes OUT (JSON) and OUT with suffix .txt (a table and notes).
Usage: python turnbench.py --run RUN/step_N --replies REPLIES.json --out OUT/turnbench.json [--n 128] [--bs 8]
           [--seen_numbers NUMBERS.json]
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import coldstart
import games as G
import payload_ctl
import session as S
from link import SIDES, partner
from stats import median_ci

KINDS = ('words', 'names', 'cities', 'numbers')
NEUTRAL = 'Ready.'
DEFAULT_PAUSES = (1, 4, 8)


def receiver_task(holder, noun):
    """The receiver's fixed task."""
    return f'Read the {noun} {G.MODELS[holder]["name"]} sends you through the link and write it.'


def pen_instruction(holder, receiver, noun):
    """The pen game's added instruction: a four-word answer, one word each, the holder first."""
    return (f' Share one four-word answer beginning "The {noun} is", alternating one word each. '
            f'{G.MODELS[holder]["name"]} starts; {G.MODELS[receiver]["name"]} writes the secret last.')


def carrier_text():
    """Every fixed prompt text of every structure; secrets are drawn from values that occur in none of it."""
    parts = [G.system(x, mode) for x in SIDES for mode in ('duplex', 'dialogue', 'pen')]
    parts += [NEUTRAL, 'The word is . The number is .']
    for holder in SIDES:
        for noun in ('word', 'number'):
            parts += [receiver_task(holder, noun), pen_instruction(holder, partner(holder), noun)]
    return '\n'.join(parts)


def secrets(n=128, seed=4242, seen_numbers=None):
    """n items balanced over kinds and receivers; each secret and its donor are drawn with replacement (the test pool
    has only 18 names), the donor differing from its own item's secret. seen_numbers: the training numbers to exclude
    (an empty list counts as supplied)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 8 or n % 8:
        raise ValueError('n must be a positive multiple of 8 (four kinds, two receivers)')
    rng = random.Random(seed)
    pools = {}
    public = carrier_text().casefold()
    for kind in KINDS[:-1]:
        train = {s.casefold() for s in G.POOLS[kind]['train']}
        pools[kind] = [v for v in G.POOLS[kind]['test'] if v.casefold() not in train and len(v.split()) == 1
                       and v.casefold() not in public]
    excluded = {str(int(v)) for v in (seen_numbers or [])}
    pools['numbers'] = [str(v) for v in range(1000, 10000) if str(v) not in excluded]
    if any(len(v) < 2 for v in pools.values()):
        raise ValueError('every pool needs at least two distinct single-word secrets')
    out = []
    for i in range(n):
        kind, receiver = KINDS[(i // 2) % 4], SIDES[i % 2]
        value = rng.choice(pools[kind])
        donor = rng.choice([v for v in pools[kind] if v != value])
        if kind != 'numbers':
            status = 'pools_test_not_train'
        elif seen_numbers is not None:
            status = 'excluded_from_supplied_training_list'
        else:
            status = 'unverified_training_overlap'
        out.append(dict(item=i, kind=kind, receiver=receiver, holder=partner(receiver), value=value,
                        donor_value=donor, holdout=status))
    return out


def episode(games, item, structure, k=0, donor=False):
    """The item's episode in structure with k silent ticks (donor: with the donor secret): a coldstart relay
    episode in the item's direction with the secret and the receiver's task replaced, rebuilt for dialogue or pen."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('pause must be a nonnegative integer')
    if structure not in ('duplex', 'dialogue', 'pen') or (structure == 'pen' and k):
        raise ValueError('unknown structure or unsupported pen pause')
    value = item['donor_value'] if donor else item['value']
    receiver, holder = item['receiver'], item['holder']
    game = 'relay_number' if item['kind'] == 'numbers' else 'relay_word'
    ep = coldstart.relay_episode(games, (game,))
    while ep['meta']['receiver'] != receiver:
        ep = coldstart.relay_episode(games, (game,))
    old = ep['meta']['value']
    ep['task'][holder] = ep['task'][holder].replace('"' + old + '"', '"' + value + '"')
    noun = 'number' if game == 'relay_number' else 'word'
    target = f'The {noun} is {value}.'
    ep['reply'][receiver] = (target, 1.0)
    ep['task'][receiver] = receiver_task(holder, noun)
    ep['meta'] = dict(item=item['item'], kind=item['kind'], holder=holder, sender=holder,
                      receiver=receiver, value=value)
    if structure == 'duplex':
        if k:
            ep['pause'] = k
        return ep
    ep['system'] = {x: G.system(x, structure) for x in SIDES}
    if structure == 'dialogue':
        ep.update(game='co_turnbench', mode='dialogue', turns=[(holder, NEUTRAL, 0.0), (receiver, target, 1.0)],
                  forced=[0])
        ep['task'][holder] += f' First say exactly "{NEUTRAL}"; do not write your secret.'
        ep['meta']['payload'] = {receiver: [(len(f'The {noun} is '), len(target) - 1)]}
        del ep['reply']
        if k:
            ep['pause_turns'] = {1: k}
        return ep
    pen = G.pen_episode(games, 'pen_secret', chunk=1)
    pen.update(system=ep['system'], task=ep['task'], text=target, pen_start=holder, pen_words=4, meta=ep['meta'])
    for x in SIDES:
        pen['task'][x] += pen_instruction(holder, receiver, noun)
    return pen


def episodes(replies, items, pauses=DEFAULT_PAUSES):
    """{structure name: [(item episode, donor episode)]}."""
    specs = [('duplex', 'duplex', 0)] + [(f'duplex+{k}', 'duplex', k) for k in pauses]
    specs += [('dialogue', 'dialogue', 0)] + [(f'dialogue+{k}', 'dialogue', k) for k in pauses]
    specs += [('pen', 'pen', 0)]
    out = {}
    for name, structure, k in specs:
        games = G.Games(replies, 'test', seed=103)
        out[name] = [(episode(games, it, structure, k), episode(games, it, structure, k, donor=True)) for it in items]
    return out


def contains(seq, sub):
    """Whether sub is a nonempty contiguous part of seq."""
    return bool(sub) and any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def assert_no_text_leak(ep, toks):
    """Raise if the secret (alone or after a space, quote or newline) is in the receiver's input up to its first
    payload token or in the holder's visible text, in either tokenizer, or in the receiver's token ids; returns
    session.build(ep). Later input tokens are the teacher-forced answer itself."""
    r, value = ep['meta']['receiver'], ep['meta']['value']
    built = S.build(ep, toks)
    js = S.payload_labels(ep, toks).get(r, [])
    if not js:
        raise ValueError('missing receiver payload')
    first = built[r]['labels'][js[0]][0]
    context_ids = built[r]['ids'][:first + 1]
    context = toks[r].decode(context_ids)
    if ep['mode'] == 'duplex':
        public = ep['reply'][partner(r)][0]
    elif ep['mode'] == 'dialogue':
        public = '\n'.join(t for x, t, _ in ep['turns'] if x != r)
    else:
        words = ep['text'].split()
        public = ' '.join(w for j, w in enumerate(words) if S.pen_author(ep, j) != r)
    for tok in toks.values():
        for text in (context, public):
            seq = S.ids(tok, text)
            for prefix in ('', ' ', '"', '\n'):
                secret_ids = S.ids(tok, prefix + value)
                if contains(seq, secret_ids):
                    raise ValueError(f'secret in visible text: {value}')
    # The receiver's token ids as built, without decoding.
    for prefix in ('', ' ', '"', '\n'):
        if contains(context_ids, S.ids(toks[r], prefix + value)):
            raise ValueError(f'secret in receiver token IDs: {value}')
    return built


def costs(ep, built, toks):
    """Ticks (all, and up to the last payload prediction), forward tokens and prompt tokens of one episode."""
    r = ep['meta']['receiver']
    js = S.payload_labels(ep, toks)[r]
    times = [built[r]['time'][built[r]['labels'][j][0]] for j in js]
    return dict(ticks=1 + max(t for x in SIDES for t in built[x]['time']),
                payload_ticks=1 + max(times), prompt_ticks=1,
                forward_tokens=sum(len(built[x]['ids']) for x in SIDES),
                prompt_tokens=sum(t < 0 for x in SIDES for t in built[x]['time']))


@torch.no_grad()
def measure(pair, toks, plans, bs=8):
    """One record per item: payload_ctl.table's link and cut CE, the shuffled CE with the donor's holder, and costs.
    Plans of several kinds or receivers are measured one group at a time."""
    if bs < 1:
        raise ValueError('bs must be positive')
    groups = {}
    for ep, donor in plans:
        groups.setdefault((ep['meta']['kind'], ep['meta']['receiver']), []).append((ep, donor))
    if len(groups) > 1:
        return sorted([row for group in groups.values() for row in measure(pair, toks, group, bs)],
                      key=lambda row: row['item'])
    pads = {x: toks[x].pad_token_id for x in SIDES}
    records = []
    for start in range(0, len(plans), bs):
        chunk = plans[start:start + bs]
        eps = [ep for ep, _ in chunk]
        built = [assert_no_text_leak(ep, toks) for ep in eps]
        donors = [assert_no_text_leak(d, toks) for _, d in chunk]
        mixed = []
        for (ep, donor), a, b in zip(chunk, built, donors):
            r, h = ep['meta']['receiver'], ep['meta']['holder']
            assert donor['meta']['receiver'] == r and donor['meta']['kind'] == ep['meta']['kind']
            assert ep['meta']['value'] != donor['meta']['value']
            mixed.append({r: a[r], h: b[h]})
        # The chunk in one table batch (table's bs must be at least 2; link and cut need no donor).
        base = payload_ctl.table(pair, toks, eps, bs=max(2, len(eps)), conds=payload_ctl.CONDS[:2])
        batch = S.collate(mixed, pads)
        hidden, _ = pair.joint(batch)
        shuffled = {x: F.cross_entropy(pair.token_logits(x, hidden[x], batch[x]['idx']), batch[x]['lab'],
                                       reduction='none').tolist() for x in SIDES}
        offsets = {x: 0 for x in SIDES}
        by_row = {row['episode']: row for row in base}
        padded = 2 * sum(len(eps) * max(len(e[x]['ids']) for e in built) for x in SIDES)
        padded += sum(batch[x]['ids'].numel() for x in SIDES)
        for i, ((ep, donor), b, mix) in enumerate(zip(chunk, built, mixed)):
            r = ep['meta']['receiver']
            js = S.payload_labels(ep, toks)[r]
            row = dict(by_row[i], item=ep['meta']['item'], kind=ep['meta']['kind'], value=ep['meta']['value'],
                       donor_value=donor['meta']['value'], **costs(ep, b, toks))
            row['episode'] = ep['meta']['item']
            row['shuffle_ce'] = sum(shuffled[r][offsets[r] + j] for j in js) / len(js)
            row['shuffle_minus_link'] = row['shuffle_ce'] - row['link_ce']
            row['cut_minus_link'] = row['cut_ce'] - row['link_ce']
            row['payload_nats_per_tick'] = row['shuffle_minus_link'] * len(js) / row['ticks']
            row['evaluation_forward_tokens'] = 2 * row['forward_tokens'] + sum(len(mix[x]['ids']) for x in SIDES)
            row['evaluation_padded_tokens_share'] = padded / len(eps)
            records.append(row)
            for x in SIDES:
                offsets[x] += len(mix[x]['labels'])
    return records


def comparison(values):
    """One-sided exact sign test and conservative exact 95% median interval."""
    wins, losses = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = wins + losses
    p = sum(math.comb(n, j) for j in range(wins, n + 1)) / (1 << n)
    lo, hi = median_ci(values)
    return dict(wins=wins, n=n, ties=len(values) - n, p=p, median_nats=statistics.median(values),
                median_ci95=[lo if math.isfinite(lo) else None, hi if math.isfinite(hi) else None])


def summarize(rows):
    """comparison() of shuffle minus link and cut minus link, with mean ticks and token counts."""
    return dict(items=len(rows), shuffle=comparison([r['shuffle_minus_link'] for r in rows]),
                cut=comparison([r['cut_minus_link'] for r in rows]),
                ticks_per_item=statistics.mean(r['ticks'] for r in rows),
                payload_ticks_per_item=statistics.mean(r['payload_ticks'] for r in rows),
                forward_tokens_per_item=statistics.mean(r['forward_tokens'] for r in rows),
                evaluation_forward_tokens=sum(r['evaluation_forward_tokens'] for r in rows),
                evaluation_padded_tokens=sum(r['evaluation_padded_tokens_share'] for r in rows),
                median_payload_nats_per_tick=statistics.median(r['payload_nats_per_tick'] for r in rows))


def text_table(summaries):
    """The .txt table: per structure, the shuffle comparison, ticks and tokens per item."""
    lines = ['structure       wins/n (ties)       p    median nats       median 95% CI   ticks/item   tokens/item']
    for name, row in summaries.items():
        s = row['shuffle']
        lo, hi = s['median_ci95']
        interval = f'[{lo:.5g}, {hi:.5g}]' if lo is not None and hi is not None else '[-inf, inf]'
        lines.append(f'{name:<15} {s["wins"]:>3}/{s["n"]:<3} ({s["ties"]:>3}) {s["p"]:>9.3g} '
                     f'{s["median_nats"]:>12.6g} {interval:>22} {row["ticks_per_item"]:>12.2f} '
                     f'{row["forward_tokens_per_item"]:>13.1f}')
    return '\n'.join(lines)


def main():
    from evaluate import load_pair
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=128, help='items per structure, multiple of 8; default 128')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--seen_numbers', help='JSON list of ALL number secrets seen in training; excludes those values')
    args = ap.parse_args()
    items = secrets(args.n, args.seed, json.loads(Path(args.seen_numbers).read_text()) if args.seen_numbers else None)
    plans = episodes(json.loads(Path(args.replies).read_text()), items)
    pair, toks = load_pair(args.run)
    pair.eval()
    summaries, records, breakdown = {}, {}, {}
    started = time.perf_counter()
    for name, rows in plans.items():
        at = time.perf_counter()
        records[name] = measure(pair, toks, rows, args.bs)
        summaries[name] = summarize(records[name])
        summaries[name]['seconds'] = time.perf_counter() - at
        breakdown[name] = {f'{kind}/{r}': summarize([v for v in records[name]
                                                     if v['kind'] == kind and v['receiver'] == r])
                           for kind in KINDS for r in SIDES}
        print(f'{name}: {len(rows)} items, {summaries[name]["seconds"]:.2f}s', flush=True)
    notes = [
        'CE and median nats are mean nats per payload token; positive shuffle-minus-link favors the real holder.',
        'Exact one-sided binomial sign test drops exact ties. Median intervals are exact binomial order statistics; '
        'null JSON endpoints mean unbounded. Inference assumes independent item signs; stratified kinds/directions '
        'are also reported. P values are unadjusted for multiple structures.',
        'Primary and distinct donor draws are independent across items, with replacement from small finite pools. '
        'The same items/donors recur across structures, so comparisons across structures are paired.',
        'Numbers have no pools.json test split and may overlap training unless a complete --seen_numbers list is '
        'supplied.',
        'Lexical values are single-word entries from test pools, disjoint from same-kind train pools and absent '
        'from the union of fixed carrier texts in all structures; '
        'this does not claim absence from base pretraining or arbitrary training text.',
        'Ticks exclude the common prompt tick (-1); include final flush/EOS prediction. payload_ticks stops at the '
        'last payload prediction. Pen ticks are words, duplex/dialogue ticks are tokens; use forward-token costs too.',
        'Payload nats per tick is a CE advantage rate, not a Shannon channel-capacity estimate.',
    ]
    result = dict(run=args.run, n=args.n, bs=args.bs, seed=args.seed, notes=notes, items=items,
                  summaries=summaries, by_kind_receiver=breakdown, records=records,
                  evaluation_seconds=time.perf_counter() - started,
                  code_sha256={p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                               for p in ('turnbench.py', 'session.py', 'link.py', 'payload_ctl.py', 'games.py',
                                         'coldstart.py')})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table = text_table(summaries) + '\n\n' + '\n'.join(notes) + '\n'
    out.with_suffix('.txt').write_text(table)
    print(table, end='')
    print('wrote', out, 'and', out.with_suffix('.txt'))


if __name__ == '__main__':
    main()
