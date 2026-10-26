"""Merges evalfree --shard co_chat outputs (DIR/episodes.jsonl) and summarizes them with chateval.summary_rows; with
--before, also pairs the same plans (episode_id, receiver) across the two checkpoints: per condition and metric, the
mean change after - before with an exact sign test.

Usage: python chat_merge.py --after DIR [DIR ...] [--before DIR [DIR ...]] [--out FILE.json]
"""
import argparse
import json
from pathlib import Path

from chateval import METRICS, estimate, summary_rows
from stats import sign_test


def load(dirs):
    """The records of every DIR/episodes.jsonl; ValueError when a chat occurs in two shards."""
    records = [json.loads(line) for d in dirs for line in open(Path(d) / 'episodes.jsonl') if line.strip()]
    keys = [(r['episode_id'], r['receiver']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('a chat appears in two shards')
    return records


def paired(before, after):
    """dict(n_pairs, changes={'condition/metric': mean change with its confidence interval and sign test}) over the
    records of the same plan in both checkpoints."""
    old = {(r['episode_id'], r['receiver']): r for r in before}
    both = [(old[k], r) for r in after for k in [(r['episode_id'], r['receiver'])] if k in old]
    out = {}
    for c in sorted({c for _, r in both for c in r['conditions']}):
        for m in METRICS:
            ds = [a['conditions'][c][m] - b['conditions'][c][m] for b, a in both
                  if c in a['conditions'] and c in b['conditions']
                  and a['conditions'][c][m] is not None and b['conditions'][c][m] is not None]
            if ds:
                out[f'{c}/{m}'] = dict(estimate(ds, bounds=(-1., 1.)), sign=sign_test(ds))
    return dict(n_pairs=len(both), changes=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--after', nargs='+', required=True)
    ap.add_argument('--before', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    result = dict(after=summary_rows(load(args.after)))
    if args.before:
        before = load(args.before)
        result['before'] = summary_rows(before)
        result['after_minus_before'] = paired(before, load(args.after))
    for k in ('before', 'after'):
        for row in result.get(k, []):
            cells = '  '.join(f"{c}: " + ' '.join(f"{m} {v['mean']:.2f}" for m, v in row['conditions'][c].items()
                                                 if v['mean'] is not None
                                                 and m in ('names', 'loop', 'game', 'distinct2'))
                              for c in row['conditions'])
            extra = '  '.join(f"{t} {row[t]['sign']['positive']}-{row[t]['sign']['negative']} "
                              f"p>{row[t]['sign']['p_greater']:.3g}" for t in ('link-cut', 'link-crossed') if t in row)
            group = f" {row['game']} {row['subset']}" if row['game'] == 'co_chat2' else ''
            counts = ('  ' + ' '.join(f"{c} {v['successes']}/{v['n']}" for c, v in row['names_counts'].items())
                      if 'names_counts' in row else '')
            print(f"{k:6s} {row['receiver']:8s}{group} n {row['n']:3d}  {cells}  {extra}{counts}")
    for name, v in result.get('after_minus_before', {}).get('changes', {}).items():
        print(f"after-before {name:22s} mean {v['mean']:+.3f}  +{v['sign']['positive']} -{v['sign']['negative']}  "
              f"p {v['sign']['p']:.3g}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
