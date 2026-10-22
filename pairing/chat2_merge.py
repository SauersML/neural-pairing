"""Merge complete chat2 shards and print counts and paired exact tests by kind.

Unlike the legacy merger, require every index from all requested shards and
verify completed manifests/hashes before summarizing. chat_merge.py also prints
the same per-kind rows, with its existing permissive partial-input behavior.
"""
import argparse
import json
from pathlib import Path

import chateval
from evalfree import file_hash
from link import SIDES


def load(dirs):
    records, manifests = [], []
    for d in map(Path, dirs):
        m = json.loads((d / 'manifest.json').read_text())
        if not m['complete'] or m['games'] != ['co_chat2'] or not m['crossed']:
            raise ValueError('expected complete co_chat2 link/cut/crossed shard')
        if file_hash(d / 'episodes.jsonl') != m['episodes_sha256']:
            raise ValueError('shard hash mismatch')
        rows = [json.loads(line) for line in (d / 'episodes.jsonl').read_text().splitlines()]
        chateval.validate(rows, dict(m, chat_plans=m['chat_plans_by_game']['co_chat2']))
        manifests.append(m)
        records += rows
    if not manifests:
        raise ValueError('at least one shard required')
    for key in ('run', 'seed', 'n', 'temperature', 'code_hashes'):
        if any(m[key] != manifests[0][key] for m in manifests):
            raise ValueError('shards disagree on ' + key)
    keys = [(r['episode_index'], r['receiver']) for r in records]
    expected = {(i, x) for i in range(manifests[0]['n']) for x in SIDES}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('duplicate or missing shard records')
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--after', required=True, nargs='+')
    ap.add_argument('--before', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    result = {}
    for label, dirs in (('before', args.before), ('after', args.after)):
        if not dirs:
            continue
        result[label] = chateval.summary_rows(load(dirs))
        for row in result[label]:
            if row['receiver'] == 'dialogue':
                continue
            counts = ' '.join(f"{c}={v['successes']}/{v['n']}" for c, v in row['names_counts'].items())
            tests = ' '.join(f"{k} +{row[k]['sign']['positive']} -{row[k]['sign']['negative']} "
                             f"p={row[k]['sign']['p_greater']:.6g}" for k in ('link-cut', 'link-crossed'))
            print(f"{label} {row['receiver']} {row['subset']} {counts} {tests}")
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
