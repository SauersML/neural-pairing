"""Merge complete chat2 shards and print counts and paired exact tests by kind.

Unlike the legacy merger, require every index from all requested shards and
verify completed manifests/hashes before summarizing. chat_merge.py also prints
the same per-kind rows, with its existing permissive partial-input behavior.
--code_ok FILE lets shards differ in that source file's hash only (a rerun shard with a fix that loosens a check the
other shards passed); the output records every shard's hash for each such file.
"""
import argparse
import json
from pathlib import Path

import chateval
from evalfree import file_hash
from link import SIDES


def load(dirs, code_ok=()):
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
    for key in ('run', 'seed', 'n', 'temperature'):
        if any(m[key] != manifests[0][key] for m in manifests):
            raise ValueError('shards disagree on ' + key)
    strict = [{f: h for f, h in m['code_hashes'].items() if f not in code_ok} for m in manifests]
    if any(h != strict[0] for h in strict):
        raise ValueError('shards disagree on code_hashes')
    for key in ('frozen', 'receivers'):
        if any(m.get(key) != manifests[0].get(key) for m in manifests):
            raise ValueError('shards disagree on ' + key)
    keys = [(r['episode_index'], r['receiver']) for r in records]
    expected = {(i, x) for i in range(manifests[0]['n']) for x in manifests[0].get('receivers', SIDES)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('duplicate or missing shard records')
    return records, {f: [m['code_hashes'].get(f) for m in manifests] for f in code_ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--after', required=True, nargs='+')
    ap.add_argument('--before', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--code_ok', nargs='*', default=[], help='source files whose hash may differ between shards')
    args = ap.parse_args()
    result = {}
    for label, dirs in (('before', args.before), ('after', args.after)):
        if not dirs:
            continue
        records, hashes = load(dirs, args.code_ok)
        result[label] = chateval.summary_rows(records)
        if hashes:
            result[label + '_code_ok'] = dict(dirs=dirs, hashes=hashes)
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
