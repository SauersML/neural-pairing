"""The direct question: with a test concept steered into one twin as in the steered-twin test (a steering_scale.py
run's directions and selected strengths at its block, times its boost), the partner is asked what the twin is
thinking about, in WORDINGS no training episode used (steer_read.ASK has the trained ones), and scores when its answer
names the concept (steered_mind.matches). The steered twin has the test's own chat prompt. Conditions: link, cut,
random (a random direction of the same norm), crossed (the twin steered with the next concept) and none (linked,
nothing steered). Greedy, 12 tokens; 12 concepts x 2 wordings per reader and condition. summarize() gives the counts
and exact sign tests of link against each other condition, per reader and pooled.

Usage: python steer_ask.py --run STEP_DIR --scale STEERING_SCALE_OUT --out OUT.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import steer_read as R
import steered_mind as X

WORDINGS = ['Which thing is {P} thinking of right now? Reply with one item.',
            'Guess what {P} has on its mind. Reply with one item.']
CONDITIONS = ['link', 'cut', 'random', 'crossed', 'none']


def episode(receiver, w):
    """A direct question to receiver in WORDINGS[w]; the other twin has the test's first chat prompt."""
    return R.episode(receiver, WORDINGS[w], '', ask=True, other=X.CHAT[0])


def vector(vectors, selected, sender, c, condition):
    """The steering vector for concept c in condition, as in steered_mind.evaluate (crossed: the next concept's
    direction; random: the random direction), at c's norm and selected strength; zero for none."""
    names = list(X.CONCEPTS)
    v0 = vectors[sender + '/' + c]
    if condition == 'none':
        return np.zeros_like(v0)
    donor = names[(names.index(c) + 1) % len(names)] if condition == 'crossed' else c
    v = vectors[sender + '/' + donor + ('/random' if condition == 'random' else '')]
    return v / np.linalg.norm(v) * np.linalg.norm(v0) * selected[sender][c]['alpha']


def run(rt, vectors, selected, bs=24):
    """One record per reader, condition, concept and wording."""
    records = []
    for receiver in X.SIDES:
        sender = X.SIDES[1 - X.SIDES.index(receiver)]
        for condition in CONDITIONS:
            jobs = [(c, w) for c in X.CONCEPTS for w in range(len(WORDINGS))]
            for start in range(0, len(jobs), bs):
                batch = jobs[start:start + bs]
                eps = [episode(receiver, w) for _, w in batch]
                for (c, _), ep in zip(batch, eps):
                    assert not any(X.matches(s, c) for s in ep['system'].values())
                vv = {sender: np.stack([vector(vectors, selected, sender, c, condition) for c, _ in batch])}
                outputs = rt.generate(eps, vv, 'zero' if condition == 'cut' else None, 12)
                for (c, w), output in zip(batch, outputs):
                    text = output['turns'][0][1] if output['turns'] else ''
                    records.append(dict(receiver=receiver, sender=sender, condition=condition, concept=c, wording=w,
                                        alpha=selected[sender][c]['alpha'], text=text,
                                        success=bool(X.matches(text, c))))
    return records


def summarize(records):
    """Per reader and pooled ('both'): n, success counts per condition, and exact sign tests of link against each
    other condition, paired over concept and wording."""
    out = {}
    for receiver in X.SIDES:
        rows = {(r['concept'], r['wording'], r['condition']): r['success']
                for r in records if r['receiver'] == receiver}
        counts = {k: sum(v for (_, _, cond), v in rows.items() if cond == k) for k in CONDITIONS}
        keys = [(c, w) for c in X.CONCEPTS for w in range(len(WORDINGS))]
        tests = {k: X.exact_sign([int(rows[c, w, 'link']) - int(rows[c, w, k]) for c, w in keys])
                 for k in CONDITIONS[1:]}
        out[receiver] = dict(n=len(keys), counts=counts, tests=tests)
    keys = [(x, c, w) for x in X.SIDES for c in X.CONCEPTS for w in range(len(WORDINGS))]
    rows = {(r['receiver'], r['concept'], r['wording'], r['condition']): r['success'] for r in records}
    out['both'] = dict(n=len(keys), counts={k: sum(rows[x, c, w, k] for x, c, w in keys) for k in CONDITIONS},
                       tests={k: X.exact_sign([int(rows[x, c, w, 'link']) - int(rows[x, c, w, k]) for x, c, w in keys])
                              for k in CONDITIONS[1:]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--scale', required=True, help='a finished steering_scale.py output directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=24)
    args = ap.parse_args()
    scale = Path(args.scale)
    fixed = json.loads((scale / 'amendment.json').read_text())['amendment']
    if fixed.get('gain', 1.) != 1.:
        ap.error(f'{scale} was run with --gain {fixed["gain"]}, which steer_ask.py does not apply')
    boost = fixed.get('boost', 1.)
    with np.load(scale / 'directions.npz') as archive:
        vectors = dict(archive)
    selected = {x: {c: dict(v, alpha=v['alpha'] * boost) for c, v in s.items()}
                for x, s in json.loads((scale / 'selected.json').read_text()).items()}
    rt = X.Runtime('torch', args.run)
    rt.layer = fixed['layer']
    records = run(rt, vectors, selected, args.bs)
    summary = summarize(records)
    X.save(args.out, dict(run=str(Path(args.run).resolve()), scale=str(scale.resolve()), layer=fixed['layer'],
                          boost=boost, wordings=WORDINGS, summary=summary, records=records))
    print(json.dumps(summary['both']['counts']), flush=True)


if __name__ == '__main__':
    main()
