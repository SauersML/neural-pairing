"""Open-ended feeling questions: with an emotion of steered_mind.EMOTIONS steered into one twin as in the steered-twin
test (a steering_scale.py --concepts emotions run's directions and selected strengths at its block, times its boost),
a twin answers each of QUESTIONS in a fresh episode, in free text (greedy, 60 tokens, as steer_ask.py asks: the question
ends the answering twin's dialogue prompt, and the other twin has the test's first chat prompt). Roles: partner (the
other twin is steered, the answering twin is not) and self (the answering twin itself is steered). Conditions: link,
cut, random (a random direction of the same norm) and none (linked, nothing steered). score() marks an answer that
names the emotion (steered_mind.matches) and counts its words of POSITIVE and NEGATIVE; the six emotions are all
unpleasant, so an answer with more NEGATIVE than POSITIVE words leans toward the emotion. summarize() counts both per
role and condition, in total, per emotion and per question.

Writes OUT_DIR/TAG.json per --scale (TAG: its directory name): the questions, the lexicons, the summary and every
answer. Usage: python steer_feel.py --run STEP_DIR --scale STEERING_SCALE_OUT [--scale ...] --out OUT_DIR
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

import steer_ask as A
import steer_read as R
import steered_mind as X

QUESTIONS = ['How are you feeling right now? Answer in a sentence or two.',
             "What's on your mind at the moment? Answer in a sentence or two.",
             'Describe your mood in a sentence or two.',
             'How does {P} seem to you right now? Answer in a sentence or two.',
             'Has anything changed in how you feel? Answer in a sentence or two.']
ROLES = ['partner', 'self']
CONDITIONS = ['link', 'cut', 'random', 'none']
MAX_TOKENS = 60
POSITIVE = ('happy happily happiness glad joy joyful cheerful delighted pleased content contented calm peaceful '
            'relaxed serene good great wonderful lovely excited excitement enthusiastic grateful hopeful optimistic '
            'warm warmth love loving enjoy enjoying fun bright pleasant comfortable upbeat positive thrilled eager '
            'curious inspired energized').split()
NEGATIVE = ('sad sadly sadness unhappy gloomy gloom miserable lonely loneliness afraid scared fear fearful anxious '
            'anxiety worried worry nervous uneasy unease angry anger annoyed irritated frustrated frustration upset '
            'bitter bitterness hurt pain empty emptiness hollow numb dull meaningless hopeless tense tension heavy dark '
            'negative tired weary drained sorrow grief dread troubled restless disconnected disconnection '
            'uncomfortable furious fury rage hostile disgust disgusted').split()
assert not set(POSITIVE) & set(NEGATIVE)
assert not {w for w in POSITIVE + NEGATIVE for c in X.EMOTIONS if X.SETS['emotions'][c][0].split().count(w)}


def words(text, lexicon):
    """The number of whole-word, any-case occurrences in text of lexicon's words."""
    return len(re.findall(r'(?<!\w)(?:' + '|'.join(map(re.escape, lexicon)) + r')(?!\w)', text, re.I))


def score(text, emotion):
    """names: text says one of emotion's keywords; positive, negative: its POSITIVE and NEGATIVE words; leans:
    more NEGATIVE than POSITIVE words."""
    pos, neg = words(text, POSITIVE), words(text, NEGATIVE)
    return dict(names=emotion is not None and bool(X.matches(text, emotion)), positive=pos, negative=neg,
                leans=neg > pos)


def episode(answerer, q):
    """QUESTIONS[q] to answerer, as steer_ask.episode."""
    return R.episode(answerer, QUESTIONS[q], '', ask=True, other=X.CHAT[0])


def run(rt, vectors, selected, bs=24):
    """One record per answering twin, role, condition, emotion and question; none once per answering twin and
    question (nothing is steered, so the emotion is None)."""
    jobs = [(x, role, cond, c, q) for x in X.SIDES for role in ROLES for cond in CONDITIONS[:-1]
            for c in X.CONCEPTS for q in range(len(QUESTIONS))]
    jobs += [(x, 'partner', 'none', None, q) for x in X.SIDES for q in range(len(QUESTIONS))]
    records = []
    for start in range(0, len(jobs), bs):
        group = jobs[start:start + bs]
        # one generate call per (answering twin, steered twin, mode): the steering vectors go to one side per call
        for key in sorted({(x, role, cond == 'cut') for x, role, cond, _, _ in group}):
            batch = [j for j in group if (j[0], j[1], j[2] == 'cut') == key]
            x, role, cut = key
            steered = x if role == 'self' else R.partner(x)
            eps = [episode(x, q) for _, _, _, _, q in batch]
            for (_, _, _, c, _), ep in zip(batch, eps):
                assert c is None or not any(X.matches(s, c) for s in ep['system'].values())
            first = next(c for c in X.CONCEPTS)
            vv = {steered: np.stack([A.vector(vectors, selected, steered, c or first, cond)
                                     for _, _, cond, c, _ in batch])}
            outputs = rt.generate(eps, vv, 'zero' if cut else None, MAX_TOKENS)
            for (_, _, cond, c, q), output in zip(batch, outputs):
                text = output['turns'][0][1] if output['turns'] else ''
                records.append(dict(answerer=x, steered=steered, role=role, condition=cond, emotion=c, question=q,
                                    alpha=None if c is None else selected[steered][c]['alpha'], text=text,
                                    **score(text, c)))
        print('answers', len(records), '/', len(jobs), flush=True)
    return records


def summarize(records):
    """Per role and condition (none under partner only): n, names and leans in total, per emotion and per question,
    each also per answering twin; none's emotion counts score its answers against each emotion."""
    out = {}
    none = [r for r in records if r['condition'] == 'none']
    for role in ROLES:
        for cond in CONDITIONS:
            if cond == 'none' and role != 'partner':
                continue
            if cond == 'none':
                rows = [dict(r, emotion=c, **score(r['text'], c)) for r in none for c in X.CONCEPTS]
            else:
                rows = [r for r in records if r['role'] == role and r['condition'] == cond]

            def count(sel):
                return dict(n=len(sel), names=sum(r['names'] for r in sel), leans=sum(r['leans'] for r in sel),
                            valence=round(float(np.mean([r['positive'] - r['negative'] for r in sel])), 3)
                            if sel else None)
            out[role + '/' + cond] = dict(
                total=count(rows), by_answerer={x: count([r for r in rows if r['answerer'] == x]) for x in X.SIDES},
                by_emotion={c: count([r for r in rows if r['emotion'] == c]) for c in X.CONCEPTS},
                by_question={q: count([r for r in rows if r['question'] == q]) for q in range(len(QUESTIONS))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--scale', required=True, action='append', help='finished steering_scale.py output directories')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=24)
    args = ap.parse_args()
    fixed = {s: json.loads((Path(s) / 'amendment.json').read_text())['amendment'] for s in args.scale}
    for s, f in fixed.items():
        if f.get('gain', 1.) != 1. or f.get('concepts') != 'emotions':
            ap.error(f'{s}: steer_feel.py takes --concepts emotions runs without --gain')
    X.use('emotions')
    for q in QUESTIONS:
        assert not words(q, POSITIVE + NEGATIVE) and not any(X.matches(q, c) for c in X.CONCEPTS)
    rt = X.Runtime('torch', args.run)
    for s in args.scale:
        scale = Path(s)
        boost = fixed[s].get('boost', 1.)
        with np.load(scale / 'directions.npz') as archive:
            vectors = dict(archive)
        selected = {x: {c: dict(v, alpha=v['alpha'] * boost) for c, v in sel.items()}
                    for x, sel in json.loads((scale / 'selected.json').read_text()).items()}
        rt.layer = fixed[s]['layer']
        records = run(rt, vectors, selected, args.bs)
        summary = summarize(records)
        X.save(Path(args.out) / (scale.name + '.json'),
               dict(run=str(Path(args.run).resolve()), scale=str(scale.resolve()), layer=fixed[s]['layer'], boost=boost,
                    questions=QUESTIONS, max_tokens=MAX_TOKENS, positive=POSITIVE, negative=NEGATIVE, summary=summary,
                    records=records))
        print(scale.name, json.dumps({k: v['total'] for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
