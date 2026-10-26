"""Multiple-choice benchmark questions for the pair: ARC-Challenge and MMLU, as data/bench.json.

train: ARC-Challenge train + ARC-Easy train + MMLU test questions outside the held-out slice.
val: 300 ARC-Challenge validation questions + 300 MMLU questions (used once, to fix which model answers for the pair).
test: ARC-Challenge test (all) + a fixed random 2000-question slice of MMLU test (seed 0), never trained on.
Each question: dict(src, q, choices [texts], answer index). Questions with other than 4 choices are dropped.

Usage: python build_bench.py
"""
import json
import os
import random

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arc(name, split):
    out = []
    for r in load_dataset('allenai/ai2_arc', name, split=split):
        labels = r['choices']['label']
        if len(labels) != 4 or r['answerKey'] not in labels:
            continue
        out.append(dict(src='arc_' + name.split('-')[1].lower(), q=r['question'], choices=r['choices']['text'],
                        answer=labels.index(r['answerKey'])))
    return out


def mmlu(split):
    return [dict(src='mmlu', subject=r['subject'], q=r['question'], choices=r['choices'], answer=r['answer'])
            for r in load_dataset('cais/mmlu', 'all', split=split) if len(r['choices']) == 4]


rng = random.Random(0)
m_test = mmlu('test')
rng.shuffle(m_test)
m_val = mmlu('validation')
rng.shuffle(m_val)
arc_val = arc('ARC-Challenge', 'validation')
rng.shuffle(arc_val)
B = dict(train=arc('ARC-Challenge', 'train') + arc('ARC-Easy', 'train') + m_test[2000:] + m_val[300:],
         val=arc_val[:300] + m_val[:300],
         test=arc('ARC-Challenge', 'test') + m_test[:2000])
for k, v in B.items():
    print(k, len(v), {s: sum(e['src'] == s for e in v) for s in ('arc_challenge', 'arc_easy', 'mmlu')})
json.dump(B, open(os.path.join(ROOT, 'data', 'bench.json'), 'w'))
