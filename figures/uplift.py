"""Does the link add capability? ARC-Challenge and MMLU test questions (pairing/bench_eval.py). Top: the percent
answered right by each way of answering: each model alone after the same question training, the two models with their
letter probabilities averaged, the two models answering in turn in text, the trained pair with the link cut, and
linked. Below: the numbers of questions on which the linked pair and the averaged models disagree, by which is right.
All counts leave out the 31 test questions whose input also appears in the training questions
(results/magic-nl28/bench_audited.json, exclude.exact_input_overlap), which leaves 3,134.

Reads results/magic-nl23/bench.json (per-question records; no model) and the audit, and writes
figures/uplift.{svg,png,pdf}.
Usage: python figures/uplift.py [BENCH_JSON]
"""
import json
import sys
from pathlib import Path

from chat_names import COOL, INK, LINK, SVG, WARM, mascot, pair_icon, width
from steered_depth import lines

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'results/magic-nl23/bench.json'
AUDIT = HERE.parent / 'results/magic-nl28/bench_audited.json'
OTHER = {'quill': 'ember', 'ember': 'quill'}
TITLE = ('Linked, the twins answer fewer exam questions right', 'than Quill and Ember averaged')


def mean(v):
    return sum(v) / len(v)


def audited(R, audit=AUDIT):
    """R without the test questions whose input also appears in the training questions."""
    drop = set(json.load(open(audit))['exclude']['exact_input_overlap'])
    keep = [i for i in range(len(R['questions'])) if i not in drop]
    return dict(R, questions=[R['questions'][i] for i in keep],
                per_question={k: [v[i] for i in keep] for k, v in R['per_question'].items()})


def arms(R):
    """(key, label, icon kind) of each way of answering, the linked pair last."""
    D = R['answerer']
    return [('solo/quill', 'Quill alone', 'quill'), ('solo/ember', 'Ember alone', 'ember'),
            ('ensemble', 'Quill and Ember\naveraged', 'both'), (f'text/{OTHER[D]}-first/{D}', 'Talking in text', 'text'),
            (f'pair/zero/{D}', 'Link cut', 'cut'), (f'pair/{D}', 'Linked', 'link')]


def icon_draw(s, x, y, size, kind, uris):
    if kind in ('quill', 'ember'):
        s.image(x, y, size, uris[kind])
    elif kind == 'both':
        s.image(x, y, size, uris['quill'])
        s.image(x + size + 10, y, size, uris['ember'])
    elif kind == 'text':
        s.image(x, y, size, uris['quill'])
        s.image(x + size + 86, y, size, uris['ember'])
        for k in range(3):
            s.circle(x + size + 25 + 18 * k, y + size * 0.62, 5, INK)
    else:
        pair_icon(s, x, y, size, kind, uris)


def draw(R, stem):
    C = R['per_question']
    W, M, G, P = 2000, 30, 30, 24
    top, row_h, icon, count_h = 180, 140, 100, 250
    bars_h = 6 * row_h + 2 * P
    H = top + bars_h + G + count_h + M
    s = SVG(W, H)
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    for k, line in enumerate(TITLE):
        s.text(M, 76 + 70 * k, line, 56, 700)

    s.rect(M, top, W - 2 * M, bars_h, COOL, 32)
    lx = M + P + 2 * icon + 86 + 30
    bx = lx + 400
    bw = W - M - P - 200 - bx
    for j, (key, label, kind) in enumerate(arms(R)):
        y0 = top + P + j * row_h
        icon_draw(s, M + P, y0 + (row_h - icon) / 2, icon, kind, uris)
        color = LINK if kind == 'link' else INK
        text = lines(label, bx - lx - 20)
        for i, line in enumerate(text):
            s.text(lx, y0 + (row_h - 56 * len(text)) / 2 + 42 + 56 * i, line, 46, 700, color)
        acc = mean(C[key])
        s.rect(bx, y0 + (row_h - 70) / 2, bw * acc, 70, color, 12)
        s.text(bx + bw * acc + 20, y0 + row_h / 2 + 17, f'{100 * acc:.1f}%', 46, 700, color)

    D = R['answerer']
    pair, ens = C[f'pair/{D}'], C['ensemble']
    sides = ((sum(a and not b for a, b in zip(pair, ens)), LINK, 'only when linked'),
             (sum(b and not a for a, b in zip(pair, ens)), INK, 'only when averaged'))
    cy = top + bars_h + G
    s.rect(M, cy, W - 2 * M, count_h, WARM, 32)
    half = (W - 2 * M) / 2
    for k, (n, color, label) in enumerate(sides):
        x = M + 44 + k * half
        s.text(x, cy + count_h / 2 + 34, str(n), 96, 700, color)
        tx = x + width(str(n), 96, 700) + 30
        s.text(tx, cy + count_h / 2 - 8, 'questions right', 46)
        s.text(tx, cy + count_h / 2 + 48, label, 46)
    s.save(stem)


def main():
    R = audited(json.load(open(sys.argv[1] if len(sys.argv) > 1 else BENCH)))
    draw(R, HERE / 'uplift')
    print(HERE / 'uplift.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
