"""Does the link add capability? ARC-Challenge + MMLU test questions (bench_eval.py). Left: the share answered right by
each way of answering: each model alone after the same question training, the two alone with their letter probabilities
averaged, the pair answering in turn in text, the trained pair with the link cut, and linked. Right: the questions on
which the linked pair and the two models averaged disagree, one mark per question, under the one that gets it right.

Reads results/magic-nl23/bench.json (per-question records; no model, no torch) and writes figures/uplift.{svg,png,pdf}.
Usage: python figures/uplift.py [BENCH_JSON]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, FAINT, INK, LINK, SVG, WARM, mascot, pair_icon  # noqa: E402

BENCH = HERE.parent / 'results/magic-nl23/bench.json'
OTHER = {'quill': 'ember', 'ember': 'quill'}


def mean(v):
    return sum(v) / len(v)


def arms(R):
    """(key, label, icon kind) of each way of answering, the linked pair last."""
    D = R['answerer']
    return [('solo/quill', 'Quill alone', 'quill'), ('solo/ember', 'Ember alone', 'ember'),
            ('ensemble', 'Both, averaged', 'both'), (f'text/{OTHER[D]}-first/{D}', 'Talking in text', 'text'),
            (f'pair/zero/{D}', 'Link cut', 'cut'), (f'pair/{D}', 'Linked', 'link')]


def icon(s, x, y, size, kind, uris):
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
    W, M = 2400, 36
    top, row_h = 128, 150
    H = top + 6 * row_h + 60 + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'))
    s.text(M, 90, 'Linked, the twins answer no more exam questions right', 68, 700)

    lw = 1500
    s.rect(M, top, lw, 6 * row_h + 60, COOL, 32)
    bx, bw = M + 700, 540
    for j, (key, label, kind) in enumerate(arms(R)):
        y0 = top + 30 + j * row_h
        icon(s, M + 30, y0 + 20, 96, kind, uris)
        color = LINK if kind == 'link' else INK
        s.text(M + 330, y0 + 84, label, 44, 700, color)
        acc = mean(C[key])
        s.rect(bx, y0 + 44, bw, 54, 'white', 14)
        s.rect(bx, y0 + 44, bw * acc, 54, color, 14)
        s.text(bx + bw + 30, y0 + 88, f'{100 * acc:.1f}%', 52, 700, color)

    D = R['answerer']
    pair, ens = C[f'pair/{D}'], C['ensemble']
    sides = (('Linked', LINK, sum(a and not b for a, b in zip(pair, ens))),
             ('Both, averaged', INK, sum(b and not a for a, b in zip(pair, ens))))
    rx, rw = M + lw + 30, W - 2 * M - lw - 30
    s.rect(rx, top, rw, 6 * row_h + 60, WARM, 32)
    s.text(rx + 40, top + 76, 'Questions only one gets right', 44, 700)
    cols, pitch, r = 24, 31, 11.5
    y = top + 150
    for label, color, n in sides:
        s.text(rx + 40, y + 50, label, 44, 700, color)
        s.text(rx + rw - 40, y + 54, f'{n}', 60, 700, color, 'end')
        for i in range(n):
            s.circle(rx + 40 + r + (i % cols) * pitch, y + 100 + (i // cols) * pitch, r, color)
        y += 100 + ((n - 1) // cols + 1) * pitch + 60
    s.save(stem)


def main():
    R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else BENCH))
    draw(R, HERE / 'uplift')
    print(HERE / 'uplift.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
