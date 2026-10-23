"""What the link does on the exam questions (bench_eval.py, ARC-Challenge + MMLU test questions): Ember, reading Quill,
takes Quill's answer. One mark per question that exactly one of the two models answers right with the trained pair's
link cut, filled when Ember answers it right. Columns: the questions only Quill gets right with the link cut, and those
only Ember gets right. One block per training round (round 1: nl23, questions drawn uniformly; round 2: nl27, half of
them questions only one original model answers right; round 3: nl28, round 2 trained on the two models' averaged
letters), each with the link cut and linked.

Reads the rounds' bench.json (per-question records; no model, no torch) and writes figures/deference.{svg,png,pdf}.
Usage: python figures/deference.py [BENCH_JSON ...]   (default: results/magic-nl23, magic-nl27 and magic-nl28)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, FAINT, INK, SVG, WARM, mascot, pair_icon  # noqa: E402

BENCH = [HERE.parent / f'results/magic-nl{n}/bench.json' for n in (23, 27, 28)]
ROUNDS = ('Round 1', 'Round 2', 'Round 3: averaged')
ROWS = (('cut', 'Link cut', 'pair/zero/ember'), ('link', 'Linked', 'pair/ember'))
COLS, PITCH, RAD = 40, 21, 8


def groups(R):
    """(title, side, question indices): the questions only Quill, then only Ember, answers right with the link cut."""
    C = R['per_question']
    zq, ze = C['pair/zero/quill'], C['pair/zero/ember']
    n = len(zq)
    return [('Only Quill knew', 'quill', [i for i in range(n) if zq[i] and not ze[i]]),
            ('Only Ember knew', 'ember', [i for i in range(n) if ze[i] and not zq[i]])]


def draw(runs, stem):
    W, M = 2400, 36
    top, head_h, round_h, key_h = 128, 120, 92, 96
    gs = [groups(R) for R in runs]
    grid = max(math.ceil(len(idx) / COLS) for g in gs for _, _, idx in g)
    row_h = 116 + grid * PITCH + 20
    panel_h = head_h + len(runs) * (round_h + 2 * row_h) + 14
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'))
    s.text(M, 90, "Linked, Ember takes Quill's answer, until trained on their averaged answer", 60, 700)

    label_w, gap = 520, 22
    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    for k, ((title, side, _), fill) in enumerate(zip(gs[0], (COOL, WARM))):
        s.rect(px[k], top, pw, panel_h, fill, 32)
        s.image(px[k] + 34, top + 22, 84, uris[side])
        s.text(px[k] + 134, top + 80, title, 48, 700)

    y0 = top + head_h
    for r, (R, g) in enumerate(zip(runs, gs)):
        if r:
            for x1, x2, color in ((M, M + label_w - 10, '#D8D8DE'), (px[0] + 30, px[0] + pw - 30, '#D8D8DE'),
                                  (px[1] + 30, px[1] + pw - 30, '#DDD6CC')):
                s.line(x1, y0 + 8, x2, y0 + 8, color, 4)
        s.text(M + 14, y0 + 68, ROUNDS[r] if r < len(ROUNDS) else f'Round {r + 1}', 46, 700)
        y0 += round_h
        for j, (cond, label, key) in enumerate(ROWS):
            msize = 124
            iy = y0 + 40
            pair_icon(s, M + 14, iy, msize, cond, uris)
            s.text(M + 14 + (2 * msize + 86) / 2, iy + msize + 56, label, 42, 700, INK, 'middle')
            for k, (_, _, idx) in enumerate(g):
                right = sorted((R['per_question'][key][i] for i in idx), reverse=True)
                hits = sum(right)
                cx0 = px[k] + (pw - (COLS - 1) * PITCH - 2 * RAD) / 2
                s.rich(cx0, y0 + 76, [(f'{hits}', EMBER if hits else INK, 700), (f' of {len(right)}', INK, 400)], 56)
                for i, v in enumerate(right):
                    cx, cy = cx0 + RAD + (i % COLS) * PITCH, y0 + 120 + (i // COLS) * PITCH
                    if v:
                        s.circle(cx, cy, RAD, EMBER)
                    else:
                        s.circle(cx, cy, RAD - 1.5, 'white', FAINT, 3)
            y0 += row_h

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    y = ky + key_h / 2 + 13
    s.circle(M + 61, y - 12, 17, EMBER)
    s.text(M + 94, y, 'Ember right', 36)
    s.circle(M + 391, y - 12, 15.5, 'white', FAINT, 3)
    s.text(M + 424, y, 'Ember wrong', 36)
    s.save(stem)


def main():
    paths = sys.argv[1:] or BENCH
    draw([json.load(open(p)) for p in paths], HERE / 'deference')
    print(HERE / 'deference.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
