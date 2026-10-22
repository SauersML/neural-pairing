"""What the link does on the exam questions (bench_eval.py, ARC-Challenge + MMLU test questions): Ember, reading Quill,
takes Quill's answer. One mark per question that exactly one of the two models answers right with the trained pair's
link cut, filled when Ember answers it right. Rows: link cut, linked. Columns: the questions only Quill gets right with
the link cut, and those only Ember gets right.

Reads results/magic-nl23/bench.json (per-question records; no model, no torch) and writes
figures/deference.{svg,png,pdf}.
Usage: python figures/deference.py [BENCH_JSON]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, FAINT, INK, SVG, WARM, mascot, pair_icon  # noqa: E402

BENCH = HERE.parent / 'results/magic-nl23/bench.json'
ROWS = (('cut', 'Link cut', 'pair/zero/ember'), ('link', 'Linked', 'pair/ember'))


def groups(R):
    """(title, side, question indices): the questions only Quill, then only Ember, answers right with the link cut."""
    C = R['per_question']
    zq, ze = C['pair/zero/quill'], C['pair/zero/ember']
    n = len(zq)
    return [('Only Quill knew', 'quill', [i for i in range(n) if zq[i] and not ze[i]]),
            ('Only Ember knew', 'ember', [i for i in range(n) if ze[i] and not zq[i]])]


def draw(R, stem):
    W, M = 2400, 36
    top, head_h, row_h, key_h = 128, 120, 470, 96
    H = top + head_h + 2 * row_h + 14 + 24 + key_h + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'))
    s.text(M, 90, "Linked, Ember takes Quill's answer, right or wrong", 68, 700)

    label_w, gap = 520, 22
    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    panel_h = head_h + 2 * row_h + 14
    gs = groups(R)
    for k, ((title, side, _), fill) in enumerate(zip(gs, (COOL, WARM))):
        s.rect(px[k], top, pw, panel_h, fill, 32)
        s.image(px[k] + 34, top + 22, 84, uris[side])
        s.text(px[k] + 134, top + 80, title, 48, 700)

    cols, pitch, r = 32, 26, 10
    for j, (cond, label, key) in enumerate(ROWS):
        y0 = top + head_h + j * row_h
        if j:
            for k in range(2):
                s.line(px[k] + 30, y0, px[k] + pw - 30, y0, '#E3DED6' if k else '#E2E2E7', 2)
            s.line(M, y0, M + label_w - 10, y0, '#E2E2E7', 2)
        msize = 124
        iy = y0 + 60
        pair_icon(s, M + 14, iy, msize, cond, uris)
        s.text(M + 14 + (2 * msize + 86) / 2, iy + msize + 56, label, 42, 700, INK, 'middle')
        for k, (_, _, idx) in enumerate(gs):
            right = sorted((R['per_question'][key][i] for i in idx), reverse=True)
            hits = sum(right)
            cx0 = px[k] + (pw - (cols - 1) * pitch - 2 * r) / 2
            s.rich(cx0, y0 + 76, [(f'{hits}', EMBER if hits else INK, 700), (f' of {len(right)}', INK, 400)], 56)
            for i, v in enumerate(right):
                cx, cy = cx0 + r + (i % cols) * pitch, y0 + 120 + (i // cols) * pitch
                if v:
                    s.circle(cx, cy, r, EMBER)
                else:
                    s.circle(cx, cy, r - 1.5, 'white', FAINT, 3)

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    y = ky + key_h / 2 + 13
    s.circle(M + 61, y - 12, 17, EMBER)
    s.text(M + 94, y, 'Ember right', 36)
    s.circle(M + 391, y - 12, 15.5, 'white', FAINT, 3)
    s.text(M + 424, y, 'Ember wrong', 36)
    s.save(stem)


def main():
    R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else BENCH))
    draw(R, HERE / 'deference')
    print(HERE / 'deference.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
