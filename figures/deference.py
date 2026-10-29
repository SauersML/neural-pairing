"""Exam questions (pairing/bench_eval.py, ARC-Challenge and MMLU test questions) that exactly one of the two models
answers right with the trained pair's link cut: linked, Ember answers right where Quill with the link cut is right and
wrong where Quill with the link cut is wrong. All counts leave out the 31 test questions whose input also appears in
the training questions (uplift.audited), which leaves 3,134.

Top left, one real test question of round 1: among the questions only Ember answers right with the link cut that
linked Ember and linked Quill both answer wrong, the shortest (question and choices). bench.json records only right or
wrong per question, so the wrong answers are drawn without their letter. Top right, round 1's counts: of the questions
only Quill answers right with the link cut, those linked Ember answers right; of those only Ember answers right with
the link cut, those linked Ember answers wrong. Bottom, the same counts in round 2 (nl27: half of the training questions are ones only one original
model answers right) and round 3 (nl28: round 2 trained on the two models' averaged letters); round 1 is nl23.

Reads the rounds' bench.json (per-question records; no model) and the test questions from data/bench.json (written by
python pairing/build_bench.py from the public datasets; checked against each bench.json's sources and answers), and
writes figures/deference.{svg,png,pdf}.
Usage: python figures/deference.py [--questions data/bench.json] [BENCH_JSON ...]
       (default: results/magic-nl23, magic-nl27 and magic-nl28)
"""
import argparse
import json
from pathlib import Path

from chat_names import COOL, EMBER, INK, QUILL, RED, SVG, WARM, count, pair_icon, width, wrap
from steered_depth import lines
from uplift import audited

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = [ROOT / f'results/magic-nl{n}/bench.json' for n in (23, 27, 28)]
LETTERS = 'ABCD'
ROUNDS = ('Round 1', 'Round 2, half the questions only one twin answers right',
          'Round 3, trained on the twins’ averaged answer')
GROUPS = (('Only Quill right with the link cut,', 'linked Ember right', EMBER),
          ('Only Ember right with the link cut,', 'linked Ember wrong', RED))
TITLE = ('Linked, Ember answers wrong where Quill answers wrong',
         'with the link cut, even when Ember is right with the link cut,',
         'until the twins train on their averaged answer')
LABEL = 800                                         # width of the bottom card's label column


def groups(R):
    """[(questions, linked Ember's matching answers)]: of the questions only Quill answers right with the link cut,
    those linked Ember answers right; of those only Ember answers right, those linked Ember answers wrong."""
    C = R['per_question']
    zq, ze, pe = C['pair/zero/quill'], C['pair/zero/ember'], C['pair/ember']
    quill = [i for i in range(len(zq)) if zq[i] and not ze[i]]
    ember = [i for i in range(len(zq)) if ze[i] and not zq[i]]
    return [(len(quill), sum(pe[i] for i in quill)), (len(ember), sum(not pe[i] for i in ember))]


def example(R, items):
    """The shortest question only Ember answers right with the link cut that both linked models answer wrong, and the
    four answers: {(condition, twin): right}."""
    C = R['per_question']
    keys = {('cut', 'quill'): 'pair/zero/quill', ('cut', 'ember'): 'pair/zero/ember', ('link', 'quill'): 'pair/quill',
            ('link', 'ember'): 'pair/ember'}
    pick = [i for i in range(len(items)) if C['pair/zero/ember'][i] and not C['pair/zero/quill'][i]
            and not C['pair/ember'][i] and not C['pair/quill'][i]]
    i = min(pick, key=lambda i: len(items[i]['q']) + sum(map(len, items[i]['choices'])))
    return items[i], {k: C[key][i] for k, key in keys.items()}


def draw(runs, item, answers, stem):
    W, M, G, P = 2000, 40, 30, 30
    left_w = 960
    right_w = W - 2 * M - G - left_w
    pad, line_h, icon, cell_w = 28, 58, 72, 230
    q_lines = wrap(item['q'], left_w - 2 * P - 2 * pad, 46)
    c_lines = [(f'{L}. {c}', k == item['answer']) for k, (L, c) in enumerate(zip(LETTERS, item['choices']))]
    bub_h = 2 * pad + line_h * (len(q_lines) + len(c_lines) - 1) + 46
    row_h = 104
    grid_h = 70 + 2 * row_h
    top_h = P + bub_h + 30 + grid_h + P
    heads = [lines(r, (W - 2 * M - 2 * P - LABEL - 2 * G) / 2) for r in ROUNDS[1:len(runs)]]
    low_head = 56 * max(map(len, heads)) + 24
    low_row = 140
    low_h = P + low_head + 2 * low_row + P
    top = 48 + 68 * len(TITLE)
    H = top + top_h + G + low_h + M
    s = SVG(W, H)
    for k, line in enumerate(TITLE):
        s.text(M, 76 + 68 * k, line, 56, 700)

    # one real question
    s.rect(M, top, left_w, top_h, WARM, 32)
    x, y = M + P, top + P
    s.rect(x, y, left_w - 2 * P, bub_h, 'white', 24, INK, 3)
    for k, line in enumerate(q_lines):
        s.text(x + pad, y + pad + 34 + line_h * k, line, 46)
    for k, (line, right) in enumerate(c_lines):
        s.text(x + pad, y + pad + 34 + line_h * (len(q_lines) + k), line, 46, 700 if right else 400)
    y += bub_h + 30
    head_w = left_w - 2 * P - 2 * cell_w
    for k, (name, color) in enumerate((('Quill', QUILL), ('Ember', EMBER))):
        s.text(x + head_w + k * cell_w + cell_w / 2, y + 46, name, 46, 700, color, 'middle')
    y += 70
    for cond, label in (('cut', 'Link cut'), ('link', 'Linked')):
        pair_icon(s, x, y + (row_h - icon) / 2, icon, cond)
        s.text(x + 2 * icon + 86 + 20, y + row_h / 2 + 16, label, 46, 700)
        for k, twin in enumerate(('quill', 'ember')):
            right = answers[(cond, twin)]
            text = f'{LETTERS[item["answer"]]}, right' if right else 'wrong'
            cw = width(text, 46, 700) + 40
            cx = x + head_w + k * cell_w + (cell_w - cw) / 2
            s.rect(cx, y + (row_h - 72) / 2, cw, 72, 'white', 16)
            s.text(cx + 20, y + row_h / 2 + 16, text, 46, 700, INK if right else RED)
        y += row_h

    # round 1 counts
    xr = M + left_w + G
    s.rect(xr, top, right_w, top_h, WARM, 32)
    s.text(xr + P, top + P + 46, ROUNDS[0], 46, 700)
    block = (top_h - 2 * P - 70) / 2
    for k, ((a, b, color), (n, hits)) in enumerate(zip(GROUPS, groups(runs[0]))):
        by = top + P + 70 + k * block
        s.text(xr + P, by + 60, a, 46, 700)
        s.text(xr + P, by + 116, b + ' on', 46, 700)
        count(s, xr + P, by + 210, hits, n, color, size=96, small=56)

    # rounds 2 and 3
    y = top + top_h + G
    s.rect(M, y, W - 2 * M, low_h, COOL, 32)
    cols = [M + P + LABEL + G + k * ((W - 2 * M - 2 * P - LABEL - 2 * G) / 2 + G) for k in range(len(runs) - 1)]
    cw = (W - 2 * M - 2 * P - LABEL - 2 * G) / 2
    for x0, head in zip(cols, heads):
        for k, line in enumerate(head):
            s.text(x0 + cw / 2, y + P + 42 + 56 * k, line, 46, 700, INK, 'middle')
    y += P + low_head
    for k, (a, b, color) in enumerate(GROUPS):
        ry = y + k * low_row
        s.text(M + P, ry + low_row / 2 - 6, a, 46, 700)
        s.text(M + P, ry + low_row / 2 + 50, b, 46, 700)
        for x0, R in zip(cols, runs[1:]):
            n, hits = groups(R)[k]
            count(s, x0 + cw / 2, ry + low_row / 2 + 40, hits, n, color, anchor='middle')
    s.save(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default=ROOT / 'data/bench.json')
    ap.add_argument('bench', nargs='*', default=BENCH)
    args = ap.parse_args()
    items = audited(dict(questions=json.load(open(args.questions))['test'], per_question={}))['questions']
    runs = [audited(json.load(open(p))) for p in args.bench]
    for R in runs:
        if [(q['src'], q['answer']) for q in R['questions']] != [(q['src'], q['answer']) for q in items]:
            raise ValueError(f'{args.questions} does not match the questions of the bench runs')
    draw(runs, *example(runs[0], items), HERE / 'deference')
    print(HERE / 'deference.png')


if __name__ == '__main__':
    main()
