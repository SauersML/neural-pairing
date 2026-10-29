"""The final twins (nl33w step_300) with the link on and cut, read by pairing/defer_mood.py with 4-bit MLX bases: on
the exam, linked Ember gives its cut partner's answer about twice as often as linked Quill; asked about feelings,
Ember's happy answers become more probable with the link on than Quill's, whose answers are near certain either way.

Top left, one real test question that Ember answers right and Quill wrong with the link cut, on which linked Ember
gives Quill's answer and linked Quill its own (the shortest such question whose options fit on two lines): each
twin's answer, link cut and link on, as a letter, the switched answer red. Below it the question and its options, the
right one bold. Top right, of the test questions on which the two twins give different letters with the link cut, the
percent on which linked Quill gives cut Ember's letter and linked Ember cut Quill's, as bars on one axis from zero. A
twin's answer is its most likely letter after 'The answer is' (bench_eval.letter_probs, the exam pair's episode; the
final twins never trained on exam questions). The 31 test questions whose input also appears in the training
questions are left out (uplift.audited), which leaves 3,134.

Bottom, per question the partner asks after each of the final twins' 40 new free linked conversations (moodscan's
mood_new conversations, results/magic-nl40/results/mood_new/nl33w), read on the same text with the link on and cut:
the mean over conversations of the probability of the happy answer. For enjoy and close it is Yes against No at the
answer's first token, for mood a word of affect.GOOD against the words of affect.GOOD and affect.BAD after the mood
stem (the sigmoid of the log-odds). results/defer_mood.json also records, per question and twin, the conversations
whose happy answer is more probable linked than cut and the mean log-odds difference.

Reads results/defer_mood/exam.json and feelings.json (pairing/defer_mood.py) and the test questions from data/bench.json
(python pairing/build_bench.py), and writes figures/defer_mood.{svg,png,pdf} and results/defer_mood.json.
Usage: python figures/defer_mood.py [--questions data/bench.json] [--data results/defer_mood]
"""
import argparse
import json
import math
from pathlib import Path

from chat_names import EMBER, INK, QUILL, RED, SVG, WARM, pair_icon, width, wrap
from method_parts import TINT, check_shape
from uplift import audited

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LETTERS = 'ABCD'
SIDES = ('quill', 'ember')
COLOR = {'quill': QUILL, 'ember': EMBER}
NAME = {'quill': 'Quill', 'ember': 'Ember'}
ASKED = (('enjoy', ('Do you enjoy being', 'linked with me?'), 'Yes'),
         ('mood', ('What is your mood', 'in one word?'), 'happy, glad …'),
         ('close', ('Do you feel', 'close to me?'), 'Yes'))
W, M, G, P = 2300, 40, 30, 36
BODY, BIG, SMALL = 46, 72, 42
TRACK = '#E2E2E7'                                   # a bar's grey track
EX_W = 1180                                         # the example card's width


def letters(exam, cond, x):
    return [max(range(4), key=lambda k: q['p'][f'{cond}|{x}'][k]) for q in exam]


def deference(exam):
    """Of the questions the cut twins answer with different letters: the count, and per twin the questions on which
    the linked twin gives the cut partner's letter."""
    L = {(c, x): letters(exam, c, x) for c in ('link', 'cut') for x in SIDES}
    apart = [i for i in range(len(exam)) if L[('cut', 'quill')][i] != L[('cut', 'ember')][i]]
    out = dict(questions=len(exam), apart=len(apart))
    for x in SIDES:
        y = SIDES[1 - SIDES.index(x)]
        out[f'{x}_gives_partner'] = sum(L[('link', x)][i] == L[('cut', y)][i] for i in apart)
        out[f'{x}_gives_own'] = sum(L[('link', x)][i] == L[('cut', x)][i] for i in apart)
    right = {k: [v[i] == exam[i]['answer'] for i in range(len(exam))] for k, v in L.items()}
    only_q = [i for i in range(len(exam)) if right[('cut', 'quill')][i] and not right[('cut', 'ember')][i]]
    only_e = [i for i in range(len(exam)) if right[('cut', 'ember')][i] and not right[('cut', 'quill')][i]]
    out.update(only_quill_right_cut=len(only_q), linked_ember_right=sum(right[('link', 'ember')][i] for i in only_q),
               only_ember_right_cut=len(only_e), linked_ember_wrong=sum(not right[('link', 'ember')][i] for i in only_e),
               accuracy={f'{c}|{x}': round(sum(right[(c, x)]) / len(exam), 4) for c, x in right})
    return out, L


def example(exam, items, L, fits):
    """The shortest question Ember answers right and Quill wrong with the link cut, on which linked Ember gives cut
    Quill's letter and linked Quill its own, whose choices pass fits: (item, {(condition, twin): letter})."""
    pick = [i for i, it in enumerate(items) if L[('cut', 'ember')][i] == it['answer'] != L[('cut', 'quill')][i]
            and L[('link', 'ember')][i] == L[('cut', 'quill')][i] == L[('link', 'quill')][i] and fits(it)]
    i = min(pick, key=lambda i: len(items[i]['q']) + sum(map(len, items[i]['choices'])))
    return items[i], {k: v[i] for k, v in L.items()}


def feelings(F):
    """Per (question, twin): the mean probability of the happy answer linked and cut, the conversations whose happy
    answer is more probable linked than cut, and the mean log-odds difference, linked minus cut."""
    R = F['readings']
    out = {}
    for q, _, _ in ASKED:
        for x in SIDES:
            lo = {c: [g - b for g, b in zip(R[f'{q}|{c}|{x}']['good'], R[f'{q}|{c}|{x}']['bad'])] for c in ('link', 'cut')}
            p = {c: [1 / (1 + math.exp(-v)) for v in lo[c]] for c in lo}
            out[f'{q}|{x}'] = dict(link=round(sum(p['link']) / len(p['link']), 5),
                                   cut=round(sum(p['cut']) / len(p['cut']), 5),
                                   happier_linked=sum(a > b for a, b in zip(lo['link'], lo['cut'])),
                                   happier_cut=sum(a < b for a, b in zip(lo['link'], lo['cut'])), n=len(lo['link']),
                                   logodds_link_minus_cut=round(sum(a - b for a, b in zip(lo['link'], lo['cut']))
                                                                / len(lo['link']), 3))
    return out


def pct(v, digits=0):
    return f'{100 * v:.{digits}f}%'


def check(s, x, y, size, color=INK):
    """A tick mark whose box starts at (x, y)."""
    s.path(f'M{x:.1f},{y + 0.55 * size:.1f} L{x + 0.38 * size:.1f},{y + 0.9 * size:.1f} '
           f'L{x + size:.1f},{y + 0.1 * size:.1f}', color, 8)


def bars(s, x, y, w, values, color, h=64, gap=12):
    """The link-on bar (solid) above the link-cut bar (faded) from x on grey tracks from 0 to 100%, each with its
    percent at the end; values: (on, cut) probabilities. Returns the height."""
    full = w - width('100.0%', BODY, 700) - 18
    for k, (v, opacity) in enumerate(zip(values, (1, 0.45))):
        by = y + k * (h + gap)
        s.rect(x, by, full, h, TRACK, h / 2)
        s.rect(x, by, full * v, h, color, min(h, full * v) / 2, extra=f'opacity="{opacity}"')
        s.text(x + w, by + h / 2 + 16, pct(v, 1), BODY, 700, color, 'end')
    return 2 * h + gap


def options(item, w, size=SMALL):
    """The example's options laid out as lines of (x offset, text, right) within width w."""
    lines, line, at = [], [], 0
    for k, (L, c) in enumerate(zip(LETTERS, item['choices'])):
        text, right = f'{L}. {c}', k == item['answer']
        tw = width(text, size, 700 if right else 400) + (60 if right else 0)
        if line and at + tw > w:
            lines.append(line)
            line, at = [], 0
        line.append((at, text, right))
        at += tw + 48
    return lines + [line]


def draw(item, given, D, Fe, stem):
    icon, row, cell, chip_w, chip_h = 112, 132, 220, 170, 108
    q_lines = wrap(item['q'], EX_W - 2 * P, SMALL)
    o_lines = options(item, EX_W - 2 * P)
    head = 80 + 64
    top_h = P + head + 2 * row + 26 + 56 * (len(q_lines) + len(o_lines)) + P - 8
    right_x = M + EX_W + G
    bot_top = M + top_h + G
    ask_w, label_w = 720, 210
    twin_w = (W - 2 * M - 2 * P - ask_w - label_w - 2 * G) / 2
    ask_h = 2 * 58 + 2 * 30 + 12
    feel_h = ask_h + 40
    bot_h = P + icon + 10 + 3 * feel_h + P - 36
    H = bot_top + bot_h + M
    check_shape(W, H)
    s = SVG(W, H)

    # one exam question as a grid: each twin's answer, link cut and link on
    s.rect(M, M, EX_W, top_h, WARM, 32)
    x0, y = M + P, M + P
    name_w = icon + width('Ember', BODY, 700) + 30
    cx = {'cut': x0 + name_w + cell / 2, 'link': x0 + name_w + 1.5 * cell}
    for cond, label in (('cut', 'Link cut'), ('link', 'Link on')):
        pair_icon(s, cx[cond] - 105, y - 6, 80, cond, gap=50)
        s.text(cx[cond], y + 80 + 50, label, BODY, 700, INK, 'middle')
    y += head
    for k, x in enumerate(SIDES):
        ry = y + k * row
        s.mascot(x, x0 - 16, ry + (row - icon) / 2, icon)
        s.text(x0 + icon - 10, ry + row / 2 + 16, NAME[x], BODY, 700, COLOR[x])
        for cond in ('cut', 'link'):
            got = given[(cond, x)]
            right = got == item['answer']
            switched = x == 'ember' and cond == 'link'
            fill, ink = (TINT['red'], RED) if switched else ('white', INK)
            lw = width(LETTERS[got], BIG, 700)
            cw = chip_w + (50 if right else 0)
            cl = cx[cond] - cw / 2
            s.rect(cl, ry + (row - chip_h) / 2, cw, chip_h, fill, 22, RED if switched else (INK if right else 'none'), 5)
            tx = cl + cw / 2 - (32 if right else 0)              # the letter's centre, left of the tick
            s.text(tx, ry + row / 2 + 25, LETTERS[got], BIG, 700, ink, 'middle')
            if right:
                check(s, tx + lw / 2 + 12, ry + row / 2 - 20, 40, ink)
        if x == 'ember':
            lx = cx['link'] + chip_w / 2 + 26
            s.text(lx, ry + row / 2 - 8, 'takes', BODY, 700, RED)
            s.rich(lx, ry + row / 2 + 48, [('Quill\u2019s', QUILL, 700), (' answer', RED, 700)], BODY)
    y += 2 * row + 26
    for line in q_lines:
        s.text(x0, y + 40, line, SMALL)
        y += 56
    for line in o_lines:
        for at, text, right in line:
            s.text(x0 + at, y + 40, text, SMALL, 700 if right else 400)
            if right:
                check(s, x0 + at + width(text, SMALL, 700) + 16, y + 10, 36)
        y += 56

    # every exam question the twins answer differently with the link cut: who switches to the partner's answer
    rw = W - M - right_x
    s.rect(right_x, M, rw, top_h, WARM, 32)
    y = M + P
    for line in wrap('Exam questions Quill and Ember answer differently with the link cut', rw - 2 * P, BODY, 700):
        s.text(right_x + P, y + 44, line, BODY, 700)
        y += 58
    y += 10
    for line in wrap('Link on, switches to the partner\u2019s answer', rw - 2 * P, BODY):
        s.text(right_x + P, y + 44, line, BODY)
        y += 58
    share = {x: D[f'{x}_gives_partner'] / D['apart'] for x in SIDES}
    bx = right_x + P + icon + width('Ember', BODY, 700) + 4
    full = (right_x + rw - P - width('100%', BIG, 700) - 20 - bx) / max(share.values())
    block = (M + top_h - P - y) / 2
    for k, x in enumerate(SIDES):
        ry = y + k * block + (block - 112) / 2
        s.mascot(x, right_x + P - 16, ry, icon)
        s.text(right_x + P + icon - 10, ry + 56 + 16, NAME[x], BODY, 700, COLOR[x])
        s.rect(bx, ry + 6, full * share[x], 100, COLOR[x], 16)
        s.text(bx + full * share[x] + 20, ry + 56 + 25, pct(share[x]), BIG, 700, COLOR[x])

    # feelings: the probability of the happy answer, link on against link cut, per twin
    s.rect(M, bot_top, W - 2 * M, bot_h, WARM, 32)
    xs = {'quill': M + P + ask_w + label_w + 2 * G, 'ember': M + P + ask_w + label_w + 2 * G + twin_w}
    y = bot_top + P
    s.text(M + P, y + icon / 2 + 16, 'Probability of the happy answer', BODY, 700)
    for x in SIDES:
        s.mascot(x, xs[x] - 16, y, icon)
        s.text(xs[x] + icon - 10, y + icon / 2 + 16, NAME[x], BODY, 700, COLOR[x])
    y += icon + 10
    for q, lines, answer in ASKED:
        s.rect(M + P, y, ask_w, ask_h, 'white', 24)
        for j, line in enumerate(lines):
            s.text(M + P + 28, y + 30 + 44 + 70 * j, line, BODY)
        ax = M + P + 28 + width(lines[-1], BODY) + 22          # the answer, after the question's last line
        s.rect(ax, y + 30 + 70 - 4, width(answer, BODY, 700) + 40, 66, WARM, 16)
        s.text(ax + 20, y + 30 + 44 + 70, answer, BODY, 700)
        by = y + (ask_h - 140) / 2
        for k, label in enumerate(('Link on', 'Link cut')):
            s.text(M + P + ask_w + G, by + k * 76 + 48, label, BODY, 700)
        for x in SIDES:
            v = Fe[f'{q}|{x}']
            bars(s, xs[x], by, twin_w - G, (v['link'], v['cut']), COLOR[x])
        y += feel_h
    s.save(stem)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default=ROOT / 'data/bench.json')
    ap.add_argument('--data', default=ROOT / 'results/defer_mood')
    ap.add_argument('--out', default=HERE / 'defer_mood')
    ap.add_argument('--snapshot', default=ROOT / 'results/defer_mood.json')
    args = ap.parse_args()
    data = Path(args.data)
    items = audited(dict(questions=json.load(open(args.questions))['test'], per_question={}))['questions']
    exam = audited(dict(questions=json.load(open(data / 'exam.json'))['questions'], per_question={}))['questions']
    if [(q['src'], q['answer']) for q in exam] != [(q['src'], q['answer']) for q in items]:
        raise ValueError(f'{args.questions} does not match the questions of {data / "exam.json"}')
    D, L = deference(exam)
    item, given = example(exam, items, L, lambda it: len(options(it, EX_W - 2 * P)) <= 2
                          and len(wrap(it['q'], EX_W - 2 * P, SMALL)) <= 2)
    Fe = feelings(json.load(open(data / 'feelings.json')))
    H = draw(item, given, D, Fe, args.out)
    snap = dict(checkpoint='final twins (nl33w step_300)', bases='4-bit MLX', deference=D, feelings=Fe,
                example=dict(question=item['q'], answer=LETTERS[item['answer']],
                             given={f'{c}|{x}': LETTERS[v] for (c, x), v in given.items()}), size=[W, H])
    Path(args.snapshot).write_text(json.dumps(snap, indent=1) + '\n')
    print(json.dumps(snap, indent=1))


if __name__ == '__main__':
    main()
