"""The final twins (nl33w step_300) with the link on and cut, read by pairing/defer_mood.py with 4-bit MLX bases: on
the exam, how often each linked twin gives its cut partner's answer; asked about feelings, in how many conversations
each twin's happy answer is more probable with the link on than cut.

Top left, one real test question that Ember answers right and Quill wrong with the link cut and on which linked Ember
gives Quill's answer and linked Quill its own: the shortest such question whose choices fit on one line. Each twin's
mascot sits on the choice it gives, link cut and link on; the arrow is Ember's move. Top right, of the test questions
on which the two twins give different letters with the link cut, the percent on which linked Ember gives cut Quill's
letter, and linked Quill cut Ember's. A twin's answer is its most likely letter after 'The answer is'
(bench_eval.letter_probs, the exam pair's episode; the final twins never trained on exam questions). The 31 test
questions whose input also appears in the training questions are left out (uplift.audited), which leaves 3,134.

Bottom, per question the partner asks after each of the final twins' 40 new free linked conversations (moodscan's
mood_new conversations, results/magic-nl40/results/mood_new/nl33w), read on the same text with the link on and cut:
the share of conversations whose happy answer is more probable with the link on (Link on) and with the link cut (Link
cut). The happy answer's log-odds is Yes against No at the answer's first token for enjoy and close, and a word of
affect.GOOD against affect.BAD after the mood stem for mood. results/defer_mood.json also records, per question and
twin, the mean probability of the happy answer linked and cut (the sigmoid of the log-odds).

Reads results/defer_mood/exam.json and feelings.json (pairing/defer_mood.py) and the test questions from data/bench.json
(python pairing/build_bench.py), and writes figures/defer_mood.{svg,png,pdf} and results/defer_mood.json.
Usage: python figures/defer_mood.py [--questions data/bench.json] [--data results/defer_mood]
"""
import argparse
import json
import math
from pathlib import Path

from chat_names import EMBER, INK, QUILL, SVG, WARM, pair_icon, width, wrap
from method_parts import arrow, check_shape
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
BODY, BIG = 46, 64
TRACK = '#E2E2E7'                                   # a bar's grey track
CHOICE = 640                                        # widest choice of the example question, in pixels


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
            out[f'{q}|{x}'] = dict(link=round(sum(p['link']) / len(p['link']), 4),
                                   cut=round(sum(p['cut']) / len(p['cut']), 4),
                                   happier_linked=sum(a > b for a, b in zip(lo['link'], lo['cut'])),
                                   happier_cut=sum(a < b for a, b in zip(lo['link'], lo['cut'])), n=len(lo['link']),
                                   logodds_link_minus_cut=round(sum(a - b for a, b in zip(lo['link'], lo['cut']))
                                                                / len(lo['link']), 3))
    return out


def pct(v):
    return f'{round(100 * v)}%'


def check(s, x, y, size, color=INK):
    """A tick mark whose box starts at (x, y)."""
    s.path(f'M{x:.1f},{y + 0.55 * size:.1f} L{x + 0.38 * size:.1f},{y + 0.9 * size:.1f} '
           f'L{x + size:.1f},{y + 0.1 * size:.1f}', color, 8)


def bars(s, x, y, w, values, color, h=64, gap=12):
    """The link-on bar (solid) above the link-cut bar (faded) from x on grey tracks, each with its percent at the end;
    values: (on, cut) shares. Returns the height."""
    full = w - width('100%', BODY, 700) - 18
    for k, (v, opacity) in enumerate(zip(values, (1, 0.45))):
        by = y + k * (h + gap)
        s.rect(x, by, full, h, TRACK, h / 2)
        if v:                                         # the true width, its ends rounded no wider than it
            s.rect(x, by, full * v, h, color, min(h, full * v) / 2, extra=f'opacity="{opacity}"')
        s.text(x + w, by + h / 2 + 16, pct(v), BODY, 700, color, 'end')
    return 2 * h + gap


def draw(item, given, D, Fe, stem):
    ex_w, col, icon, row = 1160, 270, 112, 118
    q_lines = wrap(item['q'], ex_w - 2 * P, BODY)
    head = 80 + 62
    top_h = P + 58 * len(q_lines) + 20 + head + 4 * row + P - 10
    right_x = M + ex_w + G
    bot_top = M + top_h + G
    ask_w, label_w = 720, 210
    twin_w = (W - 2 * M - 2 * P - ask_w - label_w - 2 * G) / 2
    ask_h = 2 * 58 + 2 * 30 + 12
    feel_h = ask_h + 40
    bot_h = P + icon + 10 + 3 * feel_h + P - 36
    H = bot_top + bot_h + M
    check_shape(W, H)
    s = SVG(W, H)

    # one exam question: each twin's mascot on the choice it gives, link cut and link on
    s.rect(M, M, ex_w, top_h, WARM, 32)
    y = M + P
    for k, line in enumerate(q_lines):
        s.text(M + P, y + 44 + 58 * k, line, BODY)
    y += 58 * len(q_lines) + 20
    cx = {'cut': M + ex_w - P - 1.5 * col, 'link': M + ex_w - P - col / 2}
    for cond, label in (('cut', 'Link cut'), ('link', 'Link on')):
        pair_icon(s, cx[cond] - 105, y - 6, 80, cond, gap=50)
        s.text(cx[cond], y + 80 + 46, label, BODY, 700, INK, 'middle')
    y += head
    ys = {}
    for k, (L, c) in enumerate(zip(LETTERS, item['choices'])):
        ry = y + k * row
        right = k == item['answer']
        cw = width(f'{L}. {c}', BODY, 700 if right else 400) + 48 + (70 if right else 0)
        s.rect(M + P, ry + 14, cw, row - 28, 'white', 18, INK if right else 'none', 4)
        s.text(M + P + 24, ry + row / 2 + 16, f'{L}. {c}', BODY, 700 if right else 400)
        if right:
            check(s, M + P + cw - 70, ry + row / 2 - 22, 44)
        ys[k] = ry + (row - icon) / 2
    spots = {}
    for cond in ('cut', 'link'):
        at = {}
        for x in SIDES:
            at.setdefault(given[(cond, x)], []).append(x)
        for k, xs in at.items():
            for j, x in enumerate(xs):
                px = cx[cond] - icon / 2 + (j - (len(xs) - 1) / 2) * 0.62 * icon
                s.mascot(x + ('_cut' if cond == 'cut' else ''), px, ys[k], icon)
                spots[(cond, x)] = (px, ys[k])
    # from the side of cut Ember's head to the top (or bottom) of linked Ember's
    (ax, ay), (bx, by) = spots[('cut', 'ember')], spots[('link', 'ember')]
    down = by > ay
    a, b = (ax + 0.7 * icon, ay + 0.3 * icon), (bx + 0.5 * icon, by + (0.02 if down else 1.0) * icon)
    ctrl = (b[0], a[1])
    curve = [((1 - t) ** 2 * a[0] + 2 * t * (1 - t) * ctrl[0] + t * t * b[0],
              (1 - t) ** 2 * a[1] + 2 * t * (1 - t) * ctrl[1] + t * t * b[1]) for t in [k / 24 for k in range(25)]]
    arrow(s, curve, EMBER, 8, 28)

    # every exam question the cut twins answer differently: who gives the partner's answer
    rw = W - M - right_x
    s.rect(right_x, M, rw, top_h, WARM, 32)
    y = M + P
    for line in wrap('Exam questions Quill and Ember answer differently with the link cut', rw - 2 * P, BODY, 700):
        s.text(right_x + P, y + 44, line, BODY, 700)
        y += 58
    block = (M + top_h - P - y) / 2
    full = rw - 2 * P - width('100%', BIG, 700) - 24
    for k, x in enumerate(('ember', 'quill')):
        y0 = y + k * block + (block - icon - 12 - 100) / 2
        y_ = SIDES[1 - SIDES.index(x)]
        v = D[f'{x}_gives_partner'] / D['apart']
        s.mascot(x, right_x + P - 16, y0, icon)
        s.rich(right_x + P + icon - 4, y0 + icon / 2 + 16,
               [(f'Link on, {NAME[x]} gives ', INK, 700), (NAME[y_] + '\u2019s', COLOR[y_], 700), (' answer', INK, 700)],
               BODY)
        by = y0 + icon + 12
        s.rect(right_x + P, by, max(10, full * v), 100, COLOR[x], 16)
        s.text(right_x + P + max(10, full * v) + 20, by + 74, pct(v), BIG, 700, COLOR[x])

    # feelings: the probability of the happy answer, link cut against link on, per twin
    s.rect(M, bot_top, W - 2 * M, bot_h, WARM, 32)
    xs = {'quill': M + P + ask_w + label_w + 2 * G, 'ember': M + P + ask_w + label_w + 2 * G + twin_w}
    y = bot_top + P
    s.text(M + P, y + icon / 2 + 16, 'Happier answer, per conversation', BODY, 700)
    for x in SIDES:
        s.mascot(x, xs[x] - 16, y, icon)
        s.text(xs[x] + icon - 4, y + icon / 2 + 16, NAME[x], BODY, 700, COLOR[x])
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
            bars(s, xs[x], by, twin_w - G, (v['happier_linked'] / v['n'], v['happier_cut'] / v['n']), COLOR[x])
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
    item, given = example(exam, items, L, lambda it: max(width(c, BODY, 700) for c in it['choices']) <= CHOICE)
    Fe = feelings(json.load(open(data / 'feelings.json')))
    H = draw(item, given, D, Fe, args.out)
    snap = dict(checkpoint='final twins (nl33w step_300)', bases='4-bit MLX', deference=D, feelings=Fe,
                example=dict(question=item['q'], answer=LETTERS[item['answer']],
                             given={f'{c}|{x}': LETTERS[v] for (c, x), v in given.items()}), size=[W, H])
    Path(args.snapshot).write_text(json.dumps(snap, indent=1) + '\n')
    print(json.dumps(snap, indent=1))


if __name__ == '__main__':
    main()
