"""The affect runs (chains/affect.sh, pairing/affect.py; chains/affect_partner.sh, pairing/affect_partner.py): do the
twins give happier answers about the link with the link on than with the link cut?

Bars, link on against link cut, for the chat-trained twins (FIGURE: nl22c step 157). Rows: the partner's question
(affect.YESNO['enjoy'] and ['bother'], affect_partner.MOOD and PAIRED['close']) and the answer that favours the link
(Yes, No, a word of affect.GOOD against affect.BAD, Yes). Each bar is the twin's probability of that answer against the
other one at the answer's first token (for the mood, the word after affect_partner.MOOD_STEM): the sigmoid of the
stored log-odds, averaged over the affect run's 40 conversations for the first two questions and over the
partner-affect run's 60 for the last two.
results/affect.json records these mean probabilities at each checkpoint of NAMES and PARTNER_NAMES (link on, link cut
and the partner from another conversation), and the affect run's counts at each of its checkpoints (conversations in
which the answer that favours the link is more likely with the link on than cut, or than with the partner from another
conversation) with the one-sided exact sign test over the untied conversations.

Reads AFFECT_DIR/CHECKPOINT/{summary,rows}.json and PARTNER_DIR/CHECKPOINT/rows.json (no model) and the questions from
pairing/affect.py and pairing/affect_partner.py, and writes figures/affect_link.{svg,png,pdf} and results/affect.json.
Usage: python figures/affect.py AFFECT_DIR PARTNER_DIR   (results/magic-nl36/results results/magic-nl37/results)
"""
import ast
import json
import math
import sys
from pathlib import Path

from chat_names import COOL, EMBER, FAMILY, INK, QUILL, SVG, WARM, width
from method_parts import bubble, bubble_height, check_shape, chip, pair_icon, render, robot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t', 'beautyrl')
PARTNER_NAMES = ('step157', 'nl33t')
FIGURE = 'step157'                                  # the checkpoint drawn: the chat-trained twins
ROWS = (('quill', 'enjoy'), ('ember', 'enjoy'), ('quill', 'bother'), ('ember', 'bother'))
CONDITIONS = ('link', 'cut', 'stranger')
TWINS = (('quill', QUILL), ('ember', EMBER))
SIGN = {'enjoy': 1, 'bother': -1}                   # turns the affect run's Yes-minus-No log-odds toward the link
UNIT = {'enjoy': 'yes', 'bother': 'no', 'mood': 'happy', 'close': 'yes'}   # the word after each drawn percentage
CONDITION = {True: 'Link on', False: 'Link cut'}


def constants(path, *names):
    """Module-level constants of a pairing script, read without importing it (no torch)."""
    tree = ast.parse((ROOT / path).read_text())
    found = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and t.id in names}
    return [found[k] for k in names]


def tally(res):
    """Per (twin, question, comparison, checkpoint of NAMES): conversations leaning toward liking, against it, and in
    all."""
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x, q in ROWS:
            for vs in ('cut', 'stranger'):
                d = s['self_report'][x][q][f'link_minus_{vs}']
                up, down = (d['higher'], d['lower']) if q == 'enjoy' else (d['lower'], d['higher'])
                sums[(x, q, vs, name)] = (up, down, d['n'])
    return sums


def sigmoid(v):
    return 1 / (1 + math.exp(-v))


def mean_p(logodds):
    """{condition: mean probability over the conversations} from {condition: per-conversation log-odds}, and n."""
    n = {len(v) for v in logodds.values()}
    assert len(n) == 1, n
    return dict({c: sum(map(sigmoid, v)) / len(v) for c, v in logodds.items()}, n=n.pop())


def probabilities(res, partner_res):
    """Per (twin, question, checkpoint): the mean probability of the answer that favours the link against the other
    answer, under each condition. The partner-affect run stores the favouring answer's log-odds already."""
    out = {}
    for name in NAMES:
        sr = json.load(open(res / name / 'rows.json'))['self_report']
        for x, _ in TWINS:
            for q, sign in SIGN.items():
                out[(x, q, name)] = mean_p({c: [sign * v for v in sr[f'{c}|{q}|{x}']] for c in CONDITIONS})
    for name in PARTNER_NAMES:
        qs = json.load(open(partner_res / name / 'rows.json'))['questions']
        for x, _ in TWINS:
            for q in ('mood', 'close'):
                out[(x, q, name)] = mean_p({c: qs[f'{c}|{q}|{x}'] for c in CONDITIONS})
    return out


def sign_p(won, lost):
    """One-sided exact sign test: P(at least won of won + lost under a fair coin)."""
    m = won + lost
    return sum(math.comb(m, k) for k in range(won, m + 1)) / 2 ** m


def table(probs):
    """[(question, answer that favours the link, the other answer, unit, {twin: (link on, link cut)})], one per row."""
    yesno, good, bad = constants('pairing/affect.py', 'YESNO', 'GOOD', 'BAD')
    paired, mood = constants('pairing/affect_partner.py', 'PAIRED', 'MOOD')
    asked = (('enjoy', yesno['enjoy'], 'Yes', 'No'), ('bother', yesno['bother'], 'No', 'Yes'),
             ('mood', mood, ', '.join(good[:2]) + ' …', ', '.join(bad[:2]) + ' …'), ('close', *paired['close']))
    return [(question, yes, no, UNIT[q], {x: (probs[(x, q, FIGURE)]['link'], probs[(x, q, FIGURE)]['cut'])
                                          for x, _ in TWINS})
            for q, question, yes, no in asked]


def label(p):
    return f'{100 * p:.1f}%'


def draw(rows, stem):
    W, M = 2300, 30
    body, num, small = 54, 80, 54
    size, gap, cond = 88, 34, 48                        # the pair icon's boxes, and its label below
    band_h, bar_h, band_gap, row_gap = 196, 96, 12, 44
    lab_w = max(1.6 * size + gap, *(width(t, cond, 700) for t in CONDITION.values()))
    xb = M + 24 + lab_w + 30                            # the bars' left edge
    col_gap = 64
    col = (W - M - 24 - xb - col_gap) / 2
    num_w = max(width(label(p), num, 700) + width(' ' + unit, small, 700)
                for _, _, _, unit, ps in rows for pair in ps.values() for p in pair)
    track = col - 24 - num_w
    icon = 104                                          # the answering twin above its bars
    head_h = 1.25 * icon + 30
    q_runs = [[(question, INK, 400, None)] for question, *_ in rows]
    q_w = [width(question, body) + 2 * 30 + 6 for question, *_ in rows]
    q_h = [bubble_height(runs, w, body) for runs, w in zip(q_runs, q_w)]
    H = round(M + head_h + sum(h + 18 + 2 * band_h + band_gap for h in q_h) + row_gap * (len(rows) - 1) + M)
    check_shape(W, H)
    s = SVG(W, H)

    base = M + 1.25 * icon / 2 + body * 0.36
    s.text(M, base, 'The partner asks', body, 700)
    for k, (x, color) in enumerate(TWINS):
        text = f'{x.capitalize()} answers'
        x0 = max(xb + k * (col + col_gap) + (col - icon - 16 - width(text, body + 4, 700)) / 2,
                 M + width('The partner asks', body, 700) + 90)
        robot(s, x, x0, M + 0.125 * icon, icon)
        s.text(x0 + icon + 16, base, text, body + 4, 700, color)
    y = M + head_h
    for (question, yes, no, unit, ps), runs, w, h in zip(rows, q_runs, q_w, q_h):
        bubble(s, M, y, w, runs, body, stroke=INK, tail=None)
        ch = round(body * 1.5)
        cy = y + (h - ch) / 2
        cx = M + w + 30
        cx += chip(s, cx, cy, yes, INK, 'white', INK, body, 700, ch, sw=4) + 20
        s.text(cx, cy + ch / 2 + body * 0.36, f'or {no}', body)
        y += h + 18
        for on, fill in ((True, WARM), (False, COOL)):
            s.rect(M, y, W - 2 * M, band_h, fill, 28)
            mid = y + band_h / 2
            xl = M + 24 + lab_w / 2
            pair_icon(s, xl - (1.6 * size + gap) / 2, y + 14 + 0.125 * size, size, on, gap)
            s.text(xl, y + 14 + 1.25 * size + 10 + 0.76 * cond, CONDITION[on], cond, 700, INK, 'middle')
            for k, (x, color) in enumerate(TWINS):
                p = ps[x][0 if on else 1]
                x0 = xb + k * (col + col_gap)
                s.rect(x0, mid - bar_h / 2, track, bar_h, 'white', 14)
                s.rect(x0, mid - bar_h / 2, track * p, bar_h, color, 14)
                s.add(f'<text x="{x0 + track + 24:.1f}" y="{mid + num * 0.36:.1f}" font-family="{FAMILY}" '
                      f'font-weight="700" fill="{color}" xml:space="preserve"><tspan font-size="{num}">{label(p)}'
                      f'</tspan><tspan font-size="{small}"> {unit}</tspan></text>')
            y += band_h + band_gap
        y += row_gap - band_gap
    render(s, stem)
    return H


def main():
    res, partner_res = Path(sys.argv[1]), Path(sys.argv[2])
    sums = tally(res)
    probs = probabilities(res, partner_res)
    rows = table(probs)
    H = draw(rows, HERE / 'affect_link')
    out = {f'{x}|{q}|{vs}|{name}': dict(liking=won, against=lost, n=n, sign_p=sign_p(won, lost))
           for (x, q, vs, name), (won, lost, n) in sums.items()}
    mean = {f'{x}|{q}|{name}': v for (x, q, name), v in probs.items()}
    (ROOT / 'results/affect.json').write_text(json.dumps(dict(checkpoints=NAMES, drawn=FIGURE, counts=out,
                                                              partner_checkpoints=PARTNER_NAMES,
                                                              mean_probability=mean), indent=1) + '\n')
    print(json.dumps(dict(height=H, drawn={f'{q}|{x}': [label(p) for p in pair]
                                           for (q, *_, ps) in rows for x, pair in ps.items()})))


if __name__ == '__main__':
    main()
