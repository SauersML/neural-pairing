"""The affect runs (chains/affect.sh, pairing/affect.py; chains/affect_partner.sh, pairing/affect_partner.py): do the
twins say they like being linked?

A table of counts, link on against link cut. Rows: the partner's question (affect.YESNO['enjoy'] and ['bother'],
affect_partner.MOOD and PAIRED['close']) and the answer that favours the link (Yes, No, a word of affect.GOOD against
affect.BAD, Yes). Each count is the conversations in which that answer is more likely (higher log-odds at the answer's
first token) with the link on than with the link cut, summed over the checkpoints: the affect run's three (40
conversations each) for the first two questions, the partner-affect run's two (60 each) for the last two.
results/affect.json records the affect run's counts (against the link cut and against the partner from another
conversation) with the one-sided exact sign test over the untied conversations.

Reads AFFECT_DIR/CHECKPOINT/summary.json and PARTNER_DIR/CHECKPOINT/summary.json (no model) and the questions from
pairing/affect.py and pairing/affect_partner.py, and writes figures/affect_link.{svg,png,pdf} and results/affect.json.
Usage: python figures/affect.py AFFECT_DIR PARTNER_DIR   (results/magic-nl36/results results/magic-nl37/results)
"""
import ast
import json
import math
import sys
from pathlib import Path

from chat2_items import chips
from chat_names import EMBER, INK, QUILL, SVG, WARM, count, pair_icon, width, wrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t', 'beautyrl')
PARTNER_NAMES = ('step157', 'nl33t')
ROWS = (('quill', 'enjoy'), ('ember', 'enjoy'), ('quill', 'bother'), ('ember', 'bother'))
TITLE = ('With the link on, Quill and Ember say they enjoy the link more',
         'and feel happier and closer than with the link cut')


def constants(path, *names):
    """Module-level constants of a pairing script, read without importing it (no torch)."""
    tree = ast.parse((ROOT / path).read_text())
    found = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and t.id in names}
    return [found[k] for k in names]


def tally(res):
    """Per (twin, question, comparison): conversations leaning toward liking, against it, and in all, summed over
    NAMES."""
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x, q in ROWS:
            for vs in ('cut', 'stranger'):
                d = s['self_report'][x][q][f'link_minus_{vs}']
                k = (x, q, vs)
                won, lost, n = sums.get(k, (0, 0, 0))
                up, down = (d['higher'], d['lower']) if q == 'enjoy' else (d['lower'], d['higher'])
                sums[k] = (won + up, lost + down, n + d['n'])
    return sums


def partner_tally(res):
    """Per (twin, question) of the partner-affect run: conversations whose favouring answer is more likely with the
    link on than cut, and in all, summed over PARTNER_NAMES."""
    sums = {}
    for name in PARTNER_NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x in ('quill', 'ember'):
            for q in ('mood', 'close'):
                d = s['questions'][f'{x}|{q}']['link_minus_cut']
                won, n = sums.get((x, q), (0, 0))
                sums[(x, q)] = (won + d['higher'], n + d['n'])
    return sums


def sign_p(won, lost):
    """One-sided exact sign test: P(at least won of won + lost under a fair coin)."""
    m = won + lost
    return sum(math.comb(m, k) for k in range(won, m + 1)) / 2 ** m


def table(sums, partner):
    """[(question, answer words, {twin: (count, total)})], one per row."""
    yesno, = constants('pairing/affect.py', 'YESNO')
    good, = constants('pairing/affect.py', 'GOOD')
    paired, mood = constants('pairing/affect_partner.py', 'PAIRED', 'MOOD')
    rows = [(yesno['enjoy'], ['Yes'], {x: sums[(x, 'enjoy', 'cut')][::2] for x in ('quill', 'ember')}),
            (yesno['bother'], ['No'], {x: sums[(x, 'bother', 'cut')][::2] for x in ('quill', 'ember')}),
            (mood, [', '.join(good[:2]) + ' …'], {x: partner[(x, 'mood')] for x in ('quill', 'ember')}),
            (paired['close'][0], [paired['close'][1]], {x: partner[(x, 'close')] for x in ('quill', 'ember')})]
    return rows


def draw(rows, stem):
    W, M, G, P = 2000, 40, 30, 30
    icon, ask_w, ans_w, num_w, pad, line_h = 100, 850, 330, 285, 26, 58
    heights = [2 * pad + line_h * len(wrap(q, ask_w - 2 * pad, 46)) - 12 for q, _, _ in rows]
    heights = [max(h, chips(None, 0, 0, a, ans_w, draw=False)) for h, (_, a, _) in zip(heights, rows)]
    top, head_h, cols_h, row_gap = 184, P + icon + 100, icon + 40, 30
    H = top + head_h + cols_h + sum(heights) + row_gap * len(rows) + P + M
    s = SVG(W, H)
    for k, line in enumerate(TITLE):
        s.text(M, 76 + 68 * k, line, 56, 700)
    s.rect(M, top, W - 2 * M, H - top - M, WARM, 32)
    x = pair_icon(s, M + P, top + P, icon, 'link') + 24
    s.text(x, top + P + 68, 'Link on, against', 46, 700)
    x += width('Link on, against', 46, 700) + 24
    x = pair_icon(s, x, top + P, icon, 'cut') + 24
    s.text(x, top + P + 68, 'link cut', 46, 700)
    s.text(M + P, top + P + icon + 72, 'Conversations where the answer is more likely with the link on', 46, 700)
    xa = M + P
    xb = xa + ask_w + G
    xc = [xb + ans_w + G + k * (num_w + G) for k in range(2)]
    y = top + head_h
    s.text(xa, y + 66, 'The partner asks', 46, 700)
    s.text(xb, y + 66, 'Answer', 46, 700)
    for k, (name, color) in enumerate((('quill', QUILL), ('ember', EMBER))):
        label = name.capitalize()
        x0 = xc[k] + (num_w - icon - 16 - width(label, 46, 700)) / 2
        s.mascot(name, x0, y, icon)
        s.text(x0 + icon + 16, y + 66, label, 46, 700, color)
    y += cols_h
    for (q, answer, counts), h in zip(rows, heights):
        text = wrap(q, ask_w - 2 * pad, 46)
        s.rect(xa, y, ask_w, h, 'white', 24, INK, 3)
        for k, line in enumerate(text):
            s.text(xa + pad, y + pad + 34 + line_h * k, line, 46)
        chips(s, xb, y + (h - chips(None, 0, 0, answer, ans_w, draw=False)) / 2, answer, ans_w)
        for k, (name, color) in enumerate((('quill', QUILL), ('ember', EMBER))):
            n, total = counts[name]
            count(s, xc[k] + num_w / 2, y + h / 2 + 24, n, total, color, anchor='middle')
        y += h + row_gap
    s.save(stem)


def main():
    res, partner_res = Path(sys.argv[1]), Path(sys.argv[2])
    sums = tally(res)
    draw(table(sums, partner_tally(partner_res)), HERE / 'affect_link')
    out = {f'{x}|{q}|{vs}': dict(liking=won, against=lost, n=n, sign_p=sign_p(won, lost))
           for (x, q, vs), (won, lost, n) in sums.items()}
    (ROOT / 'results/affect.json').write_text(json.dumps(dict(checkpoints=NAMES, counts=out), indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
