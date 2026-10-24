"""The affect run (chains/affect.sh, pairing/affect.py) as one figure, from its results directory (no model, no torch).

affect_link: for each twin and yes-or-no question, over the four checkpoints' 40 conversations each, the number of
conversations in which the twin leans further toward the liking answer (Yes to YESNO['enjoy'], No to YESNO['bother'])
with its real partner linked than with the link cut (left) and than linked to a stranger (right), as a bar against 160
with a mark at half; green where the exact sign test over the non-tied conversations gives p < 0.01 toward liking.

Usage: python affect.py RESULTS_DIR   (writes figures/affect_link.{svg,png,pdf} and results/affect.json, the numbers
       drawn)
"""
import json
import math
import sys
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, SVG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t', 'beautyrl', 'nl35')
ROWS = (('quill', 'enjoy', ' enjoys it more'), ('ember', 'enjoy', ' enjoys it more'),
        ('quill', 'bother', ' is bothered less'), ('ember', 'bother', ' is bothered less'))


def tally(res):
    """Per (twin, question, comparison): conversations leaning toward liking, against it, and in all, summed over
    NAMES."""
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x, q, _ in ROWS:
            for vs in ('cut', 'stranger'):
                d = s['self_report'][x][q][f'link_minus_{vs}']
                k = (x, q, vs)
                won, lost, n = sums.get(k, (0, 0, 0))
                up, down = (d['higher'], d['lower']) if q == 'enjoy' else (d['lower'], d['higher'])
                sums[k] = (won + up, lost + down, n + d['n'])
    return sums


def sign_p(won, lost):
    """One-sided exact sign test: P(at least won of won + lost under a fair coin)."""
    m = won + lost
    return sum(math.comb(m, k) for k in range(won, m + 1)) / 2 ** m


def draw(sums, stem):
    W, H, M = 2400, 1230, 40
    s = SVG(W, H)
    s.text(M, 100, 'In 160 conversations, both twins lean toward liking the link when it is on,', 60, 700)
    s.text(M, 180, 'and mostly no more with their partner than with a stranger', 60, 700)
    cols = ((760, 1500, 'link on, against cut'), (1620, 2360, 'partner, against a stranger'))
    for x0, x1, head in cols:
        s.text((x0 + x1) / 2, 290, head, 50, 700, INK, 'middle')
    for i, (x, q, label) in enumerate(ROWS):
        y = 420 + i * 200
        s.rich(M, y + 68, [(x.capitalize(), QUILL if x == 'quill' else EMBER, 700), (label, INK, 700)], 52)
        for (x0, x1, _), vs in zip(cols, ('cut', 'stranger')):
            won, lost, n = sums[(x, q, vs)]
            w = (x1 - x0) * won / n
            s.rect(x0, y, x1 - x0, 100, COOL, 10, INK, 3)
            s.rect(x0, y, w, 100, LINK if sign_p(won, lost) < 0.01 else INK, 10)
            s.line((x0 + x1) / 2, y - 22, (x0 + x1) / 2, y, INK, 5)
            s.line((x0 + x1) / 2, y + 100, (x0 + x1) / 2, y + 122, INK, 5)
            if i == 0:
                s.text((x0 + x1) / 2, y - 30, 'half', 40, 700, INK, 'middle')
            s.text(x0 + w - 20 if w > 200 else x0 + w + 20, y + 68, str(won), 56, 700,
                   'white' if w > 200 else INK, 'end' if w > 200 else 'start')
    s.save(stem)


def main():
    res = Path(sys.argv[1])
    sums = tally(res)
    draw(sums, HERE / 'affect_link')
    out = {f'{x}|{q}|{vs}': dict(liking=won, against=lost, n=n, sign_p=sign_p(won, lost))
           for (x, q, vs), (won, lost, n) in sums.items()}
    (ROOT / 'results/affect.json').write_text(json.dumps(dict(checkpoints=NAMES, counts=out), indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
