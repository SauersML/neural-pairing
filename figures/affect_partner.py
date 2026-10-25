"""The partner-affect run (chains/affect_partner.sh, pairing/affect_partner.py) as one figure, from its results directory
(no model, no torch).

affect_partner: for each twin, the number of conversations in which the twin names a happier mood (affect.GOOD against
affect.BAD after affect_partner.MOOD_STEM), says it feels close (Yes to PAIRED['close']) and reads happier inside
(affect's valence direction, refit per checkpoint, after block 33, past the last link write), with its real partner
linked than with the link cut, at each checkpoint (60 conversations each), and than linked to a stranger, over both
checkpoints, as a bar against all conversations with a mark at half; green where the exact median interval of the
paired differences is above zero at that checkpoint, or at both for the stranger column (for the inside reading, also
above the random direction's upper end).

Usage: python affect_partner.py RESULTS_DIR   (writes figures/affect_partner.{svg,png,pdf} and
       results/affect_partner.json, the numbers drawn)
"""
import json
import sys
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, SVG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t')
BLOCK = 33
ROWS = (('quill', 'mood', ' names a happier mood'), ('ember', 'mood', ' names a happier mood'),
        ('quill', 'close', ' says it feels close'), ('ember', 'close', ' says it feels close'),
        ('quill', 'valence', ' reads happier inside'), ('ember', 'valence', ' reads happier inside'))


def stat(s, x, what, vs):
    """(higher, lower, n, claim) for one twin, row and comparison in one checkpoint's summary."""
    if what == 'valence':
        d = s['internal'][f'{x}|valence|{BLOCK}']
        r = s['internal'][f'{x}|random|{BLOCK}'][f'link_minus_{vs}']
        e = d[f'link_minus_{vs}']
        return e['higher'], e['lower'], e['n'], d['usable'] and e['median'][0] > max(0, r['median'][1])
    e = s['questions'][f'{x}|{what}'][f'link_minus_{vs}']
    return e['higher'], e['lower'], e['n'], e['median'][0] > 0


def tally(res):
    """Per (twin, row, column): conversations higher linked, lower, in all, and the claim; columns are each checkpoint
    against cut, then both against a stranger."""
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x, what, _ in ROWS:
            sums[(x, what, name)] = stat(s, x, what, 'cut')
            hi, lo, n, ok = stat(s, x, what, 'stranger')
            a, b, m, c = sums.get((x, what, 'stranger'), (0, 0, 0, True))
            sums[(x, what, 'stranger')] = (a + hi, b + lo, m + n, c and ok)
    return sums


def draw(sums, stem):
    W, H, M = 2400, 1680, 40
    s = SVG(W, H)
    s.text(M, 100, 'Linked rather than cut, the twins mostly say they feel happier and closer,', 60, 700)
    s.text(M, 180, 'but never more with their partner than with a stranger', 60, 700)
    cols = ((840, 1300, 'step157', ('link on against cut', 'at step 157')),
            (1360, 1820, 'nl33t', ('link on against cut', "at nl33's step 300")),
            (1900, 2360, 'stranger', ('partner against stranger', 'both checkpoints')))
    for x0, x1, _, head in cols:
        for k, line in enumerate(head):
            s.text((x0 + x1) / 2, 290 + 56 * k, line, 44, 700, INK, 'middle')
    for i, (x, what, label) in enumerate(ROWS):
        y = 470 + i * 200
        s.rich(M, y + 68, [(x.capitalize(), QUILL if x == 'quill' else EMBER, 700), (label, INK, 700)], 50)
        for x0, x1, col, _ in cols:
            won, lost, n, ok = sums[(x, what, col)]
            w = (x1 - x0) * won / n
            s.rect(x0, y, x1 - x0, 100, COOL, 10, INK, 3)
            s.rect(x0, y, w, 100, LINK if ok else INK, 10)
            s.line((x0 + x1) / 2, y - 22, (x0 + x1) / 2, y, INK, 5)
            s.line((x0 + x1) / 2, y + 100, (x0 + x1) / 2, y + 122, INK, 5)
            if i == 0:
                s.text((x0 + x1) / 2, y - 30, 'half', 40, 700, INK, 'middle')
            t = f'{won}/{n}' if i == 0 else str(won)
            fits = w > 40 + 34 * len(t)
            s.text(x0 + w - 20 if fits else x0 + w + 20, y + 68, t, 56, 700, 'white' if fits else INK,
                   'end' if fits else 'start')
    s.save(stem)


def main():
    res = Path(sys.argv[1])
    sums = tally(res)
    draw(sums, HERE / 'affect_partner')
    out = {f'{x}|{w}|{c}': dict(higher=a, lower=b, n=n, claim=ok) for (x, w, c), (a, b, n, ok) in sums.items()}
    (ROOT / 'results/affect_partner.json').write_text(json.dumps(dict(checkpoints=NAMES, block=BLOCK, counts=out),
                                                                 indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
