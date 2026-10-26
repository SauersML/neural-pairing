"""The partner-affect run (chains/affect_partner.sh, pairing/affect_partner.py) with the affect run's yes-or-no
questions (pairing/affect.py): linked, do the twins answer differently with their real partner than with the
partner's hidden states from another conversation (the 'stranger' condition)?

Top card, per twin, over both checkpoints: the conversations in which the twin is more likely to say a happier mood
word (affect.GOOD against affect.BAD after affect_partner.MOOD_STEM) with its real partner than with the partner from
another conversation (of 120), and the mirrored question pairs (affect_partner.MIRRORS) whose two questions both
favour the real partner by their exact median interval (the claims['partner'] entries; 3 pairs at 2 checkpoints).
Bottom card: for every yes-or-no question (affect.YESNO and the yes-or-no ones of affect_partner.PAIRED), the
conversations in which No, and those in which Yes, is more likely with the real partner (ties left out), summed over
each run's checkpoints (the partner-affect run's two, 60 conversations each; the affect run's three, 40 each).
The checkpoints of the partner-affect run: nl22c step 157 (after chat training) and nl33t step 300 (after training to
read steered concepts); the affect run adds beautyrl.

Reads PARTNER_DIR/CHECKPOINT/summary.json and AFFECT_DIR/CHECKPOINT/summary.json (no model) and the questions from
pairing/affect.py and pairing/affect_partner.py, and writes figures/affect_partner.{svg,png,pdf} and
results/affect_partner.json (per twin, row and column the conversations higher linked, lower and in all, and the
claim, for the mood word, the close question and the internal happiness reading, against the link cut at each
checkpoint and against the partner from another conversation over both).
Usage: python figures/affect_partner.py PARTNER_DIR AFFECT_DIR   (results/magic-nl37/results results/magic-nl36/results)
"""
import json
import sys
from pathlib import Path

from affect import NAMES as AFFECT_NAMES
from affect import constants
from chat_names import EMBER, INK, QUILL, SVG, WARM, count, pair_icon, width, wrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t')
BLOCK = 33
ROWS = tuple((x, what) for what in ('mood', 'close', 'valence') for x in ('quill', 'ember'))
TWINS = (('quill', QUILL), ('ember', EMBER))
GROUPS = (('enjoy', 'bother'), ('close', 'distant'), ('miss', 'lonely'), ('switch',))
SUFFIX = ' Answer yes or no.'
TITLE = ('Quill and Ember report the same mood and closeness',
         'to their real partner as to a partner from another conversation')
NO_TITLE = 'Quill is more likely to answer No to its real partner on every question'
TOP_ROWS = ('Conversations where the twin is more likely to say a happier mood word',
            'Mirrored question pairs where the twin clearly favours its real partner on both')
NO_HEAD = ('Conversations where the twin', 'is more likely to answer')


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
        for x, what in ROWS:
            sums[(x, what, name)] = stat(s, x, what, 'cut')
            hi, lo, n, ok = stat(s, x, what, 'stranger')
            a, b, m, c = sums.get((x, what, 'stranger'), (0, 0, 0, True))
            sums[(x, what, 'stranger')] = (a + hi, b + lo, m + n, c and ok)
    return sums


def pairs(res):
    """Per twin: (mirrored pairs whose claim holds, pairs tested) over NAMES."""
    out = {x: (0, 0) for x, _ in TWINS}
    for name in NAMES:
        claims = json.load(open(res / name / 'summary.json'))['claims']
        for x, _ in TWINS:
            held, n = out[x]
            out[x] = (held + sum(claims[x]['partner'].values()), n + len(claims[x]['partner']))
    return out


def no_yes(res, affect_res):
    """[(question, {twin: (No more likely, Yes more likely)})] in GROUPS order: conversations with the real partner
    against the partner from another conversation, summed over each run's checkpoints."""
    yesno, = constants('pairing/affect.py', 'YESNO')
    paired, = constants('pairing/affect_partner.py', 'PAIRED')
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for q, (_, good, _) in paired.items():
            if good not in ('Yes', 'No'):
                continue
            for x, _ in TWINS:
                d = s['questions'][f'{x}|{q}']['link_minus_stranger']
                no, yes = (d['higher'], d['lower']) if good == 'No' else (d['lower'], d['higher'])
                a, b = sums.get((q, x), (0, 0))
                sums[(q, x)] = (a + no, b + yes)
    for name in AFFECT_NAMES:
        s = json.load(open(affect_res / name / 'summary.json'))
        for q in yesno:
            for x, _ in TWINS:
                d = s['self_report'][x][q]['link_minus_stranger']      # log-odds of Yes against No
                a, b = sums.get((q, x), (0, 0))
                sums[(q, x)] = (a + d['lower'], b + d['higher'])
    text = {**{q: v[0] for q, v in paired.items()}, **yesno}
    return [(text[q].removesuffix(SUFFIX), {x: sums[(q, x)] for x, _ in TWINS}) for g in GROUPS for q in g]


def draw(sums, held, rows, stem):
    W, M, G, P = 2040, 40, 30, 30
    icon, num_w, sub_g, pad, line_h = 100, 162, 20, 22, 58
    tw = 2 * num_w + sub_g
    label_w = W - 2 * M - 2 * P - 2 * tw - 2 * G
    xt = [M + P + label_w + G + k * (tw + G) for k in range(2)]
    top_lines = [wrap(t, label_w, 46, 700) for t in TOP_ROWS]
    top_h = [max(56 * len(t) + 40, 130) for t in top_lines]
    bub = [wrap(q, label_w - 2 * pad, 46) for q, _ in rows]
    bub_h = [2 * pad + line_h * (len(t) - 1) + 46 for t in bub]
    starts = {sum(len(g) for g in GROUPS[:k]) for k in range(1, len(GROUPS))}
    gaps = [0] + [44 if k in starts else 16 for k in range(1, len(rows))]
    head_y, top = 184, 184 + icon + 40
    card1_h = P + icon + 20 + sum(top_h) + P
    no_head = NO_HEAD
    card2_head = P + 70 + icon + 20 + max(66, 56 * len(no_head)) + 24
    card2_h = card2_head + sum(bub_h) + sum(gaps) + P
    H = top + card1_h + G + card2_h + M
    s = SVG(W, H)
    for k, line in enumerate(TITLE):
        s.text(M, 76 + 68 * k, line, 56, 700)
    x = pair_icon(s, M, head_y, icon, 'link') + 24
    s.text(x, head_y + 68, 'Real partner, against', 46, 700)
    x += width('Real partner, against', 46, 700) + 24
    x = pair_icon(s, x, head_y, icon, 'crossed') + 24
    s.text(x, head_y + 68, 'partner from another conversation', 46, 700)

    def twin_heads(y):
        for (name, color), x0 in zip(TWINS, xt):
            label = name.capitalize()
            x1 = x0 + (tw - icon - 16 - width(label, 46, 700)) / 2
            s.mascot(name, x1, y, icon)
            s.text(x1 + icon + 16, y + 66, label, 46, 700, color)

    y = top
    s.rect(M, y, W - 2 * M, card1_h, WARM, 32)
    twin_heads(y + P)
    y += P + icon + 20
    for lines, h, key in zip(top_lines, top_h, ('mood', 'pairs')):
        for k, line in enumerate(lines):
            s.text(M + P, y + h / 2 + 16 + 56 * (k - (len(lines) - 1) / 2), line, 46, 700)
        for (name, color), x0 in zip(TWINS, xt):
            n, total = (sums[(name, 'mood', 'stranger')][0::2] if key == 'mood' else held[name])
            count(s, x0 + tw / 2, y + h / 2 + 24, n, total, color, anchor='middle')
        y += h
    y = top + card1_h + G
    s.rect(M, y, W - 2 * M, card2_h, WARM, 32)
    s.text(M + P, y + P + 50, NO_TITLE, 50, 700)
    twin_heads(y + P + 70)
    y += P + 70 + icon + 20
    for (name, color), x0 in zip(TWINS, xt):
        for k, word in enumerate(('No', 'Yes')):
            s.text(x0 + k * (num_w + sub_g) + num_w / 2, y + 46, word, 46, 700, color if word == 'No' else INK,
                   'middle')
    for k, line in enumerate(no_head):
        s.text(M + P, y + 46 + 56 * k, line, 46, 700)
    y += max(66, 56 * len(no_head)) + 24
    for (q, counts), text, h, gap in zip(rows, bub, bub_h, gaps):
        y += gap
        s.rect(M + P, y, label_w, h, 'white', 24, INK, 3)
        for k, line in enumerate(text):
            s.text(M + P + pad, y + pad + 34 + line_h * k, line, 46)
        for (name, color), x0 in zip(TWINS, xt):
            for k, (n, c) in enumerate(zip(counts[name], (color, INK))):
                count(s, x0 + k * (num_w + sub_g) + num_w / 2, y + h / 2 + 25, n, None, c, anchor='middle')
        y += h
    s.save(stem)


def main():
    res, affect_res = Path(sys.argv[1]), Path(sys.argv[2])
    sums = tally(res)
    draw(sums, pairs(res), no_yes(res, affect_res), HERE / 'affect_partner')
    out = {f'{x}|{w}|{c}': dict(higher=a, lower=b, n=n, claim=ok) for (x, w, c), (a, b, n, ok) in sums.items()}
    (ROOT / 'results/affect_partner.json').write_text(json.dumps(dict(checkpoints=NAMES, block=BLOCK, counts=out),
                                                                 indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
