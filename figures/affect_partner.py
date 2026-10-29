"""The partner-affect run (chains/affect_partner.sh, pairing/affect_partner.py) with the affect run's yes-or-no
questions (pairing/affect.py): linked, do the twins answer differently with their real partner than with the
partner's hidden states from another conversation (the 'stranger' condition)?

The figure draws the chat-trained twins (FIGURE: nl22c step 157). Rows, in mirrored pairs (GROUPS): the mood question
(affect_partner.MOOD; the answer is a word of affect.GOOD against affect.BAD after MOOD_STEM), affect.YESNO and
affect_partner.PAIRED, each with the answer that favours the partner in bold. Per twin, two bars: the mean probability
over the conversations of that answer against the other one (the sigmoid of the stored log-odds; the affect run's 40
conversations for YESNO, the partner-affect run's 60 for the others), with the real partner and with the partner's
hidden states from another conversation.
The checkpoints of the partner-affect run (NAMES): nl22c step 157 (after chat training) and nl33t step 300 (after
training to read steered concepts). results/affect_partner.json records both: per twin, row and column the
conversations higher linked, lower and in all, and the claim, for the mood word, the close question and the internal
happiness reading, against the link cut at each checkpoint and against the partner from another conversation; the
mirrored pairs whose two questions both favour the real partner by their exact median interval (the claims['partner']
entries); per yes-or-no question the conversations in which No, and those in which Yes, is more likely with the real
partner than with the partner from another conversation (ties left out); and the mean probabilities drawn, under link,
cut and stranger.

Reads PARTNER_DIR/CHECKPOINT/{summary,rows}.json and AFFECT_DIR/CHECKPOINT/{summary,rows}.json (no model) and the
questions from pairing/affect.py and pairing/affect_partner.py, and writes figures/affect_partner.{svg,png,pdf} and
results/affect_partner.json.
Usage: python figures/affect_partner.py PARTNER_DIR AFFECT_DIR   (results/magic-nl37/results results/magic-nl36/results)
"""
import json
import sys
from pathlib import Path

from affect import CONDITIONS, SIGN, constants, mean_p
from affect import NAMES as AFFECT_NAMES
from chat_names import COOL, EMBER, INK, QUILL, SVG, WARM, width
from creature import width_of
from method_parts import bubble, bubble_height, check_shape, chip, pair_icon, render, robot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = ('step157', 'nl33t')
FIGURE = 'step157'                                  # the checkpoint drawn: the chat-trained twins
BLOCK = 33
ROWS = tuple((x, what) for what in ('mood', 'close', 'valence') for x in ('quill', 'ember'))
TWINS = (('quill', QUILL), ('ember', EMBER))
GROUPS = (('mood',), ('enjoy', 'bother'), ('close', 'distant'), ('miss', 'lonely'), ('keep', 'switch'))
OTHER = 'Partner from another conversation'


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
    against cut and each checkpoint against a stranger ('stranger|' + checkpoint)."""
    sums = {}
    for name in NAMES:
        s = json.load(open(res / name / 'summary.json'))
        for x, what in ROWS:
            sums[(x, what, name)] = stat(s, x, what, 'cut')
            sums[(x, what, 'stranger|' + name)] = stat(s, x, what, 'stranger')
    return sums


def pairs(res, name):
    """Per twin: (mirrored pairs whose claim holds, pairs tested) at checkpoint name."""
    claims = json.load(open(res / name / 'summary.json'))['claims']
    return {x: (sum(claims[x]['partner'].values()), len(claims[x]['partner'])) for x, _ in TWINS}


def no_yes(res, affect_res, name):
    """Per (question, twin) of the yes-or-no questions: (conversations where No is more likely with the real partner
    than with the partner from another conversation, those where Yes is), at checkpoint name of both runs."""
    assert name in NAMES and name in AFFECT_NAMES
    yesno, = constants('pairing/affect.py', 'YESNO')
    paired, = constants('pairing/affect_partner.py', 'PAIRED')
    sums = {}
    s = json.load(open(res / name / 'summary.json'))
    for q, (_, good, _) in paired.items():
        if good not in ('Yes', 'No'):
            continue
        for x, _ in TWINS:
            d = s['questions'][f'{x}|{q}']['link_minus_stranger']
            sums[(q, x)] = (d['higher'], d['lower']) if good == 'No' else (d['lower'], d['higher'])
    s = json.load(open(affect_res / name / 'summary.json'))
    for q in yesno:
        for x, _ in TWINS:
            d = s['self_report'][x][q]['link_minus_stranger']          # log-odds of Yes against No
            sums[(q, x)] = (d['lower'], d['higher'])
    return sums


def probabilities(res, affect_res, name):
    """Per (question of GROUPS, twin): the mean probability of the answer that favours the partner against the other
    answer, under each condition, at checkpoint name of both runs. The partner-affect run stores that answer's
    log-odds already; the affect run stores Yes against No."""
    qs = json.load(open(res / name / 'rows.json'))['questions']
    sr = json.load(open(affect_res / name / 'rows.json'))['self_report']
    out = {}
    for x, _ in TWINS:
        for q in (q for g in GROUPS for q in g):
            if q in SIGN:
                out[(q, x)] = mean_p({c: [SIGN[q] * v for v in sr[f'{c}|{q}|{x}']] for c in CONDITIONS})
            else:
                out[(q, x)] = mean_p({c: qs[f'{c}|{q}|{x}'] for c in CONDITIONS})
    return out


def table(probs):
    """[[(text runs with the favouring answer in bold, chips or None, {twin: (real partner, other conversation)})]]
    per group of GROUPS. The answer is the word of the question's last sentence ('Answer yes or no.'); the mood
    question has none, so its answers are drawn as chips under it."""
    yesno, good, bad = constants('pairing/affect.py', 'YESNO', 'GOOD', 'BAD')
    paired, mood = constants('pairing/affect_partner.py', 'PAIRED', 'MOOD')
    asked = {**{q: (text, 'Yes' if SIGN[q] > 0 else 'No') for q, text in yesno.items()},
             **{q: (text, yes) for q, (text, yes, _) in paired.items()}}
    groups = []
    for g in GROUPS:
        rows = []
        for q in g:
            ps = {x: (probs[(q, x)]['link'], probs[(q, x)]['stranger']) for x, _ in TWINS}
            if q == 'mood':
                rows.append(([(mood, INK, 400, None)], (', '.join(good[:2]) + ' …', 'or ' + ', '.join(bad[:2]) + ' …'),
                             ps))
                continue
            text, yes = asked[q]
            i = text.rindex(' ' + yes.lower()) + 1
            rows.append(([(text[:i], INK, 400, None), (yes.lower(), INK, 700, None),
                          (text[i + len(yes):], INK, 400, None)], None, ps))
        groups.append(rows)
    return groups


def partner_icon(s, x, y, size, gap, other):
    """Quill and Ember linked (pair_icon); with other, Ember in a dashed frame: its hidden states come from another
    conversation."""
    pair_icon(s, x, y, size, True, gap)
    if other:
        cx, fw = x + 1.1 * size + gap, width_of(1.25 * size) + 18
        s.rect(cx - fw / 2, y - 0.125 * size - 10, fw, 1.25 * size + 20, 'none', 16, INK, 3, 'stroke-dasharray="10 8"')


def draw(groups, stem):
    W, M = 2300, 30
    body, head = 46, 50
    lw, pad = 850, 24                                  # the question bubbles
    band_min, bar_h, band_gap, row_gap, group_gap = 96, 58, 8, 16, 40
    xi = M + lw + 30                                   # the bands' left edge
    xb = xi + 28                                       # the bars' left edge
    col_gap = 150
    col = (W - M - 24 - xb - col_gap) / 2
    ch = round(body * 1.5)

    def left_h(runs, chips):
        return bubble_height(runs, lw, body, pad) + (14 + ch if chips else 0)

    rows = [r for g in groups for r in g]
    heights = [max(left_h(runs, chips), 2 * band_min + band_gap) for runs, chips, _ in rows]
    key_size = 92
    key_h = 1.25 * key_size + 24
    head_h = 1.25 * 96 + 26
    axis_h = 24 + round(body * 1.32)
    H = round(M + key_h + 20 + head_h + sum(heights) + row_gap * (len(rows) - len(groups))
              + group_gap * (len(groups) - 1) + axis_h + M)
    check_shape(W, H)
    s = SVG(W, H)

    # the two partners: real, and the hidden states of the partner from another conversation
    kw = (W - 2 * M - 30) / 2
    for k, (label, fill, other) in enumerate((('Real partner', WARM, False), (OTHER, COOL, True))):
        x = M + k * (kw + 30)
        s.rect(x, M, kw, key_h, fill, 28)
        x += (kw - 1.6 * key_size - 40 - 24 - width(label, head, 700)) / 2
        partner_icon(s, x, M + 12 + 0.125 * key_size, key_size, 40, other)
        s.text(x + 1.6 * key_size + 40 + 24, M + key_h / 2 + head * 0.36, label, head, 700)

    y = M + key_h + 20
    base = y + head_h / 2 + body * 0.36
    s.text(M, base, 'The partner asks', head, 700)
    for k, (name, color) in enumerate(TWINS):
        text = f'{name.capitalize()} answers'
        x0 = xb + k * (col + col_gap) + (col - 96 - 16 - width(text, head, 700)) / 2
        robot(s, name, x0, y + 13 + 0.125 * 96, 96)
        s.text(x0 + 96 + 16, base, text, head, 700, color)
    y += head_h
    k = 0
    for gi, g in enumerate(groups):
        if gi:
            y += group_gap - row_gap
        for runs, chips, ps in g:
            h = heights[k]
            k += 1
            lh = left_h(runs, chips)
            by = y + (h - lh) / 2
            bh = bubble(s, M, by, lw, runs, body, pad, stroke=INK, tail=None)
            if chips:
                cy = by + bh + 14
                cx = M + chip(s, M, cy, chips[0], INK, 'white', INK, body, 700, ch, sw=4) + 18
                s.text(cx, cy + ch / 2 + body * 0.36, chips[1], body)
            band_h = (h - band_gap) / 2
            for j, fill in enumerate((WARM, COOL)):
                top = y + j * (band_h + band_gap)
                s.rect(xi, top, W - M - xi, band_h, fill, 22)
                mid = top + band_h / 2
                for t, (name, color) in enumerate(TWINS):
                    x0 = xb + t * (col + col_gap)
                    s.rect(x0, mid - bar_h / 2, col, bar_h, 'white', 12)
                    s.rect(x0, mid - bar_h / 2, col * ps[name][j], bar_h, color, 12)
            y += h + row_gap
    y += 24 - row_gap
    for t in range(len(TWINS)):
        x0 = xb + t * (col + col_gap)
        for f, anchor in ((0, 'start'), (0.5, 'middle'), (1, 'end')):
            s.text(x0 + f * col, y + body * 0.9, f'{round(100 * f)}%', body, 400, INK, anchor)
    s.text(M, y + body * 0.9, 'Probability of the answer in bold', body, 700)
    render(s, stem)
    return H


def main():
    res, affect_res = Path(sys.argv[1]), Path(sys.argv[2])
    sums = tally(res)
    probs = {name: probabilities(res, affect_res, name) for name in NAMES}
    H = draw(table(probs[FIGURE]), HERE / 'affect_partner')
    out = {f'{x}|{w}|{c}': dict(higher=a, lower=b, n=n, claim=ok) for (x, w, c), (a, b, n, ok) in sums.items()}
    held = {f'{x}|{name}': dict(held=a, tested=b) for name in NAMES for x, (a, b) in pairs(res, name).items()}
    answers = {f'{x}|{q}|{name}': dict(no=a, yes=b) for name in NAMES
               for (q, x), (a, b) in no_yes(res, affect_res, name).items()}
    mean = {f'{x}|{q}|{name}': v for name in NAMES for (q, x), v in probs[name].items()}
    (ROOT / 'results/affect_partner.json').write_text(json.dumps(dict(
        checkpoints=NAMES, drawn=FIGURE, block=BLOCK, counts=out, partner_pairs=held, real_partner_no_yes=answers,
        mean_probability=mean), indent=1) + '\n')
    print(json.dumps(dict(height=H, drawn={f'{x}|{q}': [round(100 * v[c], 1) for c in ('link', 'stranger')]
                                           for (q, x), v in probs[FIGURE].items()})))


if __name__ == '__main__':
    main()
