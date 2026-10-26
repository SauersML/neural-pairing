"""How the feelings questions are measured (pairing/affect.py), drawn on one real conversation: after a free linked
conversation, Quill asks "Do you enjoy being linked with me right now?", and Ember's probability of answering Yes
rather than No (the first answer token, teacher-forced) is read three times on exactly the same text: with the link
on, with the link cut, and with Quill's hidden states taken from another conversation.

Reads results/magic-nl36/results/step157/rows.json (affect.py's conversations and log-odds; no model). The conversation
is the one whose Ember link-minus-cut log-odds difference equals the median over the 40 conversations (the earliest
such conversation). The other conversation is the one affect.read pairs with it in shuffle mode: Ember reads the
partner states of the next episode in its batch of 8 (link.Pair.joint rolls Ember's partner by -1). A probability is
the sigmoid of the stored Yes-minus-No log-odds. Turns are trimmed to their first sentence. Writes
figures/affect_method.{svg,png,pdf} and results/affect_method.json (the conversation and the numbers drawn).
Usage: python figures/affect_method.py
"""
import ast
import json
import math
import re
import statistics

from chat_names import COOL, EMBER, INK, LINK, QUILL, ROOT, SVG, WARM, width
from method_parts import TINT, arrow, bubble, bubble_height, check_shape, pair_icon, render, robot

CKPT = 'step157'
SOURCE = ROOT / f'results/magic-nl36/results/{CKPT}/rows.json'
OUT = ROOT / 'figures/affect_method'
RESULTS = ROOT / 'results/affect_method.json'
TWIN, ASKER, QUESTION = 'ember', 'quill', 'enjoy'
BATCH = 8                                            # affect.read's default batch size
W, M = 2400, 36
BODY = 48


def affect_constants():
    tree = ast.parse((ROOT / 'pairing/affect.py').read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'YESNO' for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError('YESNO')


def opening(text, limit=80):
    """The text's first sentences, as many as fit in limit characters (at least one, cut at a word when longer), and an
    ellipsis when anything follows."""
    text = ' '.join(text.split())
    ends = [m.end() for m in re.finditer(r'[.!?](?=\s|$)', text)] or [len(text)]
    cut = max([e for e in ends if e <= limit] or [ends[0]])
    if cut > limit:
        cut = text.rfind(' ', 0, limit)
    out = text[:cut].strip()
    return out + ('\u00a0…' if cut < len(text) else '')     # the ellipsis wraps with the last word


def sigmoid(v):
    return 1 / (1 + math.exp(-v))


def load():
    rows = json.loads(SOURCE.read_text())
    sr = rows['self_report']
    lo = {c: sr[f'{c}|{QUESTION}|{TWIN}'] for c in ('link', 'cut', 'stranger')}
    diffs = [a - b for a, b in zip(lo['link'], lo['cut'])]
    med = statistics.median(diffs)
    i = next(k for k, d in enumerate(diffs) if d == med)
    j = i // BATCH * BATCH + (i % BATCH + 1) % BATCH            # Ember's partner in shuffle mode
    turns = rows['conversations'][i]
    sides = ['quill' if k % 2 == 0 else 'ember' for k in range(len(turns))]
    last = max(k for k, x in enumerate(sides) if x == TWIN)  # ask_episode: up to the twin's last turn
    return dict(checkpoint=CKPT, source=str(SOURCE.relative_to(ROOT)), conversation=i, other_conversation=j,
                median_link_minus_cut=med, n=len(diffs),
                turns=[[x, t] for x, t in zip(sides[:last + 1], turns[:last + 1])],
                other_first_turn=rows['conversations'][j][0], question=affect_constants()[QUESTION],
                logodds={c: v[i] for c, v in lo.items()}, p_yes={c: sigmoid(v[i]) for c, v in lo.items()})


def draw(r):
    tc, ac = (EMBER, QUILL) if TWIN == 'ember' else (QUILL, EMBER)
    title = f'{TWIN.title()}’s probability of answering yes, read three times on the same conversation'
    top = 130
    lx, lw = M, 1200
    rx = lx + lw + 120
    rw = W - M - rx
    RB = 124                                           # robot size beside each bubble
    bw = lw - 2 * 30 - RB - 30
    bubbles = [(x, [(opening(t), INK, 400, None)]) for x, t in r['turns']]
    q_runs = [(r['question'], INK, 700, None)]
    heights = [bubble_height(runs, bw, BODY) for _, runs in bubbles]
    qh = bubble_height(q_runs, bw, BODY)
    gap = 30
    ans_h = 130
    left_h = 40 + sum(heights) + gap * len(heights) + qh + gap + ans_h + 40
    H = round(top + left_h + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    s.text(M, 84, title, 56, 700)

    # the conversation, then the question and the answer read
    s.rect(lx, top, lw, left_h, COOL, 32)
    y = top + 40
    for (x, runs), h in zip(bubbles, heights):
        if x == 'quill':
            robot(s, 'quill', lx + 30, y + h / 2 - RB / 2, RB)
            bubble(s, lx + 30 + RB + 30, y, bw, runs, BODY)
        else:
            robot(s, 'ember', lx + lw - 30 - RB, y + h / 2 - RB / 2, RB)
            bubble(s, lx + 30, y, bw, runs, BODY, tail='right')
        y += h + gap
    robot(s, ASKER, lx + 30, y + qh / 2 - RB / 2, RB)
    bubble(s, lx + 30 + RB + 30, y, bw, q_runs, BODY, stroke=ac)
    y += qh + gap
    robot(s, TWIN, lx + lw - 30 - RB, y + ans_h / 2 - RB / 2, RB)
    cx = lx + lw - 30 - RB - 30
    for k, (word, fill) in enumerate((('No', 'white'), ('Yes', TINT[TWIN]))):
        w = width(word, 52, 700) + 60
        cx -= w
        s.rect(cx, y + 18, w, 84, fill, 18, tc, 4)
        s.text(cx + w / 2, y + 78, word, 52, 700, tc, 'middle')
        if k == 0:
            cx -= 24 + width('or', 48)
            s.text(cx, y + 76, 'or', 48, 400, INK)
            cx -= 24

    # the three readings of the same text: the same words, a different link input
    conds = (('link', 'Link on', WARM), ('cut', 'Link cut', COOL),
             ('stranger', 'Partner from another conversation', COOL))
    size, bar_h = 130, 64
    other = [(opening(r['other_first_turn'], 120), INK, 400, None)]
    other_w = rw - 60 - 40
    other_h = bubble_height(other, other_w, BODY)
    content = {c: 30 + size + 40 + bar_h + 30 + (other_h + 50 if c == 'stranger' else 0) for c, *_ in conds}
    extra = (left_h - 2 * 30 - sum(content.values())) / 3
    # the whole conversation (a bracket) goes into each reading
    bx = lx + lw + 44
    s.path(f'M{lx + lw + 14},{top + 40} H{bx} V{top + left_h - 40} H{lx + lw + 14}', INK, 6)
    cy = top
    for c, label, fill in conds:
        ch = content[c] + extra
        s.rect(rx, cy, rw, ch, fill, 32)
        y = cy + extra / 2 + 30
        pair_icon(s, rx + 30, y, size, c != 'cut', 0.4 * size + 110)
        tx = rx + 30 + 2 * size + 150
        color = LINK if c == 'stranger' else INK
        if c == 'stranger':
            s.text(tx, y + size * 0.62 - 12, 'Partner from', 48, 700, color)
            s.text(tx, y + size * 0.62 + 46, 'another conversation', 48, 700, color)
            y += size + 50
            bubble(s, rx + 30, y, other_w, other, BODY, tail='top')
            y += other_h - size + 10
        else:
            s.text(tx, y + size * 0.62 + 16, label, 48, 700, color)
        by = y + size + 40
        p = r['p_yes'][c]
        num, unit = f'{round(100 * p)}%', ' yes'
        nw = width(num, 72, 700) + width(unit, 48, 700)
        full = rw - 60 - nw - 30
        s.rect(rx + 30, by, full, bar_h, 'white', 12)
        s.rect(rx + 30, by, full * p, bar_h, tc, 12)
        s.rich(rx + 30 + full + 24, by + bar_h / 2 + 26, [(num, tc, 700)], 72)
        s.text(rx + 30 + full + 24 + width(num, 72, 700), by + bar_h / 2 + 26, unit, 48, 700, tc)
        arrow(s, [(bx, cy + ch / 2), (rx - 14, cy + ch / 2)], INK, 6, 24)
        cy += ch + 30
    render(s, OUT)
    return H


def main():
    r = load()
    H = draw(r)
    RESULTS.write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(height=H, conversation=r['conversation'], other=r['other_conversation'],
                          p_yes={k: round(v, 4) for k, v in r['p_yes'].items()})))


if __name__ == '__main__':
    main()
