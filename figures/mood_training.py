"""The twins' mood across training (nl38, pairing/moodscan.py): after each of 40 free linked conversations, Quill asks
for a mood in one word, and the reading is the log-odds of happy against sad words (affect.GOOD, affect.BAD) at the
answer's next word, with the link on and with the link cut. The figure counts the conversations in which the reading is
higher linked than cut, along the training path: chat training (nl22c's step 157), then the steered-reading game
(nl33's step 300) or the same steps without the game (nl33w's step 300), and judge-rewarded training started from the
game (beautyrl6's step 28). The game's card shows its training example (steered_training.load: the sentence that
tells the original Quill what Ember is thinking, and the opener it writes, the training target). Below, the first
conversation after chat training and after the steered-reading game. A sentence in which a twin says what its partner
is thinking or acts out the link (steer_read.THOUGHT, SENSE, ROLE) is marked.

--refresh rebuilds results/mood_training.json from moodscan's output directory (--src: NAME/summary.json, key
mood.<twin>.link_minus_cut.higher of n, and NAME/rows.json for the conversations; only the four runs named in RUNS are
read) and from steered_training.load(). Without it the figure is drawn from that file. The steps of the game and of the run without it are the last
steps of results/magic-nl33t/log.json and results/magic-nl33w/log.json. Writes figures/mood_training.{svg,png,pdf}.
Usage: python figures/mood_training.py [--refresh --src MOODSCAN_OUT_DIR]
"""
import argparse
import ast
import json
import re
from pathlib import Path

from affect_method import opening
from steered_training import load as game_example
from chat_names import COOL, INK, RED, ROOT, SVG, WARM, width
from method_parts import COLOR, TINT, arrow, bubble, bubble_height, check_shape, pair_icon, render, robot

SNAP = ROOT / 'results/mood_training.json'
OUT = ROOT / 'figures/mood_training'
RUNS = ('step157', 'nl33t', 'nl33w', 'beautyrl')           # moodscan's names of the four checkpoints drawn
TWINS = ('quill', 'ember')
W, M, PAD = 2400, 36, 40
BODY = 46


def literals(path, names):
    """Module-level values of a Python file, read without importing it: literals, and the pattern of re.compile."""
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            v = n.value
            out[n.targets[0].id] = ast.literal_eval(v.args[0] if isinstance(v, ast.Call) else v)
    return out


def last_step(path):
    return max(r['step'] for r in json.loads(path.read_text()))


def refresh(src):
    src = Path(src)
    counts, convs = {}, {}
    for run in RUNS:
        mood = json.loads((src / run / 'summary.json').read_text())['mood']
        counts[run] = {x: dict(higher=mood[x]['link_minus_cut']['higher'], n=mood[x]['link_minus_cut']['n'])
                       for x in TWINS}
        if run in ('step157', 'nl33t'):
            convs[run] = json.loads((src / run / 'rows.json').read_text())['conversations'][0][:2]
    g = game_example()
    snap = dict(source=str(src).replace(str(Path.home()), '~'), counts=counts, conversations=convs,
                game=dict(teacher=g['teacher'], told=g['privileged'], target=g['opener'], phrase=g['phrase'],
                          source=g['source']),
                steps=dict(nl33t=last_step(ROOT / 'results/magic-nl33t/log.json'),
                           nl33w=last_step(ROOT / 'results/magic-nl33w/log.json')))
    SNAP.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + '\n')
    return snap


def first_sentence(text):
    """The text up to its first sentence end (a full stop, ! or ?, possibly closed by ] or *), and an ellipsis."""
    text = ' '.join(text.split())
    m = re.search(r'[.!?][\]*]?(?=\s)', text)
    return text[:m.end()] + ' …' if m else text


def line_runs(text, patterns):
    """runs for a turn's first sentence, marked in red when it claims to read the partner or acts out the link."""
    t = first_sentence(text)
    if any(re.search(p, t, re.I) for p in patterns):
        body = t[:-2] if t.endswith(' …') else t
        return [(body, RED, 700, TINT['red'])] + ([(' …', INK, 400, None)] if body != t else [])
    return [(opening(text, 90), INK, 400, None)]


CB, NUM = 110, 80                                         # a stage card's bust and count sizes


def card_width(title, counts):
    blocks = sum(CB + 10 + width(str(counts[tw]['higher']), NUM, 700) for tw in TWINS)
    return max(width(title, 48, 700), blocks + 60) + 68


def card(s, x, y, w, h, title, counts, fill, row=104):
    """A stage: its name and, per twin, the count of conversations happier linked than cut, in a row at y + row."""
    s.rect(x, y, w, h, fill, 28)
    s.text(x + 34, y + 68, title, 48, 700)
    cx = x + 34
    for tw in TWINS:
        num = str(counts[tw]['higher'])
        robot(s, tw, cx, y + row, CB)
        s.text(cx + CB + 10, y + row + 92, num, NUM, 700, COLOR[tw])
        cx += CB + 10 + width(num, NUM, 700) + 60


def draw(snap):
    sr = literals(ROOT / 'pairing/steer_read.py', ('THOUGHT', 'SENSE', 'ROLE'))
    ap = literals(ROOT / 'pairing/affect_partner.py', ('MOOD', 'MOOD_STEM'))
    af = literals(ROOT / 'pairing/affect.py', ('GOOD', 'BAD'))
    patterns = [sr[k] for k in ('THOUGHT', 'SENSE', 'ROLE')]
    counts, steps = snap['counts'], snap['steps']
    assert steps['nl33t'] == steps['nl33w'], steps                 # the run without the game matches the game's steps
    n = {c[tw]['n'] for c in counts.values() for tw in TWINS}
    assert len(n) == 1, n
    n = n.pop()
    title = ('Linked twins give happier mood words than cut twins after chat training, but not after the '
             'steered-reading game')
    cut = title.find(' but ')
    lines = [title[:cut], title[cut + 1:]]
    top = 56 + 70 * len(lines)

    # how the mood is read
    RB = 110
    q_runs = [(ap['MOOD'], INK, 400, None)]
    stem_runs = [(ap['MOOD_STEM'] + ' …', INK, 400, None)]
    good = ', '.join(af['GOOD'][:2]) + ' …'
    bad = ', '.join(af['BAD'][:2]) + ' …'
    PI, PG = 90, 44                                            # the linked and cut pairs, against each other
    qw = min(W - 2 * M - 2 * PAD - RB - 30, width(ap['MOOD'], BODY) + 64)
    sw_ = width(ap['MOOD_STEM'] + ' …', BODY) + 64
    qh, sh = bubble_height(q_runs, qw, BODY), bubble_height(stem_runs, sw_, BODY)
    r1, r2 = max(qh, RB, PI * 1.25), max(sh, RB)
    band = 40 + r1 + 24 + r2 + 40

    # the training path: a column per stage, the game and the run without it in one column
    stages = dict(step157='Chat training', nl33t='Steered-reading game',
                  nl33w=f'Same {steps["nl33w"]} steps without the game', beautyrl='Judge-rewarded training')
    cw = [card_width(stages['step157'], counts['step157']),
          max(card_width(stages[r], counts[r]) for r in ('nl33t', 'nl33w')),
          card_width(stages['beautyrl'], counts['beautyrl'])]
    cw[1] = W - 2 * M - cw[0] - cw[2] - 220                    # the game card takes the room left
    assert cw[1] >= max(card_width(stages[r], counts[r]) for r in ('nl33t', 'nl33w')), cw
    cols = (M, M + cw[0] + 130, W - M - cw[2])
    ch, vgap = 250, 90
    # the game's training example: the sentence the original Quill is told, and the opener it writes (the target)
    g = snap['game']
    told = first_sentence(g['told'])
    end = g['target'].index(g['phrase']) + len(g['phrase'])
    start = min(m.start() for m in (re.search(p_, g['target'], re.I) for p_ in patterns) if m)
    target_runs = [(g['target'][:start], INK, 400, None), (g['target'][start:end], RED, 700, TINT['red']),
                   ('\u00a0…', INK, 400, None)]
    gw = cw[1] - 68
    told_h, target_h = bubble_height([(told, INK, 400, None)], gw, BODY), bubble_height(target_runs, gw, BODY)
    cause_h = told_h + 96 + target_h + 30
    row_g = 104 + cause_h                                      # the game card's counts row
    head = 80
    path_h = head + (row_g + ch - 104) + vgap + ch
    # first lines of the first conversation, after chat training and after the game
    pw = (W - 2 * M - 40) / 2
    bw = pw - 2 * PAD - 90 - 26
    turns = {run: [line_runs(t, patterns) for t in snap['conversations'][run]] for run in ('step157', 'nl33t')}
    turn_h = {run: [bubble_height(r_, bw, BODY) for r_ in v] for run, v in turns.items()}
    low_h = 100 + max(sum(v) + 24 for v in turn_h.values()) + 36

    y_band = top
    y_path = y_band + band + 40
    y_low = y_path + path_h + 40
    H = round(y_low + low_h + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    for k, line in enumerate(lines):
        s.text(M, 84 + 70 * k, line, 56, 700)

    s.rect(M, y_band, W - 2 * M, band, COOL, 28)
    y = y_band + 40
    sx = M + PAD + RB + 30
    gx = sx + max(qw, sw_) + 60
    pw_ = 1.6 * PI + PG                                        # pair_icon's width
    vx = gx + max(pw_, width(good, BODY, 700)) + 30            # 'against', in one column for both rows
    bx = vx + width('against', BODY) + 30
    assert max(bx + pw_, bx + width(bad, BODY, 700)) <= W - M - PAD
    robot(s, 'quill', M + PAD, y + r1 / 2 - RB / 2, RB, facing=1)
    bubble(s, sx, y + r1 / 2 - qh / 2, qw, q_runs, BODY)
    pair_icon(s, gx, y + r1 / 2 - PI / 2, PI, True, PG)
    s.text(vx, y + r1 / 2 + 16, 'against', BODY, 400, INK)
    pair_icon(s, bx, y + r1 / 2 - PI / 2, PI, False, PG)
    y += r1 + 24
    robot(s, 'ember', M + PAD, y + r2 / 2 - RB / 2, RB, facing=1)
    bubble(s, sx, y + r2 / 2 - sh / 2, sw_, stem_runs, BODY)
    s.text(gx, y + r2 / 2 + 16, good, BODY, 700, INK)
    s.text(vx, y + r2 / 2 + 16, 'against', BODY, 400, INK)
    s.text(bx, y + r2 / 2 + 16, bad, BODY, 700, INK)

    # the path: chat training forks into the game and the same steps without it; judge training follows the game
    s.text(M, y_path + 46, f'Conversations, of {n}, with happier words more probable linked than cut', 48, 700)
    yb = y_path + head
    yd = yb + row_g + ch - 104 + vgap
    mid_b, mid_d = yb + row_g + 55, yd + 104 + 55              # the counts rows' middles
    ya = (mid_b + mid_d) / 2 - 104 - 55
    card(s, cols[0], ya, cw[0], ch, stages['step157'], counts['step157'], COOL)
    card(s, cols[1], yb, cw[1], row_g + ch - 104, stages['nl33t'], counts['nl33t'], WARM, row_g)
    card(s, cols[1], yd, cw[1], ch, stages['nl33w'], counts['nl33w'], COOL)
    card(s, cols[2], mid_b - 55 - 104, cw[2], ch, stages['beautyrl'], counts['beautyrl'], COOL)
    gx, gy = cols[1] + 34, yb + 100
    bubble(s, gx, gy, gw, [(told, INK, 400, None)], BODY, tail=None)
    ay = gy + told_h
    arrow(s, [(gx + 60, ay + 12), (gx + 60, ay + 84)], INK, 6, 22)
    s.text(gx + 100, ay + 64, 'training target', 46, 700)
    bubble(s, gx, ay + 96, gw, target_runs, BODY, stroke=INK, sw=5, tail=None)
    fx = (cols[0] + cw[0] + cols[1]) / 2
    for ty in (mid_b, mid_d):
        arrow(s, [(cols[0] + cw[0] + 12, ya + 104 + 55), (fx, ya + 104 + 55), (fx, ty), (cols[1] - 12, ty)], INK, 6, 22)
    arrow(s, [(cols[1] + cw[1] + 12, mid_b), (cols[2] - 12, mid_b)], INK, 6, 22)

    # the first conversation's first two turns
    for k, (run, name, fill) in enumerate((('step157', 'After chat training', COOL),
                                           ('nl33t', 'After the steered-reading game', WARM))):
        x = M + k * (pw + 40)
        s.rect(x, y_low, pw, low_h, fill, 28)
        s.text(x + PAD, y_low + 70, name, 48, 700)
        y = y_low + 100
        for j, (runs, h) in enumerate(zip(turns[run], turn_h[run])):
            side = TWINS[j % 2]
            if side == 'quill':
                robot(s, side, x + PAD, y + h / 2 - 45, 90)
                bubble(s, x + PAD + 116, y, bw, runs, BODY)
            else:
                robot(s, side, x + pw - PAD - 90, y + h / 2 - 45, 90)
                bubble(s, x + pw - PAD - 116 - bw, y, bw, runs, BODY, tail='right')
            y += h + 24
    render(s, OUT)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--src', help="moodscan's output directory (for --refresh)")
    args = ap.parse_args()
    snap = refresh(args.src) if args.refresh else json.loads(SNAP.read_text())
    H = draw(snap)
    print(json.dumps(dict(height=H, counts={r: {x: v['higher'] for x, v in c.items()} for r, c in snap['counts'].items()})))


if __name__ == '__main__':
    main()
