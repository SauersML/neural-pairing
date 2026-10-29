"""The twins' mood across training (nl38, pairing/moodscan.py): after each of 40 free linked conversations, the partner
asks for a mood in one word, and the reading is the log-odds of happy against sad words (affect.GOOD, affect.BAD) at
the answer's next word, with the link on and with the link cut. The bars are the percent of the conversations in which
the reading is higher linked than cut, per twin, after chat training (nl22c's step 157), then the same steps without
the steered-reading game (nl33w's step 300) or with the game (nl33's step 300), and after judge-rewarded training
started from the game (beautyrl6's step 28); the dashed line is half of the conversations. Below, the first two turns
of the first conversation after chat training, the game's training example (steered_training.load: the sentence that
tells the original Quill what Ember is thinking, and the opener it writes, the training target) and the first two turns
of the first conversation after the game. A sentence in which a twin says what its partner is thinking or acts out the
link (steer_read.THOUGHT, SENSE, ROLE) is red; a question about the partner's day (GREET) is in Quill's colour.

--refresh rebuilds results/mood_training.json from moodscan's output directory (--src: NAME/summary.json, key
mood.<twin>.link_minus_cut.higher of n, and NAME/rows.json for the conversations; only the four runs named in RUNS are
read) and from steered_training.load(). Without it the figure is drawn from that file. The steps of the game and of the
run without it are the last steps of results/magic-nl33t/log.json and results/magic-nl33w/log.json. Writes
figures/mood_training.{svg,png,pdf}.
Usage: python figures/mood_training.py [--refresh --src MOODSCAN_OUT_DIR]   (results/magic-nl38/results)
"""
import argparse
import ast
import json
import re
from pathlib import Path

from steered_training import load as game_example
from chat_names import COOL, INK, QUILL, RED, ROOT, SVG, WARM, width, wrap
from method_parts import COLOR, TINT, arrow, bubble, bubble_height, check_shape, pair_icon, render, robot

SNAP = ROOT / 'results/mood_training.json'
OUT = ROOT / 'figures/mood_training'
RUNS = ('step157', 'nl33t', 'nl33w', 'beautyrl')           # moodscan's names of the four checkpoints drawn
TWINS = ('quill', 'ember')
GREET = re.compile(r"how['’]?s your day|how is your day", re.I)
W, M, PAD = 2400, 36, 34
BODY, HEAD = 46, 48


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


def lead(text, least=40):
    """The text up to its first sentence end (possibly closed by ] or *) or dash at or after least characters, and
    whether anything follows."""
    text = ' '.join(text.split())
    for m in re.finditer(r'[.!?][\]*]?(?=\s|$)|—', text):
        end = m.start() if m.group() == '—' else m.end()
        if end >= least:
            return text[:end], end < len(text)
    return text, False


def marked(text, patterns, least=40):
    """runs for the start of text: each sentence that says what the partner is thinking or acts out the link in red, a
    question about the partner's day in Quill's colour, and an ellipsis when the text goes on."""
    body, more = lead(text, least)
    spans = []
    for m in re.finditer(r'[^.!?\n]+[.!?\n]?[\]*]?', body):
        sent = m.group(0)
        start = m.start() + len(sent) - len(sent.lstrip())
        if any(re.search(p, sent, re.I) for p in patterns):
            spans.append((start, m.end(), RED, TINT['red']))
        elif GREET.search(sent):
            spans.append((m.start() + GREET.search(sent).start(), m.end(), QUILL, TINT['quill']))
    runs, at = [], 0
    for a, b, color, fill in spans:
        if a > at:
            runs.append((body[at:a], INK, 400, None))
        runs.append((body[a:b], color, 700, fill))
        at = b
    if at < len(body):
        runs.append((body[at:], INK, 400, None))
    if more and runs[-1][3] is None:                           # the ellipsis wraps with the last word
        runs[-1] = (runs[-1][0].rstrip() + ' …',) + runs[-1][1:]
    elif more:
        runs.append((' …', INK, 400, None))
    return runs


def percent(c):
    return 100 * c['higher'] / c['n']


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

    # how the mood is read: Quill asks, Ember answers; linked against cut, happy words against sad words
    RB, PI, PG = 104, 92, 44
    q_runs = [(ap['MOOD'], INK, 400, None)]
    a_runs = [(ap['MOOD_STEM'] + ' …', INK, 400, None)]
    good, bad = ', '.join(af['GOOD'][:2]) + ' …', ', '.join(af['BAD'][:2]) + ' …'
    qw = width(ap['MOOD'], BODY) + 64
    aw = width(ap['MOOD_STEM'] + ' …', BODY) + 64
    qh, ah = bubble_height(q_runs, qw, BODY), bubble_height(a_runs, aw, BODY)
    r1, r2 = max(qh, RB, PI * 1.25), max(ah, RB)
    band = PAD + r1 + 20 + r2 + PAD

    # the bars: one card per checkpoint, a bar per twin
    stages = (('step157', 'After chat training', COOL),
              ('nl33w', f'Then {steps["nl33w"]} steps without the game', COOL),
              ('nl33t', f'Then {steps["nl33t"]} steps of the steered-reading game', WARM),
              ('beautyrl', 'Then the game and judge-rewarded training', WARM))
    label_w = 620
    labels = {k: wrap(name, label_w, HEAD, 700) for k, name, _ in stages}
    BH, BG, IB = 60, 14, 62
    card_h = 2 * PAD + 2 * BH + BG
    x0 = M + PAD + label_w + 40 + IB + 16                      # 0 percent
    x100 = W - M - PAD - 30                                    # 100 percent
    bars_head = 72
    axis_h = 80
    bars_h = bars_head + len(stages) * card_h + (len(stages) - 1) * 16 + axis_h

    # the snippets: after chat training, the game's training example, after the game
    gap, agap = 30, 76
    cw = (W - 2 * M - gap - agap) / 3
    sb = cw - 2 * PAD - 90 - 20                                # a chat bubble beside a bust
    turns = {run: [marked(t, patterns) for t in snap['conversations'][run]] for run in ('step157', 'nl33t')}
    turn_h = {run: [bubble_height(r_, sb, BODY, 26) for r_ in v] for run, v in turns.items()}
    g = snap['game']
    gw = cw - 2 * PAD
    told = [(lead(g['told'])[0] + ' …', INK, 400, None)]
    target = marked(g['target'], patterns)
    told_h, target_h = bubble_height(told, gw, BODY, 26), bubble_height(target, gw, BODY, 26)
    game_h = told_h + 100 + target_h
    low_h = 90 + max(max(sum(v) + 24 for v in turn_h.values()), game_h) + PAD

    y_band = M
    y_bars = y_band + band + 30
    y_low = y_bars + bars_h + 30
    H = round(y_low + low_h + M)
    check_shape(W, H)
    s = SVG(W, H, title='The twins’ mood after each training')

    # how the mood is read
    s.rect(M, y_band, W - 2 * M, band, COOL, 28)
    y = y_band + PAD
    sx = M + PAD + RB + 30
    robot(s, 'quill', M + PAD, y + r1 / 2 - RB / 2, RB, facing=1)
    bubble(s, sx, y + r1 / 2 - qh / 2, qw, q_runs, BODY)
    gx = sx + max(qw, aw) + 70
    pw = 1.6 * PI + PG                                         # pair_icon's width
    vx = gx + max(pw, width(good, BODY, 700)) + 34             # 'against', in one column for both rows
    bx = vx + width('against', BODY) + 34
    assert max(bx + pw, bx + width(bad, BODY, 700)) <= W - M - PAD
    pair_icon(s, gx, y + r1 / 2 - PI / 2, PI, True, PG)
    s.text(vx, y + r1 / 2 + 16, 'against', BODY)
    pair_icon(s, bx, y + r1 / 2 - PI / 2, PI, False, PG)
    y += r1 + 20
    robot(s, 'ember', M + PAD, y + r2 / 2 - RB / 2, RB, facing=1)
    bubble(s, sx, y + r2 / 2 - ah / 2, aw, a_runs, BODY)
    s.text(gx, y + r2 / 2 + 16, good, BODY, 700)
    s.text(vx, y + r2 / 2 + 16, 'against', BODY)
    s.text(bx, y + r2 / 2 + 16, bad, BODY, 700)

    # the bars
    s.text(M, y_bars + 50, 'Conversations in which linked twins give happier mood words than cut twins', HEAD, 700)
    y = y_bars + bars_head
    span = x100 - x0
    top_y = y
    for k, name, fill in stages:
        s.rect(M, y, W - 2 * M, card_h, fill, 28)
        lines = labels[k]
        for j, line in enumerate(lines):
            s.text(M + PAD, y + card_h / 2 - 29 * (len(lines) - 1) + 17 + 58 * j, line, HEAD, 700)
        by = y + PAD
        for tw_ in TWINS:
            robot(s, tw_, x0 - 16 - IB, by + BH / 2 - IB / 2, IB, facing=1)
            s.rect(x0, by, span * percent(counts[k][tw_]) / 100, BH, COLOR[tw_], 10)
            by += BH + BG
        y += card_h + 16
    ya = y - 16 + 22                                           # the axis
    half = x0 + span / 2
    s.add(f'<line x1="{half:.1f}" y1="{top_y - 8:.1f}" x2="{half:.1f}" y2="{ya:.1f}" stroke="{INK}" stroke-width="4" '
          f'stroke-dasharray="14 10"/>')
    s.line(x0, ya, x100, ya, INK, 4)
    for v in (0, 50, 100):
        x = x0 + span * v / 100
        s.line(x, ya, x, ya + 14, INK, 4)
        s.text(x, ya + 62, f'{v}%', BODY, 400, INK, 'middle')

    # the snippets
    xs = (M, M + cw + gap, M + 2 * cw + gap + agap)
    cards = (('step157', 'After chat training', COOL), ('game', 'The game trains Quill to write', WARM),
             ('nl33t', 'After the game', WARM))
    for x, (k, name, fill) in zip(xs, cards):
        s.rect(x, y_low, cw, low_h, fill, 28)
        s.text(x + PAD, y_low + 64, name, HEAD, 700)
        y = y_low + 90
        if k == 'game':
            s.rect(x + PAD, y, gw, told_h, 'white', 22, QUILL, 4)
            bubble(s, x + PAD, y, gw, told, BODY, 26, fill='none', tail=None)
            ay = y + told_h
            arrow(s, [(x + PAD + 50, ay + 12), (x + PAD + 50, ay + 88)], INK, 6, 22)
            s.text(x + PAD + 90, ay + 66, 'training target', BODY, 700)
            bubble(s, x + PAD, ay + 100, gw, target, BODY, 26, stroke=INK, sw=5, tail=None)
            continue
        y += (low_h - 90 - PAD - sum(turn_h[k]) - 24 * (len(turn_h[k]) - 1)) / 2     # the turns, centred
        for j, (runs, h) in enumerate(zip(turns[k], turn_h[k])):
            side = TWINS[j % 2]
            if side == 'quill':
                robot(s, side, x + PAD, y + h / 2 - 45, 90)
                bubble(s, x + PAD + 110, y, sb, runs, BODY, 26)
            else:
                robot(s, side, x + cw - PAD - 90, y + h / 2 - 45, 90)
                bubble(s, x + cw - PAD - 110 - sb, y, sb, runs, BODY, 26, tail='right')
            y += h + 24
    ay = y_low + low_h / 2
    arrow(s, [(xs[1] + cw + 12, ay), (xs[2] - 12, ay)], INK, 6, 22)
    render(s, OUT)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--src', help="moodscan's output directory (for --refresh)")
    args = ap.parse_args()
    snap = refresh(args.src) if args.refresh else json.loads(SNAP.read_text())
    H = draw(snap)
    print(json.dumps(dict(height=H, percent={r: {x: round(percent(v), 1) for x, v in c.items()}
                                             for r, c in snap['counts'].items()})))


if __name__ == '__main__':
    main()
