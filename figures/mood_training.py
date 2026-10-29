"""The twins' mood across training as a picture (nl38, pairing/moodscan.py): after each of 40 free linked
conversations, each twin is asked for a mood in one word, and the reading is the log-odds of happy against sad words
(affect.GOOD, affect.BAD) at the answer's next word, with the link on and with the link cut. Each gauge is the percent of
the 80 answers (40 conversations, both twins) whose reading is higher linked than cut; its mark at the top is half, no
difference between linked and cut. Chat training (nl22c's step 157) forks into the steered-reading game (nl33's step
300, the red arrow) and the same steps without the game (nl33w's step 300). After the game the linked pair is drawn
with lids slanting over the eyes (twin(gloomy=True)): its mood words are lower linked, cut and from the original models
(summary.json mood.<twin>.level). On the red arrow, the game's training example (steered_training.load): the greeting
chat-trained Quill opens with, crossed out, and the opener original Quill writes when told what Ember is thinking, the
training target. Under each pair, Quill's first message in conversation EXAMPLE[run]. A sentence in which a twin says
what its partner is thinking (steer_read.THOUGHT, SENSE, ROLE) is red; a question about the partner's day (GREET) is in
Quill's colour. The note beside the first greeting: conversations whose first message has GREET get higher readings,
linked and from the original models, than the others of the same checkpoint (a correlation). Judge-rewarded training
from the game (beautyrl) is read into results/mood_training.json for the README and not drawn.

--refresh rebuilds results/mood_training.json from moodscan's output directory (--src, default
results/magic-nl38/results: NAME/summary.json, key mood.<twin>.link_minus_cut.higher of n, and NAME/rows.json for the
conversations; only the runs named in RUNS are read) and from steered_training.load(). Without it the figure is drawn
from that file. The steps of the game and of the run without it are the last steps of results/magic-nl33t/log.json and
results/magic-nl33w/log.json. Writes figures/mood_training.{svg,png,pdf}.
Usage: python figures/mood_training.py [--refresh --src MOODSCAN_OUT_DIR]
"""
import argparse
import ast
import json
import math
import re
from pathlib import Path

from steered_training import load as game_example
from chat_names import COOL, INK, LINK, QUILL, RED, ROOT, SVG, WARM, width, wrap
import creature
from method_parts import TINT, arrow, bubble, bubble_height, check_shape, layout, link_icon, pair_icon, render

SNAP = ROOT / 'results/mood_training.json'
OUT = ROOT / 'figures/mood_training'
SRC = ROOT / 'results/magic-nl38/results'
RUNS = ('step157', 'nl33t', 'nl33w', 'beautyrl')           # moodscan's names of the checkpoints read
EXAMPLE = {'step157': 0, 'nl33t': 0, 'nl33w': 4}           # the conversation whose first message is drawn
TWINS = ('quill', 'ember')
GREET = re.compile(r"how['’]?s your day|how is your day", re.I)
W, M, PAD = 2400, 36, 34
BODY, HEAD, BIG = 46, 52, 100
TRACK = '#E2E2E7'


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
        if run in EXAMPLE:
            convs[run] = json.loads((src / run / 'rows.json').read_text())['conversations'][EXAMPLE[run]][:2]
    g = game_example()
    snap = dict(source=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), counts=counts,
                example=EXAMPLE, conversations=convs,
                game=dict(teacher=g['teacher'], told=g['privileged'], target=g['opener'], phrase=g['phrase'],
                          source=g['source']),
                steps=dict(nl33t=last_step(ROOT / 'results/magic-nl33t/log.json'),
                           nl33w=last_step(ROOT / 'results/magic-nl33w/log.json')))
    SNAP.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + '\n')
    return snap


def lead(text, least=30):
    """The text up to its first sentence end (possibly closed by ] or *) or dash at or after least characters, and
    whether anything follows."""
    text = ' '.join(text.split())
    for m in re.finditer(r'[.!?][\]*]?(?=\s|$)|—', text):
        end = m.start() if m.group() == '—' else m.end()
        if end >= least:
            return text[:end], end < len(text)
    return text, False


def marked(text, patterns, least=30):
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
    """The percent of both twins' answers whose mood reading is higher linked than cut."""
    return 100 * sum(c[x]['higher'] for x in TWINS) / sum(c[x]['n'] for x in TWINS)


R, SW = 180, 48                                            # a gauge's radius and stroke
GAUGE_W = 2 * R + SW
PAIR, PAIR_GAP = 76, 30
THAN = 1.6 * PAIR + PAIR_GAP                               # pair_icon's width
GAUGE_H = 64 + SW / 2 + R + 122 + 24 + PAIR * 1.25


def gauge(s, cx, top, pct, color, half_label=False):
    """A half dial filled to pct in color with a mark at half and the percent inside; below, what the percent counts:
    answers that give happier mood words linked than cut. Returns the bottom."""
    cy = top + 64 + SW / 2 + R
    s.add(f'<path d="M{cx - R:.1f},{cy:.1f} A{R},{R} 0 0 1 {cx + R:.1f},{cy:.1f}" fill="none" stroke="{TRACK}" '
          f'stroke-width="{SW}"/>')
    a = math.pi * (1 - pct / 100)
    s.add(f'<path d="M{cx - R:.1f},{cy:.1f} A{R},{R} 0 0 1 {cx + R * math.cos(a):.1f},{cy - R * math.sin(a):.1f}" '
          f'fill="none" stroke="{color}" stroke-width="{SW}"/>')
    s.line(cx, cy - R - SW / 2 - 12, cx, cy - R + SW / 2 + 12, INK, 6, 'butt')
    if half_label:
        s.text(cx, cy - R - SW / 2 - 22, 'no difference', BODY, 400, INK, 'middle')
    s.text(cx, cy - 16, f'{math.floor(pct + 0.5)}%', BIG, 700, color, 'middle')
    s.text(cx, cy + 64, 'of answers give', BODY, 400, INK, 'middle')
    s.text(cx, cy + 122, 'happier mood words', BODY, 700, INK, 'middle')
    row = 2 * THAN + 2 * 22 + width('than', BODY)
    x = cx - row / 2
    py = cy + 122 + 24
    pair_icon(s, x, py, PAIR, True, PAIR_GAP)
    s.text(x + THAN + 22, py + PAIR / 2 + 16, 'than', BODY)
    pair_icon(s, x + THAN + 44 + width('than', BODY), py, PAIR, False, PAIR_GAP)
    return py + PAIR * 1.25


def strike(s, x, y, w, runs, pad=26):
    """Red lines through each line of a bubble's text drawn by bubble(s, x, y, w, runs, BODY, pad)."""
    lead_ = round(BODY * 1.32)
    for k, line in enumerate(layout(runs, w - 2 * pad, BODY)):
        dx, word, _, weight, _ = line[-1]
        ly = y + pad + BODY * 0.84 + k * lead_ - BODY * 0.32
        s.line(x + pad - 8, ly, x + pad + dx + width(word, BODY, weight) + 8, ly, RED, 7)


def fit(runs, most, pad=26):
    """A bubble width that holds runs on as few lines as at most most pixels allow."""
    lines = layout(runs, most - 2 * pad, BODY)
    return min(most, max(line[-1][0] + width(line[-1][1], BODY, line[-1][3]) for line in lines) + 2 * pad + 4)


TWIN, TWIN_GAP = 150, 40                                   # the big pair in each state: box size and gap
TWINS_W = 1.6 * TWIN + TWIN_GAP
GAUGE_COL = max(GAUGE_W, width('happier mood words', BODY, 700), 2 * THAN + 44 + width('than', BODY))


def twin(s, name, x, y, size, gloomy=False):
    """Quill or Ember as method_parts.robot draws it linked; gloomy: a lid slanting down to the outside over each
    open eye (lower mood words), unlike the level lid of the link-cut pose."""
    facing = 1 if name == 'quill' else -1
    h = size * 1.25
    g, _ = creature.creature(name, x + size / 2, y + size / 2 - h / 2, h, look=facing, sway=0.5 * facing)
    if gloomy:
        head = re.search(rf'<ellipse cx="0" cy="([\d.]+)" rx="{creature.HEAD_RX}"', g)
        cy, r, c = float(head.group(1)) - 6, creature.EYE_R + 3, creature.PALETTE[name]
        lids = []
        for side in (-1, 1):
            ex = side * creature.EYE_DX
            a, b = (200, 45) if side < 0 else (135, -20)       # degrees: the left and right ends, the inner one higher
            x1, y1 = ex + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))
            x2, y2 = ex + r * math.cos(math.radians(b)), cy - r * math.sin(math.radians(b))
            lids.append(f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}Z" fill="{c["main"]}"/>'
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c["deep"]}" '
                        f'stroke-width="7" stroke-linecap="round"/>')
        g = g[:-len('</g>')] + ''.join(lids) + '</g>'
    s.add(g)


def twins(s, x, y, gloomy=False):
    """The linked pair at (x, y) in TWIN boxes, as pair_icon draws it; occupies y - TWIN / 8 to y + 9 TWIN / 8."""
    ex = x + 0.6 * TWIN + TWIN_GAP
    link_icon(s, x + TWIN / 2, ex + TWIN / 2, y, TWIN, True)
    twin(s, 'quill', x, y, TWIN, gloomy)
    twin(s, 'ember', ex, y, TWIN, gloomy)


def state(s, x, y, w, h, name, name_color, fill, runs, bw, pct, color, gloomy, note=None):
    """A state of the twins: its name, the linked pair, Quill's first message below Quill, a note beside the message,
    and the gauge at the right."""
    s.rect(x, y, w, h, fill, 28)
    s.text(x + PAD, y + 66, name, HEAD, 700, name_color)
    twins(s, x + PAD, y + 110, gloomy)
    by = y + 110 + 9 * TWIN / 8 + 34
    bh = bubble(s, x + PAD, by, bw, runs, BODY, 26, tail='top')
    if note:
        nx = x + PAD + bw + 60
        lines = wrap(note, x + w - PAD - GAUGE_COL - 60 - nx, BODY)
        for j, line in enumerate(lines):
            s.text(nx, by + bh / 2 - 31 * (len(lines) - 1) + 16 + 62 * j, line, BODY)
    gauge(s, x + w - PAD - GAUGE_COL / 2, y + 40, pct, color, half_label=note is not None)


def draw(snap):
    sr = literals(ROOT / 'pairing/steer_read.py', ('THOUGHT', 'SENSE', 'ROLE'))
    patterns = [sr[k] for k in ('THOUGHT', 'SENSE', 'ROLE')]
    counts, steps, convs = snap['counts'], snap['steps'], snap['conversations']
    assert steps['nl33t'] == steps['nl33w'], steps                 # the run without the game matches the game's steps
    assert GREET.search(convs['step157'][0]) and GREET.search(convs['nl33w'][0])
    assert any(re.search(p, convs['nl33t'][0], re.I) for p in patterns)
    pct = {k: percent(counts[k]) for k in ('step157', 'nl33t', 'nl33w')}
    BP = 26
    opener = {k: marked(convs[k][0], patterns) for k in pct}

    # the two outcomes side by side; the start above both
    ew = (W - 2 * M - 30) / 2
    xl, xr = M, M + ew + 30
    cl, cr = xl + ew / 2, xr + ew / 2                          # the arrows' x
    col = ew - 2 * PAD - 50 - GAUGE_COL                        # the twins' column in an outcome
    bw = {k: fit(opener[k], col) for k in ('nl33t', 'nl33w')}
    bw['step157'] = fit(opener['step157'], 1.6 * col)
    bh = {k: bubble_height(opener[k], bw[k], BODY, BP) for k in bw}
    left = lambda k: 110 + 9 * TWIN / 8 + 34 + bh[k] + PAD
    right = 40 + GAUGE_H + PAD
    start_h = max(left('step157'), right)
    end_h = max(left('nl33t'), left('nl33w'), right)
    # the game's example on the red arrow: the greeting crossed out, then the training target
    g = snap['game']
    greet_runs = [(lead(convs['step157'][0])[0], INK, 400, None)]
    target = marked(g['target'], patterns, least=40)
    pw = cr - cl - 80 - 70                                     # the panel between the arrows
    gw1, gw2 = fit(greet_runs, pw - 2 * PAD), fit(target, pw - 2 * PAD)
    g1h, g2h = bubble_height(greet_runs, gw1, BODY, BP), bubble_height(target, gw2, BODY, BP)
    panel_h = 90 + g1h + 96 + g2h + PAD                        # title, crossed greeting, arrow, target
    fork_h = panel_h + 60

    y_start = M
    y_fork = y_start + start_h
    y_end = y_fork + fork_h
    H = round(y_end + end_h + M)
    check_shape(W, H)
    s = SVG(W, H, title='The twins\u2019 mood after the steered-reading game and without the game')

    state(s, M, y_start, W - 2 * M, start_h, 'After chat training', INK, COOL, opener['step157'], bw['step157'],
          pct['step157'], LINK, False, 'Conversations that open by asking about Ember\u2019s day tend to end with happier mood words')

    # the fork: the red arrow of the game with its example, the black arrow without the game
    arrow(s, [(cl, y_fork + 10), (cl, y_end - 12)], RED, 18, 50)
    arrow(s, [(cr, y_fork + 10), (cr, y_end - 12)], INK, 9, 32)
    s.text(cl - 44, y_fork + fork_h / 2 + 34, 'worse', BIG, 700, RED, 'end')
    px, py = cl + 70, y_fork + 30
    s.rect(px, py, pw, panel_h, WARM, 28, RED, 4)
    s.text(px + PAD, py + 64, 'The steered-reading game', HEAD, 700, RED)
    by = py + 90
    bubble(s, px + PAD, by, gw1, greet_runs, BODY, BP, tail=None)
    strike(s, px + PAD, by, gw1, greet_runs, BP)
    ay = by + g1h
    arrow(s, [(px + PAD + 60, ay + 12), (px + PAD + 60, ay + 84)], RED, 7, 24)
    s.text(px + PAD + 100, ay + 64, 'training target', BODY, 700)
    bubble(s, px + PAD, ay + 96, gw2, target, BODY, BP, stroke=INK, sw=5, tail=None)
    label = wrap(f'Same {steps["nl33w"]} steps without the game', W - M - (cr + 40), HEAD, 700)
    for j, line in enumerate(label):
        s.text(cr + 40, y_fork + fork_h / 2 - 31 * (len(label) - 1) + 18 + 62 * j, line, HEAD, 700)

    state(s, xl, y_end, ew, end_h, 'After the game', RED, WARM, opener['nl33t'], bw['nl33t'], pct['nl33t'], RED,
          True)
    state(s, xr, y_end, ew, end_h, 'Without the game', INK, COOL, opener['nl33w'], bw['nl33w'], pct['nl33w'], LINK,
          False)
    render(s, OUT)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--src', default=str(SRC), help="moodscan's output directory (for --refresh)")
    args = ap.parse_args()
    snap = refresh(args.src) if args.refresh else json.loads(SNAP.read_text())
    H = draw(snap)
    print(json.dumps(dict(height=H, percent={r: round(percent(c), 2) for r, c in snap['counts'].items()})))


if __name__ == '__main__':
    main()
