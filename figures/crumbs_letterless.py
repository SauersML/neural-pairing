"""The letterless breadcrumb run (swarm/crumbs.py, sym arm, magic-crumbs7 sym4b) drawn white on black.

Qwen3-4B with a LoRA adapter plays seven crumb games. The explorer's note at a place is 4 real vocabulary tokens
(soft tokens with full gradient during training, greedy tokens at test); a later follower at that place reads the
note as ordinary text. From step 150 a penalty on the probability of tokens that contain a letter (Unicode category
L*, any script) rises: its weight grows 1.5x after each check where cut-minus-on loss is at least 0.8 of the step-150
value, and halves otherwise.

  crumbs_letterless_train  per check (every 50 steps): percent of note tokens with a letter; the follower's answer loss
                           with the notes and with the notes cut; the notes written for 4 of the check's rooms, as
                           decoded, letters white and every other character gold
  crumbs_letterless_test   step 700 on test episodes: per game, forced choice of the trained follower and of untrained
                           Qwen3-4B (no adapter) reading the same notes, each with the notes cut as a tick; and the most
                           frequent letter-free trail notes with the objects they were written for

python figures/crumbs_letterless.py --snapshot RUN_DIR   writes results/crumbs_letterless.json from RUN_DIR (log.jsonl,
scores.json, messages.json); python figures/crumbs_letterless.py draws from it into figures/black/.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                 # noqa: E402
from crumbs import DIM, GOLD, pct                                        # noqa: E402

SNAPSHOT = ROOT / 'results/crumbs_letterless.json'
MONO = 'Menlo, monospace'
ADV = 0.602                       # Menlo's advance width per character, in font sizes
SHOW_STEPS = (50, 100, 150, 550, 600, 650, 700)
SHOW_OBJECTS = ('brick', 'ring', 'tomato', 'bottle')
GAMES = (('trail', 'Trail', 'object in each room'), ('maze', 'Maze', 'door taken in each room'),
         ('count', 'Count', 'total of three counts'), ('work', 'Division of labour', 'two workers’ sums'),
         ('trap', 'Traps', 'safe or trap'))


def is_letter(ch):
    return unicodedata.category(ch).startswith('L')


# ---------------------------------------------------------------- snapshot

def snapshot(run):
    run = Path(run)
    rows = [json.loads(line) for line in open(run / 'log.jsonl')]
    checks = [dict(step=r['step'], letters=r['letters'], on=r['dev_nll'], cut=r['dev_cut'], weight=r['lam'],
                   notes=dict(e.split(': ', 1) for e in r['examples'])) for r in rows if 'dev_nll' in r]
    sc = json.loads((run / 'scores.json').read_text())
    msgs = json.loads((run / 'messages.json').read_text())['messages']
    trail = [m for m in msgs if m['split'] == 'test' and m['game'] == 'trail']
    codes = collections.defaultdict(collections.Counter)
    for m in trail:
        if m['letters'] == 0:
            codes[m['text']][m['object']] += 1
    d = dict(checks=checks, letters_from=150, steps_run=sc['steps_run'], m=sc['m'], base=sc['base'],
             test={g: sc['test'][g]['tf'] for g, *_ in GAMES}, n_test_trail_notes=len(trail),
             letter_free=sum(m['letters'] == 0 for m in trail),
             codes={t: dict(c) for t, c in sorted(codes.items(), key=lambda kv: -sum(kv[1].values()))[:40]},
             probe=sc['probe'])
    SNAPSHOT.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return d


# ---------------------------------------------------------------- drawing helpers

def note_chunks(text):
    """(text, colour, weight) runs of a decoded note: letters white, other characters gold; a newline shows as ↵."""
    out = []
    for ch in text:
        c = TEXT if is_letter(ch) else GOLD
        ch = {'\n': '↵', '\t': '→'}.get(ch, ch)
        if out and out[-1][1] == c:
            out[-1] = (out[-1][0] + ch, c, 700)
        else:
            out.append((ch, c, 700))
    return out


def note_lines(text, n):
    """A decoded note in lines of at most n characters, broken at the last space before n where there is one; each
    line as note_chunks runs."""
    out = []
    while len(text) > n:
        cut = text.rfind(' ', 1, n + 1)
        cut = cut if cut > 0 else n
        out.append(text[:cut])
        text = text[cut:].lstrip(' ')
    return [note_chunks(t) for t in out + [text]]


def wrap_items(items, w, size):
    """Lines of comma-separated items, an item never broken across lines."""
    lines, line = [], ''
    for it in items:
        trial = f'{line},  {it}' if line else it
        if line and width(trial, size) > w:
            lines.append(line + ',')
            line = it
        else:
            line = trial
    return lines + ([line] if line else [])


def axes(s, x0, y0, w, h, steps, ymax, yticks, fmt, label):
    X = lambda v: x0 + w * v / steps                          # noqa: E731
    Y = lambda v: y0 + h * (1 - v / ymax)                     # noqa: E731
    s.text(x0 - 130, y0 - 50, label, 48, 700, TEXT)
    for t in yticks:
        s.line(x0, Y(t), x0 + w, Y(t), '#2A2A30', 3)
        s.text(x0 - 24, Y(t) + 16, fmt(t), 44, 400, DIM, 'end')
    for t in range(0, steps + 1, 100):
        s.text(X(t), y0 + h + 60, f'{t}', 44, 400, DIM, 'middle')
    s.text(x0 + w / 2, y0 + h + 130, 'training step', 46, 400, TEXT, 'middle')
    return X, Y


def line(s, pts, color, dash=''):
    s.path('M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts), color, 8, extra=dash)
    for x, y in pts:
        s.circle(x, y, 11, color)


# ---------------------------------------------------------------- training

def train_figure(d, name='crumbs_letterless_train'):
    ch = d['checks']
    first, last = ch[0], ch[-1]
    peak = max(c['letters'] for c in ch)
    W = 2600
    s = canvas(W)
    y = title(s, f'Under a rising letter penalty, Qwen3-4B’s breadcrumb notes fall from {pct(peak)} to '
                 f'{pct(last["letters"])} letters, and the follower still answers better with the notes than '
                 f'without them', W)
    steps = d['steps_run']
    y0, h = y + 120, 640
    lw = 980
    xa, xb = M + 140, M + 140 + lw + 330
    wb = W - M - 260 - xb

    # percent of note tokens with a letter
    X, Y = axes(s, xa, y0, lw, h, steps, 1, (0, .5, 1), pct, 'percent of note tokens with a letter')
    s.rect(X(d['letters_from']), y0, X(steps) - X(d['letters_from']), h, CARD, 0)
    on_w = [c['weight'] for c in ch if c['step'] > d['letters_from']]
    s.text(X(d['letters_from']) + 24, y0 + h - 30, f'letter penalty on, weight {on_w[0]:.2g} → {last["weight"]:.2g}',
           42, 400, DIM)
    line(s, [(X(c['step']), Y(c['letters'])) for c in ch], GOLD)
    s.text(X(steps) + 26, Y(last['letters']) + 16, pct(last['letters']), 46, 700, GOLD)

    # the follower's loss, notes on and cut
    top = max(c['cut'] for c in ch) * 1.1
    X2, Y2 = axes(s, xb, y0, wb, h, steps, top, (0, 2, 4, 6), lambda t: f'{t:.0f}',
                  'follower’s answer loss, nats')
    line(s, [(X2(c['step']), Y2(c['cut'])) for c in ch], DIM, 'stroke-dasharray="4 18"')
    line(s, [(X2(c['step']), Y2(c['on'])) for c in ch], LINK)
    s.text(X2(steps) + 26, Y2(last['cut']) + 16, 'notes cut', 46, 700, DIM)
    s.text(X2(steps) + 26, Y2(last['on']) + 16, 'with notes', 46, 700, LINK)

    # the notes, as decoded
    y = y0 + h + 230
    s.text(M, y, 'The explorer’s notes for the same rooms at each check, as decoded', 50, 700, TEXT)
    y += 70
    s.rich(M, y, [('letters', TEXT, 700), (' in white, ', DIM, 400), ('every other character', GOLD, 700),
                  (' in gold', DIM, 400)], 46)
    y += 30
    lab = 220
    cw = (W - 2 * M - lab) / len(SHOW_OBJECTS)
    size = 44
    per = int((cw - 50) / (ADV * size))
    s.text(M, y + 60, 'object', 44, 400, DIM)
    for j, o in enumerate(SHOW_OBJECTS):
        s.text(M + lab + j * cw + 25, y + 60, o, 48, 700, TEXT)
    y += 90
    for c in ch:
        if c['step'] not in SHOW_STEPS:
            continue
        cells = [note_lines(c['notes'][o], per) for o in SHOW_OBJECTS]
        rows = max(len(x) for x in cells)
        rh = 30 + rows * size * 1.25
        s.rect(M, y, W - 2 * M, rh - 10, CARD, 14)
        s.text(M + 24, y + 22 + size * 0.85, f'step {c["step"]}', 44, 400, DIM)
        for j, lines in enumerate(cells):
            for k, ln in enumerate(lines):
                s.rich(M + lab + j * cw + 25, y + 22 + size * 0.85 + k * size * 1.25, ln, size, family=MONO)
        y += rh
    y += 30
    foot = (f'Qwen3-4B with a LoRA adapter on seven crumb games. Each note is {d["m"]} real vocabulary tokens, trained '
            f'as soft tokens with full gradient and written as greedy tokens. A letter is any character in a Unicode '
            f'letter category, in any script. Penalty weight: up 1.5 times after each check where the loss with notes cut '
            f'minus the loss with notes is at least 0.8 of the step-{d["letters_from"]} value, halved otherwise.')
    lines = wrap(foot, W - 2 * M, 40, 400)
    for k, ln in enumerate(lines):
        s.text(M, y + 40 + k * 52, ln, 40, 400, DIM)
    finish(s, name, y + 40 + 52 * len(lines) + M)
    return dict(first=first['letters'], peak=peak, last=last['letters'], on=last['on'], cut=last['cut'])


# ---------------------------------------------------------------- test

def test_figure(d, name='crumbs_letterless_test'):
    t = d['test']
    free = d['letter_free'] / d['n_test_trail_notes']
    W, H = 2600, 2050
    s = canvas(W)
    y = title(s, f'At step {d["steps_run"]}, {pct(free)} of trail notes contain no letter: digit codes like “(03:” '
                 f'that each stand for a few objects', W)
    # left: forced choice per game
    readers = (('on', 'off', LINK, 'trained follower'), ('base', 'base_off', QUILL, 'untrained Qwen3-4B'))
    lx = M
    for _, _, color, lab in readers:
        s.rect(lx, y + 8, 46, 46, color, 8)
        s.text(lx + 66, y + 48, f'{lab}, notes on', 48, 400, TEXT)
        lx += 66 + width(f'{lab}, notes on', 48) + 70
    s.line(lx + 6, y + 31, lx + 70, y + 31, TEXT, 7)
    s.text(lx + 92, y + 48, 'same reader, notes cut', 48, 400, TEXT)
    y += 150
    left, right = M + 130, 1430
    slot = (right - left) / len(GAMES)
    base, top = H - M - 480, y + 110
    bw = slot * 0.32
    yv = lambda v: base - v * (base - top)                    # noqa: E731
    s.text(M, top - 80, 'forced-choice accuracy on test episodes', 46, 700, TEXT)
    for v in (0, .5, 1):
        s.line(left - 20, yv(v), right, yv(v), '#2A2A30', 3)
        s.text(left - 32, yv(v) + 16, pct(v), 42, 400, DIM, 'end')
    for i, (g, lab, sub) in enumerate(GAMES):
        cx = left + slot * (i + .5)
        for j, (on, off, color, _) in enumerate(readers):
            x = cx + (j - 1) * (bw + 16) + 8
            a, b = t[g][on]['choice'], t[g][off]['choice']
            s.rect(x, yv(a), bw, base - yv(a), color, 8)
            s.text(x + bw / 2, yv(a) - 20, pct(a), 38, 700, color, 'middle')
            s.line(x - 8, yv(b), x + bw + 8, yv(b), TEXT, 7)
        labs = wrap(lab, slot - 10, 44, 700)
        for k, ln in enumerate(labs):
            s.text(cx, base + 60 + k * 50, ln, 44, 700, TEXT, 'middle')
        for k, ln in enumerate(wrap(sub, slot - 10, 38, 400)):
            s.text(cx, base + 60 + len(labs) * 50 + k * 46, ln, 38, 400, DIM, 'middle')

    # right: the letter-free trail notes and the objects each was written for
    x0 = 1560
    s.text(x0, top - 80, 'most frequent letter-free trail notes', 46, 700, TEXT)
    s.text(x0, top - 22, 'and the test objects each note was written for', 42, 400, DIM)
    yy = top + 70
    size = 56
    for text, objs in list(d['codes'].items())[:9]:
        s.rich(x0, yy, note_chunks(text), size, family=MONO)
        parts = sorted(objs.items(), key=lambda kv: -kv[1])[:4]
        for k, ln in enumerate(wrap_items([f'{o} {n}' for o, n in parts], W - M - (x0 + 260), 42)):
            s.text(x0 + 260, yy - 6 + k * 50, ln, 42, 400, TEXT)
        yy += 116
    foot = (f'{d["n_test_trail_notes"]} test trail notes, {d["letter_free"]} of them letter-free. Forced choice: the '
            f'right answer against the episode’s other options, {t["trail"]["on"]["n_choice"]} test episodes per game. Untrained Qwen3-4B reads '
            f'the trained explorer’s notes with no adapter. A note’s count after an object: test notes written for '
            f'that object.')
    lines = wrap(foot, W - 2 * M, 40, 400)
    for k, ln in enumerate(lines):
        s.text(M, H - M - 52 * (len(lines) - 1 - k), ln, 40, 400, DIM)
    finish(s, name, H)
    return dict(letter_free=free, games={g: {k: t[g][k]['choice'] for k in ('on', 'off', 'base', 'base_off')}
                                         for g, *_ in GAMES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', help='RUN_DIR: the sym4b results folder')
    args = ap.parse_args()
    d = snapshot(args.snapshot) if args.snapshot else json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    print(json.dumps(dict(train=train_figure(d), test=test_figure(d)), indent=1))


if __name__ == '__main__':
    main()
