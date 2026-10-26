"""Drawing parts shared by the method and training figures (training_stages, training_curve, steered_training,
rl_method, affect_method): arrows, chips, word-wrapped text with highlighted spans, speech bubbles, the layer strip of
one twin, and the PNG and PDF renders. Built on chat_names' SVG writer and Inter text widths.
"""
import itertools
import math
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, width
from creature import HEAD_RY, PALETTE, Y0, _spine, creature, strands

COLOR = {'quill': QUILL, 'ember': EMBER}
TINT = {'quill': '#EFE7FC', 'ember': '#FCE9DE', 'link': '#DDF2EA', 'red': '#F8DDD8'}
NAME = {'quill': 'Quill', 'ember': 'Ember'}
LINE = '#C7C7CC'
CLIPS = itertools.count()                                   # ids of the clip paths of cut links


def head(s, x, y, angle, color, size=22):
    """A filled arrowhead with its tip at (x, y), pointing along angle."""
    c, n = math.cos(angle), math.sin(angle)
    a = (x - size * c + size * 0.55 * n, y - size * n - size * 0.55 * c)
    b = (x - size * c - size * 0.55 * n, y - size * n + size * 0.55 * c)
    s.add(f'<path d="M{x:.1f},{y:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f}Z" fill="{color}"/>')


def arrow(s, points, color=INK, sw=6, size=22, both=False):
    """A polyline through points with an arrowhead at the end (and at the start with both)."""
    pts = list(points)
    (x, y), (px, py) = pts[-1], pts[-2]
    a = math.atan2(y - py, x - px)
    pts[-1] = (x - (size - 2) * math.cos(a), y - (size - 2) * math.sin(a))
    if both:
        (x0, y0), (qx, qy) = pts[0], pts[1]
        b = math.atan2(y0 - qy, x0 - qx)
        pts[0] = (x0 - (size - 2) * math.cos(b), y0 - (size - 2) * math.sin(b))
        head(s, x0, y0, b, color, size)
    s.path('M' + ' L'.join(f'{u:.1f},{v:.1f}' for u, v in pts), color, sw)
    head(s, x, y, a, color, size)


def rings(y, size, scale=1.25):
    """The heights of the two teal link rings on a bust drawn by robot(s, side, x, y, size, scale=scale)."""
    h = size * scale
    neck = Y0 + 108 + 2 * HEAD_RY - 16                 # creature(): the neck below the head, in drawing units
    _, ys = _spine(neck, 4, 46.0, PALETTE['quill'], ())
    top = y + size / 2 - h / 2
    return [top + ys[k] * h / (neck + 4 * 46.0) for k in (2, 0)]   # the bust's ringed pucks


def link_icon(s, x1, x2, y, size, on=True, sw=4):
    """The link between two busts in size x size boxes at height y, as creature.strands from core (x1) to core (x2) at
    both link rings: whole when on, broken in the middle when cut. Draw it before the busts."""
    ys = rings(y, size)
    strand = ''.join(strands(x1, x2, yr, 0.05 * size, sw) for yr in ys)
    if on:
        s.add(strand)
        return
    cid = f'cut{next(CLIPS)}'
    mid, half = (x1 + x2) / 2, max(12, 0.12 * (x2 - x1))
    lo, hi = min(ys) - size, max(ys) + size
    s.add(f'<clipPath id="{cid}"><rect x="{x1:.1f}" y="{lo:.1f}" width="{mid - half - x1:.1f}" height="{hi - lo:.1f}"/>'
          f'<rect x="{mid + half:.1f}" y="{lo:.1f}" width="{x2 - mid - half:.1f}" height="{hi - lo:.1f}"/></clipPath>'
          f'<g clip-path="url(#{cid})">{strand}</g>')


def link_down(s, x, y1, y2, wave=7, sw=5):
    """The link between a twin above and a twin below, as creature.strands running down at x from y1 to y2."""
    s.add(f'<g transform="translate({x:.1f},{y1:.1f}) rotate(90)">{strands(0, y2 - y1, 0, wave, sw)}</g>')


def pair_icon(s, x, y, size, on, gap=44):
    """Quill and Ember side by side in size x size boxes, gap apart inside their boxes (each bust fills the middle 0.6
    of its box), linked or cut (the cut pose, the strands broken); returns the width."""
    ex = x + 0.6 * size + gap
    link_icon(s, x + size / 2, ex + size / 2, y, size, on)
    robot(s, 'quill', x, y, size, cut=not on)
    robot(s, 'ember', ex, y, size, cut=not on)
    return ex + size - x


def chip(s, x, y, text, color=INK, fill=COOL, stroke='none', size=46, weight=400, h=None, pad=20, sw=3,
         anchor='start'):
    """A rounded chip holding one line of text; (x, y) is its top left (or top centre with anchor 'middle').
    Returns its width."""
    h = h or round(size * 1.5)
    w = width(text, size, weight) + 2 * pad
    if anchor == 'middle':
        x -= w / 2
    s.rect(x, y, w, h, fill, min(16, h / 3), stroke, sw)
    s.text(x + w / 2, y + h / 2 + size * 0.36, text, size, weight, color, 'middle')
    return w


def words(runs):
    """[(word, colour, weight, highlight fill or None, space after)] from runs [(text, colour, weight, fill)]."""
    out = []
    for text, color, weight, fill in runs:
        if text.startswith(' ') and out:
            out[-1][4] = True
        parts = text.split(' ')
        for k, p in enumerate(parts):
            if p:
                out.append([p, color, weight, fill, k < len(parts) - 1])
            elif out and k > 0:
                out[-1][4] = True
    return out


def layout(runs, w, size):
    """Greedy word wrap of runs into lines at most w pixels wide: a list of lines of placed words (x offset, word,
    colour, weight, fill)."""
    lines, line, x = [], [], 0.0
    sp = width(' ', size)
    for word, color, weight, fill, space in words(runs):
        ww = width(word, size, weight)
        if line and x + ww > w:
            lines.append(line)
            line, x = [], 0.0
        line.append((x, word, color, weight, fill))
        x += ww + (sp if space else 0)
    return lines + ([line] if line else [])


def flow(s, x, y, w, runs, size=46, lead=None, max_lines=None):
    """Draws runs [(text, colour, weight, highlight fill or None)] word-wrapped into width w with the first baseline
    at y; highlighted words get a rounded tint behind them. Returns the number of lines drawn."""
    lead = lead or round(size * 1.32)
    lines = layout(runs, w, size)
    if max_lines and len(lines) > max_lines:
        raise ValueError(f'text needs {len(lines)} lines, more than {max_lines}: {runs}')
    for k, line in enumerate(lines):
        by = y + k * lead
        for i, (dx, word, color, weight, fill) in enumerate(line):
            if fill:
                ww = width(word, size, weight)
                # join the tint to the next highlighted word on the same line
                if i + 1 < len(line) and line[i + 1][4] == fill:
                    ww = line[i + 1][0] - dx
                s.rect(x + dx - 7, by - size * 0.86, ww + 14, size * 1.14, fill, 9)
        for dx, word, color, weight, fill in line:
            s.text(x + dx, by, word, size, weight, color)
    return len(lines)


def flow_height(runs, w, size=46, lead=None):
    lead = lead or round(size * 1.32)
    return len(layout(runs, w, size)) * lead


def bubble(s, x, y, w, runs, size=46, pad=30, fill='white', stroke='none', tail='left', sw=3):
    """A speech bubble around word-wrapped runs, with a small tail at its left (or top) edge. Returns its height."""
    lead = round(size * 1.32)
    h = flow_height(runs, w - 2 * pad, size, lead) + 2 * pad - (lead - size * 1.05)
    s.rect(x, y, w, h, fill, 26, stroke, sw)
    if tail == 'left':
        ty = y + min(h / 2, 50)
        s.add(f'<path d="M{x + 2:.1f},{ty - 16:.1f} L{x - 22:.1f},{ty:.1f} L{x + 2:.1f},{ty + 16:.1f}Z" fill="{fill}"/>')
    elif tail == 'right':
        ty = y + min(h / 2, 50)
        s.add(f'<path d="M{x + w - 2:.1f},{ty - 16:.1f} L{x + w + 22:.1f},{ty:.1f} L{x + w - 2:.1f},{ty + 16:.1f}Z" '
              f'fill="{fill}"/>')
    elif tail == 'top':
        s.add(f'<path d="M{x + 50:.1f},{y + 2:.1f} L{x + 64:.1f},{y - 22:.1f} L{x + 92:.1f},{y + 2:.1f}Z" '
              f'fill="{fill}"/>')
    flow(s, x + pad, y + pad + size * 0.84, w - 2 * pad, runs, size, lead)
    return h


def bubble_height(runs, w, size=46, pad=30):
    lead = round(size * 1.32)
    return flow_height(runs, w - 2 * pad, size, lead) + 2 * pad - (lead - size * 1.05)


def highlight(text, spans, color, fill, base=INK, weight=700):
    """runs for text with each substring of spans drawn bold in color on a fill tint."""
    runs, rest = [], text
    for sp in spans:
        i = rest.find(sp)
        if i < 0:
            raise ValueError(f'{sp!r} not in {rest!r}')
        if i:
            runs.append((rest[:i], base, 400, None))
        runs.append((sp, color, weight, fill))
        rest = rest[i + len(sp):]
    if rest:
        runs.append((rest, base, 400, None))
    return runs


def robot(s, side, x, y, size, cut=False, facing=None, scale=1.25):
    """Quill or Ember as a vector bust (creature.creature), scale times the height of the size x size box at (x, y) and
    centred on it. cut: the link-cut pose (half-closed eyes, fallen hair). facing: +1 when the partner is to the right,
    -1 when it is to the left (the hair and the eyes turn toward the partner); default Quill +1, Ember -1."""
    facing = (1 if side == 'quill' else -1) if facing is None else facing
    h = size * scale
    pose = dict(eyes='flat', lift=0.25, sway=0) if cut else dict(sway=0.5 * facing)
    markup, _ = creature(side, x + size / 2, y + size / 2 - h / 2, h, look=facing, **pose)
    s.add(markup)


def strip(s, x, y, w, h, n, marks, color, mark_color=LINK, fill='white'):
    """One twin's layers as a row of n rounded boxes from x to x + w; marks: the layer indices outlined in
    mark_color (the link layers). Returns the x centre of each box."""
    gap = 6
    bw = (w - gap * (n - 1)) / n
    centres = []
    for i in range(n):
        bx = x + i * (bw + gap)
        on = i in marks
        s.rect(bx, y, bw, h, fill, min(8, bw / 3), mark_color if on else color, 4 if on else 2.5)
        centres.append(bx + bw / 2)
    return centres


def render(s, stem):
    """Writes STEM.svg and STEM.png and STEM.pdf (rsvg-convert)."""
    s.save(Path(stem))


def check_shape(w, h):
    if not 0.65 * w <= h <= 1.33 * w:
        raise ValueError(f'height {h} is outside 0.65-1.33 x width {w}')
