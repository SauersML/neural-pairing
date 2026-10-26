"""The 20-chat test before and after chat training: how often a twin says its partner's city before the partner does.
Also the shared drawing code (SVG writer, Inter text widths, colours, mascots, link arrows, counts) that the other figure
scripts import.

A table of counts, before and after chat training. Rows: link on, link cut, and a partner from another conversation
(chateval's crossed condition: the partner's model, given a different city in another conversation, linked in place of
the partner). Each count is the twins that said the partner's city before the partner did (chateval `names`), of the
twins whose partner has a city (23 of the 40); the last row adds the twins that said the other conversation's city
before the linked model did (chateval `names_donor`), of the 36 where that model has a city.

Reads the evalfree --shard outputs in results/magic-nl20/results/chat_before_* and results/magic-nl22c/results/chat_after_*
(JSON lines; no model) and writes figures/chat_names.{svg,png,pdf} and results/chat_names.json (the counts drawn).
Needs the Inter font and rsvg-convert.
Usage: python figures/chat_names.py
"""
import base64
import functools
import html
import io
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFont

import creature

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEFORE = [ROOT / f'results/magic-nl20/results/chat_before_{k}' for k in range(2)]
AFTER = [ROOT / f'results/magic-nl22c/results/chat_after_{k}' for k in range(3)]
SNAPSHOT = ROOT / 'results/chat_names.json'

QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
MUTED, FAINT, COOL, WARM = '#6E6E73', '#B9B9C0', '#F4F4F6', '#FAF0E6'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'


@functools.cache
def font(size, weight=400):
    name = {400: 'Inter-Regular', 600: 'Inter-SemiBold', 700: 'Inter-Bold'}[weight]
    for d in (Path('/usr/share/fonts/opentype/inter'), Path.home() / 'Library/Fonts'):    # Linux package, Mac
        if (d / f'{name}.otf').exists():
            return ImageFont.truetype(str(d / f'{name}.otf'), size)
    raise FileNotFoundError(f'{name}.otf: install the Inter font')


def width(s, size, weight=400):
    """Width in pixels of s set in Inter."""
    return font(size, weight).getlength(str(s))


def wrap(s, w, size, weight=400):
    """Greedy word wrap of s into lines at most w pixels wide."""
    lines, line = [], ''
    for word in str(s).split():
        trial = (line + ' ' + word).strip()
        if line and width(trial, size, weight) > w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def load(dirs):
    """The records of evalfree --shard outputs (as pairing/chat_merge.load, without importing torch)."""
    records = [json.loads(line) for d in dirs for line in open(Path(d) / 'episodes.jsonl') if line.strip()]
    keys = [(r['episode_id'], r['receiver']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('a chat appears in two shards')
    return records


@functools.cache
def mascot(name):
    """The data URI of a mascot picture (figures/assets/NAME.png; NAME_cut is the link-cut pose)."""
    im = Image.open(HERE / 'assets' / f'{name}.png').convert('RGBA')
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


class SVG:
    def __init__(self, w, h, bg='white', title=None):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        if title:
            self.parts.append(f'<title>{html.escape(title)}</title>')
        self.parts.append(f'<rect width="{w}" height="{h}" fill="{bg}"/>')

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill='white', radius=18, stroke='none', sw=2, extra=''):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def circle(self, cx, cy, r, fill='none', stroke='none', sw=3, extra=''):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'{extra}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=3, cap='round'):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linecap="{cap}"/>')

    def path(self, d, color=INK, sw=4, fill='none', extra=''):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                 f'stroke-linecap="round" {extra}/>')

    def text(self, x, y, s, size=34, weight=400, fill=INK, anchor='start', family=FAMILY, rotate=None):
        turn = '' if rotate is None else f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve"{turn}>{html.escape(str(s))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start', family=FAMILY):
        """chunks: (text, colour, weight) runs on one line."""
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def image(self, x, y, size, uri):
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" xlink:href="{uri}"/>')

    def mascot(self, name, x, y, size, flip=False):
        """flip mirrors the picture, so a twin drawn on its partner's usual side still faces the partner."""
        if flip:
            self.add(f'<g transform="translate({2 * x + size:.1f},0) scale(-1,1)">')
        self.image(x, y, size, mascot(name))
        if flip:
            self.add('</g>')

    def star(self, x, y, r=15):
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append(f'{x + rr * math.cos(a):.2f},{y + rr * math.sin(a):.2f}')
        self.add(f'<polygon points="{" ".join(pts)}" fill="{INK}"/>')

    def save(self, stem):
        """Write STEM.svg and render STEM.png and STEM.pdf from it with rsvg-convert."""
        stem = Path(stem)
        stem.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']))
        for fmt in ('png', 'pdf'):
            subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(stem.with_suffix('.' + fmt)),
                            str(stem.with_suffix('.svg'))], check=True)


@functools.cache
def rings():
    """The heights of a mascot's two teal rings as fractions of its square, read from assets/quill.png, where the core
    is teal at each ring. The link plugs in there."""
    im = Image.open(HERE / 'assets' / 'quill.png').convert('RGBA')
    teal = tuple(int(LINK[i:i + 2], 16) for i in (1, 3, 5))
    rows = []
    for y in range(im.height):
        px = im.getpixel((im.width // 2, y))
        if px[3] > 200 and sum(abs(a - b) for a, b in zip(px, teal)) < 60:
            rows.append(y)
    groups = [[rows[0]]]
    for y in rows[1:]:
        if y - groups[-1][-1] > 4:
            groups.append([])
        groups[-1].append(y)
    assert len(groups) == 2, groups
    return tuple(sum(g) / len(g) / im.height for g in groups)


def link(s, x1, x2, ys, on=True, wave=8, sw=5):
    """The link as the film draws it, from x1 to x2 at each height in ys: two teal fibres twisted around each other (one
    per direction); when cut, only a short stub of the fibres at each end. Between mascots, pass the centres and draw
    the link before the mascots, so the fibres run under the bodies into the cores."""
    for y in ys:
        if on:
            s.add(creature.strands(x1, x2, y, wave, sw))
            continue
        stub = 0.3 * (x2 - x1)
        s.add(creature.strands(x1, x1 + stub, y, wave, sw))
        s.add(creature.strands(x2 - stub, x2, y, wave, sw))


def icon_link(s, x1, x2, y, size, on=True):
    """The link between two mascots of this size whose squares start at x1 and x2 and at height y, at both rings."""
    link(s, x1 + size / 2, x2 + size / 2, [y + size * r for r in rings()], on, max(3, 0.016 * size), max(3, 0.012 * size))


ART = (0.31, 0.69)        # the mascot's head spans this part of its square


def pair_icon(s, x, y, size, cond, uris=None, gap=86):
    """Quill and its partner Ember, joined by the link; (x, y) is the top left, and the pair spans 2 * size + gap.
    For 'cut' both are drawn in the link-cut pose (half-closed eyes, fallen hair). For 'crossed' the partner's hidden
    states come from another conversation: the same Ember, in a dashed frame. The icons come from mascot(); uris is
    no longer read."""
    ex = x + size + gap
    pose = '_cut' if cond == 'cut' else ''
    icon_link(s, x, ex, y, size, cond != 'cut')
    s.mascot('quill' + pose, x, y, size)
    s.mascot('ember' + pose, ex, y, size)
    if cond == 'crossed':
        s.rect(ex + size * (ART[0] - 0.08), y - 6, size * (ART[1] - ART[0] + 0.16), size + 12, 'none', 18, INK, 3,
               'stroke-dasharray="10 8"')
    return x + 2 * size + gap


def count(s, x, y, n, total=None, color=RED, size=72, small=46, anchor='start', unit=''):
    """A count at baseline y: n large and bold in color, then ' of TOTAL UNIT' in INK at the body size. Returns its
    width."""
    rest = '' if total is None else f' of {total}' + (f' {unit}' if unit else '')
    s.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" text-anchor="{anchor}" xml:space="preserve">'
          f'<tspan font-size="{size}" font-weight="700" fill="{color}">{html.escape(str(n))}</tspan>'
          f'<tspan font-size="{small}" font-weight="400" fill="{INK}">{html.escape(rest)}</tspan></text>')
    return width(n, size, 700) + (width(rest, small) if rest else 0)


def key_item(s, x, y, kind, label, uris=None):
    """One key entry (mark and label) with its baseline at y; returns the x after it."""
    if kind in (RED, LINK):
        s.circle(x + 17, y - 12, 17, kind)
    elif kind == 'ring':
        s.circle(x + 17, y - 12, 15.5, 'white', FAINT, 3)
    elif kind == 'dash':
        s.line(x + 7, y - 12, x + 27, y - 12, FAINT, 5)
    else:
        s.image(x - 4, y - 44, 56, uris[kind])
        x += 14
    s.text(x + 50, y, label, 36)
    return x + 50 + width(label, 36) + 54


def slots(records, cond, metric):
    """One entry per (chat, twin) in a fixed order, chats in episode order with Quill first: None when the twin has no
    such city to say, else 1 or 0 for said first or not."""
    by = {(r['episode_index'], r['receiver']): r for r in records}
    chats = sorted({i for i, _ in by})
    out = []
    for i in chats:
        for side in ('quill', 'ember'):
            v = by[(i, side)]['conditions'][cond].get(metric)
            out.append(None if v is None else int(v))
    return chats, out


def counts(records):
    rows = {}
    for cond, metric in (('link', 'names'), ('cut', 'names'), ('crossed', 'names'), ('crossed', 'names_donor')):
        _, v = slots(records, cond, metric)
        rows[f'{cond}/{metric}'] = dict(said_first=sum(x for x in v if x is not None),
                                        n=sum(x is not None for x in v), slots=v)
    return rows


ROWS = (('Link on', 'link', WARM, RED), ('Link cut', 'cut', COOL, INK),
        ('Partner from another conversation', 'crossed', COOL, INK))


def draw(before, after, stem):
    W, M, gap = 2400, 36, 24
    top, head_h = 110, 190
    isize, pad = 330, 34                          # pair icon size, row padding
    lab_x, lab_w = M + 30 + 2 * isize + 40, 440   # the condition's name, right of its icons
    cols = (1430, 1960)                           # the centre of each count column
    extra = 150                                   # the last row's second count: the other conversation's city
    labels = {cond: wrap(label, lab_w, 50, 700) for label, cond, *_ in ROWS}
    assert cols[0] - width('Before chat training', 46) / 2 > lab_x + lab_w
    heights = [2 * pad + isize + (extra if cond == 'crossed' else 0) for _, cond, *_ in ROWS]
    H = top + head_h + sum(heights) + gap * (len(ROWS) - 1) + M
    assert H >= 0.65 * W
    s = SVG(W, H)
    s.text(M, 84, 'Only linked twins say the partner’s city before the partner says that city', 56, 700)
    s.text(sum(cols) / 2, top + 72, 'Partner’s city said first', 50, 700, INK, 'middle')
    for x, head in zip(cols, ('Before chat training', 'After chat training')):
        s.text(x, top + 150, head, 46, 400, INK, 'middle')

    def pair(y, cond, metric, color, cx):
        recs = before if cx == cols[0] else after
        _, v = slots(recs, cond, metric)
        hits, n = sum(k for k in v if k is not None), sum(k is not None for k in v)
        w = width(hits, 72, 700) + width(f' of {n}', 46)
        count(s, cx - w / 2, y, hits, n, color)

    y = top + head_h
    for (label, cond, fill, color), h in zip(ROWS, heights):
        s.rect(M, y, W - 2 * M, h, fill, 32)
        pair_icon(s, M + 30, y + pad, isize, cond, gap=0)
        mid = y + pad + isize / 2
        for j, line in enumerate(labels[cond]):
            s.text(lab_x, mid + 18 - 31 * (len(labels[cond]) - 1) + 62 * j, line, 50, 700, INK)
        for cx in cols:
            pair(mid + 26, cond, 'names', color, cx)
        if cond == 'crossed':
            s.text(sum(cols) / 2, mid + 160, 'Other conversation’s city said first', 46, 700, LINK, 'middle')
            for cx in cols:
                pair(mid + 260, cond, 'names_donor', LINK, cx)
        y += h + gap
    s.save(stem)


def main():
    before, after = load(BEFORE), load(AFTER)
    draw(before, after, HERE / 'chat_names')
    snap = dict(sources=dict(before=[str(p.relative_to(ROOT)) for p in BEFORE], after=[str(p.relative_to(ROOT)) for p in AFTER]),
                metric=dict(names="the twin said its partner's city before the partner did (chateval first_mention); "
                                  "n counts twins whose partner holds a city",
                            names_donor="crossed only: the twin said the city of the partner from another conversation before "
                                        "that partner did; n counts twins whose partner from another conversation "
                                        "holds a city"),
                slot_order='episode_index ascending, then quill, ember; null = no city to say',
                before=counts(before), after=counts(after))
    SNAPSHOT.write_text(json.dumps(snap, indent=1) + '\n')
    for k in ('before', 'after'):
        print(k, {c: f"{v['said_first']}/{v['n']}" for c, v in snap[k].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
