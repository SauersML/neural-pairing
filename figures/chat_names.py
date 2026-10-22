"""The 20-chat test before and after chat training: who says whose city first.

One mark per twin per chat (20 chats x 2 twins, in the same place in every row). Rows: link on, link cut, and linked
to a stranger (chateval's crossed condition: an isolated twin of the partner's model holding a different city, linked
in place of the partner). For the first three rows the mark is filled red when the twin said its partner's city before
the partner did (chateval `names`); in the last row it is filled teal when the twin said the stranger's city before
the stranger did (chateval `names_donor`). A faint dash marks a twin with no such city to say (the partner, or the
stranger, holds no city), which is why the denominators are 23 and 36.

Reads the evalfree --shard outputs in results/magic-nl20/results/chat_before_* and
results/magic-nl22c/results/chat_after_* (plain JSON lines; no model, no torch) and writes
figures/chat_names.{svg,png,pdf} and results/chat_names.json (the counts drawn).
"""
import base64
import html
import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEFORE = [ROOT / f'results/magic-nl20/results/chat_before_{k}' for k in range(2)]
AFTER = [ROOT / f'results/magic-nl22c/results/chat_after_{k}' for k in range(3)]
SNAPSHOT = ROOT / 'results/chat_names.json'

QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
MUTED, FAINT, COOL, WARM = '#6E6E73', '#B9B9C0', '#F4F4F6', '#FAF0E6'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
_FONTS = {}


def font(size, weight=400):
    name = {400: 'Inter-Regular', 600: 'Inter-SemiBold', 700: 'Inter-Bold'}[weight]
    key = (size, weight)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(f'/usr/share/fonts/opentype/inter/{name}.otf', size)
    return _FONTS[key]


def width(s, size, weight=400):
    return font(size, weight).getlength(s)


def load(dirs):
    """The records of evalfree --shard outputs (as pairing/chat_merge.load, without importing chateval/torch)."""
    records = [json.loads(line) for d in dirs for line in open(Path(d) / 'episodes.jsonl') if line.strip()]
    keys = [(r['episode_id'], r['receiver']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('a chat appears in two shards')
    return records


class SVG:
    def __init__(self, w, h, bg='white'):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<rect width="{w}" height="{h}" fill="{bg}"/>']

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill='white', radius=18, stroke='none', sw=2, extra=''):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def circle(self, cx, cy, r, fill='none', stroke='none', sw=3):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=3, cap='round'):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linecap="{cap}"/>')

    def path(self, d, color=INK, sw=4, fill='none', extra=''):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                 f'stroke-linecap="round" {extra}/>')

    def text(self, x, y, s, size=34, weight=400, fill=INK, anchor='start'):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start'):
        """chunks: (text, colour, weight) runs on one line."""
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def image(self, x, y, size, uri):
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" xlink:href="{uri}"/>')

    def save(self, stem):
        stem = Path(stem)
        stem.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']))
        for fmt in ('png', 'pdf'):
            subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(stem.with_suffix('.' + fmt)),
                            str(stem.with_suffix('.svg'))], check=True)


def mascot(name, grey=False):
    im = Image.open(HERE / 'assets' / f'{name}.png').convert('RGBA')
    if grey:
        alpha = im.getchannel('A')
        g = im.convert('L').point(lambda v: int(90 + v * 0.62))
        im = Image.merge('RGBA', (g, g, g, alpha))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def link_arrow(s, x1, x2, y, on=True):
    """A two-way link between two mascots: teal when on, grey and broken when cut."""
    color = LINK if on else FAINT
    head = 16
    for xa, d in ((x1, 1), (x2, -1)):
        s.path(f'M{xa + d * head:.1f},{y - 12} L{xa},{y} L{xa + d * head:.1f},{y + 12}', color, 6)
    if on:
        s.line(x1 + 3, y, x2 - 3, y, color, 6)
    else:
        mid = (x1 + x2) / 2
        s.line(x1 + 3, y, mid - 16, y, color, 6)
        s.line(mid + 16, y, x2 - 3, y, color, 6)
        s.line(mid - 9, y - 11, mid + 9, y + 11, MUTED, 5)
        s.line(mid - 9, y + 11, mid + 9, y - 11, MUTED, 5)


def pair_icon(s, x, y, size, cond, uris):
    """Quill and its partner (Ember, or a grey stranger), joined by the link; (x, y) is the top left."""
    gap = 86
    s.image(x, y, size, uris['quill'])
    s.image(x + size + gap, y, size, uris['stranger' if cond == 'crossed' else 'ember'])
    link_arrow(s, x + size - 2, x + size + gap + 2, y + size * 0.62, cond != 'cut')
    return x + 2 * size + gap


def slots(records, cond, metric):
    """One entry per (chat, twin) in a fixed order, chats in episode order with Quill first: None when the twin has no
    such city to say, else 1/0 for said first / not."""
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


ROWS = (('link', 'names', 'Link on', None, RED),
        ('cut', 'names', 'Link cut', None, RED),
        ('crossed', 'names', 'Linked to stranger', "partner's city", RED),
        ('crossed', 'names_donor', 'Linked to stranger', "stranger's city", LINK))


def key_item(s, x, y, kind, label, uris=None):
    """One key entry (mark + label) with its baseline at y; returns the x after it."""
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


def draw(before, after, stem):
    W, M = 2400, 36
    top, head_h, row_h, key_h = 128, 96, 214, 96
    H = top + head_h + 4 * row_h + 14 + 24 + key_h + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'), stranger=mascot('ember', grey=True))
    s.text(M, 90, "Twins name each other's city through the link", 68, 700)

    label_w, gap = 520, 22
    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    panel_h = head_h + 4 * row_h + 14
    for k, (title, fill) in enumerate((('Before chat training', COOL), ('After chat training', WARM))):
        s.rect(px[k], top, pw, panel_h, fill, 32)
        s.text(px[k] + 40, top + 70, title, 48, 700)

    pitch, r = 40, 16
    for j, (cond, metric, label, sub, color) in enumerate(ROWS):
        y0 = top + head_h + j * row_h
        if j:
            for k in range(2):
                s.line(px[k] + 30, y0, px[k] + pw - 30, y0, '#E3DED6' if k else '#E2E2E7', 2)
            if j != 3:
                s.line(M, y0, M + label_w - 10, y0, '#E2E2E7', 2)
        msize = 124
        if j < 3:
            iy = y0 + (row_h - msize) / 2 - (34 if j < 2 else -row_h / 2 + 64)
            pair_icon(s, M + 14, iy, msize, cond, uris)
            ly = iy + msize + 56
            s.text(M + 14 + (2 * msize + 86) / 2, ly, label, 42, 700, INK, 'middle')
        for k, recs in enumerate((before, after)):
            chats, v = slots(recs, cond, metric)
            hits, n = sum(x for x in v if x is not None), sum(x is not None for x in v)
            cx0 = px[k] + (pw - 19 * pitch - 2 * r) / 2
            strong = color if hits and (cond == 'link' or color == LINK) else INK
            s.rich(cx0, y0 + 72, [(f'{hits}', strong, 700), (f' of {n}', INK, 400)] +
                   ([(f'   {sub}', color, 600)] if sub else []), 56)
            gx = cx0 + r
            for i in range(len(chats)):
                for t in range(2):
                    val = v[2 * i + t]
                    cx, cy = gx + i * pitch, y0 + 120 + t * pitch
                    if val is None:
                        s.line(cx - 9, cy, cx + 9, cy, FAINT, 5)
                    elif val:
                        s.circle(cx, cy, r, color)
                    else:
                        s.circle(cx, cy, r - 1.5, 'white', FAINT, 3)

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    x, y = M + 44, ky + key_h / 2 + 13
    for kind, label in ((RED, "partner's city first"), (LINK, "stranger's city first"), ('ring', 'not said'),
                        ('dash', 'no city'), ('quill', 'top row'), ('ember', 'bottom row')):
        x = key_item(s, x, y, kind, label, uris)
    s.save(stem)


def main():
    try:
        before, after = load(BEFORE), load(AFTER)
    except FileNotFoundError as e:
        print('missing', e.filename)
        return 1
    stem = HERE / 'chat_names'
    draw(before, after, stem)
    snap = dict(sources=dict(before=[str(p.relative_to(ROOT)) for p in BEFORE], after=[str(p.relative_to(ROOT)) for p in AFTER]),
                metric=dict(names="the twin said its partner's city before the partner did (chateval first_mention); "
                                  "n counts twins whose partner holds a city",
                            names_donor="crossed only: the twin said the linked stranger's city before the stranger did; "
                                        "n counts twins whose stranger holds a city"),
                slot_order='episode_index ascending, then quill, ember; null = no city to say',
                before=counts(before), after=counts(after))
    SNAPSHOT.write_text(json.dumps(snap, indent=1) + '\n')
    for k in ('before', 'after'):
        print(k, {c: f"{v['said_first']}/{v['n']}" for c, v in snap[k].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
