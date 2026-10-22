"""The link's architecture, drawn from pairing/link.py and pairing/session.py.

link_mechanism        one read at one exchange depth: query, keys, scores, recency bias, weights, null key, gated
                      values, weighted sum, output map, write size, added to the reader's residual stream
link_mechanism_depths both models, 36 blocks each, the 8 exchange depths, and how a write rises and is read back
link_mechanism_ticks  which partner positions each position may read (the time rule), both directions

Renders from results/link_mechanism.json (counts, token timing and private cities extracted once from the author's
saved chat and model configs) and checks the exchange blocks against pairing/link.py. Bars, gate meters and weight bars
are schematic: no activations, gates or attention weights were measured for these drawings.
`python figures/link_mechanism.py --refresh` rebuilds the snapshot from the original files (author's Mac only).
"""
import ast
import base64
import hashlib
import html
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
SNAP = ROOT / 'results/link_mechanism.json'

# Shared palette and type (keep in step with the other figures).
Q, E, G, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
QL, EL, GL, REDL = '#EFE7FC', '#FCE9DE', '#DDF2EA', '#F8DDD8'     # light tints for fills
MUTE, LINE = '#6E6E73', '#C7C7CC'
FONT = 'Inter, Helvetica, Arial, sans-serif'
W = 2300
BODY, SMALL, TITLE = 36, 32, 62


def xblocks():
    tree = ast.parse((ROOT / 'pairing/link.py').read_text())
    return next(list(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))


def refresh():
    """Rebuild the snapshot from the author's saved chat, tokenizer and configs (only on the author's Mac)."""
    configs = {}
    for side, repo in [('quill', 'Qwen--Qwen3-4B'), ('ember', 'arcee-ai--AFM-4.5B')]:
        p = sorted((Path.home() / '.cache/huggingface/hub' / ('models--' + repo) / 'snapshots').glob('*/config.json'))[0]
        configs[side] = dict(path=str(p), blocks=json.loads(p.read_text())['num_hidden_layers'])
    chat_path = Path('/Users/user/mpd-data/neural-pairing/free_chat/nl22c_s100/chat_link_s7.json')
    chat = json.loads(chat_path.read_text())
    private = {side: chat['episode']['meta']['secrets'][side]['city'] for side in configs}
    first = chat['turns'][0]
    pieces = [v['text'] for v in first['tokens'][:4]]
    from tokenizers import Tokenizer
    tok_path = Path(configs['ember']['path']).with_name('tokenizer.json')
    tok = Tokenizer.from_file(str(tok_path))
    listener, prior = [], []
    for i in range(len(pieces)):
        stable = tok.encode(''.join(pieces[:i + 1]), add_special_tokens=False).ids[:-1]
        assert stable[:len(prior)] == prior
        listener.append([tok.decode([t]) for t in stable[len(prior):]])
        prior = stable
    old = json.loads(SNAP.read_text()) if SNAP.exists() else {}
    xb = xblocks()
    old.update(exchange_blocks=xb, exchange_count=len(xb), configs=configs, private_text=private,
               private_text_source=str(chat_path), timing_tokens=pieces,
               timing_emission_ticks=[v['tick'] for v in first['tokens'][:4]],
               timing_token_ids=[v['id'] for v in first['tokens'][:4]], timing_listener_chunks=listener,
               timing_tokenizer=str(tok_path),
               source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [ROOT / 'pairing/link.py', ROOT / 'pairing/session.py', chat_path, tok_path]
                              + [Path(c['path']) for c in configs.values()]})
    SNAP.write_text(json.dumps(old, indent=2) + '\n')


def facts():
    f = json.loads(SNAP.read_text())
    assert f['exchange_blocks'] == xblocks(), 'snapshot and pairing/link.py disagree on the exchange blocks'
    src = (ROOT / 'pairing/link.py').read_text()
    assert 'heads=8, head_dim=128' in src and 'AGES = 8' in src
    return f


# ---------------------------------------------------------------- drawing helpers (also used by hero_link.py)

_FONTS = {}


def tw(s, size, weight=400):
    """Width of s in px when set in Inter (estimate if Pillow or the font is missing)."""
    name = {400: 'Regular', 600: 'SemiBold', 700: 'Bold'}.get(weight, 'Regular')
    try:
        from PIL import ImageFont
        key = (name, size)
        if key not in _FONTS:
            _FONTS[key] = ImageFont.truetype(f'/usr/share/fonts/opentype/inter/Inter-{name}.otf', size)
        return _FONTS[key].getlength(s)
    except Exception:
        return 0.56 * size * len(s)


def esc(s):
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                  f'<title>{esc(title)}</title>', f'<desc>{esc(desc)}</desc>',
                  f'<rect width="{w}" height="{h}" fill="white"/>']

    def add(self, s):
        self.a.append(s)

    def text(self, x, y, s, size=BODY, color=INK, weight=400, anchor='start', italic=False, opacity=None):
        """s: a string, or a list of (text, color, weight) runs."""
        runs = [(s, color, weight)] if isinstance(s, str) else s
        body = ''.join(f'<tspan fill="{c or color}" font-weight="{wt or weight}">{esc(t)}</tspan>' for t, c, wt in runs)
        extra = (' font-style="italic"' if italic else '') + (f' opacity="{opacity}"' if opacity else '')
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                 f'font-weight="{weight}" text-anchor="{anchor}" xml:space="preserve"{extra}>{body}</text>')

    def title(self, s, x=60, y=88, size=TITLE):
        self.text(x, y, s, size, INK, 700)

    def rect(self, x, y, w, h, fill='white', stroke='none', r=16, sw=3, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        o = f' opacity="{opacity}"' if opacity is not None else ''
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def path(self, d, color=INK, sw=4, fill='none', dash=None, opacity=None, cap='round'):
        extra = (f' stroke-dasharray="{dash}"' if dash else '') + (f' opacity="{opacity}"' if opacity is not None else '')
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}" stroke-linecap="{cap}" '
                 f'stroke-linejoin="round"{extra}/>')

    def line(self, pts, color=INK, sw=4, dash=None, opacity=None):
        self.path('M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts), color, sw, dash=dash, opacity=opacity)

    def arrow(self, pts, color=INK, sw=4, head=None, dash=None, opacity=None):
        """Polyline with a filled head at the last point (the line stops at the head's base)."""
        head = head or 7 + 2.6 * sw
        (a, b), (x, y) = pts[-2:]
        t = math.atan2(y - b, x - a)
        bx, by = x - head * 0.8 * math.cos(t), y - head * 0.8 * math.sin(t)
        self.line(pts[:-1] + [(bx, by)], color, sw, dash, opacity)
        hw = head * 0.55
        p = [(x, y), (x - head * math.cos(t) - hw * math.sin(t), y - head * math.sin(t) + hw * math.cos(t)),
             (x - head * math.cos(t) + hw * math.sin(t), y - head * math.sin(t) - hw * math.cos(t))]
        o = f' opacity="{opacity}"' if opacity is not None else ''
        self.add(f'<polygon points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in p)}" fill="{color}"{o}/>')

    def circle(self, x, y, r, fill='white', stroke=G, sw=4):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def plus(self, x, y, r=20, color=G, sw=4):
        self.circle(x, y, r, 'white', color, sw)
        k = r * 0.55
        self.path(f'M{x - k:.1f},{y:.1f} L{x + k:.1f},{y:.1f} M{x:.1f},{y - k:.1f} L{x:.1f},{y + k:.1f}', color, sw)

    def times(self, x, y, r=18, color=INK, sw=4):
        self.circle(x, y, r, 'white', color, sw)
        k = r * 0.42
        self.path(f'M{x - k:.1f},{y - k:.1f} L{x + k:.1f},{y + k:.1f} M{x - k:.1f},{y + k:.1f} L{x + k:.1f},{y - k:.1f}', color, sw)

    def robot(self, x, y, side, size=130):
        data = base64.b64encode((OUT / 'assets' / f'{side}.png').read_bytes()).decode()
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>')

    def vector(self, x, y, color, w=64, h=76, seed=0, scale=1.0, fill='white', stroke=None, sw=3, opacity=1.0):
        """A hidden-state / key / value icon: a box of 4 bars (schematic heights)."""
        self.rect(x, y, w, h, fill, stroke or color, 10, sw)
        for j in range(4):
            bar = (h - 22) * (.3 + ((seed * 3 + j * 5) % 7) / 10) * scale
            self.rect(x + 9 + j * (w - 18) / 4, y + h - 11 - bar, (w - 18) / 4 - 5, bar, color, r=3, sw=0,
                      opacity=opacity)

    def meter(self, x, y, frac, color=E, w=26, h=76):
        """A gate: one number between 0 and 1, drawn as a filled bar."""
        self.rect(x, y, w, h, 'white', color, 8, 3)
        self.rect(x + 4, y + h - 4 - (h - 8) * frac, w - 8, (h - 8) * frac, color, r=5, sw=0)

    def save(self, name, width=None):
        p = OUT / (name + '.svg')
        p.write_text('\n'.join(self.a + ['</svg>']) + '\n')
        subprocess.run(['rsvg-convert', '-w', str(width or self.w), '-o', str(p.with_suffix('.png')), str(p)], check=True)
        subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(p.with_suffix('.pdf')), str(p)], check=True)


def chip(s, x, y, w, h, label, color, fill='white', size=BODY, weight=600, sw=3, tcolor=None):
    s.rect(x, y, w, h, fill, color, 14, sw)
    s.text(x + w / 2, y + h / 2 + size * 0.36, label, size, tcolor or color, weight, 'middle')


# ---------------------------------------------------------------- timing (session.py dialogue rule)

def timeline(f):
    """Positions of each model in the opening of the linked chat, with the time at which each is read.
    Quill writes turn 0: prompt at -1, its turn header at 0, and each generated piece is read one time after it is
    emitted. Ember listens: prompt at -1, the user-message header at 0, then at each time the pieces of Quill's text
    that can no longer change (all but the last)."""
    q = [('prompt', -1), ('turn header', 0)]
    q += [(t.strip(), e + 1) for t, e in zip(f['timing_tokens'], f['timing_emission_ticks'])]
    e = [('prompt', -1), ('turn header', 0)]
    for i, chunk in enumerate(f['timing_listener_chunks']):
        e += [(c.strip(), i + 1) for c in chunk]
    return {'quill': q, 'ember': e}


def bucket(tq, tk):
    """link.age_buckets for one pair of times: 'prompt' for a partner prompt position, else log-spaced."""
    if tk < 0:
        return 'prompt'
    d = max(tq - tk, 0)
    if d == 0:
        return '0'
    b = int(math.floor(math.log2(d))) + 1
    lo, hi = 2 ** (b - 1), 2 ** b - 1
    return str(lo) if lo == hi else f'{lo}–{hi}'


# ---------------------------------------------------------------- figure: which partner positions are visible

def ticks(f):
    tl = timeline(f)
    H = 1090
    s = SVG(W, H, 'Readable partner positions: same time or earlier',
            'For the opening of the linked chat (Quill writes "So, you\'re", Ember reads it as it streams), the grid '
            'of which partner positions each position may read at every exchange depth, with the recency bucket of '
            'each readable pair. Derived from pairing/session.py and pairing/link.py; token timing from '
            'results/link_mechanism.json.')
    s.title('Readable partner positions: same time or earlier')
    cell, ch, gap, lab = 132, 100, 8, 238
    top = 130
    ph = H - top - 30

    def tokchip(x, y, w, tok, color, fill):
        if tok in ('prompt', 'turn header'):
            s.text(x + w / 2, y + 44, 'prompt' if tok == 'prompt' else 'header', BODY - 2, color, 600, 'middle')
        else:
            s.rect(x + 8, y, w - 16, 64, fill, color, 12, 3)
            s.text(x + w / 2, y + 45, tok, BODY + 2, INK, 600, 'middle')

    def grid(px, pw, reader, sender):
        rows, cols = tl[reader], tl[sender]
        rc, sc = (Q, E) if reader == 'quill' else (E, Q)
        rl, sl = (QL, EL) if reader == 'quill' else (EL, QL)
        s.rect(px, top, pw, ph, COOL, r=36)
        s.robot(px + 24, top + 18, reader, 108)
        s.text(px + 148, top + 88, [(reader.title(), rc, 700), (' reads ', INK, 400), (sender.title(), sc, 700)], 50)
        gx = px + 30 + lab
        gy = top + 268
        s.text(px + 30, gy - 92, 'time', SMALL, MUTE)
        for j, (tok, t) in enumerate(cols):
            x = gx + j * (cell + gap)
            s.text(x + cell / 2, gy - 92, str(t).replace('-', '−'), SMALL, MUTE, 400, 'middle')
            tokchip(x, gy - 78, cell, tok, sc, sl)
        for i, (tok, tq) in enumerate(rows):
            y = gy + i * (ch + gap)
            if tok in ('prompt', 'turn header'):
                s.text(px + 30, y + 62, 'prompt' if tok == 'prompt' else 'header', BODY - 2, rc, 600)
            else:
                s.rect(px + 30, y + 18, 92, 64, rl, rc, 12, 3)
                s.text(px + 76, y + 63, tok, BODY + 2, INK, 600, 'middle')
            s.text(px + 176, y + 62, str(tq).replace('-', '−'), SMALL, MUTE, 400, 'middle')
            for j, (_, tk) in enumerate(cols):
                x = gx + j * (cell + gap)
                if tk <= tq:
                    s.rect(x, y, cell, ch, GL, G, 14, 3)
                    b = bucket(tq, tk)
                    s.text(x + cell / 2, y + ch / 2 + 14, b, BODY + 4 if b != 'prompt' else SMALL, G, 700, 'middle')
                else:
                    s.rect(x, y, cell, ch, '#E3E3E8', 'none', 14, 0)
        return gx, gy, rows, cols

    lw = 2 * 30 + lab + 5 * (cell + gap) + 10
    gx, gy, rows, cols = grid(60, lw, 'quill', 'ember')
    i = next(k for k, (t, _) in enumerate(rows) if t == 'you')
    s.rect(60 + 14, gy + i * (ch + gap) - 9, lw - 28, ch + 18, 'none', RED, 20, 5)
    rx = 60 + lw + 30
    gx2, gy2, rows2, _ = grid(rx, W - 60 - rx, 'ember', 'quill')
    ky = gy2 + len(rows2) * (ch + gap) + 40
    items = [('1', GL, G, 3, 'time gap'), ('', '#E3E3E8', 'none', 0, 'hidden'), ('', 'none', RED, 5, 'previous figure')]
    kx = rx + 30
    for lab_, fill, stroke, sw, words in items:
        s.rect(kx, ky, 80, 62, fill, stroke, 12, sw)
        if lab_:
            s.text(kx + 40, ky + 44, lab_, BODY, G, 700, 'middle')
        s.text(kx + 100, ky + 44, words, BODY - 2)
        kx += 100 + tw(words, BODY - 2) + 70
    s.save('link_mechanism_ticks')


# ---------------------------------------------------------------- figure: the 8 depths in one pass

def depths(f):
    xb = f['exchange_blocks']
    nb = f['configs']['quill']['blocks']
    assert nb == f['configs']['ember']['blocks']
    H = 1270
    s = SVG(W, H, f'Both models read each other at {len(xb)} depths',
            'Two 36-block stacks run block by block together. After blocks ' + ', '.join(map(str, xb)) +
            ' each model reads the other, both ways at once, from the states before that exchange, and adds the '
            'result to its own residual stream. Schematic; counts from pairing/link.py and the model configs.')
    s.title(f'Both models read each other at {len(xb)} depths')
    yq, ye = 318, 598
    x0, pitch, bw, bh, extra, half = 350, 36, 27, 124, 72, 24

    def bx(i):
        return x0 + i * pitch + extra * sum(1 for l in xb if l < i)

    def xc(l):
        return (bx(l) + bw + bx(l + 1)) / 2

    s.rect(40, 130, W - 80, 650, COOL, r=36)
    end = bx(nb - 1) + bw + 40
    for side, y, col, light in (('quill', yq, Q, QL), ('ember', ye, E, EL)):
        s.robot(52, y - 134, side, 132)
        s.text(118, y + 34, side.title(), BODY + 2, col, 700, 'middle')
        chip(s, 190, y - 34, 140, 62, f['private_text'][side], col, light, SMALL, 700, tcolor=INK)
        s.arrow([(332, y - 3), (x0 - 2, y - 3)], col, 4, head=14)
    pts = [(x0 - 2, ye)]
    cur = 'ember'
    for l in xb:
        ya, yb = (ye, yq) if cur == 'ember' else (yq, ye)
        pts += [(xc(l) - half, ya), (xc(l) + half, yb)]
        cur = 'quill' if cur == 'ember' else 'ember'
    pts.append((end, ye if cur == 'ember' else yq))
    s.line(pts, RED, 24, opacity=0.34)
    for side, y, col in (('quill', yq, Q), ('ember', ye, E)):
        for i in range(nb):
            s.rect(bx(i), y - bh / 2, bw, bh, 'white', '#8E8E93', 7, 2.5)
        s.arrow([(x0 - 2, y), (end, y)], col, 6)
    for i in [0] + xb + [nb - 1]:
        s.text(bx(i) + bw / 2 if i in (0, nb - 1) else xc(i), yq - bh / 2 - 18, str(i), BODY - 2,
               INK if i in (0, nb - 1) else G, 700 if i in xb else 400, 'middle')
    s.text(bx(0) - 6, yq - bh / 2 - 62, 'block', SMALL, MUTE)
    for l in xb:
        a, b = xc(l) - half, xc(l) + half
        for (ys, yt) in ((yq, ye), (ye, yq)):
            sg = 1 if ys < yt else -1
            s.arrow([(a, ys + 9 * sg), (b, yt - 14 * sg)], G, 5, head=20)
            s.circle(a, ys, 7.5, G, G, 0)
            s.plus(b, yt, 12, G, 3.5)
    z = xb[1]
    s.rect(xc(z) - 46, yq - 74, 92, ye - yq + 148, 'none', INK, 18, 3, dash='10 9')
    ky = 730
    kx = (W - (44 + tw('block', SMALL) + 90 + 128 + tw('read, added', SMALL) + 90 + 84 + tw('one path', SMALL))) / 2
    s.rect(kx, ky - 30, 24, 60, 'white', '#8E8E93', 6, 2.5)
    s.text(kx + 44, ky + 12, 'block', SMALL)
    kx += 44 + tw('block', SMALL) + 90
    s.circle(kx + 8, ky, 7.5, G, G, 0)
    s.arrow([(kx + 18, ky), (kx + 78, ky)], G, 5, head=20)
    s.plus(kx + 98, ky, 12, G, 3.5)
    s.text(kx + 128, ky + 12, 'read, added', SMALL)
    kx += 128 + tw('read, added', SMALL) + 90
    s.line([(kx, ky), (kx + 64, ky)], RED, 24, opacity=0.34)
    s.text(kx + 84, ky + 12, 'one path', SMALL)
    # zoom: one exchange (block z)
    zy = 812
    zh = H - zy - 30
    pw = (W - 120 - 2 * 30) / 3
    heads = [f'After block {z}', 'Both read at once', 'Each adds its write']
    for k in range(3):
        px = 60 + k * (pw + 30)
        s.rect(px, zy, pw, zh, 'white', LINE, 28, 3)
        s.text(px + 34, zy + 62, heads[k], BODY + 2, INK, 700)
    p1, p2, p3 = (60 + k * (pw + 30) for k in range(3))
    vw, vh, sp_ = 78, 88, 96
    ra, rb = zy + 96, zy + 96 + vh + 112
    for j in range(5):
        s.vector(p1 + 34 + j * sp_, ra, Q, vw, vh, 1 + j)
        s.vector(p1 + 34 + j * sp_, rb, E, vw, vh, 4 + j)
    s.text(p1 + 34 + 5 * sp_ + 10, ra + 56, 'Quill', BODY, Q, 700)
    s.text(p1 + 34 + 5 * sp_ + 10, rb + 56, 'Ember', BODY, E, 700)
    qx = [p2 + 34 + j * sp_ for j in range(5)]
    qr, er = 4, 2
    for j in range(5):
        s.vector(qx[j], ra, Q, vw, vh, 1 + j, 1, 'white', None if j == qr else '#CDB9F3')
        s.vector(qx[j], rb, E, vw, vh, 4 + j, 1, 'white', None if j == er else '#F5C6AA')
    for j in range(qr + 1):
        s.arrow([(qx[j] + 39, rb - 4), (qx[qr] + 22 + 8 * j, ra + vh + 6)], G, 3.5, head=15)
    for j in range(er + 1):
        s.arrow([(qx[j] + 39, ra + vh + 4), (qx[er] + 24 + 15 * j, rb - 6)], G, 3.5, head=15)
    s.text(qx[4] + vw + 22, ra + 40, 'Quill', SMALL, G, 700)
    s.text(qx[4] + vw + 22, ra + 76, 'reads', SMALL, G, 700)
    s.text(qx[4] + vw + 22, rb + 40, 'Ember', SMALL, G, 700)
    s.text(qx[4] + vw + 22, rb + 76, 'reads', SMALL, G, 700)
    for y, col, seed in ((ra, Q, 1), (rb, E, 4)):
        s.vector(p3 + 30, y, col, vw, vh, seed)
        s.plus(p3 + 142, y + vh / 2, 18, G, 3.5)
        s.vector(p3 + 176, y + 10, G, 66, vh - 20, seed + 2, 0.7)
        s.arrow([(p3 + 252, y + vh / 2), (p3 + 300, y + vh / 2)], col, 4)
        s.vector(p3 + 308, y, col, vw, vh, seed + 1)
        s.arrow([(p3 + 396, y + vh / 2), (p3 + 436, y + vh / 2)], col, 4)
        s.rect(p3 + 444, y + 4, 190, vh - 8, 'white', '#8E8E93', 12, 2.5)
        s.text(p3 + 539, y + vh / 2 + 12, f'block {z + 1}', SMALL, INK, 600, 'middle')
    s.text(p3 + 209, ra + vh + 44, 'write', SMALL, G, 700, 'middle')
    s.save('link_mechanism_depths')


# ---------------------------------------------------------------- figure: one read in full

def mechanism(f):
    tl = timeline(f)
    xb = f['exchange_blocks']
    z = xb[1]
    reader_tok, reader_t = next((t, k) for t, k in tl['quill'] if t == 'you')
    H = 1300
    s = SVG(W, H, f'Quill reads Ember after block {z}',
            'One read through the link at one exchange depth, as in pairing/link.py Side.show and Side.read: Quill\'s '
            f'position "{reader_tok}" (time {reader_t}) attends over every Ember position read at time {reader_t} or '
            'earlier plus its null key. Purple parts belong to the reader (Quill), orange parts to the sender (Ember). '
            'Bars, gate meters and weights are drawings, not measured values.')
    s.title(f'Quill reads Ember after block {z}')
    top, rh = 130, 118
    s.rect(40, top, W - 80, H - top - 30, COOL, r=36)
    C = dict(state=450, key=630, score=830, bias=1025, weight=1215, gate=1395, value=1540, wv=1700)
    heads = [('state', 'state', E), ('key', 'key', E), ('score', 'score', G), ('bias', 'bias', Q),
             ('weight', 'weight', G), ('gate', 'gate', E), ('value', 'value', E), ('wv', 'weighted', G)]
    hy = top + 66
    s.text(70, hy, 'position', BODY, E, 700)
    for k, h1, col in heads:
        s.text(C[k], hy, h1, BODY, col, 700, 'middle')
    kx, ky = 1880, top + 48
    for j, (col, fill, lab) in enumerate([(Q, QL, 'Quill’s parts'), (E, EL, 'Ember’s parts'), (G, GL, 'computed')]):
        s.rect(kx, ky + j * 50 - 24, 32, 32, fill, col, 8, 3)
        s.text(kx + 46, ky + j * 50 + 3, lab, SMALL, col, 700)
    rows = [('null key', None)] + [(t, k) for t, k in tl['ember']]
    weights = [0.16, 0.34, 0.06, 0.14, 0.30, 0.0]
    scores = [0.45, 0.7, 0.25, 0.5, 0.65, 0.6]
    gates = [None, 0.88, 0.55, 0.8, 0.9, 0.85]
    vw, vh = 76, 76
    y0 = top + 100
    faded = '#F5C6AA'
    for i, (tok, t) in enumerate(rows):
        y = y0 + i * rh
        cy = y + rh / 2 - 6
        hidden = t is not None and t > reader_t
        if i % 2 == 0:
            s.rect(56, y - 2, 1760, rh - 8, '#E9E9EE', 'none', 18, 0)
        if tok == 'null key':
            chip(s, 70, cy - 32, 190, 64, 'null key', Q, QL, BODY - 2, 700, tcolor=INK)
        elif tok in ('prompt', 'turn header'):
            s.text(70, cy + 2, 'prompt' if tok == 'prompt' else 'header', BODY, E, 700)
            s.text(70, cy + 40, f'time {t}'.replace('-', '−'), SMALL, MUTE)
        else:
            chip(s, 70, cy - 32, 100, 64, tok, E, EL, BODY + 2, 600, tcolor=INK)
            s.text(186, cy + 11, f'time {t}', SMALL, MUTE)
        vy = cy - vh / 2
        if tok != 'null key':
            if tok == 'prompt':
                for d in (12, 6):
                    s.rect(C['state'] - vw / 2 + d, vy - d, vw, vh, 'white', E, 10, 2.5)
            s.vector(C['state'] - vw / 2, vy, E, vw, vh, i, opacity=0.35 if hidden else 1, stroke=faded if hidden else E)
            s.vector(C['key'] - vw / 2, vy, E, vw, vh, i + 3, 0.9, opacity=0.35 if hidden else 1,
                     stroke=faded if hidden else E)
            s.arrow([(C['state'] + vw / 2 + 4, cy), (C['key'] - vw / 2 - 4, cy)], faded if hidden else E, 3.5, head=15)
        else:
            s.vector(C['key'] - vw / 2, vy, Q, vw, vh, 2, 0.9)
        if hidden:
            s.rect(C['score'] - 86, cy - 32, C['wv'] + 56 - C['score'] + 86, 64, 'white', LINE, 14, 2.5)
            s.text((C['score'] - 86 + C['wv'] + 56) / 2, cy + 12, 'hidden', BODY - 2, MUTE, 400, 'middle')
            continue
        s.rect(C['score'] - 76, cy - 15, 152, 30, 'white', LINE, 9, 2)
        s.rect(C['score'] - 76, cy - 15, 152 * scores[i], 30, G, 'none', 9, 0, opacity=0.55)
        b = '0' if tok == 'null key' else bucket(reader_t, t)
        if tok == 'null key':
            s.text(C['bias'], cy + 12, '–', BODY, MUTE, 400, 'middle')
        else:
            chip(s, C['bias'] - 72, cy - 30, 144, 60, b, Q, QL, BODY - 2, 700, tcolor=Q)
        s.rect(C['weight'] - 88, cy - 16, 176, 32, 'white', LINE, 9, 2)
        s.rect(C['weight'] - 88, cy - 16, 176 * weights[i] / max(weights), 32, G, 'none', 9, 0)
        if tok == 'null key':
            s.text(C['gate'], cy + 12, '–', BODY, MUTE, 400, 'middle')
            s.rect(C['value'] - vw / 2, vy, vw, vh, 'white', Q, 10, 2.5, dash='7 6')
            s.text(C['value'], cy + 13, '0', BODY, Q, 700, 'middle')
            s.text(C['wv'], cy + 13, '0', BODY, G, 700, 'middle')
        else:
            s.meter(C['gate'] - 15, vy - 2, gates[i], E, 30, vh + 4)
            s.vector(C['value'] - vw / 2, vy, E, vw, vh, i + 5, gates[i])
            sc = 0.45 + 0.55 * weights[i] / max(weights)
            s.vector(C['wv'] - vw / 2 * sc, cy - vh / 2 * sc, G, vw * sc, vh * sc, i + 5, gates[i])
    by = y0 + len(rows) * rh
    qy, sy = by + 62, H - 132
    s.vector(70, sy - 40, Q, 80, 80, 1)
    s.text(70, sy + 80, f'Quill: “{reader_tok}”, time {reader_t}', BODY - 2, Q, 700)
    s.arrow([(154, sy), (C['wv'] + 140, sy)], Q, 9)
    s.circle(250, sy, 8, Q, Q, 0)
    s.arrow([(250, sy - 8), (250, qy + 38)], Q, 4, head=17)
    chip(s, 110, qy - 36, 280, 72, 'query map', Q, 'white', BODY - 2, 600, tcolor=INK)
    s.arrow([(392, qy), (452, qy)], Q, 4, head=17)
    s.vector(458, qy - 38, Q, 78, 76, 2)
    s.text(497, qy + 82, 'query', BODY - 2, Q, 700, 'middle')
    bus = C['score'] - 112
    s.line([(538, qy), (bus, qy), (bus, y0 + rh / 2 - 6)], Q, 4)
    for i, (tok, t) in enumerate(rows):
        if t is not None and t > reader_t:
            continue
        cy = y0 + i * rh + rh / 2 - 6
        s.arrow([(bus, cy), (C['score'] - 80, cy)], Q, 4, head=15)
    ox = C['wv'] + 170
    bx_ = C['wv'] + 72
    s.line([(C['wv'] + 46, y0 + rh / 2 - 6), (bx_, y0 + rh / 2 - 6), (bx_, y0 + 4.5 * rh - 6),
            (C['wv'] + 46, y0 + 4.5 * rh - 6)], G, 3.5)
    sy_ = y0 + 2.5 * rh - 6
    s.arrow([(bx_, sy_), (ox - 44, sy_)], G, 4, head=17)
    s.vector(ox - 40, sy_ - 40, G, 80, 80, 7)
    s.text(ox + 58, sy_ + 12, 'sum', BODY, G, 700)
    s.arrow([(ox, sy_ + 42), (ox, qy - 38)], G, 4, head=17)
    chip(s, ox - 140, qy - 36, 280, 72, 'output map', Q, 'white', BODY - 2, 600, tcolor=INK)
    s.arrow([(ox, qy + 38), (ox, sy - 108)], G, 4, head=17)
    s.times(ox, sy - 82, 24, Q, 4)
    s.text(ox + 40, sy - 70, 'write size', BODY - 2, Q, 700)
    s.arrow([(ox, sy - 57), (ox, sy - 28)], G, 4, head=15)
    s.plus(ox, sy, 26, Q, 5)
    s.arrow([(ox + 27, sy), (ox + 100, sy)], Q, 9)
    s.rect(ox + 106, sy - 40, 196, 80, 'white', '#8E8E93', 14, 3)
    s.text(ox + 204, sy + 12, f'block {z + 1}', BODY - 2, INK, 600, 'middle')
    s.save('link_mechanism')

if __name__ == '__main__':
    if '--refresh' in sys.argv:
        refresh()
    f = facts()
    mechanism(f)
    depths(f)
    ticks(f)
