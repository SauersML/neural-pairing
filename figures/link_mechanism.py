"""Three drawings of the link's architecture, from pairing/link.py and pairing/session.py.

link_mechanism        one read after one exchange layer: query, keys, scores, time-gap bias, weights, null key, gated
                      values, weighted sum, output map, write size, added to the reader's residual stream
link_mechanism_depths both models as the film draws them, 36 layers each, the link as two strands at each of the 8
                      exchange layers, and a zoom on one exchange
link_mechanism_ticks  which partner tokens each token may read (the time rule), both directions

Reads results/link_mechanism.json (layer counts from the model configs, and the token timing and private cities of
the linked chat figures/inputs/chat_link_s7.json.gz) and checks its exchange layers against pairing/link.py (XBLOCKS). Writes
figures/NAME.{svg,png,pdf} for each NAME above. Bars, gate meters and weight bars are schematic: no activations,
gates or attention weights were measured for these drawings. --refresh rebuilds the snapshot from the chat and the
model configs and tokenizer in the local Hugging Face cache.
Usage: python figures/link_mechanism.py [--refresh]
"""
import ast
import gzip
import hashlib
import html
import json
import math
import subprocess
import sys
from pathlib import Path

import creature
from chat_names import ROOT, mascot, width as tw

OUT = ROOT / 'figures'
SNAP = ROOT / 'results/link_mechanism.json'
CHAT = ROOT / 'figures/inputs/chat_link_s7.json.gz'
REPOS = {'quill': 'Qwen/Qwen3-4B', 'ember': 'arcee-ai/AFM-4.5B'}

Q, E, G, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
QL, EL, GL, REDL = '#EFE7FC', '#FCE9DE', '#DDF2EA', '#F8DDD8'     # light tints for fills
LINE = '#C7C7CC'
FONT = 'Inter, Helvetica, Arial, sans-serif'
W = 2300
BODY, SMALL, TITLE = 36, 32, 56


def xblocks():
    tree = ast.parse((ROOT / 'pairing/link.py').read_text())
    return next(list(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))


def refresh():
    """The snapshot rebuilt from the linked chat and the cached model configs and Ember tokenizer: layer counts, both
    private cities, and the timing of the first four tokens of turn 0 as Quill writes them and as Ember reads them."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    files, configs = {}, {}
    for side, repo in REPOS.items():
        p = Path(hf_hub_download(repo, 'config.json', local_files_only=True))
        name = f'{repo}@{p.parent.name}/config.json'
        files[name] = p
        configs[side] = dict(path=name, blocks=json.loads(p.read_text())['num_hidden_layers'])
    raw = gzip.open(CHAT).read()
    chat = json.loads(raw)
    private = {side: chat['episode']['meta']['secrets'][side]['city'] for side in configs}
    first = chat['turns'][0]
    pieces = [v['text'] for v in first['tokens'][:4]]
    tok_path = Path(hf_hub_download(REPOS['ember'], 'tokenizer.json', local_files_only=True))
    tok_name = f'{REPOS["ember"]}@{tok_path.parent.name}/tokenizer.json'
    files[tok_name] = tok_path
    tok = Tokenizer.from_file(str(tok_path))
    listener, prior = [], []
    for i in range(len(pieces)):
        # Ember reads the tokens of Quill's text that can no longer change: all but the last of the re-encoding
        stable = tok.encode(''.join(pieces[:i + 1]), add_special_tokens=False).ids[:-1]
        assert stable[:len(prior)] == prior
        listener.append([tok.decode([t]) for t in stable[len(prior):]])
        prior = stable
    chat_name = str(CHAT.relative_to(ROOT)) + ' (uncompressed)'
    sha = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('pairing/link.py', 'pairing/session.py')}
    sha[chat_name] = hashlib.sha256(raw).hexdigest()
    sha.update({name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()})
    snap = json.loads(SNAP.read_text()) if SNAP.exists() else {}
    xb = xblocks()
    snap.update(exchange_blocks=xb, exchange_count=len(xb), configs=configs, private_text=private,
                private_text_source=chat_name, timing_tokens=pieces,
                timing_emission_ticks=[v['tick'] for v in first['tokens'][:4]],
                timing_token_ids=[v['id'] for v in first['tokens'][:4]], timing_listener_chunks=listener,
                timing_tokenizer=tok_name, source_sha256=sha)
    return snap


def facts():
    f = json.loads(SNAP.read_text())
    assert f['exchange_blocks'] == xblocks(), 'snapshot and pairing/link.py disagree on the exchange layers'
    return f


# ---------------------------------------------------------------- drawing helpers (also used by hero_link.py)

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

    def mascot(self, x, y, side, size=130):
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" href="{mascot(side)}"/>')

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


# ---------------------------------------------------------------- figure: which partner tokens can be read

def ticks(f):
    tl = timeline(f)
    title = 'Each model reads only the partner’s tokens from the same time or earlier'
    M, gap, lab, ch, head = 40, 12, 300, 104, 296       # margin, cell gap, row-label width, cell height, header
    top = 130
    hs = {x: head + len(tl[x]) * (ch + gap) - gap + 34 for x in ('quill', 'ember')}
    H = round(top + hs['quill'] + 24 + hs['ember'] + 30)
    s = SVG(W, H, title,
            'For the opening of the linked chat (Quill writes "So, you\'re", Ember reads it as it streams), the grid '
            'of which partner tokens each token may read after every exchange layer, with the time-gap bucket of '
            'each readable pair. Derived from pairing/session.py and pairing/link.py; token timing from '
            'results/link_mechanism.json.')
    s.title(title)

    def tokchip(x, y, w, tok, color, fill):
        if tok in ('prompt', 'turn header'):
            s.text(x + w / 2, y + 46, 'prompt' if tok == 'prompt' else 'header', 42, color, 600, 'middle')
        else:
            s.rect(x + w / 2 - 80, y, 160, 64, fill, color, 12, 3)
            s.text(x + w / 2, y + 46, tok, 44, INK, 600, 'middle')

    def grid(pt, reader, sender):
        rows, cols = tl[reader], tl[sender]
        rc, sc = (Q, E) if reader == 'quill' else (E, Q)
        rl, sl = (QL, EL) if reader == 'quill' else (EL, QL)
        px = M
        s.rect(px, pt, W - 2 * M, hs[reader], COOL, r=36)
        s.mascot(px + 30 - 0.27 * 154, pt + 8, reader, 154)     # the art starts at 0.27 of its square
        s.text(px + 130, pt + 100, [(reader.title(), rc, 700), (' reads ', INK, 400), (sender.title(), sc, 700)], 52)
        gx = px + 30 + lab
        gy = pt + head
        cell = (W - 2 * M - 60 - lab - (len(cols) - 1) * gap) / len(cols)
        cx = [gx + j * (cell + gap) for j in range(len(cols))]
        s.text(px + 30, gy - 96, 'time', 42, INK)
        for j, (tok, t) in enumerate(cols):
            s.text(cx[j] + cell / 2, gy - 96, str(t).replace('-', '−'), 42, INK, 400, 'middle')
            tokchip(cx[j], gy - 80, cell, tok, sc, sl)
        for i, (tok, tq) in enumerate(rows):
            y = gy + i * (ch + gap)
            if tok in ('prompt', 'turn header'):
                s.text(px + 30, y + ch / 2 + 15, 'prompt' if tok == 'prompt' else 'header', 42, rc, 600)
            else:
                s.rect(px + 30, y + ch / 2 - 32, 110, 64, rl, rc, 12, 3)
                s.text(px + 85, y + ch / 2 + 15, tok, 44, INK, 600, 'middle')
            s.text(px + 236, y + ch / 2 + 15, str(tq).replace('-', '−'), 42, INK, 400, 'middle')
            for j, (_, tk) in enumerate(cols):
                if tk <= tq:
                    s.rect(cx[j], y, cell, ch, GL, G, 14, 3)
                    b = bucket(tq, tk)
                    s.text(cx[j] + cell / 2, y + ch / 2 + (18 if b != 'prompt' else 15), b,
                           52 if b != 'prompt' else 42, G, 700, 'middle')
            # the partner's later tokens, masked: one grey run per row (times only grow along a row)
            n = sum(tk <= tq for _, tk in cols)
            if n < len(cols):
                s.rect(cx[n], y, cx[-1] + cell - cx[n], ch, '#E3E3E8', 'none', 14, 0)
        # direct labels in the masked corner: the first row's masked cells, and the second row's time gap
        readable = [sum(tk <= tq for _, tk in cols) for _, tq in rows[:2]]
        x1 = cx[readable[0]]
        s.text((x1 + cx[-1] + cell) / 2, gy + ch / 2 + 15, 'masked: later time', 42, INK, 400, 'middle')
        y2 = gy + ch + gap + ch / 2
        xa = cx[readable[1] - 1] + cell
        s.arrow([(xa + 120, y2), (xa + 10, y2)], G, 4, head=18)
        s.text(xa + 136, y2 + 15, 'time gap', 42, G, 700)

    grid(top, 'quill', 'ember')
    grid(top + hs['quill'] + 24, 'ember', 'quill')
    s.save('link_mechanism_ticks')


# ---------------------------------------------------------------- figure: the 8 exchange layers

def depths(f):
    xb = f['exchange_blocks']
    nb = f['configs']['quill']['blocks']
    assert nb == f['configs']['ember']['blocks'] == creature.LAYERS and tuple(xb) == creature.PORTS
    title = f'After {len(xb)} of the {nb} layers, Quill and Ember read each other at the same time'
    top, M = 130, 40
    # the zoom column on the right: three steps of one exchange, stacked
    vw, vh, sp_ = 88, 110, 104                      # zoom vectors: width, height, pitch
    step_h = 100 + vh + 140 + vh + 56
    zx, zw = 1210, W - M - 1210
    H = round(top + 3 * step_h + 40 + M)
    s = SVG(W, H, title,
            'Quill and Ember drawn as two stacks of 36 layers, with the input at the base and the head at the top. '
            'After layers ' + ', '.join(map(str, xb)) + ' two teal strands join the stacks, one per direction: each '
            'model reads the other from the states before that exchange and adds the result to its own residual '
            'stream. A zoom on layer 9 shows the read in three steps. Schematic; counts from pairing/link.py and the '
            'model configs.')
    s.title(title)

    # anatomy: both models at full height, the link as two strands at each exchange layer
    ah = 1290                                       # creature height, hair tip to base
    cq, ce, ctop = 400, 1000, top + 40
    gq, pq = creature.creature('quill', cq, ctop, ah, full=True)
    ge, pe = creature.creature('ember', ce, ctop, ah, full=True)
    d = (pq[xb[0]][1] - pq[xb[1]][1]) / (xb[1] - xb[0])      # pixels per layer

    def ly(k):
        return pq[xb[0]][1] - (k - xb[0]) * d

    base = ctop + ah
    s.rect(M, top, zx - 30 - M, H - top - M, COOL, r=36)
    for k in xb:
        s.add(creature.strands(cq, ce, ly(k), wave=8, sw=5))
    s.add(gq)
    s.add(ge)
    lx = cq - creature.width_of(ah, True) / 2 - 24  # layer numbers end here
    s.text(lx, ly(nb - 1) - 62, 'layer', 44, INK, 700, 'end')
    for k in [nb - 1] + xb[::-1] + [0]:     # blocks counted from 0, as in the code
        s.text(lx, ly(k) + 15, str(k), 44, G if k in xb else INK, 700, 'end')
    for side, cx, col, light in (('quill', cq, Q, QL), ('ember', ce, E, EL)):
        chip(s, cx - 120, base + 80, 240, 76, f['private_text'][side], col, light, 46, 700, tcolor=INK)
        s.arrow([(cx, base + 78), (cx, base + 10)], col, 5, head=20)
        s.text(cx, base + 80 + 76 + 62, side.title(), 50, col, 700, 'middle')
    assert base + 80 + 76 + 62 + 20 < H - M
    # direct label on the top strand pair: one strand per direction
    yt = ly(xb[-1])
    s.text((cq + ce) / 2, yt - 34, 'both ways', 44, G, 700, 'middle')
    # the zoomed exchange, with a wedge to the column of steps
    z = xb[1]
    rw = creature.BODY_W * creature.width_of(ah, True) / (2 * (creature.HEAD_RX + 6)) / 2 + 26
    bx0, bx1, by0, by1 = cq - rw, ce + rw, ly(z) - 34, ly(z) + 34
    s.rect(bx0, by0, bx1 - bx0, by1 - by0, 'none', INK, 18, 3, dash='10 9')
    s.line([(bx1, by0), (zx, top)], INK, 2.5, dash='8 8')
    s.line([(bx1, by1), (zx, H - M)], INK, 2.5, dash='8 8')
    s.rect(zx, top, zw, H - top - M, 'white', LINE, 28, 3)
    heads = [f'After layer {z}', 'Quill and Ember read at once', 'Each model adds its write']
    sy = [top + 20 + k * step_h for k in range(3)]
    for k in range(1, 3):
        s.line([(zx + 30, sy[k] - 10), (zx + zw - 30, sy[k] - 10)], LINE, 3)
    px = zx + 40
    for k in range(3):
        s.text(px, sy[k] + 62, heads[k], 46, INK, 700)
    ra, rb = (sy[0] + 100, sy[0] + 100 + vh + 140)
    for j in range(5):
        s.vector(px + j * sp_, ra, Q, vw, vh, 1 + j)
        s.vector(px + j * sp_, rb, E, vw, vh, 4 + j)
    s.text(px + 5 * sp_ + 10, ra + vh / 2 + 16, 'Quill', 44, Q, 700)
    s.text(px + 5 * sp_ + 10, rb + vh / 2 + 16, 'Ember', 44, E, 700)
    ra, rb = (sy[1] + 100, sy[1] + 100 + vh + 140)
    qx = [px + j * sp_ for j in range(5)]
    qr, er = 4, 2
    for j in range(5):
        s.vector(qx[j], ra, Q, vw, vh, 1 + j, 1, 'white', None if j == qr else '#CDB9F3')
        s.vector(qx[j], rb, E, vw, vh, 4 + j, 1, 'white', None if j == er else '#F5C6AA')
    for j in range(qr + 1):
        s.arrow([(qx[j] + vw / 2, rb - 4), (qx[qr] + 24 + 9 * j, ra + vh + 6)], G, 3.5, head=15)
    for j in range(er + 1):
        s.arrow([(qx[j] + vw / 2, ra + vh + 4), (qx[er] + 26 + 17 * j, rb - 6)], G, 3.5, head=15)
    tx = qx[4] + vw + 24
    for y, who in ((ra, 'Quill reads Ember'), (rb, 'Ember reads Quill')):
        s.text(tx, y + vh / 2 + 16, who, 44, G, 700)
    assert tx + tw('Ember reads Quill', 44, 700) < zx + zw - 20
    ra, rb = (sy[2] + 100, sy[2] + 100 + vh + 140)
    for y, col, seed in ((ra, Q, 1), (rb, E, 4)):
        s.vector(px, y, col, vw, vh, seed)
        s.plus(px + vw + 26, y + vh / 2, 18, G, 3.5)
        s.vector(px + vw + 60, y + 14, G, 66, vh - 28, seed + 2, 0.7)
        s.arrow([(px + vw + 136, y + vh / 2), (px + vw + 180, y + vh / 2)], col, 4)
        s.vector(px + vw + 188, y, col, vw, vh, seed + 1)
        s.arrow([(px + 2 * vw + 198, y + vh / 2), (px + 2 * vw + 236, y + vh / 2)], col, 4)
        s.rect(px + 2 * vw + 244, y + 10, 210, vh - 20, 'white', '#8E8E93', 12, 2.5)
        s.text(px + 2 * vw + 349, y + vh / 2 + 15, f'layer {z + 1}', 44, INK, 600, 'middle')
    s.text(px + vw + 93, ra + vh + 56, 'write', 44, G, 700, 'middle')
    s.save('link_mechanism_depths')


# ---------------------------------------------------------------- figure: one read in full

def mechanism(f):
    tl = timeline(f)
    xb = f['exchange_blocks']
    z = xb[1]
    reader_tok, reader_t = next((t, k) for t, k in tl['quill'] if t == 'you')
    top, rh = 130, 140
    rows = [('null key', None)] + [(t, k) for t, k in tl['ember']]
    y0 = top + 100
    by = y0 + len(rows) * rh
    qy, sy = by + 72, by + 272
    H = sy + 170
    title = f'After layer {z}, Quill reads Ember’s tokens at its token “{reader_tok}” and adds the result'
    s = SVG(W, H, title,
            'One read through the link after one exchange layer, as in pairing/link.py Side.show and Side.read: '
            f'Quill\'s token "{reader_tok}" (time {reader_t}) attends over every Ember token read at time {reader_t} '
            'or earlier plus an empty slot that adds nothing. Purple parts belong to the reader (Quill), orange parts to the sender '
            '(Ember). Bars, gate meters and weights are drawings, not measured values.')
    s.title(title)
    s.rect(40, top, W - 80, H - top - 30, COOL, r=36)
    C = dict(state=450, key=630, score=830, bias=1025, weight=1215, gate=1395, value=1530, wv=1725)
    heads = [('state', 'state', E), ('key', 'key', E), ('score', 'score', G), ('bias', 'bias', Q),
             ('weight', 'weight', G), ('gate', 'gate', E), ('value', 'value', E), ('wv', 'weighted', G)]
    hy = top + 66
    s.text(70, hy, 'position', 42, E, 700)
    for k, h1, col in heads:
        s.text(C[k], hy, h1, 42, col, 700, 'middle')
    weights = [0.16, 0.34, 0.06, 0.14, 0.30, 0.0]
    scores = [0.45, 0.7, 0.25, 0.5, 0.65, 0.6]
    gates = [None, 0.88, 0.55, 0.8, 0.9, 0.85]
    vw, vh = 88, 88
    faded = '#F5C6AA'
    for i, (tok, t) in enumerate(rows):
        y = y0 + i * rh
        cy = y + rh / 2 - 6
        hidden = t is not None and t > reader_t
        if i % 2 == 0:
            s.rect(56, y - 2, 1760, rh - 8, '#E9E9EE', 'none', 18, 0)
        if tok == 'null key':
            chip(s, 70, cy - 32, 250, 64, 'empty slot', Q, QL, 40, 700, tcolor=INK)
        elif tok in ('prompt', 'turn header'):
            s.text(70, cy, 'prompt' if tok == 'prompt' else 'header', 42, E, 700)
            s.text(70, cy + 48, f'time {t}'.replace('-', '−'), 42, INK)
        else:
            chip(s, 70, cy - 32, 100, 64, tok, E, EL, 42, 600, tcolor=INK)
            s.text(186, cy + 15, f'time {t}', 42, INK)
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
            s.text((C['score'] - 86 + C['wv'] + 56) / 2, cy + 14, 'masked: later time', 40, INK, 400, 'middle')
            continue
        s.rect(C['score'] - 76, cy - 15, 152, 30, 'white', LINE, 9, 2)
        s.rect(C['score'] - 76, cy - 15, 152 * scores[i], 30, G, 'none', 9, 0, opacity=0.55)
        b = '0' if tok == 'null key' else bucket(reader_t, t)
        if tok == 'null key':
            s.text(C['bias'], cy + 14, '–', 42, INK, 400, 'middle')
        else:
            chip(s, C['bias'] - 84, cy - 30, 168, 60, b, Q, QL, 40, 700, tcolor=Q)
        s.rect(C['weight'] - 88, cy - 16, 176, 32, 'white', LINE, 9, 2)
        s.rect(C['weight'] - 88, cy - 16, 176 * weights[i] / max(weights), 32, G, 'none', 9, 0)
        if tok == 'null key':
            s.text(C['gate'], cy + 14, '–', 42, INK, 400, 'middle')
            s.rect(C['value'] - vw / 2, vy, vw, vh, 'white', Q, 10, 2.5, dash='7 6')
            s.text(C['value'], cy + 15, '0', 42, Q, 700, 'middle')
            s.text(C['wv'], cy + 15, '0', 42, G, 700, 'middle')
        else:
            s.meter(C['gate'] - 15, vy - 2, gates[i], E, 30, vh + 4)
            s.vector(C['value'] - vw / 2, vy, E, vw, vh, i + 5, gates[i])
            sc = 0.45 + 0.55 * weights[i] / max(weights)
            s.vector(C['wv'] - vw / 2 * sc, cy - vh / 2 * sc, G, vw * sc, vh * sc, i + 5, gates[i])
    s.vector(70, sy - 40, Q, 80, 80, 1)
    s.text(70, sy + 82, f'Quill: “{reader_tok}”, time {reader_t}', 40, Q, 700)
    s.arrow([(154, sy), (C['wv'] + 140, sy)], Q, 9)
    s.circle(250, sy, 8, Q, Q, 0)
    s.arrow([(250, sy - 8), (250, qy + 38)], Q, 4, head=17)
    chip(s, 110, qy - 36, 280, 72, 'query map', Q, 'white', 40, 600, tcolor=INK)
    s.arrow([(392, qy), (452, qy)], Q, 4, head=17)
    s.vector(458, qy - 38, Q, 78, 76, 2)
    s.text(497, qy + 84, 'query', 40, Q, 700, 'middle')
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
    s.text(ox + 58, sy_ + 14, 'sum', 42, G, 700)
    s.arrow([(ox, sy_ + 42), (ox, qy - 38)], G, 4, head=17)
    chip(s, ox - 140, qy - 36, 280, 72, 'output map', Q, 'white', 40, 600, tcolor=INK)
    s.arrow([(ox, qy + 38), (ox, sy - 108)], G, 4, head=17)
    s.times(ox, sy - 82, 24, Q, 4)
    s.text(ox + 40, sy - 68, 'write size', 40, Q, 700)
    s.arrow([(ox, sy - 57), (ox, sy - 28)], G, 4, head=15)
    s.plus(ox, sy, 26, Q, 5)
    s.arrow([(ox + 27, sy), (ox + 100, sy)], Q, 9)
    s.rect(ox + 106, sy - 40, 196, 80, 'white', '#8E8E93', 14, 3)
    s.text(ox + 204, sy + 14, f'layer {z + 1}', 40, INK, 600, 'middle')
    s.save('link_mechanism')


if __name__ == '__main__':
    if '--refresh' in sys.argv:
        SNAP.write_text(json.dumps(refresh(), indent=2) + '\n')
    f = facts()
    mechanism(f)
    depths(f)
    ticks(f)
