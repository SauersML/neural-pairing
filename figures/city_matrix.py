"""Which city a twin scores highest after an opening line, given the city its linked partner holds.

Reads results/city_matrix.json (written by pairing/city_matrix.py on the Mac: nl22c step 157, both models with 4-bit
MLX weights) and draws one figure per receiving twin: figures/city_matrix_{quill,ember}.{svg,png,pdf}.

Each cell is the receiver's share of probability over the 12 candidate cities after an opening line, from the full-city
log probability averaged over the 3 openings (softmax over the 12 cities of the mean log probability). Panels: link on,
link cut, and linked to a stranger (a twin of the partner holding a different city). The stranger panel is drawn with
its rows sorted by the stranger's city (data['donor_indices'] says which stranger each row was linked to), so a
diagonal there means the receiver names the stranger's city.

Under each matrix: the exact chance distribution of how many of the 12 rows put the row's own city on top when the 12
city labels are shuffled (all 12! orderings, computed here from the rook polynomial of the top-city board and checked
against the counts stored in the JSON). No model is loaded.
"""
import base64
import html
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/city_matrix.json'

QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
MUTED, FAINT, COOL, WARM = '#6E6E73', '#B9B9C0', '#F4F4F6', '#FAF0E6'
COLORS = {'quill': QUILL, 'ember': EMBER}
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
FONT_DIR = Path('/usr/share/fonts/opentype/inter')
_FONTS = {}


def font(size, weight=400):
    key = (size, weight)
    if key not in _FONTS:
        name = {400: 'Inter-Regular', 600: 'Inter-SemiBold', 700: 'Inter-Bold'}[weight]
        try:
            _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f'{name}.otf'), size)
        except OSError:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def width(s, size, weight=400):
    return font(size, weight).getlength(str(s))


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<rect width="{w}" height="{h}" fill="white"/>']

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill='white', radius=18, stroke='none', sw=2, extra=''):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=3):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linecap="round"/>')

    def path(self, d, color=INK, sw=4, fill='none'):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')

    def text(self, x, y, s, size=34, weight=400, fill=INK, anchor='start', rotate=None):
        tr = '' if rotate is None else f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{tr}>{html.escape(str(s))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start'):
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def mascot(self, side, x, y, size, grey=False):
        data = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        filt = ' filter="url(#grey)"' if grey else ''
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}"{filt} '
                 f'xlink:href="data:image/png;base64,{data}"/>')

    def save(self, stem):
        stem.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']))
        for fmt in ('png', 'pdf'):
            subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(stem.with_suffix('.' + fmt)),
                            str(stem.with_suffix('.svg'))], check=True)


def link_arrow(s, x1, x2, y, on=True):
    """The two-way link: teal with a head at both ends; grey and crossed out when cut."""
    color = LINK if on else FAINT
    for xa, d in ((x1, 1), (x2, -1)):
        s.path(f'M{xa + d * 16:.1f},{y - 12} L{xa},{y} L{xa + d * 16:.1f},{y + 12}', color, 6)
    if on:
        s.line(x1 + 3, y, x2 - 3, y, color, 6)
    else:
        mid = (x1 + x2) / 2
        s.line(x1 + 3, y, mid - 16, y, color, 6)
        s.line(mid + 16, y, x2 - 3, y, color, 6)
        s.line(mid - 9, y - 11, mid + 9, y + 11, MUTED, 5)
        s.line(mid - 9, y + 11, mid + 9, y - 11, MUTED, 5)


def blend(p):
    low, high = np.array([255, 255, 255]), np.array([26, 158, 119])
    return '#%02x%02x%02x' % tuple(np.rint(low + float(p) * (high - low)).astype(int))


def shares(mean_logp):
    a = np.asarray(mean_logp, dtype=float)
    a = np.exp(a - a.max(1, keepdims=True))
    return a / a.sum(1, keepdims=True)


def chance_counts(top):
    """For each k, the number of the n! labellings under which exactly k rows have their own label as the top column.
    The board has one cell per row, (i, top[i]); r_j (ways to place j non-attacking rooks) is the j-th elementary
    symmetric polynomial of the column multiplicities, and inclusion-exclusion gives the exact counts."""
    n = len(top)
    mult = np.bincount(top, minlength=n)
    r = [1] + [0] * n
    for m in mult:
        for j in range(n, 0, -1):
            r[j] += r[j - 1] * int(m)
    return [sum((-1) ** (j - k) * math.comb(j, k) * r[j] * math.factorial(n - j) for j in range(k, n + 1))
            for k in range(n + 1)]


def panels(data, side):
    """[(condition, title, row labels, matrix rows, top-city indices, observed, counts)] with the stranger panel's
    rows sorted by the stranger's city."""
    cities = data['design']['cities']
    k = len(cities)
    donor = data['donor_indices']
    out = []
    for cond in ('link', 'cut', 'shuffle'):
        a = shares(data['mean_logp'][side][cond])
        top = np.asarray(data['mean_logp'][side][cond]).argmax(1)
        st = data['stats'][side][cond]
        counts = chance_counts(list(top))
        assert counts == st['counts'], (side, cond)          # the stored exact distribution
        assert int((top == np.arange(k)).sum()) == st['observed']
        if cond == 'shuffle':
            # row i was linked to the stranger holding cities[donor[i]]; sort rows by that city
            order = np.argsort(donor)
            assert all(donor[order[j]] == j for j in range(k))
            a, top = a[order], top[order]
            labels = [cities[donor[i]] for i in order]
            partner_city = [cities[i] for i in order]
            # the stranger panel equals the link panel of the stranger's city (audited on the Mac)
            assert data['audits'][side]['shuffle_vs_actual_donor_max_difference'] == 0.0
        else:
            labels, partner_city = list(cities), list(cities)
        observed = int((top == np.arange(k)).sum())
        counts = chance_counts(list(top))
        assert sum(counts) == math.factorial(k)
        out.append(dict(cond=cond, labels=labels, partner=partner_city, a=a, top=top, observed=observed,
                        counts=counts, stored_observed=st['observed']))
    return out


def figure(data, side):
    cities = data['design']['cities']
    k = len(cities)
    other = 'ember' if side == 'quill' else 'quill'
    name, partner = side.title(), other.title()
    me, you = COLORS[side], COLORS[other]
    W, M = 2400, 36
    top_y, strip_h = 130, 300
    cell = 46
    head_h, cols_h = 150, 140
    hist_h = 230
    py = top_y + strip_h + 24
    ph = head_h + cols_h + k * cell + hist_h
    H = py + ph + M
    s = SVG(W, H)
    s.add('<defs><filter id="grey"><feColorMatrix type="saturate" values="0"/>'
          '<feComponentTransfer><feFuncR type="linear" slope="0.62" intercept="0.35"/>'
          '<feFuncG type="linear" slope="0.62" intercept="0.35"/><feFuncB type="linear" slope="0.62" intercept="0.35"/>'
          '</feComponentTransfer></filter>'
          f'<linearGradient id="scale"><stop stop-color="{blend(0)}"/><stop offset="1" stop-color="{blend(1)}"/>'
          '</linearGradient></defs>')
    s.text(M, 92, f'{name} names its linked partner’s city', 64, 700)
    # colour key, top right
    kx = W - M - 360
    s.text(kx, 52, 'probability share', 30, 600, MUTED)
    s.rect(kx, 66, 360, 30, 'url(#scale)', 6, '#D8D8DE', 1)
    s.text(kx, 128, '0%', 28, 400, MUTED)
    s.text(kx + 360, 128, '100%', 28, 400, MUTED, 'end')

    # Setup strip: the partner's city, the link, the receiver's openings and its top city after each.
    s.rect(M, top_y, W - 2 * M, strip_h, WARM, 32)
    shown = cities[0]
    x0 = M + 40
    s.text(x0, top_y + 70, f'{partner}’s city', 38, 700, you)
    s.rect(x0, top_y + 100, 320, 86, 'white', 18, you, 3)
    s.text(x0 + 160, top_y + 158, shown, 44, 700, you, 'middle')
    mx = x0 + 380
    msize = 170
    s.mascot(other, mx, top_y + 60, msize)  # partner left, receiver right, as in the panels
    link_arrow(s, mx + msize + 4, mx + msize + 140, top_y + 60 + msize * 0.6)
    s.text(mx + msize + 72, top_y + 60 + msize * 0.6 - 28, 'link', 32, 600, LINK, 'middle')
    s.mascot(side, mx + msize + 144, top_y + 60, msize)
    s.text(mx + msize / 2, top_y + 272, partner, 36, 700, you, 'middle')
    s.text(mx + msize + 144 + msize / 2, top_y + 272, name, 36, 700, me, 'middle')
    ox = mx + 2 * msize + 200
    col_city = W - M - 40 - 260
    s.text(ox, top_y + 62, f'{name}’s opening', 34, 700, me)
    s.text(col_city + 130, top_y + 62, 'top city', 34, 700, MUTED, 'middle')
    for j, template in enumerate(data['design']['templates'][side]):
        y = top_y + 86 + j * 68
        s.rect(ox, y, col_city - ox - 16, 58, 'white', 14)
        s.text(ox + 22, y + 41, '“' + template['prefix'].rstrip() + ' …”', 36, 400, me)
        choice = cities[int(np.argmax(data['raw_logp'][side]['link'][j][0]))]
        ok = choice == shown
        s.rect(col_city, y, 260, 58, '#DDF0E9' if ok else '#FBE3DE', 14)
        s.text(col_city + 130, y + 42, choice, 38, 700, LINK if ok else RED, 'middle')

    # Three matrices
    gap = 24
    pw = (W - 2 * M - 2 * gap) / 3
    my_off = head_h + cols_h
    titles = {'link': 'Link on', 'cut': 'Link cut', 'shuffle': 'Linked to a stranger'}
    for z, p in enumerate(panels(data, side)):
        cond = p['cond']
        left = M + z * (pw + gap)
        s.rect(left, py, pw, ph, WARM if cond == 'link' else COOL, 32)
        isz = 84
        s.mascot(other, left + 30, py + 26, isz, grey=cond == 'shuffle')
        link_arrow(s, left + 30 + isz + 2, left + 30 + isz + 70, py + 26 + isz * 0.62, cond != 'cut')
        s.mascot(side, left + 30 + isz + 74, py + 26, isz)
        tx = left + 30 + 2 * isz + 100
        s.text(tx, py + 82, titles[cond], 42, 700)
        good = p['observed'] == k
        col = (LINK if cond == 'shuffle' else RED) if good else INK
        s.text(tx, py + 140, f'{p["observed"]} of {k}', 44, 700, col)
        # matrix
        mx0 = left + pw - 30 - k * cell
        my0 = py + my_off
        for j, city in enumerate(cities):
            s.text(mx0 + j * cell + cell / 2 + 10, my0 - 12, city, 30, 400, me, rotate=-62)
        stranger = cond == 'shuffle'
        row_head = 'stranger' if stranger else f'{partner}’s city'
        s.text(mx0 - 14, my0 - 16, row_head, 28, 600, MUTED if stranger else you, 'end')
        for i in range(k):
            s.text(mx0 - 14, my0 + i * cell + 34, p['labels'][i], 32, 400, MUTED if stranger else you, 'end')
            for j in range(k):
                s.rect(mx0 + j * cell + 1, my0 + i * cell + 1, cell - 2, cell - 2, blend(p['a'][i, j]), 5,
                       '#E6E6EA', 1)
        # chance distribution of the count, over all orderings of the city names
        counts = np.asarray(p['counts'], dtype=float)
        bpitch = k * cell / (k + 1)
        bw = bpitch - 10
        bh = 110
        base = my0 + k * cell + 40 + bh + 20
        for j, c in enumerate(counts):
            h = bh * c / counts.max()
            if c > 0:
                s.rect(mx0 + j * bpitch + 5, base - max(h, 3), bw, max(h, 3), FAINT, 4)
        s.line(mx0, base + 1, mx0 + k * cell, base + 1, MUTED, 2)
        for j in (0, 6, 12):
            s.text(mx0 + j * bpitch + bpitch / 2, base + 38, j, 30, 400, MUTED, 'middle')
        s.text(mx0 - 14, base - 6, 'chance', 30, 600, MUTED, 'end')
        ox = mx0 + p['observed'] * bpitch + bpitch / 2
        tip = base - bh - 12
        s.path(f'M{ox - 14:.1f},{tip - 24} L{ox + 14:.1f},{tip - 24} L{ox:.1f},{tip} Z', col, 2, col)
        s.line(ox, tip, ox, base, col, 3)
    s.save(ROOT / 'figures' / f'city_matrix_{side}')


def main():
    data = json.loads(DATA.read_text())
    if not data.get('complete'):
        raise ValueError('only complete measured results can be rendered')
    for side in ('quill', 'ember'):
        figure(data, side)
        print('figures/city_matrix_' + side)


if __name__ == '__main__':
    main()
