"""Hand-built SVG city-intervention heatmaps, from measured full-city log P.

Each direction has its own landscape figure. Color is softmax of mean log P
across the three openings, conditional on the twelve candidate cities (geometric
pooling), using the same 0..1 scale in every panel. The exact permutation count
distribution and observed rank-one count are drawn below the corresponding map.
All reported values, tokens, example openings and city lists come from JSON.
"""
import argparse
import base64
import html
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = '#1D1D1F'
GREEN = '#1A9E77'
RED = '#D2432F'
COLORS = {'quill': '#7B3FE4', 'ember': '#E8590C'}
W, H = 2300, 1630


class SVG:
    def __init__(self):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                      '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, x, y, w, h, fill, radius=0, stroke='none', sw=0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=36, color=INK, bold=False, anchor='start', rotate=None):
        transform = '' if rotate is None else f' transform="rotate({rotate} {x} {y})"'
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}"{transform}>{html.escape(str(s))}</text>')

    def line(self, x, y, xx, yy, color, width=5):
        self.parts.append(f'<path d="M{x},{y} L{xx},{yy}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linecap="round"/>')

    def robot(self, side, x, y, size=130):
        data = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        self.parts.append(f'<image href="data:image/png;base64,{data}" x="{x}" y="{y}" width="{size}" height="{size}"/>')

    def arrow(self, x, y, xx, yy, color):
        self.line(x, y, xx, yy, color)
        self.parts.append(f'<path d="M{xx - 13},{yy - 9} L{xx},{yy} L{xx - 13},{yy + 9}" fill="none" stroke="{color}" stroke-width="5"/>')

    def finish(self):
        return '\n'.join(self.parts + ['</svg>'])


def blend(p):
    low = np.array([244, 244, 246])
    high = np.array([26, 158, 119])
    c = np.rint(low + float(p) * (high - low)).astype(int)
    return '#%02x%02x%02x' % tuple(c)


def figure(data, side):
    s = SVG()
    design = data['design']
    cities = design['cities']
    k = len(cities)
    other = 'ember' if side == 'quill' else 'quill'
    name, partner = side.title(), other.title()
    s.text(55, 68, f'The private city changes {name}’s city scores', 52, bold=True)
    s.text(1910, 39, 'Relative city score', 36, anchor='middle')
    s.parts.append(f'<defs><linearGradient id="scale" color-interpolation="sRGB"><stop stop-color="{blend(0)}"/><stop offset="1" stop-color="{blend(1)}"/></linearGradient></defs>')
    s.rect(1725, 56, 350, 18, 'url(#scale)')
    s.text(1695, 85, '0', anchor='end')
    s.text(2100, 85, '100%')
    # A separate training card makes the split visible without explanatory prose.
    s.rect(40, 103, 320, 208, '#F4F4F6', 25)
    s.text(65, 151, 'Training', 38, bold=True)
    for j, city in enumerate(design['training_cities'][:2]):
        s.rect(65, 174 + 57 * j, 270, 48, 'white', 14)
        s.text(200, 210 + 57 * j, city, anchor='middle')
    s.rect(385, 103, 1875, 208, '#FAF0E6', 25)
    s.text(415, 151, 'Never seen in training', 38, bold=True)
    shown = cities[0]
    s.text(420, 205, f'{partner}’s private city', color=COLORS[other])
    s.rect(420, 224, 290, 58, 'white', 15)
    s.text(565, 265, shown, color=COLORS[other], bold=True, anchor='middle')
    s.arrow(720, 249, 790, 249, COLORS[other])
    s.robot(other, 800, 161, 130)
    s.text(1040, 198, 'Hidden states', color=GREEN, anchor='middle')
    s.line(925, 249, 1145, 249, GREEN)
    for j, height in enumerate([35, 57, 44, 68, 40, 52, 31]):
        s.rect(969 + j * 20, 249 - height / 2, 14, height, GREEN, 3)
    s.arrow(1110, 249, 1150, 249, GREEN)
    s.robot(side, 1160, 161, 130)
    for j, template in enumerate(design['templates'][side]):
        s.rect(1295, 168 + j * 43, 940, 40, 'white', 12)
        s.text(1315, 198 + j * 43, template['prefix'].rstrip(), color=COLORS[side])
        choice = cities[int(np.argmax(data['raw_logp'][side]['link'][j][0]))]
        color = GREEN if choice == shown else RED
        s.rect(2050, 168 + j * 43, 175, 40, '#E2F1EB' if choice == shown else '#FBE8E4', 10)
        s.text(2137, 198 + j * 43, choice, color=color, anchor='middle')
    s.text(1295, 150, f'{name} (4-bit): supplied openings', color=COLORS[side])
    s.text(2137, 150, 'Top city', anchor='middle')
    cell = 43
    base_y = 705
    panels = [('link', 'Link open'), ('cut', 'Link cut'), ('shuffle', 'Partner swapped')]
    for z, (cond, title) in enumerate(panels):
        left = 40 + z * 750
        mx = left + 180
        s.rect(left, 338, 720, 1248, '#FAF0E6' if cond == 'link' else '#F4F4F6', 27)
        s.text(left + 360, 398, title, 44, GREEN if cond == 'link' else INK, True, 'middle')
        st = data['stats'][side][cond]
        s.text(left + 360, 465, f'{st["percent"]:.0f}% first by average score', 40,
               RED if cond == 'link' else INK, True, 'middle')
        s.text(mx + k * cell / 2, 524, f'{k} candidate city tokens', color=COLORS[side], anchor='middle')
        s.text(left + 16, base_y - 62, 'Private', color=COLORS[other], bold=True)
        s.text(left + 16, base_y - 18, 'city', color=COLORS[other], bold=True)
        for j, city in enumerate(cities):
            s.text(mx + j * cell + cell / 2 + 9, base_y - 20, city, rotate=-72)
            s.text(mx - 18, base_y + j * cell + 33, city, color=COLORS[other], anchor='end')
        a = np.asarray(data['mean_logp'][side][cond])
        a = np.exp(a - a.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        for i in range(k):
            for j in range(k):
                s.rect(mx + j * cell, base_y + i * cell, cell - 2, cell - 2, blend(a[i, j]), 4)
        # Exact null distribution as marks; the observed count is a red triangle.
        hist_y, hist_x = 1465, mx
        s.text(left + 360, 1280, 'Permuted city labels', 36, anchor='middle')
        counts = np.asarray(st['counts'], dtype=float)
        for j, count in enumerate(counts):
            height = 90 * count / counts.max()
            s.rect(hist_x + j * 38, hist_y - height, 30, height, '#1D1D1F', 3)
        obsx = hist_x + st['observed'] * 38 + 15
        s.parts.append(f'<path d="M{obsx - 13},1345 L{obsx + 13},1345 L{obsx},1362 Z" fill="{RED}"/>')
        for j in sorted({0, k, st['observed']}):
            s.text(hist_x + j * 38 + 15, hist_y + 52, j, anchor='middle',
                   color=RED if j == st['observed'] else INK, bold=j == st['observed'])
        s.text(min(left + 625, max(left + 220, obsx)), 1327, 'Observed', 36, anchor='middle', color=RED)
        s.text(left + 360, 1560, 'Partner cities ranked first', 36, anchor='middle')
    return s.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=str(ROOT / 'results/city_matrix.json'))
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    if not data.get('complete'):
        raise ValueError('only complete measured results can be rendered')
    for side in ('quill', 'ember'):
        path = ROOT / 'figures' / f'city_matrix_{side}'
        path.with_suffix('.svg').write_text(figure(data, side))
        subprocess.run(['rsvg-convert', '-o', str(path.with_suffix('.png')), str(path.with_suffix('.svg'))], check=True)
        subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(path.with_suffix('.pdf')), str(path.with_suffix('.svg'))], check=True)
        print(path)


if __name__ == '__main__':
    main()
