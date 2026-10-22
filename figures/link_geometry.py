"""Hand-built SVG figures of measured link writes; no plotting defaults.

Reads only results/link_geometry.json (made by pairing/link_geometry.py from a 4-bit MLX run of magic-nl22c).
Each write is what the link adds to the reader's residual stream at an exchange depth, at the position that predicts
the reader's first reply token; the partner's prompt holds one of 20 cities never used in training. The projection and
the probe are fitted on 3 phrasings of the prompt (60 writes per reader) and every mark drawn here is one of the 60
writes from the 3 other phrasings. The fitted writes' coordinates are not stored, so they are not drawn.

Outputs SVG, 2300-pixel PNG and vector PDF via rsvg-convert:
  link_geometry_probe   probe accuracy on new phrasings at all 8 exchange depths, both readers
  link_geometry_quill   city-guided projection of writes into Quill: best depth and worst depth
  link_geometry_ember   the same for Ember
"""
import argparse
import base64
from html import escape
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INK = '#1D1D1F'
MUTED = '#5E5E66'
PURPLE, ORANGE, GREEN, RED = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
CREAM, GREY = '#FAF0E6', '#F4F4F6'
SIDE = {'quill': PURPLE, 'ember': ORANGE}
COLORS = ['#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1565C0', '#A61E6E', '#697900', '#007D8A',
          '#9C4A14', '#4444A6', '#A34F66', '#526B37', '#006E5C', '#9D6500', '#5B407D', '#B52A35',
          '#335D84', '#70523B', '#5F6D9C', '#13782F']
WIDTH = 2300
FAMILY = 'Arial,Helvetica,sans-serif'
SHAPES = {3: 'circle', 4: 'square', 5: 'triangle'}


def font_file(bold):
    """Font used to measure text: the face rsvg-convert picks for Arial here (Liberation Sans) or macOS Arial."""
    names = (['LiberationSans-Bold.ttf', 'Arial Bold.ttf'] if bold else ['LiberationSans-Regular.ttf', 'Arial.ttf'])
    for base in ('/usr/share/fonts/truetype/liberation', '/System/Library/Fonts/Supplemental', '/Library/Fonts'):
        for n in names:
            if Path(base, n).exists():
                return str(Path(base, n))
    out = subprocess.run(['fc-match', '-f', '%{file}', 'Arial:bold' if bold else 'Arial'],
                         capture_output=True, text=True).stdout
    return out


_FONTS = {}


def tw(s, size, bold=False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(font_file(bold), size)
    return _FONTS[key].getlength(str(s))


class Canvas:
    def __init__(self, height):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
                      f'viewBox="0 0 {WIDTH} {height}">',
                      '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto" '
                      'markerUnits="userSpaceOnUse"><path d="M0 0 L12 6 L0 12 Z" fill="#1A9E77"/></marker></defs>',
                      '<rect width="100%" height="100%" fill="white"/>']

    def text(self, x, y, s, color=INK, size=36, weight=400, anchor='start', halo=None, italic=False):
        style = (f' paint-order="stroke" stroke="{halo}" stroke-width="9" stroke-linejoin="round"' if halo else '')
        style += ' font-style="italic"' if italic else ''
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" font-weight="{weight}" '
                          f'font-family="{FAMILY}" text-anchor="{anchor}" xml:space="preserve"{style}>{escape(str(s))}</text>')

    def runs(self, x, y, runs, size=36):
        """One line of mixed text: runs of (text, color, weight)."""
        for s, color, weight in runs:
            self.text(x, y, s, color, size, weight)
            x += tw(s, size, weight >= 600)
        return x

    def rect(self, x, y, w, h, fill, radius=32, stroke='none', sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=3, dash=None, arrow=False, cap='butt'):
        attrs = f' stroke-dasharray="{dash}"' if dash else ''
        attrs += ' marker-end="url(#arrow)"' if arrow else ''
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                          f'stroke-width="{sw}" stroke-linecap="{cap}"{attrs}/>')

    def mark(self, shape, x, y, size, color, stroke='white', sw=2.5):
        if shape == 'circle':
            self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{color}" stroke="{stroke}" '
                              f'stroke-width="{sw}"/>')
        elif shape == 'square':
            s = size * .9
            self.parts.append(f'<rect x="{x-s:.1f}" y="{y-s:.1f}" width="{2*s:.1f}" height="{2*s:.1f}" rx="2" '
                              f'fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')
        else:
            s = size * 1.25
            self.parts.append(f'<path d="M{x:.1f} {y-s:.1f} L{x+s*.93:.1f} {y+s*.62:.1f} L{x-s*.93:.1f} '
                              f'{y+s*.62:.1f} Z" fill="{color}" stroke="{stroke}" stroke-width="{sw}" '
                              f'stroke-linejoin="round"/>')

    def robot(self, side, x, y, size=110):
        uri = base64.b64encode((ROOT / f'figures/assets/{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" '
                          f'href="data:image/png;base64,{uri}"/>')

    def save(self, stem, svg_only=False):
        path = ROOT / 'figures' / stem
        path.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']) + '\n')
        if not svg_only:
            for ext in ('png', 'pdf'):
                subprocess.run(['rsvg-convert', '-f', ext, '-o', str(path.with_suffix('.' + ext)),
                                str(path.with_suffix('.svg'))], check=True)


def partner(side):
    return 'ember' if side == 'quill' else 'quill'


def overlap(a, b, pad=0):
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def seg_hits_box(p, q, box):
    low, high = 0., 1.
    for d in (0, 1):
        delta = q[d] - p[d]
        if abs(delta) < 1e-9:
            if not box[d] <= p[d] <= box[d + 2]:
                return False
        else:
            u, v = sorted(((box[d] - p[d]) / delta, (box[d + 2] - p[d]) / delta))
            low, high = max(low, u), min(high, v)
            if low > high:
                return False
    return True


def place_labels(items, marks, bounds, size, bold=False):
    """Collision-free labels near measured clusters; marks never move.

    items: (cx, cy, radius, text). marks: (x, y, r). Returns (box, leader_needed) per item.
    """
    left, top, right, bottom = bounds
    mark_boxes = [(x - r - 3, y - r - 3, x + r + 3, y + r + 3) for x, y, r in marks]
    placed = {}
    crowd = [sum(math.dist(a[:2], b[:2]) < 160 for b in items) for a in items]
    order = sorted(range(len(items)), key=lambda i: (-crowd[i], items[i][1]))
    h = size * 1.0
    for i in order:
        cx, cy, rad, s = items[i]
        w = tw(s, size, bold)
        best = None
        for dist in range(int(rad) + 8, 900, 6):
            for k in range(32):
                a = 2 * math.pi * k / 32
                ux, uy = math.cos(a), math.sin(a)
                # Box edge nearest the cluster sits at `dist` along the direction.
                bx = cx + ux * (dist + w / 2 * abs(ux) ** .5) - w / 2
                by = cy + uy * (dist + h / 2 * abs(uy) ** .5) - h / 2
                box = (bx, by, bx + w, by + h)
                if box[0] < left or box[2] > right or box[1] < top or box[3] > bottom:
                    continue
                if any(overlap(box, b, 4) for b in mark_boxes):
                    continue
                if any(overlap(box, b[0], 10) for b in placed.values()):
                    continue
                far = dist > rad + 30
                if far:
                    end = (min(max(cx, box[0]), box[2]), min(max(cy, box[1]), box[3]))
                    if any(seg_hits_box((cx, cy), end, b[0]) for b in placed.values()):
                        continue
                cost = dist + 25 * abs(uy)  # Prefer labels beside the cluster.
                if best is None or cost < best[0]:
                    best = (cost, box, far)
            if best is not None and dist > best[0] + 30:
                break
        if best is None:
            raise ValueError(f'No room for label {s}')
        placed[i] = (best[1], best[2])
    return [placed[i] for i in range(len(items))]


def layer_at(r, block):
    return next(v for v in r['layers'] if v['block'] == block)


def fit_points(pts, area, margin):
    """Uniform scale (both projection axes are in within-city spread units), centred in the area."""
    left, top, right, bottom = area
    lo = [min(p[d] for p in pts) for d in (0, 1)]
    hi = [max(p[d] for p in pts) for d in (0, 1)]
    scale = min((right - left - 2 * margin[0]) / (hi[0] - lo[0]), (bottom - top - 2 * margin[1]) / (hi[1] - lo[1]))
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return [(cx + (p[0] - mid[0]) * scale, cy - (p[1] - mid[1]) * scale) for p in pts]


def swap_axes(pts, area):
    """True when the transposed view gets the larger uniform scale in this area."""
    w, h = area[2] - area[0], area[3] - area[1]
    xr = max(p[0] for p in pts) - min(p[0] for p in pts)
    yr = max(p[1] for p in pts) - min(p[1] for p in pts)
    return min(w / yr, h / xr) > min(w / xr, h / yr)


def map_panel(c, data, side, block, box, good):
    """City-guided projection of the 60 held-out writes at one exchange depth."""
    r = data['per_receiver'][side]
    layer = layer_at(r, block)
    x0, y0, w, h = box
    fill = CREAM if good else GREY
    c.rect(x0, y0, w, h, fill)
    c.text(x0 + 40, y0 + 78, f'After block {block}', size=52, weight=700)
    c.text(x0 + w - 40, y0 + 78, f'probe {layer["correct"]}/{layer["total"]}', SIDE[side], 44, 700, 'end')
    samples = [r['samples'][i] for i in r['test_indices']]
    cities = [r['cities'][e] for e in layer['expected']]
    assert cities == [s['city'] for s in samples]
    size = 16
    area = (x0 + 40, y0 + 110, x0 + w - 40, y0 + h - 30)
    pts = layer['lda']
    if not good and swap_axes(pts, area):
        # The two projection axes are unlabelled; showing axis 2 across uses the panel better.
        pts = [(p[1], p[0]) for p in pts]
    xy = fit_points(pts, area, (95, 30) if good else (30, 30))
    if good:
        marks = [(x, y, size + 5) for x, y in xy]
        items = []
        for n in sorted(set(cities)):
            group = [p for p, cn in zip(xy, cities) if cn == n]
            cx, cy = sum(p[0] for p in group) / len(group), sum(p[1] for p in group) / len(group)
            items.append((cx, cy, max(math.dist((cx, cy), p) for p in group) + size, n))
        labels = place_labels(items, marks, area, 36, True)
        for (cx, cy, _, n), (lb, far) in zip(items, labels):
            if far:
                end = (min(max(cx, lb[0] - 3), lb[2] + 3), min(max(cy, lb[1] + 4), lb[3] - 4))
                c.line(cx, cy, *end, COLORS[r['cities'].index(n)], 3)
    for (x, y), s, e in zip(xy, samples, layer['expected']):
        c.mark(SHAPES[s['wording']], x, y, size, COLORS[e])
    if good:
        for (cx, cy, _, n), (lb, far) in zip(items, labels):
            c.text(lb[0], lb[3] - 7, n, COLORS[r['cities'].index(n)], 36, 700, halo=fill)


def maps(data, side, good_block, bad_block, svg_only=False):
    other = partner(side)
    # Panel height follows the best-depth map's shape at full width, so the map fills its panel.
    pts = layer_at(data['per_receiver'][side], good_block)['lda']
    xr, yr = (max(p[d] for p in pts) - min(p[d] for p in pts) for d in (0, 1))
    height = round(min(1360, max(1040, 110 + 30 + 60 + yr * (1450 - 80 - 190) / xr)))
    c = Canvas(115 + height + 25)
    c.text(45, 78, f'Writes into {side.title()} group by city or phrasing', size=60, weight=700)
    map_panel(c, data, side, good_block, (45, 115, 1450, height), True)
    # Key: icons and a few words.
    c.rect(1520, 115, 735, 300, GREY)
    c.robot(other, 1555, 132, 84)
    c.line(1655, 174, 1755, 174, GREEN, 6, arrow=True)
    c.robot(side, 1770, 132, 84)
    for j, wd in enumerate((3, 4, 5)):
        c.mark(SHAPES[wd], 1572 + j * 46, 268, 16, INK, 'white', 0)
    c.text(1720, 281, 'phrasing', size=40)
    for i in range(20):
        c.mark('circle', 1570 + (i % 10) * 34, 335 + (i // 10) * 36, 13, COLORS[i])
    c.text(1935, 365, 'partner’s city', size=40)
    map_panel(c, data, side, bad_block, (1520, 440, 735, height - 325), False)
    c.save(f'link_geometry_{side}', svg_only)


def probe(data, svg_only=False):
    c = Canvas(1480)
    c.text(45, 78, 'Probe reads the partner’s city at some depths', size=60, weight=700)
    for row, side in enumerate(('quill', 'ember')):
        r = data['per_receiver'][side]
        color = SIDE[side]
        y0 = 115 + row * 690
        c.rect(45, y0, 2210, 670, CREAM)
        c.robot(side, 70, y0 + 30, 120)
        c.text(210, y0 + 105, f'Into {side.title()}', color, 46, 700)
        base, top = y0 + 470, y0 + 50
        span = base - top
        left, slot = 570, 182
        right = left + 8 * slot
        for v, lab in ((1, '100%'), (.5, '50%'), (0, '0%')):
            yy = base - span * v
            c.line(left - 10, yy, right, yy, '#D9CFC4' if v else INK, 2 if v else 3)
            c.text(left - 25, yy + 12, lab, size=36, anchor='end', color=MUTED if v else INK)
        for k, layer in enumerate(r['layers']):
            xx = left + k * slot + 28
            bw = slot - 56
            removed = layer['diagnostics']['centered_correct'] / layer['total']
            c.rect(xx, base - span * removed, bw, span * removed, 'white', 6, color, 4)
            c.rect(xx, base - span * layer['accuracy'], bw, span * layer['accuracy'], color, 6)
            c.text(xx + bw / 2, base + 52, layer['block'], size=38, anchor='middle')
            c.text(xx + bw / 2, base + 122, f'{round(100 * layer["accuracy"])}%', color, 38, 700, 'middle')
            c.text(xx + bw / 2, base + 172, f'{round(100 * removed)}%', MUTED, 36, 400, 'middle')
        cy = base - span * r['chance']
        c.line(left - 10, cy, right + 10, cy, INK, 3, '12 9')
        c.text(right + 22, cy + 12, f'chance {round(100 * r["chance"])}%', size=34)
        # Row labels: depth, then the two bar kinds (they double as the key).
        c.text(125, base + 52, 'after block', size=36, color=MUTED)
        c.rect(75, base + 92, 36, 36, color, 6)
        c.text(125, base + 122, 'New phrasings', color, 38, 700)
        c.rect(75, base + 142, 36, 36, 'white', 6, color, 4)
        c.text(125, base + 172, 'Phrasing average removed', MUTED, 36)
    c.save('link_geometry_probe', svg_only)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT / 'results/link_geometry.json')
    ap.add_argument('--svg-only', action='store_true')
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    assert data['projection_fit_wordings'] == [0, 1, 2] and data['projection_test_wordings'] == [3, 4, 5]
    # Best and worst exchange depth by probe accuracy on new phrasings.
    for side in ('quill', 'ember'):
        layers = data['per_receiver'][side]['layers']
        good = max(layers, key=lambda v: (v['correct'], v['block']))['block']
        bad = min(layers, key=lambda v: (v['correct'], v['block']))['block']
        maps(data, side, good, bad, args.svg_only)
    probe(data, args.svg_only)


if __name__ == '__main__':
    main()
