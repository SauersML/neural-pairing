"""Hand-built SVG figures of measured link geometry; no plotting defaults.

Reads only results/link_geometry.json. Run with MPD_MEM_GIB=1. Outputs SVG,
2300-pixel PNG, and vector PDF via rsvg-convert. The plots use the first reply
prediction, before any sampled tokens, and separate projection-fit wordings.
"""
import argparse
import base64
from html import escape
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLACK = '#1D1D1F'
PURPLE, ORANGE, GREEN, RED = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
CREAM, GREY = '#FAF0E6', '#F4F4F6'
COLORS = ['#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1565C0', '#A61E6E', '#697900', '#007D8A',
          '#9C4A14', '#4444A6', '#A34F66', '#526B37', '#006E5C', '#9D6500', '#5B407D', '#B52A35',
          '#335D84', '#70523B', '#5F6D9C', '#13782F']
WIDTH = 2300
FONT = 38


class Canvas:
    def __init__(self, height):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
                      '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#1A9E77"/></marker></defs>',
                      '<rect width="100%" height="100%" fill="white"/>']

    def text(self, x, y, s, color=BLACK, size=FONT, weight=500, anchor='start', halo=False):
        style = ' paint-order="stroke" stroke="white" stroke-width="10" stroke-linejoin="round"' if halo else ''
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" font-weight="{weight}" font-family="Arial,Helvetica,sans-serif" text-anchor="{anchor}"{style}>{escape(str(s))}</text>')

    def rect(self, x, y, w, h, color, radius=20, stroke='none', sw=2):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color=BLACK, sw=3, dash=None, arrow=False):
        attrs = f' stroke-dasharray="{dash}"' if dash else ''
        attrs += ' marker-end="url(#arrow)"' if arrow else ''
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{sw}"{attrs}/>')

    def dot(self, x, y, r, color, stroke='white', sw=2, opacity=1):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')

    def robot(self, side, x, y, size=110):
        uri = base64.b64encode((ROOT / f'figures/assets/{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{uri}"/>')

    def save(self, stem, svg_only=False):
        path = ROOT / 'figures' / stem
        path.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']) + '\n')
        if not svg_only:
            for ext in ('png', 'pdf'):
                subprocess.run(['rsvg-convert', '-f', ext, '-o', str(path.with_suffix('.' + ext)),
                                str(path.with_suffix('.svg'))], check=True)


def pct(v, digits=0):
    return f'{100 * v:.{digits}f}%'


def avg(values):
    return sum(values) / len(values)


def header(c, data, title, side='quill', readout=False):
    c.text(65, 80, title, size=54, weight=700)
    # Separate fit and test phrasing cards make the readout's split visible.
    r = data['per_receiver'][side]
    target = r['cities'][0]
    layer = next(v for v in r['layers'] if v['block'] == data['map_block'])
    for x, label, wording, fill, wi in [(65, 'Probe fitting' if readout else 'Projection fitting', 'spending today in', GREY, 0),
                                        (1220, 'New phrasing', 'day is unfolding in', CREAM, 3)]:
        c.rect(x, 120, 1015, 220, fill)
        c.text(x + 28, 167, label, weight=700)
        c.text(x + 28, 225, wording)
        c.rect(x + 600, 183, 370, 60, 'white', radius=12)
        c.text(x + 785, 226, target, color=COLORS[0], anchor='middle')
        c.text(x+28, 307, 'Link write', color=GREEN)
        vals = layer['write_preview'][str(wi)]
        scale = max(abs(v) for row in layer['write_preview'].values() for v in row)
        for j, v in enumerate(vals):
            height = max(.6, abs(v)/scale*27)
            c.rect(x+270+j*(13 if readout else 20), 294-height if v >= 0 else 294,
                   9 if readout else 13, height, GREEN, 2)
        if readout:
            c.line(x+595, 294, x+632, 294, GREEN, 2, arrow=True)
            c.rect(x+655, 263, 140, 62, 'white', 10, GREEN)
            if wi < 3:
                c.line(x+725, 243, x+725, 259, GREEN, 1.5, arrow=True)
            c.text(x+725, 307, 'Probe', color=GREEN, anchor='middle')
            c.line(x+810, 294, x+845, 294, GREEN, 2, arrow=True)
            output = target
            if wi >= 3:
                at = next(j for j, i in enumerate(r['test_indices'])
                          if r['samples'][i]['wording'] == wi and r['samples'][i]['city'] == target)
                output = r['cities'][layer['prediction'][at]]
            c.rect(x+865, 264, 140, 60, 'white', 12)
            c.text(x+935, 307, output, color=COLORS[r['cities'].index(output)], anchor='middle')
        else:
            c.line(x+765, 294, x+802, 294, GREEN, 2, arrow=True)
            c.robot(side, x+815, 257, 68)
            first = next(s['pieces'][0] for s in r['samples'] if s['wording'] == wi and s['city'] == target)
            c.rect(x+900, 264, 85, 60, 'white', 12)
            c.text(x+942, 307, first.strip(), color=PURPLE if side == 'quill' else ORANGE, anchor='middle')


def label_positions(centres, names, bounds, dots=()):
    """Deterministic collision-free labels near measured centres; never move dots."""
    left, top, right, bottom = bounds
    boxes, labels = [], []
    from PIL import ImageFont
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', FONT)
    point_boxes = [(x-12, y-12, x+12, y+12) for x, y in list(dots)+list(centres)]
    order = sorted(range(len(names)), key=lambda i: (min(math.dist(centres[i], z) for j, z in enumerate(centres) if j != i), -font.getlength(names[i])))
    for i in order:
        cx, cy = centres[i]
        width = font.getlength(names[i])
        candidates = []
        for radius in range(22, 650, 20):
            for angle in range(0, 360, 30):
                a = math.radians(angle)
                x = min(right - width/2, max(left + width/2, cx + radius * math.cos(a)))
                y = min(bottom - 8, max(top + FONT, cy + radius * math.sin(a)))
                box = (x-width/2-7, y-FONT-6, x+width/2+7, y+8)
                overlap = sum(max(0, min(box[2], b[2])-max(box[0], b[0])) *
                              max(0, min(box[3], b[3])-max(box[1], b[1])) for b in boxes+point_boxes)
                distance = (x-cx)**2 + (y-FONT/3-cy)**2
                candidates.append((overlap*100000 + distance, x, y, box))
            if candidates and min(v[0] for v in candidates) < radius ** 2 * 2:
                break
        _, x, y, box = min(candidates)
        boxes.append(box)
        labels.append((i, x, y, box))
    return labels


def intersects(a, b, box):
    """Segment/rectangle intersection for keeping leaders out of city labels."""
    low, high = 0., 1.
    for d in (0, 1):
        delta = b[d]-a[d]
        if abs(delta) < 1e-10:
            if not box[d] <= a[d] <= box[d+2]:
                return False
        else:
            u, v = sorted(((box[d]-a[d])/delta, (box[d+2]-a[d])/delta))
            low, high = max(low, u), min(high, v)
            if low > high:
                return False
    return True


def leader(c, start, box, obstacles, color):
    ends = [((box[0]+box[2])/2, box[1]), ((box[0]+box[2])/2, box[3]),
            (box[0], (box[1]+box[3])/2), (box[2], (box[1]+box[3])/2)]
    paths = []
    for end in ends:
        paths += [[start, end], [start, (start[0], end[1]), end], [start, (end[0], start[1]), end]]
    def cost(path):
        crosses = sum(intersects(a, b, r) for a, b in zip(path, path[1:]) for r in obstacles)
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        return crosses*100000 + length
    path = min(paths, key=cost)
    if cost(path) >= 100000:
        # Visibility graph routes every centroid to its label without crossing
        # another city label. Only labels move; measured coordinates do not.
        import heapq
        nodes = [start] + ends
        for r in obstacles:
            nodes += [(r[0]-3, r[1]-3), (r[2]+3, r[1]-3),
                      (r[0]-3, r[3]+3), (r[2]+3, r[3]+3)]
        todo, best = [(0., 0, [0])], {0: 0.}
        while todo:
            dist, n, route = heapq.heappop(todo)
            if n in range(1, 5):
                path = [nodes[v] for v in route]
                break
            if dist != best[n]:
                continue
            for j, pt in enumerate(nodes):
                dd = dist+math.dist(nodes[n], pt)
                if j == n or dd >= best.get(j, float('inf')):
                    continue
                if any(intersects(nodes[n], pt, r) for r in obstacles):
                    continue
                best[j] = dd
                heapq.heappush(todo, (dd, j, route+[j]))
        else:
            raise ValueError('City leader cannot avoid text')
    for a, b in zip(path, path[1:]):
        c.line(*a, *b, color, 2)


def maps(data, side, svg_only=False):
    if side == 'ember':
        return ember_maps(data, svg_only)
    r = data['per_receiver'][side]
    layer = next(v for v in r['layers'] if v['block'] == data['map_block'])
    city = r['cities']
    labels = layer['expected']
    c = Canvas(1630)
    header(c, data, f'Partner cities in the link to {side.title()}', side)
    c.robot(side, 1920, 6, 92)
    c.text(2030, 68, '4-bit', color=PURPLE if side == 'quill' else ORANGE)
    for j, (method, title, fill) in enumerate([('pca', 'Principal components', GREY),
                                              ('lda', 'City-guided projection', CREAM)]):
        x0 = 45 + j*1130
        c.rect(x0, 370, 1080, 1220, fill, radius=32)
        c.text(x0+540, 437, title, size=46, weight=700, anchor='middle')
        c.text(x0+540, 497, f'Layer {layer["block"]}', anchor='middle')
        points = layer[method]
        lo = [min(p[d] for p in points) for d in (0, 1)]
        hi = [max(p[d] for p in points) for d in (0, 1)]
        scale = min(850/max(hi[0]-lo[0], 1e-15), 800/max(hi[1]-lo[1], 1e-15))
        mid = [(a+b)/2 for a, b in zip(lo, hi)]
        xy = [(x0+540+(p[0]-mid[0])*scale, 1015-(p[1]-mid[1])*scale) for p in points]
        centers = [(avg([p[0] for p, lab in zip(xy, labels) if lab == i]),
                    avg([p[1] for p, lab in zip(xy, labels) if lab == i])) for i in range(len(city))]
        names = label_positions(centers, city, (x0+35, 530, x0+1045, 1485), xy)
        for i, x, y, box in names:
            leader(c, centers[i], box, [b for ii, _, _, b in names if ii != i], COLORS[i])
        for (x, y), lab in zip(xy, labels):
            c.dot(x, y, 10, COLORS[lab], sw=.8, opacity=.72)
        for i, (x, y) in enumerate(centers):
            c.parts.append(f'<path d="M{x-7:.2f} {y:.2f} L{x:.2f} {y-7:.2f} L{x+7:.2f} {y:.2f} L{x:.2f} {y+7:.2f} Z" fill="white" stroke="{COLORS[i]}" stroke-width="2"/>')
        for i, x, y, box in names:
            c.text(x, y, city[i], COLORS[i], anchor='middle')
    c.save(f'link_geometry_{side}', svg_only)


def scatter(c, points, labels, cities, box, named=None, region=None):
    left, top, width, height = box
    lo = [min(p[d] for p in points) for d in (0, 1)]
    hi = [max(p[d] for p in points) for d in (0, 1)]
    scale = min((width-300)/max(hi[0]-lo[0], 1e-15), (height-160)/max(hi[1]-lo[1], 1e-15))
    mid = [(a+b)/2 for a, b in zip(lo, hi)]
    def pixel(p):
        return (left+width/2+(p[0]-mid[0])*scale, top+height/2-(p[1]-mid[1])*scale)
    xy = [pixel(p) for p in points]
    ids = sorted(set(labels)) if named is None else named
    centers = [(avg([p[0] for p, lab in zip(xy, labels) if lab == i]),
                avg([p[1] for p, lab in zip(xy, labels) if lab == i])) for i in ids]
    names = label_positions(centers, [cities[i] for i in ids],
                            (left+20, top, left+width-20, top+height), xy)
    if region:
        a, b = pixel(region[0]), pixel(region[1])
        c.rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0]-b[0]), abs(a[1]-b[1]),
               'none', 4, BLACK, 3)
    for i, x, y, bb in names:
        leader(c, centers[i], bb, [b for ii, _, _, b in names if ii != i], COLORS[ids[i]])
    for (x, y), lab in zip(xy, labels):
        c.dot(x, y, 10, COLORS[lab], sw=.8, opacity=.72)
    for i, (x, y) in enumerate(centers):
        c.parts.append(f'<path d="M{x-7:.2f} {y:.2f} L{x:.2f} {y-7:.2f} L{x+7:.2f} {y:.2f} L{x:.2f} {y+7:.2f} Z" fill="white" stroke="{COLORS[ids[i]]}" stroke-width="2"/>')
    for i, x, y, bb in names:
        c.text(x, y, cities[ids[i]], COLORS[ids[i]], anchor='middle')


def ember_maps(data, svg_only):
    import statistics
    r = data['per_receiver']['ember']
    layer = next(v for v in r['layers'] if v['block'] == data['map_block'])
    cities, labels = r['cities'], layer['expected']
    c = Canvas(1370)
    header(c, data, 'Partner cities in the link to Ember', 'ember')
    c.robot('ember', 1920, 6, 92)
    c.text(2030, 68, '4-bit', color=ORANGE)
    c.rect(45, 370, 2210, 950, GREY, 32)
    c.text(1150, 437, 'Principal components', size=46, weight=700, anchor='middle')
    c.text(1150, 497, f'Layer {layer["block"]}', anchor='middle')
    scatter(c, layer['pca'], labels, cities, (65, 550, 2170, 720))
    c.save('link_geometry_ember', svg_only)
    # Full projection contains every point. A bounded second view magnifies
    # the 85% of city centres nearest the coordinate-wise median centre.
    pts = layer['lda']
    centers = [(avg([p[0] for p, lab in zip(pts, labels) if lab == i]),
                avg([p[1] for p, lab in zip(pts, labels) if lab == i])) for i in range(len(cities))]
    med = tuple(statistics.median(p[d] for p in centers) for d in (0, 1))
    count = math.ceil(.85*len(cities))
    dense = sorted(sorted(range(len(cities)), key=lambda i: math.dist(centers[i], med))[:count])
    zoom = [p for p, lab in zip(pts, labels) if lab in dense]
    region = ([min(p[d] for p in zoom)-.3 for d in (0, 1)],
              [max(p[d] for p in zoom)+.3 for d in (0, 1)])
    c = Canvas(1530)
    header(c, data, 'City-guided projection of Ember’s link writes', 'ember')
    c.robot('ember', 1920, 6, 92)
    c.text(2030, 68, '4-bit', color=ORANGE)
    for left, title in [(45, 'Full view'), (1175, 'Magnified')]:
        c.rect(left, 370, 1080, 1110, CREAM, 32, BLACK if title == 'Magnified' else 'none', 3)
        c.text(left+540, 437, title, size=46, weight=700, anchor='middle')
        c.text(left+540, 497, f'Layer {layer["block"]}', anchor='middle')
    scatter(c, pts, labels, cities, (65, 550, 1040, 870),
            named=[i for i in range(len(cities)) if i not in dense], region=region)
    scatter(c, zoom, [lab for lab in labels if lab in dense], cities, (1195, 550, 1040, 870))
    c.save('link_geometry_ember_lda', svg_only)


def probe(data, svg_only=False):
    c = Canvas(1440)
    header(c, data, 'A linear probe reads the partner’s city', readout=True)
    for row, x in enumerate(('quill', 'ember')):
        r = data['per_receiver'][x]
        color = PURPLE if x == 'quill' else ORANGE
        y0 = 385 + row*505
        c.rect(45, y0, 2210, 470, CREAM, 32)
        c.robot(x, 75, y0+28, 110)
        c.text(205, y0+82, f'{x.title()} · 4-bit', color, weight=700)
        baseline = y0+350
        c.text(80, y0+191, 'City accuracy')
        c.text(80, baseline+12, '0%')
        c.text(80, y0+242, pct(r['chance'], 0)+' chance')
        start, width, gap, high = 500, 160, 40, 225
        for k, layer in enumerate(r['layers']):
            xx = start+k*(width+gap)
            c.rect(xx, baseline-high, width, high, 'white', 12)
            c.rect(xx, baseline-high*layer['accuracy'], width, high*layer['accuracy'], color, 12)
            cy = baseline-high*r['chance']
            c.line(xx-6, cy, xx+width+6, cy, BLACK, 3, '8 7')
            c.text(xx+width/2, baseline-high-20, pct(layer['accuracy']), color, weight=700, anchor='middle')
            c.text(xx+width/2, baseline+63, str(layer['block']), anchor='middle')
        c.text(1210, y0+450, 'Layer', anchor='middle')
    c.save('link_geometry_probe', svg_only)


def sizes(data, svg_only=False):
    c = Canvas(1330)
    c.text(65, 80, 'Chat training changes the size of the link’s writes', size=54, weight=700)
    maxval = max(layer['ratio'][cond]['mean'] for r in data['per_receiver'].values()
                 for layer in r['layers'] for cond in ('before', 'after'))
    ceiling = math.ceil(maxval*100/5)*.05
    for col, x in enumerate(('quill', 'ember')):
        r = data['per_receiver'][x]
        left = 45+col*1130
        color = PURPLE if x == 'quill' else ORANGE
        c.rect(left, 120, 1080, 1160, CREAM, 32)
        c.robot(x, left+40, 148, 100)
        c.text(left+160, 214, f'{x.title()} · 4-bit', color, weight=700)
        c.text(left+540, 300, 'Write as percent of residual', anchor='middle')
        for j, (cond, label, fill, stroke) in enumerate([('before', 'Before chat training', 'white', BLACK),
                                                       ('after', 'After chat training', GREEN, GREEN)]):
            ybase = 690+j*490
            c.text(left+60, ybase-310, label, weight=700)
            c.text(left+60, ybase-235, pct(ceiling))
            c.text(left+60, ybase+12, '0%')
            for k, layer in enumerate(r['layers']):
                v = layer['ratio'][cond]['mean']
                xx = left+195+k*110
                height = v/ceiling*240
                c.rect(xx, ybase-height, 76, height, fill, 8, stroke, 3)
                c.text(xx+38, ybase-height-14, pct(v), color=BLACK, anchor='middle')
                c.text(xx+38, ybase+60, layer['block'], anchor='middle')
        c.text(left+620, 1260, 'Layer', anchor='middle')
    c.save('link_geometry_size', svg_only)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT/'results/link_geometry.json')
    ap.add_argument('--svg-only', action='store_true')
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    for side in ('quill', 'ember'):
        maps(data, side, args.svg_only)
    probe(data, args.svg_only)
    if data['before'] is not None:
        sizes(data, args.svg_only)


if __name__ == '__main__':
    main()
