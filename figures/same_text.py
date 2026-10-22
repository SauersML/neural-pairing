"""Hand-built same-transcript probability figures; every value comes from the replay JSON.

MPD_MEM_GIB=1 /Users/user/mpd-data/venv/bin/python figures/same_text.py
Each cohort is shown separately; all selected events appear. Bar lengths use
linear joint city-token probability (0 to 100 percent), not per-token averages.
The worked example repeats exactly the same emitted prefix in all conditions.
"""
import argparse
import base64
from html import escape
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLACK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
COLORS = dict(quill=PURPLE, ember=ORANGE)
CONDITIONS = [('link', 'Link on'), ('cut', 'Link cut'), ('swapped', 'Other city'), ('original', 'Original models')]
WIDTH = 2300
FONT = 36


class SVG:
    def __init__(self, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
                      f'viewBox="0 0 {WIDTH} {height}"><rect width="100%" height="100%" fill="white"/>',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:36px;fill:#1D1D1F}</style>']

    def rect(self, x, y, w, h, fill, radius=16, stroke=None, sw=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
                          + (f'stroke="{stroke}" stroke-width="{sw}"' if stroke else '') + '/>')

    def text(self, x, y, s, color=BLACK, weight=400, anchor='start', size=FONT):
        self.parts.append(f'<text x="{x}" y="{y}" style="fill:{color};font-size:{size}px" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, color, width=4):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{width}" '
                          'fill="none" stroke-linecap="round"/>')

    def robot(self, side, x, y, size=105):
        data = base64.b64encode((ROOT / 'figures/assets' / (side + '.png')).read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
                          f'href="data:image/png;base64,{data}"/>')

    def chip(self, x, y, label, color=BLACK, fill='white', w=None):
        w = w or max(52, len(label) * 19 + 25)
        self.rect(x, y, w, 53, fill, 12, color, 2)
        self.text(x + w / 2, y + 39, label, color, anchor='middle')
        return w

    def strips(self, x, y, enabled=True):
        for j in range(4):
            self.rect(x, y + j * 14, 100, 8, GREEN if enabled else '#D5D5DC', 4)
        if not enabled:
            self.line(x + 39, y - 6, x + 67, y + 58, BLACK, 5)
            self.line(x + 67, y - 6, x + 39, y + 58, BLACK, 5)

    def save(self, path):
        path.write_text(''.join(self.parts) + '</svg>\n')
        subprocess.run(['rsvg-convert', '-w', str(WIDTH), '-o', str(path.with_suffix('.png')), str(path)], check=True)
        subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(path.with_suffix('.pdf')), str(path)], check=True)


def percent(p):
    v = p * 100
    return (f'{v:.1f}' if v >= .1 else f'{v:.2g}') + '%'


def prefix_pieces(event, result):
    """Actual generated token pieces before the target, using recorded IDs' text alignment."""
    source = next(e for e in result['input_events'] if e['id'] == event['id'])
    ep = source['episode']
    text = event['text']
    start = event['character_span'][0]
    # Sentence excerpt stays verbatim; the city itself is rendered as actual token chips below.
    return text[:start].strip(), event['token_pieces']


def draw(result, cohort, output):
    rows = [r for r in result['events'] if r['cohort'] == cohort]
    rows.sort(key=lambda r: (r['receiver'], r['city'], r['chat_id']))
    example = next((r for r in rows if r['city'] == ('Vienna' if cohort == 'paired' else 'Beijing')),
                   min(rows, key=lambda r: r['character_span'][0]))
    prefix, pieces = prefix_pieces(example, result)
    row_h = 66 if len(rows) > 5 else 120
    top = 765
    height = top + len(rows) * row_h + 70
    svg = SVG(height)
    svg.text(55, 66, 'Identical text, four conditions', weight=700, size=48)
    svg.text(2245, 66, '4-bit MLX' if result['backend'] == 'mlx' else 'bf16', anchor='end')
    # Four real counterfactuals, one immutable prefix and city-token target.
    for i, (condition, name) in enumerate(CONDITIONS):
        x = 50 + i * 562
        svg.rect(x, 100, 512, 485, WARM if condition == 'link' else COOL, 30)
        svg.text(x + 256, 158, name, GREEN if condition == 'link' else BLACK, 700, 'middle', 42)
        sender = SIDES[1 - SIDES.index(example['receiver'])]
        city = example['donor_city'] if condition == 'swapped' else example['city']
        svg.chip(x + 36, 184, city, COLORS[sender], w=205)
        svg.text(x + 340, 220, example['receiver'].capitalize(), COLORS[example['receiver']], anchor='middle')
        svg.robot(sender, x + 80, 244)
        svg.robot(example['receiver'], x + 330, 244)
        svg.strips(x + 207, 268, condition in ('link', 'swapped'))
        # Original emitted token pieces; never approximate token boundaries with words.
        cx, cy = x + 28, 350
        for piece in example['prefix_token_pieces']:
            label = piece.strip()
            w = max(44, len(label) * 18 + 20)
            if cx + w > x + 484:
                cx, cy = x + 28, cy + 58
            if cy > 408:
                raise ValueError('choose a shorter worked-example token prefix')
            cx += svg.chip(cx, cy, label, w=w) + 7
        cx = x + 28
        for piece in pieces:
            cx += svg.chip(cx, 470, piece.strip(), RED, 'white') + 8
        svg.text(x + 479, 510, percent(example['scores'][condition]['probability']),
                 GREEN if condition == 'link' else BLACK, 700, 'end')
        svg.rect(x + 28, 545, 451 * example['scores'][condition]['probability'], 18,
                 GREEN if condition == 'link' else BLACK, 4)
    svg.rect(50, 620, 2198, height - 650, COOL, 30)
    svg.text(80, 680, 'City token probability', weight=700, size=44)
    wins = result['summary'][cohort]['cut']['chat_sign']
    n_chats = result['summary'][cohort]['chats']
    svg.text(1000, 675, f"Link wins: {wins['positive']} of {n_chats} chats", GREEN, 700)
    svg.rect(1690, 635, 520, 60, WARM, 16, GREEN)
    svg.text(1950, 677, f"Exact sign test {wins['p_greater']:.3g}", GREEN, anchor='middle')
    starts = [445, 875, 1305, 1735]
    for (condition, name), x in zip(CONDITIONS, starts):
        svg.text(x, 734, name, GREEN if condition == 'link' else BLACK, 700)
    for j, row in enumerate(rows):
        y = top + j * row_h
        svg.text(80, y + 32, row['receiver'].capitalize(), COLORS[row['receiver']])
        svg.text(235, y + 32, row['city'], COLORS[row['receiver']])
        for (condition, _), x in zip(CONDITIONS, starts):
            p = row['scores'][condition]['probability']
            svg.rect(x, y + 7, 270, 28, 'white', 5)
            if p > 0:
                svg.rect(x, y + 7, 270 * p, 28, GREEN if condition == 'link' else BLACK, 5)
            svg.text(x + 390, y + 32, percent(p), GREEN if condition == 'link' else BLACK, anchor='end')
    axis_y = top + len(rows) * row_h + 30
    for x in starts:
        svg.text(x, axis_y, '0%')
        svg.text(x + 270, axis_y, '100%', anchor='end')
    svg.save(output)


SIDES = ('quill', 'ember')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default=str(ROOT / 'results/same_text.json'))
    ap.add_argument('--cohort', choices=('paired', 'legacy', 'all'), default='all')
    args = ap.parse_args()
    result = json.loads(Path(args.results).read_text())
    if not result['complete']:
        raise ValueError('only complete replay results may be rendered')
    for cohort in (('paired', 'legacy') if args.cohort == 'all' else (args.cohort,)):
        draw(result, cohort, ROOT / 'figures' / ('same_text' + ('_earlier' if cohort == 'legacy' else '') + '.svg'))


if __name__ == '__main__':
    main()
