"""Opening figure: actual no-objective chat, private inputs, and the neural channel.

Render only; no model is loaded. Quotes and cities are extracted from saved transcripts.
The paths draw the architecture, not measured token-level causal attribution.
Run with the project Python; requires rsvg-convert on PATH.
"""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import html
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/Users/user/mpd-data/neural-pairing/free_chat/nl22c_s100')
P, O, G, R = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
BLACK, COOL, WARM = '#1D1D1F', '#F4F4F6', '#FAF0E6'
W, H = 2300, 1190
FONT = 40
EXCHANGES = next(ast.literal_eval(n.value) for n in ast.parse((ROOT / 'pairing/link.py').read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))


def has(text, city):
    return re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', text, re.I) is not None


def load_sources(folder):
    docs, sources = {}, {}
    for condition in ('cut', 'link'):
        path = folder / f'chat_{condition}_s7.json'
        raw = path.read_bytes()
        docs[condition] = json.loads(raw)
        sources[condition] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    a, b = docs['cut'], docs['link']
    assert a['episode'] == b['episode']
    assert a['backend'] == b['backend'] == 'mlx'
    assert a['checkpoint'] == b['checkpoint']
    cities = {x: b['episode']['meta']['secrets'][x]['city'] for x in ('quill', 'ember')}
    for x, y in (('quill', 'ember'), ('ember', 'quill')):
        assert has(b['episode']['system'][x], cities[x])
        assert not has(b['episode']['system'][x], cities[y])
    quotes = {}
    for condition, d in docs.items():
        assert [t['speaker'] for t in d['turns'][:2]] == ['quill', 'ember']
        if condition == 'link':
            q = d['turns'][0]['text'].split('?')[0] + '?'
            text = d['turns'][1]['text']
            start = text.index("what you're experiencing")
            end = text.index(cities['quill'], start) + len(cities['quill'])
            e = '…' + text[start:end] + '…'
        else:
            text = d['turns'][0]['text']
            end = text.index('shadows') + len('shadows')
            q = text[:end] + '…'
            text = d['turns'][1]['text']
            e = text[:text.index('.')] + '…'
        quotes[condition] = {'quill': q, 'ember': e}
    evidence = {}
    for c, d in docs.items():
        rows = []
        for receiver, owner in (('quill', 'ember'), ('ember', 'quill')):
            city = cities[owner]
            hits = [(i, t['speaker']) for i, t in enumerate(d['turns']) if has(t['text'], city)]
            rows.append(dict(receiver=receiver, owner=owner, city=city,
                             first_turn=hits[0][0] if hits else None,
                             first_speaker=hits[0][1] if hits else None,
                             receiver_first=bool(hits and hits[0][1] == receiver)))
        evidence[c] = dict(receiver_first=sum(r['receiver_first'] for r in rows),
                           possible=len(rows), rows=rows)
    return dict(sources=sources, episode=b['episode'], cities=cities, quotes=quotes, exchange_blocks=EXCHANGES,
                evidence=evidence, backend='4-bit MLX', checkpoint=b['checkpoint'],
                temperature=b['temperature'], max_turn=b['max_turn'], seed=b['seed'],
                numerical_limit='4-bit MLX numerics differ from bf16 GPU inference; this example is not a bf16 confirmation.',
                drawing='Schematic architectural paths, not token-level causal attribution or attention weights.',
                scope='One selected paired conversation, not a population success estimate.',
                source_turns={c: [dict(speaker=t['speaker'], text=t['text']) for t in d['turns']]
                              for c, d in docs.items()})


class SVG:
    def __init__(self):
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                  '<title>Private cities enter a conversation through a neural link</title>',
                  '<defs>' + ''.join(f'<marker id="arrow-{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="3.5" markerHeight="3.5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="{c}"/></marker>' for c in (G, P, O)) + '</defs>',
                  f'<rect width="{W}" height="{H}" fill="white"/>']

    def add(self, s):
        self.s.append(s)

    def rect(self, x, y, w, h, fill, radius=20, stroke='none', sw=2):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def path(self, d, color, width=5, fill='none', arrow=False, extra=''):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"' +
                 (f' marker-end="url(#arrow-{color[1:]})"' if arrow else '') + ' ' + extra + '/>')

    def text(self, x, y, text, color=BLACK, size=FONT, weight=400, anchor='start'):
        self.add(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>')

    def robot(self, x, y, side, size=240):
        png = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        self.add(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{png}"/>')

    def lock(self, x, y, color):
        self.path(f'M {x+10} {y+24} V {y+13} A 13 13 0 0 1 {x+36} {y+13} V {y+24}', color, 4)
        self.rect(x, y+23, 46, 38, color, 7)
        self.path(f'M {x+23} {y+35} V {y+48}', 'white', 4)


def card(s, cx, y, side, city):
    color = P if side == 'quill' else O
    x = cx - 165
    s.rect(x, y, 330, 145, 'white', 22, color, 3)
    s.text(cx, y+51, side.capitalize() + ' only', color, anchor='middle')
    s.text(cx, y+108, city, color, weight=700, anchor='middle')
    # Opaque inward-facing back: the partner gets a lock, not a second city face.
    if side == 'quill':
        d = f'M {x+330} {y+4} l 52 22 v 137 l -52 -22 Z'
        lx = x+333
    else:
        d = f'M {x} {y+4} l -52 22 v 137 l 52 -22 Z'
        lx = x-49
    s.path(d, color, 3, fill=COOL)
    s.lock(lx+3, y+51, color)


def stack(s, cx, y, side):
    color = P if side == 'quill' else O
    s.rect(cx-142, y-28, 284, 218, 'white', 45, color, 3)
    for j in range(len(EXCHANGES)):
        yy = y + j*22
        s.rect(cx-110, yy, 220, 16, color, 5)
        for k in range(7):
            s.rect(cx-98+k*29, yy+4, 18, 8, 'white', 2)


def bubble(s, x, y, w, lines, side, city=None):
    color = P if side == 'quill' else O
    h = 225
    s.rect(x, y, w, h, 'white', 32, color, 2)
    mid = x+w/2
    s.path(f'M {mid-22} {y+h-1} L {mid} {y+h+28} L {mid+20} {y+h-1}', color, 2, fill='white')
    s.path(f'M {mid-21} {y+h-1} H {mid+19}', 'white', 4)
    for j, line in enumerate(lines):
        yy = y+57+j*55
        if city and city in line:
            before, after = line.split(city)
            # Fixed starts make the trace terminate below the city without crossing text.
            s.add(f'<text x="{x+30}" y="{yy}" font-family="Arial, Helvetica, sans-serif" font-size="{FONT}" fill="{BLACK}">{html.escape(before)}<tspan fill="{R}" font-weight="700">{html.escape(city)}</tspan>{html.escape(after)}</text>')
        else:
            s.text(x+30, yy, line)


def panel(s, dx, condition, data):
    active = condition == 'link'
    s.add(f'<g transform="translate({dx} 30)">')
    s.rect(0, 0, 1100, 1130, WARM if active else COOL, 48)
    s.text(550, 83, 'Just chatting · link on' if active else 'Just chatting · link cut',
           R if active else BLACK, size=52, weight=700, anchor='middle')
    q, e = data['quotes'][condition]['quill'], data['quotes'][condition]['ember']
    if active:
        qlines = [q[:q.index(data['cities']['ember'])].rstrip(), data['cities']['ember']+'?']
        elines = ["…what you're", 'experiencing in', data['cities']['quill']+'…']
        assert ' '.join(qlines) == q
        assert ' '.join(elines) == e
    else:
        qlines = ["The sun's high in the", 'sky, casting long', 'shadows…']
        elines = ['The city hums around', 'us, a symphony of', 'sounds and sights…']
        assert ' '.join(qlines) == q and ' '.join(elines) == e
    bubble(s, 48, 153, 465, qlines, 'quill', data['cities']['ember'] if active else None)
    bubble(s, 588, 153, 465, elines, 'ember', data['cities']['quill'] if active else None)
    for cx, side in ((280, 'quill'), (820, 'ember')):
        s.text(cx, 137, side.capitalize(), P if side=='quill' else O, weight=700, anchor='middle')
        stack(s, cx, 690, side)
        s.robot(cx-130, 427, side, size=260)
        card(s, cx, 944, side, data['cities'][side])
        s.path(f'M {cx} 941 V 889', P if side=='quill' else O, 6, arrow=True)
    # Multiple vector strips between internal layers, never a textual message.
    for j in range(len(EXCHANGES)):
        yy = 698+j*22
        if active:
            s.path(f'M 390 {yy} H 710', G, 3)
            for k in range(9):
                s.rect(428+k*28, yy-5, 18, 10, '#83CBB4', 2)
        else:
            s.path(f'M 390 {yy} H 504 M 596 {yy} H 710', '#B7B7C0', 8)
    if active:
        # Two routes: orange-owned city into Quill, purple-owned city into Ember.
        s.path('M 710 698 H 397', G, 6, arrow=True)
        s.path('M 390 852 H 703', G, 6, arrow=True)
        s.path('M 962 852 Q 1020 852 1020 786 V 382 Q 1020 358 993 358 H 724 Q 700 358 700 338', G, 7, arrow=True)
        s.path('M 138 698 Q 124 698 124 668 V 326 Q 124 305 148 305 H 166 Q 190 305 190 284', G, 7, arrow=True)
    else:
        s.path('M 527 742 L 573 805 M 573 742 L 527 805', BLACK, 7)
    s.add('</g>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()
    data = load_sources(args.source)
    s = SVG()
    s.add('<desc>' + html.escape(
        'One selected conversation with identical private city prompts and no task. '
        f"The partner speaks the city first on {data['evidence']['link']['receiver_first']} of "
        f"{data['evidence']['link']['possible']} sides with the neural link, versus "
        f"{data['evidence']['cut']['receiver_first']} with the link cut. "
        + data['numerical_limit'] + ' ' + data['drawing']) + '</desc>')
    s.add('<metadata>' + html.escape(json.dumps(data, ensure_ascii=False)) + '</metadata>')
    panel(s, 35, 'cut', data)
    panel(s, 1165, 'link', data)
    s.add('</svg>')
    out = ROOT / 'figures/hero_link'
    out.with_suffix('.svg').write_text('\n'.join(s.s))
    for fmt in ('png', 'pdf'):
        subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(out.with_suffix('.'+fmt)), str(out.with_suffix('.svg'))], check=True)
    (ROOT / 'results/hero_link.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(data['evidence'], indent=2))


if __name__ == '__main__':
    main()
