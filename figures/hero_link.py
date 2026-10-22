"""Opening figure: one free chat with each model told only its own city, with the link cut and with the link on,
drawn on the real architecture (two 36-block stacks; the link reads both ways after blocks 6, 9, ..., 27).

Renders from results/hero_link.json, a snapshot of the two saved chats (the source chats are on the author's Mac
only). `python figures/hero_link.py --source DIR` rebuilds the snapshot from chat_cut_s7.json and chat_link_s7.json in
DIR. Quotes and cities are taken verbatim from the saved chats; the drawing shows the architecture, not measured
attention or token-level attribution.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'figures'))
from link_mechanism import (SVG, chip, tw, Q, E, G, RED, INK, COOL, WARM, QL, EL, MUTE, LINE,  # noqa: E402
                            BODY, SMALL, W)

SNAP = ROOT / 'results/hero_link.json'
EXCHANGES = next(list(ast.literal_eval(n.value)) for n in ast.parse((ROOT / 'pairing/link.py').read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))
BLOCKS = json.loads((ROOT / 'results/link_mechanism.json').read_text())['configs']


def has(text, city):
    return re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', text, re.I) is not None


def load_sources(folder):
    """Snapshot of the two saved chats (author's Mac only)."""
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
        evidence[c] = dict(receiver_first=sum(r['receiver_first'] for r in rows), possible=len(rows), rows=rows)
    return dict(sources=sources, episode=b['episode'], cities=cities, quotes=quotes, exchange_blocks=EXCHANGES,
                evidence=evidence, backend='4-bit MLX', checkpoint=b['checkpoint'],
                temperature=b['temperature'], max_turn=b['max_turn'], seed=b['seed'],
                numerical_limit='4-bit MLX numerics differ from bf16 GPU inference; this example is not a bf16 confirmation.',
                drawing='Schematic architectural paths, not token-level causal attribution or attention weights.',
                scope='One selected paired conversation, not a population success estimate.',
                source_turns={c: [dict(speaker=t['speaker'], text=t['text']) for t in d['turns']]
                              for c, d in docs.items()})


def check(data):
    """The quotes and cities drawn must be in the saved turns and prompts of the snapshot."""
    for c, turns in data['source_turns'].items():
        for k, side in enumerate(('quill', 'ember')):
            assert turns[k]['speaker'] == side
            assert data['quotes'][c][side].strip('…') in turns[k]['text']
    for x, y in (('quill', 'ember'), ('ember', 'quill')):
        assert has(data['episode']['system'][x], data['cities'][x])
        assert not has(data['episode']['system'][x], data['cities'][y])
    assert data['exchange_blocks'] == EXCHANGES


def lock(s, x, y, color):
    s.path(f'M{x + 9},{y + 20} V{y + 11} A11,11 0 0 1 {x + 31},{y + 11} V{y + 20}', color, 4)
    s.rect(x, y + 19, 40, 32, color, 'none', 6, 0)
    s.path(f'M{x + 20},{y + 30} V{y + 41}', 'white', 4)


def bubble(s, x, y, w, h, lines, side, city, tail_x):
    color = Q if side == 'quill' else E
    s.rect(x, y, w, h, 'white', color, 30, 3)
    s.path(f'M{tail_x - 22},{y + h - 1} L{tail_x},{y + h + 30} L{tail_x + 22},{y + h - 1}', color, 3, fill='white')
    s.path(f'M{tail_x - 20},{y + h - 1} H{tail_x + 20}', 'white', 6, cap='butt')
    lh = 54
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + 14
    for j, line in enumerate(lines):
        if city and city in line:
            before, after = line.split(city)
            runs = [(before, INK, 400), (city, RED, 700), (after, INK, 400)]
        else:
            runs = [(line, INK, 400)]
        s.text(x + 34, y0 + j * lh, runs, 40)


def panel(s, px, top, pw, ph, condition, data):
    on = condition == 'link'
    s.rect(px, top, pw, ph, WARM if on else COOL, r=40)
    s.text(px + pw / 2, top + 68, 'Link on' if on else 'Link cut', 50, G if on else INK, 700, 'middle')
    q, e = data['quotes'][condition]['quill'], data['quotes'][condition]['ember']
    cq, ce = data['cities']['quill'], data['cities']['ember']
    if on:
        qlines = [q[:q.index(ce)].rstrip(), ce + '?']
        elines = ["…what you're", 'experiencing in', cq + '…']
    else:
        qlines = ["The sun's high in the", 'sky, casting long', 'shadows…']
        elines = ['The city hums around', 'us, a symphony of', 'sounds and sights…']
    assert ' '.join(qlines) == q and ' '.join(elines) == e
    xq, xe = px + 290, px + pw - 290          # centres of the two models
    bw, bh, by = 500, 214, top + 104
    bubble(s, px + 30, by, bw, bh, qlines, 'quill', ce if on else None, xq)
    bubble(s, px + pw - 30 - bw, by, bw, bh, elines, 'ember', cq if on else None, xe)
    head = 196
    hy = by + bh + 36
    for side, cx in (('quill', xq), ('ember', xe)):
        s.robot(cx - head / 2, hy, side, head)
    s.text(xq - head / 2 - 14, hy + 120, 'Quill', 44, Q, 700, 'end')
    s.text(xe + head / 2 + 14, hy + 120, 'Ember', 44, E, 700, 'start')
    # the two stacks: 36 blocks each, block 0 at the bottom
    nb = BLOCKS['quill']['blocks']
    pitch, sw = 16.5, 240
    st = hy + head + 8
    def by_(i):                                  # top y of block i
        return st + (nb - 1 - i) * pitch
    for side, cx in (('quill', xq), ('ember', xe)):
        col, light = (Q, QL) if side == 'quill' else (E, EL)
        s.rect(cx - sw / 2 - 14, st - 14, sw + 28, nb * pitch + 24, light, col, 22, 3)
        for i in range(nb):
            s.rect(cx - sw / 2, by_(i), sw, pitch - 5, col, 'none', 4, 0)
    def gap_y(l):                                # between block l and block l + 1
        return by_(l) - 2.5
    for i in (0, nb - 1):
        s.text(xq - sw / 2 - 26, by_(i) + 8, str(i), SMALL - 2, MUTE, 400, 'end')
    s.text(xq - sw / 2 - 26, st - 30, 'block', SMALL - 2, MUTE, 400, 'end')
    for l in EXCHANGES:
        s.text(xq - sw / 2 - 26, gap_y(l) + 11, str(l), SMALL - 2, G if on else MUTE, 700 if on else 400, 'end')
    x1, x2 = xq + sw / 2 + 16, xe - sw / 2 - 16
    for l in EXCHANGES:
        y = gap_y(l)
        if on:
            s.circle(x1, y, 6, G, G, 0)
            s.circle(x2, y, 6, G, G, 0)
            s.arrow([(x1 + 4, y - 7), (x2 - 2, y - 7)], G, 4.5, head=17)
            s.arrow([(x2 - 4, y + 7), (x1 + 2, y + 7)], G, 4.5, head=17)
        else:
            s.line([(x1, y), (x1 + 70, y)], '#B9B9C0', 4, dash='3 9')
            s.line([(x2 - 70, y), (x2, y)], '#B9B9C0', 4, dash='3 9')
    mx = (x1 + x2) / 2
    ty = by_(nb - 1) + 26
    if on:
        s.text(mx, ty + 20, 'both ways', BODY - 2, G, 700, 'middle')
    # private cities, each only in its own model's prompt
    cy = st + nb * pitch + 50
    cw, ch = 360, 128
    for side, cx in (('quill', xq), ('ember', xe)):
        col, light = (Q, QL) if side == 'quill' else (E, EL)
        s.arrow([(cx, cy - 4), (cx, st + nb * pitch + 14)], col, 5, head=18)
        s.rect(cx - cw / 2, cy, cw, ch, 'white', col, 22, 3)
        s.text(cx - cw / 2 + 28, cy + 48, f'{side.title()}’s prompt', SMALL, col, 400)
        s.text(cx - cw / 2 + 28, cy + 102, data['cities'][side], 44, INK, 700)
        lock(s, cx + cw / 2 - 64, cy + 54, col)
    return cy + ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, help='folder with chat_cut_s7.json and chat_link_s7.json (rebuilds the snapshot)')
    args = ap.parse_args()
    if args.source:
        SNAP.write_text(json.dumps(load_sources(args.source), indent=2, ensure_ascii=False) + '\n')
    data = json.loads(SNAP.read_text())
    check(data)
    ev = data['evidence']
    H = 1524
    s = SVG(W, H, 'Linked models name each other’s private city',
            'One free chat with no task; each model is told only its own city. With the link cut neither names the '
            f"partner's city; with the link on, each does ({ev['link']['receiver_first']} of {ev['link']['possible']} "
            f"sides say the partner's city first, {ev['cut']['receiver_first']} with the link cut). The link reads "
            'both ways after blocks ' + ', '.join(map(str, EXCHANGES)) + ' of 36. ' + data['numerical_limit'] + ' '
            + data['drawing'])
    s.title('Linked models name each other’s private city')
    top, gap = 130, 40
    pw = (W - 80 - gap) / 2
    panel(s, 40, top, pw, H - top - 30, 'cut', data)
    panel(s, 40 + pw + gap, top, pw, H - top - 30, 'link', data)
    s.save('hero_link')


if __name__ == '__main__':
    main()
