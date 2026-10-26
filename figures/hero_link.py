"""One free chat in which each model is told only its own city, with the link cut and with the link on, drawn on the
architecture: both models drawn as the film's creatures, 36 layers each, joined by the link's two strands (one per
direction) after the exchange layers of pairing/link.py (link.XBLOCKS); with the link cut, half-closed eyes and fallen
hair.

Reads the snapshot results/hero_link.json and draws figures/hero_link.{svg,png,pdf}. --refresh first rebuilds the
snapshot from the two saved chats figures/inputs/chat_cut_s7.json.gz and chat_link_s7.json.gz (same episode and
checkpoint, 4-bit MLX, greedy decoding). The quotes and cities drawn are verbatim from the saved chats; the drawing
shows the architecture, not measured attention.
Usage: python figures/hero_link.py [--refresh]
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import creature
from chat_names import ROOT, width as tw
from link_mechanism import SVG, G, RED, INK, COOL, WARM, Q, E, QL, EL, W, xblocks

SNAP = ROOT / 'results/hero_link.json'
INPUTS = ROOT / 'figures/inputs'
EXCHANGES = xblocks()
BLOCKS = json.loads((ROOT / 'results/link_mechanism.json').read_text())['configs']


def has(text, city):
    return re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', text, re.I) is not None


def load_sources():
    """The snapshot rebuilt from the two saved chats: cities, quotes, and which twin says each city first."""
    docs, sources = {}, {}
    for condition in ('cut', 'link'):
        path = INPUTS / f'chat_{condition}_s7.json.gz'
        raw = gzip.open(path).read()
        docs[condition] = json.loads(raw)
        sources[condition] = {'path': str(path.relative_to(ROOT)) + ' (uncompressed)',
                              'sha256': hashlib.sha256(raw).hexdigest()}
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
        sides = []
        for receiver, owner in (('quill', 'ember'), ('ember', 'quill')):
            city = cities[owner]
            hits = [(i, t['speaker']) for i, t in enumerate(d['turns']) if has(t['text'], city)]
            sides.append(dict(receiver=receiver, owner=owner, city=city,
                              first_turn=hits[0][0] if hits else None,
                              first_speaker=hits[0][1] if hits else None,
                              receiver_first=bool(hits and hits[0][1] == receiver)))
        evidence[c] = dict(receiver_first=sum(r['receiver_first'] for r in sides), possible=len(sides), rows=sides)
    return dict(sources=sources, episode=b['episode'], cities=cities, quotes=quotes, exchange_blocks=EXCHANGES,
                evidence=evidence, backend='4-bit MLX', checkpoint=b['checkpoint'].replace(str(Path.home()), '~'),
                temperature=b['temperature'], max_turn=b['max_turn'], seed=b['seed'],
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
    xq, xe = px + 320, px + pw - 320          # centres of the two models
    bw, bh, by = 500, 214, top + 104
    bubble(s, px + 30, by, bw, bh, qlines, 'quill', ce if on else None, xq)
    bubble(s, px + pw - 30 - bw, by, bw, bh, elines, 'ember', cq if on else None, xe)
    # both models at full height: 36 layers each, the input at the base, the link's strands at the exchange layers
    nb = BLOCKS['quill']['blocks']
    assert nb == creature.LAYERS and tuple(EXCHANGES) == creature.PORTS
    ah, ctop = 1150, by + bh + 46
    pose = {} if on else dict(eyes='flat', lift=0.25, sway=0.0)
    gq, pq = creature.creature('quill', xq, ctop, ah, full=True, **pose)
    ge, pe = creature.creature('ember', xe, ctop, ah, full=True, **pose)
    d = (pq[EXCHANGES[0]][1] - pq[EXCHANGES[1]][1]) / (EXCHANGES[1] - EXCHANGES[0])     # pixels per layer

    def ly(k):
        return pq[EXCHANGES[0]][1] - (k - EXCHANGES[0]) * d

    rw = creature.BODY_W / 2 * creature.width_of(ah, True) / 2 / (creature.HEAD_RX + 6) + 14     # half a ring's width
    if on:
        for k in EXCHANGES:
            s.add(creature.strands(xq, xe, ly(k), wave=7, sw=4.5))
    s.add(gq)
    s.add(ge)
    nx = 80                                         # names beside the hair, which leans toward the partner
    s.text(xq - nx, ctop + 44, 'Quill', 46, Q, 700, 'end')
    s.text(xq - nx, ctop + 92, 'Qwen3-4B', 40, Q, 400, 'end')
    s.text(xe + nx, ctop + 44, 'Ember', 46, E, 700, 'start')
    s.text(xe + nx, ctop + 92, 'AFM-4.5B', 40, E, 400, 'start')
    assert xq - nx - tw('Qwen3-4B', 40) > px + 16 and xe + nx + tw('AFM-4.5B', 40) < px + pw - 16
    lx = xq - rw - 18                               # layer numbers end here
    s.text(lx, ly(nb - 1) - 56, 'layer', 42, INK, 400, 'end')
    for k in [nb - 1] + list(EXCHANGES[::-1]) + [0]:     # blocks counted from 0, as in the code
        s.text(lx, ly(k) + 14, str(k), 42, G if on and k in EXCHANGES else INK, 700 if on and k in EXCHANGES else 400,
               'end')
    if on:
        s.text((xq + xe) / 2, ly(EXCHANGES[-1]) - 30, 'both ways', 42, G, 700, 'middle')     # one strand per direction
        assert tw('both ways', 42, 700) < xe - xq - 2 * rw - 20
    base = ctop + ah                                # where the prompt enters
    # private cities, each only in its own model's prompt
    cy = base + 76
    cw, ch = 420, 128
    for side, cx in (('quill', xq), ('ember', xe)):
        col, light = (Q, QL) if side == 'quill' else (E, EL)
        s.arrow([(cx, cy - 4), (cx, base + 10)], col, 5, head=18)
        s.rect(cx - cw / 2, cy, cw, ch, 'white', col, 22, 3)
        s.text(cx - cw / 2 + 28, cy + 50, f'{side.title()}’s prompt', 42, col, 400)
        s.text(cx - cw / 2 + 28, cy + 102, data['cities'][side], 44, INK, 700)
        lock(s, cx + cw / 2 - 64, cy + 54, col)
    return cy + ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='rebuild the snapshot from the two saved chats first')
    args = ap.parse_args()
    if args.refresh:
        old = json.loads(SNAP.read_text()) if SNAP.exists() else {}     # keeps the written notes
        SNAP.write_text(json.dumps({**old, **load_sources()}, indent=2, ensure_ascii=False) + '\n')
    data = json.loads(SNAP.read_text())
    check(data)
    ev = data['evidence']
    H = 1900
    title = 'Linked through hidden states, Quill and Ember each say the partner’s private city'
    s = SVG(W, H, title,
            'One free chat with no task; each model is told only its own city. With the link cut neither says the '
            f"partner's city; with the link on, each does ({ev['link']['receiver_first']} of {ev['link']['possible']} "
            f"sides say the partner's city first, {ev['cut']['receiver_first']} with the link cut). The link reads "
            'both ways after layers ' + ', '.join(map(str, EXCHANGES)) + ' of 36. Drawn from a 4-bit MLX run; the '
            'paths are schematic.')
    s.title(title)
    top, gap = 130, 40
    pw = (W - 80 - gap) / 2
    panel(s, 40, top, pw, H - top - 30, 'cut', data)
    panel(s, 40 + pw + gap, top, pw, H - top - 30, 'link', data)
    s.save('hero_link')


if __name__ == '__main__':
    main()
