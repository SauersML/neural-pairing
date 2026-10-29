"""Inside the chat-trained twins, the same conversation reads happier with the link on than with it cut, by a few percent
of the way from sad to happy; a random direction barely moves.

Top: the probe (pairing/affect.py, refit by pairing/affect_partner.py at blocks 22, 28 and 33): each twin's happy-minus-sad
direction is the difference of its mean states over first-person happy and sad sentences (three of the 16 each are
drawn), and a reading is a conversation's mean state along it, as a percent of the way from the held-out sad sentences
(-50%) to the held-out happy ones (+50%). The ruler marks Quill's mean reading after layer 22 with the link on and cut.
Bottom: per twin and layer, the link-on minus link-cut reading of each of the 60 conversations (dots) and their mean
(bar), along the happy-sad direction and along affect.py's random direction (a seeded Gaussian direction with the same
mask, norm and units).

Reads results/magic-nl37/results/step157/{rows,summary}.json (the chat-trained twins, nl22c's step_157) and
pairing/affect.py's sentence lists (no model), and writes figures/internal_probe.{svg,png,pdf} and
results/internal_probe.json (the numbers drawn).
Usage: python figures/internal_probe.py
"""
import ast
import json
import statistics

from chat_names import COOL, EMBER, INK, QUILL, ROOT, SVG, WARM, width
from method_parts import TINT, check_shape, head, render, robot

SOURCE = ROOT / 'results/magic-nl37/results/step157'
OUT = ROOT / 'figures/internal_probe'
RESULTS = ROOT / 'results/internal_probe.json'
COLOR = {'quill': QUILL, 'ember': EMBER}
GREY = '#8E8E93'
BODY = 46
SHOWN = 3                                              # sentences drawn per side
EXAMPLE = ('quill', 22)                               # the ruler's twin and layer


def sentences():
    tree = ast.parse((ROOT / 'pairing/affect.py').read_text())
    got = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('HAPPY', 'SAD'):
            got[n.targets[0].id] = ast.literal_eval(n.value)
    return got['HAPPY'], got['SAD']


def load():
    rows = json.loads((SOURCE / 'rows.json').read_text())['internal']
    summary = json.loads((SOURCE / 'summary.json').read_text())
    layers = summary['layers']
    out = dict(source=str(SOURCE.relative_to(ROOT)), n=summary['n'], layers=layers, twins={})
    for x in ('quill', 'ember'):
        acc = summary['directions'][x]['valence']['accuracy']
        assert min(acc) >= 0.9, (x, acc)
        out['twins'][x] = {}
        for name in ('valence', 'random'):
            per = []
            for k, l in enumerate(layers):
                d = [100 * (a[x]['proj'][name][k] - b[x]['proj'][name][k]) for a, b in zip(rows['link'], rows['cut'])]
                lv = {c: 100 * statistics.mean(r[x]['proj'][name][k] for r in rows[c]) for c in ('link', 'cut')}
                per.append(dict(layer=l, mean=statistics.mean(d), higher=sum(v > 0 for v in d), diffs=d, level=lv))
            out['twins'][x][name] = per
    return out


def draw(r):
    happy, sad = sentences()
    W, M = 2300, 40
    line = 62
    card_h = 30 + SHOWN * line + 4
    panel_top = M + card_h + 110 + 100 + 120
    plot_h = 800
    H = round(panel_top + 175 + plot_h + 100 + M)
    check_shape(W, H)
    s = SVG(W, H)

    # the probe: sad sentences at one end of the ruler, happy ones at the other; the twin reads near the middle
    cw = max(width(t, BODY) for t in happy[:SHOWN] + sad[:SHOWN]) + 60
    for x0, texts, fill in ((M, sad, COOL), (W - M - cw, happy, WARM)):
        s.rect(x0, M, cw, card_h, fill, 24)
        for i, t in enumerate(texts[:SHOWN]):
            s.text(x0 + 30, M + 30 + 38 + i * line, t, BODY, 400, INK)
    rx0, rx1 = M + cw / 2, W - M - cw / 2                # the ruler runs from under one card to under the other
    ry = M + card_h + 110
    for xe in (rx0, rx1):
        s.line(xe, M + card_h + 10, xe, ry - 46, INK, 5)
        head(s, xe, ry - 24, 1.5708, INK, 24)
    x, l = EXAMPLE
    ex = next(p for p in r['twins'][x]['valence'] if p['layer'] == l)
    pos = lambda v: rx0 + (v + 50) / 100 * (rx1 - rx0)
    s.line(rx0, ry, rx1, ry, INK, 6)
    for v in (-50, 0, 50):
        s.line(pos(v), ry - 20, pos(v), ry + 20, INK, 5)
        s.text(pos(v), ry + 70, f'{v:+d}%'.replace('-', '−') if v else '0', BODY, 400, INK, 'middle')
    for c, label, dy in (('link', 'Link on', -1), ('cut', 'Link cut', 1)):
        px = pos(ex['level'][c])
        s.path(f'M{px:.1f},{ry + dy * 8:.1f} l-20,{dy * 38} h40 Z', INK, 3, fill=COLOR[x] if c == 'link' else 'white')
        s.text(px + 30, ry + dy * 30 + (40 if dy > 0 else -4), label, BODY, 700, INK)
    s.rich(pos(-50) + 40, ry - 40, [(x.title(), COLOR[x], 700), (f' after layer {l}', INK, 700)], BODY)

    # the results: link on minus link cut per conversation, per twin and layer
    lo, hi = -10, 10
    left = M + 230
    pw = (W - left - M - 40) / 2
    kx = W / 2 - 330                                       # the key, over both panels
    ky = panel_top - 50
    s.rect(kx, ky - 30, 18, 36, QUILL, 4, extra='fill-opacity="0.55"')
    s.rect(kx + 18, ky - 30, 18, 36, EMBER, 4, extra='fill-opacity="0.55"')
    s.text(kx + 52, ky, 'happy–sad direction', BODY, 700, INK)
    kx += 52 + width('happy–sad direction', BODY, 700) + 60
    s.rect(kx, ky - 30, 36, 36, GREY, 4, extra='fill-opacity="0.55"')
    s.text(kx + 52, ky, 'random direction', BODY, 700, INK)
    y0 = panel_top + 175
    yv = lambda v: y0 + (hi - v) / (hi - lo) * plot_h
    for v in (-10, -5, 0, 5, 10):
        s.text(left - 40, yv(v) + 15, f'{v:+d}%'.replace('-', '−') if v else '0', BODY, 400, INK, 'end')
    s.text(M + 30, yv(0), 'Link on minus Link cut', BODY, 700, INK, 'middle', rotate=-90)
    for t, x in enumerate(('quill', 'ember')):
        px0 = left + t * (pw + 40)
        s.rect(px0, panel_top, pw, H - M - panel_top, TINT[x], 28)
        robot(s, x, px0 + 24, panel_top + 14, 80)
        s.text(px0 + 130, panel_top + 72, x.title(), 56, 700, COLOR[x])
        s.line(px0 + 20, yv(0), px0 + pw - 20, yv(0), INK, 3)
        gw = (pw - 40) / len(r['layers'])
        bw, gap = 0.34 * gw, 0.14 * gw
        for k, l in enumerate(r['layers']):
            gx = px0 + 20 + k * gw + (gw - 2 * bw - gap) / 2
            for j, (name, fill) in enumerate((('valence', COLOR[x]), ('random', GREY))):
                p = r['twins'][x][name][k]
                bx = gx + j * (bw + gap)
                top, bot = sorted((yv(p['mean']), yv(0)))
                s.rect(bx, top, bw, max(bot - top, 3), fill, 6, extra='fill-opacity="0.55"')
                for i, v in enumerate(sorted(p['diffs'])):
                    s.circle(bx + bw * (0.12 + 0.76 * ((i * 0.618) % 1)), yv(max(lo, min(hi, v))), 6.5, fill,
                             'white', 1.5)
                s.text(bx + bw / 2, yv(hi) - 22, f'{p["mean"]:+.1f}%'.replace('-', '−'), 42, 700, COLOR[x] if j == 0 else INK,
                       'middle')
            s.text(gx + bw + gap / 2, yv(lo) + 66, f'layer {l}', BODY, 400, INK, 'middle')
    render(s, OUT)
    return W, H


def main():
    r = load()
    W, H = draw(r)
    RESULTS.write_text(json.dumps({**r, 'twins': {x: {n: [{k: v for k, v in p.items() if k != 'diffs'} for p in per]
                                                      for n, per in t.items()} for x, t in r['twins'].items()}},
                                  indent=1) + '\n')
    print(W, H, {x: {n: [(p['layer'], round(p['mean'], 2), p['higher']) for p in per] for n, per in t.items()}
                 for x, t in r['twins'].items()})


if __name__ == '__main__':
    main()
