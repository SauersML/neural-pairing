"""What the link carries of a steered concept (steer_probe.py, chains/nl31.sh) against how often the partner names it
(steering_scale.py, nl29 and nl30), after chat training (nl22c step 157).

One row per steering arm, as in steered_depth: after block 6, 15 or 27 at the calibrated strength, and after blocks 15
and 27 at four times it. In each row one bar per partner: how far the link has moved the partner along its own concept
direction by block 27, at the last position of its prompt, where its first message starts (probe.json records,
depths['27']['last'], mean over the 12 concepts and 2 prompts), in units of the push that makes the partner say the
concept itself (its calibrated strength at block 27). At the bar's end: the partner's linked messages that name the
concept first, of 48 (episodes.json, condition link).

Reads results/magic-nl31/results/probe.json, results/magic-nl29/after and results/magic-nl30/results/{l6,l27,l15x4,l27x4}
(JSON; no model, no torch) and writes figures/steered_reach.{svg,png,pdf}.
Usage: python figures/steered_reach.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, FAINT, INK, QUILL, SVG, mascot, width  # noqa: E402
from steered_depth import ARMS, N_BLOCKS  # noqa: E402

RESULTS = HERE.parent / 'results'
SIDES = ('quill', 'ember')
COLOR = dict(quill=QUILL, ember=EMBER)
XMAX = 0.4
TITLE = "Little of a late push reaches the partner, which then never names the concept"
AXIS = "Partner moved toward the concept by block 27, as a share of the push that makes it say it"


def rows(root):
    """[(title, layer, boost, {partner: (reach, named, n)})] in ARMS order."""
    probe = json.loads((root / 'magic-nl31/results/probe.json').read_text())['records']
    out = []
    for path, title, layer, boost in ARMS:
        episodes = json.loads((root / path / 'episodes.json').read_text())
        bars = {}
        for partner in SIDES:
            steered = SIDES[1 - SIDES.index(partner)]
            reach = [r['depths']['27']['last'] for r in probe
                     if r['layer'] == layer and r['boost'] == boost and r['steered'] == steered]
            named = [r['success'] for r in episodes if r['sender'] == steered and r['condition'] == 'link']
            assert len(reach) == 24 and len(named) == 48
            bars[partner] = (sum(reach) / len(reach), sum(named), len(named))
        out.append((title, layer, boost, bars))
    return out


def draw(data, stem):
    W, M = 2400, 36
    label_w, gap = 430, 22
    top, row_h, bar_h, axis_h, key_h = 128, 176, 52, 150, 96
    panel_h = 30 + len(data) * row_h + axis_h
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = {x: mascot(x) for x in SIDES}
    s.text(M, 90, TITLE, min(60, int(60 * (W - 2 * M) / width(TITLE, 60, 700))), 700)

    px, pw = M + label_w + gap, W - 2 * M - label_w - gap
    s.rect(px, top, pw, panel_h, COOL, 32)
    x0, bw = px + 140, pw - 140 - 420
    for i in range(5):
        x = x0 + bw * i * 0.1 / XMAX
        s.line(x, top + 24, x, top + 30 + len(data) * row_h, '#DCDCE2' if i else '#B9B9C0', 3)
        s.text(x, top + 30 + len(data) * row_h + 50, f'{i * 0.1:.1f}', 36, 400, INK, 'middle')
    s.text(x0, top + 30 + len(data) * row_h + 108, AXIS, min(36, int(36 * (px + pw - 30 - x0) / width(AXIS, 36, 700))),
           700)

    y0 = top + 30
    for a, (title, layer, boost, bars) in enumerate(data):
        if a and boost != data[a - 1][2]:
            s.line(M, y0 - 4, M + label_w - 10, y0 - 4, '#D8D8DE', 4)
        s.text(M + 14, y0 + 62, title, 44, 700)
        bx, by, blen = M + 14, y0 + 102, label_w - 60
        s.rect(bx, by - 9, blen, 18, '#E4E4EA', 9)
        s.circle(bx + 9 + (blen - 18) * layer / (N_BLOCKS - 1), by, 9 + 5 * (boost > 1), INK)
        for k, partner in enumerate(SIDES):
            reach, named, n = bars[partner]
            cy = y0 + 22 + k * (bar_h + 18)
            s.image(px + 50, cy - 6, 64, uris[partner])
            length = max(6., bw * min(reach, XMAX) / XMAX)
            s.rect(x0, cy, length, bar_h, COLOR[partner], 10)
            s.rich(x0 + length + 22, cy + bar_h - 12,
                   [(f'{named}', COLOR[partner] if named else INK, 700), (f' of {n} named it', INK, 400)], 40)
        y0 += row_h

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    y, x = ky + key_h / 2 + 13, M + 44
    for partner, label in (('quill', 'Quill reading a steered Ember'), ('ember', 'Ember reading a steered Quill'),
                           ('depth', 'steered block, larger at 4×')):
        if partner == 'depth':
            s.rect(x - 6, y - 21, 60, 18, '#E4E4EA', 9)
            s.circle(x + 24, y - 12, 10, INK)
            x += 26
        else:
            s.rect(x, y - 30, 40, 36, COLOR[partner], 8)
            x += 6
        s.text(x + 50, y, label, 36)
        x += 50 + width(label, 36) + 60
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(rows(root), HERE / 'steered_reach')
    print(HERE / 'steered_reach.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
