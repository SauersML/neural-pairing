"""The partner's layer-27 shift toward a concept steered into its twin (pairing/steer_probe.py, chains/nl31.sh), against
how often the partner says the concept (pairing/steering_scale.py, nl29 and nl30), after chat training (nl22c step 157).

One group per steering arm, as in figures/steered_depth.py: after layer 6, 15 or 27 at the calibrated steering
coefficient, and after layers 15 and 27 at four times it. In each group one bar per partner: the shift of the partner's
layer-27 state along its own concept direction, linked, at the last position of its prompt, where its first message
starts (probe.json records, depths['27']['last'], mean over the 12 concepts and 2 prompts), in units of the steering
at which the partner says the concept itself (its calibrated steering coefficient at layer 27). Beside each bar: the
partner's linked messages that say the concept first, of 48 (episodes.json, condition link).

Reads results/magic-nl31/results/probe.json, results/magic-nl29/after and results/magic-nl30/results/{l6,l27,l15x4,l27x4}
(JSON; no model) and writes figures/steered_reach.{svg,png,pdf}.
Usage: python figures/steered_reach.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

from chat_names import COOL, EMBER, INK, QUILL, SVG, count, mascot
from steered_depth import ARMS, lines

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
SIDES = ('quill', 'ember')
COLOR = dict(quill=QUILL, ember=EMBER)
XMAX = 0.4
TITLE = ('Before training, the partner says a steered concept most often',
         "when the partner's layer-27 state is closest to the concept")
AXIS = ("Partner's layer-27 shift toward the concept, as a fraction",
        'of the steering at which the partner says the concept')


def per_arm(root):
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
    W, M, G, P = 2200, 30, 30, 24
    label_w, top, head_h, bar_h, bar_gap, arm_gap, split = 350, 182, 156, 72, 16, 36, 24
    arm_h = 2 * bar_h + bar_gap
    gap = [0] + [arm_gap + (split if data[a][2] != data[a - 1][2] else 0) for a in range(1, len(data))]
    body_h = head_h + len(data) * arm_h + sum(gap)
    H = top + body_h + 40 + 70 + 2 * 56 + 30 + M
    s = SVG(W, H)
    uris = {x: mascot(x) for x in SIDES}
    for i, line in enumerate(TITLE):
        s.text(M, 76 + 70 * i, line, 56, 700)

    px = M + label_w + G
    pw = W - M - px
    s.rect(px, top, pw, H - top - M, COOL, 32)
    cx = px + pw - P - 150                                           # centre of the count column
    s.text(cx, top + 58, 'Partner says', 46, 700, INK, 'middle')
    s.text(cx, top + 114, 'the concept', 46, 700, INK, 'middle')
    x0 = px + P + bar_h + 24
    bw = cx - 190 - x0

    y0 = top + head_h
    for a, (title, layer, boost, bars) in enumerate(data):
        y0 += gap[a]
        text = lines(title, label_w - 10)
        for i, line in enumerate(text):
            s.text(M, y0 + (arm_h - 56 * len(text)) / 2 + 42 + 56 * i, line, 46, 700)
        for k, partner in enumerate(SIDES):
            reach, named, n = bars[partner]
            by = y0 + k * (bar_h + bar_gap)
            s.image(px + P - 4, by - 4, bar_h + 8, uris[partner])
            s.rect(x0, by, max(6., bw * min(reach, XMAX) / XMAX), bar_h, COLOR[partner], 10)
            count(s, cx, by + bar_h / 2 + 26, named, n, COLOR[partner] if named else INK, anchor='middle')
        y0 += arm_h

    ay = y0 + 30
    s.line(x0, ay, x0 + bw, ay, INK, 3)
    for i in range(5):
        x = x0 + bw * i * 0.1 / XMAX
        s.line(x, ay, x, ay + 16, INK, 3)
        s.text(x, ay + 68, f'{i * 0.1:.1f}', 46, 400, INK, 'middle')
    for i, line in enumerate(AXIS):
        s.text(x0, ay + 140 + 56 * i, line, 46, 700)
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(per_arm(root), HERE / 'steered_reach')
    print(HERE / 'steered_reach.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
