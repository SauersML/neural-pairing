"""The steered-twin test at several layers and strengths (pairing/steering_scale.py --layer/--boost, chains/nl29.sh and
chains/nl30.sh), after chat training (nl22c step 157): does the partner's first message say a concept steered into its
twin when the steering is applied after an earlier or later layer, or at four times the strength?

One row per arm: steered after layer 6, 15 (nl29) or 27 at the calibrated steering coefficient, and after layers 15 and
27 at four times it. One card per steered twin; in each, three counts of texts that say the concept, over 12 concepts:
the steered twin's own sentence with the link cut (cut_check.json, 3 prompts per concept), and the partner's first
message while it reads the steered twin, linked and with the link cut (episodes.json, 4 prompts per concept; counted
when the partner says the concept before the steered twin does). figures/steered_gain.py uses draw() and marks();
figures/steered_reach.py uses ARMS.

Reads results/magic-nl29/after and results/magic-nl30/results/{l6,l27,l15x4,l27x4} (JSON; no model) and writes
figures/steered_depth.{svg,png,pdf}.
Usage: python figures/steered_depth.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

from chat_names import EMBER, INK, QUILL, SVG, WARM, count, mascot, pair_icon, wrap

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
ARMS = (('magic-nl30/results/l6', 'After layer 6', 6, 1), ('magic-nl29/after', 'After layer 15', 15, 1),
        ('magic-nl30/results/l27', 'After layer 27', 27, 1),
        ('magic-nl30/results/l15x4', 'After layer 15,\n4× stronger', 15, 4),
        ('magic-nl30/results/l27x4', 'After layer 27,\n4× stronger', 27, 4))
SIDES = ('quill', 'ember')
COLOR = dict(quill=QUILL, ember=EMBER)
TITLE = ('Before training to read steering, the partner says a concept', 'steered after layer 6, or 4× stronger after layer 15')
N_LAYERS = 36


def marks(folder):
    """{steered: {part: [[marks] per concept]}} for the parts sender / link / cut, concepts in protocol order: 1 when
    the text says the concept (for the partner: before the steered twin does), else 0."""
    episodes = json.loads((folder / 'episodes.json').read_text())
    checks = json.loads((folder / 'cut_check.json').read_text())
    concepts = list(dict.fromkeys(r['concept'] for r in episodes))
    out = {}
    for steered in SIDES:
        parts = dict(sender=[[int(r['sender_mention']) for r in sorted(
            (r for r in checks if r['sender'] == steered and r['concept'] == c), key=lambda r: r['prompt'])]
            for c in concepts])
        for cond in ('link', 'cut'):
            parts[cond] = [[int(r['success']) for r in sorted(
                (r for r in episodes if r['sender'] == steered and r['condition'] == cond and r['concept'] == c),
                key=lambda r: (r['structure'], r['prompt']))] for c in concepts]
        out[steered] = parts
    return out


def lines(text, w):
    """text wrapped at width w in bold body text, breaking at each newline too."""
    return [line for part in text.split('\n') for line in wrap(part, w, 46, 700)]


def draw(arms, stem, title=TITLE):
    """arms: [(name, big, marks)], one row each, name in bold (newlines break it); extra space before the first big
    arm. title: one line or a tuple of lines. One card per steered twin with three columns: the steered twin, the
    partner linked, the partner cut."""
    W, M, G, P = 2100, 30, 30, 16
    titles = (title,) if isinstance(title, str) else title
    label_w, head_h, sub_h, row_h, split, icon = 350, 130, 236, 160, 24, 76
    top = 40 + 70 * len(titles)
    cw = (W - 2 * M - label_w - 2 * G) / 2
    sw = (cw - 2 * P) / 3
    gap = [split if a and big != arms[a - 1][1] else 0 for a, (_, big, _) in enumerate(arms)]
    H = top + head_h + sub_h + len(arms) * row_h + sum(gap) + 20 + M
    s = SVG(W, H)
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    for k, line in enumerate(titles):
        s.text(M, 76 + 70 * k, line, 56, 700)

    sy = top + head_h + sub_h - 30                                   # baseline of the column labels' last line
    s.text(M, sy - 56, 'Says the', 46, 700)
    s.text(M, sy, 'concept', 46, 700)
    for k, steered in enumerate(SIDES):
        partner = SIDES[1 - k]
        x0 = M + label_w + G + k * (cw + G)
        s.rect(x0, top, cw, H - top - M, WARM, 32)
        s.image(x0 + P + 10, top + 16, 100, uris[steered])
        s.text(x0 + P + 124, top + 82, f'{steered.title()} steered', 46, 700)
        for c, (who, cond, label) in enumerate(((steered, None, (steered.title(), 'itself')),
                                                (partner, 'link', (partner.title(), 'linked')),
                                                (partner, 'cut', (partner.title(), 'cut')))):
            cx = x0 + P + (c + 0.5) * sw
            if cond is None:
                s.image(cx - icon / 2, top + head_h, icon, uris[who])
            else:
                pair_icon(s, cx - (2 * icon + 86) / 2, top + head_h, icon, cond, uris)
            for i, line in enumerate(label):
                s.text(cx, sy - 56 * (1 - i), line, 46, 700, COLOR[who], 'middle')

    y0 = top + head_h + sub_h
    for a, (name, big, data) in enumerate(arms):
        y0 += gap[a]
        text = lines(name, label_w - 10)
        for i, line in enumerate(text):
            s.text(M, y0 + (row_h - 56 * len(text)) / 2 + 42 + 56 * i, line, 46, 700)
        for k, steered in enumerate(SIDES):
            partner = SIDES[1 - k]
            x0 = M + label_w + G + k * (cw + G)
            for c, (part, who) in enumerate((('sender', steered), ('link', partner), ('cut', partner))):
                grid = data[steered][part]
                n = sum(map(sum, grid))
                count(s, x0 + P + (c + 0.5) * sw, y0 + row_h / 2 + 26, n, sum(map(len, grid)),
                      COLOR[who] if n else INK, anchor='middle')
        y0 += row_h
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw([(name, boost > 1, marks(root / path)) for path, name, _, boost in ARMS], HERE / 'steered_depth')
    print(HERE / 'steered_depth.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
