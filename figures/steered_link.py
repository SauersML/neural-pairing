"""The steered-twin test in bf16 (steering_scale.py, chains/nl29.sh): a concept pushed into one twin's hidden states
shows up in that twin's words, and not in its partner's.

One panel per steered twin. In each, one column per concept (12) and one mark per text, filled when the text says the
concept. Top row: the steered twin's own sentence, link cut (cut_check.json, 3 prompts per concept). Below: the
partner's first message while it reads the steered twin, linked and with the link cut (episodes.json, 2 chat prompts
and 2 one-item prompts per concept; filled when the partner says the concept, before the steered twin does). One block
per checkpoint: before chat training (nl20 step 1,320) and after it (nl22c step 157).

Reads results/magic-nl29/{before,after}/ (JSON; no model, no torch) and writes figures/steered_link.{svg,png,pdf}.
Usage: python figures/steered_link.py [RESULTS_DIR]   (default: results/magic-nl29)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, FAINT, INK, QUILL, SVG, WARM, mascot, pair_icon, width  # noqa: E402

RESULTS = HERE.parent / 'results/magic-nl29'
BLOCKS = (('before', 'Before chat training'), ('after', 'After chat training'))
SIDES = ('quill', 'ember')
COLOR = dict(quill=QUILL, ember=EMBER)
COLS_PITCH, LINE_PITCH, RAD = 64, 32, 11


def marks(folder):
    """{steered: {row: [[concept's marks] per concept]}} with rows sender / link / cut, concepts in protocol order."""
    episodes = json.loads((folder / 'episodes.json').read_text())
    checks = json.loads((folder / 'cut_check.json').read_text())
    concepts = list(dict.fromkeys(r['concept'] for r in episodes))
    out = {}
    for steered in SIDES:
        rows = dict(sender=[[int(r['sender_mention']) for r in sorted(
            (r for r in checks if r['sender'] == steered and r['concept'] == c), key=lambda r: r['prompt'])]
            for c in concepts])
        for cond in ('link', 'cut'):
            rows[cond] = [[int(r['success']) for r in sorted(
                (r for r in episodes if r['sender'] == steered and r['condition'] == cond and r['concept'] == c),
                key=lambda r: (r['structure'], r['prompt']))] for c in concepts]
        out[steered] = rows
    return out


def draw(blocks, stem):
    W, M = 2400, 36
    label_w, gap = 470, 22
    top, head_h, block_h, key_h = 128, 128, 78, 96
    row_h = dict(sender=82 + 3 * LINE_PITCH + 36, link=82 + 4 * LINE_PITCH + 36, cut=82 + 4 * LINE_PITCH + 36)
    panel_h = head_h + len(blocks) * (block_h + sum(row_h.values())) + 10
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = {x: mascot(x) for x in SIDES}
    s.text(M, 90, "Pushed into one twin, a concept reaches its words, not its partner's", 64, 700)

    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    for k, steered in enumerate(SIDES):
        s.rect(px[k], top, pw, panel_h, (COOL, WARM)[k], 32)
        s.image(px[k] + 34, top + 24, 84, uris[steered])
        s.text(px[k] + 134, top + 82, f'{steered.title()} steered', 48, 700)

    y0 = top + head_h
    for b, (name, title, data) in enumerate(blocks):
        if b:
            for x1, x2, color in ((M, M + label_w - 10, '#D8D8DE'), (px[0] + 30, px[0] + pw - 30, '#D8D8DE'),
                                  (px[1] + 30, px[1] + pw - 30, '#DDD6CC')):
                s.line(x1, y0 + 8, x2, y0 + 8, color, 4)
        s.text(M + 14, y0 + 62, title, 46, 700)
        y0 += block_h
        for row in ('sender', 'link', 'cut'):
            if row == 'sender':
                s.text(M + 14, y0 + 72, 'Steered twin', 42, 700)
                s.text(M + 14, y0 + 124, 'says it', 42, 700)
            else:
                msize = 104
                iy = y0 + 26
                pair_icon(s, M + 14, iy, msize, row, uris)
                s.text(M + 14 + (2 * msize + 86) / 2, iy + msize + 46, 'Partner, ' + ('linked' if row == 'link'
                       else 'link cut'), 38, 700, INK, 'middle')
            for k, steered in enumerate(SIDES):
                who = steered if row == 'sender' else SIDES[1 - SIDES.index(steered)]
                grid = data[steered][row]
                hits, n = sum(map(sum, grid)), sum(map(len, grid))
                cx0 = px[k] + (pw - (len(grid) - 1) * COLS_PITCH) / 2
                s.rich(px[k] + 40, y0 + 66, [(f'{hits}', COLOR[who] if hits else INK, 700), (f' of {n}', INK, 400)]
                       + ([(f'   {who.title()} says it', INK, 400)] if b == 0 else []), 52)
                for j, column in enumerate(grid):
                    for i, v in enumerate(column):
                        cx, cy = cx0 + j * COLS_PITCH, y0 + 104 + i * LINE_PITCH
                        if v:
                            s.circle(cx, cy, RAD, COLOR[who])
                        else:
                            s.circle(cx, cy, RAD - 1.5, 'white', FAINT, 3)
            y0 += row_h[row]

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    y = ky + key_h / 2 + 13
    x = M + 44
    for fill, label in ((QUILL, 'Quill says the concept'), (EMBER, 'Ember says the concept'), (None, 'does not'),
                        ('cols', 'one column per concept')):
        if fill == 'cols':
            for i in range(3):
                s.circle(x + 17, y - 30 + i * 18, 7, FAINT)
            x += 6
        elif fill:
            s.circle(x + 17, y - 12, 17, fill)
        else:
            s.circle(x + 17, y - 12, 15.5, 'white', FAINT, 3)
        s.text(x + 50, y, label, 36)
        x += 50 + width(label, 36) + 60
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw([(name, title, marks(root / name)) for name, title in BLOCKS], HERE / 'steered_link')
    print(HERE / 'steered_link.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
