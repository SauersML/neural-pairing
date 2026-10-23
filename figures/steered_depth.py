"""The steered-twin test at other depths and strengths (steering_scale.py --layer/--boost, chains/nl30.sh), after chat
training (nl22c step 157): does a concept pushed into one twin reach its partner when the push is deeper, shallower or
four times stronger?

One row per arm: steered after block 6, 15 (nl29) or 27 at the calibrated strength, and after blocks 15 and 27 at four
times it. One panel per steered twin; in each, three grids with one column per concept (12) and one mark per text,
filled when the text says the concept: the steered twin's own sentence with the link cut (cut_check.json, 3 prompts per
concept), and the partner's first message while it reads the steered twin, linked and with the link cut (episodes.json,
4 prompts per concept; filled when the partner says the concept before the steered twin does). The bar under each arm's
name marks the steered block among the 36.

Reads results/magic-nl29/after and results/magic-nl30/results/{l6,l27,l15x4,l27x4} (JSON; no model, no torch) and
writes figures/steered_depth.{svg,png,pdf}.
Usage: python figures/steered_depth.py [RESULTS_ROOT]   (default: results)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, FAINT, INK, QUILL, SVG, WARM, mascot, width  # noqa: E402
from steered_link import marks  # noqa: E402

RESULTS = HERE.parent / 'results'
ARMS = (('magic-nl30/results/l6', 'Block 6', 6, 1), ('magic-nl29/after', 'Block 15', 15, 1),
        ('magic-nl30/results/l27', 'Block 27', 27, 1), ('magic-nl30/results/l15x4', 'Block 15, 4×', 15, 4),
        ('magic-nl30/results/l27x4', 'Block 27, 4×', 27, 4))
SIDES = ('quill', 'ember')
COLOR = dict(quill=QUILL, ember=EMBER)
COLUMNS = (('sender', 'Says it'), ('link', 'Partner, linked'), ('cut', 'Partner, cut'))
TITLE = "Pushed in early or hard enough, a concept crosses into the partner's words"
PITCH, RAD, N_BLOCKS = 22, 8, 36


def draw(arms, stem):
    W, M = 2400, 36
    label_w, gap = 430, 22
    top, head_h, sub_h, row_h, key_h = 128, 116, 64, 196, 96
    panel_h = head_h + sub_h + len(arms) * row_h + 10
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = {x: mascot(x) for x in SIDES}
    s.text(M, 90, TITLE, min(60, int(60 * (W - 2 * M) / width(TITLE, 60, 700))), 700)

    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    grid_w = 11 * PITCH + 2 * RAD
    cell = (pw - 60) / 3
    for k, steered in enumerate(SIDES):
        s.rect(px[k], top, pw, panel_h, (COOL, WARM)[k], 32)
        s.image(px[k] + 34, top + 20, 84, uris[steered])
        s.text(px[k] + 134, top + 78, f'{steered.title()} steered', 48, 700)
        for c, (_, label) in enumerate(COLUMNS):
            s.text(px[k] + 30 + c * cell + (cell - grid_w) / 2, top + head_h + 40, label, 34, 700)

    y0 = top + head_h + sub_h
    for a, (_, title, layer, boost, data) in enumerate(arms):
        if a and boost != arms[a - 1][3]:
            for x1, x2, color in ((M, M + label_w - 10, '#D8D8DE'), (px[0] + 30, px[0] + pw - 30, '#D8D8DE'),
                                  (px[1] + 30, px[1] + pw - 30, '#DDD6CC')):
                s.line(x1, y0 - 4, x2, y0 - 4, color, 4)
        s.text(M + 14, y0 + 62, title, 44, 700)
        bx, by, bw = M + 14, y0 + 102, label_w - 60
        s.rect(bx, by - 9, bw, 18, '#E4E4EA', 9)
        s.circle(bx + 9 + (bw - 18) * layer / (N_BLOCKS - 1), by, 9 + 5 * (boost > 1), INK)
        for k, steered in enumerate(SIDES):
            partner = SIDES[1 - k]
            for c, (row, _) in enumerate(COLUMNS):
                who = steered if row == 'sender' else partner
                grid = data[steered][row]
                hits, n = sum(map(sum, grid)), sum(map(len, grid))
                cx0 = px[k] + 30 + c * cell + (cell - grid_w) / 2 + RAD
                s.rich(cx0 - RAD, y0 + 52, [(f'{hits}', COLOR[who] if hits else INK, 700), (f' of {n}', INK, 400)], 40)
                for j, column in enumerate(grid):
                    for i, v in enumerate(column):
                        cx, cy = cx0 + j * PITCH, y0 + 82 + i * PITCH
                        if v:
                            s.circle(cx, cy, RAD, COLOR[who])
                        else:
                            s.circle(cx, cy, RAD - 1.5, 'white', FAINT, 2.5)
        y0 += row_h

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    y = ky + key_h / 2 + 13
    x = M + 44
    for fill, label in ((QUILL, 'Quill says the concept'), (EMBER, 'Ember says the concept'), (None, 'does not'),
                        ('cols', 'one column per concept'), ('depth', 'steered block, larger at 4×')):
        if fill == 'cols':
            for i in range(3):
                s.circle(x + 17, y - 30 + i * 18, 7, FAINT)
            x += 6
        elif fill == 'depth':
            s.rect(x - 6, y - 21, 60, 18, '#E4E4EA', 9)
            s.circle(x + 24, y - 12, 10, INK)
            x += 26
        elif fill:
            s.circle(x + 17, y - 12, 17, fill)
        else:
            s.circle(x + 17, y - 12, 15.5, 'white', FAINT, 3)
        s.text(x + 50, y, label, 36)
        x += 50 + width(label, 36) + 54
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw([(path, title, layer, boost, marks(root / path)) for path, title, layer, boost in ARMS], HERE / 'steered_depth')
    print(HERE / 'steered_depth.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
