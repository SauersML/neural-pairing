"""The steered-twin test after training the link to read a steered concept (chains/nl33.sh): how often the partner says
a concept steered into its twin, before (nl22c step 157: nl29, nl30), after 300 more steps without the steered-reading game
(nl33w), and after 300 steps with the steered-reading game (pairing/steer_read.py, nl33t),
whose concepts never include the test's 12. figures/steered_method.py draws one of these episodes.

Top: the partner's first message (pairing/steering_scale.py, the nl29/nl30 protocol: 96 linked messages per setting,
over both turn structures). Bottom: the partner asked what its twin is thinking about (pairing/steer_ask.py, wordings
never trained; 48 per setting). With the link cut the partner says the concept in no setting (checked in load).

Reads results/magic-nl29/after, results/magic-nl30/results/*, results/magic-nl33w/results/* and
results/magic-nl33t/results/* (JSON; no model) and writes figures/steered_trained.{svg,png,pdf}.
Usage: python figures/steered_trained.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

from chat_names import LINK, SVG, WARM, width
from steered_depth import lines

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
SETTINGS = (('l6', 'After layer 6'), ('l15', 'After layer 15'), ('l27', 'After layer 27'),
            ('l15x4', 'After layer 15,\n4× stronger'), ('l27x4', 'After layer 27,\n4× stronger'))
BEFORE = dict(l6='magic-nl30/results/l6', l15='magic-nl29/after', l27='magic-nl30/results/l27',
              l15x4='magic-nl30/results/l15x4', l27x4='magic-nl30/results/l27x4')
ARMS = (('before', 'Chat-trained twins, before', '#D9D9DE'),
        ('write', '300 more steps without the steered-reading game', '#9FD8C5'),
        ('steer', '300 more steps with the steered-reading game', LINK))
RUNS = dict(write='magic-nl33w', steer='magic-nl33t')
TITLE = ('After training to read steering, the partner says the steered concept',
         'after every layer, including layer 27, and never with the link cut')
PANELS = ('The partner says the concept in the first message, of 96',
          'Asked about the twin, the partner says the concept, of 48')


def chat_counts(folder):
    """(linked, cut) messages, of 96, in which the partner says the concept first."""
    t = json.loads((folder / 'summary.json').read_text())
    return tuple(sum(c['counts'][k] for r in t for c in r['concepts']) for k in ('link', 'cut'))


def ask_counts(path):
    """(linked, cut, n) answers that say the concept."""
    s = json.loads(path.read_text())['summary']['both']
    return s['counts']['link'], s['counts']['cut'], s['n']


def load(root):
    """(chat, ask): counts per (arm, setting) for the partner's first message and the steer_ask question."""
    chat, ask = {}, {}
    for t, _ in SETTINGS:
        chat['before', t] = chat_counts(root / BEFORE[t])
        for arm, run in RUNS.items():
            chat[arm, t] = chat_counts(root / run / 'results' / t)
            ask[arm, t] = ask_counts(root / run / 'results' / f'ask_{t}.json')
        ask['before', t] = ask_counts(root / RUNS['write'] / 'results' / f'ask_before_{t}.json')
    assert all(v[1] == 0 for v in list(chat.values()) + list(ask.values())), 'a partner said the concept with the link cut'
    return chat, ask


def draw(chat, ask, stem):
    W, M, G, P = 2000, 30, 30, 30
    label_w, head_h, bar_h, bar_gap, group_gap = 400, 100, 50, 8, 28
    group_h = len(ARMS) * (bar_h + bar_gap) - bar_gap + group_gap
    panel_h = P + head_h + len(SETTINGS) * group_h - group_gap + P
    top = 180
    H = top + 2 * panel_h + M
    s = SVG(W, H)
    for k, line in enumerate(TITLE):
        s.text(M, 76 + 70 * k, line, 56, 700)
    x0 = M + P + label_w
    room = min(SETTINGS, key=lambda st: max(chat[arm, st[0]][0] for arm, _, _ in ARMS))[0]  # labels the arms
    for p, (data, n, head, fill) in enumerate(((chat, 96, PANELS[0], WARM), (ask, 48, PANELS[1], WARM))):
        y = top + p * (panel_h + G)
        s.rect(M, y, W - 2 * M, panel_h, fill, 32)
        s.text(M + P, y + P + 54, head, 46, 700)
        bw = W - M - P - x0 - 140
        for i, (t, name) in enumerate(SETTINGS):
            gy = y + P + head_h + i * group_h
            text = lines(name, label_w - 20)
            mid = gy + (group_h - group_gap) / 2
            for k, line in enumerate(text):
                s.text(M + P, mid - 56 * len(text) / 2 + 42 + 56 * k, line, 46, 700)
            for j, (arm, label, color) in enumerate(ARMS):
                by = gy + j * (bar_h + bar_gap)
                v = data[arm, t][0]
                length = max(6, bw * v / n)
                s.rect(x0, by, length, bar_h, color, 8)
                tx = x0 + length + 16
                s.text(tx, by + bar_h / 2 + 19, str(v), 54, 700)
                if p == 0 and t == room:
                    s.text(tx + width(str(v), 54, 700) + 30, by + bar_h / 2 + 16, label, 46)
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(*load(root), HERE / 'steered_trained')
    print(HERE / 'steered_trained.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
