"""The beauty RL run (chains/beautyrl.sh) as two figures, from its results directory (no model, no torch).

The held-out judge is gate.json's. beauty_rl_prompts: each held-out prompt's held-out-judge score difference (trained
minus untrained, linked plays, EVAL rubric) as a bar, sorted, under the number of prompts on which it is positive.
beauty_rl_cut: the held-out judge's strict win rate of linked over cut plays, before and after RL, as bars.

Usage: python beauty_rl.py RESULTS_DIR [CUT_TITLE, lines split at a newline]   (writes figures/beauty_rl_{prompts,cut}.{svg,png,pdf} and
       results/beauty_rl.json, the numbers drawn)
"""
import json
import math
import sys
from pathlib import Path

from chat_names import COOL, INK, LINK, SVG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def prompts(judged, held, stem):
    """Each held-out prompt's held-out-judge score difference, trained minus untrained (linked plays, EVAL rubric, mean
    over the pairs of plays), as one bar, sorted."""
    per = judged['judges'][held]['eval']['diff']['per_prompt']['rl>init']
    summ = judged['judges'][held]['eval']['diff']['summary']['rl>init']
    W, H, M = 2400, 1250, 40
    s = SVG(W, H)
    s.rich(M, 100, [(f'{summ["above"]} of {summ["n"]}', LINK, 700), (' new prompts: the judge from another family', INK, 700)], 64)
    s.text(M, 180, "scores the trained twins' conversation higher", 64, 700)
    left, right, top, bot = M + 150, W - M, 260, H - 110
    hi, lo = max(max(per), 0) * 1.05, min(min(per), 0) * 1.05
    sc = (bot - top) / (hi - lo)
    y0 = top + hi * sc
    step = (right - left) / len(per)
    for t in range(math.ceil(lo), math.floor(hi) + 1):
        y = y0 - t * sc
        s.text(left - 24, y + 15, f'{t:+d}' if t else '0', 44, 400, INK, 'end')
    for i, v in enumerate(sorted(per, reverse=True)):
        x = left + i * step + step * 0.12
        h = abs(v) * sc
        s.rect(x, y0 - h if v > 0 else y0, step * 0.76, max(h, 2), LINK if v > 0 else COOL, 4, 'none' if v > 0 else INK, 3)
    s.line(left, y0, right, y0, INK, 4)
    s.text(left + 20, top + 10, 'trained higher', 46, 700, LINK)
    s.text(right, H - 40, 'untrained higher', 46, 700, INK, 'end')
    s.save(stem)
    return dict(higher=summ['above'], lower=summ['below'], n=summ['n'], mean_diff=summ['mean'])


def cut(judged, held, stem, title):
    st = judged['judges'][held]['eval']['strict']['summary']
    vals = [st['init:link>cut']['mean'], st['rl:link>cut']['mean']]
    W, H, M = 2000, 1100, 40
    s = SVG(W, H)
    for i, line in enumerate(title.split('\n')):
        s.text(M, 100 + 80 * i, line, 60, 700)
    base, top = H - 170, 300
    scale = (base - top) / (1.15 * max(vals + [0.5]))
    bw = 420
    xs = [W / 2 - bw - 120, W / 2 + 120]
    y50 = base - 0.5 * scale
    s.line(M + 60, y50, W - M - 60, y50, INK, 4)
    s.text(W - M - 60, y50 - 18, 'even', 44, 600, INK, 'end')
    for x, v, label, fill in zip(xs, vals, ('Before RL', 'After RL'), (COOL, LINK)):
        s.rect(x, base - v * scale, bw, v * scale, fill, 12, INK if fill == COOL else 'none', 3)
        s.text(x + bw / 2, base - v * scale - 30, f'{round(100 * v)}%', 72, 700, INK, 'middle')
        s.text(x + bw / 2, base + 70, label, 52, 700, INK, 'middle')
    s.line(M + 60, base, W - M - 60, base, INK, 4)
    s.save(stem)
    return dict(before=st['init:link>cut'], after=st['rl:link>cut'], change=st['rl-init:link>cut'])


def main():
    res = Path(sys.argv[1])
    judged = json.load(open(res / 'judged.json'))
    held = json.load(open(res / 'gate.json'))['held']
    out = dict(source='results/' + res.parent.name + '/results', held=held,
               prompts=prompts(judged, held, HERE / 'beauty_rl_prompts'),
               cut=cut(judged, held, HERE / 'beauty_rl_cut', sys.argv[2] if len(sys.argv) > 2 else
                       'How often the linked conversation\nbeats the link-cut one'))
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/beauty_rl.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
