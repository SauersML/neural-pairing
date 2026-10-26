"""The beauty RL run (chains/beautyrl.sh) as two figures. The judge drawn is the one gate.json names as held out, a
model from a family neither twin comes from, scoring with a rubric training never used.

beauty_rl_prompts: each of the 48 prompts never seen in training, that judge's score difference (trained minus
untrained, linked plays, EVAL rubric) as a bar, sorted, under the number of prompts on which it is positive.
beauty_rl_cut: that judge's strict win rate of linked over link-cut plays, before and after training, as bars.

Reads RESULTS_DIR/judged.json and RESULTS_DIR/gate.json (no model) and writes figures/beauty_rl_{prompts,cut}.{svg,png,pdf}
and results/beauty_rl.json (the numbers drawn).
Usage: python figures/beauty_rl.py RESULTS_DIR   (results/magic-beautyrl6/results)
"""
import json
import math
import sys
from pathlib import Path

from chat_names import COOL, INK, LINK, SVG, WARM, pair_icon, width

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def prompts(judged, held, stem):
    """Each prompt's score difference from the judge of another family, trained minus untrained (linked plays, EVAL
    rubric, mean over the pairs of plays), as one bar, sorted."""
    per = judged['judges'][held]['eval']['diff']['per_prompt']['rl>init']
    summ = judged['judges'][held]['eval']['diff']['summary']['rl>init']
    W, H, M = 2400, 1600, 40
    s = SVG(W, H)
    s.rich(M, 92, [('On ', INK, 700), (f'{summ["above"]} of {summ["n"]}', LINK, 700),
                   (' prompts never seen in training,', INK, 700)], 56)
    s.text(M, 162, 'the trained twins’ conversations are scored higher', 56, 700)
    left, right, top, bot = M + 250, W - M, 270, H - 110
    hi, lo = max(max(per), 0) * 1.05, min(min(per), 0) * 1.05
    sc = (bot - top) / (hi - lo)
    y0 = top + hi * sc
    step = (right - left) / len(per)
    for t in range(math.ceil(lo), math.floor(hi) + 1):
        y = y0 - t * sc
        s.text(left - 24, y + 15, f'+{t}' if t > 0 else f'−{-t}' if t < 0 else '0', 46, 400, INK, 'end')
    # the axis label, rotated, centred on the bars
    ym = (top + bot) / 2
    s.text(M + 50, ym, 'judge score from another model family', 44, 700, INK, 'middle', rotate=-90)
    s.text(M + 104, ym, 'trained twins minus untrained twins', 44, 400, INK, 'middle', rotate=-90)
    for i, v in enumerate(sorted(per, reverse=True)):
        x = left + i * step + step * 0.12
        h = abs(v) * sc
        s.rect(x, y0 - h if v > 0 else y0, step * 0.76, max(h, 2), LINK if v > 0 else COOL, 4, 'none' if v > 0 else INK, 3)
    s.line(left, y0, right, y0, INK, 4)
    s.text(left + 20, top + 10, 'trained higher', 48, 700, LINK)
    s.text(right, H - 44, 'untrained higher', 48, 700, INK, 'end')
    s.save(stem)
    return dict(higher=summ['above'], lower=summ['below'], n=summ['n'], mean_diff=summ['mean'])


def cut(judged, held, stem):
    """The other-family judge's strict win rate of linked over link-cut plays (mean over prompts), before and after
    training, as two large numbers, each beside a linked pair drawn above a cut pair."""
    st = judged['judges'][held]['eval']['strict']['summary']
    vals = [st['init:link>cut']['mean'], st['rl:link>cut']['mean']]
    W, H, M = 2000, 1300, 40
    s = SVG(W, H)
    s.text(M, 88, 'The linked conversation is scored above the cut conversation', 56, 700)
    s.text(M, 160, f'{round(100 * vals[0])}% of the time before training and {round(100 * vals[1])}% after', 56, 700)
    top, gap, size, g = 220, 30, 290, 20
    pw, ph = (W - 2 * M - gap) / 2, H - M - top
    for k, (v, head, fill) in enumerate(zip(vals, ('Before training', 'After training'), (COOL, WARM))):
        x = M + k * (pw + gap)
        s.rect(x, top, pw, ph, fill, 32)
        s.text(x + pw / 2, top + 84, head, 52, 700, INK, 'middle')
        # the comparison, stacked: the linked pair, "above", the cut pair; the rate beside it, in ink (55% to 64% is
        # within noise)
        sx = x + 30 - 0.25 * size                  # the mascot art starts at 0.25 of its square
        cx = sx + size + g / 2
        art = sx + 1.75 * size + g                 # where the pair's art ends
        p = top + 130
        pair_icon(s, sx, p, size, 'link', gap=g)
        s.text(cx, p + size + 55, 'linked', 46, 400, INK, 'middle')
        s.text(cx, p + size + 140, 'above', 52, 700, INK, 'middle')
        pair_icon(s, sx, p + size + 185, size, 'cut', gap=g)
        s.text(cx, p + 2 * size + 240, 'cut', 46, 400, INK, 'middle')
        assert top + ph - (p + 2 * size + 240) <= 100      # a thin band below the stack
        nx = (art + x + pw) / 2
        label = f'{round(100 * v)}%'
        assert nx - width(label, 170, 700) / 2 > art + 30 and nx + width(label, 170, 700) / 2 < x + pw - 30
        s.text(nx, p + size + 180, label, 170, 700, INK, 'middle')
    s.save(stem)
    return dict(before=st['init:link>cut'], after=st['rl:link>cut'], change=st['rl-init:link>cut'])


def main():
    res = Path(sys.argv[1])
    judged = json.load(open(res / 'judged.json'))
    held = json.load(open(res / 'gate.json'))['held']
    out = dict(source='results/' + res.parent.name + '/results', held=held,
               prompts=prompts(judged, held, HERE / 'beauty_rl_prompts'),
               cut=cut(judged, held, HERE / 'beauty_rl_cut'))
    (ROOT / 'results/beauty_rl.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
