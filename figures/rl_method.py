"""Judge-rewarded training of the twins' open conversations (pairing/rl.py --open_frac, chains/beautyrl.sh) as a
picture: each prompt is played linked and cut, a judge model scores each conversation from 1 to 9, and a linked play's
reward is its mean strict win over the same prompt's cut plays (judge.strict_win: 1 when the score difference is above
the judge's noise margin, 0 below minus the margin, 1/2 between).

Reads results/magic-beautyrl6/results/gen.json (the plays of the checkpoint training starts from, 'init': 2 linked and
2 cut plays of 48 prompts), judged.json (per prompt, the training judge's mean score difference and mean strict win
over the 2 x 2 linked-against-cut pairs, on the training rubric rl.RUBRIC), gate.json (the training judge and its
margin) and the group sizes (--G, --open_cut) and judge from chains/beautyrl.sh. Per-conversation scores are not
stored; the figure shows the stored per-prompt mean difference, and every one of the 4 pairs is a win when the stored
mean strict win is 1. The prompt is the one with the largest mean difference among the prompts whose 4 pairs all win;
the cut play drawn is the first that repeats a turn word for word, the linked play drawn is the first. Turns are
trimmed to their opening. Writes figures/rl_method.{svg,png,pdf} and results/rl_method.json.
Usage: python figures/rl_method.py
"""
import ast
import json
import re

from chat_names import COOL, INK, LINK, RED, ROOT, SVG, WARM, width
from method_parts import (TINT, arrow, bubble, bubble_height, check_shape, flow, highlight, pair_icon, render,
                          robot)
from affect_method import opening

R = ROOT / 'results/magic-beautyrl6/results'
OUT = ROOT / 'figures/rl_method'
RESULTS = ROOT / 'results/rl_method.json'
CHAIN = ROOT / 'chains/beautyrl.sh'
W, M, PAD = 2400, 36, 40
BODY = 46
NAMES = {'quill': 'Quill', 'ember': 'Ember'}


def judge_scale():
    """The lowest and highest score the judge may give (judge.DIGITS)."""
    tree = ast.parse((ROOT / 'pairing/judge.py').read_text())
    digits = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'DIGITS' for t in n.targets))
    return int(digits[0]), int(digits[-1])


def norm(t):
    return re.sub(r'[\W_]+', ' ', t.lower()).strip()


def flag(text, name):
    m = re.search(rf'--{name} (\S+)', text)
    if not m:
        raise KeyError(name)
    return m.group(1)


def load():
    gen = json.loads((R / 'gen.json').read_text())
    judged = json.loads((R / 'judged.json').read_text())
    gate = json.loads((R / 'gate.json').read_text())
    chain = CHAIN.read_text()
    train = re.search(r'TRAIN=(\S+)', chain).group(1)
    assert train == gate['train_arm']
    G, n_cut = int(flag(chain, 'G')), int(flag(chain, 'open_cut'))
    q = judged['judges'][train]['train']
    strict = q['strict']['per_prompt']['init:link>cut']
    diff = q['diff']['per_prompt']['init:link>cut']
    margin = gate['margins'][train]
    i = max((k for k, v in enumerate(strict) if v == 1.0), key=lambda k: diff[k])
    links = [gen['init']['link'][k][i] for k in range(judged['samples'])]
    cuts = [gen['init']['cut'][k][i] for k in range(judged['samples'])]

    def repeats(conv):
        seen = set()
        for _, t in conv:
            if norm(t) in seen:
                return True
            seen.add(norm(t))
        return False

    c = next(k for k, conv in enumerate(cuts) if repeats(conv))
    p = gen['prompts'][i]
    return dict(source=str(R.relative_to(ROOT)), prompt_index=i, prompt=p, judge=train, margin=margin,
                linked_plays=G, cut_plays=n_cut, mean_difference=diff[i], mean_strict_win=strict[i],
                pairs=judged['samples'] ** 2, link_play=0, cut_play=c, link=links[0], cut=cuts[c],
                abs_max_difference=max(abs(v) for v in diff), scale=judge_scale())


def conversation(s, x, y, w, turns, n=3, mark=None, cut=False):
    """The first n turns as bubbles beside the twins (in the link-cut pose with cut); mark: the turn indices outlined in
    red. Returns the height."""
    RB = 96
    y0 = y
    for k, (sp, t) in enumerate(turns[:n]):
        runs = [(opening(t, 64), INK, 400, None)]
        bw = w - RB - 26
        h = bubble_height(runs, bw, BODY)
        stroke = RED if mark and k in mark else 'none'
        if sp == turns[0][0]:
            robot(s, sp, x, y + h / 2 - RB / 2, RB, cut=cut, facing=1)
            bubble(s, x + RB + 26, y, bw, runs, BODY, stroke=stroke, sw=5)
        else:
            robot(s, sp, x + w - RB, y + h / 2 - RB / 2, RB, cut=cut, facing=-1)
            bubble(s, x, y, bw, runs, BODY, tail='right', stroke=stroke, sw=5)
        y += h + 22
    return y - y0 - 22


def conversation_height(turns, w, n=3):
    RB = 96
    return sum(bubble_height([(opening(t, 64), INK, 400, None)], w - RB - 26, BODY) + 22 for _, t in turns[:n]) - 22


def draw(r):
    p = r['prompt']
    first = p['first']
    title = 'The twins earn reward when the judge scores a linked conversation more than ' \
            f'{r["margin"]:.2f} points above a cut conversation of the same prompt'
    cut_at = title.rfind(' more than ')
    title_lines = [title[:cut_at], title[cut_at + 1:]]
    top = 56 + 70 * len(title_lines)

    # the prompt and the group of plays
    prompt_text = p['template'].format(P=NAMES['quill' if first == 'ember' else 'ember'], X=p['topic'])
    prompt_runs = highlight(prompt_text, [p['topic']], INK, TINT['red'])
    pw = 760
    size, row = 90, 140
    band = max(110 + bubble_height(prompt_runs, pw, BODY) - 20, 40 + 2 * row) + 30

    cw = (W - 2 * M - 30) / 2
    inner = cw - 2 * PAD
    rep = {k for k, (_, t) in enumerate(r['cut'])
           if any(norm(t) == norm(u) for _, u in r['cut'][:k])}
    k_rep = min(rep)
    j_rep = next(j for j, (_, u) in enumerate(r['cut']) if norm(u) == norm(r['cut'][k_rep][1]))
    rep_label = f'{NAMES[r["cut"][k_rep][0]]} repeats turn {j_rep + 1}'
    conv_h = max(conversation_height(r['link'], inner), conversation_height(r['cut'], inner))
    card_h = 110 + conv_h + 40 + 70
    judge_h = 330
    H = round(top + band + 30 + card_h + 30 + judge_h + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    for k, line in enumerate(title_lines):
        s.text(M, 84 + 70 * k, line, 56, 700)

    y = top
    s.rect(M, y, W - 2 * M, band, COOL, 32)
    s.text(M + PAD, y + 72, 'One prompt', 48, 700)
    flow(s, M + PAD, y + 140, pw, prompt_runs, BODY)
    gx = M + PAD + pw + 70
    for k, (n, label, on, color) in enumerate(((r['linked_plays'], 'plays linked', True, LINK),
                                                (r['cut_plays'], 'plays cut', False, INK))):
        ry = y + 40 + k * row
        s.text(gx, ry + size * 0.62 + 16, f'{n} {label}', 48, 700, color)
        ix = gx + width(f'{r["linked_plays"]} plays linked', 48, 700) + 50
        for j in range(n):
            ix += pair_icon(s, ix, ry, size, on) + 72
    band_bottom = y + band

    # one linked play and one cut play, real turns
    y = band_bottom + 30
    for k, (label, turns, fill, on, mark) in enumerate((('Linked', r['link'], WARM, True, None),
                                                        ('Cut', r['cut'], COOL, False, rep))):
        x = M + k * (cw + 30)
        s.rect(x, y, cw, card_h, fill, 32)
        w_ = pair_icon(s, x + PAD, y + 26, 70, on)
        s.text(x + PAD + w_ + 30, y + 80, f'{label} conversation', 48, 700, LINK if on else INK)
        h_ = conversation(s, x + PAD, y + 110, inner, turns, mark=mark, cut=not on)
        if mark:            # under the repeated turn
            s.text(x + cw - PAD, y + 110 + h_ + 64, rep_label, 46, 700, RED, 'end')

    # the judge's score difference and the reward
    y = y + card_h + 30
    s.rect(M, y, W - 2 * M, judge_h, COOL, 32)
    lo, hi = r['scale']
    s.text(M + PAD, y + 72, f'Judge ({r["judge"].split("/")[-1]}) score from {lo} to {hi}, linked minus cut', 48, 700)
    span = int(r['abs_max_difference']) + 1
    ax0, ax1 = M + PAD + 40, W - M - PAD - 560
    ay = y + 230
    xs = lambda v: ax0 + (v + span) / (2 * span) * (ax1 - ax0)
    m = r['margin']
    s.rect(xs(-m), ay - 50, xs(m) - xs(-m), 100, '#E4E4E8', 12)
    s.line(ax0, ay, ax1, ay, INK, 4)
    for v in range(-span, span + 1):
        s.line(xs(v), ay - 12, xs(v), ay + 12, INK, 3)
    for v in (-span, 0, span):
        s.text(xs(v), ay + 62, f'{v:+d}'.replace('-', '−') if v else '0', 46, 400, INK, 'middle')
    s.text((ax0 + xs(-m)) / 2, ay - 66, 'loss: 0', 46, 700, INK, 'middle')
    s.text(0.5 * (xs(-m) + xs(m)), ay - 66, 'within noise: ½', 46, 700, INK, 'middle')
    s.text((xs(m) + ax1) / 2, ay - 66, 'win: 1', 46, 700, LINK, 'middle')
    d = r['mean_difference']
    s.circle(xs(d), ay, 20, RED)
    s.text(xs(d), ay + 62, f'{d:+.1f}', 46, 700, RED, 'middle')
    rx = ax1 + 80
    arrow(s, [(ax1 + 14, ay), (rx - 10, ay)], INK, 6, 22)
    s.text(rx, ay - 20, 'reward', 48, 700, INK)
    s.text(rx, ay + 60, f'{r["mean_strict_win"]:g}', 72, 700, RED)
    s.text(rx + width(f'{r["mean_strict_win"]:g}', 72, 700) + 16, ay + 60, f'every pair a win', 46, 700, RED)
    render(s, OUT)
    return H


def main():
    r = load()
    H = draw(r)
    RESULTS.write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(height=H, prompt=r['prompt'], diff=r['mean_difference'], strict=r['mean_strict_win'],
                          margin=r['margin'], cut_play=r['cut_play'])))


if __name__ == '__main__':
    main()
