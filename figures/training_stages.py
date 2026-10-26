"""The whole training in order as one picture: the fitted start, then the four stages of the main line to the final
model (warm-up on the value games, the full mix, chat training with the privileged teacher, and chat training
continued for 300 steps without the steered-reading game, nl33w), each with its step count (and, before the last, a
small real example), and the side branches trained from earlier checkpoints: exam questions (from the full mix);
private items and one-way links, the steered-reading game with original Quill's targets (nl33) and judge-rewarded
reinforcement learning after it, and the game with targets written by the twins themselves (nl40), all from chat
training.

Step counts are the last step of each run's training log: results/magic-nl20/warm/log.json (warm-up), the full mix's
steps from results/training_curve.json (from the repository's history, see training_curve.py), results/magic-nl22c/
log.json (chat training; nl22c is chains/nl22.sh run on the cleaned privileged-teacher chats), magic-nl33t (steered
game), magic-beautyrl6 (reinforcement learning), the last checkpoint saved in magic-nl23/qa_pair.log (exam questions),
magic-nl26/log.json (private items, chains/nl24.sh), magic-nl26/nl25/oneway_*/log.json (one-way links), magic-nl33w
(chat training continued without the game) and --nl40's results/log.json (nl40's game). Each later run's start is checked
against its parent's last step in its chain script (or chain log).

Examples: the fitted start's text is the first of the 192 texts align.content_init reads (rebuilt with align.texts from
nl20's replies and data/coread.json; the rebuild must give the 11,777 matched token pairs content_init.json records),
cut by both tokenizers; the warm-up episode is the first word-game episode games.Games draws from the training split
(seed 0); the full mix's example is the training row of results/shared_reading.json; chat training's is
results/teacher_method.json's. --refresh rebuilds results/training_stages.json (needs nl20's replies.json,
data/coread.json and the two tokenizers in the local Hugging Face cache); without it the figure is drawn from that file.
Writes figures/training_stages.{svg,png,pdf}.
Usage: python figures/training_stages.py [--refresh --replies REPLIES --coread COREAD [--nl40 NL40_DIR]]
"""
import argparse
import ast
import inspect
import json
import re
import sys
from pathlib import Path

from affect_method import opening
from chat_names import COOL, INK, LINK, ROOT, SVG, WARM, width
from method_parts import (COLOR, NAME, TINT, arrow, bubble, bubble_height, check_shape, flow, flow_height, layout,
                          link_down, render, robot)

OUT = ROOT / 'figures/training_stages'
SNAP = ROOT / 'results/training_stages.json'
R = ROOT / 'results'
W, M = 2400, 36
BODY = 46
FIT_CHARS = 160                       # the fitted start's text is stored up to this character
FIT_LABEL = 'matched token pairs'
FIT_RULE = ('both tokens end at', 'the same character')


def last_step(path):
    return max(r['step'] for r in json.loads(path.read_text()))


def started_at(path, pattern):
    """The step a run starts from, as its chain script (or chain log) gives it."""
    return int(re.search(pattern, path.read_text()).group(1))


def refresh(replies_path, coread_path, nl40):
    sys.path.insert(0, str(ROOT / 'pairing'))
    import align
    import games
    from coread import request
    from link import MODELS, SIDES
    from transformers import AutoTokenizer

    curve = json.loads((R / 'training_curve.json').read_text())
    steps = dict(warm_up=last_step(R / 'magic-nl20/warm/log.json'), full_mix=curve['mix_steps'],
                 chat=last_step(R / 'magic-nl22c/log.json'), steered=last_step(R / 'magic-nl33t/log.json'),
                 rl=last_step(R / 'magic-beautyrl6/log.json'),
                 exam=max(int(v) for v in re.findall(r'qa_pair/step_(\d+)', (R / 'magic-nl23/qa_pair.log').read_text())),
                 private=last_step(R / 'magic-nl26/log.json'), steered40=last_step(Path(nl40) / 'results/log.json'),
                 same33w=last_step(R / 'magic-nl33w/log.json'))
    oneway = {x: last_step(R / f'magic-nl26/nl25/oneway_{x}/log.json') for x in SIDES}
    assert len(set(oneway.values())) == 1, oneway
    steps['oneway'] = oneway[SIDES[0]]
    assert steps['warm_up'] == curve['warm_up_steps']
    for path, pattern, parent in ((ROOT / 'chains/nl22.sh', r'INIT=\$J/step_(\d+)', 'full_mix'),
                                  (ROOT / 'chains/nl23.sh', r'--init \$J/step_(\d+)', 'full_mix'),
                                  (ROOT / 'chains/nl24.sh', r'INIT=\$J/step_(\d+)', 'chat'),
                                  (ROOT / 'chains/nl33.sh', r'--init \$J/step_(\d+)', 'chat'),
                                  (ROOT / 'chains/nl40.sh', r'--init \$J/step_(\d+)', 'chat'),
                                  (ROOT / 'chains/beautyrl.sh', r'INIT=\$J/step_(\d+)', 'steered'),
                                  (R / 'magic-nl26/nl25/chain.log', r'magic-nl26/run/step_(\d+)', 'private')):
        if started_at(path, pattern) != steps[parent]:
            raise ValueError(f'{path.name} starts from step {started_at(path, pattern)}, not {parent}\'s last step')
    chain20 = (ROOT / 'chains/nl20.sh').read_text()
    value_games = re.search(r'VALUES=(\S+)', chain20).group(1).split(',')
    texts = int(re.search(r"'--content_texts', type=int, default=(\d+)", (ROOT / 'pairing/train.py').read_text()).group(1))
    assert '--content_texts' not in chain20

    # the fitted start: the fit's texts rebuilt, checked against the recorded number of matched token pairs
    replies = json.loads(open(replies_path).read())
    held = set(json.load(open(ROOT / 'data/pools.json'))['questions']['test'])
    held |= {request(it)[2] for it in json.load(open(coread_path))['test'] if request(it)}
    align.held_out_prompts = lambda: held
    items = align.texts(replies, texts)
    toks = {x: AutoTokenizer.from_pretrained(MODELS[x]['hf'], local_files_only=True) for x in SIDES}
    recorded = json.loads((R / 'magic-nl20/warm/content_init.json').read_text())
    n_test = inspect.signature(align.content_init).parameters['n_test'].default
    pairs = 0
    for j, item in enumerate(items[:len(items) - n_test]):
        ends = set(align.sequence(toks[SIDES[0]], SIDES[0], item)[1]) & set(align.sequence(toks[SIDES[1]], SIDES[1], item)[1])
        pairs += len(ends) if len(ends) >= 2 else 0
    if {r['pairs'] for r in recorded} != {pairs}:
        raise ValueError(f'rebuilt fit texts give {pairs} matched pairs, content_init.json has '
                         f'{sorted({r["pairs"] for r in recorded})}')
    text = items[0][1][:align.MAX_CHARS]
    spans = {x: [[a, b] for a, b in toks[x](text, add_special_tokens=False, return_offsets_mapping=True)
                 ['offset_mapping'] if b > a and b <= FIT_CHARS] for x in SIDES}
    fit = dict(texts=texts, test_texts=n_test, pairs=pairs, role=items[0][2], text=text[:FIT_CHARS], tokens=spans)

    # the warm-up: the first word-game episode of the training split
    ep = games.Games(replies, 'train', seed=0).episode('relay_word')
    meta = ep['meta']
    warm = dict(sender=meta['sender'], receiver=meta['receiver'], value=meta['value'], question=meta['question'],
                send=ep['task'][meta['sender']], recv=ep['task'][meta['receiver']],
                target=ep['reply'][meta['receiver']][0])

    tree = ast.parse((ROOT / 'pairing/chat2.py').read_text())
    kinds = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'TRAIN_KINDS' for t in n.targets))
    shared = json.loads((R / 'shared_reading.json').read_text())['setup']['training']
    teacher = json.loads((R / 'teacher_method.json').read_text())['example']['conversation']
    snap = dict(steps=steps, value_games=value_games, private_kinds=list(kinds), fit=fit, warm=warm,
                mix=dict(context=shared['context'], question=shared['question'], answer=shared['answer']),
                chat=dict(private=teacher['private'], system=teacher['system'], opener=teacher['opener'],
                          first_turn=teacher['turns'][0]))
    SNAP.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + '\n')
    return snap


def plain(text):
    return [(text, INK, 400, None)]


def marked(text, span, color, fill):
    """runs for text with span bold in color on fill."""
    i = text.find(span)
    if i < 0:
        raise ValueError(f'{span!r} not in {text!r}')
    runs = [(text[:i], INK, 400, None)] if i else []
    runs.append((span, color, 700, fill))
    if text[i + len(span):]:
        runs.append((text[i + len(span):], INK, 400, None))
    return runs


def natural(runs, size=BODY, pad=30):
    return sum(width(t, size, wt) for t, _, wt, _ in runs) + 2 * pad + 4


def private_sentence(system, city):
    """The sentence of a system prompt that gives the twin's city."""
    part = system.split('):', 1)[1]
    return next(m.group(0).strip() for m in re.finditer(r'[^.]*\.', part) if city in m.group(0))


RB = 90                                          # robot size beside each prompt
GAP = 26                                         # between a robot and its prompt
ROW = 22                                         # gap between the rows of an example


def said(s, side, x, y, w, runs, draw=True):
    """A robot and a box of text to its right, at most w wide; returns (height, width)."""
    bw = min(w - RB - GAP, natural(runs))
    h = bubble_height(runs, bw, BODY)
    if draw:
        robot(s, side, x, y + h / 2 - RB / 2, RB, facing=1)
        bubble(s, x + RB + GAP, y, bw, runs, BODY)
    return max(h, RB), RB + GAP + bw


def two_rows(s, x, y, w, first, second, t_runs, label, frac=0.6, draw=True):
    """An example of two rows and a training target: the first row's twin (a prompt, or a tag when its text is a str),
    the second row's twin and prompt (at most frac of w wide), an arrow to the target box (outlined in black) beside the
    second row, and the link between the two twins. A short label goes above the arrow and the first row may span w; a
    long one goes above the target box, and the first row stays clear of it. Returns the height."""
    (s1, r1), (s2, r2) = first, second
    lab = [(label, INK, 700, None)]
    short = width(label, 46, 700) < 240
    w_first = w if short else w * frac
    if isinstance(r1, str):                     # a tag beside the twin, not a prompt
        h1, w1 = RB, RB + GAP + width(r1, BODY, 700) + 40
    else:
        h1, w1 = said(s, s1, x, y, w_first, r1, draw=False)
    y2 = y + h1 + ROW
    h2, w2 = said(s, s2, x, y2, w * frac, r2, draw=False)
    tx = x + (w2 + width(label, 46, 700) + 60 if short else max(w1, w2) + 80)
    tw = x + w - tx
    bw = min(tw, natural(t_runs))
    bh, lh = bubble_height(t_runs, bw, BODY), (0 if short else flow_height(lab, tw, 46) + 6)
    mid = max(y2 + max(h2, bh) / 2, y + lh + bh / 2)
    if draw:
        if isinstance(r1, str):
            robot(s, s1, x, y, RB, facing=1)
            s.rect(x + RB + GAP, y + RB / 2 - 38, w1 - RB - GAP, 76, TINT[s1], 16)
            s.text(x + RB + GAP + (w1 - RB - GAP) / 2, y + RB / 2 + 16, r1, BODY, 700, COLOR[s1], 'middle')
        else:
            said(s, s1, x, y, w_first, r1)
        said(s, s2, x, mid - h2 / 2, w * frac, r2)
        if short:
            s.text((x + w2 + tx) / 2, mid - 20, label, 46, 700, INK, 'middle')
        else:
            flow(s, tx, mid - bh / 2 - lh + 46 * 0.86, tw, lab, 46)
        bubble(s, tx, mid - bh / 2, bw, t_runs, BODY, stroke=INK, sw=5, tail=None)
        arrow(s, [(x + w2 + 12, mid), (tx - 12, mid)], INK, 5, 18)    # the second twin writes the target
        link_down(s, x - 36, y + h1 / 2, mid)
    return max(mid + h2 / 2, mid + bh / 2) - y


def ex_fit(s, x, y, w, snap, draw=True):
    """Both tokenizers' cuts of the same text, joined in green where both tokens end at the same character. Every token
    boundary of either cut opens a small gap, so the two rows line up character by character."""
    fit = snap['fit']
    text = fit['text']
    size, g = BODY, 12
    # one token per end character (the byte pieces of one character end together and draw as one)
    toks = {}
    for side, v in fit['tokens'].items():
        first = {}
        for a, b in v:
            first[b] = min(first.get(b, a), a)
        toks[side] = sorted((a, b) for b, a in first.items())
    cuts = sorted({c for v in toks.values() for t in v for c in t})
    tx = x + RB + GAP
    left = lambda a: tx + width(text[:a], size) + g * sum(c <= a for c in cuts)
    right = lambda b: tx + width(text[:b], size) + g * sum(c < b for c in cuts)
    ends = {side: {b for _, b in v} for side, v in toks.items()}
    room = x + w - max(width(t, 46, 700) for t in FIT_RULE) - 50
    last = max(b for b in ends['quill'] & ends['ember'] if right(b) <= room and text[b:b + 1] == ' ')    # a word end
    ch, gap = 72, 84
    rows = {'quill': y, 'ember': y + ch + gap}
    if draw:
        for side, ry in rows.items():
            robot(s, side, x, ry + ch / 2 - RB / 2, RB, facing=1)
            for a, b in toks[side]:
                if b > last:
                    continue
                s.rect(left(a), ry, right(b) - left(a), ch, TINT[side], 12, COLOR[side], 3)
                word = text[a:b]
                s.text((left(a) + right(b)) / 2, ry + ch / 2 + 16, word.strip(), size, 400, INK, 'middle')
        for b in sorted(ends['quill'] & ends['ember']):
            if b <= last:
                s.line(right(b) - 3, rows['quill'] + ch + 10, right(b) - 3, rows['ember'] - 10, LINK, 6)
        for k, line in enumerate(FIT_RULE):
            s.text(right(last) + 50, rows['quill'] + ch + gap / 2 - 12 + 56 * k, line, 46, 700, LINK)
    return 2 * ch + gap




def ex_warm(s, x, y, w, snap, draw=True):
    e = snap['warm']
    snd, rcv, v = e['sender'], e['receiver'], e['value']
    send = opening(e['send'], e['send'].find(e['question']) - 1)
    return two_rows(s, x, y, w, (snd, marked(send, f'"{v}"', COLOR[snd], TINT[snd])), (rcv, plain(e['recv'])),
                    marked(e['target'], v, COLOR[snd], TINT[snd]), 'target', draw=draw)


def ex_mix(s, x, y, w, snap, draw=True):
    parts = ['value games', 'shared reading', 'play', 'dialogue', 'silent rounds']
    cx = x
    if draw:
        for p in parts:
            cw = width(p, BODY, 700) + 40
            s.rect(cx, y, cw, 70, 'white', 16, LINK if p == 'shared reading' else 'none', 4)
            s.text(cx + cw / 2, y + 50, p, BODY, 700, INK, 'middle')
            cx += cw + 20
    m = snap['mix']
    return 70 + 28 + two_rows(s, x, y + 70 + 28, w, ('quill', plain(m['context'])), ('ember', plain(m['question'])),
                              plain(m['answer']), 'target', 0.66, draw=draw)


def ex_chat(s, x, y, w, snap, draw=True):
    c = snap['chat']
    first, other = c['first_turn'][0], 'quill' if c['first_turn'][0] == 'ember' else 'ember'
    p_first = private_sentence(c['system'][first], c['private'][first])
    p_other = private_sentence(c['system'][other], c['private'][other])
    return two_rows(s, x, y, w, (other, marked(p_other, c['private'][other], COLOR[other], TINT[other])),
                    (first, marked(p_first, c['private'][first], COLOR[first], TINT[first])),
                    marked(c['first_turn'][1], c['private'][other], COLOR[other], TINT[other]),
                    f'what original {NAME[first]} writes when given {NAME[other]}’s city', 0.5, draw=draw)


# (key, name, step key, example, level, parent): level 0 is the main line, 1 a branch from it, 2 a branch from parent
COMMON = (('fit', 'Fitted start', None, ex_fit, 0, None),
          ('warm_up', 'Warm-up: {n} value games', 'warm_up', ex_warm, 0, None),
          ('full_mix', 'Full mix of games', 'full_mix', ex_mix, 0, None),
          ('exam', 'Exam questions', 'exam', None, 1, None),
          ('chat', 'Chat training', 'chat', ex_chat, 0, None),
          ('private', '{k} kinds of private item', 'private', None, 1, None),
          ('oneway', 'One-way links', 'oneway', None, 2, 'private'))
FINAL = (('steered', 'Game with original Quill’s targets', 'steered', None, 1, None),
         ('rl', 'Judge-rewarded reinforcement learning', 'rl', None, 2, 'steered'),
         ('steered40', 'Game with the twins’ own targets', 'steered40', None, 1, None),
         ('same33w', 'Chat training continued, without the steered game', 'same33w', None, 0, None))
STAGES = COMMON + FINAL
LEVEL = {k: lvl for k, _, _, _, lvl, _ in STAGES if lvl}
PARENT = {k: par for k, _, _, _, _, par in STAGES if par}
NUMBER = dict(enumerate('zero one two three four five six seven eight nine ten'.split()))


def draw(snap):
    steps = snap['steps']
    main = [step_key for k, _, step_key, _, lvl, _ in STAGES if step_key and not lvl]
    title = (f'After a fitted start, the twins train in {NUMBER[len(main)]} stages for '
             f'{sum(steps[k] for k in main):,} steps')
    top = 130
    sx = M + 30
    card_x = M + 80
    left_w = 520
    pad = 28
    indent = 120
    ex_x = card_x + left_w + 50
    ex_w = W - M - pad - ex_x
    gap = 20
    names = {k: name.format(n=len(snap['value_games']), k=len(snap['private_kinds'])) for k, name, *_ in STAGES}

    def height(key, fn):
        if fn is None:
            return 110
        lines = len(layout([(names[key], INK, 700, None)], left_w - 2 * pad, 48))
        left = 72 + 63 * (lines - 1) + 116 + pad                       # to the bottom of the step count
        if key == 'fit':
            left += flow_height([(FIT_LABEL, LINK, 700, None)], left_w - pad, 46) - 16
        return max(left, fn(SVG(W, 100), ex_x, 0, ex_w, snap, draw=False) + 2 * pad)

    heights = {k: height(k, fn) for k, _, _, fn, _, _ in STAGES}
    H = round(top + sum(heights.values()) + gap * (len(STAGES) - 1) + 40 + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    s.text(M, 84, title, 56, 700)
    compact = [k for k, _, _, fn, _, _ in STAGES if fn is None]   # rows of a name and a step count only
    branch_x = max(card_x + indent * LEVEL.get(k, 0) + pad + width(names[k], 48, 700) for k in compact) + 50
    y = top
    nodes, x_of = [], {}
    for key, _, step_key, fn, _, _ in STAGES:
        h, name = heights[key], names[key]
        lvl = LEVEL.get(key, 0)
        x0 = x_of[key] = card_x + indent * lvl
        s.rect(x0, y, W - M - x0, h, COOL if lvl else WARM, 32)
        if lvl == 0:
            nodes.append(y + 62)
        elif lvl == 1:                          # a branch from the main line at this point
            s.line(sx, y + h / 2, x0 - 4, y + h / 2, INK, 5)
        else:                                   # a branch from the branch above
            px = x_of[PARENT[key]] + 40
            s.path(f'M{px},{y - gap} V{y + h / 2} H{x0 - 4}', INK, 5)
        lx = x0 + pad
        if fn is None:
            s.text(lx, y + h / 2 + 16, name, 48, 700)
            n = f'{steps[step_key]:,}'
            s.text(branch_x, y + h / 2 + 24, n, 72, 700)
            s.text(branch_x + width(n, 72, 700) + 14, y + h / 2 + 24, 'steps', 48, 700)
        else:
            lines = flow(s, lx, y + 72, left_w - 2 * pad, [(name, INK, 700, None)], 48)
            ny = y + 72 + 63 * (lines - 1)
            if step_key:
                n = f'{steps[step_key]:,}'
                s.text(lx, ny + 96, n, 72, 700)
                s.text(lx + width(n, 72, 700) + 14, ny + 96, 'steps', 48, 700)
            else:
                f = snap['fit']
                s.text(lx, ny + 96, f'{f["pairs"]:,}', 72, 700, LINK)
                flow(s, lx, ny + 150, left_w - pad, [(FIT_LABEL, LINK, 700, None)], 46)
            fn(s, ex_x, y + pad, ex_w, snap)
        y += h + gap
    arrow(s, [(sx, nodes[0]), (sx, y - gap + 30)], INK, 6, 24)
    for ny in nodes:
        s.circle(sx, ny, 16, INK)
    render(s, OUT)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--replies', help="nl20's replies.json (for --refresh)")
    ap.add_argument('--coread', help='data/coread.json (for --refresh)')
    ap.add_argument('--nl40', default=str(Path.home() / 'mpd-data/runpod/magic-nl40'), help="nl40's output directory")
    args = ap.parse_args()
    snap = refresh(args.replies, args.coread, args.nl40) if args.refresh else json.loads(SNAP.read_text())
    H = draw(snap)
    print(json.dumps(dict(height=H, steps=snap['steps'], fit_pairs=snap['fit']['pairs'])))


if __name__ == '__main__':
    main()
