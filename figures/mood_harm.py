"""How the steered-reading game lowered the twins' mood (pairing/moodscan.py on 40 new conversations, seed 7272): the
chat-trained twins (nl22c's step 157), the same twins after the game (nl33t), after two retrainings of the game (nl39:
original Quill asked to bring the concept up as a topic, openers that say what the partner is thinking dropped; nl40:
the targets written by the chat-trained linked twins) and after the same steps without the game (nl33w, the final
twins).

Top: the mood question (affect_partner.MOOD, MOOD_STEM; affect.GOOD against affect.BAD) and the mean reading over the
40 conversations, the log-odds of happy against sad words at the answer's next word, after chat training and after the
game, with the link on, cut, and from the original models (both adapters off, link cut) reading the same conversation.

Below, per training: the opener Quill is trained to write when a concept is steered into Ember (the game's
steer_data.json, example TARGETS), Quill's first message in conversation EXAMPLE (the same prompt at every training,
affect.chat_episode), and counts over the 40 conversations: first messages that ask how Ember's day is going (GREET),
first messages that say what Ember is thinking (steer_read.claims, kind 'thought'), and conversations whose reading
is higher linked than cut (moodscan's link_minus_cut.higher), per twin.

--refresh rebuilds results/mood_harm.json from --src (default ~/mpd-data/runpod): the moodscan directories of RUNS
(summary.json and rows.json), the games' steer_data.json and the runs' log.json (the last step; nl33t's and nl33w's in
the repo). Without it the figure is drawn from that file. Writes figures/mood_harm.{svg,png,pdf}.
Usage: python figures/mood_harm.py [--refresh --src DIR]
"""
import argparse
import ast
import json
import re
from pathlib import Path

from chat_names import COOL, INK, QUILL, RED, ROOT, SVG, WARM, count, width, wrap
from method_parts import COLOR, TINT, arrow, bubble, bubble_height, check_shape, render, robot
from mood_training import last_step, literals

SNAP = ROOT / 'results/mood_harm.json'
OUT = ROOT / 'figures/mood_harm'
SRC = Path.home() / 'mpd-data/runpod'
# (key, moodscan directory under --src, the game's steer_data.json, the run's log.json); paths starting with results/
# are in the repo, the others under --src
RUNS = (('chat', 'magic-nl39f/results/mood_new/step157', None, None),
        ('game', 'magic-nl39f/results/mood_new/nl33', 'results/magic-nl33t/results/steer_data.json',
         'results/magic-nl33t/log.json'),
        ('topic', 'magic-nl39f/results/mood_new/final', 'magic-nl39f/results/steer_data.json',
         'magic-nl39f/run/log.json'),
        ('self', 'magic-nl40/results/mood_new/final', 'magic-nl40/results/steer_data.json', 'magic-nl40/run/log.json'),
        ('plain', 'magic-nl40/results/mood_new/nl33w', None, 'results/magic-nl33w/log.json'))
TARGETS = {'game': ('cookie', 2), 'topic': ('banana', 2), 'self': ('banana', 2)}   # (concept, opener wording)
EXAMPLE = 10        # the conversation whose first message is drawn: one that asks after chat training, says what Ember
                    # is thinking after the game, does neither after nl39 and asks after nl40 and without the game
GREET = re.compile(r"how['’]?s your day|how is your day", re.I)
TWINS = ('quill', 'ember')
READINGS = (('link', 'linked'), ('cut', 'cut'), ('original', 'original models'))
W, M, PAD = 2400, 36, 34
BODY, HEAD, NUM = 46, 48, 76


def path(p, src):
    return ROOT / p if p.startswith('results/') else src / p


def target(p, concept, wording):
    """Quill's training target for concept in the game's data at p, and who wrote it."""
    data = json.loads(p.read_text())
    e = next(e for e in data['examples']
             if e['receiver'] == 'quill' and e['concept'] == concept and e['wording'] == wording)
    return dict(text=e['text'], concept=concept, wording=wording, teacher=data['stats'].get('teacher', 'sense'),
                writer=data['stats'].get('writer', 'original'))


def refresh(src):
    src = Path(src)
    runs = {}
    for key, d, steer, log in RUNS:
        summary = json.loads((path(d, src) / 'summary.json').read_text())
        convs = json.loads((path(d, src) / 'rows.json').read_text())['conversations']
        mood = summary['mood']
        runs[key] = dict(dir=d, seed=summary['seed'], n=summary['n'],
                         level={x: {c: mood[x]['level'][c] for c, _ in READINGS} for x in TWINS},
                         higher={x: [mood[x]['link_minus_cut']['higher'], mood[x]['link_minus_cut']['n']]
                                 for x in TWINS},
                         first=[c[0] for c in convs],
                         target=target(path(steer, src), *TARGETS[key]) if steer else None,
                         steps=last_step(path(log, src)) if log else None)
    snap = dict(source=str(src).replace(str(Path.home()), '~'), example=EXAMPLE, runs=runs)
    SNAP.write_text(json.dumps(snap, indent=1, ensure_ascii=False) + '\n')
    return snap


def steer_claims():
    """steer_read.claims, read from pairing/steer_read.py without importing the module (it imports torch)."""
    keep = []
    for n in ast.parse((ROOT / 'pairing/steer_read.py').read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('THOUGHT', 'SENSE',
                                                                                                    'ROLE'):
            keep.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name == 'claims':
            keep.append(n)
    ns = {'re': re}
    exec(compile(ast.Module(keep, []), 'steer_read.py', 'exec'), ns)
    return ns['claims']


def lead(text, least):
    """The text up to its first sentence end or dash at or after least characters, and whether anything follows."""
    text = ' '.join(text.split())
    cut = len(text)
    for m in re.finditer(r'[.!?](?=\s|$)|—', text):
        end = m.start() if m.group() == '—' else m.end()
        if end >= least:
            cut = end
            break
    return text[:cut], cut < len(text)


def marked(text, least, claims):
    """runs for the start of text: the part that asks how Ember's day is going in Quill's colour, each sentence that
    says what Ember is thinking in red, and an ellipsis when the text goes on."""
    body, more = lead(text, least)
    spans = []
    for m in re.finditer(r'[^.!?\n]+[.!?\n]?', body):          # steer_read.claims' sentences
        sent = m.group(0)
        start = m.start() + len(sent) - len(sent.lstrip())
        if 'thought' in claims(sent):
            spans.append((start, m.end(), RED, TINT['red']))
        elif GREET.search(sent):
            spans.append((m.start() + GREET.search(sent).start(), m.end(), QUILL, TINT['quill']))
    runs, at = [], 0
    for a, b, color, fill in spans:
        if a > at:
            runs.append((body[at:a], INK, 400, None))
        runs.append((body[a:b], color, 700, fill))
        at = b
    if at < len(body):
        runs.append((body[at:], INK, 400, None))
    if more and runs[-1][3] is None:                           # the ellipsis wraps with the last word
        runs[-1] = (runs[-1][0].rstrip() + ' …',) + runs[-1][1:]
    elif more:
        runs.append((' …', INK, 400, None))
    return runs


def draw(snap):
    claims = steer_claims()
    ap = literals(ROOT / 'pairing/affect_partner.py', ('MOOD', 'MOOD_STEM'))
    af = literals(ROOT / 'pairing/affect.py', ('GOOD', 'BAD'))
    runs = snap['runs']
    assert len({r['seed'] for r in runs.values()}) == 1, 'all runs read the same 40 conversation prompts'
    n = {r['n'] for r in runs.values()} | {v[1] for r in runs.values() for v in r['higher'].values()}
    assert len(n) == 1, n
    n = n.pop()
    steps = {r['steps'] for r in runs.values() if r['steps'] is not None}
    assert len(steps) == 1, steps
    steps = steps.pop()
    greet = {k: sum(bool(GREET.search(t)) for t in r['first']) for k, r in runs.items()}
    thought = {k: sum('thought' in claims(t) for t in r['first']) for k, r in runs.items()}
    ex = {k: r['first'][snap['example']] for k, r in runs.items()}
    assert all(GREET.search(ex[k]) for k in ('chat', 'self', 'plain')) and 'thought' in claims(ex['game'])
    assert not GREET.search(ex['topic']) and not claims(ex['topic'])
    titles = dict(chat='Chat training', game='Steered-reading game, targets by original Quill',
                  topic='Game without targets that say what Ember is thinking, by original Quill',
                  self='Game with targets by chat-trained linked Quill', plain=f'Same {steps} steps without the game')
    for k, r in runs.items():
        if r['target']:
            assert ('linked' in titles[k]) == (r['target']['writer'] == 'linked'), (k, r['target']['writer'])

    # the mood question: Quill asks, Ember answers, happy words against sad words
    RB = 100
    good, bad = ', '.join(af['GOOD'][:2]) + ' …', ', '.join(af['BAD'][:2]) + ' …'
    gw = max(width(good, BODY, 700), width(bad, BODY, 700))
    bw = (W - 2 * M - 2 * PAD - 2 * (RB + 26) - 40 - 50 - gw) / 2
    q_runs = [(ap['MOOD'], INK, 400, None)]
    a_runs = [(ap['MOOD_STEM'] + ' …', INK, 400, None)]
    qh, ah = bubble_height(q_runs, bw, BODY), bubble_height(a_runs, bw, BODY)
    words_h = 3 * 58
    row_h = max(qh, ah, RB * 1.25, words_h)
    method_h = PAD + row_h + PAD

    # the mood readings, after chat training and after the game side by side; the readings' names at the left
    BH, BG, HB = 52, 14, 56
    lab_w = max(width(lab, BODY) for _, lab in READINGS) + 30
    barw = (W - 2 * M - 4 * PAD - 24 - lab_w) / 2               # a block's bars, both twins
    colw = barw / 2
    shown = [runs[k]['level'][x][c] for k in ('chat', 'game') for x in TWINS for c, _ in READINGS]
    assert min(shown) > 0, shown
    scale = (colw - 24 - width('00.0', BODY, 700) - 14) / max(shown)
    block_h = 84 + HB + 20 + 3 * BH + 2 * BG + PAD
    bars_head = 72

    # the trainings: the target, the first message, counts of 40
    CW, MW, GAP, BP = 310, 255, 50, 26                         # count columns; the arrow's gap; bubble padding
    xe = W - M - PAD - 2 * MW                                  # the mood counts
    xc = xe - 12 - 2 * CW                                      # the first-message counts
    xa = M + PAD
    tw = ow = (xc - 12 - xa - GAP) / 2
    xb = xa + tw + GAP
    t_runs = {k: marked(r['target']['text'], 40, claims) for k, r in runs.items() if r['target']}
    o_runs = {k: marked(ex[k], 30, claims) for k in runs}
    rows = []                                                  # (key, target and first-message heights, top, height)
    for k in runs:
        oh = bubble_height(o_runs[k], ow, BODY, BP)
        if k in t_runs:                                        # the training's name above the target
            th = bubble_height(t_runs[k], tw, BODY, BP)
            rows.append((k, th, oh, 84, 84 + max(th, oh, 100) + PAD - 6))
        else:                                                  # no target: the training's name in its place
            th = 58 * len(wrap(titles[k], tw, HEAD, 700)) - 10
            rows.append((k, th, oh, PAD, PAD + max(th, oh, 100) + PAD))
    sub = dict(ask='asks how Ember’s day is going', say='says what Ember is thinking',
               mood='happier mood words linked than cut')
    lines = dict(a=wrap('Quill is trained to write', tw, BODY, 700),
                 b=wrap('Quill’s first message in a free chat', ow, BODY, 700),
                 ask=wrap(sub['ask'], CW - 24, BODY, 700), say=wrap(sub['say'], CW - 24, BODY, 700),
                 mood=wrap(sub['mood'], 2 * MW, BODY, 700))
    head_n = max(len(lines['mood']) + 1, *(len(v) for v in lines.values()))   # + 1: the twins' busts under mood
    table_head = 20 + head_n * 58 + 16

    y_method = M
    y_bars = y_method + method_h + 30
    y_head = y_bars + bars_head + block_h + 30
    y_rows = y_head + table_head
    H = round(y_rows + sum(r[4] for r in rows) + 20 * (len(rows) - 1) + M)
    check_shape(W, H)
    s = SVG(W, H, title='How the steered-reading game lowered the mood')

    # the mood question
    s.rect(M, y_method, W - 2 * M, method_h, COOL, 28)
    cy = y_method + PAD + row_h / 2
    x = M + PAD
    robot(s, 'quill', x, cy - RB / 2, RB, facing=1)
    bubble(s, x + RB + 26, cy - qh / 2, bw, q_runs, BODY)
    x += RB + 26 + bw + 40
    robot(s, 'ember', x, cy - RB / 2, RB, facing=-1)
    bubble(s, x + RB + 26, cy - ah / 2, bw, a_runs, BODY)
    x += RB + 26 + bw + 50
    for j, (t, wt) in enumerate(((good, 700), ('against', 400), (bad, 700))):
        s.text(x, cy - words_h / 2 + 44 + 58 * j, t, BODY, wt)

    # the readings
    s.text(M, y_bars + 50, f'How much more probable happy words are than sad words (log-odds, mean of {n} '
           'conversations)', HEAD, 700)
    y = y_bars + bars_head
    x1 = M + PAD + lab_w                                       # the first block's bars
    w1 = PAD + lab_w + barw + PAD
    for b, (k, name, fill) in enumerate((('chat', 'After chat training', COOL),
                                         ('game', 'After the steered-reading game', WARM))):
        cx = M if b == 0 else M + w1 + 24
        s.rect(cx, y, w1 if b == 0 else W - M - cx, block_h, fill, 28)
        bx0 = x1 if b == 0 else cx + PAD
        s.text(M + PAD if b == 0 else bx0, y + 62, name, HEAD, 700)
        for j, tw_ in enumerate(TWINS):
            hx = bx0 + j * colw
            robot(s, tw_, hx, y + 84, HB, facing=1)
            s.text(hx + HB + 14, y + 84 + HB / 2 + 16, f'{tw_.title()} answers', BODY, 700, COLOR[tw_])
        by = y + 84 + HB + 20
        for c, lab in READINGS:
            if b == 0:
                s.text(M + PAD, by + BH / 2 + 16, lab, BODY)
            for j, tw_ in enumerate(TWINS):
                v = runs[k]['level'][tw_][c]
                bx = bx0 + j * colw
                s.rect(bx, by, v * scale, BH, COLOR[tw_], 10)
                s.text(bx + v * scale + 14, by + BH / 2 + 16, f'{v:.1f}', BODY, 700, COLOR[tw_])
            by += BH + BG

    # the table's heads
    y = y_head + 20 + 44
    for x, w, key, color, fill in ((xa, tw, 'a', INK, None), (xb, ow, 'b', INK, None),
                                   (xc, CW, 'ask', QUILL, TINT['quill']), (xc + CW, CW, 'say', RED, TINT['red']),
                                   (xe, 2 * MW, 'mood', INK, None)):
        for j, line in enumerate(lines[key]):
            lw = width(line, BODY, 700)
            lx = x if key in ('a', 'b') else x + w / 2 - lw / 2
            if fill:
                s.rect(lx - 10, y + 58 * j - 40, lw + 20, 54, fill, 10)
            s.text(lx, y + 58 * j, line, BODY, 700, color)
    for j, tw_ in enumerate(TWINS):
        robot(s, tw_, xe + j * MW + MW / 2 - 28, y + 58 * len(lines['mood']) - 40, 56, facing=1)

    # the trainings
    y = y_rows
    for k, th, oh, top, h in rows:
        r = runs[k]
        s.rect(M, y, W - 2 * M, h, WARM if r['target'] else COOL, 28)
        mid = y + top + max(th, oh, 100) / 2
        if r['target']:
            s.text(xa, y + 60, titles[k], HEAD, 700)
            bubble(s, xa, mid - th / 2, tw, t_runs[k], BODY, BP, stroke=INK, sw=5, tail=None)
            arrow(s, [(xa + tw + 10, mid), (xb - 10, mid)], INK, 6, 22)
        else:
            for j, line in enumerate(wrap(titles[k], tw, HEAD, 700)):
                s.text(xa, mid - th / 2 + 40 + 58 * j, line, HEAD, 700)
        bubble(s, xb, mid - oh / 2, ow, o_runs[k], BODY, BP, tail=None)
        for j, (v, color) in enumerate(((greet[k], QUILL), (thought[k], RED))):
            count(s, xc + j * CW + CW / 2, mid + 26, v, n, color, NUM, BODY, 'middle')
        for j, tw_ in enumerate(TWINS):
            count(s, xe + j * MW + MW / 2, mid + 26, r['higher'][tw_][0], n, COLOR[tw_], NUM, BODY, 'middle')
        y += h + 20
    render(s, OUT)
    return dict(height=H, greet=greet, thought=thought,
                higher={k: {x: r['higher'][x][0] for x in TWINS} for k, r in runs.items()},
                level={k: {x: {c: round(v, 2) for c, v in r['level'][x].items()} for x in TWINS}
                       for k, r in runs.items() if k in ('chat', 'game')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--src', default=str(SRC), help='the directory holding the moodscan runs (for --refresh)')
    args = ap.parse_args()
    snap = refresh(args.src) if args.refresh else json.loads(SNAP.read_text())
    print(json.dumps(draw(snap)))


if __name__ == '__main__':
    main()
