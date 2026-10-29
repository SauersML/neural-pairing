"""How the twins are trained to stay themselves, and whether they do: real training episodes of the identity games, and
the final twins' answers linked and cut.

Training (top): episodes drawn by games.Games (training split, seed 0), the first of each kind: 'identity' with Ember
asked who it is while Quill answers its own question; 'identity' with Quill asked who its partner is; 'plain', each twin
writing its original model's reply to its own question; 'chat_identity' with Quill asked who it is after a chat. A grey
chip is a prompt, a white bubble a training target, written by the twin in its colour.

Results (bottom), from results/identity_check.json (pairing/identity_check.py on the final twins, nl33w step 300):
after each of 40 free linked conversations, the partner asks who the twin is, or who its partner is, in wordings never
seen in training; the twin answers linked and with the link cut. Bars, per twin and condition: the share of answers
with the twin's own name, and, asked about the partner, with the partner's name; and the share of never-seen questions
on which the twin's own original reply is more probable than the partner model's reply (one shown). identity.json
also keeps the answers with the twin's own maker (Alibaba or Qwen for Quill, Arcee or AFM for Ember).

Also counts, in every saved free conversation under results/, the turns in which a twin introduces itself as its
partner (evaluate.claims), linked and cut. Writes results/identity.json (the episodes drawn and every count) and
figures/identity.{svg,png,pdf}. --refresh redraws the episodes (needs the original models' replies, prep.py's
replies.json, and torch for pairing/games.py); without it the episodes come from results/identity.json.
Usage: python figures/identity.py [--refresh --replies REPLIES]
"""
import argparse
import json
import re
import sys

from chat_names import COOL, INK, ROOT, SVG, icon_link, width
from method_parts import COLOR, NAME, TINT, check_shape, chip, flow, flow_height, layout, render

OUT = ROOT / 'figures/identity'
RESULTS = ROOT / 'results/identity.json'
CHECK = ROOT / 'results/identity_check.json'
W, F, M = 2400, 46, 30
SIDES = ('quill', 'ember')
MAKER = {'quill': 'Alibaba', 'ember': 'Arcee AI'}
MODEL = {'quill': 'Qwen3-4B', 'ember': 'AFM-4.5B'}
GREY = '#E2E2E7'          # a prompt's box
CHIP = round(F * 1.6)     # a one-line prompt's height
GAP = 42                  # between a prompt and the training target below it
STAGGER = 0.62            # a dialogue turn starts this far down the previous turn (in the other column)
CONVERSATIONS = [  # saved free conversations: (file, key path to a list of conversations, condition, speakers given)
    ('results/magic-nl38/results/nl33w/rows.json', ('conversations',), 'link', False),
    ('results/magic-nl40/results/mood_new/nl33w/rows.json', ('conversations',), 'link', False),
    ('results/magic-beautyrl6/results/gen.json', ('init', 'link'), 'link', True),
    ('results/magic-beautyrl6/results/gen.json', ('init', 'cut'), 'cut', True),
    ('results/magic-beautyrl6/results/gen.json', ('rl', 'link'), 'link', True),
    ('results/magic-beautyrl6/results/gen.json', ('rl', 'cut'), 'cut', True),
]


def other(x):
    return SIDES[1 - SIDES.index(x)]


def claims(t, x):
    """pairing/evaluate.claims: whether text t introduces its writer as twin x."""
    return re.search(r"\b(i'?m|i am|my name is|this is)\s+" + NAME[x].lower(), t.lower()) is not None


def draw_episodes(replies):
    """The first training episode of each kind drawn (module docstring), as plain lists."""
    sys.path.insert(0, str(ROOT / 'pairing'))
    import games as G

    def first(game, ok):
        g = G.Games(replies, 'train', seed=0)
        for _ in range(10000):
            e = g.episode(game)
            if ok(e):
                return e
        raise RuntimeError(game)
    ident = lambda r, kind: (lambda e: e['meta']['receiver'] == r
                             and e['reply'][r][0].startswith("I'm" if kind == 'self' else 'My partner'))
    out = dict(system={x: G.system(x) for x in SIDES})
    for key, e in (('self', first('identity', ident('ember', 'self'))),
                   ('partner', first('identity', ident('quill', 'partner'))),
                   ('plain', first('plain', lambda e: True))):
        out[key] = {x: dict(asked=e['task'][x], target=e['reply'][x][0]) for x in SIDES}
    e = first('chat_identity', lambda e: e['meta']['receiver'] == 'quill' and e['meta']['kind'] == 'self'
              and e['meta']['exchanges'] == 1)
    out['chat'] = [[sp, t] for sp, t, _ in e['turns']]
    return out


def free_claims():
    """Turns that introduce the writer as its partner, in the saved free conversations of CONVERSATIONS."""
    rows = []
    for path, keys, cond, speakers in CONVERSATIONS:
        d = json.loads((ROOT / path).read_text())
        for k in keys:
            d = d[k]
        turns = ([tuple(t) for play in d for c in play for t in c] if speakers
                 else [(SIDES[j % 2], t) for c in d for j, t in enumerate(c)])          # Quill speaks first
        rows.append(dict(file=path, part='/'.join(keys), condition=cond, turns=len(turns),
                         claims_partner=sum(claims(t, other(x)) for x, t in turns)))
    return rows


def trim(text, w, lines, size=F):
    """text cut to its first words that fit in the given lines of width w, with '…' when cut."""
    words = text.split()
    for n in range(len(words), 0, -1):
        t = ' '.join(words[:n])
        if n < len(words):
            t = t.rstrip('.,;:') + '…'
        if len(layout([(t, INK, 400, None)], w, size)) <= lines:
            return t
    raise ValueError(text)


def runs_for(text):
    """The text with each twin's name and maker bold in that twin's colour on its tint."""
    spans = [(s, y) for y in SIDES for s in (NAME[y], MAKER[y]) if s in text]
    spans.sort(key=lambda p: text.index(p[0]))
    runs, rest = [], text
    for s, y in spans:
        i = rest.find(s)
        if i < 0:
            continue
        if i:
            runs.append((rest[:i], INK, 400, None))
        runs.append((s, COLOR[y], 700, TINT[y]))
        rest = rest[i + len(s):]
    return runs + ([(rest, INK, 400, None)] if rest else [])


def bubble_runs(text, w, lines):
    return runs_for(trim(text, w - 60, lines))


def target_height(runs, w):
    return flow_height(runs, w - 60) + 60 - (round(F * 1.32) - F * 1.05)


def target(s, x, x0, y, w, runs):
    """A training target or an answer: a white bubble with the twin's border. Returns its height."""
    h = target_height(runs, w)
    s.rect(x0, y, w, h, 'white', 24, COLOR[x], 4)
    flow(s, x0 + 30, y + 30 + F * 0.84, w - 60, runs)
    return h


def original(t):
    """Whether a duplex training target is the original model's reply (not an identity answer)."""
    return not t.startswith(("I'm ", 'My partner'))


def badge(s, x, right, y):
    """A tag on a bubble's top edge, right-aligned at right: the reply is the original model's."""
    t = MODEL[x] + '’s reply'
    w = width(t, 42, 700) + 36
    s.rect(right - w, y - 30, w, 58, TINT[x], 29, COLOR[x], 4)
    s.text(right - w / 2, y + 14, t, 42, 700, COLOR[x], 'middle')


def asked(s, x0, y, w, text):
    """A prompt: a grey chip."""
    chip(s, x0, y, trim(text, w - 40, 1), INK, GREY, size=F, h=CHIP, pad=20)


def draw(r, stem):
    cw = (W - 2 * M - 30) / 2                 # one twin's column
    cx = {'quill': M, 'ember': M + cw + 30}
    pad, inner = 26, (W - 2 * M - 30) / 2 - 2 * 26
    s = SVG(W, 0)
    # the twins over their columns, linked
    y = M
    s.text(M, y + 79, 'Training', 52, 700)
    size = 170
    ix = {x: cx[x] + cw / 2 - size / 2 for x in SIDES}
    icon_link(s, ix['quill'], ix['ember'], y, size)
    for x in SIDES:
        s.mascot(x, ix[x], y, size)
        side = -1 if x == 'quill' else 1
        tx = ix[x] + size / 2 + side * (size * 0.24 + 24)
        anchor = 'end' if x == 'quill' else 'start'
        s.text(tx, y + size / 2 - 6, NAME[x], 56, 700, COLOR[x], anchor)
        s.text(tx, y + size / 2 + 52, MODEL[x], F, 400, COLOR[x], anchor)
    y += size + 24
    ep = r['episodes']
    hp = 0
    for x in SIDES:                           # every episode's system prompt
        hp = max(hp, question(s, cx[x] + pad, y, inner, trim(ep['system'][x], inner - 40, 2)))
    y += hp + 22
    for key in ('self', 'partner', 'plain'):
        runs = {x: bubble_runs(t, inner, 1 if original(t) else 2)
                for x, t in ((x, ep[key][x]['target']) for x in SIDES)}
        h = CHIP + GAP + max(target_height(runs[x], inner) for x in SIDES) + 2 * pad
        s.rect(M, y, W - 2 * M, h, COOL, 22)
        for x in SIDES:
            asked(s, cx[x] + pad, y + pad, inner, ep[key][x]['asked'])
            target(s, x, cx[x] + pad, y + pad + CHIP + GAP, inner, runs[x])
            if original(ep[key][x]['target']):
                badge(s, x, cx[x] + pad + inner - 24, y + pad + CHIP + GAP)
        y += h + 22
    # a chat, then the identity question: each turn in its writer's column, in order
    turns = [(sp, bubble_runs(t, inner, 1)) for sp, t in ep['chat']]
    hs = [target_height(rr, inner) for _, rr in turns]
    h = sum(STAGGER * v for v in hs[:-1]) + hs[-1] + 2 * pad
    s.rect(M, y, W - 2 * M, h, COOL, 22)
    yy = y + pad
    for sp, rr in turns:
        yy += STAGGER * target(s, sp, cx[sp] + pad, yy, inner, rr)
    y += h + 50
    y = results(s, r, y, cx, pad, inner)
    H = round(y + M)
    check_shape(W, H)
    out = SVG(W, H)
    out.parts += s.parts[2:]
    render(out, stem)
    return W, H


def question(s, x, y, w, text):
    """A prompt as a grey box, its text wrapped into width w. Returns its height."""
    runs = [(text, INK, 400, None)]
    lines = layout(runs, w - 40, F)
    cw = max(dx + width(word, F) for line in lines for dx, word, *_ in line[-1:]) + 40
    h = len(lines) * round(F * 1.32) + 26
    s.rect(x, y, min(cw, w), h, GREY, 16)
    flow(s, x + 20, y + 13 + F * 0.9, w - 40, runs)
    return h


ROWS = (('self', 'own_name', 'Says its own name'), ('partner', 'partner_name', 'Names its partner'),
        ('voice', 'own_more_probable', 'Its own model’s reply is more likely than its partner model’s'))
BAR_H, BAR_GAP = 62, 18


def bar(s, x0, y, w, label, frac, color, opacity):
    """One labelled bar, its top at y and spanning width w: the label, the bar on a grey track, the percent."""
    lw, pw = width('Link cut', F, 700) + 26, width('100%', 56, 700) + 22
    s.text(x0, y + BAR_H / 2 + F * 0.36, label, F, 700)
    s.rect(x0 + lw, y, w - lw - pw, BAR_H, GREY, BAR_H / 2)
    if frac:
        s.rect(x0 + lw, y, max(BAR_H, frac * (w - lw - pw)), BAR_H, color, BAR_H / 2, extra=f'opacity="{opacity}"')
    s.text(x0 + w, y + BAR_H / 2 + 20, f'{round(100 * frac)}%', 56, 700, color, 'end')


def results(s, r, y, cx, pad, inner):
    """Per measure, a band: its label and the never-seen questions, then each twin's share with the link on and cut,
    in the twin's column."""
    c = r['check']['summary']
    s.text(M, y + 44, 'After training', 52, 700)
    y += 44 + 30
    asks = dict(r['check']['questions'], voice=[r['check']['voice_example']])
    for kind, m, label in ROWS:
        h = pad + 64 + CHIP + 34 + 2 * BAR_H + BAR_GAP + pad
        s.rect(M, y, W - 2 * M, h, COOL, 22)
        s.text(M + pad, y + pad + 44, label, 52, 700)
        x = M + pad
        for q in asks[kind]:
            x += chip(s, x, y + pad + 64, q, INK, GREY, size=F, h=CHIP, pad=20) + 18
        yy = y + pad + 64 + CHIP + 34
        for j, (cond, word, opacity) in enumerate((('link', 'Link on', 1), ('cut', 'Link cut', 0.45))):
            for tw in SIDES:
                v = c[f'{cond}|{tw}'][kind]
                bar(s, cx[tw] + pad, yy + j * (BAR_H + BAR_GAP), inner, word, v[m] / v['n'], COLOR[tw], opacity)
        y += h + 22
    return y - 22


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--replies')
    args = ap.parse_args()
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if args.refresh:
        episodes = draw_episodes(json.loads(open(args.replies).read()))
    else:
        episodes = old['episodes']
    check = json.loads(CHECK.read_text())
    r = dict(episodes=episodes, check={k: check[k] for k in ('run', 'rows', 'backend', 'questions', 'summary')},
             free_claims=free_claims())
    r['check']['voice_example'] = check['voice'][0]['question']
    RESULTS.write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(summary=check['summary'], free_claims=r['free_claims']), indent=1))
    print(draw(r, OUT))


if __name__ == '__main__':
    main()
