"""The open-ended, multi-turn training data (swarm/opendata.py) drawn white on black: four real episodes from the first
eval shard (eval_seen-0000, Qwen3-1.7B teacher) and, over the shard's 600 episodes, the episodes per family and the
percent of user turns that copy an earlier teacher answer or repeat an earlier user turn.

An episode splits one real item between agents A and B. At each task turn one agent gets a user message and answers;
a frozen Qwen3-1.7B that sees both halves writes the answer the agents are trained to match, token by token.

python figures/opendata_examples.py --snapshot SHARD   writes results/opendata_examples.json; python
figures/opendata_examples.py draws figures/black/opendata_examples.{svg,png,pdf} from it.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                          # noqa: E402
from crumbs import DIM, GOLD, pct                                                 # noqa: E402

SNAPSHOT = ROOT / 'results/opendata_examples.json'
PICK = (('055946cdb6c2ea81', 'A real chat (WildChat, Portuguese) with the user turns dealt between A and B'),
        ('d32f4f4efb296ef0', 'Debugging: A holds the code, B the failing tests'),
        ('90b95dc3d7f150b4', 'A holds a Russian article; B asks about it in English'),
        ('544257ae6df194b1', 'Fill-in: each asks for one of the partner’s text chunks'))
FAMILY = dict(chat_alt='real chat, turns split', doc='document vs requests', fill='fill-in of chunks',
              pair='two documents', chat_doc='pasted material vs requests', code='code vs requests',
              plan='plan, constraints split', debug='code vs failing tests')


def grams(t, n=12):
    w = re.findall(r'\w+', t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def degenerate(r):
    """Per user turn: True when the turn shares a 12-word run with an earlier teacher answer or repeats an earlier
    user turn."""
    out, prev_r, prev_u = [], [], []
    for t in r['turns']:
        u = t['user'] or ''
        out.append(any(grams(u) & grams(p) for p in prev_r) or u.strip() in prev_u)
        prev_r.append(t['response'] or '')
        prev_u.append(u.strip())
    return out


def snapshot(shard):
    R = [json.loads(line) for line in gzip.open(shard, 'rt')]
    fam = collections.Counter(r['family'] for r in R)
    bad = collections.Counter()
    turns = collections.Counter()
    for r in R:
        d = degenerate(r)
        bad[r['family']] += sum(d)
        turns[r['family']] += len(d)
    ex = {r['id']: dict(family=r['family'], source=r['source'], teacher=r['teacher'], views=r['views'],
                        turns=r['turns'][:2])
          for r in R if r['id'] in dict(PICK)}
    d = dict(shard=str(shard), episodes=len(R), families=dict(fam), bad=dict(bad), turns=dict(turns),
             sources=dict(collections.Counter(r['source'] for r in R)),
             turns_per_episode=dict(collections.Counter(len(r['turns']) for r in R)), examples=ex)
    SNAPSHOT.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return d


def clean(t):
    """Markdown marks, rules and emoji (not in the figure font) removed; whitespace runs as one space."""
    t = re.sub(r'\*\*|__|^#+ |-{3,}', '', t or '', flags=re.M)
    t = ''.join(c for c in t if ord(c) < 0x2190 or 0x3000 <= ord(c) < 0xE000)
    return re.sub(r'\s+', ' ', t).strip()


def block(s, x, y, w, label, color, text, size, lines_max):
    """A labelled text block, at most lines_max lines; returns the y below it."""
    s.text(x, y + size, label, size, 700, color)
    y += size * 1.35
    lines = wrap(clean(text), w, size, 400)
    if len(lines) > lines_max:
        lines = lines[:lines_max]
        lines[-1] = lines[-1] + ' …'
    for k, ln in enumerate(lines):
        s.text(x, y + size + k * size * 1.28, ln, size, 400, TEXT)
    return y + len(lines) * size * 1.28 + 24


def episode_card(s, x, y, w, ep, head):
    pad, size = 34, 38
    iw = w - 2 * pad
    yy = y + pad
    for k, ln in enumerate(wrap(head, iw, 44, 700)):
        s.text(x + pad, yy + 44 + k * 54, ln, 44, 700, GOLD)
    yy += 54 * len(wrap(head, iw, 44, 700)) + 20
    t0 = ep['turns'][0]
    asker = t0['who']
    other = 'B' if asker == 'A' else 'A'
    col = dict(A=QUILL, B=EMBER)
    if ep['family'] != 'chat_alt':
        yy = block(s, x + pad, yy, iw, f'{other} sees', col[other], ep['views'][other], size, 3)
        if ep['views'][asker]:
            yy = block(s, x + pad, yy, iw, f'{asker} sees', col[asker], ep['views'][asker], size, 2)
    else:
        t1 = ep['turns'][1]
        yy = block(s, x + pad, yy, iw, f'user to {asker}, turn 1', col[asker], t0['user'], size, 2)
        yy = block(s, x + pad, yy, iw, f'teacher’s answer to {asker}', LINK, t0['response'], size, 2)
        yy = block(s, x + pad, yy, iw, f'user to {t1["who"]}, turn 2', col[t1['who']], t1['user'], size, 2)
        yy = block(s, x + pad, yy, iw, f'teacher’s answer to {t1["who"]}, the target for {t1["who"]}', LINK,
                   t1['response'], size, 2)
        return yy + pad - 24
    yy = block(s, x + pad, yy, iw, f'user to {asker}', col[asker], t0['user'], size, 2)
    target = (f'teacher’s answer, the target for {asker}' if ep['teacher'] else
              f'target for {asker}: the real text of {other}’s chunk {t0["user"].strip()}')
    yy = block(s, x + pad, yy, iw, target, LINK, t0['response'], size, 3)
    return yy + pad - 24


def bars(s, x, y, w, rows, fmt, color, size=40):
    """Horizontal bars: rows [(label, value, frac)]; returns the y below them."""
    lab = max(width(r[0], size) for r in rows) + 30
    full = w - lab - 140
    for label, v, f in rows:
        s.text(x + lab - 30, y + 38, label, size, 400, TEXT, 'end')
        s.rect(x + lab, y + 6, max(6, f * full), 40, color, 6)
        s.text(x + lab + max(6, f * full) + 18, y + 40, fmt(v), size, 700, color)
        y += 60
    return y


def figure(d, name='opendata_examples'):
    W = 2600
    s = canvas(W)
    y = title(s, 'Open-ended training data: real chats, documents and code split between agents A and B; a frozen '
                 'Qwen3-1.7B that sees both halves writes the answers to match', W)
    gap = 40
    cw = (W - 2 * M - gap) / 2
    ex = d['examples']
    hs = []
    for k, (eid, head) in enumerate(PICK):
        hs.append((k, eid, head))
    for row in range(2):
        tops = []
        for col in range(2):
            k, eid, head = hs[2 * row + col]
            x = M + col * (cw + gap)
            from chat_names import SVG
            probe = SVG(10, 10)
            tops.append(episode_card(probe, x, y, cw, ex[eid], head))
        bottom = max(tops)
        for col in range(2):
            k, eid, head = hs[2 * row + col]
            x = M + col * (cw + gap)
            s.rect(x, y, cw, bottom - y, CARD, 26)
            episode_card(s, x, y, cw, ex[eid], head)
        y = bottom + gap

    # the shard in numbers
    y += 30
    fam = sorted(d['families'].items(), key=lambda kv: -kv[1])
    s.text(M, y + 46, f'Episodes per family, {d["episodes"]} test episodes', 46, 700, TEXT)
    top = max(v for _, v in fam)
    yb = bars(s, M, y + 80, cw, [(FAMILY[f], v, v / top) for f, v in fam], str, LINK)
    x2 = M + cw + gap
    s.text(x2, y + 46, 'User turns copying an earlier answer or request', 46, 700, TEXT)
    rows = [(FAMILY[f], d['bad'][f] / d['turns'][f], d['bad'][f] / d['turns'][f]) for f, _ in fam if f != 'fill']
    rows.sort(key=lambda r: -r[1])
    yb2 = bars(s, x2, y + 80, cw, rows, pct, GOLD)
    y = max(yb, yb2) + 20
    foot = ('Fill-in turns are fixed requests (“[2]”) and are left out of the right chart. User turns are real in '
            'chats; elsewhere the frozen Qwen3-1.7B writes them while playing a person who holds one half.')
    lines = wrap(foot, W - 2 * M, 40, 400)
    for k, ln in enumerate(lines):
        s.text(M, y + 40 + k * 52, ln, 40, 400, DIM)
    finish(s, name, y + 40 + 52 * len(lines) + M)
    return dict(families=d['families'], bad={f: round(d['bad'][f] / d['turns'][f], 3) for f in d['bad']})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', help='an opendata shard (jsonl.gz)')
    args = ap.parse_args()
    d = snapshot(args.snapshot) if args.snapshot else json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    print(json.dumps(figure(d), indent=1))


if __name__ == '__main__':
    main()
