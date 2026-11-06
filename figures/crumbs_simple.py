"""The two communication results in plain terms, from results/crumbs_status.json (figures/crumbs_status.py --snapshot).

  memory_message    Model A reads half of a chat and Model B gets part of Model A's memory (KV cache); how close Model
                    B's answers get to those of a model that read the text
  letterfree_notes  Model A writes notes with no letters; right answers for the trained partner and an untrained model

python figures/crumbs_simple.py draws both into figures/black/.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import wrap                                                      # noqa: E402
from crumbs import GOLD                                                          # noqa: E402
from crumbs_status import SNAPSHOT, arrow, note, token_lines                     # noqa: E402


def closed(r):
    return (r['cut'] - r['on']) / r['cut']


def memory_figure(d, name='memory_message'):
    W = 2400
    c = {k: closed(v[-1]) for k, v in d['curves'].items()}
    s = canvas(W)
    y = title(s, f'Reading a quarter of another model’s memory gets a model {100 * c["even64"]:.0f}% of the way to '
                 f'having read the text itself', W)
    ex = d['example']
    gap, lw = 130, 1340
    rw = W - 2 * M - lw - gap
    h = 610
    pad = 34
    s.rect(M, y, lw, h, CARD, 24)
    s.rect(M + lw + gap, y, rw, h, CARD, 24)
    s.text(M + pad, y + 68, 'Model A reads half of a chat', 54, 700, EMBER)
    picked = set(ex['picked'])
    lines = token_lines(ex['tokens'], picked, lw - 2 * pad - 50, 42)
    for k, ln in enumerate(lines[:8]):
        chunks = [(t, GOLD if p else TEXT, 700 if p else 400) for t, p in ln]
        if k == 7:
            chunks.append((' …', TEXT, 400))
        s.rich(M + pad, y + 134 + k * 52, chunks, 42)
    s.text(M + pad, y + h - 30, 'Model B gets the memory (KV cache) of the gold tokens only', 44, 700, GOLD)
    arrow(s, M + lw + 14, y + h / 2, M + lw + gap - 14, GOLD, 9)
    bx = M + lw + gap + pad
    s.text(bx, y + 68, 'Model B never sees the text', 54, 700, QUILL)
    s.text(bx, y + 150, 'A user asks Model B:', 44, 700, TEXT)
    for k, ln in enumerate(wrap('“' + ex['user'] + '”', rw - 2 * pad, 46, 400)):
        s.text(bx, y + 214 + k * 60, ln, 46, 400, TEXT)
    s.text(bx, y + 432, 'Goal: answer like a model', 44, 700, LINK)
    s.text(bx, y + 484, 'with the whole chat:', 44, 700, LINK)
    answer = wrap('“' + ex['answer'], rw - 2 * pad, 40, 400)[:2]
    for k, ln in enumerate(answer):
        s.text(bx, y + 538 + k * 50, ln + (' …' if k == 1 else ''), 40, 400, TEXT)
    y += h + 90

    s.text(M, y, 'How close Model B’s answers get', 58, 700, TEXT)
    y += 50
    rows = (('All of the memory (525 tokens)', c['full'], LINK),
            ('A quarter of the memory (125 tokens)', c['even64'], LINK),
            ('Compressed into 125 learned vectors', c['slots64'], QUILL),
            ('Compressed into 30 learned vectors', c['slots16'], QUILL))
    x0 = M + 1000
    full = W - M - x0 - 180
    bh, row = 74, 104
    y0 = y
    for lab, v, color in rows:
        s.text(x0 - 30, y + bh - 20, lab, 50, 700, TEXT, 'end')
        s.rect(x0, y, full * v, bh, color, 8)
        s.text(W - M, y + bh - 18, f'{100 * v:.0f}%', 58, 700, color, 'end')
        y += row
    s.line(x0, y0 - 10, x0, y, TEXT, 4)
    s.line(x0 + full, y0 - 10, x0 + full, y, TEXT, 4)
    s.text(x0, y + 50, '0%: no memory', 44, 700, TEXT)
    s.text(W - M, y + 50, '100%: same answers', 44, 700, TEXT, 'end')
    s.text(W - M, y + 102, 'as reading the text', 44, 700, TEXT, 'end')
    y += 150
    foot = (f'Two copies of Qwen3-1.7B on a mix of chats, documents and code, measured during training. On real chats '
            f'alone, the 125 learned vectors reach {100 * d["chats"]["slots64"]:.0f}%; the quarter of the memory is '
            f'still being tested.')
    got = wrap(foot, W - 2 * M, 44, 400)
    for k, ln in enumerate(got):
        s.text(M, y + 44 + k * 58, ln, 44, 400, TEXT)
    finish(s, name, y + 44 + 58 * len(got) + M)
    return dict(closed=c)


def notes_figure(d, name='letterfree_notes'):
    W = 2400
    r = d['readers']
    s = canvas(W)
    y = title(s, f'Notes with no letters: in the apple game the trained partner gets {100 * r["count"]["trained"]["own"]:.0f}'
                 f'% right, an untrained model {100 * r["count"]["base"]["own"]:.0f}%', W)
    cw, h = 800, 170
    s.rect(M, y, cw, h, CARD, 22)
    s.text(M + 30, y + 72, 'Three models each count apples', 46, 700, TEXT)
    s.text(M + 30, y + 134, 'at a stall and leave a note', 46, 400, TEXT)
    fx = W - M - cw
    s.rect(fx, y, cw, h, CARD, 22)
    s.text(fx + 30, y + 72, 'A fourth model reads the notes', 46, 700, TEXT)
    s.text(fx + 30, y + 134, 'and says the total', 46, 400, TEXT)
    arrow(s, M + cw + 14, y + h / 2 + 20, fx - 14, GOLD, 9)
    s.text(W / 2, y + h / 2 - 12, 'notes', 54, 700, GOLD, 'middle')
    y += h + 100

    row = next(x for x in d['notes'] if x['step'] == 200)
    s.text(M, y, 'Real notes from the rooms game, written for each object', 54, 700, TEXT)
    y += 36
    objs = list(row['notes'])
    col = (W - 2 * M) / 4
    for i, o in enumerate(objs):
        x = M + (i % 4) * col
        yy = y + (i // 4) * 104
        s.rect(x, yy, col - 24, 88, CARD, 14)
        s.text(x + 22, yy + 60, o, 44, 700, TEXT)
        note(s, x + col - 24 - 22 - 8 * .602 * 46, yy + 60, row['notes'][o], 46)
    y += (len(objs) + 3) // 4 * 104 + 90

    s.rich(M, y, [('Right answers using the notes: ', TEXT, 700), ('trained partner', LINK, 700), (', ', TEXT, 700),
                  ('untrained model', EMBER, 700), (', white line = chance', TEXT, 700)], 54)
    y += 44
    games = (('count', 'Apples: the total', 'from 7 choices', 1 / 7),
             ('trail', 'Rooms: the object in a room', 'from 6 choices', 1 / 6),
             ('relay', 'Key: the room with the key', 'from 6 choices', 1 / 6),
             ('maze', 'Maze: the door to take', 'from 3 choices', 1 / 3))
    x0 = M + 800
    full = W - M - x0 - 150
    scale = lambda v: full * v / .8                           # noqa: E731
    bh = 56
    for g, lab, sub, chance in games:
        s.text(x0 - 30, y + 50, lab, 46, 700, TEXT, 'end')
        s.text(x0 - 30, y + 102, sub, 42, 400, TEXT, 'end')
        top = y
        for reader, color in (('trained', LINK), ('base', EMBER)):
            v = r[g][reader]['own']
            s.rect(x0, y, scale(v), bh, color, 6)
            s.text(W - M, y + bh - 10, f'{100 * v:.0f}%', 52, 700, color, 'end')
            y += bh + 8
        s.line(x0 + scale(chance), top - 8, x0 + scale(chance), y, TEXT, 5)
        y += 34
    foot = ('Trained partner: Qwen3-1.7B after training. Untrained model: the same Qwen3-1.7B before training. 96 test '
            'games per row. Two more games, where a note also helps the partner, have not been scored this way yet. '
            'Untrained Qwen3-8B and AFM-4.5B come next.')
    got = wrap(foot, W - 2 * M, 44, 400)
    for k, ln in enumerate(got):
        s.text(M, y + 30 + k * 58, ln, 44, 400, TEXT)
    finish(s, name, y + 30 + 58 * len(got) + M)
    return dict(readers=r)


def main():
    d = json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    print(json.dumps(dict(memory=memory_figure(d), notes=notes_figure(d)), indent=1)[:600])


if __name__ == '__main__':
    main()
