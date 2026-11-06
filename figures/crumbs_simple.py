"""The two communication results in plain terms, from results/crumbs_status.json (figures/crumbs_status.py --snapshot).

  memory_message           Model A reads half of a chat and Model B gets part of Model A's memory (KV cache); how close
                           Model B's answers get to those of a model that read the text
  kv_real_chats            the same with all of the KV
  letterfree_notes         notes with no letters: right answers for the trained partner and an untrained model
  letter_penalty_messages  open conversation: real messages token by token under a penalty on letters

python figures/crumbs_simple.py [--talk-snapshot] [--notes-snapshot] draws the last two into figures/black/.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                                # noqa: E402
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


KV_ROWS = (('chat_alt', 'Real chats'), ('fill', 'Fill-in'), ('code', 'Code'), ('doc', 'Documents'),
           ('debug', 'Debugging'), ('chat_doc', 'Pasted material with requests'), ('pair', 'Two documents'),
           ('plan', 'Plans'))


def kv_figure(d, name='kv_real_chats'):
    W = 2400
    head = json.loads((ROOT / 'results/crumbs_open.json').read_text())['split_full']['headline']
    chat = head['chat_alt eval_seen']
    s = canvas(W)
    y = title(s, f'With all of another model’s KV, a model gets {100 * chat["closed_on"]:.0f}% of the way to having '
                 f'read a real chat itself', W)
    ex = d['example']
    text = ''.join(ex['tokens'])
    question = text.split('A user asked you:')[1].split('You answered:')[0].strip()
    reply = text.split('You answered:')[1].strip()
    gap, lw = 130, 1340
    rw = W - 2 * M - lw - gap
    h, pad = 560, 34
    s.rect(M, y, lw, h, CARD, 24)
    s.rect(M + lw + gap, y, rw, h, CARD, 24)
    s.text(M + pad, y + 68, 'Model A has this half of the chat', 54, 700, EMBER)
    yy = y + 134
    s.text(M + pad, yy, 'User:', 42, 700, TEXT)
    yy += 52
    for ln in wrap(question, lw - 2 * pad, 42, 400)[:2]:
        s.text(M + pad, yy, ln, 42, 400, TEXT)
        yy += 52
    s.text(M + pad, yy + 8, 'Model A:', 42, 700, TEXT)
    yy += 60
    got = wrap(reply, lw - 2 * pad, 42, 400)[:3]
    for k, ln in enumerate(got):
        s.text(M + pad, yy, ln + (' …' if k == len(got) - 1 else ''), 42, 400, TEXT)
        yy += 52
    arrow(s, M + lw + 14, y + h / 2 + 20, M + lw + gap - 14, GOLD, 9)
    s.text(M + lw + gap / 2, y + h / 2 - 12, 'KV', 54, 700, GOLD, 'middle')
    bx = M + lw + gap + pad
    s.text(bx, y + 68, 'Model B never sees the text', 54, 700, QUILL)
    s.text(bx, y + 140, 'A user asks Model B:', 44, 700, TEXT)
    for k, ln in enumerate(wrap('“' + ex['user'] + '”', rw - 2 * pad, 44, 400)):
        s.text(bx, y + 204 + k * 56, ln, 44, 400, TEXT)
    s.text(bx, y + 400, 'Goal: answer like a model', 44, 700, LINK)
    s.text(bx, y + 452, 'with the whole chat', 44, 700, LINK)
    y += h + 96

    s.rich(M, y, [('How close Model B’s answers get: ', TEXT, 700), ('after training', LINK, 700), (', ', TEXT, 700),
                  ('no training', EMBER, 700)], 54)
    y += 40
    x0 = M + 780
    full = W - M - x0 - 300
    bh = 40
    y0 = y
    for fam, lab in KV_ROWS:
        r = head[f'{fam} eval_seen']
        s.text(x0 - 30, y + bh + 10, lab, 46, 700, TEXT, 'end')
        for k, color in (('closed_on', LINK), ('closed_zero', EMBER)):
            v = r[k]
            s.rect(x0, y, full * v, bh, color, 6)
            s.text(W - M, y + bh - 2, f'{100 * v:.0f}%', 44, 700, color, 'end')
            y += bh + 6
        y += 22
    s.line(x0, y0 - 8, x0, y - 16, TEXT, 4)
    s.line(x0 + full, y0 - 8, x0 + full, y - 16, TEXT, 4)
    s.text(x0, y + 34, '0%: no KV', 44, 700, TEXT)
    s.text(W - M, y + 34, '100%: same answers as reading the text', 44, 700, TEXT, 'end')
    y += 80
    worse = (1 - chat['closed_transplant'])
    foot = (f'Two copies of Qwen3-1.7B on test data. On real chats, the KV of a different chat makes answers about '
            f'{worse:.0f} times worse than no KV, so Model B really is reading this chat’s KV.')
    got = wrap(foot, W - 2 * M, 44, 400)
    for k, ln in enumerate(got):
        s.text(M, y + 44 + k * 58, ln, 44, 400, TEXT)
    finish(s, name, y + 44 + 58 * len(got) + M)
    return {k: head[f'{k} eval_seen'] for k, _ in KV_ROWS}


NOTES = ROOT / 'results/notes_final.json'
NOTES_RUN = Path('/Users/user/mpd-data/runpod/magic-crumbs66/results/bar_crumbs65_best.json')
NOTES_ROWS = (('trail', 'Rooms', 0, 'a violin'), ('maze', 'Maze', 0, 'the purple door'), ('work', 'Sums', 0, '31 + 32'),
              ('trap', 'Traps', 1, 'a trap'), ('count', 'Apples', 0, '5 apples'))


def notes_snapshot():
    """Writes results/notes_final.json: the numbers and example notes the notes figure draws, from the final read."""
    best = json.loads(NOTES_RUN.read_text())['ckpts']['best.pt']
    out = {}
    for game, _, i, _ in NOTES_ROWS:
        r = best[game]
        one = r['one_note']['Qwen/Qwen3-1.7B']
        shots = one.get('shots8') or one['shots0']
        out[game] = dict(trained=r['trained']['choice_on'], trained_swap=r['trained']['choice_swap'],
                         chance=r['chance'], read_chance=shots['chance'], read=shots['own'], read_swap=shots['swap'],
                         read_ci=shots['own_minus_swap_ci'], english=shots['english_own'],
                         shots=8 if 'shots8' in one else 0, note=r['notes'][0]['notes'][i]['note'],
                         saw=r['notes'][0]['notes'][i]['saw'], says=r['notes'][0].get('says'))
    NOTES.write_text(json.dumps(out, ensure_ascii=False, indent=0))


def notes_figure(name='letterfree_notes'):
    W = 2400
    d = json.loads(NOTES.read_text())
    s = canvas(W)
    y = title(s, 'Trained models read each other’s no-letter notes; in two of five games an untrained copy can’t', W)

    side, h = 600, 330
    s.rect(M, y, side, h, CARD, 24)
    s.text(M + 34, y + 70, 'Model A, trained', 54, 700, LINK)
    for k, ln in enumerate(wrap(d['trail']['saw'], side - 68, 46, 400)):
        s.text(M + 34, y + 150 + k * 60, ln, 46, 400, TEXT)
    rx = W - M - 760
    half = (h - 22) / 2
    s.rect(rx, y, 760, half, CARD, 20)
    s.rect(rx, y + half + 22, 760, half, CARD, 20)
    s.text(rx + 30, y + 62, 'Model B, trained', 50, 700, LINK)
    said = d['trail']['says'][0].split('\n')[0]
    assert 'violin' in said, said
    s.text(rx + 30, y + 122, f'Writes “{said}”', 44, 400, TEXT)
    s.text(rx + 30, y + half + 84, 'Same model, untrained', 50, 700, EMBER)
    s.text(rx + 30, y + half + 144, 'Sees 8 notes with their answers', 44, 400, TEXT)
    cx = (M + side + rx) / 2
    size = 92
    note(s, cx - 8 * .602 * size / 2, y + 150, d['trail']['note'], size)
    s.text(cx, y + 46, 'A note of 8 tokens', 50, 700, GOLD, 'middle')
    s.text(cx, y + 222, 'no letters in any language', 46, 700, GOLD, 'middle')
    low, up = y + half + 22 + half / 2, y + half / 2
    arrow(s, M + side + 14, low, rx - 14, GOLD, 8)
    s.line(rx - 64, low, rx - 64, up, GOLD, 8)
    arrow(s, rx - 64, up, rx - 14, GOLD, 8)
    y += h + 104

    s.rich(M, y, [('Right answers: ', TEXT, 700), ('trained partner', LINK, 700), (', ', TEXT, 700),
                  ('untrained model', EMBER, 700), (', white line = chance', TEXT, 700)], 52)
    y += 42
    x0 = M + 860
    full = W - M - x0 - 200
    bh = 60
    for game, lab, _, seen in NOTES_ROWS:
        r = d[game]
        s.text(M, y + 52, lab, 58, 700, TEXT)
        s.text(M, y + 118, seen + ':', 44, 400, TEXT)
        note(s, M + 30 + width(seen + ':', 44, 400), y + 118, r['note'], 50)
        for v, chance, color in ((r['trained'], r['chance'], LINK), (r['read'], r['read_chance'], EMBER)):
            s.rect(x0, y, full, bh, CARD, 6)
            s.rect(x0, y, full * v, bh, color, 6)
            s.line(x0 + full * chance, y - 4, x0 + full * chance, y + bh + 4, TEXT, 5)
            s.text(W - M, y + bh - 10, f'{100 * v:.0f}%', 58, 700, color, 'end')
            y += bh + 8
        y += 40
    y += 40
    low = min(r['english'] for r in d.values())
    amount = '100%' if low == 1 else f'at least {100 * low:.0f}%'
    s.text(M, y, f'Given an English note, the same untrained model gets {amount} right in all five games.', 50, 700, TEXT)
    y += 64
    foot = ('Qwen3-1.7B, 192 test games per game. Two more games are left out: in one the notes never worked, and one has '
            'no right-answer score.')
    got = wrap(foot, W - 2 * M, 44, 400)
    for k, ln in enumerate(got):
        s.text(M, y + k * 58, ln, 44, 400, TEXT)
    finish(s, name, y + 58 * (len(got) - 1) + M)
    return {k: {x: round(v[x], 4) for x in ('trained', 'read', 'read_swap', 'english')} | dict(shots=v['shots'])
            for k, v in d.items()}


TALK = ROOT / 'results/dialogue_status.json'
TALK_RUNS = (('v15', 1400), ('v17', 1421), ('v19', 1433))
TALK_FAMS = ('chat_alt', 'fill', 'code', 'doc', 'debug', 'chat_doc', 'pair', 'plan')
BLANK, SYMBOL = '#26262E', QUILL


def talk_snapshot():
    """Writes results/dialogue_status.json from the dialogue run's logs: the overall scores at each stage, and each
    example message as text and as one class per token. A newline is one token, as in the run's blank messages; other
    text is tokenized again with the model's tokenizer, so a few rows differ from what was decoded (24 rows, at most
    144 tokens before cutting to 128)."""
    import unicodedata
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', local_files_only=True)

    def cells(text):
        out = []
        for k, part in enumerate(text.split('\n')):
            if k:
                out.append('b')
            for i in tok.encode(part, add_special_tokens=False) if part else ():
                t = tok.decode([i])
                if '�' in t or any(unicodedata.category(c)[0] == 'L' for c in t):
                    out.append('l')
                else:
                    out.append('b' if not t.strip() else 's')
        return ''.join(out[:128]).ljust(128, 'b')

    stages = []
    for v, step in TALK_RUNS:
        logs = Path('/Users/user/mpd-data/runpod/magic-opentalk5/results') / v / 'log.jsonl'
        rows = [json.loads(x) for x in open(logs)]
        d = next(r for r in rows if r.get('step') == step and 'dev' in r)['dev']
        a = d['_all']
        ex = [(f, i, e) for f in TALK_FAMS for i, e in enumerate(d[f]['examples'])]
        stages.append(dict(run=v, step=step, writer=a['value_nf'], copy=a['copy_head_nf'], letters=a['letters'],
                           filler=a['filler'], symbols=1 - a['letters'] - a['filler'],
                           rows={f'{f} {i}': cells(e[3:]) for f, i, e in ex},
                           text={f'{f} {i}': e[3:] for f, i, e in ex}))
    TALK.write_text(json.dumps(dict(stages=stages), ensure_ascii=False, indent=0))


def coloured(s, x, y, text, size, limit, lines):
    """Real message text, letter by letter: letters orange, other symbols purple, a newline as a white return arrow."""
    import unicodedata
    adv = .602 * size
    per = int(limit // adv)
    rows, row = [], ''
    for word in text.replace('\n', '↵ ').split(' '):
        while len(word) > per:                                    # a long run of returns has no spaces to break at
            rows.append(row) if row else None
            row, word = '', word[per:] if rows.append(word[:per]) is None else word
        if len(row) + len(word) + (1 if row else 0) > per:
            rows.append(row)
            row = word
        else:
            row = f'{row} {word}' if row else word
    rows.append(row)
    more = len(rows) > lines
    rows = rows[:lines]
    if more:
        rows[-1] = rows[-1][:per - 1] + '…'
    for k, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == ' ':
                continue
            color = EMBER if unicodedata.category(ch)[0] == 'L' else TEXT if ch in '↵…' else SYMBOL
            s.text(x + j * adv, y + k * size * 1.3, ch, size, 700, color, family='Menlo, monospace')


def dialogue_figure(name='letter_penalty_messages'):
    W = 2400
    st = json.loads(TALK.read_text())['stages']
    s = canvas(W)
    y = title(s, 'A penalty on letters broke the messages between two models before any code formed', W)

    side, h = 520, 250
    s.rect(M, y, side, h, CARD, 24)
    s.rect(W - M - side, y, side, h, CARD, 24)
    for x, who, what in ((M, 'Model A', 'half of a chat'), (W - M - side, 'Model B', 'the other half')):
        s.text(x + side / 2, y + 100, who, 62, 700, TEXT, 'middle')
        s.text(x + side / 2, y + 176, what, 48, 400, TEXT, 'middle')
    left, right = M + side + 22, W - M - side - 22
    s.text(W / 2, y + 46, 'Messages of 128 tokens, both ways', 50, 700, GOLD, 'middle')
    s.line(left + 16, y + 84, right - 16, y + 84, GOLD, 8)
    s.path(f'M{left + 34},{y + 62} L{left + 6},{y + 84} L{left + 34},{y + 106}', GOLD, 8)
    s.path(f'M{right - 34},{y + 62} L{right - 6},{y + 84} L{right - 34},{y + 106}', GOLD, 8)
    real = st[0]['text']['plan 0'].split(' The estate')[0]
    for k, ln in enumerate(wrap('“' + real + '”', right - left - 40, 44, 400)[:2]):
        s.text(W / 2, y + 156 + k * 56, ln, 44, 400, TEXT, 'middle')
    y += h + 76
    s.text(W / 2, y, 'Goal: each model answers like a model that read the whole chat', 54, 700, LINK, 'middle')
    y += 96

    legend = [('Each row is one real message, each cell one token: ', TEXT, 700), ('letters', EMBER, 700),
              (', ', TEXT, 700), ('other symbols', SYMBOL, 700), (', blank', TEXT, 700)]
    s.rich(M, y, legend, 50)
    s.rect(M + sum(width(t, 50, w) for t, _, w in legend) + 20, y - 36, 64, 42, BLANK, 4, stroke=TEXT, sw=2)
    y += 54

    gap = 56
    pw = (W - 2 * M - 2 * gap) / 3
    heads = (('Weak penalty on letters,', '1,400 steps'), ('Strong penalty on letters,', '21 more steps'),
             ('Strong penalty on letters', 'and blank lines, 33 more steps'))
    keys = sorted(st[0]['rows'], key=lambda k: -sum(c != 'b' for c in st[0]['rows'][k]))
    rh, cw = 17, pw / 128
    colour = dict(l=EMBER, s=SYMBOL, b=BLANK)
    top = y
    for j, (r, head) in enumerate(zip(st, heads)):
        x, y = M + j * (pw + gap), top
        s.text(x, y + 50, head[0], 48, 700, TEXT)
        s.text(x, y + 108, head[1], 48, 700, TEXT)
        y += 136
        for i, key in enumerate(keys):
            row = r['rows'][key]
            a = 0
            while a < 128:
                b = a
                while b < 128 and row[b] == row[a]:
                    b += 1
                s.rect(x + a * cw, y + i * rh, (b - a) * cw + .4, rh - 2, colour[row[a]], 0)
                a = b
        y += len(keys) * rh + 56
        coloured(s, x, y, r['text']['doc 0'], 40, pw, 3)
        y += 3 * 52 + 22
        s.text(x, y, f'Letters: {100 * r["letters"]:.0f}% of all tokens', 48, 700, EMBER)
        y += 100
        for lab, v, color in (('These messages', r['writer'], TEXT), ('Copy of first 128 tokens', r['copy'], LINK)):
            s.text(x, y, lab, 44, 700, color)
            s.text(x + pw, y + 4, f'{100 * v:.0f}%', 64, 700, color, 'end')
            s.rect(x, y + 20, pw, 40, CARD, 6)
            s.rect(x, y + 20, pw * v, 40, color, 6)
            y += 118
    bars = y
    y += 6
    s.text(M, y, 'Bars: how close the answers get. 0%: no message. 100%: same answers as reading the whole chat.', 46, 700, TEXT)
    y += 62
    foot = ('Two copies of Qwen3-1.7B with trained adapters. Scores over 128 chats, documents and code tasks not used '
            'in training. Both strong penalties start from the weights on the left.')
    got = wrap(foot, W - 2 * M, 44, 400)
    for k, ln in enumerate(got):
        s.text(M, y + k * 58, ln, 44, 400, TEXT)
    finish(s, name, y + 58 * (len(got) - 1) + M)
    return [{k: round(r[k], 4) for k in ('writer', 'copy', 'letters', 'filler', 'symbols')} for r in st]


def main():
    OUT.mkdir(exist_ok=True)
    if '--talk-snapshot' in sys.argv:
        talk_snapshot()
    if '--notes-snapshot' in sys.argv:
        notes_snapshot()
    print(json.dumps(dialogue_figure(), indent=0)[:600])
    print(json.dumps(notes_figure(), indent=0)[:900])


if __name__ == '__main__':
    main()
