"""Where the breadcrumb runs stand, white on black.

  crumbs_open_status       breadcrumbs on the open-ended data: one real chat with the tokens agent B can read in gold,
                           and per run the KL of agent B's answers to the model that saw both halves, per training step
  crumbs_letterfree_status letter-free notes: the notes at each check, the loss from reading another episode's note,
                           the letters in the worst game, and four readers' right-object percent on trail

python figures/crumbs_status.py --snapshot reads the run folders and writes results/crumbs_status.json; python
figures/crumbs_status.py draws from it into figures/black/.
"""
import argparse
import base64
import gzip
import io
import json
from pathlib import Path
import sys
import unicodedata

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, HOT, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                               # noqa: E402
from crumbs import GOLD                                                                # noqa: E402

SNAPSHOT = ROOT / 'results/crumbs_status.json'
RUNS = Path('/Users/user/mpd-data/runpod')
SHARD = RUNS / 'magic-opentalk4/results/opendata/Qwen3-1.7B/shards/eval_seen-0000.jsonl.gz'
EXAMPLE = 'a33b060c51c57522'
MONO = 'Menlo, monospace'
ADV = 0.602
OBJECTS = ('wagon', 'ring', 'valley', 'tomato')
EMOJI_FONT = '/System/Library/Fonts/Apple Color Emoji.ttc'
GRID = '#2A2A30'


def devs(path):
    return [json.loads(line) for line in open(path) if '"dev_' in line]


def example():
    """Agent A's first visit of the example chat, as crumb_open.parts cuts it, with the even64 positions of
    crumbs.taken; agent B's first request and the teacher's answer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', local_files_only=True)
    rec = next(r for r in map(json.loads, gzip.open(SHARD, 'rt')) if r['id'] == EXAMPLE)
    ids = lambda t: tok(t, add_special_tokens=False).input_ids                            # noqa: E731
    text = '\n\n'.join(f'A user asked you:\n{t["user"]}\n\nYou answered:\n{t["response"]}' for t in rec['turns']
                       if t['who'] == 'A')
    cur, n = [], 0
    for line in text.split('\n'):                 # the first part only: at most 384 tokens, cut at a line end
        got = ids(line + '\n')
        if n + len(got) > 384 and cur:
            break
        cur.append(tok.decode(got))
        n += len(got)
    span = ids('Part 1:\n' + ''.join(cur))
    picked = {round(j * (len(span) - 1) / 63) for j in range(64)}
    b = next(t for t in rec['turns'] if t['who'] == 'B')
    return dict(id=EXAMPLE, source=rec['source'], tokens=[tok.decode([i]) for i in span],
                picked=sorted(picked), user=b['user'], answer=b['response'])


def snapshot():
    d = json.loads((ROOT / 'results/crumbs_open.json').read_text())
    curve = lambda run: [dict(step=r['step'], on=r['dev_nll'], cut=r['dev_cut']) for r in devs(  # noqa: E731
        RUNS / run / 'log.jsonl')]
    curves = dict(slots16=[{k: r[k] for k in ('step', 'on', 'cut')} for r in d['slots']['dev']],
                  slots64=[{k: r[k] for k in ('step', 'on', 'cut')} for r in d['slots64']['dev']],
                  full=[{k: r[k] for k in ('step', 'on', 'cut')} for r in d['sentence']['dev']],
                  even64=curve('magic-crumbs36/results/open_even64'),
                  mix48=curve('magic-crumbs37/results/open_mix48'))
    first = devs(RUNS / 'magic-crumbs30/results/sym_emoji/log.jsonl')
    later = devs(RUNS / 'magic-crumbs39/results/sym_emoji/log.jsonl')
    rows = [r for r in first if r['step'] in (100, 150, 200)] + [r for r in later if r['step'] > 200]
    notes = [dict(step=r['step'], notes=dict(e.split(': ', 1) for e in r['examples']), swap=r['gap_swap'],
                  distinct=r['distinct'], letters=r['letters_text'], worst=r.get('letters_max'),
                  by_game=r.get('letters_by_game')) for r in rows]
    eaves = json.loads((RUNS / 'magic-crumbs40/results/eaves_crumbs30.json').read_text())['ckpts']['gate.pt']
    readers = {}
    for game in ('trail', 'count', 'relay', 'maze'):
        got = eaves.get(game, {})
        row = {k: got[k] for k in ('trained', 'base') if k in got}
        row.update({f'crack{n}': v for n, v in got.get('crack', {}).items()})
        readers[game] = {k: dict(own=v['choice_on'], other=v['choice_swap'], episodes=v['episodes'])
                         for k, v in row.items()}
    chats = {k: d[k]['test']['eval_seen']['by_family']['chat_alt']['closed_on'] for k in ('slots', 'slots64')}
    out = dict(curves=curves, example=example(), notes=notes, readers=readers, chats=chats)
    SNAPSHOT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def is_letter(ch):
    return unicodedata.category(ch).startswith('L')


def is_emoji(ch):
    return ord(ch) >= 0x1F000


def emoji_uri(ch, cache={}):
    if ch not in cache:
        font = ImageFont.truetype(EMOJI_FONT, 160)
        im = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((20, 20), ch, font=font, embedded_color=True)
        im = im.crop(im.getbbox())
        side = max(im.size)
        sq = Image.new('RGBA', (side, side), (0, 0, 0, 0))
        sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        buf = io.BytesIO()
        sq.save(buf, 'PNG')
        cache[ch] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    return cache[ch]


def note(s, x, y, text, size):
    """A note as decoded: letters white, other characters gold, a newline as the return arrow, an emoji as a picture."""
    a = ADV * size
    for ch in text:
        if ch in '‍️':
            continue
        if is_emoji(ch):
            s.image(x, y - size * 0.92, round(size * 1.12), emoji_uri(ch))
            x += 2 * a
            continue
        shown = {'\n': '↵', '\t': '→'}.get(ch, ch)
        s.text(x, y, shown, size, 700, TEXT if is_letter(ch) else GOLD, family=MONO)
        x += a


def arrow(s, x1, y, x2, color=TEXT, sw=7):
    s.line(x1, y, x2 - 18, y, color, sw)
    s.path(f'M{x2 - 30},{y - 20} L{x2},{y} L{x2 - 30},{y + 20}', color, sw)


# ---------------------------------------------------------------- breadcrumbs on the open-ended data

def token_lines(tokens, picked, w, size):
    """Lines of (text, picked) runs of tokens, broken between tokens to fit w."""
    words = []
    for i, t in enumerate(tokens):
        t = t.replace('\n', ' ')
        if not words or t.startswith(' '):
            words.append([])
        words[-1].append((t, i in picked))
    lines, line, lw = [], [], 0
    for word in words:
        ww = sum(width(t, size, 700 if p else 400) for t, p in word)
        if line and lw + ww > w:
            lines.append(line)
            line, lw = [], 0
        if not line:
            word = [(word[0][0].lstrip(), word[0][1])] + word[1:]
        line.extend(word)
        lw += ww
    return lines + ([line] if line else [])


def open_figure(d, name='crumbs_open_status'):
    W = 2600
    s = canvas(W)
    y = title(s, 'Trained on 64 evenly spaced keys and values from agent A, agent B answers closer to the model that saw '
                 'both halves than with 64 learned slots', W)
    ex = d['example']

    # one real chat: agent A's half with the 64 tokens in gold, then agent B's turn
    gap, lw = 150, 1500
    rw = W - 2 * M - lw - gap
    h = 650
    s.rect(M, y, lw, h, CARD, 24)
    s.rect(M + lw + gap, y, rw, h, CARD, 24)
    size = 40
    pad = 34
    s.text(M + pad, y + 66, f'Agent A reads half of a real chat, {len(ex["tokens"])} tokens', 50, 700, EMBER)
    picked = set(ex['picked'])
    lines = token_lines(ex['tokens'], picked, lw - 2 * pad - 50, size)
    shown = lines[:9]
    for k, ln in enumerate(shown):
        chunks = [(t, GOLD if p else TEXT, 700 if p else 400) for t, p in ln]
        if k == len(shown) - 1 and len(lines) > len(shown):
            chunks.append((' …', TEXT, 400))
        s.rich(M + pad, y + 130 + k * 52, chunks, size)
    s.text(M + pad, y + h - 28, f'gold: the {len(picked)} tokens whose keys and values agent B can read', 44, 700, GOLD)

    ax = M + lw
    arrow(s, ax + 14, y + h / 2, ax + gap - 14, GOLD, 9)
    bx = M + lw + gap + pad
    bw = rw - 2 * pad
    s.text(bx, y + 66, 'Agent B never sees A’s half', 50, 700, QUILL)
    yy = y + 130
    s.text(bx, yy, 'The user asks agent B:', 40, 700, TEXT)
    for k, ln in enumerate(wrap(ex['user'], bw, 44, 400)):
        s.text(bx, yy + 56 + k * 56, ln, 44, 400, TEXT)
    yy += 56 + 56 * len(wrap(ex['user'], bw, 44, 400)) + 20
    s.text(bx, yy, 'Target answer, written by', 40, 700, LINK)
    s.text(bx, yy + 50, 'Qwen3-1.7B with both halves:', 40, 700, LINK)
    tgt = ex['answer'].split('.')[0].replace('**', '') + ' …'
    got = wrap(tgt, bw - 40, 40, 400)
    for k, ln in enumerate(got[:3]):
        s.text(bx, yy + 104 + k * 50, ln + (' …' if k == 2 and len(got) > 3 else ''), 40, 400, TEXT)
    y += h + 70

    # the KL per step
    c = d['curves']
    x0, x1 = M + 110, W - 700
    ytop = y + 90
    ch = 860
    ymax = .55
    X = lambda v: x0 + (x1 - x0) * v / 1500                   # noqa: E731
    Y = lambda v: ytop + ch * (1 - v / ymax)                  # noqa: E731
    s.text(M, y + 44, 'Distance from the target answer (KL per answer token), lower is better', 52, 700, TEXT)
    for t in (0, .1, .2, .3, .4, .5):
        s.line(x0, Y(t), x1, Y(t), GRID, 3)
        s.text(x0 - 24, Y(t) + 15, f'{t:.1f}', 44, 400, TEXT, 'end')
    for t in range(0, 1501, 300):
        s.text(X(t), ytop + ch + 60, f'{t}', 44, 400, TEXT, 'middle')
    s.text((x0 + x1) / 2, ytop + ch + 124, 'training step', 48, 400, TEXT, 'middle')

    arms = (('slots16', EMBER, '16 learned slots per visit', 'about 30 positions'),
            ('slots64', QUILL, '64 learned slots per visit', 'about 125 positions'),
            ('mix48', GOLD, '16 slots + 48 evenly spaced', 'about 125 positions'),
            ('even64', LINK, '64 evenly spaced per visit', 'about 125 positions'),
            ('full', TEXT, 'All keys and values', 'about 525 positions'))
    for k, color, *_ in arms:
        pts = [(X(r['step']), Y(r['cut'])) for r in c[k]]
        s.path('M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts), color, 5, extra='stroke-dasharray="4 16"')
    top = max(r['cut'] for k, *_ in arms for r in c[k])
    s.text(x0 + 10, Y(top) - 26, 'Dashed: the same run with the breadcrumbs cut', 46, 700, TEXT)
    for k, color, lab, sub in arms:
        pts = [(X(r['step']), Y(r['on'])) for r in c[k]]
        s.path('M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts), color, 10)
        ex_, ey = pts[-1]
        last = c[k][-1]['on']
        dy = dict(slots16=-30, slots64=14, mix48=-34, even64=38, full=0)[k]
        s.text(ex_ + 26, ey + dy, lab, 46, 700, color)
        s.text(ex_ + 26, ey + dy + 50, f'KL {last:.2f}, {sub}', 42, 400, color)
    y = ytop + ch + 180
    foot = (f'Qwen3-1.7B with a LoRA adapter. Positions are per episode. Measured on 32 training-set episodes across '
            f'8 data families; the two evenly spaced runs are still training. On test real chats alone, learned slots '
            f'remove {100 * d["chats"]["slots"]:.0f}% (16 slots) and {100 * d["chats"]["slots64"]:.0f}% (64 slots) of '
            f'the KL with the breadcrumbs cut.')
    for k, ln in enumerate(wrap(foot, W - 2 * M, 42, 400)):
        s.text(M, y + 40 + k * 54, ln, 42, 400, TEXT)
    finish(s, name, y + 40 + 54 * len(wrap(foot, W - 2 * M, 42, 400)) + M)
    return {k: c[k][-1] for k in c}


# ---------------------------------------------------------------- letter-free notes

def letter_cell(r):
    if r['by_game'] is None:
        return f'{100 * r["letters"]:.0f}% overall', TEXT
    if r['worst'] < .005:
        return '0% in every game', LINK
    game = max(r['by_game'], key=r['by_game'].get)
    return f'{100 * r["worst"]:.0f}% in {game}', HOT


def letter_figure(d, name='crumbs_letterfree_status'):
    W = 2600
    s = canvas(W)
    y = title(s, 'Letter-free notes help the trained follower and untrained Qwen3-1.7B cannot read them, but a fresh '
                 'adapter learns them and full privacy training erases them', W)

    # the method as a picture, with one real note
    cw = 900
    h = 170
    mid = (W - 2 * M - 2 * cw) / 2
    s.rect(M, y, cw, h, CARD, 22)
    s.text(M + 30, y + 72, 'Explorer sees a wagon in a room,', 48, 700, EMBER)
    s.text(M + 30, y + 134, 'writes a note of 8 tokens there', 48, 400, TEXT)
    fx = W - M - cw
    s.rect(fx, y, cw, h, CARD, 22)
    s.text(fx + 30, y + 72, 'Follower enters the same room,', 48, 700, QUILL)
    s.text(fx + 30, y + 134, 'reads the note, names the object', 48, 400, TEXT)
    arrow(s, M + cw + 14, y + h / 2, fx - 14, GOLD, 9)
    note(s, M + cw + mid - ADV * 62 * 4, y + h / 2 - 24, '65.65.65', 62)
    y += h + 130

    # the notes per check
    size = 46
    lab, oc = 170, 290
    xo = M + lab
    xd = xo + oc * len(OBJECTS)
    xs = xd + 240
    xl = xs + 530
    s.text(M, y, 'step', 44, 700, TEXT)
    for j, o in enumerate(OBJECTS):
        s.text(xo + j * oc, y, o, 48, 700, TEXT)
    s.text(xd, y - 54, 'different', 40, 700, TEXT)
    s.text(xd, y, 'of 12 notes', 40, 700, TEXT)
    s.text(xs, y - 54, 'extra loss with another', 40, 700, TEXT)
    s.text(xs, y, 'episode’s note, nats', 40, 700, TEXT)
    s.text(xl, y - 54, 'letters', 40, 700, TEXT)
    y += 26
    top = max(r['swap'] for r in d['notes'])
    barw = 380
    for r in d['notes']:
        rh = 84
        s.rect(M - 14, y, W - 2 * M + 28, rh - 10, CARD, 12)
        base = y + rh / 2 + 10
        s.text(M + 10, base, str(r['step']), 48, 700, TEXT)
        for j, o in enumerate(OBJECTS):
            note(s, xo + j * oc, base, r['notes'][o], size)
        s.text(xd + 40, base, str(r['distinct']), 48, 700, TEXT)
        w = max(4, barw * r['swap'] / top)
        s.rect(xs, y + 18, w, rh - 46, LINK, 6)
        s.text(xs + w + 14, base, f'{r["swap"]:.1f}', 46, 700, LINK)
        cell, color = letter_cell(r)
        s.text(xl, base, cell, 40, 700, color)
        y += rh
        if r['step'] == 200:
            s.text(M + 10, y + 46, 'Privacy training starts at step 200 and reaches full strength at step 500',
                   44, 700, GOLD)
            y += 72
    y += 60

    # the readers: trail and count side by side, same scale
    s.text(M, y + 40, 'Notes from step 200, before privacy training: percent of test answers that are right', 52, 700, TEXT)
    y += 120
    readers = (('trained', 'Trained follower'), ('base', 'Untrained Qwen3-1.7B'),
               ('crack64', 'Fresh adapter, 64 examples'), ('crack1024', 'Fresh adapter, 1,024 examples'))
    games = (('trail', 'Trail: which of 6 objects', 1 / 6), ('count', 'Count: which of 7 totals', 1 / 7))
    x0s = (M + 760, M + 760 + 840)
    full = 640
    scale = lambda v: full * v / .8                           # noqa: E731
    bh, row = 44, 2 * 44 + 40
    for (g, lab, _), x0 in zip(games, x0s):
        s.text(x0, y + 10, lab, 48, 700, TEXT)
    y += 40
    ty = y
    for i, (k, lab) in enumerate(readers):
        s.text(x0s[0] - 30, y + bh + 16, lab, 48, 700, TEXT, 'end')
        y += row
    for (g, _, chance), x0 in zip(games, x0s):
        y = ty
        for i, (k, _) in enumerate(readers):
            v = d['readers'][g].get(k)
            if v is None:
                s.text(x0 + scale(chance) + 18, y + bh + 16, 'still running', 44, 400, TEXT)
                y += row
                continue
            s.rect(x0, y, max(4, scale(v['own'])), bh, LINK, 6)
            s.rect(x0, y + bh + 6, max(4, scale(v['other'])), bh, '#8A8A96', 6)
            y += row
        cx = x0 + scale(chance)
        s.line(cx, ty - 8, cx, y - 30, TEXT, 4)
        y = ty
        for i, (k, _) in enumerate(readers):
            v = d['readers'][g].get(k)
            if v is not None:
                for yy, val, color, tail in ((y + bh - 8, v['own'], LINK, ' own note'),
                                             (y + 2 * bh - 2, v['other'], TEXT, ' other episode’s note')):
                    txt = f'{100 * val:.0f}%' + (tail if i == 0 and g == 'trail' else '')
                    lx = x0 + scale(val) + 12
                    s.rect(lx - 6, yy - 36, width(txt, 42, 700) + 12, 46, '#000000', 4)
                    s.text(lx, yy, txt, 42, 700, color)
            y += row
        s.text(cx, y + 6, f'chance, {100 * chance:.0f}%', 42, 700, TEXT, 'middle')
    y += 16
    finish(s, name, y + 50 + M)
    return dict(last=d['notes'][-1]['step'], readers=d['readers'], chats=d['chats'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', action='store_true')
    args = ap.parse_args()
    d = snapshot() if args.snapshot else json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    print(json.dumps(dict(open=open_figure(d), letter=letter_figure(d)), indent=1))


if __name__ == '__main__':
    main()
