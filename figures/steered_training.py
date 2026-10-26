"""The steered-reading training game (pairing/steer_read.py, train.py --steer) as a picture: the writer (original Quill,
or the chat-trained linked pair for data written with --writer linked), given one more sentence naming the concept,
writes a chat opener; linked Quill, whose twin Ember has the concept's direction added to its residual stream after a
random link layer, is trained on that opener. Below: the episodes with nothing steered (target: the opener written
without the sentence) and the direct question (target: the concept).

Reads results/magic-nl33t/results/steer_data.json (the teacher openers kept for training, as written by steer_read.py
on nl22c's step 157), the wordings and episode shares (CHAT, TEACHER, ASK, NONE_P, ASK_P) from pairing/steer_read.py,
the link layers (XBLOCKS) from pairing/link.py and the layer count from results/link_mechanism.json. The teacher's
sentence is the one steer_data.json records, or TEACHER['sense'] for data written before --teacher existed (nl33t's).
The writer is the one steer_data.json records (stats.writer, 'original' before --writer existed). The steering is
drawn after link layer 15 as an example of the random layer. Writes figures/NAME.{svg,png,pdf} and results/NAME.json
(the example and the shares drawn; NAME defaults to steered_training). No model is loaded.
Usage: python figures/steered_training.py [--data STEER_DATA --concept C --wording K --none_share P --name NAME]
"""
import argparse
import ast
import json
from fractions import Fraction
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, ROOT, SVG, WARM, width
from creature import creature, strands, width_of
from method_parts import (TINT, arrow, bubble, bubble_height, check_shape, flow, flow_height, highlight, link_down,
                          render, robot)

DATA = ROOT / 'results/magic-nl33t/results/steer_data.json'
OUT = ROOT / 'figures/steered_training'
CONCEPT, WORDING, READER = 'cookie', 2, 'quill'      # the default example: a kept opener and a plain one
SHOWN_LAYER = 15                                      # where the drawing puts the random link layer
W, M, PAD = 2400, 36, 40
BODY = 46


def constants(path, names):
    """The module-level literals names of a Python file, read without importing it."""
    tree = ast.parse(path.read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(n.value)
    missing = set(names) - set(out)
    if missing:
        raise KeyError(f'{sorted(missing)} not found in {path}')
    return out


def load(data_path=DATA, teacher=None, none_share=None, concept=CONCEPT, wording=WORDING):
    """The example and the game's settings. teacher: the key of steer_read.TEACHER (default: the one the data records,
    or 'sense' for data written before --teacher, such as nl33t's); none_share: the share of episodes with nothing
    steered (default steer_read.NONE_P, the share nl33 trained with); concept, wording: the example's concept and
    opener wording (the plain opener has the same wording when the data holds one, else the first plain one)."""
    data = json.loads(data_path.read_text())
    sr = constants(ROOT / 'pairing/steer_read.py', ('CHAT', 'TEACHER', 'ASK', 'NONE_P', 'ASK_P', 'NOTHING'))
    teacher = teacher or data['stats'].get('teacher', 'sense')
    none_share = sr['NONE_P'] if none_share is None else none_share
    xb = constants(ROOT / 'pairing/link.py', ('XBLOCKS',))['XBLOCKS']
    blocks = json.loads((ROOT / 'results/link_mechanism.json').read_text())['configs']['ember']['blocks']
    phrase = dict((a, b) for a, b in data['concepts'])[concept]
    steered = next(e for e in data['examples']
                   if e['receiver'] == READER and e['concept'] == concept and e['wording'] == wording)
    plains = [e for e in data['examples'] if e['receiver'] == READER and e['concept'] is None]
    plain = next((e for e in plains if e['wording'] == wording), plains[0])
    assert phrase in steered['text'] and SHOWN_LAYER in xb
    names = {'quill': 'Quill', 'ember': 'Ember'}
    other = 'ember' if READER == 'quill' else 'quill'
    line = sr['CHAT'][wording]
    return dict(reader=READER, steered_twin=other, concept=concept, phrase=phrase,
                reader_prompt=line.format(P=names[other]), steered_prompt=line.format(P=names[READER]),
                privileged=sr['TEACHER'][teacher].format(P=names[other], X=phrase).strip(),
                opener=steered['text'], plain_opener=plain['text'],
                ask=sr['ASK'][0].format(P=names[other]), ask_answer=phrase[0].upper() + phrase[1:] + '.',
                teacher=teacher, writer=data['stats'].get('writer', 'original'), none_share=none_share,
                ask_share=sr['ASK_P'], xblocks=list(xb), blocks=blocks, shown_layer=SHOWN_LAYER,
                concepts=data['stats']['concepts'],
                source=str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT)
                else str(data_path).replace(str(Path.home()), '~'))


def share(p):
    """'1 in 5 episodes' when p is a fraction with a small denominator, else '36% of episodes'."""
    f = Fraction(p).limit_denominator(10)
    if abs(f - p) < 1e-9:
        return f'{f.numerator} in {f.denominator} episodes'
    return f'{round(100 * p)}% of episodes'


def prompt_card(s, x, y, w, owner, title, runs):
    """A prompt card: a title in the owner's colour, then word-wrapped runs. Returns its height."""
    color = QUILL if owner == 'quill' else EMBER
    h = 40 + 54 + flow_height(runs, w - 64, BODY) + 26
    s.rect(x, y, w, h, 'white', 22, color, 3)
    s.text(x + 32, y + 66, title, 46, 700, color)
    flow(s, x + 32, y + 66 + 72, w - 64, runs, BODY)
    return h


def card_height(runs, w):
    return 40 + 54 + flow_height(runs, w - 64, BODY) + 26


def vector_icon(s, x, y, color, fill, heights=(0.55, 0.9, 0.35, 0.75, 0.5)):
    """A small state vector: a rounded box of bars."""
    w, h = 100, 84
    s.rect(x, y, w, h, fill, 14, color, 4)
    bw = 11
    for k, f in enumerate(heights):
        bx = x + 16 + k * (bw + 6)
        s.rect(bx, y + h - 14 - f * (h - 28), bw, f * (h - 28), color, 2)
    return w


def draw(r, out=OUT):
    reader, twin = r['reader'], r['steered_twin']
    rc, tc = (QUILL, EMBER) if reader == 'quill' else (EMBER, QUILL)
    RNAME, TNAME = reader.title(), twin.title()
    linked = r['writer'] == 'linked'               # the writer: the chat-trained linked pair, else original Quill
    teacher = f'chat-trained linked {RNAME}' if linked else f'original {RNAME}'
    title = f'Linked {RNAME} is trained to write what {teacher} writes when told the concept steered into {TNAME}'
    title_lines = [title] if width(title, 56, 700) <= W - 2 * M else None
    if title_lines is None:
        cut = title.rfind(' when ')
        title_lines = [title[:cut], title[cut + 1:]]
    top = 56 + 70 * len(title_lines)

    cx, cw = M + PAD, 860
    RS = 300
    rx = cx + cw + 100
    bx = rx + RS + 70
    bw = W - M - PAD - bx
    tgt_x = W - M - PAD - 70                       # the training-target arrow, right of the strip

    def hl(text):
        return highlight(text, [r['phrase']], tc, TINT[twin])

    # the extra sentence on a tint, its concept in the steered twin's colour
    teacher_runs = [(r['reader_prompt'] + ' ', INK, 400, None)] + \
        [(t, c, w if f else 600, TINT[twin]) for t, c, w, f in hl(r['privileged'])]
    student_runs = [(r['reader_prompt'], INK, 400, None)]
    twin_runs = [(r['steered_prompt'], INK, 400, None)]
    bub = hl(r['opener'])

    t_card = card_height(teacher_runs, cw)
    TS = 200                                       # the linked writer: Quill above, Ember below, the link between
    t_row = max(t_card, 2 * TS * 1.25 + 10 if linked else RS, bubble_height(bub, bw))
    th = 116 + t_row + 44
    # the student: both twins drawn whole, the link's strands at the link layers between their cores
    HC = 900                                       # a twin's full height
    half = width_of(HC, full=True) / 2
    tx_, qx_ = 880, 1330                           # the steered twin's and the reader's centre
    pc_w = 620
    q_card_x = qx_ + half + 110
    tgt_x = W - M - PAD - 70                       # the training-target arrow
    q_card_w = tgt_x - 60 - q_card_x
    s_card = card_height(student_runs, q_card_w)
    sb_h = bubble_height(bub, W - M - PAD - q_card_x)
    sh = 100 + HC + 50
    low_w = (W - 2 * M - 2 * PAD - 40) / 2
    plain_runs = [(r['plain_opener'], INK, 400, None)]
    ask_runs = [(r['ask'], INK, 400, None)]
    low_h = 110 + max(bubble_height(plain_runs, low_w - 150) + 10, card_height(ask_runs, low_w) + 30 + 120) + 40
    y1 = top + th + 30
    y2 = y1 + sh + 30
    H = round(y2 + low_h + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    for k, line in enumerate(title_lines):
        s.text(M, 84 + 70 * k, line, 56, 700)

    # teacher: original model, one more sentence in its prompt
    s.rect(M, top, W - 2 * M, th, COOL, 32)
    s.text(M + PAD, top + 76, f'Teacher: {teacher}', 52, 700)
    ty = top + 116
    tm = ty + t_row / 2
    prompt_card(s, cx, tm - t_card / 2, cw, reader, f'{RNAME}’s prompt', teacher_runs)
    qm = tm                                        # the writing twin's middle: the prompt goes in, the opener comes out
    if linked:
        tx0 = rx + (RS - TS) / 2
        qm, em = tm - (TS * 1.25 + 10) / 2, tm + (TS * 1.25 + 10) / 2
        link_down(s, tx0 - 10, qm, em)
        robot(s, reader, tx0, qm - TS / 2, TS, facing=1)
        robot(s, twin, tx0, em - TS / 2, TS, facing=1)
    else:
        robot(s, reader, rx, tm - RS / 2, RS, facing=1)
    arrow(s, [(cx + cw + 10, qm), (rx - 14, qm)], rc, 6)
    tb_h = bubble_height(bub, bw)
    bubble(s, bx, qm - tb_h / 2, bw, bub)
    teacher_bottom = qm + tb_h / 2

    # student: the steered twin on the left, the reader on the right
    s.rect(M, y1, W - 2 * M, sh, WARM, 32)
    s.text(M + PAD, y1 + 76, f'Student: linked {RNAME}', 52, 700)
    ctop = y1 + 70
    t_svg, ports = creature(twin, tx_, ctop, HC, full=True, sway=0.5, look=1)
    q_svg, q_ports = creature(reader, qx_, ctop, HC, full=True, sway=-0.5, look=-1)
    for k in r['xblocks']:
        s.add(strands(tx_, qx_, ports[k][1]))
    s.add(t_svg)
    s.add(q_svg)
    gap_y = ports[max(r['xblocks'])][1]
    s.text((tx_ + qx_) / 2, gap_y - 120, 'link,', 46, 700, LINK, 'middle')
    s.text((tx_ + qx_) / 2, gap_y - 66, 'both ways', 46, 700, LINK, 'middle')
    head_y = ctop + 0.21 * HC                      # the heads' middle
    # the steered twin's prompt goes in at its head
    t_card = card_height(twin_runs, pc_w)
    prompt_card(s, cx, head_y - t_card / 2, pc_w, twin, f'{TNAME}’s prompt', twin_runs)
    arrow(s, [(cx + pc_w + 10, head_y), (tx_ - half - 14, head_y)], tc, 6)
    # the concept direction is added to the steered twin's residual stream after one link layer
    step = (ports[r['xblocks'][0]][1] - ports[r['xblocks'][1]][1]) / (r['xblocks'][1] - r['xblocks'][0])
    py = ports[r['shown_layer']][1] - step / 2
    iy = py - 42
    vector_icon(s, cx + 10, iy, tc, TINT[twin])
    s.text(cx + 130, iy + 36, f'{r["phrase"]} direction', 46, 700, tc)
    s.text(cx + 130, iy + 92, 'after a random', 46, 700, INK)
    s.text(cx + 130, iy + 146, 'link layer', 46, 700, INK)
    lab_w = max(width(f'{r["phrase"]} direction', 46, 700), width('after a random', 46, 700))
    arrow(s, [(cx + 150 + lab_w, py), (tx_ - 26, py)], tc, 6, 20)
    s.circle(tx_, py, 20, 'white', tc, 5)
    s.line(tx_ - 10, py, tx_ + 10, py, tc, 5)
    s.line(tx_, py - 10, tx_, py + 10, tc, 5)
    # the reader's prompt goes in at its head; its opener is trained toward the teacher's
    prompt_card(s, q_card_x, head_y - s_card / 2, q_card_w, reader, f'{RNAME}’s prompt', student_runs)
    arrow(s, [(q_card_x - 10, head_y), (qx_ + half + 14, head_y)], rc, 6)
    bm = ports[r['shown_layer']][1] + 40
    bubble(s, q_card_x, bm - sb_h / 2, W - M - PAD - q_card_x, bub)
    arrow(s, [(tgt_x, teacher_bottom + 12), (tgt_x, bm - sb_h / 2 - 14)], INK, 6, 24)
    s.text(tgt_x - 24, bm - sb_h / 2 - 40, 'training target', 46, 700, INK, 'end')

    # the other episodes: nothing steered, and the direct question
    s.rect(M, y2, W - 2 * M, low_h, COOL, 32)
    x0 = M + PAD
    s.text(x0, y2 + 76, f'{share(r["none_share"])}: nothing steered', 50, 700)
    robot(s, reader, x0, y2 + 110, 120)
    bubble(s, x0 + 150, y2 + 120, low_w - 150, plain_runs)
    x1 = x0 + low_w + 40
    s.text(x1, y2 + 76, f'{share(r["ask_share"])}: asked directly', 50, 700)
    ch = prompt_card(s, x1, y2 + 110, low_w, reader, f'{RNAME}’s prompt', ask_runs)
    robot(s, reader, x1, y2 + 110 + ch + 30, 120)
    bubble(s, x1 + 150, y2 + 110 + ch + 40, width(r['ask_answer'], BODY, 700) + 80,
           highlight(r['ask_answer'], [r['ask_answer']], tc, TINT[twin]))
    render(s, out)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DATA), help="a run's steer_data.json")
    ap.add_argument('--teacher', help="the teacher's sentence, a key of steer_read.TEACHER (default: as recorded)")
    ap.add_argument('--none_share', type=float, help='share of episodes with nothing steered (default NONE_P)')
    ap.add_argument('--concept', default=CONCEPT, help="the example's concept (a key of the data's concepts)")
    ap.add_argument('--wording', type=int, default=WORDING, help="the example's opener wording (steer_read.CHAT)")
    ap.add_argument('--name', default='steered_training', help='output name: figures/NAME.png and results/NAME.json')
    args = ap.parse_args()
    r = load(Path(args.data).resolve(), args.teacher, args.none_share, args.concept, args.wording)
    H = draw(r, ROOT / 'figures' / args.name)
    (ROOT / 'results' / f'{args.name}.json').write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(height=H, concept=r['phrase'], none=r['none_share'], ask=r['ask_share'])))


if __name__ == '__main__':
    main()
