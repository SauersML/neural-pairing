"""The shared-reading games: one model gets a text, and its partner is scored on an answer that needs the text.

Left: the setup, drawn from three corpus items: a question-answering item never seen in training, the text the partner
from another episode gets instead, and a training item (stored, not drawn). They come from data/coread.json (built by pairing/build_coread.py,
test items filtered by coread_data.held_out) when it is present, else from the 'setup' entry of
results/shared_reading.json. They show the setup; they are not measured episodes.
Right: from every receiver episode of the shared-reading games in results/magic-nl20/results/payload_final.json
(final checkpoint), one bar per game and receiver: the fraction of episodes in which the receiver's mean cross-entropy
on the answer tokens is lower with its real partner than with the partner of another episode of the same game, with
the count at the bar's end (ties count as not lower). The JSON adds the one-sided exact sign test per game and receiver,
ties left out. Writes figures/shared_reading.{svg,png,pdf} and
results/shared_reading.json (counts, sign-test p per game and receiver, the setup items and every episode).
No model is loaded.
Usage: python figures/shared_reading.py
"""
from collections import Counter
import hashlib
import json
import math
import sys

from chat_names import COOL, EMBER, FAINT, INK, LINK, MUTED, QUILL, RED, ROOT, SVG, WARM, ART, icon_link, rings, width, wrap

SOURCE = ROOT / 'results/magic-nl20/results/payload_final.json'
CORPUS = ROOT / 'data/coread.json'
SNAPSHOT = ROOT / 'results/shared_reading.json'

GROUPS = [
    ('Passages', [('co_transfer', 'Copy a passage'), ('co_swap', 'Swap passages'), ('co_split', 'Join two halves'),
                  ('co_continue', 'Continue a text')]),
    ('Tasks', [('co_qa', 'Answer a question'), ('co_hop', 'Combine two facts'), ('co_translate', 'Translate'),
               ('co_paraphrase', 'Paraphrase'), ('co_nli', 'Compare two claims'), ('co_math', 'Solve a math problem'),
               ('co_code_out', 'Predict code output'), ('co_code_write', 'Write code'),
               ('co_instruct', 'Follow an instruction'), ('co_mcq', 'Multiple choice')]),
    ('Conversation', [('co_distill', 'Match the original reply'), ('co_talk', 'Answer, then talk'),
                      ('co_play', 'Play with secrets')]),
]
LABELS = {g: label for _, games in GROUPS for g, label in games}
CONTEXTS = dict(test='Volcanoes can be active, dormant, or extinct.',
                shuffle_example='Amphibians may attract mates with calls or scents.',
                training='The corona is the halo around the Sun.')

def outcome(r):
    d = r['shuffle_ce'] - r['link_ce']
    return 'real' if d > 0 else 'shuffled' if d < 0 else 'tie'


def sign_p(wins, losses):
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.


def setup_items():
    """The three corpus items of the setup, with the file they were read from."""
    if CORPUS.exists():
        sys.path.insert(0, str(ROOT / 'pairing'))
        from coread_data import held_out
        corpus = json.loads(CORPUS.read_text())
        fresh = held_out(corpus['train'], corpus['test'])
        return dict(test=next(r for r in fresh if r.get('context') == CONTEXTS['test']),
                    shuffle_example=next(r for r in fresh if r.get('context') == CONTEXTS['shuffle_example']),
                    training=next(r for r in corpus['train'] if r.get('context') == CONTEXTS['training']),
                    source='data/coread.json')
    setup = json.loads(SNAPSHOT.read_text())['setup']
    for k, ctx in CONTEXTS.items():
        assert setup[k]['context'] == ctx
    return dict(test=setup['test'], shuffle_example=setup['shuffle_example'], training=setup['training'],
                source='results/shared_reading.json (snapshot of data/coread.json rows)')


def main():
    eps = [r for r in json.loads(SOURCE.read_text()) if r['game'].startswith('co_')]
    assert set(LABELS) == {r['game'] for r in eps}
    assert len({(r['game'], r['receiver'], r['episode']) for r in eps}) == len(eps)
    assert all(math.isfinite(r[k]) for r in eps for k in ('shuffle_ce', 'link_ce'))
    totals = Counter(outcome(r) for r in eps)
    p_all = sign_p(totals['real'], totals['shuffled'])
    items = setup_items()
    sample, donor, training = items['test'], items['shuffle_example'], items['training']

    W, M = 2400, 36
    top = 120
    gx = M + 918                       # bar panel
    gw = W - M - gx
    head_h, row_h, group_gap = 120, 74, 40
    grid_h = head_h + len(LABELS) * row_h + (len(GROUPS) - 1) * group_gap + 24
    H = top + grid_h + M
    title = f'The answer is more probable with the real partner in {totals["real"]} of {len(eps)} episodes'
    s = SVG(W, H, title=title)
    s.text(M, 84, title, 56, 700)

    # Left: one item never seen in training, and the partner from another episode
    lx, lw = M, gx - 24 - M
    s.rect(lx, top, lw, grid_h, WARM, 32)
    cw = (lw - 60 - 30) / 2
    cards_y = top + 40
    ans = sample['answer'].rstrip('.').lower()
    qlines = wrap(sample['context'], cw - 44, 42)
    elines = wrap(sample['question'], cw - 44, 42)
    card_h = 90 + max(len(qlines), len(elines)) * 54 + 20
    for k, (side, color, lines, head) in enumerate((('quill', QUILL, qlines, 'Quill’s text'),
                                                     ('ember', EMBER, elines, 'Ember’s question'))):
        x = lx + 30 + k * (cw + 30)
        s.rect(x, cards_y, cw, card_h, 'white', 22, color, 3)
        s.text(x + 22, cards_y + 58, head, 42, 700, color)
        y = cards_y + 118
        for line in lines:
            if side == 'quill' and ans in line:
                pre, post = line.split(ans, 1)
                s.rich(x + 22, y, [(pre, INK, 400), (ans, QUILL, 700), (post, INK, 400)], 42)
            else:
                s.text(x + 22, y, line, 42)
            y += 54
    # vertical budget: the pair, Ember's answer, the partner from another episode; slack is shared between the gaps
    ms, gs = 270, 230
    dl = wrap(donor['context'], lw - 140 - 60, 42)
    donor_h = 44 + len(dl) * 54
    sh_need = 80 + gs + 24 + donor_h + 30
    used = card_h + ms + 100 + sh_need
    gap = (grid_h - 40 - 30 - used) / 2.5
    assert gap > 24, 'left column overflows'
    # the pair, linked both ways
    my = cards_y + card_h + gap
    qmx, emx = lx + 30 + cw / 2 - ms / 2, lx + 30 + cw + 30 + cw / 2 - ms / 2
    for x in (qmx, emx):
        s.line(x + ms / 2, cards_y + card_h + 4, x + ms / 2, my + 4, MUTED, 3)
    icon_link(s, qmx, emx, my, ms)
    s.mascot('quill', qmx, my, ms)
    s.mascot('ember', emx, my, ms)
    s.text((qmx + ms + emx) / 2, my + ms * rings()[0] - 30, 'link', 42, 700, LINK, 'middle')
    # Ember's answer
    ty = my + ms + gap * 0.5
    s.line(emx + ms / 2, my + ms + 2, emx + ms / 2, ty, MUTED, 3)
    s.rect(lx + 40, ty, lw - 80, 100, 'white', 22, '#D8D2C8', 2)
    s.text(lx + 70, ty + 64, 'Ember’s target answer', 42, 700, EMBER)
    s.text(lx + lw - 70, ty + 66, sample['answer'], 48, 700, RED, 'end')
    # the partner from another episode: Ember linked to Quill given another text
    sy = ty + 100 + gap
    sh = top + grid_h - 30 - sy
    s.rect(lx + 40, sy, lw - 80, sh, COOL, 22)
    s.text(lx + 70, sy + 58, 'Partner from another episode', 42, 700)
    gy = sy + 80
    icon_link(s, lx + 70, lx + lw - 70 - gs, gy, gs)
    s.mascot('quill', lx + 70, gy, gs)
    s.mascot('ember', lx + lw - 70 - gs, gy, gs)
    s.rect(lx + 70 + gs * (ART[0] - 0.08), gy - 6, gs * (ART[1] - ART[0] + 0.16), gs + 12, 'none', 18, INK, 3,
           'stroke-dasharray="10 8"')      # Quill's hidden states come from another episode
    dy_ = gy + gs + 24
    s.rect(lx + 70, dy_, lw - 140, donor_h, 'white', 18, FAINT, 2)
    for j, line in enumerate(dl):
        s.text(lx + 100, dy_ + 58 + j * 54, line, 42, 400, INK)

    # Right: per game and receiver, the percent of episodes more probable with the real partner
    s.rect(gx, top, gw, grid_h, WARM, 32)
    name_x = gx + 40
    track, cnt = 200, 225          # bar length; room for the count after the bar
    b0 = name_x + max(width(label, 42) for _, games in GROUPS for _, label in games) + 30
    bx = {'quill': b0, 'ember': b0 + track + cnt}
    assert bx['ember'] + track + cnt + 20 <= gx + gw
    for side, color in (('quill', QUILL), ('ember', EMBER)):
        w = 96 + width(side.title() + ' receives', 42, 700)
        s.mascot(side, bx[side] + (track + 170) / 2 - w / 2 - 20, top + 6, 112)
        s.text(bx[side] + (track + 170) / 2 - w / 2 + 96, top + 76, side.title() + ' receives', 42, 700, color)
    y = top + head_h
    groups_out = []
    for k, (gname, games) in enumerate(GROUPS):
        if k:
            y += group_gap
        for game, label in games:
            s.text(name_x, y + 48, label, 42)
            for side, color in (('quill', QUILL), ('ember', EMBER)):
                mine = sorted([r for r in eps if r['game'] == game and r['receiver'] == side],
                              key=lambda r: r['episode'])
                c = Counter(outcome(r) for r in mine)
                p = sign_p(c['real'], c['shuffled'])
                groups_out.append(dict(game=game, receiver=side, episodes=len(mine), real=c['real'],
                                       shuffled=c['shuffled'], tie=c['tie'], sign_p=p))
                s.rect(bx[side], y + 12, track, 46, '#EFE5D8', 10)
                length = track * c['real'] / len(mine)
                s.rect(bx[side], y + 12, length, 46, color, 10,
                       extra=f'data-game="{game}" data-receiver="{side}" data-real="{c["real"]}"')
                s.rich(bx[side] + track + 16, y + 50, [(f'{c["real"]}', color, 700), (f' of {len(mine)}', INK, 400)],
                       44)
            y += row_h
    s.save(ROOT / 'figures/shared_reading')

    manifest = dict(source=str(SOURCE.relative_to(ROOT)), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    metric='Mean answer/payload-token cross entropy, real partner versus same-game shuffled partner; '
                           'not free-generation accuracy.',
                    tie_rule='Exact equality; ties displayed and omitted only from sign-test denominator.',
                    all_receiver_episodes=len(eps), non_tied_receiver_episodes=totals['real'] + totals['shuffled'],
                    counts=dict(totals), one_sided_sign_test_p=p_all, groups=groups_out,
                    setup=dict(training=training, test=sample, shuffle_example=donor, source=items['source'],
                               interpretation='Verbatim corpus examples of the setup, not claimed to be a measured '
                                              'evaluation episode or a generated answer.'),
                    rows=[dict(r, outcome=outcome(r)) for r in eps])
    SNAPSHOT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(episodes=len(eps), counts=dict(totals), p=p_all)))


if __name__ == '__main__':
    main()
