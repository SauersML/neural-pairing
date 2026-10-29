"""The value games on values never used in training: one worked episode, and every episode against a partner from
another episode of the same game.

Reads results/magic-nl20/results/payload_final.json (the final checkpoint's per-episode teacher-forced cross-entropy
on the value's tokens, with the real partner and with a shuffled partner) and the worked example stored in
results/values_unseen.json. Writes results/values_unseen.json (sign counts and one-sided sign-test p per game and
receiver, the example, and the median probability per game) and draws figures/values_unseen_game.{svg,png,pdf} and
figures/values_unseen_results.{svg,png,pdf}. A probability drawn is exp(-cross-entropy), the geometric mean
probability of the value's tokens. The worked example shows the target replies the cross-entropy is measured on.
--refresh-examples rebuilds the example from the original models' stored replies (REPLIES) with the Games and
payload_ctl code and the local tokenizers; no model is loaded either way.
Usage: python figures/values_unseen.py [--refresh-examples --replies REPLIES]
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, RED, ROOT, SVG, WARM, ART, icon_link, link, width

sys.path.insert(0, str(ROOT / 'pairing'))
GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')
LABELS = ('Words', 'Numbers', 'Phrases', 'Digit strings', 'Sums', 'Name and city', 'Code lookups')
SIDES = ('quill', 'ember')
SOURCE = ROOT / 'results/magic-nl20/results/payload_final.json'
OUT = ROOT / 'results/values_unseen.json'


def chip(s, x, y, t, color=INK, fill=COOL, stroke='none', size=34, weight=400, h=60, pad=18):
    w = width(t, size, weight) + 2 * pad
    s.rect(x, y, w, h, fill, 14, stroke, 3)
    s.text(x + w / 2, y + h / 2 + size * 0.36, t, size, weight, color, 'middle')
    return w


def down_arrow(s, x, y1, y2, color=INK):
    s.line(x, y1, x, y2, color, 4)
    s.path(f'M{x - 12},{y2 - 14} L{x},{y2} L{x + 12},{y2 - 14}', color, 4)


def summary(records):
    """Per game and receiver: how many episodes have shuffle_ce > link_ce, < and ==, and the one-sided sign-test p."""
    out = {}
    for g in GAMES:
        for side in SIDES:
            rec = sorted((r for r in records if r['game'] == g and r['receiver'] == side), key=lambda r: r['episode'])
            assert rec and len({r['episode'] for r in rec}) == len(rec)
            ds = [r['shuffle_ce'] - r['link_ce'] for r in rec]
            assert all(math.isfinite(v) for v in ds)
            pos, neg, ties = sum(v > 0 for v in ds), sum(v < 0 for v in ds), sum(v == 0 for v in ds)
            n = pos + neg
            out[f'{g}/{side}'] = dict(
                total=len(rec), positive=pos, negative=neg, ties=ties,
                one_sided_sign_p=sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n if n else 1.,
                minimum_shuffle_minus_link=min(ds), records=rec)
    return out


def extract_examples(replies, records):
    """The worked example: the relay_word episode where Quill sends 'cipher', its target replies split into tokens,
    the episode whose partner it is scored with as the partner from another episode, and four training words and
    four never seen in training, for the side panels."""
    import games
    import payload_ctl
    from transformers import AutoTokenizer
    from link import MODELS
    eps = payload_ctl.episodes(json.loads(replies.read_text()), 20)
    i, ep = next((i, e) for i, e in enumerate(eps['relay_word'])
                 if e['meta'].get('sender') == 'quill' and e['meta']['value'] == 'cipher')
    donor = next(r['donor'] for r in records
                 if r['game'] == 'relay_word' and r['episode'] == i and r['receiver'] == 'ember')
    pools = games.POOLS
    train_words = [v for v in ('apple', 'cloud', 'river', 'garden', 'stone', 'cat', 'book')
                   if v in pools['words']['train']][:4]
    if len(train_words) < 4:
        train_words = pools['words']['train'][:4]
    test_words = [e['meta']['value'] for e in eps['relay_word']
                  if e['meta']['sender'] == 'quill' and e['meta']['value'] in ('cipher', 'mouth', 'sling', 'project')]
    assert len(test_words) == 4 and set(train_words).isdisjoint(test_words)
    assert all(v in pools['words']['test'] for v in test_words)
    toks = {x: AutoTokenizer.from_pretrained(MODELS[x]['hf'], local_files_only=True) for x in SIDES}
    pieces = {x: [toks[x].decode([v]) for v in toks[x](ep['reply'][x][0], add_special_tokens=False).input_ids]
              for x in SIDES}
    assert ep['meta']['value'] not in ep['task']['ember']
    return dict(episode=i, example=ep, training_values=train_words, test_values=test_words, target_pieces=pieces,
                other_examples={g: es[0]['meta'] for g, es in eps.items()}, donor=eps['relay_word'][donor])


def game_figure(ex):
    W, M = 2400, 36
    value = ex['example']['meta']['value']
    pieces = ex['target_pieces']
    task = ex['example']['task']
    assert f'"{value}"' in task['quill'] and value not in task['ember']
    assert ''.join(pieces['ember']).strip() == ex['example']['reply']['ember'][0]
    # shortest real excerpts of the two prompts
    q_ex = task['quill'][task['quill'].index('tell Ember'):task['quill'].index(f'"{value}"') + len(value) + 3]
    e_ex = task['ember'][:task['ember'].index(' through the link')]
    assert q_ex == f'tell Ember the word "{value}".' and e_ex == 'Read the word Quill sends you'
    top = 130
    lh_ = 240                              # the two value lists, side by side
    ey = top + lh_ + 24                    # the episode panel
    py, ph = ey + 40, 190                  # prompts
    my, mh = py + ph + 56, 548             # models
    ry, rh = my + mh + 56, 210             # replies
    H = ry + rh + 40 + M
    title = 'Linked Ember is scored on writing a word that only Quill’s prompt contains'
    s = SVG(W, H, title=title)
    s.text(M, 88, title, 56, 700)
    # value lists
    lw_ = (W - 2 * M - 24) / 2
    for k, (head, words, fill) in enumerate((('In training', ex['training_values'], COOL),
                                             ('Never seen in training', ex['test_values'], WARM))):
        x0 = M + k * (lw_ + 24)
        s.rect(x0, top, lw_, lh_, fill, 32)
        s.text(x0 + 40, top + 70, head, 46, 700)
        cw_ = (lw_ - 80 - 3 * 20) / 4
        for i, word in enumerate(words):
            cx, cy = x0 + 40 + i * (cw_ + 20), top + 104
            on = word == value
            s.rect(cx, cy, cw_, 100, 'white', 20, QUILL if on else 'none', 4)
            s.text(cx + cw_ / 2, cy + 50 + 17, word, 48, 700 if on else 400, QUILL if on else INK, 'middle')
    # the episode
    px = M
    pw = W - 2 * M
    s.rect(px, ey, pw, H - ey - M, WARM, 32)
    cw = (pw - 80 - 60) / 2
    qx, ex_ = px + 40, px + pw - 40 - cw
    for x, side, color, excerpt in ((qx, 'quill', QUILL, q_ex), (ex_, 'ember', EMBER, e_ex)):
        s.rect(x, py, cw, ph, 'white', 24, color, 3)
        s.text(x + 30, py + 62, f'{side.title()}’s prompt', 44, 700, color)
        if side == 'quill':
            pre, post = excerpt.split(f'"{value}"')
            s.rich(x + 30, py + 140, [('… ' + pre + '“', INK, 400), (value, QUILL, 700), ('”' + post, INK, 400)],
                   48)
        else:
            s.text(x + 30, py + 140, excerpt + ' …', 48)
    # the two models side by side, with the link between them at its 8 layers
    bw, ms = 360, 360
    boxes = {'quill': (W / 2 - 170 - bw, QUILL), 'ember': (W / 2 + 170, EMBER)}     # close, so the strands stay short
    for side, (x, color) in boxes.items():
        down_arrow(s, x + bw / 2, py + ph + 8, my - 8)
        s.rect(x, my, bw, mh, 'white', 24, color, 3)
        s.mascot(side, x + (bw - ms) / 2, my + 40, ms)
        s.text(x + bw / 2, my + mh - 56, side.title(), 50, 700, color, 'middle')
    lx1, lx2 = boxes['quill'][0] + bw, boxes['ember'][0]
    step = (mh - 190) / 7
    link(s, lx1 + 4, lx2 - 4, [my + 60 + i * step for i in range(8)], wave=9, sw=5)
    mid = (lx1 + lx2) / 2
    s.text(mid, my - 16, 'link', 46, 700, LINK, 'middle')
    s.text(mid, my + 60 + 7 * step + 76, '8 layers', 46, 700, LINK, 'middle')
    # target replies, split into tokens
    for x, side, color in ((qx, 'quill', QUILL), (ex_, 'ember', EMBER)):
        bx = boxes[side][0]
        down_arrow(s, bx + bw / 2, my + mh + 8, ry - 8)
        s.rect(x, ry, cw, rh, 'white', 24, color, 3)
        s.text(x + 30, ry + 60, f'{side.title()}’s target reply', 44, 700, color)
        cx, cy = x + 30, ry + 98
        shown = pieces[side] if side == 'ember' else pieces[side][:5]
        for piece in shown:
            t = piece.strip() or '·'
            on = side == 'ember' and t == value
            w = chip(s, cx, cy, t, RED if on else color, WARM if on else COOL, RED if on else 'none', 46,
                     700 if on else 400, 80, 16)
            cx += w + 10
        assert cx < x + cw
        if side == 'quill':
            s.text(cx + 4, cy + 54, '…', 46, 400, color)
    s.save(ROOT / 'figures/values_unseen_game')


def fmt_p(p):
    if p >= 0.9995:
        return f'{100 * p:.2f}%'
    if p >= 0.001:
        return f'{100 * p:.1f}%'
    if p >= 0.0001:
        return f'{100 * p:.2f}%'
    return '< 0.01%'


def one_in(p):
    n = 1 / p
    for unit, v in (('billion', 1e9), ('million', 1e6), ('thousand', 1e3)):
        if n >= v:
            return f'1 in {n / v:.0f} {unit}'
    return f'1 in {n:.0f}'


def results_figure(stats, ex):
    """Draws the results figure and returns the median probability per game with each partner."""
    W, M = 2400, 36
    total = sum(v['total'] for v in stats.values())
    assert sum(v['positive'] for v in stats.values()) == total     # the title says all
    sample = next(r for r in stats['relay_word/ember']['records'] if r['episode'] == ex['episode'])
    value = ex['example']['meta']['value']
    pr = math.exp(-sample['link_ce'])
    ps = math.exp(-sample['shuffle_ce'])
    assert sample['n_tokens'] == 1
    top = 130
    ex_h = 390
    grid_y = top + ex_h + 24
    row_h, head_h = 115, 220
    grid_h = head_h + len(GAMES) * row_h + 20
    H = grid_y + grid_h + M
    title = f'Values never seen in training are more probable with the real partner in all {total} episodes'
    s = SVG(W, H, title=title)
    s.text(M, 88, title, 54, 700)

    # one episode, both partners
    cw = (W - 2 * M - 24) / 2
    for k, (head, p, grey, fill) in enumerate((('Real partner', pr, False, WARM),
                                               ('Partner from another episode', ps, True, COOL))):
        x = M + k * (cw + 24)
        s.rect(x, top, cw, ex_h, fill, 32)
        s.text(x + 40, top + 72, head, 46, 700, RED if not grey else INK)
        my = top + 100
        icon_link(s, x + 20, x + 270, my, 190)
        s.mascot('quill', x + 20, my, 190)
        s.mascot('ember', x + 270, my, 190)
        if grey:        # Quill's hidden states come from another episode
            s.rect(x + 20 + 190 * (ART[0] - 0.08), my - 6, 190 * (ART[1] - ART[0] + 0.16), 202, 'none', 18, INK, 3,
                   'stroke-dasharray="10 8"')
        tx = x + 530
        s.text(tx, top + 170, f'probability of “{value}”', 44, 400, INK)
        s.text(tx, top + 290, fmt_p(p) if p > 0.001 else one_in(p), 96, 700, RED if not grey else INK)
        chip(s, x + 30, my + 206, value if not grey else 'other word', QUILL if not grey else INK,
             '#EFE8FC' if not grey else 'white', 'none', 42, 700 if not grey else 400, 60, 14)

    # every game: episodes where the value is more probable with the real partner, and the median probabilities
    s.rect(M, grid_y, W - 2 * M, grid_h, WARM, 32)
    name_x = M + 40
    cx = {'quill': M + 720, 'ember': M + 1270}          # each count ends here and its ' of N' starts
    real_x, shuf_x = M + 1840, W - M - 60
    for side, color in (('quill', QUILL), ('ember', EMBER)):
        w = 96 + width(side.title() + ' receives', 46, 700)
        s.mascot(side, cx[side] + 40 - w / 2 - 22, grid_y + 18, 112)
        s.text(cx[side] + 40 - w / 2 + 96, grid_y + 86, side.title() + ' receives', 46, 700, color)
    # a value of several tokens is drawn as its geometric mean probability per token
    head = 'median probability per token'
    s.text(shuf_x, grid_y + 70, head, 44, 700, INK, 'end')
    assert shuf_x - width(head, 44, 700) > cx['ember'] + 40 + (96 + width('Ember receives', 46, 700)) / 2 + 30
    s.text(real_x, grid_y + 184, 'real partner', 44, 700, RED, 'end')
    s.text(shuf_x, grid_y + 134, 'partner from', 44, 700, INK, 'end')
    s.text(shuf_x, grid_y + 184, 'another episode', 44, 700, INK, 'end')
    y = grid_y + head_h
    medians = {}
    for g, label in zip(GAMES, LABELS):
        base = y + row_h / 2 + 24
        s.text(name_x, base, label, 46, 700)
        for side in SIDES:
            st = stats[f'{g}/{side}']
            s.text(cx[side], base + 4, st['positive'], 72, 700, RED, 'end')
            s.text(cx[side], base, f' of {st["total"]}', 46, 400, INK)
        recs = stats[f'{g}/quill']['records'] + stats[f'{g}/ember']['records']
        real = statistics.median(math.exp(-r['link_ce']) for r in recs)
        shuf = statistics.median(math.exp(-r['shuffle_ce']) for r in recs)
        medians[g] = dict(real=real, shuffled=shuf)
        s.text(real_x, base, fmt_p(real), 46, 700, RED, 'end')
        s.text(shuf_x, base, fmt_p(shuf), 46, 400, INK, 'end')
        y += row_h
    s.save(ROOT / 'figures/values_unseen_results')
    return medians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-examples', action='store_true', help='rebuild the worked example from --replies')
    ap.add_argument('--replies', help="the original models' stored replies (JSON), needed with --refresh-examples")
    args = ap.parse_args()
    if args.refresh_examples and not args.replies:
        ap.error('--refresh-examples needs --replies')
    data = json.loads(SOURCE.read_text())
    selected = [r for r in data if r['game'] in GAMES]
    st = summary(selected)
    if args.refresh_examples:
        ex = extract_examples(Path(args.replies), data)
        replies_source = str(Path(args.replies).resolve()).replace(str(Path.home()), '~')
    else:
        old = json.loads(OUT.read_text())
        ex, replies_source = old['examples'], old['example_replies_source']
    report = dict(
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measurement='Mean teacher-forced cross-entropy on payload tokens; positive iff shuffle_ce - link_ce > 0; '
                    'ties exactly equal.',
        precision='Existing real-model GPU result; no Mac model inference performed.',
        records=len(selected), positive=sum(v['positive'] for v in st.values()),
        ties=sum(v['ties'] for v in st.values()), groups=st, examples=ex, example_replies_source=replies_source,
        example_note='The Games target replies, not generated linked-model replies. Vector strips are schematic, '
                     'not activation measurements. Word pools are disjoint as values, not a claim about pretraining '
                     'or incidental text.')
    game_figure(ex)
    report['median_probability_per_value_token'] = results_figure(st, ex)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('records', 'positive', 'ties')}))


if __name__ == '__main__':
    main()
