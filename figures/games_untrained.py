"""The five games the nl21 checkpoint never saw in training: one real example of each, and per receiver the count of
episodes where the value is more probable with the real partner.

Reads results/magic-nl21/results/payload_final.json and the worked examples stored in results/games_untrained.json
(real evaluation episodes built with payload_ctl.episodes(n=40) from the run's stored replies and the coread data:
inputs and target replies, shortened as that file's example_meaning describes). Writes results/games_untrained.json
(per game and receiver, trained games included: episode counts, wins, ties, one-sided sign-test p) and
figures/games_untrained.{svg,png,pdf}. An episode counts when the receiver's mean cross-entropy on the value's tokens
is lower with its real partner than with a partner from another episode of the same game. No model is loaded.
Usage: python figures/games_untrained.py
"""
import hashlib
import json
import math

from scipy.stats import binomtest

from chat_names import EMBER, INK, LINK, QUILL, RED, ROOT, SVG, WARM, link, width, wrap

SOURCE = ROOT / 'results/magic-nl21/results/payload_final.json'
OUT = ROOT / 'figures/games_untrained'
REPORT = ROOT / 'results/games_untrained.json'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
MONO = 'DejaVu Sans Mono, Menlo, monospace'

NEW = ['lookup', 'relay_digits', 'co_translate', 'co_code_out', 'co_mcq']
TRAINED = ['relay_word', 'relay_number', 'relay_phrase', 'merge_sum', 'merge_fact',
           'co_transfer', 'co_swap', 'co_split', 'co_continue', 'co_distill', 'co_talk',
           'co_qa', 'co_hop', 'co_paraphrase', 'co_nli', 'co_math', 'co_code_write', 'co_instruct']
LABELS = dict(zip(NEW, ['Look up a code', 'Copy digits', 'Translate', 'Read code', 'Choose an answer']))
LABELS.update(dict(zip(TRAINED, ['Copy a word', 'Copy a number', 'Copy a phrase', 'Add numbers', 'Join name and city',
    'Copy a passage', 'Swap passages', 'Join text halves', 'Continue a text', 'Match original reply',
    'Answer, then talk', 'Answer a question', 'Join two facts', 'Paraphrase', 'Compare sentences', 'Solve a problem',
    'Write code', 'Follow a request'])))

def collect():
    records = json.loads(SOURCE.read_text())
    assert set(r['game'] for r in records) == set(NEW + TRAINED + ['co_play'])
    groups = {}
    for game in NEW + TRAINED:
        for side in ('quill', 'ember'):
            rs = sorted([r for r in records if (r['game'], r['receiver']) == (game, side)], key=lambda r: r['episode'])
            assert len({r['episode'] for r in rs}) == len(rs)
            differences = [r['shuffle_ce'] - r['link_ce'] for r in rs]
            assert all(math.isfinite(d) for d in differences)
            wins = sum(d > 0 for d in differences)
            losses = sum(d < 0 for d in differences)
            groups[game + '/' + side] = dict(n=len(rs), wins=wins, losses=losses, ties=len(rs) - wins - losses,
                p=binomtest(wins, wins + losses, .5, alternative='greater').pvalue if wins + losses else 1.,
                records=rs, differences=differences)
    example_source = json.loads(REPORT.read_text())['examples']
    selected = [v for k, v in groups.items() if k.split('/')[0] in NEW]
    report = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measurement='Per receiver episode: real-partner payload CE strictly below same-game shuffled-partner payload CE.',
        significance=dict(test='scipy.stats.binomtest', alternative='greater', null_probability=.5,
            threshold=.05, adjustment='none', mark='Star: p below 0.05, one-sided.',
            ties='Exact zero differences excluded from the test; all episodes drawn.'),
        dot_order='Original episode ID within each game and receiver; no sorting by outcome.',
        omitted_game='co_play: rhyme is mixed with trained play kinds and cannot be isolated in this file.',
        example_meaning='Actual task inputs and target replies of single evaluation episodes, not generated text. '
            'The figure shortens some prompts: the lookup table shows 3 of its 8 codes and drops the closing request, '
            'the reading prompts show the content and the reader\'s own question without the framing sentence, and '
            'the options prompt shows the options and its last sentence. Examples show Quill holding the content; '
            'the dots test each direction.',
        examples=example_source, groups=groups,
        untrained=dict(wins=sum(v['wins'] for v in selected), n=sum(v['n'] for v in selected),
            ties=sum(v['ties'] for v in selected), directions=len(selected),
            maximum_p=max(v['p'] for v in selected)),
        precision='Existing GPU payload evaluation; no Mac model inference used.',
        findings=['The first translation example considered (episode 3: English "Who is this?" against Spanish '
                  '"¿Quién la llama?") is not a literal translation, so the diagram uses episode 25; every '
                  'translation episode is in the dot grid.'],
        trained=dict(wins=sum(groups[g + '/' + s]['wins'] for g in TRAINED for s in ('quill', 'ember')),
                     n=sum(groups[g + '/' + s]['n'] for g in TRAINED for s in ('quill', 'ember'))))
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def scene(ex, g):
    """(Quill's prompt lines, Ember's prompt lines, whose target reply, the target as (text, scored) runs, whether
    Quill's lines are code): the shortest real excerpts of the two prompts that show the game."""
    ep = ex['episode_data']
    meta, reply, task = ep['meta'], ep['reply'], ep['task']
    if g == 'lookup':
        k, v = meta['key'], meta['value']
        entries = [f'{a}: {b}' for a, b in meta['table'].items()]
        assert meta['asker'] == 'ember' and task['ember'].startswith(f'Your code is {k}.') \
            and reply['ember'][0] == f'Code {k} holds the word {v}.' and all(e in task['quill'] for e in entries)
        return [entries[0], entries[1], '…'], [f'Your code is {k}.'], 'Ember’s target', \
            [(f'Code {k} holds the word ', False), (v, True), ('.', False)], False
    if g == 'relay_digits':
        v = meta['value']
        q = f'send "{v}" to Ember.'
        e = 'Read the string of digits Quill sends you'
        assert meta['sender'] == 'quill' and reply['ember'][0] == f'The digits are {v}.'
        assert q in task['quill'] and task['ember'].startswith(e)
        return ['… ' + q], [e + ' …'], 'Ember’s target', [('The digits are ', False), (v, True), ('.', False)], False
    if g == 'co_translate':
        it = meta['item']
        e = f'write it in {it["target_lang"]}.'
        assert meta['reader'] == 'quill' and reply['ember'][0] == it['target'] and it['source'] in task['quill']
        assert task['ember'].endswith(e)
        return [it['source']], ['… ' + e], 'Ember’s target', [(it['target'], True)], False
    if g == 'co_code_out':
        it = meta['item']
        assert meta['reader'] == 'quill' and reply['ember'][0] == f"It returns {it['output']}."
        assert it['code'] in task['quill'] and it['call'] in task['quill']
        return it['code'].splitlines() + [it['call']], [task['ember']], 'Ember’s target', [(reply['ember'][0], True)], True
    it = meta['item']
    assert meta['question'] == 'quill' and reply['quill'][0] == reply['ember'][0] and it['question'] in task['quill']
    L = 'ABCD'
    opts = [f'({L[i]}) {o}' for i, o in enumerate(it['options'])]
    assert all(o in task['ember'] for o in opts)
    return [it['question']], opts, 'Each twin’s target', [(reply['ember'][0], True)], False


def rich_wrap(runs, w, size):
    """Greedy word wrap of (text, scored) runs; returns lines of runs."""
    words = []
    for txt, scored in runs:
        if scored and len(txt) <= 16:          # a short value (a digit string) stays on one line
            words.append((txt, scored))
            continue
        for k, part in enumerate(txt.split(' ')):
            if k:
                words.append((' ', scored))
            if part:
                words.append((part, scored))
    lines, cur, cur_w = [], [], 0
    for word, scored in words:
        ww = width(word, size, 700 if scored else 400)
        if word != ' ' and cur and cur_w + ww > w:
            while cur and cur[-1][0] == ' ':
                cur.pop()
            lines.append(cur)
            cur, cur_w = [], 0
        if word == ' ' and not cur:
            continue
        cur.append((word, scored))
        cur_w += ww
    return lines + ([cur] if cur else [])


def render(report):
    W, M = 2400, 36
    body, lh, mono_lh = 44, 56, 50
    qx, qw = M + 40, 630          # Quill card (40 px code lines fit)
    lw = 96                       # link
    ex_, ew = qx + qw + lw, 440   # Ember card
    wx, ww = ex_ + ew + 50, 420   # target reply
    cx = {'quill': W - M - 520, 'ember': W - M - 220}     # each count ends here and its ' of N' starts
    head = 78                     # card header height

    games = []
    for g in NEW:
        q, e, who, runs, mono = scene(report['examples'][g], g)
        qlines = []
        for t in q:
            qlines += [t] if mono else wrap(t, qw - 48, body)
        elines = []
        for t in e:
            elines += wrap(t, ew - 48, body)
        wlines = rich_wrap(runs, ww - 48, 40 if mono else body)
        qh = len(qlines) * (mono_lh if mono else lh)
        card_h = head + max(qh, len(elines) * lh) + 26
        write_h = head + len(wlines) * lh + 26
        games.append(dict(g=g, q=qlines, e=elines, who=who, w=wlines, mono=mono, card_h=card_h,
                         write_h=write_h, h=70 + max(card_h, write_h) + 44))

    top = 120
    p_head = 126
    p_h = p_head + sum(r['h'] for r in games) - 14
    H = top + p_h + M
    u = report['untrained']
    assert len(NEW) == 5
    title = f'In games never seen in training, {u["wins"]} of {u["n"]} values are more probable with the real partner'
    s = SVG(W, H, title=title)
    s.text(M, 84, title, 52, 700)

    s.rect(M, top, W - 2 * M, p_h, WARM, 32)
    for side, color in (('quill', QUILL), ('ember', EMBER)):
        x0 = cx[side] + 40 - (76 + width('receives', 40, 700)) / 2
        s.mascot(side, x0 - 14, top + 22, 90)
        s.text(x0 + 76, top + 62, side.title(), 40, 700, color)
        s.text(x0 + 76, top + 106, 'receives', 40, 700, color)
    y = top + p_head
    for j, r in enumerate(games):
        if j:
            s.line(M + 30, y - 22, W - M - 30, y - 22, '#E6DDD2', 2)
        s.text(qx, y + 44, LABELS[r['g']], 46, 700)
        cy = y + 70
        # Quill card
        s.rect(qx, cy, qw, r['card_h'], 'white', 20, QUILL, 3)
        s.mascot('quill', qx - 2, cy + 3, 74)
        s.text(qx + 70, cy + 48, 'Quill', 42, 700, QUILL)
        ty = cy + head + 42
        for line in r['q']:
            if r['mono']:
                s.text(qx + 24, ty, line, 40, 400, INK, family=MONO)
                ty += mono_lh
            else:
                s.text(qx + 24, ty, line, body, 400, INK)
                ty += lh
        # Ember card
        s.rect(ex_, cy, ew, r['card_h'], 'white', 20, EMBER, 3)
        s.mascot('ember', ex_ - 2, cy + 3, 74)
        s.text(ex_ + 70, cy + 48, 'Ember', 42, 700, EMBER)
        ty = cy + head + 42
        for line in r['e']:
            s.text(ex_ + 24, ty, line, body, 400, INK)
            ty += lh
        # the link between the two models, both ways
        link(s, qx + qw + 3, ex_ - 3, [cy + 36], wave=5, sw=4)
        s.text(qx + qw + lw / 2, cy + 90, 'link', 40, 700, LINK, 'middle')
        # the target reply, sized to its text
        s.path(f'M{ex_ + ew + 8},{cy + 36} H{wx - 10}', INK, 4)
        s.path(f'M{wx - 24},{cy + 24} L{wx - 10},{cy + 36} L{wx - 24},{cy + 48}', INK, 4)
        s.rect(wx, cy, ww, r['write_h'], 'white', 20, '#D8D2C8', 2)
        s.text(wx + 24, cy + 48, r['who'], 42, 700, INK)
        fam = MONO if r['mono'] else FAMILY
        size = 40 if r['mono'] else body
        ty = cy + head + 42
        for line in r['w']:
            s.rich(wx + 24, ty, [(txt, RED if scored else INK, 700 if scored else 400) for txt, scored in line],
                   size, family=fam)
            ty += lh
        # the count per receiver: episodes where the value is more probable with the real partner
        for side in ('quill', 'ember'):
            row = report['groups'][f'{r["g"]}/{side}']
            s.text(cx[side], cy + 74, row['wins'], 72, 700, RED, 'end')
            s.text(cx[side], cy + 70, f' of {row["n"]}', 46, 400, INK)
        y += r['h']
    s.save(OUT)
    print(json.dumps(report['untrained']))


if __name__ == '__main__':
    render(collect())
