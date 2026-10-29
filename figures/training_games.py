"""Every task the final twins trained on, one row each and grouped by family: the percent of each training stage's
episodes that were the task, and the task's episodes in all. A last row gives the episodes with silent time steps.

The numbers are the episodes training drew, not probabilities. --refresh replays the draws of the four runs on the
final twins' path: nl20's warm-up and full mix, nl22c's chat training and nl33w (chat training continued). Each run
draws from a fresh random.Random(0) and fresh games.Games and coread.Coread drawers of seed 0, so the replay builds
them the same way and calls train.py's own draw_any, draw and make_batch (compiled from pairing/train.py's source)
with each stage's flags, read from its chain script, for each run's number of steps (the last step of its training
log). link.py and coldstart.py import torch, so the replay compiles the parts of them that drawing uses (MODELS, SIDES,
partner; relay_episode) from their source and needs no torch. The replay is checked against the training logs: every 20
steps a log lists the games drawn in those steps that have a target of weight 1, and the replay must list the same
games in every window (a game absent from a window must be absent from the replay's window too). --refresh also writes
the probabilities the flags give, as expected episodes, next to the counts.

A task is a game name of the logs or of draw_any, and each co_play game (guess, gift, where, story, rhyme) is its own
task. 'plain' and 'chat' have no target of weight 1, so the logs never list them.

--refresh needs nl33's data (replies.json and data/{coread,talk,play,chat,chat2,pools}.json in one folder; the full
mix ran on nl20's inputs, which the log check covers) and nl20's full-mix log (run/log.json; results/magic-nl20/log.json
stops at step 100). It writes results/training_games.json; without --refresh the figure is drawn from that file.
Writes figures/training_games.{svg,png,pdf}. Needs the Inter font and rsvg-convert.
Usage: python figures/training_games.py [--refresh --data DATA_DIR --mix_log NL20_RUN/log.json]
"""
import argparse
import ast
import collections
import json
import os
import random
import sys
import types

from chat_names import COOL, EMBER, INK, LINK, QUILL, ROOT, SVG, icon_link, width, wrap
from method_parts import check_shape

OUT = ROOT / 'figures/training_games'
SNAP = ROOT / 'results/training_games.json'
CODE = ROOT / 'pairing'
LOGS = {'warm_up': ROOT / 'results/magic-nl20/warm/log.json', 'chat': ROOT / 'results/magic-nl22c/log.json',
        'final': ROOT / 'results/magic-nl33w/log.json'}

# stage, its chain script, the train.py flags there that set the draws (the warm-up's VALUES and the full mix's PEN
# are the chain's variables), the key of its step count in results/training_stages.json, and its title on the figure
STAGES = [
    ('warm_up', 'nl20.sh', ('--relay_first 100000 --relay_games $VALUES --steps 900 --bs 16 --coread 0 --anchor 0 '
                            '--dialogue 0'), 'warm_up', 'Warm-up'),
    ('full_mix', 'nl20.sh', 'MIX="--coread 0.35 --dialogue 0.15 --anchor 0.1 $PEN"|--pause 4 --pause_p 0.5',
     'full_mix', 'Full mix'),
    ('chat', 'nl22.sh', 'MIX="--coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2"|--pause 4 --pause_p 0.5',
     'chat', 'Chat training'),
    ('final', 'nl33.sh', ('write) MIX= ;;|--coread 0.35 --dialogue 0.15 --anchor 0.1 --pen 0.05 --chat 0.2 --chat2 0.25|'
                          '--pause 4 --pause_p 0.5'), 'same33w', 'More chat training'),
]
CHAIN_VARS = ('$VALUES', '$PEN')
# the flags draw_any reads, in its order
DRAW_ORDER = ['relay_first', 'solo', 'text_pair', 'steer', 'anchor', 'chat', 'chat2', 'pen', 'pen_chunk', 'coread', 'qa',
              'dialogue']

# families, each with its tasks in plain words
FAMILIES = [
    ('Value games', [
        ('relay_word', "Write the partner's secret word"),
        ('relay_number', "Write the partner's secret number"),
        ('relay_phrase', "Write the partner's two-word phrase"),
        ('relay_digits', "Write the partner's string of digits"),
        ('merge_sum', "Add own number to the partner's"),
        ('merge_fact', 'Join a name and a city, one each'),
        ('lookup', "Look up a code in the partner's table")]),
    ('Steering, identity, own answers', [
        ('steer', 'Make the partner mention a concept'),
        ('identity', 'Say who you are, or who the partner is'),
        ('plain', 'Answer as the original model did')]),
    ('Dialogue games', [
        ('chat', 'Ask and answer questions in turn'),
        ('chat_identity', 'Chat, then say who you are'),
        ('long_relay', 'Recall a word sent early in the chat'),
        ('teach', 'Learn a made-up fact, then take a quiz'),
        ('hide_seek', "Guess the partner's unspoken concept"),
        ('mirror', 'Agree on one of two private concepts'),
        ('listen_steer', 'Steer the partner while only listening'),
        ('listen_report', 'Write a word sent during an answer')]),
    ('Shared pen', [
        ('pen_free', 'Write a poem together, a word each'),
        ('pen_secret', "End the poem on the partner's word")]),
    ('Shared reading', [
        ('co_transfer', 'Write out the text the partner reads'),
        ('co_swap', "Each writes out the partner's text"),
        ('co_split', 'Join two halves of a text, one each'),
        ('co_continue', 'Continue the text the partner reads'),
        ('co_distill', "Reply as if reading the partner's text"),
        ('co_talk', "Answer about the partner's text, then talk")]),
    ("Tasks on the partner's text", [
        ('co_qa', "Answer a question from the partner's text"),
        ('co_hop', 'Answer from two paragraphs, one each'),
        ('co_translate', "Translate the partner's sentence"),
        ('co_paraphrase', "Paraphrase the partner's sentence"),
        ('co_nli', 'Say if one sentence implies the other'),
        ('co_math', "Solve the partner's word problem"),
        ('co_code_out', "Say what the partner's code returns"),
        ('co_code_write', "Write code for the partner's problem"),
        ('co_instruct', "Follow the partner's instruction"),
        ('co_mcq', 'Join a question and options, one each')]),
    ('Play', [
        ('co_play:guess', "Guess the partner's secret word"),
        ('co_play:gift', 'Give the gift the partner wishes for'),
        ('co_play:where', 'Say which city the partner is in'),
        ('co_play:story', "Put the partner's character in a story"),
        ('co_play:rhyme', "End a rhyme on the partner's word")]),
    ('Free chat', [
        ('co_chat', 'Chat when a twin has a private city'),
        ('co_chat2', 'Chat when a twin has a private pet, job, …')]),
]
SILENT = ('Silent steps', 'Twins write 4 spaces before replies')


def literal(path, names):
    """{name: value} of the module-level literal assignments names in path."""
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    assert set(out) == set(names), (path, names)
    return out


def code_tasks():
    """Every task the drawers can make: the game names of games.py and coread.py, with co_play split by game."""
    g = literal(CODE / 'games.py', ['GAMES', 'DIALOGUE', 'PEN'])
    c = literal(CODE / 'coread.py', ['FAMILY', 'TASK_GAME', 'PLAY'])
    assert "game not in ('co_chat', 'co_chat2')" in (CODE / 'coread.py').read_text()
    fams = [f for f in c['FAMILY'] if f not in ('co_task', 'co_play')]
    return (list(g['GAMES']) + list(g['DIALOGUE']) + list(g['PEN']) + fams + list(c['TASK_GAME'].values())
            + [f'co_play:{k}' for k in c['PLAY']] + ['co_chat', 'co_chat2'])


def compiled(path, names, ns):
    """Run the module-level definitions names (functions, classes, assignments) of path's source in ns."""
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    found = {n.name if hasattr(n, 'name') else n.targets[0].id for n in body}
    assert found == set(names), (path, set(names) - found)
    exec(compile(ast.Module(body, []), str(path), 'exec'), ns)
    return ns


def defaults():
    """The defaults of train.py's and coldstart.py's command-line flags."""
    out = {}
    for path in (CODE / 'train.py', CODE / 'coldstart.py'):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'add_argument' and n.args and \
                    isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith('--'):
                kw = {k.arg: k.value for k in n.keywords}
                name = n.args[0].value[2:]
                if 'default' in kw:
                    try:
                        out[name] = ast.literal_eval(kw['default'])
                    except ValueError:              # a computed default (--xblocks): no draw reads it
                        pass
                elif isinstance(kw.get('action'), ast.Constant) and kw['action'].value == 'store_true':
                    out[name] = False
    return out


def stage_flags(chain, spec):
    """The flags of one stage: each '|'-separated part of spec must be in the chain script; $VALUES and $PEN are the
    chain's values of those variables."""
    text = (ROOT / 'chains' / chain).read_text()
    flags = {}
    for part in spec.split('|'):
        assert part in text, (chain, part)
        if part.startswith('MIX='):
            part = part[5:-1]
        elif part.startswith('write)'):
            continue
        for var in CHAIN_VARS:
            if var in part:
                line = next(ln for ln in text.splitlines() if ln.startswith(var[1:] + '='))
                part = part.replace(var, line.split('=', 1)[1].strip('"'))
        words = part.split()
        for k in range(len(words)):
            if words[k].startswith('--'):
                if k + 1 < len(words) and not words[k + 1].startswith('--'):
                    flags[words[k][2:]] = words[k + 1]
                else:
                    flags[words[k][2:]] = True
    return flags


def task_of(ep):
    return f"co_play:{ep['meta']['play']}" if ep['game'] == 'co_play' else ep['game']


def weighted(ep):
    """Whether the episode has a target of weight 1 (train.py's per-game log counts only those)."""
    if ep['mode'] == 'duplex':
        return any(w >= 1 for _, w in ep['reply'].values())
    if ep['mode'] == 'pen':
        return ep.get('pen_weight', 1.0) >= 1
    return any(w >= 1 for _, _, w in ep['turns'])


def expected(a, G, co, steps):
    """{task: probability} of one stage's draws, from draw_any's order and the drawers' weights."""
    if a.relay_first >= steps:
        games = a.relay_games.split(',')
        return {g: 1 / len(games) for g in games}
    p, rest = collections.Counter(), [1.0]

    def take(prob, dist):
        for k, v in dist.items():
            p[k] += rest[0] * prob * v
        rest[0] *= 1 - prob
    import coread
    take(a.anchor, {'identity': .5, 'plain': .5})
    if a.chat:
        take(a.chat, {'co_chat': 1})
    if a.chat2:
        take(a.chat2, {'co_chat2': 1})
    if a.pen:
        take(a.pen, {g: 1 / len(G.PEN) for g in G.PEN})
    fams = {f: w for f, w in coread.FAMILY.items() if (f != 'co_distill' or co.distill) and (f != 'co_talk' or co.talk)
            and (f != 'co_play' or co.play)}
    tasks = collections.Counter()
    for src, w in zip(co.task_src, co.tw):            # a source drawn by weight, then one of its items
        items = co.by_src[src]
        for t, n in collections.Counter(it['task'] for it in items).items():
            tasks[coread.TASK_GAME[t]] += w * n / len(items)
    kinds = collections.Counter(c['game'] for c in co.play)
    dist = collections.Counter()
    for f, w in fams.items():
        share = w / sum(fams.values())
        if f == 'co_task':
            for t, tw in tasks.items():
                dist[t] += share * tw / sum(tasks.values())
        elif f == 'co_play':
            for k, n in kinds.items():
                dist[f'co_play:{k}'] += share * n / sum(kinds.values())
        else:
            dist[f] += share
    take(a.coread, dist)
    assert a.qa == 0
    take(a.qa, {})
    take(a.dialogue, {g: 1 / len(G.DIALOGUE) for g in G.DIALOGUE})
    take(1.0, {g: 1 / len(G.GAMES) for g in G.GAMES})
    assert abs(sum(p.values()) - 1) < 1e-9
    return dict(p)


def refresh(data, mix_log):
    link = types.ModuleType('link')
    link.os = os
    compiled(CODE / 'link.py', ['MODELS', 'SIDES', 'partner', 'XBLOCKS'], link.__dict__)
    sys.modules['link'] = link
    cold = types.ModuleType('coldstart')
    cold.MODELS, cold.SIDES = link.MODELS, link.SIDES
    compiled(CODE / 'coldstart.py', ['KIND', 'VALUE_GAMES', 'relay_episode'], cold.__dict__)
    sys.modules['coldstart'] = cold
    sys.path.insert(0, str(CODE))
    import coread
    import games as G

    draw_any = next(n for n in ast.parse((CODE / 'train.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'draw_any')
    order = sorted((n.lineno, n.col_offset, n.attr) for n in ast.walk(draw_any) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id == 'args')
    assert list(dict.fromkeys(a for *_, a in order)) == DRAW_ORDER, order

    replies = json.load(open(os.path.join(data, 'replies.json')))
    co_path = os.path.join(data, 'data', 'coread.json')
    base = defaults()
    steps_ref = json.loads((ROOT / 'results/training_stages.json').read_text())['steps']
    logs = dict(LOGS, full_mix=mix_log)
    co, fresh = None, None
    out = dict(stages=[], tasks=code_tasks())
    for name, chain, spec, ref, _ in STAGES:
        flags = stage_flags(chain, spec)
        a = types.SimpleNamespace(**base)
        for k, v in flags.items():
            setattr(a, k, type(base[k])(v) if not isinstance(base[k], bool) else v)
        log = json.load(open(logs[name]))
        steps = max(r['step'] for r in log)
        assert steps == steps_ref[ref], (name, steps, steps_ref[ref])
        ns = dict(args=a, G=G, coldstart=cold, RELAY_GAMES=tuple(a.relay_games.split(',')), ONLY_QA=False,
                  STEER=None, HOLD_OUT=set(), FROZEN=None, BENCH=None, FOCUS=None)
        compiled(CODE / 'train.py', ['draw_any', 'draw', 'make_batch', 'held_out', 'to_target', 'bench_item',
                                     'ANCHOR'], ns)
        games = G.Games(replies, 'train', seed=a.seed, max_digits=a.max_digits)
        stage_co = None
        if a.coread > 0 or a.chat2 > 0:
            if co is None:
                co = coread.Coread(replies, 'train', seed=a.seed, path=co_path)
                fresh = dict(co.__dict__)
            co.__dict__.update(fresh)
            co.rng = random.Random(a.seed)              # the only state a draw changes
            stage_co = co
        rng = random.Random(a.seed)
        counts, silent, windows, win = collections.Counter(), 0, {}, set()
        for step in range(1, steps + 1):
            for ep in ns['make_batch'](rng, games, step, stage_co):
                counts[task_of(ep)] += 1
                silent += bool(ep.get('pause') or ep.get('pause_turns'))
                if weighted(ep):
                    win.add(ep['game'])
            if step % 20 == 0:
                windows[step], win = win, set()
        logged = {r['step']: set(r['games']) for r in log if 'games' in r}
        names = set().union(*logged.values())
        bad = [s for s in logged if windows[s] != logged[s]]
        if bad:
            raise ValueError(f'{name}: the replay differs from the log at steps {bad[:10]}')
        absent = sum(len(names - logged[s]) for s in logged)
        prob = expected(a, G, stage_co, steps)
        eps = steps * a.bs
        assert sum(counts.values()) == eps and set(counts) <= set(out['tasks']), name
        print(f'{name}: {steps} steps, {eps} episodes, {len(counts)} tasks; {len(logged)} log windows match '
              f'({absent} absences)', flush=True)
        out['stages'].append(dict(stage=name, chain=chain, flags=flags, steps=steps, episodes=eps,
                                  log=('nl20 run/log.json' if name == 'full_mix' else
                                       str(logs[name].relative_to(ROOT))),
                                  log_windows=len(logged), log_absences=absent,
                                  counts=dict(sorted(counts.items())), silent=silent,
                                  expected={k: round(v * eps, 1) for k, v in sorted(prob.items())}))
    SNAP.write_text(json.dumps(out, indent=1) + '\n')
    return out


def render(snap):
    stages = snap['stages']
    tasks = [k for _, rows in FAMILIES for k, _ in rows]
    assert sorted(tasks) == sorted(snap['tasks']) == sorted(set(tasks)), set(tasks) ^ set(snap['tasks'])
    drawn = set().union(*[s['counts'] for s in stages])
    assert drawn == set(tasks), set(tasks) - drawn
    total = sum(s['episodes'] for s in stages)
    titles = {name: t for name, *_, t in STAGES}

    W, M = 2400, 36
    body, row, pad, gap = 44, 56, 10, 12
    fam_x, fam_w = M + 26, 280
    cell_w, total_w = 220, 200
    total_x = W - M - 16 - total_w
    cell_x = [total_x - (len(stages) - k) * cell_w for k in range(len(stages))]
    text_x = fam_x + fam_w + 32
    text_w = cell_x[0] - 12 - text_x
    for _, rows in FAMILIES:
        for k, t in rows:
            assert width(t, body) <= text_w, (t, width(t, body), text_w)
    top = 112
    head_lines = {s['stage']: wrap(titles[s['stage']], cell_w, body, 700) for s in stages}
    n_head = max(len(v) for v in head_lines.values())
    span_y = top + 44                                     # what the stage columns give
    name_y = span_y + 66                                  # the first line of a stage's title
    num_y = name_y + n_head * 50 + 6                      # its steps
    head_h = num_y + 48 + 22 - top
    y0 = top + head_h
    H = y0 + sum(len(rows) * row + 2 * pad for _, rows in FAMILIES) + (len(FAMILIES) - 1) * gap + \
        gap + 8 + row + 2 * pad + M
    check_shape(W, H)
    title = f'The final twins trained on {len(tasks)} tasks in {total:,} episodes'
    s = SVG(W, H, title=title)
    s.text(M, 76, title, 56, 700)

    # the two twins, joined by the link, each named on its outer side
    size = 168
    qname = max(width('Quill', body + 4, 700), width('Qwen3-4B', body))
    qx = M + 10 + qname
    ex_ = qx + size + 90
    iy = top + (head_h - size) / 2 - 4
    icon_link(s, qx, ex_, iy, size)
    s.mascot('quill', qx, iy, size)
    s.mascot('ember', ex_, iy, size)
    for x, anchor, name, model, color in ((qx + 4, 'end', 'Quill', 'Qwen3-4B', QUILL),
                                          (ex_ + size - 4, 'start', 'Ember', 'AFM-4.5B', EMBER)):
        s.text(x, iy + size / 2 - 6, name, body + 4, 700, color, anchor)
        s.text(x, iy + size / 2 + 48, model, body, 400, color, anchor)
    assert ex_ + size + max(width('Ember', body + 4, 700), width('AFM-4.5B', body)) < cell_x[0] - 20

    # stage headers: what the columns give, then each stage and its steps
    span = "Percent of each stage's episodes"
    assert width(span, body) <= len(stages) * cell_w - 40
    s.text(cell_x[0] + len(stages) * cell_w / 2, span_y, span, body, 400, INK, 'middle')
    s.line(cell_x[0] + 12, span_y + 18, cell_x[-1] + cell_w - 12, span_y + 18, INK, 2)
    for k, st in enumerate(stages):
        cx = cell_x[k] + cell_w / 2
        lines = head_lines[st['stage']]
        y = name_y + (n_head - len(lines)) * 50
        for line in lines:
            s.text(cx, y, line, body, 700, INK, 'middle')
            y += 50
        s.text(cx, num_y, f"{st['steps']:,}", 56, 700, INK, 'middle')
        s.text(cx, num_y + 48, 'steps', body, 400, INK, 'middle')
    tx = total_x + total_w / 2
    s.text(tx, name_y + (n_head - 1) * 50, 'Episodes', body, 700, INK, 'middle')
    s.text(tx, num_y, 'in all', body, 700, INK, 'middle')

    peak = max(c / st['episodes'] for st in stages for c in st['counts'].values())

    def cells(y, values, n_all, tint=True):
        for k, (st, c) in enumerate(zip(stages, values)):
            if not c:
                continue
            p = c / st['episodes']
            assert p >= 0.0005
            if tint:                    # deeper teal for a larger percent of the stage
                alpha = 0.10 + 0.55 * (p / peak) ** 0.5
                s.rect(cell_x[k] + 8, y + 5, cell_w - 16, row - 10, LINK, 12, extra=f'fill-opacity="{alpha:.3f}"')
            else:
                s.rect(cell_x[k] + 8, y + 5, cell_w - 16, row - 10, 'white', 12)
            s.text(cell_x[k] + cell_w / 2, y + row / 2 + 15, f'{100 * p:.1f}%', body, 400, INK, 'middle')
        s.text(total_x + total_w - 24, y + row / 2 + 15, f'{n_all:,}', body, 700, INK, 'end')

    y = y0
    for fam, rows in FAMILIES:
        h = len(rows) * row + 2 * pad
        s.rect(M, y, W - 2 * M, h, COOL, 22)
        fam_lines = wrap(fam, fam_w, body, 700)
        assert len(fam_lines) <= len(rows) and all(width(t, body, 700) <= fam_w for t in fam_lines), fam
        for j, line in enumerate(fam_lines):
            s.text(fam_x, y + pad + row / 2 + 15 + j * 52, line, body, 700, INK)
        ry = y + pad
        for key, text in rows:
            s.text(text_x, ry + row / 2 + 15, text, body, 400, INK)
            cells(ry, [st['counts'].get(key, 0) for st in stages], sum(st['counts'].get(key, 0) for st in stages))
            ry += row
        y += h + gap
    # silent time steps: any task's episode, so a card of its own
    y += 8
    s.rect(M, y, W - 2 * M, row + 2 * pad, COOL, 22)
    s.text(fam_x, y + pad + row / 2 + 15, SILENT[0], body, 700, INK)
    s.text(text_x, y + pad + row / 2 + 15, SILENT[1], body, 400, INK)
    cells(y + pad, [st['silent'] for st in stages], sum(st['silent'] for st in stages), tint=False)
    s.save(OUT)
    print(title)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--data', help="nl33's data folder: replies.json and data/*.json")
    ap.add_argument('--mix_log', help="nl20's full-mix training log (run/log.json)")
    a = ap.parse_args()
    if a.refresh and not (a.data and a.mix_log):
        ap.error('--refresh needs --data and --mix_log')
    render(refresh(a.data, a.mix_log) if a.refresh else json.loads(SNAP.read_text()))
