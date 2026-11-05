"""Zero-training KV crumbs: the untrained model (its LoRA adapter disabled) on trail test episodes, per crumb variant:
the K/V of the writer's own room text at that place (a; with the writer's positions or re-rotated to just before the
reader's first token there; and with each reader token seeing every place it has been at), of its last 4 tokens (b),
of its last token (c), and the kv arm's 4 slot tokens with fresh slot embeddings (d). Each with crumbs on, cut and
transplanted (crumbs.evaluate: teacher-forced NLL and forced choice; greedy answers on, off, swap and room). The
answer turn has no place, so without remember it reads the crumbs only through the follower's own earlier tokens; with
remember every reader token, the answer's included, sees the crumbs of every place visited so far.
Usage: python swarm/kv_zero.py OUT_JSON N N_GEN N_VARIANTS MODEL,...  (figures/crumbs_kv.py --zero draws it)"""
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch                                            # noqa: E402
from swarm import crumbs as C                           # noqa: E402

VARIANTS = (('a_text_remember', 'sentence', False, True), ('a_text_rerope_remember', 'sentence', True, True),
            ('b_last4_remember', 'last4', False, True), ('c_last1_remember', 'last1', False, True),
            ('d_slots_remember', 'slots', False, True), ('a_text', 'sentence', False, False),
            ('a_text_rerope', 'sentence', True, False), ('b_last4', 'last4', False, False),
            ('c_last1', 'last1', False, False), ('d_slots_untrained', 'slots', False, False))
out, n, n_gen = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
VARIANTS, models = VARIANTS[:int(sys.argv[4])], sys.argv[5].split(',')


def mentions(rows):
    """Format-free reading of the greedy answers (the untrained model does not answer "room: object"): the share of
    visited rooms whose object word appears in the answer, crumbs on and cut; for room (one visited room's crumbs
    from another episode's room) and swap (a visited room reads an unvisited room's crumbs), the share of answers
    that contain the moved crumbs' object, and that contain the room's own object."""
    by = {m: [r for r in rows if r['mode'] == m] for m in ('on', 'off', 'swap', 'room')}
    has = lambda w, t: bool(re.search(rf'\b{re.escape(w)}\b', t.lower()))          # noqa: E731
    final = lambda r: r['says'][max((k for k in r['says'] if k.startswith('1/')),     # noqa: E731
                                    key=lambda k: int(k.split('/')[1]))]
    out = {}
    for m in ('on', 'off'):
        hits = [has(r['episode']['trail']['objects'][q], final(r))
                for r in by[m] for q in r['episode']['trail']['visit']]
        out[f'named_{m}'] = sum(hits) / max(1, len(hits))
    eps = [r['episode'] for r in by['on']]
    for m in ('room', 'swap'):
        follow, own = [], []
        for r in by[m]:
            o, t = r['episode']['trail']['objects'], final(r)
            if m == 'room':
                pa, j, pc = r['plan']
                moved = eps[j]['trail']['objects'][pc]
            else:
                pa, pb = r['plan']
                moved = o[pb]
            if moved == o[pa]:
                continue
            follow.append(has(moved, t))
            own.append(has(o[pa], t))
        out[f'{m}_follows'] = sum(follow) / max(1, len(follow))
        out[f'{m}_keeps_own'] = sum(own) / max(1, len(own))
    return out


res = json.loads(out.read_text()) if out.exists() else {}
for base in models:
    net = C.build('cuda', 'kv', 4, base=base)
    for name, source, rerope, remember in VARIANTS:
        key = f'{base}/{name}'
        if 'mentions' in res.get(key, {}):
            continue
        net.source, net.rerope, net.remember = source, rerope, remember
        t0 = time.time()
        with net.model.disable_adapter():
            test, rows = C.evaluate(net, 'cuda', n=n, n_choice=n, n_gen=n_gen, games=('trail',), bs=25,
                                    keep_rows=n_gen)
        tr = test['trail']
        res[key] = dict(tf={m: {k: v for k, v in d.items()} for m, d in tr['tf'].items()}, gen=tr['gen'],
                        mentions=mentions(rows), s=round(time.time() - t0),
                        says=[dict(mode=r['mode'], say=r['says'], objects=r['episode']['trail']['objects'],
                                   visit=r['episode']['trail']['visit']) for r in rows[:4] + rows[n_gen:n_gen + 4]])
        out.write_text(json.dumps(res, indent=1))
        g, t = tr['gen'], tr['tf']
        print(json.dumps({key: dict(**{k: round(v, 3) for k, v in res[key]['mentions'].items()},
                                    objects_on=g['on']['objects'], objects_cut=g['off']['objects'],
                                    choice_on=t['on']['choice'], choice_cut=t['off']['choice'],
                                    choice_transplant=t['transplant']['choice'],
                                    follows=g['room']['follows_crumb'], keeps_own=g['room']['keeps_own'],
                                    s=res[key]['s'])}), flush=True)
    del net
    torch.cuda.empty_cache()
