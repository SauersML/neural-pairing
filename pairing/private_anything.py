"""Pre-registered private-item free-chat transfer experiment; no model API calls.

Mac: MPD_MEM_GIB=12 /usr/bin/lockf -k /Users/user/mpd-data/locks/macgpu.lock \
 /Users/user/mpd-data/venv/bin/python pairing/private_anything.py run --checkpoint after
Generate uses session.generate, the exact generator used by play_live/free_chat.
Crossed conversations replace the original partner entirely, as chateval.inputs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/Users/user/mpd-data/scratch/codex-swarm/wave21/private_anything')
SIDES = ('quill', 'ember')
CHECKPOINTS = {'after': '/Users/user/mpd-data/runpod/magic-nl22c/run/step_157',
               'before': '/Users/user/mpd-data/runpod/magic-nl20/run/step_1320'}
# Written before any generation: first spelling is the displayed value, others
# are the complete allowed alias list. Never expand aliases after seeing outputs.
KINDS = {
 'job': ('Your job is {v}.', 'carpenter|carpentry;paramedic;beekeeper|beekeeping;librarian;plumber|plumbing;ceramic artist|ceramicist;air traffic controller;marine biologist;locksmith;pastry chef;surveyor;sign language interpreter;arborist;costume designer;geologist;watchmaker'),
 'pet': ('Your pet is a {v}.', 'ferret;cockatiel;gecko;chinchilla;goldfish;dachshund;axolotl;canary;guinea pig;tarantula;tortoise;budgerigar|budgie;hedgehog;ragdoll cat;lop rabbit;corn snake'),
 'mood': ('Today you feel {v}.', 'homesick|homesickness;elated|elation;resentful|resentment;nostalgic|nostalgia;apprehensive|apprehension;contented|contentment;lonely|loneliness;inspired|inspiration;irritable|irritability;grateful|gratitude;melancholy|melancholic;restless|restlessness;embarrassed|embarrassment;relieved|relief;envious|envy;hopeful|hopefulness'),
 'food': ('Your favourite food is {v}.', 'pierogi;bibimbap;falafel;baklava;risotto;ceviche;lasagna|lasagne;kimchi;paella;gazpacho;gnocchi;shakshuka;poutine;spanakopita;dumplings;ramen'),
 'hobby': ('Your favourite hobby is {v}.', 'origami;birdwatching|bird watching;woodcarving|wood carving;stargazing;embroidery;rock climbing;beekeeping;calligraphy;pottery;geocaching;fencing;quilting;archery;juggling;knitting;lockpicking|lock picking'),
 'secret_word': ('Your private word today is "{v}".', 'thimble;quartz;marigold;velvet;compass;acorn;saffron;lantern;pebble;gossamer;turnip;cobalt;dandelion;obsidian;hammock;zephyr'),
 'trip': ('Your upcoming trip is to {v}.', 'the Galapagos Islands|Galapagos;the Scottish Highlands|Scottish Highlands;the Atacama Desert|Atacama;the Dolomites|Dolomites;the Serengeti|Serengeti;the Faroe Islands|Faroe;the Great Barrier Reef|Great Barrier Reef;the Carpathian Mountains|Carpathians|Carpathian;the Azores|Azores;the Black Forest|Black Forest;the Okavango Delta|Okavango;the Lake District|Lake District;the Lofoten Islands|Lofoten;the Gobi Desert|Gobi;the Amalfi Coast|Amalfi;the Pyrenees|Pyrenees'),
 'recent_event': ('Something happened recently: {v}.', 'you won a chess tournament|chess tournament;you adopted a rescue dog|rescue dog;you completed a marathon|marathon;you broke your wrist|broke your wrist|broken wrist;you baked sourdough for the first time|sourdough;you watched a lunar eclipse|lunar eclipse;you lost your passport|lost your passport|lost passport;you passed your driving test|driving test;you found a fossil|fossil;you planted an apple tree|apple tree;you attended a pottery workshop|pottery workshop;you repaired a bicycle|repaired a bicycle|bicycle repair;you sang at a wedding|sang at a wedding|wedding singing;you went scuba diving|scuba diving;you rescued a stranded kitten|stranded kitten;you won a spelling bee|spelling bee'),
 'instrument': ('The musical instrument you play is the {v}.', 'bassoon;harp;ukulele;clarinet;mandolin;oboe;accordion;banjo;theremin;viola;sitar;bagpipes;marimba;flugelhorn;dulcimer;harmonica'),
 'project': ('Your current personal project is {v}.', 'building a treehouse|treehouse;restoring a grandfather clock|grandfather clock;writing a detective novel|detective novel;making a stained glass window|stained glass;designing a board game|board game;sewing a patchwork blanket|patchwork blanket;constructing a sundial|sundial;painting a mural|mural;assembling a model railway|model railway|model railroad;growing oyster mushrooms|oyster mushrooms;making a stop motion film|stop motion;building a terrarium|terrarium;translating a sonnet|sonnet;carving a chess set|chess set;weaving a tapestry|tapestry;binding a handmade journal|handmade journal')}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def other(x):
    return SIDES[1 - SIDES.index(x)]


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def hits(text, aliases):
    norm = ' ' + normalize(text) + ' '
    return [a for a in aliases if ' ' + normalize(a) + ' ' in norm]


def exact_sign(differences):
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    return dict(wins=wins, losses=losses, ties=len(differences)-n,
                p_greater=(sum(math.comb(n, j) for j in range(wins, n+1)) / 2**n) if differences else None)


def score(turns, receiver, aliases):
    for k, (speaker, text) in enumerate(turns):
        found = hits(text, aliases)
        if found:
            return dict(success=int(speaker == receiver), owner_first=int(speaker != receiver),
                        first_turn=k, first_speaker=speaker, matched=found)
    return dict(success=0, owner_first=0, first_turn=None, first_speaker=None, matched=[])


def make_plan():
    import coread
    kinds, episodes = {}, []
    for kind, (template, raw) in KINDS.items():
        values = [dict(value=v.split('|')[0], aliases=list(dict.fromkeys(v.split('|')))) for v in raw.split(';')]
        assert len(values) == 16
        kinds[kind] = dict(template=template, values=values)
        for i in range(16):
            private = {x: values[(i+8*j)%16] for j, x in enumerate(SIDES)}
            opener = SIDES[i % 2]
            systems = {}
            for x in SIDES:
                wording = template.format(v=private[x]['value']) + ' ' + coread.CHAT_TEST.split('. ', 1)[1]
                systems[x] = coread.chat_system(x, wording=wording)
                assert hits(systems[x], private[x]['aliases'])
                assert not hits(systems[x], private[other(x)]['aliases']), (kind, i, x)
            episodes.append(dict(id=f'{kind}:{i}', kind=kind, index=i, opener=opener,
                                 private=private, system=systems, donor_id=f'{kind}:{(i+2)%16}'))
    plan = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), kinds=kinds, episodes=episodes,
                checkpoints=CHECKPOINTS, turns=6, max_turn=48, temperature=0, seed=4242,
                inference='Exact one-sided sign tests of paired per-chat success totals (0,1,2); ties removed. '
                  'Primary after link > cut and link > crossed, per kind; Holm correction across 20 primary tests. '
                  'Also raw receiver-specific sign tests and after > before paired chat-total sign tests. '
                  'No treating the two sides of one chat as independent observations.',
                metric='For each receiver, first whole-token normalized match of a preregistered target alias in '
                  'the six chronological turns must be written by the receiver. Owner-first and no mention score 0. '
                  'All sides remain in denominator; no post-generation exclusions. Lexical mention, not semantic '
                  'attribution. Mood and other common words can match by chance; cut and crossed estimate this.',
                crossed='For each receiver run a separate isolated six-turn conversation with that receiver prompt '
                  'and the same-kind same-opener donor partner prompt. Original partner absent; target remains '
                  'original item. Donor-target first mentions are scored separately. Text context also changes.',
                scope='Kinds are new private situations in city-only chat fine-tuning. Secret-word communication '
                  'was trained in the earlier games. No claim of unseen tokens, unseen pretraining concepts, or '
                  'all kinds absent from all earlier game training.',
                stopping='All 160 focal chats, 320 sides, 10 kinds x 16 authored values per checkpoint. '
                  'Every value appears once as each model\'s private item; openers are balanced. '
                  'Each checkpoint has 640 six-turn conversations (link, cut, two isolated crossed). '
                  'Save every completed batch. Interrupted batches are rerun unchanged. No outcome-based stopping.',
                precision='Real models on Mac use 4-bit MLX bases with trained adapters/bridge; numerics differ from bf16.')
    for ep in episodes:
        donor = next(e for e in episodes if e['id'] == ep['donor_id'])
        for x in SIDES:
            assert not any(hits(donor['system'][other(x)], ep['private'][y]['aliases']) for y in SIDES)
    return plan


def register(args):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / 'preregistered.json'
    if path.exists():
        raise FileExistsError('Preregistration already exists; never overwrite it')
    plan = make_plan()
    plan['code_hashes'] = {n: hash_file(ROOT/'pairing'/n) for n in
                           ('private_anything.py', 'coread.py', 'session.py', 'mlx_eval.py', 'mlx_pair.py')}
    path.write_text(json.dumps(plan, indent=2)+'\n')
    with (SCRATCH/'rounds.txt').open('a') as f:
        f.write(f"PRE-REGISTRATION {plan['created_utc']} SHA256 {hash_file(path)}\n")
        for k in ('metric', 'inference', 'crossed', 'stopping', 'scope', 'precision'):
            f.write(k.upper()+': '+plan[k]+'\n')
        f.write('All values, aliases, prompts, donor assignments and generation parameters frozen in '+str(path)+'\n')
    print(path, hash_file(path), flush=True)


def inputs(ep, donor=None, receiver=None):
    system = dict(ep['system'])
    if donor is not None:
        assert receiver in SIDES and ep['kind'] == donor['kind'] and ep['opener'] == donor['opener']
        y = other(receiver)
        assert not hits(donor['system'][y], ep['private'][y]['aliases'])
        system[y] = donor['system'][y]
    return dict(mode='dialogue', system=system, task=dict.fromkeys(SIDES),
                turns=[(x, '', 1.) for x in (ep['opener'], other(ep['opener']))*3])


def load_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args):
    plan_path = Path(args.plan or SCRATCH/'preregistered.json')
    plan = json.loads(plan_path.read_text())
    fingerprint = hash_file(plan_path)
    os.environ.setdefault('NL_DEVICE', 'cpu')
    if args.backend == 'torch' and sys.platform == 'darwin':
        raise ValueError('Real torch models are forbidden on Mac')
    if args.backend == 'mlx' and os.environ.get('MPD_MEM_GIB') != '12':
        raise ValueError('Real MLX run requires MPD_MEM_GIB=12 and external macgpu lockf')
    import torch
    import session
    torch.set_num_threads(2)
    directory = Path(args.out_dir or SCRATCH)
    directory.mkdir(parents=True, exist_ok=True)
    rawpath = directory/f'{args.checkpoint}.jsonl'
    records = load_records(rawpath)
    done = {(r['episode_id'], r['condition'], r['receiver']) for r in records}
    assert all(r['preregistration_sha256'] == fingerprint for r in records)
    jobs = []
    eps = {ep['id']: ep for ep in plan['episodes']}
    # Fixed ordering: complete every condition of each kind before the next kind.
    for kind in plan['kinds']:
        for condition, receiver in [('link', None), ('cut', None), ('crossed', 'quill'), ('crossed', 'ember')]:
            for ep in plan['episodes']:
                if ep['kind'] == kind and (ep['id'], condition, receiver) not in done:
                    jobs.append((ep, condition, receiver))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print('All requested conversations complete', flush=True)
        return
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(int(10.5*2**30))
        mx.set_cache_limit(32*2**20)
    else:
        from evaluate import load_pair
    checkpoint = args.run or plan['checkpoints'][args.checkpoint]
    started = time.monotonic()
    pair, _ = load_pair(checkpoint)
    provenance = dict(preregistration_sha256=fingerprint, checkpoint=checkpoint, backend=args.backend,
                      checkpoint_bridge_sha256=hash_file(Path(checkpoint)/'bridge.pt'),
                      code_hashes={n: hash_file(ROOT/'pairing'/n) for n in
                        ('private_anything.py', 'coread.py', 'session.py', 'mlx_eval.py', 'mlx_pair.py')})
    j = 0
    while j < len(jobs):
        chunk = [jobs[j]]
        while j+len(chunk) < len(jobs) and len(chunk) < args.bs and jobs[j+len(chunk)][1:] == jobs[j][1:]:
            chunk.append(jobs[j+len(chunk)])
        input_episodes = [inputs(ep, eps[ep['donor_id']] if cond == 'crossed' else None, receiver)
                          for ep, cond, receiver in chunk]
        t0 = time.monotonic()
        with torch.no_grad():
            generated = session.generate(pair, input_episodes, max_turn=plan['max_turn'], temperature=0,
                                         mode='zero' if chunk[0][1] == 'cut' else None, keep_tokens=True)
        with rawpath.open('a') as f:
            for (ep, cond, receiver), res, inp in zip(chunk, generated, input_episodes):
                assert len(res['turns']) == plan['turns']
                row = dict(**provenance, episode_id=ep['id'], kind=ep['kind'], condition=cond, receiver=receiver,
                           donor_id=ep['donor_id'] if cond == 'crossed' else None, inputs=inp,
                           turns=res['turns'], tokens=res['tokens'], sampled=res['sampled'],
                           batch_size=len(chunk), batch_seconds=time.monotonic()-t0,
                           elapsed_seconds=time.monotonic()-started)
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        j += len(chunk)
        print(json.dumps(dict(checkpoint=args.checkpoint, complete=len(done)+j, total=len(done)+len(jobs),
                              kind=chunk[0][0]['kind'], condition=chunk[0][1], seconds=round(time.monotonic()-t0, 1))), flush=True)
        if args.backend == 'mlx':
            mx.clear_cache()
    summarize(args)


def summarize(args):
    plan_path = Path(args.plan or SCRATCH/'preregistered.json')
    plan = json.loads(plan_path.read_text())
    out = dict(schema=1, preregistration=plan, preregistration_sha256=hash_file(plan_path), checkpoints={})
    audit_path = plan_path.with_name('training_audit.json')
    if audit_path.exists():
        out['training_audit'] = json.loads(audit_path.read_text())
    first_plan = plan_path.with_name('preregistered_v1.json')
    if first_plan.exists():
        out['preregistration_amendment'] = dict(original_sha256=hash_file(first_plan),
            reason='Before any generation, increased 8 to 16 chats per kind: with 8 signs, minimum p=1/256 '
                   'cannot pass Holm over 20 primary tests. Values, aliases and metric unchanged. '
                   'Each of the 16 authored values now occurs once per model. Original file retained.',
            rounds_file=str(plan_path.with_name('rounds.txt')))
    score_by_checkpoint = {}
    for checkpoint in CHECKPOINTS:
        rawpath = Path(args.out_dir or SCRATCH)/f'{checkpoint}.jsonl'
        raw = load_records(rawpath)
        index = {(r['episode_id'], r['condition'], r['receiver']): r for r in raw}
        assert len(index) == len(raw), 'duplicate conversations'
        scores, incomplete = [], []
        for ep in plan['episodes']:
            needed = [(ep['id'], 'link', None), (ep['id'], 'cut', None)] + [(ep['id'], 'crossed', x) for x in SIDES]
            if not all(k in index for k in needed):
                incomplete.append(ep['id'])
                continue
            donor = next(e for e in plan['episodes'] if e['id'] == ep['donor_id'])
            for x in SIDES:
                row = dict(episode_id=ep['id'], kind=ep['kind'], receiver=x, opener=ep['opener'],
                           value=ep['private'][other(x)]['value'], aliases=ep['private'][other(x)]['aliases'], conditions={})
                for cond in ('link', 'cut', 'crossed'):
                    r = index[ep['id'], cond, x if cond == 'crossed' else None]
                    assert r['preregistration_sha256'] == out['preregistration_sha256']
                    turns = r['turns']
                    row['conditions'][cond] = dict(**score(turns, x, row['aliases']), turns=turns,
                        donor_first=score(turns, x, donor['private'][other(x)]['aliases'])['success'],
                        truncated_turns=sum(not s[-1] for s in r['sampled']))
                scores.append(row)
        score_by_checkpoint[checkpoint] = scores
        groups = {}
        for kind in plan['kinds']:
            rows = [r for r in scores if r['kind'] == kind]
            groups[kind] = group_summary(rows)
        out['checkpoints'][checkpoint] = dict(complete=not incomplete, missing_episodes=incomplete,
            generated_conversations=len(raw), expected_conversations=4*len(plan['episodes']),
            raw_path=str(rawpath), raw_sha256=hash_file(rawpath) if rawpath.exists() else None,
            groups=groups, pooled=group_summary(scores), scores=scores)
    # Holm over exactly the pre-registered 20 primary comparisons, even incomplete.
    tests = [(kind, control, out['checkpoints']['after']['groups'][kind]['paired'][control]['p_greater'])
             for kind in plan['kinds'] for control in ('cut', 'crossed')]
    running = 0.
    for rank, (kind, control, p) in enumerate(sorted(tests, key=lambda t: 1. if t[2] is None else t[2])):
        if p is not None:
            running = max(running, min(1., (len(tests)-rank)*p))
        out['checkpoints']['after']['groups'][kind]['paired'][control]['p_holm'] = running if p is not None else None
    before = {(r['episode_id'], r['receiver']): r for r in score_by_checkpoint['before']}
    out['after_minus_before'] = {}
    for kind in plan['kinds']:
        diffs = defaultdict(int)
        for r in score_by_checkpoint['after']:
            key = r['episode_id'], r['receiver']
            if r['kind'] == kind and key in before:
                diffs[r['episode_id']] += r['conditions']['link']['success']-before[key]['conditions']['link']['success']
        out['after_minus_before'][kind] = exact_sign(list(diffs.values()))
    out['complete'] = all(c['complete'] for c in out['checkpoints'].values())
    out['status'] = ('complete' if out['complete'] else 'in_progress' if
                     any(c['generated_conversations'] for c in out['checkpoints'].values()) else 'awaiting_generation')
    dest = Path(args.output or ROOT/'results/private_anything.json')
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(complete=out['complete'], checkpoints={k: {g: v['counts'] for g, v in c['groups'].items()}
                                                            for k, c in out['checkpoints'].items()})), flush=True)


def group_summary(rows):
    counts = {c: sum(r['conditions'][c]['success'] for r in rows) for c in ('link', 'cut', 'crossed')}
    paired, per_receiver = {}, {}
    for control in ('cut', 'crossed'):
        totals = defaultdict(int)
        for r in rows:
            totals[r['episode_id']] += r['conditions']['link']['success']-r['conditions'][control]['success']
        paired[control] = exact_sign(list(totals.values()))
    for x in SIDES:
        mine = [r for r in rows if r['receiver'] == x]
        per_receiver[x] = dict(n=len(mine), counts={c: sum(r['conditions'][c]['success'] for r in mine) for c in counts},
            paired={c: exact_sign([r['conditions']['link']['success']-r['conditions'][c]['success'] for r in mine])
                    for c in ('cut', 'crossed')})
    return dict(n_sides=len(rows), n_chats=len({r['episode_id'] for r in rows}), counts=counts,
                paired=paired, per_receiver=per_receiver)


def self_test():
    # Deterministic lexical examples test only the scorer, never a measured model path.
    assert score([('ember','A ferret.'), ('quill','ferret')], 'quill', ['ferret'])['success'] == 0
    assert score([('quill','FERRET!'), ('ember','ferret')], 'quill', ['ferret'])['success'] == 1
    assert not hits('ferreting', ['ferret'])
    assert hits('rock-climbing', ['rock climbing'])
    assert exact_sign([1]*8)['p_greater'] == 1/256
    assert exact_sign([0,0])['p_greater'] == 1
    plan = make_plan()
    eps = {e['id']: e for e in plan['episodes']}
    for ep in eps.values():
        for x in SIDES:
            inp = inputs(ep, eps[ep['donor_id']], x)
            assert not any(hits(s, ep['private'][other(x)]['aliases']) for s in inp['system'].values())
            assert all(not t for _,t,_ in inp['turns'])
    print('scorer, statistics, full prompt inventory and crossed isolation: OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('register', 'run', 'summarize', 'self-test'))
    ap.add_argument('--checkpoint', choices=CHECKPOINTS, default='after')
    ap.add_argument('--backend', choices=('mlx','torch'), default='mlx')
    ap.add_argument('--run')
    ap.add_argument('--plan', help='frozen preregistration JSON; required when running on another host')
    ap.add_argument('--out-dir')
    ap.add_argument('--output')
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='operational chunk limit; does not change registered sample')
    args = ap.parse_args()
    if args.command == 'register': register(args)
    elif args.command == 'run': run(args)
    elif args.command == 'summarize': summarize(args)
    else: self_test()

if __name__ == '__main__':
    main()
