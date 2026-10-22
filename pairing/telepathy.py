"""Simultaneous open-category coordination; original session.generate and MLX pair.

Plan first, then run under the Mac GPU lock. Summarize is model-free.
The torch backend is for lead-run bf16 confirmation on a CUDA host only.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

SIDES = ('quill', 'ember')
CATEGORIES = {'animal': 'an animal', 'number': 'a number from 1 to 100 (use digits)',
              'colour': 'a colour', 'city': 'a city with a one-word name', 'fruit': 'a fruit',
              'letter': 'a letter of the English alphabet',
              'person': 'a famous person (use only the surname)', 'country': 'a country with a one-word name',
              'sport': 'a sport', 'instrument': 'a musical instrument', 'flower': 'a flower',
              'profession': 'a profession'}
SCENES = ['You are spending a quiet morning in a library.', 'You are taking an evening train journey.',
          'You are relaxing in a sunny garden.', 'You are sitting in a busy cafe at lunchtime.']
WORDINGS = [
    'Your partner is answering the same question at the same moment. Try to give {goal}. Answer with one word: name {category}.',
    'Choose {category} now. Your partner is choosing at the same time. Give {goal}. Write only one word.',
    'Without exchanging any text, you and your partner will each name {category} simultaneously. Aim to give {goal}. Reply with one word only.',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    while text and (unicodedata.category(text[0]).startswith('P') or text[0].isspace()):
        text = text[1:]
    while text and (unicodedata.category(text[-1]).startswith('P') or text[-1].isspace()):
        text = text[:-1]
    return text


def format_ok(text, category):
    if category == 'number':
        return bool(re.fullmatch(r'\d{1,3}', text)) and 1 <= int(text) <= 100
    if category == 'letter':
        return bool(re.fullmatch('[a-z]', text))
    return bool(re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text, re.UNICODE))


def plan(path, rounds):
    import games
    cats = list(CATEGORIES)
    batches = []
    for objective in ('same', 'different'):
        for wording in range(len(WORDINGS)):
            for context in range(len(SCENES)):
                for start in (0, 6):
                    bid = len(batches)
                    eps = []
                    for category in cats[start:start+6]:
                        goal = 'the same answer as your partner' if objective == 'same' else 'a DIFFERENT answer from your partner'
                        task = WORDINGS[wording].format(goal=goal, category=CATEGORIES[category])
                        eps.append(dict(game='telepathy', mode='duplex',
                                        system={x: games.system(x) + '\n\n' + SCENES[context] for x in SIDES},
                                        task={x: task for x in SIDES},
                                        meta=dict(id=f'{objective}/{wording}/{context}/{category}', objective=objective,
                                                  category=category, wording=wording, context=context, batch=bid)))
                    batches.append(dict(id=bid, seed=210221+bid, episodes=eps))
    obj = dict(preregistered=json.loads(Path(rounds).read_text()), batches=batches,
               created_utc=datetime.now(timezone.utc).isoformat(),
               source_hashes={n: digest(Path(__file__).with_name(n)) for n in
                              ('telepathy.py','session.py','link.py','games.py','coread.py','mlx_eval.py','mlx_pair.py')})
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)
    print('registered', len(batches)*6, 'episodes per checkpoint; plan SHA256', digest(path))


def run(args):
    if args.backend == 'torch' and sys.platform == 'darwin':
        raise RuntimeError('Real torch models forbidden on the Mac')
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
        import mlx.core as mx
        import mlx_eval
        mx.set_memory_limit(int(10.5*2**30)); mx.set_cache_limit(32*2**20)
        loader = mlx_eval.load_pair
    else:
        os.environ['NL_DEVICE'] = 'cuda'
        from evaluate import load_pair as loader
    import torch
    import session
    torch.set_num_threads(2)
    doc = json.loads(Path(args.plan).read_text())
    out = Path(args.raw)
    existing = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    keys = {(r['checkpoint'],r['batch'],r['condition']) for r in existing}
    for key in keys:
        assert sum((r['checkpoint'],r['batch'],r['condition']) == key for r in existing) == 6, 'partial batch; inspect file'
    pair = None
    for checkpoint, directory in [('after',args.after),('before',args.before)]:
        if not directory:
            continue
        if all((checkpoint,b['id'],c) in keys for b in doc['batches'] for c in ('link','cut','crossed')):
            continue
        if pair is None or args.backend != 'mlx':
            pair, _ = loader(directory)
        else:
            import mlx_pair
            mlx_pair.load_checkpoint(pair.mlx, directory)
        checkpoint_hash = digest(Path(directory)/'bridge.pt')
        for b in doc['batches']:
            for condition, mode in [('link',None),('cut','zero'),('crossed','shuffle')]:
                if (checkpoint,b['id'],condition) in keys:
                    continue
                started = time.monotonic()
                result = session.generate(pair,b['episodes'],max_turn=8,temperature=.7,
                                          gen=torch.Generator(device='cpu' if args.backend=='mlx' else 'cuda').manual_seed(b['seed']),
                                          mode=mode,keep_tokens=True)
                records = []
                for i,(ep,r) in enumerate(zip(b['episodes'],result)):
                    n = {x:norm(r['text'][x]) for x in SIDES}
                    ok = {x:format_ok(n[x],ep['meta']['category']) for x in SIDES}
                    match = bool(n['quill']) and n['quill']==n['ember']
                    records.append(dict(**ep['meta'],checkpoint=checkpoint,run=directory,bridge_sha256=checkpoint_hash,
                                        plan_sha256=digest(args.plan),condition=condition,text=r['text'],normalized=n,
                                        tokens=r['tokens'],sampled=r['sampled'],format_ok=ok,match=match,
                                        literal_match=bool(r['text']['quill']) and r['text']['quill']==r['text']['ember'],
                                        valid_match=all(ok.values()) and match,valid_different=all(ok.values()) and not match,
                                        donors={x:b['episodes'][(i+(-1 if x=='quill' else 1))%6]['meta']['id']
                                                for x in SIDES} if condition=='crossed' else None,
                                        backend=args.backend,utc=datetime.now(timezone.utc).isoformat()))
                with out.open('a') as f:
                    f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));f.flush();os.fsync(f.fileno())
                print(json.dumps(dict(checkpoint=checkpoint,batch=b['id'],condition=condition,complete=len(records),
                                      seconds=round(time.monotonic()-started,2))),flush=True)
                if args.backend=='mlx':
                    mx.clear_cache()


def signflip(values):
    """Exact upper tail under independent sign flips of whole cluster differences."""
    counts = Counter({0:1})
    for v in values:
        nxt = Counter()
        for s,n in counts.items():
            nxt[s+v] += n; nxt[s-v] += n
        counts = nxt
    return sum(n for s,n in counts.items() if s>=sum(values))/2**len(values)


def paired(rows, control, objective, metric='match'):
    from scipy.stats import binomtest
    by = defaultdict(dict)
    for r in rows:
        by[r['id']][r['condition']] = r
    differences, clusters = [], defaultdict(int)
    for item in by.values():
        a,b = item['link'],item[control]
        d = int(a[metric])-int(b[metric])
        if objective=='different' and metric=='match': d = -d
        differences.append(d);clusters[a['batch']] += d
    wins,losses = differences.count(1),differences.count(-1)
    return dict(wins=wins,losses=losses,ties=differences.count(0),
                exact_mcnemar_p=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.,
                cluster_differences=list(clusters.values()),exact_cluster_p=signflip(list(clusters.values())))


def summarize(args):
    rows = [json.loads(s) for s in Path(args.raw).read_text().splitlines()]
    doc = json.loads(Path(args.plan).read_text())
    assert len(rows)==len(doc['batches'])*6*3*2, ('incomplete',len(rows))
    assert len({(r['checkpoint'],r['condition'],r['id']) for r in rows})==len(rows)
    assert {r['plan_sha256'] for r in rows}=={digest(args.plan)}
    summary = {}
    for checkpoint in ('after','before'):
        summary[checkpoint] = {}
        for objective in ('same','different'):
            rr = [r for r in rows if r['checkpoint']==checkpoint and r['objective']==objective]
            per = {}
            for cat in CATEGORIES:
                cc = [r for r in rr if r['category']==cat]
                cut = [r for r in cc if r['condition']=='cut']
                dist = {x:Counter(r['normalized'][x] for r in cut) for x in SIDES}
                numerator = sum(n*dist['ember'][a] for a,n in dist['quill'].items() if a)
                diagonal = sum(r['match'] for r in cut)
                per[cat] = dict(n=len(cut),conditions={c:dict(matches=sum(r['match'] for r in cc if r['condition']==c),
                           literal_matches=sum(r['literal_match'] for r in cc if r['condition']==c),
                           format_pairs=sum(all(r['format_ok'].values()) for r in cc if r['condition']==c),
                           valid_matches=sum(r['valid_match'] for r in cc if r['condition']==c),
                           valid_different=sum(r['valid_different'] for r in cc if r['condition']==c))
                           for c in ('link','cut','crossed')},cut_distributions=dist,
                           chance=numerator/len(cut)**2,chance_offdiagonal=(numerator-diagonal)/(len(cut)*(len(cut)-1)))
            summary[checkpoint][objective] = dict(categories=per,n=sum(v['n'] for v in per.values()),
                conditions={c:{k:sum(v['conditions'][c][k] for v in per.values())
                               for k in next(iter(per.values()))['conditions'][c]} for c in ('link','cut','crossed')},
                chance=sum(v['chance'] for v in per.values())/len(per),
                chance_offdiagonal=sum(v['chance_offdiagonal'] for v in per.values())/len(per),
                tests={c:paired(rr,c,objective) for c in ('cut','crossed')},
                format_tests={c:paired(rr,c,objective,'valid_match' if objective=='same' else 'valid_different')
                              for c in ('cut','crossed')})
    family = sorted((summary['after'][o]['tests'][c]['exact_mcnemar_p' if c=='cut' else 'exact_cluster_p'],o,c)
                    for o in ('same','different') for c in ('cut','crossed'))
    adjusted = 0.
    for i,(p,o,c) in enumerate(family):
        adjusted = max(adjusted,min(1.,p*(len(family)-i)))
        summary['after'][o]['tests'][c]['holm_p'] = adjusted
    report = dict(experiment='telepathy',plan=doc,plan_sha256=digest(args.plan),raw_sha256=digest(args.raw),
                  precision=('Mac 4-bit MLX bases; numerics differ from GPU bf16' if {r['backend'] for r in rows} == {'mlx'} else 'Torch bf16 CUDA confirmation'),summary=summary,records=rows,
                  existing_measurements_changed=False)
    Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:{o:{a:b for a,b in r.items() if a!='categories'} for o,r in v.items()}
                      for k,v in summary.items()},indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['plan','run','summarize'])
    ap.add_argument('--plan',required=True);ap.add_argument('--rounds');ap.add_argument('--raw');ap.add_argument('--out')
    ap.add_argument('--after');ap.add_argument('--before');ap.add_argument('--backend',choices=['mlx','torch'],default='mlx')
    a=ap.parse_args()
    if a.command=='plan':plan(a.plan,a.rounds)
    elif a.command=='run':run(a)
    else:summarize(a)


if __name__=='__main__':main()
