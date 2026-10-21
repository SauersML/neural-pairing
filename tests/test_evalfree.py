"""Real tiny architectures/LoRA/bridges, scorers and checkpoint loading; no mocks.

NL_TINY=1 NL_DEVICE=cpu MPD_MEM_GIB=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 /Users/user/mpd-data/venv/bin/python tests/test_evalfree.py
Artifacts go to --out_dir when supplied, otherwise a temporary directory.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

if os.environ.get('NL_TINY') != '1' or os.environ.get('NL_DEVICE') != 'cpu':
    raise RuntimeError('this script requires NL_TINY=1 NL_DEVICE=cpu; never load full models')
if not 0 < float(os.environ.get('MPD_MEM_GIB', '0')) <= 6:
    raise RuntimeError('set 0 < MPD_MEM_GIB <= 6')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))

import torch
from peft import LoraConfig, get_peft_model
import coread as C
import evalfree as F
import evaluate as E
import games as G
import link
import play_tasks as P
import session as S
from payload_ctl import PAYLOAD_GAMES


def fixtures(out):
    replies = {x: dict(own={q: 'An ordinary answer.' for sp in ('train', 'test')
                            for q in G.POOLS['questions'][sp]}, steer=[]) for x in link.SIDES}
    rows = []
    for i, city in enumerate(('Lima', 'Oslo', 'Rome', 'Suva')):
        task_items = [dict(task='passage', kind='prose', text=f'{city} has a river. The boat is {i} years old.'),
                      dict(task='qa', context=f'Ada lives in {city}.', question=f'Where does person {i} live?', answer=city),
                      dict(task='hop', a=f'Person {i} lives in {city}.', b=f'{city} is the capital of country {i}.',
                           question=f'What is the capital of country {i}?', answer=city),
                      dict(task='translate', source=f'I live in {city}.', target=f'Vivo en {city}.',
                           source_lang='English', target_lang='Spanish'),
                      dict(task='paraphrase', a=f'Ada went to {city}.', b=f'Ada visited {city}.'),
                      dict(task='nli', premise=f'Ada is in {city}.', hypothesis=f'Ada visited city {i}.', label='neutral'),
                      dict(task='mcq', question=f'Which city is number {i}?', options=[city, f'Elsewhere {i}'], answer=0),
                      dict(task='math', question=f'What is {i} plus 2?', solution=f'The answer is {i + 2}.'),
                      dict(task='code_out', code=f'def f(): return {i}', call='f()', output=str(i)),
                      dict(task='code_write', problem=f'Return the integer {i}.', code=f'def f(): return {i}',
                           tests=[f'assert f() == {i}'])]
        rows.extend(dict(it, src=it['task']) for it in task_items)
    data = dict(train=[], test=rows, weights={r['src']: 1 for r in rows})
    (out / 'coread.json').write_text(json.dumps(data))
    (out / 'replies.json').write_text(json.dumps(replies))
    conversations = []
    for it in rows:
        if it['task'] != 'qa':
            continue
        for asker in link.SIDES:
            conversations.append(dict(item=it, asker=asker,
                                      system={x: C.talk_system(it, x, asker) for x in link.SIDES},
                                      turns=[(asker, C.talk_turn0(it)), (link.partner(asker), 'A later response.'),
                                             (asker, 'Thanks.')]))
    (out / 'talk.json').write_text(json.dumps(dict(train=[], test=conversations)))
    return replies


def checkpoint(out):
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], model = link.load(x)
        models[x] = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
                                                    target_modules=['q_proj', 'v_proj'])).eval()
        for name, p in models[x].named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(p, std=.03)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for bridge in pair.bridge.values():
        bridge.alpha.data.fill_(1.)
    out.mkdir()
    torch.save(pair.checkpoint(), out / 'bridge.pt')
    for x in link.SIDES:
        models[x].save_pretrained(out / x)
    del pair, models, toks, model
    gc.collect()


def assert_match_failures(plan, replies):
    ep, donor = plan['episode'], plan['donor']
    assert F.matched(ep, donor)
    assert not F.matched(ep, copy.deepcopy(ep))
    bad = copy.deepcopy(donor)
    bad['meta']['sender'], bad['meta']['receiver'] = bad['meta']['receiver'], bad['meta']['sender']
    assert not F.matched(ep, bad)
    # An actual corpus with only one distinct item must fail, never weaken matching.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'coread.json'
        it = dict(task='math', src='one', question='What is 2+2?', solution='4')
        path.write_text(json.dumps(dict(train=[], test=[it], weights={'one': 1})))
        source = C.Coread(replies, 'test', path=str(path))
        try:
            F.choose_donor(source.episode('co_math'), source, max_attempts=20)
        except ValueError as exc:
            assert 'No valid matched donor' in str(exc)
        else:
            raise AssertionError('same-item donor accepted')

    # Stale talk.json must not bypass either the corpus or conversation holdout.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixtures(root)
        data = json.loads((root / 'coread.json').read_text())
        talk = json.loads((root / 'talk.json').read_text())
        data['train'] = [talk['test'][0]['item']]
        talk['train'] = [talk['test'][2]]
        (root / 'coread.json').write_text(json.dumps(data))
        (root / 'talk.json').write_text(json.dumps(talk))
        clean, excluded = F.make_plans(replies, n=2, coread_data=root / 'coread.json', games=['co_talk'])
        assert excluded['talk_excluded'] == 4
        forbidden = {F.digest(talk['test'][i]['item']) for i in (0, 2)}
        for p in clean['co_talk']:
            assert all(F.digest(p[k]['meta']['item']) not in forbidden for k in ('episode', 'donor'))


def generation_checks(pair, plan):
    eps = [plan['episode'], plan['donor']]
    x = eps[0]['meta']['receiver']
    y = link.partner(x)
    with torch.inference_mode():
        before = S.generate(pair, eps, max_turn=3, keep=True)
        gold = copy.deepcopy(eps)
        for ep in gold:
            ep['reply'] = {s: ('FORBIDDEN GOLD ANSWER ' * 20, 1.) for s in link.SIDES}
        after = S.generate(pair, gold, max_turn=3, keep=True)
        for i in (0, 1):
            for side in link.SIDES:
                assert torch.equal(before[i]['h'][side], after[i]['h'][side]), 'gold entered generation'
        changed = copy.deepcopy(eps)
        changed[0]['task'][y] = 'A completely different private message with orange secrets.'
        for mode in ('zero', 'shuffle'):
            a = S.generate(pair, eps, mode=mode, max_turn=3, keep=True)
            b = S.generate(pair, changed, mode=mode, max_turn=3, keep=True)
            assert torch.allclose(a[0]['h'][x], b[0]['h'][x], atol=2e-6, rtol=2e-6), mode
        a = S.generate(pair, eps, max_turn=3, keep=True)
        b = S.generate(pair, changed, max_turn=3, keep=True)
        # Active bridges, not a vacuous zero-initialized isolation test.
        length = min(len(a[0]['h'][x]), len(b[0]['h'][x]))
        assert (a[0]['h'][x][:length] - b[0]['h'][x][:length]).abs().max() > 1e-4
        cut = S.generate(pair, eps, mode='zero', max_turn=3, keep=True)
        for side in link.SIDES:
            # Compare actual cut prompt states to the standalone LoRA model.
            ids = S.prefix(pair.toks[side], side, eps[0]['system'][side], eps[0]['task'][side])
            ids += S.assistant_header(pair.toks[side], side)
            inp = torch.tensor([ids[:-1]])
            hidden = link.inner(pair.models[side])[0](inp).last_hidden_state[0]
            assert torch.allclose(cut[0]['h'][side][:len(inp[0])], hidden, atol=2e-6, rtol=2e-6)
        assert all(p.requires_grad is False for p in pair.models[x].parameters())
    print('PASS free generation ignores gold; native shuffle/cut isolate true partner; cut = standalone LoRA', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir')
    args = ap.parse_args()
    tmp = tempfile.TemporaryDirectory() if not args.out_dir else None
    out = Path(args.out_dir or tmp.name)
    out.mkdir(parents=True, exist_ok=True)
    replies = fixtures(out)
    plans, excluded = F.make_plans(replies, n=2, seed=993, coread_data=out / 'coread.json')
    assert set(plans) == set(PAYLOAD_GAMES) | set(P.SCORED) | {'co_talk'}
    repeat, _ = F.make_plans(replies, n=2, seed=993, coread_data=out / 'coread.json',
                             games=['relay_word', 'co_swap'])
    assert repeat['relay_word'] == plans['relay_word'] and repeat['co_swap'] == plans['co_swap']
    for game, ps in plans.items():
        for plan in ps:
            ep, d = plan['episode'], plan['donor']
            assert F.matched(ep, d), game
            a, b = F.identity(ep), F.identity(d)
            assert a['roles'] == b['roles'] and a['item_id'] != b['item_id']
            result = dict(text={x: F.target(ep, x) for x in link.SIDES}) if ep['mode'] == 'duplex' else dict(
                turns=[ep['turns'][0][:2]])
            scores = F.receiver_scores(ep, result)
            assert scores and all(v == 1 for v in scores.values()), (game, scores)
    assert_match_failures(plans['relay_word'][0], replies)
    ep = copy.deepcopy(plans['relay_word'][0]['episode'])
    x = ep['meta']['receiver']
    ep['system'][x] += ' ' + ep['meta']['value'].upper() + '!!!'
    assert not F.flags(ep, x)['exclusive']
    assert F.contains('The CITY, is\nLiMa!', 'city is lima')
    assert not F.contains('150', '15')
    for p in plans['co_nli']:
        assert all(not F.flags(p['episode'], x)['exclusive'] for x in link.SIDES)
        assert all(F.flags(p['episode'], x)['symmetric'] for x in link.SIDES)
    print('PASS all 20 games, matched roles/private content, impossible donors, scorers and flags', flush=True)

    if not (out / 'step_tiny' / 'bridge.pt').exists():
        checkpoint(out / 'step_tiny')
    pair, _ = E.load_pair(str(out / 'step_tiny'))
    generation_checks(pair, plans['relay_word'][0])
    # Dialogue shuffle must isolate the true holder even with listener text
    # feedback. Only testing duplex or counting turns misses this path.
    talk = plans['co_talk'][0]
    ep, donor = talk['episode'], talk['donor']
    altered = copy.deepcopy(ep)
    asker = ep['meta']['asker']
    altered['system'][link.partner(asker)] = 'A new private context about violet elephants.'
    with torch.inference_mode():
        a = F.generate_shuffle(pair, ep, donor, 5, keep=True)
        b = F.generate_shuffle(pair, altered, donor, 5, keep=True)
        assert torch.equal(a['h'][asker], b['h'][asker]), 'dialogue reconnected true partner'
        assert len(a['turns']) == 1 and a['turns'][0][0] == asker
    print('PASS co_talk crossed text feedback isolates original holder', flush=True)
    os.environ['NL_MAXNEW'] = '2'
    assert F.generation_limit('co_code_write') == 2
    os.environ['NL_MAXNEW'] = '999'
    assert F.generation_limit('relay_word') == 64
    os.environ['NL_MAXNEW'] = '2'
    records = []
    for game, ps in plans.items():
        rs = list(F.evaluate_game(pair, ps, bs=2))
        assert len(rs) == sum(len(F.payloads(p['episode'])) for p in ps)
        for r in rs:
            for c in F.CONDITIONS:
                assert 0 <= r['conditions'][c]['score'] <= 1
                assert r['conditions'][c]['generated_tokens'] <= 2
                if game == 'co_talk':
                    assert set(r['conditions'][c]['texts']) == {r['receiver']}
            assert r['partner_role'] == r['donor']['partner_role']
            assert r['partner_item_id'] != r['donor']['partner_item_id']
        records.extend(rs)
        print('PASS tiny free generation', game, len(rs), 'receiver cases', flush=True)
    # Batch-size invariance of actual generated texts, not mocked logits.
    rs_one = list(F.evaluate_game(pair, plans['merge_sum'][:1], bs=2))
    rs_batch = [r for r in records if r['game'] == 'merge_sum' and r['episode_index'] == 0]
    assert [r['conditions'] for r in rs_one] == [r['conditions'] for r in rs_batch]
    manifest = dict(schema=F.SCHEMA, complete=True, run=str(out / 'step_tiny'), seed=993,
                    tiny=True, games=list(plans), receiver_records=len(records), excluded=excluded)
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with (out / 'episodes.jsonl').open('w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    manifest['episodes_sha256'] = F.file_hash(out / 'episodes.jsonl')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    dest = F.summarize(out, 'tiny_test', results_dir=out / 'results')
    table = json.loads(dest.read_text())
    assert len(table['rows']) == (len(plans) + 2) * 4
    empty = [r for r in table['rows'] if r['game'] == 'co_nli' and r['subset'] == 'exclusive']
    assert all(r['n'] == 0 and r['mean']['link'] is None and r['link-cut']['ci'] == [-1., 1.] for r in empty)
    pool = next(r for r in table['rows'] if r['game'] == 'pooled_coread' and
                r['receiver'] == 'quill' and r['subset'] == 'all')
    assert 'co_talk' not in pool['game_n'] and set(pool['game_n']) == set(P.SCORED)
    manifest['complete'] = False
    (out / 'manifest.json').write_text(json.dumps(manifest))
    try:
        F.summarize(out, 'incomplete', results_dir=out / 'results')
    except ValueError:
        pass
    else:
        raise AssertionError('partial run summarized')
    manifest['complete'] = True
    (out / 'manifest.json').write_text(json.dumps(manifest))
    print('PASS JSONL, per-game/all/exclusive/pool summaries, empty strata, incomplete-run guard', flush=True)
    print('ALL EVALFREE CHECKS PASSED; artifacts:', out, flush=True)


if __name__ == '__main__':
    main()
