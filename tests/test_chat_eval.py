"""chateval.py with evalfree.py on a tiny pair: co_chat and co_chat2 plans, donors and leak guards, the measures, real
generation under every condition, summaries, and the sharded command line with chat2_merge and chat_merge.
"""
import contextlib
import copy
import io
import json
import random
import sys

import pytest
import torch

import chat2
import chat2_merge
import chat_merge
import chateval as C
import coread
import evalfree as F
import link


def chat(opener='quill', quill='Lisbon', ember='Lima'):
    private = dict(quill=quill, ember=ember)
    return dict(opener=opener, private=private,
                system={x: coread.chat_system(x, private[x], coread.CHAT[0] if private[x] else coread.CHAT[3])
                        for x in link.SIDES},
                teach={x: 'FORBIDDEN_TEACHER ' + str(private[link.partner(x)]) for x in link.SIDES},
                turns=[(x, 'FORBIDDEN_RECORDED_TURN') for x in (opener, link.partner(opener)) * 3],
                pmi={'quill': [900, 900, 900]})


@pytest.fixture
def data(tmp_path):
    chats = [chat(opener, a, b) for opener in link.SIDES
             for a, b in [('Lisbon', 'Lima'), ('Oslo', 'Suva'), ('Rome', 'Kyoto'),
                          ('Lima', None), (None, 'Oslo'), (None, None)]]
    (tmp_path / 'chat.json').write_text(json.dumps(dict(train=[chat(quill='TRAIN_ONLY')], test=chats)))
    (tmp_path / 'coread.json').write_text(json.dumps(dict(train=[], test=[], weights={})))
    replies = {x: dict(own={}, steer=[]) for x in link.SIDES}
    (tmp_path / 'replies.json').write_text(json.dumps(replies))
    return tmp_path, replies


def test_draws_inputs_and_donor_guards(data):
    path, replies = data
    plans, _ = F.make_plans(replies, n=24, seed=93, coread_data=path / 'coread.json', games=['co_chat'])
    repeat, _ = F.make_plans(replies, n=24, seed=93, coread_data=path / 'coread.json', games=['co_chat'])
    assert plans == repeat
    text = json.dumps(plans)
    assert all(s not in text for s in ('FORBIDDEN', 'TRAIN_ONLY', 'teach', 'pmi'))
    assert {p['episode']['meta']['opener'] for p in plans['co_chat']} == set(link.SIDES)
    assert any(not any(p['episode']['meta']['private'].values()) for p in plans['co_chat'])
    for p in plans['co_chat']:
        ep = p['episode']
        for x in link.SIDES:
            d, y = p['donors'][x], link.partner(x)
            assert C.matched(ep, d, x)
            e = C.inputs(ep, d, x)
            assert e['system'] == {x: ep['system'][x], y: d['system'][y]}
            assert all(not t for _, t, _ in e['turns']) and len(e['turns']) == 6
            real, own, other = ep['meta']['private'][y], ep['meta']['private'][x], d['meta']['private'][y]
            if real:
                assert other and other not in (real, own)
    ep = C.episode(chat())
    for bad in [chat(ember='Lima'), chat(ember='Lisbon'), chat(ember=None), chat(opener='ember', ember='Kyoto')]:
        assert not C.matched(ep, C.episode(bad), 'quill')
    bad = C.episode(chat(ember='Kyoto'))
    bad['system']['ember'] += ' I also know Lima.'
    assert not C.matched(ep, bad, 'quill')
    (path / 'chat.json').unlink()
    with pytest.raises(ValueError, match='requires test-split'):
        F.make_plans(replies, n=2, coread_data=path / 'coread.json', games=['co_chat'])
    (path / 'chat.json').write_text(json.dumps(dict(test=[chat()])))
    with pytest.raises(ValueError, match='No valid co_chat donor'):
        F.make_plans(replies, n=2, coread_data=path / 'coread.json', games=['co_chat'], max_attempts=5)


def test_names_leak_precedence_and_collapse():
    private = dict(quill='Lisbon', ember='New York')
    turns = [('quill', 'NEW, YORK! How is it?'), ('ember', 'I am in New York.'), ('quill', 'Lisbon is lovely.')]
    a = C.measures(turns, 'quill', private, 'Suva')
    assert (a['names'], a['leak'], a['names_donor'], a['self_city_first']) == (1, 0, 0, 0)
    turns = [('ember', 'New York is sunny.'), ('quill', 'New York! I am in Lisbon. Suva?')]
    a = C.measures(turns, 'quill', private, 'Suva')
    assert (a['names'], a['leak'], a['names_donor'], a['self_city_first']) == (0, 1, 1, 1)
    assert C.first_mention([('ember', 'Lima'), ('quill', 'Lima')], 'quill', 'Lima') == (0, 1)
    assert C.first_mention([('quill', 'Lima')], 'quill', 'Lim') == (0, 0)
    assert C.first_mention(turns, 'quill', None) == (None, None)
    assert C.first_mention([('quill', 'Oslo'), ('ember', 'Suva')], 'quill', 'Suva') == (0, 1)
    repeated = [('quill', 'alpha beta gamma'), ('ember', 'other novel words'),
                ('quill', 'ALPHA BETA GAMMA'), ('ember', 'ALPHA BETA GAMMA')]
    a = C.measures(repeated, 'quill', private)
    assert a['loop_turns'] == [False, False, True, True]
    assert a['loop'] == .5 and a['distinct2'] == .5 and a['game'] == 0
    a = C.measures([('quill', 'abcde'), ('ember', 'abcXY')], 'quill', private)
    assert a['loop'] == 0, 'exactly 60% must not count'
    a = C.measures([('quill', ''), ('ember', '')], 'quill', dict.fromkeys(link.SIDES))
    assert a['loop'] == a['distinct2'] == a['self_city_first'] == 0
    assert a['names'] is None
    a = C.measures([('quill', 'YOUR TURN'), ('ember', 'A secret word'), ('quill', 'Just chatting')], 'quill', private)
    assert a['game_turns'] == [True, True, False] and a['game'] == 2 / 3


@torch.no_grad()
def test_actual_prompts_streaming_and_crossed_isolation(pair):
    ep, donor = C.episode(chat()), C.episode(chat(quill='Rome', ember='Suva'))
    for x in link.SIDES:
        y = link.partner(x)
        captured = {s: [] for s in link.SIDES}
        original = pair.step

        def observe(chunk, cache, qmask, tick, *args, **kw):
            for s, (ids, att, _) in chunk.items():
                captured[s].extend(ids[0][att[0, -ids.shape[1]:].bool()].tolist())
            return original(chunk, cache, qmask, tick, *args, **kw)

        pair.step = observe
        try:
            a = C.generate(pair, ep, 3, 8, donor=donor, receiver=x, keep=True)
        finally:
            del pair.step
        for s in link.SIDES:
            context = pair.toks[s].decode(captured[s])
            assert ep['meta']['private'][y] not in context
            assert 'FORBIDDEN' not in context
            assert C.inputs(ep, donor, x)['system'][s] in context
        changed = copy.deepcopy(ep)
        changed['system'][y] = 'ORIGINAL_PARTNER_SECRET ' * 50
        changed['turns'] = [(s, 'FORBIDDEN_GOLD', 1.) for s, *_ in ep['turns']]
        changed['forced'] = list(range(6))
        changed['turn_tokens'] = [[777]] * 6
        b = C.generate(pair, changed, 3, 8, donor=donor, receiver=x, keep=True)
        assert a['tokens'] == b['tokens'] and len(a['turns']) == 6
        for s in link.SIDES:
            assert torch.equal(a['h'][s], b['h'][s])
        changed_donor = copy.deepcopy(donor)
        changed_donor['system'][y] += ' Different donor context about yellow boats.'
        c = C.generate(pair, ep, 3, 8, donor=changed_donor, receiver=x, keep=True)
        assert not torch.equal(a['h'][x][:100], c['h'][x][:100]), 'link must be active'
        # The real partner's prompt is available through the open focal link.
        on = C.generate(pair, ep, 2, 8, keep=True)
        on_changed = C.generate(pair, changed, 2, 8, keep=True)
        assert not torch.equal(on['h'][x][:100], on_changed['h'][x][:100])


@torch.no_grad()
def test_conditions_sampling_summary_and_null_targets(pair, data, monkeypatch):
    path, replies = data
    monkeypatch.setenv('NL_MAXNEW', '2')
    plans, _ = F.make_plans(replies, n=4, seed=93, coread_data=path / 'coread.json', games=['co_chat'])
    # Explicitly include a city-free chat to test the non-naming denominator.
    p = copy.deepcopy(plans['co_chat'][0])
    p['episode'] = C.episode(chat(quill=None, ember=None))
    p['donors'] = {x: C.episode(chat(quill='Rome', ember='Suva')) for x in link.SIDES}
    p['episode_id'] = '93:co_chat:empty'
    plans['co_chat'] = plans['co_chat'][:1] + [p]
    for temperature in (0., .8):
        records = list(F.evaluate_game(pair, plans['co_chat'], bs=2, crossed=True, temperature=temperature))
        again = list(F.evaluate_game(pair, plans['co_chat'], bs=4, crossed=True, temperature=temperature))
        assert records == again
        assert len(records) == 4
        for r in records:
            assert set(r['conditions']) == {'link', 'cut', 'crossed'}
            assert r['temperature'] == temperature
            for c in r['conditions'].values():
                assert len(c['turns']) == 6 and c['generated_tokens'] <= 12
                assert 'FORBIDDEN' not in json.dumps(c)
        empty = [r for r in records if r['episode_id'].endswith('empty')]
        assert all(c['names'] is None and c['leak'] is None for r in empty for c in r['conditions'].values())
        rows = F.summary_rows(records, ['co_chat'])
        assert len(rows) == 3
        for row in rows:
            assert row['n'] == 2
            for c in row['conditions'].values():
                assert c['loop']['n'] == c['distinct2']['n'] == 2
            if row['receiver'] != 'dialogue':
                assert row['link-crossed']['metric'] == 'names'
                assert 'p_greater' in row['link-cut']['sign']
        manifest = dict(schema=F.SCHEMA, complete=True, run='tiny', seed=93, n=2, tiny=True,
                        crossed=True, temperature=temperature, games=['co_chat'], receiver_records=4)
        (path / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        manifest['episodes_sha256'] = F.file_hash(path / 'episodes.jsonl')
        (path / 'manifest.json').write_text(json.dumps(manifest))
        result = F.summarize(path, 'chat', results_dir=path)
        assert json.loads(result.read_text())['rows'] == rows
        broken = copy.deepcopy(records)
        broken[0]['conditions']['crossed']['loop'] = float('nan')
        with pytest.raises(ValueError, match='metrics disagree'):
            C.validate(broken, manifest)
        with pytest.raises(ValueError, match='missing chat'):
            C.validate(records[:-1], manifest)
        with pytest.raises(ValueError, match='condition'):
            C.validate(records, dict(manifest, crossed=False))


def test_cli_checkpoint_loading_and_summary(pair, data, monkeypatch):
    path, _ = data
    step = path / 'step'
    step.mkdir()
    for x in link.SIDES:
        pair.models[x].save_pretrained(step / x)
    torch.save(pair.checkpoint(), step / 'bridge.pt')
    out = path / 'out'
    monkeypatch.setenv('NL_MAXNEW', '1')
    monkeypatch.setattr(sys, 'argv', ['evalfree.py', '--run', str(step), '--replies', str(path / 'replies.json'),
                                   '--out_dir', str(out), '--coread_data', str(path / 'coread.json'),
                                   '--games', 'co_chat', '--crossed', '--n', '2', '--bs', '4', '--temperature', '.7'])
    F.main()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['receiver_records'] == 4 and manifest['temperature'] == .7
    assert manifest['hashes'][str(path / 'chat.json')] == F.file_hash(path / 'chat.json')
    assert 'chateval.py' in manifest['code_hashes']
    result = F.summarize(out, 'cli', results_dir=path)
    assert json.loads(result.read_text())['temperature'] == .7


def item(kind, index=20):
    return copy.deepcopy(chat2.CATALOG[kind][index])


def chat2_record(opener, a, b):
    private = dict(quill=a, ember=b)
    w = chat2.wording('test', random.Random(0))
    return dict(game='chat2', opener=opener, private=private,
                system={x: coread.chat_system(x, private[x], w) for x in link.SIDES},
                teach={'quill': 'FORBIDDEN_TEACHER'}, turns=[['quill', 'FORBIDDEN_TURN']], pmi={'quill': [99]})


def test_chat2_aliases_leaks_and_matched_kinds():
    private = dict(quill=item('job'), ember=item('event', 0))
    assert C.first_mention([('quill', 'Your brother got married!')], 'quill', private['ember']) == (1, 0)
    assert C.first_mention([('ember', 'My wedding.'), ('quill', 'Congratulations on the marriage.')],
                           'quill', private['ember']) == (0, 1)
    assert C.first_mention([('quill', 'unmarried')], 'quill', private['ember']) == (0, 0)
    ep = C.episode(chat2_record('quill', private['quill'], private['ember']))
    donor = C.episode(chat2_record('quill', item('job', 21), item('event', 21)))
    assert C.matched(ep, donor, 'quill')
    for wrong in (item('event', 0), item('job', 22), None):
        bad = C.episode(chat2_record('quill', item('pet'), wrong))
        assert not C.matched(ep, bad, 'quill')
    donor['system']['ember'] += ' I know about a marriage.'
    assert not C.matched(ep, donor, 'quill')


@torch.no_grad()
def test_chat2_real_generation_and_sharded_cli(pair, tmp_path, monkeypatch):
    pool = [chat2_record(opener, item(kind, i), item(kind, (i + 1) % 26))
            for opener in link.SIDES for kind in ('job', 'mood') for i in range(20, 26)]
    pool += [chat2_record(opener, None, None) for opener in link.SIDES]
    (tmp_path / 'chat2.json').write_text(json.dumps(dict(train=[], test=pool)))
    (tmp_path / 'coread.json').write_text(json.dumps(dict(train=[], test=[], weights={})))
    replies = {x: dict(own={}, steer=[]) for x in link.SIDES}
    (tmp_path / 'replies.json').write_text(json.dumps(replies))
    plans, _ = F.make_plans(replies, n=9, seed=12, coread_data=tmp_path / 'coread.json', games=['co_chat2'])
    assert 'FORBIDDEN' not in json.dumps(plans)
    for p in plans['co_chat2']:
        for x in link.SIDES:
            assert C.matched(p['episode'], p['donors'][x], x)
    monkeypatch.setenv('NL_MAXNEW', '1')
    records = list(F.evaluate_game(pair, plans['co_chat2'][:2], bs=2, crossed=True))
    assert all(r['game'] == 'co_chat2' and r['kind'] == chat2.kind_of(r['private'][link.partner(r['receiver'])])
               for r in records)
    rows = C.summary_rows(records)
    assert all('names_counts' in r for r in rows if r['receiver'] != 'dialogue')
    assert any(r['subset'].startswith('kind:') for r in rows)
    C.validate(records, dict(n=2, temperature=0., crossed=True))
    step = tmp_path / 'step'
    step.mkdir()
    for x in link.SIDES:
        pair.models[x].save_pretrained(step / x)
    torch.save(pair.checkpoint(), step / 'bridge.pt')
    all_ids = []
    for shard in range(3):
        out = tmp_path / str(shard)
        monkeypatch.setattr(sys, 'argv', ['evalfree.py', '--run', str(step), '--replies', str(tmp_path / 'replies.json'),
            '--out_dir', str(out), '--coread_data', str(tmp_path / 'coread.json'), '--games', 'co_chat2',
            '--crossed', '--n', '3', '--shard', f'{shard}/3', '--bs', '2'])
        F.main()
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['complete'] and manifest['chat_plans_by_game'] == {'co_chat2': 1}
        assert str(tmp_path / 'chat2.json') in manifest['hashes']
        F.summarize(out, f'chat2_{shard}', results_dir=tmp_path)
        all_ids += [json.loads(l)['episode_id'] for l in (out / 'plans.jsonl').read_text().splitlines()]
    assert len(set(all_ids)) == 3
    load = chat2_merge.load
    merged, _ = load([tmp_path / str(i) for i in range(3)])
    assert len(merged) == 6
    with pytest.raises(ValueError, match='missing shard'):
        load([tmp_path / '0', tmp_path / '1'])
    monkeypatch.setattr(sys, 'argv', ['chat_merge.py', '--after', *[str(tmp_path / str(i)) for i in range(3)]])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        chat_merge.main()
    assert 'co_chat2 kind:' in printed.getvalue()
    # A shard rerun with a different chateval.py merges only when that file is named, and its hashes are kept.
    m = json.loads((tmp_path / '0' / 'manifest.json').read_text())
    m['code_hashes']['chateval.py'] = '0' * 64
    (tmp_path / '0' / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='code_hashes'):
        load([tmp_path / str(i) for i in range(3)])
    merged, hashes = load([tmp_path / str(i) for i in range(3)], code_ok=('chateval.py',))
    assert len(merged) == 6 and hashes['chateval.py'][0] == '0' * 64 and len(set(hashes['chateval.py'])) == 2
