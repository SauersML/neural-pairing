"""evalfree.py (free generation under link, cut and shuffle with matched donors) on tiny random models with LoRA and open
bridges, loaded from a saved checkpoint: plans cover every game with matched donors and private content, and a
corpus without a second distinct item or a stale talk.json cannot weaken matching; the flags and scorers; generation
never reads gold text, the cut twin is the plain LoRA model, and shuffle isolates the true partner in duplex and
dialogue; every game generates and scores, independent of the batch size; the summaries, empty strata and the guard
against summarizing an incomplete run.
"""
import copy
import json

import pytest
import torch

import coread as C
import evalfree as F
import evaluate as E
import link
import play_tasks as P
import session as S
from payload_ctl import PAYLOAD_GAMES


@pytest.fixture(scope='module')
def corpus(tmp_path_factory, write_data):
    out = tmp_path_factory.mktemp('evalfree')
    replies = write_data(out)
    plans, excluded = F.make_plans(replies, n=2, seed=993, coread_data=out / 'coread.json')
    return out, replies, plans, excluded


@pytest.fixture(scope='module')
def loaded(tiny_models, corpus):
    """The pair saved as a step (q/v LoRA, alpha 1) and loaded back with evaluate.load_pair."""
    toks, models = tiny_models(['q_proj', 'v_proj'], std=.03)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for bridge in pair.bridge.values():
        bridge.alpha.data.fill_(1.)
    step = corpus[0] / 'step_tiny'
    step.mkdir()
    torch.save(pair.checkpoint(), step / 'bridge.pt')
    for x in link.SIDES:
        models[x].save_pretrained(step / x)
    return E.load_pair(str(step))[0]


@pytest.fixture(scope='module')
def records(loaded, corpus):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('NL_MAXNEW', '2')
        return [r for ps in corpus[2].values() for r in F.evaluate_game(loaded, ps, bs=2)]


def test_plans_cover_every_game_with_matched_donors(corpus):
    out, replies, plans, _ = corpus
    assert set(plans) == set(PAYLOAD_GAMES) | set(P.SCORED) | {'co_talk'}
    repeat, _ = F.make_plans(replies, n=2, seed=993, coread_data=out / 'coread.json', games=['relay_word', 'co_swap'])
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


def test_match_failures(corpus, write_data, tmp_path):
    _, replies, plans, _ = corpus
    plan = plans['relay_word'][0]
    ep, donor = plan['episode'], plan['donor']
    assert F.matched(ep, donor)
    assert not F.matched(ep, copy.deepcopy(ep))
    bad = copy.deepcopy(donor)
    bad['meta']['sender'], bad['meta']['receiver'] = bad['meta']['receiver'], bad['meta']['sender']
    assert not F.matched(ep, bad)
    # A corpus with only one distinct item has no valid donor.
    path = tmp_path / 'one' / 'coread.json'
    path.parent.mkdir()
    it = dict(task='math', src='one', question='What is 2+2?', solution='4')
    path.write_text(json.dumps(dict(train=[], test=[it], weights={'one': 1})))
    source = C.Coread(replies, 'test', path=str(path))
    with pytest.raises(ValueError, match='No valid matched donor'):
        F.choose_donor(source.episode('co_math'), source, max_attempts=20)
    # A stale talk.json bypasses neither the corpus holdout nor the conversation holdout.
    root = tmp_path / 'stale'
    root.mkdir()
    write_data(root)
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


def test_flags_and_contains(corpus):
    plans = corpus[2]
    ep = copy.deepcopy(plans['relay_word'][0]['episode'])
    x = ep['meta']['receiver']
    ep['system'][x] += ' ' + ep['meta']['value'].upper() + '!!!'
    assert not F.flags(ep, x)['exclusive']
    assert F.contains('The CITY, is\nLiMa!', 'city is lima')
    assert not F.contains('150', '15')
    for p in plans['co_nli']:
        assert all(not F.flags(p['episode'], x)['exclusive'] for x in link.SIDES)
        assert all(F.flags(p['episode'], x)['symmetric'] for x in link.SIDES)


@torch.inference_mode()
def test_generation_ignores_gold_and_shuffle_isolates(loaded, corpus):
    plan = corpus[2]['relay_word'][0]
    pair = loaded
    eps = [plan['episode'], plan['donor']]
    x = eps[0]['meta']['receiver']
    y = link.partner(x)
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
    # With the link on the private message does change the receiver: the bridges are active.
    a = S.generate(pair, eps, max_turn=3, keep=True)
    b = S.generate(pair, changed, max_turn=3, keep=True)
    length = min(len(a[0]['h'][x]), len(b[0]['h'][x]))
    assert (a[0]['h'][x][:length] - b[0]['h'][x][:length]).abs().max() > 1e-4
    cut = S.generate(pair, eps, mode='zero', max_turn=3, keep=True)
    for side in link.SIDES:
        # The cut prompt states are the standalone LoRA model's.
        ids = S.prefix(pair.toks[side], side, eps[0]['system'][side], eps[0]['task'][side])
        ids += S.assistant_header(pair.toks[side], side)
        inp = torch.tensor([ids[:-1]])
        hidden = link.inner(pair.models[side])[0](inp).last_hidden_state[0]
        assert torch.allclose(cut[0]['h'][side][:len(inp[0])], hidden, atol=2e-6, rtol=2e-6)
    assert all(p.requires_grad is False for p in pair.models[x].parameters())


@torch.inference_mode()
def test_dialogue_shuffle_isolates_the_holder(loaded, corpus):
    # With the listener's text turns fed back, a duplex test or a turn count alone would miss this path.
    talk = corpus[2]['co_talk'][0]
    ep, donor = talk['episode'], talk['donor']
    altered = copy.deepcopy(ep)
    asker = ep['meta']['asker']
    altered['system'][link.partner(asker)] = 'A new private context about violet elephants.'
    a = F.generate_shuffle(loaded, ep, donor, 5, keep=True)
    b = F.generate_shuffle(loaded, altered, donor, 5, keep=True)
    assert torch.equal(a['h'][asker], b['h'][asker]), 'dialogue reconnected true partner'
    assert len(a['turns']) == 1 and a['turns'][0][0] == asker


def test_generation_limit(monkeypatch):
    monkeypatch.setenv('NL_MAXNEW', '2')
    assert F.generation_limit('co_code_write') == 2
    monkeypatch.setenv('NL_MAXNEW', '999')
    assert F.generation_limit('relay_word') == 64


def test_every_game_generates_and_scores(corpus, records):
    plans = corpus[2]
    for game, ps in plans.items():
        rs = [r for r in records if r['game'] == game]
        assert len(rs) == sum(len(F.payloads(p['episode'])) for p in ps), game
        for r in rs:
            for c in F.CONDITIONS:
                assert 0 <= r['conditions'][c]['score'] <= 1
                assert r['conditions'][c]['generated_tokens'] <= 2
                if game == 'co_talk':
                    assert set(r['conditions'][c]['texts']) == {r['receiver']}
            assert r['partner_role'] == r['donor']['partner_role']
            assert r['partner_item_id'] != r['donor']['partner_item_id']


def test_batch_size_invariance(loaded, corpus, records, monkeypatch):
    monkeypatch.setenv('NL_MAXNEW', '2')
    one = list(F.evaluate_game(loaded, corpus[2]['merge_sum'][:1], bs=2))
    batched = [r for r in records if r['game'] == 'merge_sum' and r['episode_index'] == 0]
    assert [r['conditions'] for r in one] == [r['conditions'] for r in batched]


def test_summaries_and_incomplete_run_guard(corpus, records, tmp_path):
    _, _, plans, excluded = corpus
    out = tmp_path
    manifest = dict(schema=F.SCHEMA, complete=True, run='step_tiny', seed=993, tiny=True, games=list(plans),
                    receiver_records=len(records), excluded=excluded)
    (out / 'episodes.jsonl').write_text(''.join(json.dumps(rec) + '\n' for rec in records))
    manifest['episodes_sha256'] = F.file_hash(out / 'episodes.jsonl')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    table = json.loads(F.summarize(out, 'tiny_test', results_dir=out / 'results').read_text())
    assert len(table['rows']) == (len(plans) + 2) * 4
    empty = [r for r in table['rows'] if r['game'] == 'co_nli' and r['subset'] == 'exclusive']
    assert all(r['n'] == 0 and r['mean']['link'] is None and r['link-cut']['ci'] == [-1., 1.] for r in empty)
    pool = next(r for r in table['rows'] if r['game'] == 'pooled_coread' and r['receiver'] == 'quill'
                and r['subset'] == 'all')
    assert 'co_talk' not in pool['game_n'] and set(pool['game_n']) == set(P.SCORED)
    manifest['complete'] = False
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        F.summarize(out, 'incomplete', results_dir=out / 'results')
