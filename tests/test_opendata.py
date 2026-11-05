"""swarm/opendata.py on small fake sources: material splits, the leak and credential filters, mutations that fail real
tests, skeletons per family and split, generation with a stand-in generator, the one-pass teacher, the top-k KL, and
episodes that never repeat. CPU only (a tiny random Qwen3 with the real tokenizer)."""
import gzip
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import opendata as O                         # noqa: E402

WORDS = ('river stone lamp orbit candle maple violet harbor engine signal copper meadow lantern quartz falcon timber '
         'glacier pepper saddle wander').split()


def prose(rng, n):
    return ' '.join(rng.choice(WORDS) for _ in range(n))


def doc(rng, n=420):
    return '\n\n'.join(prose(rng, 60) + '.' for _ in range(n // 60))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Fake items for every family's sources, train and test, in a temporary item cache."""
    monkeypatch.setattr(O, 'CACHE', tmp_path)
    rng = random.Random(0)
    def chat(k, i):
        users = [prose(rng, 25) + f' turn {j}?' for j in range(k)]
        turns = []
        for u in users:
            turns += [dict(role='user', content=u), dict(role='assistant', content=prose(rng, 20))]
        return dict(id=f'c{i}', source='wildchat', kind='chat', turns=turns)
    def pasted(i):
        code = 'def f(x):\n' + '\n'.join(f'    x = x + {j}  # {prose(rng, 4)}' for j in range(12)) + '\n    return x'
        u = f'Why does this fail?\n```python\n{code}\n```'
        return dict(id=f'p{i}', source='stackexchange', kind='chat', turns=[dict(role='user', content=u)])
    fake = dict(
        wildchat=[chat(4, i) for i in range(30)], stackexchange=[pasted(i) for i in range(20)],
        fineweb_edu=[dict(id=f'd{i}', source='fineweb_edu', kind='doc', text=doc(rng)) for i in range(40)],
        github=[dict(id=f'g{i}', source='github', kind='code', text=doc(rng)) for i in range(20)],
        mbpp=[dict(id=f'm{i}', source='mbpp', kind='tests', code=f'def add(a, b):\n    return a + b + {i}',
                   tests=[f'assert add(1, 2) == {3 + i}', f'assert add(0, 0) == {i}'], setup='', text='add')
              for i in range(4)],
        arxiv=[dict(id=f'a{i}', source='arxiv', kind='doc', text=doc(rng)) for i in range(20)])
    (tmp_path / 'items').mkdir()
    for name, xs in fake.items():
        for split in ('train', 'test'):
            if name in O.HELD_OUT and split == 'train':
                continue
            with open(O.item_path(name, split), 'w') as f:
                for x in xs:
                    f.write(json.dumps(dict(x, id=f'{x["id"]}-{split}')) + '\n')
    return tmp_path


def test_split_material():
    code = 'def g(y):\n' + '\n'.join(f'    y += {j}' for j in range(40))
    kind, mat, ask = O.split_material(f'Fix this please:\n```python\n{code}\n```\nIt crashes.')
    assert kind == 'code' and 'y += 39' in mat and 'crashes' in ask and 'y += 39' not in ask
    table = '\n'.join(f'| {i} | name{i} | {i * 7} |' for i in range(8))
    kind, mat, ask = O.split_material(f'Which name has the largest value?\n{table}')
    assert kind == 'table' and 'name7' in mat and 'largest' in ask
    rng = random.Random(1)
    body = doc(rng, 300)
    kind, mat, ask = O.split_material(body + '\n\nSummarize the passage above in two lines.')
    assert kind == 'passage' and ask.startswith('Summarize') and 'Summarize' not in mat
    assert O.split_material('hi there, how are you?') is None


def test_leak_and_credential_filters():
    hidden = 'the quick brown fox jumps over the lazy dog near the old stone bridge at dawn today'
    assert O.leaks('so tell me: the quick brown fox jumps over the lazy dog near the old stone bridge', hidden)
    assert not O.leaks('what animal crosses the bridge, and when?', hidden)
    bad = dict(conversation=[dict(role='user', content='my key is sk-' + 'a1B2' * 8 + ' ok? ' + 'x ' * 10),
                             dict(role='assistant', content='sure')] * 3, toxic=False)
    good = dict(conversation=[dict(role='user', content=f'question {i} about rivers?') for i in range(3)], toxic=False)
    assert O.to_item('wildchat', bad, 0) is None
    assert O.to_item('wildchat', good, 0)['kind'] == 'chat'
    assert O.to_item('wildchat', dict(good, toxic=True), 0) is None


def test_mutants_fail_real_tests():
    code = 'def add(a, b):\n    if a < b:\n        return a + b\n    return b + a'
    tests = ['assert add(1, 2) == 3', 'assert add(5, 1) == 6']
    assert O.failing_output(code, tests) is None
    bad = O.mutants(code, random.Random(0))
    assert bad and all(b != code for b in bad)
    outs = [O.failing_output(b, tests) for b in bad]
    assert any(o and 'FAILED: assert add' in o and 'AssertionError' in o for o in outs)


def test_skeletons_cover_the_families_and_keep_splits_apart(cache):
    sks = O.skeletons('train', 120, seed=0)
    fams = {s['family'] for s in sks}
    assert {'chat_alt', 'chat_doc', 'doc', 'pair', 'code', 'debug', 'plan', 'fill'} <= fams, fams
    assert len({s['id'] for s in sks}) == len(sks)
    assert not any(s['source'] in O.HELD_OUT for s in sks)
    held = O.skeletons('eval_heldout', 20, seed=0)
    assert held and all(s['source'] in O.HELD_OUT for s in held)
    for s in sks:
        assert O.MIN_TURNS <= len(s['turns']) <= O.MAX_TURNS
        assert all(O.TALK[0] <= t['talk'] <= O.TALK[1] for t in s['turns'])
        if s['family'] == 'chat_alt':                       # alternate holders, each answering its own turns
            assert [t['who'] for t in s['turns']][:2] in (['A', 'B'], ['B', 'A'])
            assert s['views'] == dict(A='', B='')
        if s['family'] == 'chat_doc':                       # A holds the pasted code, B only the request
            assert 'x = x + 11' in s['views']['A'] and 'x = x + 11' not in s['turns'][0]['user']
            assert 'x = x + 11' in s['turns'][0]['teacher_user']
        if s['family'] == 'debug':
            assert 'FAILED:' in s['views']['B'] and 'def add' in s['views']['A']
        if s['family'] == 'fill':
            lens = [O.ntok(t['response']) for t in s['turns']]
            assert all(4 <= n <= 200 for n in lens)


def fake_gen(copy_from=None):
    """A stand-in for vllm_gen: the simulated user asks a question (or copies copy_from, to trip the leak filter),
    the planner returns JSON, and the teacher answers with a short text and its token ids."""
    calls = []

    def gen(prompts, max_tokens, temperature, topk=0):
        out = []
        for p in prompts:
            if isinstance(p[0], int):
                text = f'answer {len(calls)}: ' + ' '.join(WORDS[:5])
                ids = O.piece(text) + ([O.IM_END] if len(calls) % 2 else [])   # ended, or cut at the length
                n = len(ids)
                top = (np.arange(n * topk, dtype=np.int32).reshape(n, topk) + len(calls),
                       -np.linspace(0.1, 9, n * topk).reshape(n, topk).astype(np.float16))
                out.append((text, ids, top))
            elif 'JSON' in p[0]['content']:
                out.append((json.dumps(dict(situation='a trip for two', constraints_A=['leave after 9'],
                                            constraints_B=['budget 300'])), [], None))
            else:
                text = copy_from if copy_from and 'you have not seen' in p[1]['content'] else 'What does it say?'
                out.append((text, [], None))
            calls.append(p)
        return out
    return gen, calls


def test_generate_fills_every_turn(cache):
    sks = O.skeletons('train', 60, seed=1)
    recs = O.generate(sks, fake_gen()[0], random.Random(0), log=lambda *_: None)
    assert len(recs) >= 40
    for r in recs:
        assert O.MIN_TURNS <= len(r['turns'])
        for k, t in enumerate(r['turns']):
            assert t['user'] and t['response'] and t['response_ids'] and 'ask' not in t
            if r['teacher']:
                assert t['teacher_user']
        if r['family'] in ('doc', 'code', 'debug', 'pair'):
            assert r['turns'][0]['teacher_user'].startswith(r['material'][:50])
            assert r['material'][:50] not in r['turns'][0]['user']
    plans = [r for r in recs if r['family'] == 'plan']
    assert plans and all(r['views']['A'] == '- leave after 9' for r in plans)


def test_leaking_requests_are_cut(cache):
    sks = [s for s in O.skeletons('train', 80, seed=2) if s['family'] == 'doc']
    leak = ' '.join(sks[0]['material'].split()[:30])
    recs = O.generate([sks[0]], fake_gen(copy_from=leak)[0], random.Random(0), log=lambda *_: None)
    assert recs == []


def tiny_lm():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(O.tok()) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_implementation='sdpa')
    return Qwen3ForCausalLM(cfg).eval()


def test_one_teacher_pass_scores_every_turn_in_its_generation_context(cache):
    recs = O.generate(O.skeletons('train', 30, seed=3), fake_gen()[0], random.Random(0), log=lambda *_: None)
    rec = next(r for r in recs if r['teacher'] and len(r['turns']) >= 3)
    full, starts = O.teacher_ids(rec)
    for k in range(len(rec['turns'])):
        pre, _ = O.teacher_ids(rec, k)
        assert full[:len(pre)] == pre and starts[k] == len(pre)   # turn k's answer follows exactly its prompt
    model = tiny_lm()
    one = O.teacher_targets(model, O.tok(), rec, k=8)
    for k, t in enumerate(rec['turns']):
        pre, _ = O.teacher_ids(rec, k)
        with torch.no_grad():
            z = model(input_ids=torch.tensor([pre + t['response_ids']])).logits[0, len(pre) - 1:-1].log_softmax(-1)
        lp, ids = z.topk(8, -1)
        assert torch.equal(one[k][0], ids) and torch.allclose(one[k][1], lp, atol=1e-4)


def test_kl_topk_is_zero_at_the_teacher_and_charges_mass_outside():
    torch.manual_seed(0)
    z = torch.randn(5, 50)
    lp, ids = z.log_softmax(-1).topk(8, -1)
    assert O.kl_topk(z, ids, lp).abs().max() < 1e-5
    worse = z.clone()
    worse.scatter_(-1, ids, -5.0)                        # the student moves the teacher's top-8 mass elsewhere
    assert (O.kl_topk(worse, ids, lp) > 0.5).all()


def test_episodes_never_repeat(cache, tmp_path):
    recs = O.generate(O.skeletons('train', 40, seed=4), fake_gen()[0], random.Random(0), log=lambda *_: None)
    nxt = O.write_shards(recs[:15], tmp_path, 'train', 0, size=10)
    assert nxt == 2
    O.write_shards(recs[15:], tmp_path, 'train', nxt, size=10)
    got = [r['id'] for r in O.Episodes(tmp_path, 'train', seed=5)]
    assert sorted(got) == sorted(r['id'] for r in recs) and len(set(got)) == len(got)
    assert len(O.Episodes(tmp_path, 'train', families=['fill'])) == sum(r['family'] == 'fill' for r in recs)
    with gzip.open(tmp_path / 'shards' / 'train-0000.jsonl.gz', 'rt') as f:
        assert sum(1 for _ in f) == 10


def test_teacher_topk_is_stored_beside_the_shards_and_read_back(cache, tmp_path):
    recs = O.generate(O.skeletons('train', 40, seed=6), fake_gen()[0], random.Random(0), log=lambda *_: None)
    want = {(r['id'], k): t['_top'] for r in recs for k, t in enumerate(r['turns']) if '_top' in t}
    assert want and all(r['teacher'] == all('_top' in t for t in r['turns']) for r in recs)
    O.write_shards(recs, tmp_path, 'train', 0, size=12)
    with gzip.open(tmp_path / 'shards' / 'train-0000.jsonl.gz', 'rt') as f:
        assert '_top' not in f.read()                                 # the records hold no top-k
    eps = list(O.Episodes(tmp_path, 'train', teacher=True))
    assert len(eps) == len(recs)
    for r in eps:
        if not r['teacher']:
            assert r['topk'] == {}
            continue
        for k, t in enumerate(r['turns']):
            ids, lp = r['topk'][k]
            assert ids.shape == (len(t['response_ids']), O.TOPK) and ids.dtype == torch.int32
            assert np.array_equal(ids.numpy(), want[(r['id'], k)][0])
            assert np.array_equal(lp.numpy(), want[(r['id'], k)][1]) and lp.dtype == torch.float16
    ended = [t for r in eps for t in r['turns'] if r['teacher'] and t['response_ids'][-1] == O.IM_END]
    assert ended                                                     # an ended answer is followed by a newline only
    r = next(r for r in eps if r['teacher'] and r['turns'][0]['response_ids'][-1] == O.IM_END)
    full, starts = O.teacher_ids(r)
    n0 = len(r['turns'][0]['response_ids'])
    assert full[starts[0] + n0 - 1] == O.IM_END and O.tok().decode(full[starts[0] + n0:starts[0] + n0 + 1]) == '\n'


def test_top_arrays_place_tokens_by_rank():
    L = lambda lp, rank: types.SimpleNamespace(logprob=lp, rank=rank)      # noqa: E731
    dicts = [{5: L(-0.1, 1), 7: L(-2.0, 2), 9: L(-7.0, 40)}, {3: L(-0.5, 2), 4: L(-0.3, 1)}]
    ids, lp = O.top_arrays(dicts, 2, 2)
    assert ids.tolist() == [[5, 7], [4, 3]] and lp.dtype == np.float16
    assert np.allclose(lp, [[-0.1, -2.0], [-0.3, -0.5]], atol=1e-2)
    flat = types.SimpleNamespace(start_indices=[0, 3], end_indices=[3, 5], token_ids=[9, 5, 7, 4, 3],
                                 logprobs=[-7.0, -0.1, -2.0, -0.3, -0.5], ranks=[40, 1, 2, 1, 2])
    ids2, lp2 = O.top_arrays(flat, 2, 2)
    assert ids2.tolist() == ids.tolist() and np.array_equal(lp2, lp)
