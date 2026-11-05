"""The corpus tasks (swarm/corpus.py) on a made-up cache: every source and task builds training and test episodes in
dialogue.py's format from its own split only, deterministically given the rng; every view stays under VIEW_MAX and
every answer under ANSWER_MAX with the Qwen3-4B tokenizer (Qwen3-0.6B's where 4B's is not cached; skipped where neither
is); the gold answer scores 1 and a wrong one less; the code sandbox stops loops and contains failures; the mix weights
are 60/40 by family. test_live runs the same checks on the real cache when CORPUS_CACHE points at one.

Usage: python -m pytest -q tests/test_corpus.py   (from the repository root)
"""
import json
import os
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import corpus as C                           # noqa: E402

WORDS = ('river stone garden window market teacher letter music winter island harbor engine forest lantern violin '
         'mountain village silver orchard bridge canyon meadow compass ribbon').split()


def tok_ok():
    try:
        C.tokenizer()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not tok_ok(), reason='no Qwen3 tokenizer in the local cache')


def prose(rng, n, tag):
    out = []
    for _ in range(n):
        w = [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]
        out.append(f'The {tag} ' + ' '.join(w) + '.')
    return ' '.join(out)


def mock_rows(name, test):
    """Small made-up rows for one source or task; every text carries its split's tag."""
    rng = random.Random(f'{name}-{test}')
    tag = 'heldout' if test else 'trainpart'
    if name == 'fill_chat':
        return [dict(turns=[dict(role='user' if i % 2 == 0 else 'assistant', content=prose(rng, 4, tag))
                            for i in range(8)]) for _ in range(6)]
    if name == 'fill_code':
        return [dict(text='\n'.join(f'def f_{tag}_{i}(x, y):\n    """Adds {i}."""\n    return x + y + {i}\n'
                                    for i in range(40))) for _ in range(6)]
    if name == 'fill_wikix':
        de = f'Der Fluss {tag} fließt durch das Tal, und die Brücke über ihn ist alt. ' * 30
        ja = f'川は谷を流れ、橋はとても古い。{tag}の村には市場があります。' * 30
        return [dict(text=de), dict(text=ja)] * 3
    if name in C.SOURCES:
        return [dict(text=prose(rng, 40, tag)) for _ in range(6)]
    if name == 'gsm8k':
        return [dict(question=f'Ann has {i + 3} apples. Bo gives her {i} more. She eats 2 of them. The {tag} basket '
                              'is full. How many apples does Ann have now?', answer=str(2 * i + 1))
                for i in range(1, 7)]
    if name == 'qa_squad':
        return [dict(context=prose(rng, 6, tag) + f' The {tag} key is under the stone number {i}.',
                     question='Where is the key?', answers=[f'under the stone number {i}']) for i in range(6)]
    if name == 'qa_hotpot':
        return [dict(question='Which river flows past the town?', answer=f'River {tag} {i}',
                     titles=['Town', 'River'], pages=[prose(rng, 3, tag), f'River {tag} {i} flows past the town.'])
                for i in range(6)]
    if name == 'drop':
        return [dict(passage=prose(rng, 3, tag) + f' The team scored {i} goals. Later they scored 2 more.',
                     question='How many goals did the team score in total?', answers=[str(i + 2)], types=['number'])
                for i in range(6)]
    if name == 'mbpp':
        return [dict(text=f'Write a function to add {i} to a number.', code=f'def add_{i}(x):\r\n\treturn x + {i}',
                     test_list=[f'assert add_{i}(1) == {i + 1}', f'assert add_{i}(0) == {i}'], test_setup_code='')
                for i in range(6)]
    if name == 'translate':
        return [dict(de=f'Der {tag} Hund läuft schnell über die grüne Wiese am Fluss {i}.',
                     en=f'The {tag} dog runs quickly across the green meadow by river {i}.') for i in range(6)]
    if name == 'summarize':
        return [dict(article=prose(rng, 30, tag), highlights=f'The {tag} town opened a market .\nIt sells fish {i} .')
                for i in range(6)]
    if name == 'wordnet20':
        nouns = [f'{w}{"x" if test else ""}' for w in WORDS]
        return [dict(noun=n, definition=f'a kind of {n}', near=[m for m in nouns if m != n]) for n in nouns[:6]]
    raise ValueError(name)


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'CACHE', tmp_path)
    monkeypatch.setattr(C, '_rows', {})
    for name in C.TASKS:
        for test in (False, True):
            C.write_rows(name, test, mock_rows(name, test))
    return tmp_path


def check_episode(name, ep, test):
    assert ep['game'] == name and ep['family'] == C.FAMILY[name]
    assert set(ep['views']) == set(ep['answers']) == set(ep['weights']) == {'A', 'B'}
    assert ep['metric'] in ('chrf', 'f1', 'number', 'tests', 'rougeL', 'exact')
    for x in 'AB':
        assert C.ntok(ep['views'][x]) <= C.VIEW_MAX, (name, x, C.ntok(ep['views'][x]))
        assert C.ntok(ep['answers'][x]) <= C.ANSWER_MAX, (name, x, C.ntok(ep['answers'][x]))
        assert ep['views'][x].strip() and ep['answers'][x].strip()
        assert ep['max_new'] >= C.ntok(ep['answers'][x])
    json.dumps(ep)                                       # golds and cands are JSON-able
    if 'cands' in ep:
        assert ep['answers']['B'] in ep['cands']['B']


@pytest.mark.parametrize('name', sorted(C.TASKS))
def test_episodes(cache, name):
    for test in (False, True):
        for i in range(3):
            ep = C.TASKS[name](random.Random(f'x-{name}-{i}'), test)
            check_episode(name, ep, test)
            text = json.dumps(ep, ensure_ascii=False)
            other = 'trainpart' if test else 'heldout'
            assert other not in text, f'{name} {"test" if test else "train"} episode drew the other split'
            assert ep == C.TASKS[name](random.Random(f'x-{name}-{i}'), test), 'not a function of rng alone'


@pytest.mark.parametrize('name', sorted(C.TASKS))
def test_gold_scores_best(cache, name):
    ep = C.TASKS[name](random.Random(f'g-{name}'), False)
    gold = C.score(ep, 'B', ep['answers']['B'])['main']
    assert gold > 0.99, (name, gold)
    wrong = C.score(ep, 'B', 'I do not know.')['main']
    assert wrong < gold


def test_fill_split_and_answers(cache):
    ep = C.TASKS['fill_web'](random.Random(0), False)
    chunks = ep['golds']['chunks']
    assert len(chunks) == 2 * C.CHUNKS
    for x, own in (('A', range(0, 2 * C.CHUNKS, 2)), ('B', range(1, 2 * C.CHUNKS, 2))):
        for i in own:
            assert chunks[i].strip() in ep['views'][x]
            assert chunks[i].strip() in ep['answers']['B' if x == 'A' else 'A']
            assert chunks[i].strip() not in ep['views']['B' if x == 'A' else 'A']
    ja = [C.TASKS['fill_wikix'](random.Random(i), False) for i in range(6)]
    assert not any('�' in ep['views']['A'] + ep['answers']['A'] for ep in ja), 'a chunk split a character'


def test_mix():
    assert set(C.MIX) == set(C.TASKS) == set(C.FAMILY)
    assert abs(sum(C.MIX.values()) - 1) < 1e-9
    assert abs(sum(v for n, v in C.MIX.items() if C.FAMILY[n] == 1) - 0.6) < 1e-9


def test_mix_of(cache):
    assert C.available() == list(C.TASKS)
    m = C.mix_of(['fill_web', 'fill_code', 'gsm8k'])
    assert set(m) == {'fill_web', 'fill_code', 'gsm8k'} and abs(sum(m.values()) - 1) < 1e-9
    assert abs(m['gsm8k'] - 0.4) < 1e-9


def test_sample_draws_many(cache):
    rng = random.Random(1)
    games = [C.sample(rng)['game'] for _ in range(120)]
    assert len(set(games)) >= 10


def test_sandbox():
    assert C.run_tests('def f(x):\n    return x + 1', ['assert f(1) == 2', 'assert f(2) == 4']) == 0.5
    assert C.run_tests('while True:\n    pass', ['assert True'], timeout=3) == 0.0
    assert C.run_tests('def f(:', ['assert f(1) == 2']) == 0.0
    assert C.run_tests('import os\nos.listdir("/")', ['assert True']) == 1.0          # runs, in its own empty folder
    big = 'open("big.bin", "wb").write(b"0" * (8 << 20))'
    assert C.run_tests(big, ['assert True']) == 0.0, 'the file-size limit should stop an 8 MB write'


def test_metrics():
    assert C.chrf('the cat sat', 'the cat sat') == pytest.approx(1.0)
    assert 0 < C.chrf('the cat sat', 'a cat sits') < 1
    assert C.rouge_l('a b c d', 'a b c d') == pytest.approx(1.0)
    assert C.rouge_l('a x c', 'a b c') == pytest.approx(2 / 3)
    assert C.numbers('it is 1,234.5 or -2') == [1234.5, -2.0]


def live_cache():
    p = os.environ.get('CORPUS_CACHE')
    return bool(p) and any(Path(p).glob('*.train.jsonl'))


@pytest.mark.skipif(not live_cache(), reason='CORPUS_CACHE does not point at a cache')
@pytest.mark.parametrize('name', sorted(C.TASKS))
def test_live(name):
    for test in (False, True):
        if not C.path_of(name, test).exists():
            pytest.skip(f'{name} not in the cache')
        for i in range(3):
            ep = C.TASKS[name](random.Random(f'live-{name}-{i}'), test)
            check_episode(name, ep, test)
            assert C.score(ep, 'B', ep['answers']['B'])['main'] > 0.99
