"""Shared setup: every test runs tiny random models of both architectures (link.tiny, NL_TINY=1) on the CPU, imports
the pairing/ modules directly, and reads only the small made-up data files written here.

Usage: python -m pytest -q tests   (from the repository root; the MLX tests run only where MLX is installed)
"""
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.update(NL_TINY='1', NL_DEVICE='cpu', MLX_ENABLE_TF32='0')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))

import pytest
import torch
from peft import LoraConfig, get_peft_model

import coread
import games
import link

torch.set_num_threads(2)

TEST_CITIES = ('Lima', 'Oslo', 'Rome', 'Suva')
TRAIN_CITIES = ('Accra', 'Bern', 'Cairo', 'Hanoi', 'Kyiv', 'Perth')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: runs train.py, rl.py or another script in a subprocess')
    # PEFT looks up the base model's config when saving an adapter; the tiny models have none on the hub.
    config.addinivalue_line('filterwarnings', 'ignore:Could not find a config file:UserWarning')


@pytest.fixture(scope='session')
def tiny_models():
    """tiny_models(targets=None, r=4, lora_alpha=8, std=None) -> (toks, models): both tiny models in eval mode, with
    LoRA adapters on the modules `targets` when given, their lora_B drawn from N(0, std) when std is given."""
    def load(targets=None, r=4, lora_alpha=8, std=None):
        toks, models = {}, {}
        for x in link.SIDES:
            toks[x], m = link.load(x)
            if targets:
                m = get_peft_model(m, LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=list(targets)))
                if std:
                    for n, p in m.named_parameters():
                        if 'lora_B' in n:
                            torch.nn.init.normal_(p, std=std)
            models[x] = m.eval()
        return toks, models
    return load


@pytest.fixture(scope='session')
def plain_pair(tiny_models):
    """plain_pair(alpha) -> (pair, toks): a pair of the plain tiny models (no LoRA) in eval mode, every alpha set to
    `alpha` (at 0 the bridges write nothing)."""
    def make(alpha):
        toks, models = tiny_models()
        pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
        for side in pair.bridge.values():
            side.alpha.data.fill_(alpha)
        return pair, toks
    return make


@pytest.fixture(scope='session')
def runtime(plain_pair):
    """runtime() -> a steered_mind.Runtime on the torch backend over plain_pair(0.2)."""
    def make():
        import steered_mind
        rt = steered_mind.Runtime.__new__(steered_mind.Runtime)
        rt.backend = 'torch'
        rt.pair, rt.toks = plain_pair(0.2)
        return rt
    return make


@pytest.fixture(scope='module')
def pair(tiny_models):
    """A pair with q/v LoRA and open bridges: alpha 0.4, recency bias and gates drawn at random."""
    toks, models = tiny_models(['q_proj', 'v_proj'])
    torch.manual_seed(27)
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16, sizes={x: [.04, .05, .06] for x in link.SIDES}).eval()
    with torch.no_grad():
        for b in p.bridge.values():
            b.alpha.fill_(.4)
            b.age.normal_(0, .2)
            for gate in b.gate:
                gate.weight.normal_(0, .05)
                gate.bias.normal_(1.5, .2)
    return p


SYSTEM = {x: f'You are {link.MODELS[x]["name"]}.' for x in link.SIDES}


@pytest.fixture(scope='module')
def duplex():
    """Three teacher-forced duplex episodes."""
    return [dict(mode='duplex', system=SYSTEM, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
                 reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue are three.', 1.0)}),
            dict(mode='duplex', system=SYSTEM, task={'quill': 'Tell me a story.', 'ember': 'Why is the sea salty?'},
                 reply={'quill': ('Once upon a time', 1.0), 'ember': ('Because rivers carry minerals there over time.', 1.0)}),
            dict(mode='duplex', system=SYSTEM, task={'quill': 'Say hi.', 'ember': 'Count to three, please, slowly.'},
                 reply={'quill': ('Hi!', 1.0), 'ember': ('One, two, three.', 1.0)})]


@pytest.fixture(scope='module')
def dialogue():
    """Two teacher-forced dialogue episodes, one opened by each side."""
    return [dict(mode='dialogue', system=SYSTEM, task={'quill': 'Ask Ember a question.', 'ember': None},
                 turns=[('quill', 'Why is the sky blue, Ember?', 1.0),
                        ('ember', 'Sunlight scatters off air; blue light scatters most.', 1.0),
                        ('quill', 'Thanks! Café crème? Ünïcode works.', 1.0)]),
            dict(mode='dialogue', system=SYSTEM, task={'quill': None, 'ember': 'Talk about cats.'},
                 turns=[('ember', 'Cats sleep a lot, about sixteen hours a day.', 1.0),
                        ('quill', 'Wow.', 1.0)])]


def corpus_items(city, i):
    """One co-reading item of every task about city; i keeps the texts of different cities distinct."""
    items = [dict(task='passage', kind='prose', text=f'{city} has a river. The boat is {i} years old.'),
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
                  tests=[f'assert f() == {i}']),
             dict(task='instruct', instruction=f'Name the city of person {i}.', response=city)]
    return [dict(it, src=it['task']) for it in items]


@pytest.fixture(scope='session')
def write_data():
    """write_data(out, train=(), steer=False) -> replies: writes into out a co-reading corpus (coread.json: one item of
    every task per city, the TEST_CITIES in the test split and the cities `train` in the training split), co_talk
    conversations about its qa items (talk.json) and replies.json. Every own reply is 'An ordinary answer.'. With steer,
    the replies also list one steered reply per training question (for the steer games) and an own reply to every
    co-reading request (for co_distill)."""
    def write(out, train=(), steer=False):
        out = Path(out)
        questions = [q for sp in ('train', 'test') for q in games.POOLS['questions'][sp]]
        replies = {x: dict(own={q: 'An ordinary answer.' for q in questions}, steer=[]) for x in link.SIDES}
        split = dict(test=[it for i, c in enumerate(TEST_CITIES) for it in corpus_items(c, i)],
                     train=[it for i, c in enumerate(train, len(TEST_CITIES)) for it in corpus_items(c, i)])
        data = dict(train=split['train'], test=split['test'], weights={r['src']: 1 for r in split['test']})
        talk = {sp: [dict(item=it, asker=asker, system={x: coread.talk_system(it, x, asker) for x in link.SIDES},
                          turns=[(asker, coread.talk_turn0(it)), (link.partner(asker), 'A later response.'),
                                 (asker, 'Thanks.')])
                     for it in its if it['task'] == 'qa' for asker in link.SIDES]
                for sp, its in split.items()}
        if steer:
            concepts = games.POOLS['concepts']['train']
            for x in link.SIDES:
                replies[x]['steer'] = [(q, concepts[k % len(concepts)][0],
                                        f'Thinking of {concepts[k % len(concepts)][1]}, here is my answer.')
                                       for k, q in enumerate(games.POOLS['questions']['train'])]
                replies[x]['coread'] = {coread.request(it)[2]: f'{x} reads: ' + ' '.join(coread.request(it)[0].split()[:12])
                                        for its in split.values() for it in its if coread.request(it)}
        (out / 'coread.json').write_text(json.dumps(data))
        (out / 'talk.json').write_text(json.dumps(dict(train=talk['train'], test=talk['test'])))
        (out / 'replies.json').write_text(json.dumps(replies))
        return replies
    return write


@pytest.fixture(scope='session')
def data(tmp_path_factory, write_data):
    """A directory with replies.json (steered and distillation replies included) and a corpus of both splits."""
    out = tmp_path_factory.mktemp('data')
    write_data(out, TRAIN_CITIES, steer=True)
    return out


@pytest.fixture(scope='session')
def replies(data):
    return json.loads((data / 'replies.json').read_text())


@pytest.fixture(scope='session')
def tree(tmp_path_factory):
    """A copy of pairing/ beside a data/ directory with pools.json and a made-up bench.json (four-choice benchmark
    questions for train.py --qa), for scripts that read data files at their fixed paths."""
    root = tmp_path_factory.mktemp('tree')
    shutil.copytree(ROOT / 'pairing', root / 'pairing', ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'data').mkdir()
    shutil.copy(ROOT / 'data' / 'pools.json', root / 'data' / 'pools.json')
    bench = {sp: [dict(src='sums', q=f'What is {k} plus {k}?', choices=[str(2 * k + j - k % 4) for j in range(4)],
                       answer=k % 4) for k in range(start, start + 12)]
             for sp, start in (('train', 0), ('val', 100), ('test', 200))}
    (root / 'data' / 'bench.json').write_text(json.dumps(bench))
    return root
