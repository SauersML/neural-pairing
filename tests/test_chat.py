"""Plain chats (chat.json and chat2.json, prep_talk.py --chat; train.py --chat and --chat2): only the original model
writing a turn sees the partner's city, prep_talk's turn scores match turn-by-turn scoring, the payload is each twin's
first turn when the partner has a city, payload_ctl measures co_chat, and tiny runs with chats train.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest
import torch

import chat2
import coread as C
import link
import payload_ctl
import session as S
from link import SIDES, partner

ROOT = Path(__file__).resolve().parents[1]


def chats():
    out = []
    for i, private in enumerate(({'quill': 'Lisbon', 'ember': None}, {'quill': 'Oslo', 'ember': 'Lima'},
                                 {'quill': None, 'ember': None})):
        opener = SIDES[i % 2]
        w = {x: C.CHAT[0] if private[x] else C.CHAT[3] for x in SIDES}
        turns = [(opener if k % 2 == 0 else partner(opener), f'turn {k} of chat {i}, about the weather.') for k in range(4)]
        out.append(dict(game='chat', opener=opener, private=private, turns=turns,
                        system={x: C.chat_system(x, private[x], w[x]) for x in SIDES}))
    return out


@pytest.fixture
def chat_data(data, tmp_path):
    """The co-reading corpus with chat.json beside it; the path of its coread.json."""
    shutil.copy(data / 'coread.json', tmp_path / 'coread.json')
    (tmp_path / 'chat.json').write_text(json.dumps({'train': chats(), 'test': chats()}))
    return tmp_path / 'coread.json'


@pytest.fixture
def chat2_data(chat_data):
    """chat_data with chat2.json beside it."""
    data = {}
    for split in ('train', 'test'):
        data[split] = []
        rng = random.Random(4)
        for _ in range(12):
            c = chat2.conversation(split, rng)
            c['turns'] = [(x, 'How has your morning been?') for x in (c['opener'], partner(c['opener'])) * 3]
            data[split].append(c)
    (chat_data.parent / 'chat2.json').write_text(json.dumps(data))
    return chat_data


def train(data, out, *args, **env):
    cmd = [sys.executable, str(ROOT / 'pairing' / 'train.py'), '--replies', str(data / 'replies.json'), '--out', str(out),
           '--anchor', '0', '--comp', '0', '--first_eval', '1000', '--save_every', '1000', '--heads', '2',
           '--head_dim', '16', '--xblocks', '1,2,4', '--ckpt', 'none', '--quick_n', '1', *args]
    return subprocess.run(cmd, cwd=ROOT / 'pairing', env=dict(os.environ, MPD_MEM_GIB='8', **env), capture_output=True,
                          text=True, stdin=subprocess.DEVNULL)


def test_teacher_prompt():
    taught = C.chat_system('quill', 'Lisbon', C.CHAT[0], knows='Lima')
    plain = C.chat_system('quill', 'Lisbon', C.CHAT[0])
    assert 'Lima' in taught and 'Lima' not in plain and 'Lisbon' in plain
    assert taught.startswith(plain[:plain.index(C.CHAT_TONE)].rstrip())


def test_episodes(replies, chat_data):
    co = C.Coread(replies, 'test', seed=0, path=str(chat_data))
    assert len(co.chat) == 3
    seen = set()
    for _ in range(60):
        ep = co.chat_episode()
        pay = S.payload(ep)
        want = {x for x in SIDES if ep['meta']['private'][partner(x)]}
        assert set(pay) == want, (pay, ep['meta']['private'])
        for x, spans in pay.items():
            first = next(t for sp, t, *_ in ep['turns'] if sp == x)
            assert spans == [(0, len(first))]
        assert all(ep['meta']['private'][x] is None or ep['meta']['private'][x] in ep['system'][x] for x in SIDES)
        assert not any(ep['meta']['private'][partner(x)] and ep['meta']['private'][partner(x)] in ep['system'][x]
                       for x in SIDES)
        seen.add(frozenset(want))
    assert len(seen) == 3
    eps = payload_ctl.episodes(replies, 4, coread=True, path=str(chat_data))
    assert len(eps['co_chat']) == 4 and all(ep['game'] == 'co_chat' for ep in eps['co_chat'])


def prep_talk_functions(bs):
    """turn_nats and context from prep_talk.py (a script), with what they read."""
    src = (ROOT / 'pairing' / 'prep_talk.py').read_text()
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ('turn_nats', 'context')]
    env = dict(S=S, torch=torch, DEVICE='cpu', args=argparse.Namespace(bs=bs))
    exec(compile(ast.Module(fns, []), 'prep_talk.py', 'exec'), env)
    return env['turn_nats'], env['context']


def test_turn_nats():
    """turn_nats scores each of x's turns as generation laid it out: equal to scoring turn k after context() of the
    turns before it."""
    turn_nats, context = prep_talk_functions(bs=16)
    turns = [('quill', 'Hello there, how is it going?'), ('ember', 'Fine, thanks. And you?'), ('quill', 'Good.'),
             ('ember', 'Nice to hear.')]
    for x in SIDES:
        tok, m = link.load(x)
        m = m.float().eval()
        end = tok.convert_tokens_to_ids('<|im_end|>')
        systems = [C.chat_system(x, 'Lisbon', C.CHAT[0]), C.chat_system(x, None, C.CHAT[3])]
        got = turn_nats(tok, m, [(sy, turns, x) for sy in systems])
        for sy, row in zip(systems, got):
            want = []
            for k, (sp, text) in enumerate(turns):
                if sp != x:
                    continue
                ctx = context(tok, x, sy, turns[:k])
                tgt = S.ids(tok, text) + [end]
                with torch.no_grad():
                    lp = torch.log_softmax(m(input_ids=torch.tensor([ctx + tgt])).logits[0, :-1].float(), -1)
                want.append(float(sum(lp[len(ctx) - 1 + j, t] for j, t in enumerate(tgt))))
            assert len(row) == len(want) == 2 and all(abs(a - b) < 1e-3 * max(1, abs(b)) for a, b in zip(row, want)), (row, want)


@pytest.mark.slow
def test_tiny_run(data, chat_data, tmp_path):
    r = train(data, tmp_path / 'out', '--steps', '3', '--bs', '4', '--coread', '0.3', '--coread_data', str(chat_data),
              '--chat', '0.6')
    assert r.returncode == 0, '\n'.join(l for l in r.stderr.splitlines() if 'Warning' not in l)[-2500:]


def test_chat2_table_splits_and_prompts():
    counts = chat2.validate_catalog()
    assert set(counts) == set(chat2.TRAIN_KINDS + chat2.HELD_OUT_KINDS)
    assert all(c['train'] == 0 for k, c in counts.items() if k in chat2.HELD_OUT_KINDS)
    for split in ('train', 'test'):
        rng = random.Random(74)
        seen = set()
        for _ in range(800):
            c = chat2.conversation(split, rng)
            for x in SIDES:
                p = c['private'][x]
                if p:
                    seen.add(p['kind'])
                    assert p in chat2.pool(p['kind'], split)
                    assert p['owner'] in c['system'][x]
                other = c['private'][partner(x)]
                if other:
                    assert other['teacher'].format(P=C.MODELS[partner(x)]['name']) in c['teach'][x]
                assert ('There is no game and nothing you have to do:' in c['system'][x]) == (split == 'test')
        assert seen == set(chat2.TRAIN_KINDS if split == 'train' else counts)


def test_chat2_episodes(replies, chat2_data):
    co = C.Coread(replies, 'train', seed=0, path=str(chat2_data))
    for _ in range(24):
        ep = co.chat_episode('co_chat2')
        assert ep['game'] == 'co_chat2' and 'teach' not in ep['meta']
        assert set(S.payload(ep)) == {x for x in SIDES if ep['meta']['private'][partner(x)]}
    assert 'co_chat2' in payload_ctl.episodes(replies, 4, coread=True, path=str(chat2_data))


@pytest.mark.slow
def test_chat2_tiny_run(data, chat2_data, tmp_path):
    out = tmp_path / 'out'
    r = train(data, out, '--steps', '2', '--bs', '2', '--coread', '0', '--coread_data', str(chat2_data), '--chat2', '1',
              NL_QUICK='relay_word', NL_MAXNEW='2')
    assert r.returncode == 0, r.stderr[-3000:]
    assert 'co_chat2' in r.stdout and (out / 'step_2' / 'bridge.pt').is_file()
