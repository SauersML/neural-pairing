"""Plain chats (data/chat.json, prep_talk.py --chat; train.py --chat): only the original model writing a turn sees the
partner's city, the payload is each twin's first turn when the partner has a city, payload_ctl measures co_chat, and a
tiny run with --chat trains.

NL_TINY=1 NL_DEVICE=cpu MPD_MEM_GIB=2 python tests/test_chat.py (the tiny run reserves 3 GiB)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pairing'))
import coread as C
import payload_ctl
import session as S
from link import SIDES, partner

REPLIES = os.environ.get('HOLD_OUT_REPLIES', '/Users/user/mpd-data/runpod/magic-nl19/replies.json')
COREAD = os.path.join(ROOT, 'data', 'coread_check.json')


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


def data_dir(tmp):
    shutil.copy(COREAD, os.path.join(tmp, 'coread.json'))
    json.dump({'train': chats(), 'test': chats()}, open(os.path.join(tmp, 'chat.json'), 'w'))
    return os.path.join(tmp, 'coread.json')


def test_teacher_prompt():
    taught = C.chat_system('quill', 'Lisbon', C.CHAT[0], knows='Lima')
    plain = C.chat_system('quill', 'Lisbon', C.CHAT[0])
    assert 'Lima' in taught and 'Lima' not in plain and 'Lisbon' in plain
    assert taught.startswith(plain[:plain.index(C.CHAT_TONE)].rstrip())


def test_episodes():
    replies = json.load(open(REPLIES))
    with tempfile.TemporaryDirectory() as tmp:
        path = data_dir(tmp)
        co = C.Coread(replies, 'test', seed=0, path=path)
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
        eps = payload_ctl.episodes(replies, 4, coread=True, path=path)
        assert len(eps['co_chat']) == 4 and all(ep['game'] == 'co_chat' for ep in eps['co_chat'])


def test_tiny_run():
    env = dict(os.environ, MPD_MEM_GIB='3')
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        cmd = [sys.executable, os.path.join(ROOT, 'pairing', 'train.py'), '--replies', REPLIES, '--out', out, '--steps', '3',
               '--bs', '4', '--coread', '0.3', '--coread_data', data_dir(tmp), '--chat', '0.6', '--anchor', '0', '--comp', '0',
               '--first_eval', '1000', '--save_every', '1000', '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4',
               '--ckpt', 'none', '--quick_n', '1']
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, 'pairing'), env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, '\n'.join(l for l in r.stderr.splitlines() if 'Warning' not in l)[-2500:]


if __name__ == '__main__':
    test_teacher_prompt()
    test_episodes()
    test_tiny_run()
    print('OK')
