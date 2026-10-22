"""train.py --hold_out: the predicate flags exactly the held-out games among real episodes (by name, co_play kind and
co-reading task, distillation and talk included), and a tiny run with games held out trains.

NL_TINY=1 NL_DEVICE=cpu MPD_MEM_GIB=2 python tests/test_hold_out.py (each tiny run reserves 3 GiB)
"""
import ast
import collections
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pairing'))
import coread as C
import games as G

REPLIES = os.environ.get('HOLD_OUT_REPLIES', '/Users/user/mpd-data/runpod/magic-nl19/replies.json')
COREAD = os.path.join(ROOT, 'data', 'coread_check.json')


def predicate(hold):
    src = open(os.path.join(ROOT, 'pairing', 'train.py')).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'held_out')
    env = {'HOLD_OUT': set(hold)}
    exec(compile(ast.Module([fn], []), 'train.py', 'exec'), env)
    return env['held_out']


def test_predicate():
    hold = {'lookup', 'co_mcq', 'co_code_out', 'co_play:rhyme'}
    held_out = predicate(hold)
    replies = json.load(open(REPLIES))
    g = G.Games(replies, 'train', seed=3)
    co = C.Coread(replies, 'train', seed=3, path=COREAD)
    eps = [g.episode() for _ in range(400)] + [co.episode() for _ in range(3000)]
    seen = collections.Counter()
    for ep in eps:
        m = ep.get('meta') or {}
        task = C.TASK_GAME.get(m.get('task'))
        want = ep['game'] in hold or task in hold or (ep['game'] == 'co_play' and f"co_play:{m.get('play')}" in hold)
        assert held_out(ep) == want, (ep['game'], task, m.get('play'))
        if want:
            seen[ep['game'] if ep['game'] in hold else task if task in hold else 'co_play:' + m['play']] += 1
            if ep['game'] in ('co_distill', 'co_talk'):
                seen['via ' + ep['game']] += 1
    print('held out among', len(eps), 'drawn episodes:', dict(seen))
    assert seen['lookup'] and seen['co_mcq'] and seen['co_code_out']
    assert seen['co_play:rhyme'] or not any(c['game'] == 'rhyme' for c in co.play)


def test_tiny_run():
    env = dict(os.environ, MPD_MEM_GIB='3')
    with tempfile.TemporaryDirectory() as out:
        cmd = [sys.executable, os.path.join(ROOT, 'pairing', 'train.py'), '--replies', REPLIES, '--out', out, '--steps', '2',
               '--bs', '4', '--coread', '0.5', '--coread_data', COREAD, '--anchor', '0', '--comp', '0', '--first_eval', '1000',
               '--save_every', '1000', '--hold_out', 'lookup,merge_sum,co_mcq', '--relay_first', '1',
               '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4', '--ckpt', 'none', '--quick_n', '1', '--relay_games', 'relay_word,merge_fact']
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, 'pairing'), env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, '\n'.join(l for l in r.stderr.splitlines() if 'Warning' not in l)[-2500:]
        bad = subprocess.run(cmd[:-4] + ['--hold_out', 'lookup', '--relay_games', 'lookup'], cwd=os.path.join(ROOT, 'pairing'),
                             env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        assert bad.returncode != 0 and 'also in --relay_games' in bad.stderr, bad.stderr[-1000:]


if __name__ == '__main__':
    test_predicate()
    test_tiny_run()
    print('OK')
