"""The breadcrumb games (swarm/crumb_games.py): every game's episodes keep the interface crumbs.py reads, every reader
segment has crumbs to read at its place (in oldgrowth the asker may visit a room no walker did), the right say scores 1
and a wrong one 0, every reader's closed answer has a forced-choice group (choices: the gold say first, then the gold
say with only that answer swapped, each scoring lower), test episodes use only held-out values, oldgrowth's fixed test
sets for 1 to 32 walkers hold together, and every agent's stream (chat turns, m = 4 crumb slots) stays short with the
Qwen3-0.6B tokenizer (skipped where it is not in the local cache, like HotpotQA's validation split).

Usage: python -m pytest -q tests/test_crumb_games.py   (from the repository root)
       python tests/test_crumb_games.py                 prints one episode of every game as the model reads it, the
                                                        longest streams, and what a reader scores with crumbs cut
"""
from collections import Counter
import os
from pathlib import Path
import random
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as C                      # noqa: E402
from swarm import games as G_                           # noqa: E402

M = 4
LIMIT = 400                                             # crumbs.py keeps streams under about 400 tokens
CRUMB = '<|vision_pad|>'                                 # stands for one crumb slot when counting tokens


def hotpot_cached():
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(C.HOTPOT, C.HOTPOT_FILES[1][0], repo_type='dataset', local_files_only=True)
        return True
    except Exception:
        return False


def games_here():
    return [g for g in C.GAMES if g != 'hotpot' or hotpot_cached()]


def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    except Exception:
        return None


def stream(ep, a, text_arm=False, m=M):
    """Agent a's stream as crumbs.py lays it out: system, then per segment the text arm's notes, the user turn, the
    assistant turn when it says something, and m crumb slots when it writes at a place."""
    agent = ep['agents'][a]
    out = f'<|im_start|>system\n{agent["system"]}<|im_end|>\n'
    for sg in agent['segments']:
        notes = ''
        if text_arm and sg['place'] is not None:
            notes = ''.join(f'A note left here: {w["user"]}' + (f' {w["say"]}' if w['say'] else '') + '\n'
                            for b in ep['agents'][:a] if b['write'] for w in b['segments'] if w['place'] == sg['place'])
        out += f'<|im_start|>user\n{notes}{sg["user"]}<|im_end|>\n'
        if sg['say'] is not None:
            out += f'<|im_start|>assistant\n<think>\n\n</think>\n\n{sg["say"]}<|im_end|>\n'
        if agent['write'] and sg['place'] is not None:
            out += CRUMB * m
    return out


@pytest.mark.parametrize('game', games_here())
@pytest.mark.parametrize('test', (False, True))
def test_interface(game, test):
    rng = random.Random(1)
    for _ in range(20):
        ep = C.episode(game, rng, test)
        assert ep['game'] == game and 2 <= len(ep['agents']) <= 5
        left = set()
        said = 0
        for a, agent in enumerate(ep['agents']):
            assert set(agent) >= {'name', 'system', 'write', 'segments'}
            for s, sg in enumerate(agent['segments']):
                assert set(sg) == {'place', 'user', 'say', 'values'}
                assert sg['place'] is None or 0 <= sg['place'] < len(ep['places'])
                if sg['say'] is not None:
                    said += 1
                    assert sg['values'] and C.score(ep, a, s, sg['say']) == 1.0
                    assert C.score(ep, a, s, '') == 0.0
                if not agent['write'] and sg['place'] is not None and game != 'oldgrowth':
                    assert sg['place'] in left, 'a reader visits a place where no one left crumbs'
            if agent['write']:
                left |= {sg['place'] for sg in agent['segments'] if sg['place'] is not None}
        assert said and not ep['agents'][-1]['write']


@pytest.mark.parametrize('game', ['relay', 'maze', 'trap'])
def test_wrong_say(game):
    """A say that names another room, door or bit scores 0."""
    rng = random.Random(2)
    ep = C.episode(game, rng, True)
    a = len(ep['agents']) - 1
    s = next(i for i, sg in enumerate(ep['agents'][a]['segments']) if sg['say'])
    right = ep['agents'][a]['segments'][s]['values'][0]
    vocab = dict(relay=ep['places'], maze=C.COLOURS, trap=('safe', 'trap'))[game]
    wrong = next(v for v in vocab if v != right)
    say = re.sub(right, wrong, ep['agents'][a]['segments'][s]['say'], flags=re.I)
    assert say != ep['agents'][a]['segments'][s]['say'] and C.score(ep, a, s, say) == 0.0


def middle(a, b):
    """The parts of a and b left after their longest common prefix and suffix."""
    p = len(os.path.commonprefix([a, b]))
    q = len(os.path.commonprefix([a[p:][::-1], b[p:][::-1]]))
    return a[p:len(a) - q], b[p:len(b) - q]


@pytest.mark.parametrize('game', games_here())
@pytest.mark.parametrize('test', (False, True))
def test_choices(game, test):
    """Every reader say outside HotpotQA has at least one group; writers' says and HotpotQA have none. In a group the
    first say is the gold say, all says differ, each other say differs from it in one word only and scores lower (0
    in the one-answer games)."""
    rng = random.Random(8)
    for _ in range(20):
        ep = C.episode(game, rng, test)
        for a, agent in enumerate(ep['agents']):
            for s, sg in enumerate(agent['segments']):
                groups = C.choices(ep, a, s)
                if game == 'hotpot' or agent['write'] or sg['say'] is None:
                    assert groups == []
                    continue
                assert groups
                for group in groups:
                    assert group[0] == sg['say'] and len(group) >= 2 and len(set(group)) == len(group)
                    for wrong in group[1:]:
                        assert not set(' \n') & set(''.join(middle(group[0], wrong)))
                        got = C.score(ep, a, s, wrong)
                        assert got < 1.0 and (game in ('trail', 'oldgrowth') or got == 0.0)


def test_oldgrowth():
    """G walkers then the asker; training draws G from 1 to 4; the fixed test sets repeat exactly; seen and visits
    match the walkers' segments; a G = 32 episode keeps every stream short."""
    rng = random.Random(9)
    assert {C.episode('oldgrowth', rng)['oldgrowth']['G'] for _ in range(200)} == {1, 2, 3, 4}
    for G in C.OG_SIZES:
        eps = C.oldgrowth_eval(G, n=30)
        assert eps == C.oldgrowth_eval(G, n=30)
        for ep in eps:
            o, walkers, asker = ep['oldgrowth'], ep['agents'][:-1], ep['agents'][-1]
            assert o['G'] == G == len(walkers) and all(w['write'] for w in walkers) and not asker['write']
            assert all(p in C.ROOM_POOLS[1] for p in ep['places'])
            assert all(x in G_.WORD_POOLS[1] for x in o['objects'])
            for r in range(C.OG_ROOMS):
                visits = [sg for w in walkers for sg in w['segments'] if sg['place'] == r]
                assert o['visits'][r] == len(visits)
                assert o['seen'][r] == any(f'You see a {o["objects"][r]}.' in sg['user'] for sg in visits)
            assert [sg['place'] for sg in asker['segments'][:-1]] == o['asked']
            parts = C.oldgrowth_parts(ep, asker['segments'][-1]['say'])
            assert [p['right'] for p in parts] == [True] * C.OG_ASK
    tok = tokenizer()
    if tok is not None:
        ep = C.episode('oldgrowth', random.Random(10), True, G=32)
        assert len(ep['agents']) == 33
        assert max(len(tok(stream(ep, a), add_special_tokens=False).input_ids) for a in range(33)) <= LIMIT


def test_held_out():
    """Test episodes use only held-out rooms, colours, numbers and additions; training episodes never do."""
    rng = random.Random(3)
    for test in (False, True):
        for _ in range(50):
            for game in ('relay', 'trap', 'maze'):
                ep = C.episode(game, rng, test)
                assert all((p in C.ROOM_POOLS[1]) == test for p in ep['places'])
            ep = C.episode('maze', rng, test)
            assert all((c in C.COLOUR_POOLS[1]) == test for d in ep['maze']['doors'] for c in d)
            ep = C.episode('count', rng, test)
            assert (tuple(sorted(ep['count']['numbers'])) in C.THREES[1]) == test
            ep = C.episode('work', rng, test)
            assert all((tuple(t) in C.SUMS[1]) == test for t in ep['work']['tasks'])
    assert not set(C.THREES[0]) & set(C.THREES[1]) and not set(C.SUMS[0]) & set(C.SUMS[1])


@pytest.mark.parametrize('game', games_here())
def test_short_streams(game):
    tok = tokenizer()
    if tok is None:
        pytest.skip('Qwen3-0.6B tokenizer not in the local cache')
    rng = random.Random(4)
    longest = 0
    for _ in range(30):
        ep = C.episode(game, rng, True)
        for a in range(len(ep['agents'])):
            longest = max(longest, len(tok(stream(ep, a), add_special_tokens=False).input_ids))
    assert longest <= LIMIT, longest


def chance(game, n=4000):
    """What the reader's says score with no crumbs at test values, for games where that is a guess: the best constant
    say (count, work, trap), one door of three (maze), or for HotpotQA the share of validation answers a reader can
    guess from the question and the two titles (yes or no, or one of the titles) halved."""
    rng = random.Random(5)
    eps = [C.episode(game, rng, True) for _ in range(n if game != 'hotpot' else 1)]
    if game == 'maze':
        return 1 / 3
    if game in ('relay', 'oldgrowth'):
        return 0.0
    if game == 'hotpot':
        rows = C.hotpot_rows(True)
        easy = sum(r['answer'].lower() in ('yes', 'no') or r['answer'] in r['titles'] for r in rows)
        return 0.5 * easy / len(rows)
    out = {}
    for ep in eps:
        agent = ep['agents'][-1]
        for s, sg in enumerate(agent['segments']):
            if sg['say'] is not None:
                kind = 'total' if sg['place'] is None or game == 'count' else 'each'
                out.setdefault(kind, Counter())[sg['values'][0]] += 1
    return {k: round(max(c.values()) / sum(c.values()), 3) for k, c in out.items()}


def main():
    tok = tokenizer()
    for game in C.GAMES:
        ep = C.episode(game, random.Random(6), True)
        print(f'===== {game}: places {ep["places"]}')
        for a, agent in enumerate(ep['agents']):
            text = stream(ep, a)
            n = len(tok(text, add_special_tokens=False).input_ids) if tok else None
            nt = len(tok(stream(ep, a, True), add_special_tokens=False).input_ids) if tok else None
            print(f'--- {agent["name"]} (write={agent["write"]}, {n} tokens, text arm {nt})')
            print(text.replace(CRUMB * M, f' [{M} crumbs]'))
        rng = random.Random(7)
        longest = max(len(tok(stream(e, a, t), add_special_tokens=False).input_ids)
                      for e in [C.episode(game, rng, True) for _ in range(40)]
                      for a in range(len(e['agents'])) for t in (False, True)) if tok else None
        print(f'longest stream of 40 test episodes (text arm included): {longest} tokens; '
              f'reader with crumbs cut: {chance(game)}')
    if tok:
        for G in C.OG_SIZES:
            eps = C.oldgrowth_eval(G, n=10)
            n = [max(len(tok(stream(e, a, t), add_special_tokens=False).input_ids) for e in eps
                     for a in range(len(e['agents']))) for t in (False, True)]
            ceiling = sum(e['oldgrowth']['seen'][r] for e in C.oldgrowth_eval(G) for r in e['oldgrowth']['asked'])
            print(f'oldgrowth G={G}: longest stream {n[0]} tokens, text arm {n[1]}; '
                  f'ceiling (asked rooms some walker saw) {ceiling / (300 * C.OG_ASK):.3f}')


if __name__ == '__main__':
    main()
