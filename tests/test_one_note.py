"""swarm/one_note.py and crumbs.py's use of it on a tiny random Qwen3 sym net on the CPU: every note of every game gets a
question with the true value among its candidates; the penalty's stream is the text prompt with the note tokens in place;
the penalty charges those prompts and its gradient reaches the writers; the dev's read scores own and swapped notes; the
dev charges only the games read above noise and counts the leak against the best weights."""
import collections
import itertools
import json
import math
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402
from swarm import eavesdrop as E                        # noqa: E402
from swarm import one_note as ON                        # noqa: E402

from test_crumbs import lora_effect, tiny               # noqa: E402

GAMES = ('count', 'work', 'trap', 'trail', 'keytrail', 'relay', 'maze', 'hotpot')


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def episodes(game, n=3, test=True, seed=0):
    rng = random.Random(seed)
    return [CG.episode(game, rng, test) for _ in range(n)]


def test_questions_are_about_what_each_writer_saw():
    for g in GAMES:
        for ep in episodes(g):
            got = ON.asks(ep)
            placed = [(w, k) for w, ag in enumerate(ep['agents']) if ag['write']
                      for k in range(sum(s['place'] is not None for s in ag['segments']))]
            assert [(a['w'], a['k']) for a in got] == placed
            for a in got:
                assert a['gold'] not in a['alts'] and len(set(a['alts'])) == len(a['alts']) and a['alts']
                assert a['system'] == next(x['system'] for x in ep['agents'] if not x['write'])
                saw = [s for s in ep['agents'][a['w']]['segments'] if s['place'] is not None][a['k']]['user']
                if g == 'count':
                    assert f'there are {a["gold"]}.' in saw and len(a['alts']) == 8
                elif g == 'work':
                    x, y = saw.split('The task: ')[1].rstrip('.').split(' + ')
                    assert int(a['gold']) == int(x) + int(y) and len(a['alts']) == 5
                elif g == 'trap':
                    assert ('trap' in saw) == (a['gold'] == 'Trap') and len(a['alts']) == 1
                elif g in ('trail', 'keytrail'):
                    assert f'You see a {a["gold"]}.' in saw and len(a['alts']) == 5
                    assert all(x in a['question'] for x in a['alts'] + [a['gold']])
                elif g == 'relay':
                    assert a['gold'] == ep['places'][ep['relay']['key']] and len(a['alts']) == 5
                elif g == 'maze':
                    assert f'You take the {a["gold"]} door' in saw and len(a['alts']) == 2
                    assert all(x in a['question'] for x in a['alts'] + [a['gold']])
                elif g == 'hotpot':
                    assert ep['hotpot']['question'] in a['question'] and a['alts']
                    assert a['gold'] == [s['say'] for s in ep['agents'][-1]['segments'] if s['say']][-1]
    assert ON.chance(episodes('count')) == round(1 / 9, 4) and ON.chance(episodes('trap')) == .5


def test_targets_part_where_the_candidates_part():
    assert ON.targets(10, [5, 9], [[5, 8], [5, 7], [6, 9], [5, 9, 1]]) == [(9, 5, [6]), (10, 9, [7, 8])]


@pytest.mark.parametrize('game', ['count', 'work', 'maze', 'hotpot'])
def test_stream_is_the_text_prompt(tok, game):
    net = tiny(tok, 'sym', 2)
    ask = ON.asks(episodes(game)[0])[-1]
    gold = net._ids(ask['gold'])
    st, cut = net._one_stream(ask, gold)
    ids = st['ids']
    assert ids[cut:] == gold and len(ids) == len(st['place']) and not any(st['target'])
    first, last = st['reads'][0][0], st['reads'][-1][0]
    assert [r[1:] for r in st['reads']] == [(ask['w'], ask['k'] * 2), (ask['w'], ask['k'] * 2 + 1)] and last == first + 1
    note = 'xyz'
    assert tok.decode(ids[:first]) + note + tok.decode(ids[last + 1:cut]) == ON.prompt(tok, ask, note)


@pytest.mark.parametrize('game', ['count', 'work', 'hotpot'])
def test_penalty_charges_the_read_and_reaches_the_writers(tok, game, monkeypatch):
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.train()
    net.adv, net.one_note = 1., True

    def whole_game(*args):
        raise AssertionError('the whole-game read is not charged')

    monkeypatch.setattr(type(net), '_adversary', whole_game)
    net.sums = collections.defaultdict(float)
    net.run(episodes(game, 3, False), 'cpu')
    assert net.sums['n_adv'] > 0 and torch.isfinite(net.sums['adv'])
    grads = torch.autograd.grad(net.sums['adv'], [p for p in net.model.parameters() if p.requires_grad], allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_penalty_charges_what_the_note_gives_over_the_prior(tok, monkeypatch):
    """Under chance and still charged: a note that lifts the gold from -3 to -1 among two candidates; a note that takes
    it down, or moves nothing, is not."""
    net = tiny(tok, 'sym', 2)
    net.train()
    net.adv, net.one_note = 1., True
    for on, off, charged in ((-1., -3., 2.), (-3., -1., 0.), (-2., -2., 0.)):
        def fake(rows_t, x, a, bank, aims, read, odds=False):
            assert odds
            return torch.full((len(aims),), on if read else off), torch.full((len(aims),), 2.)

        monkeypatch.setattr(net, '_gold_logp', fake)
        net.sums = collections.defaultdict(float)
        net.run(episodes('trap', 2, False), 'cpu')
        assert net.sums['n_adv'] > 0 and float(net.sums['adv']) == pytest.approx(charged * net.sums['n_adv'])


def test_penalty_reads_notes_warmed_token_by_token(tok, monkeypatch):
    """However sure the writer, the first choice stands ONE_SURE nats over the second, and a token the writer is unsure of
    stays as the run has it. The warm notes line up with the bank's, and the gradient reaches the writers through them."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.tau = .5
    logits = torch.tensor([[900., 0., -5.], [1., 0., -5.], [50., 50., float('-inf')]])
    p = net._warm(logits)
    assert float((p[0, 0] / p[0, 1]).log()) == pytest.approx(C.ONE_SURE, abs=1e-3)
    assert torch.allclose(p[1], (logits[1] / net.tau).softmax(-1))
    assert p[2].tolist() == [.5, .5, 0.]
    net.train()
    net.adv, net.one_note = 1., True
    params = [q for q in net.model.parameters() if q.requires_grad]
    eps = episodes('count', 3, False)
    monkeypatch.setattr(C, 'ONE_SURE', 1e9)
    net.sums = collections.defaultdict(float)
    bank = net.run(eps, 'cpu')[2]
    assert bank['warm'][0].shape == bank['e'][0].shape and torch.allclose(bank['warm'][0], bank['e'][0])
    monkeypatch.setattr(C, 'ONE_SURE', .01)
    net.sums = collections.defaultdict(float)
    bank = net.run(eps, 'cpu')[2]
    hard = net.model.get_input_embeddings().weight[bank['ids'][0]]
    assert (bank['warm'][0] - hard).norm() > (bank['e'][0] - hard).norm()
    grads = torch.autograd.grad(net.sums['adv'], params, allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_log_odds_keep_a_slope_when_the_reader_is_sure(tok):
    net = tiny(tok, 'sym', 2)
    W = net.model.get_output_embeddings().weight.detach()
    h = (W[5] - W[6]).detach()[None, None].repeat(1, 1, 1) * 1e4
    h = h.requires_grad_()
    targets = [(0, (0, 5, [6]))]
    slope = [float(torch.autograd.grad(net._among(h, targets, odds)[0].sum(), h)[0].abs().sum()) for odds in (False, True)]
    assert slope[0] < 1e-6 < slope[1]
    p, odds = net._among(h, targets)[0], net._among(h, targets, True)[0]
    assert float(p) == 0. and float(odds) > 10


def test_penalty_is_off_with_no_weight(tok):
    net = tiny(tok, 'sym', 2)
    net.train()
    net.adv, net.one_note = 0., True
    net.sums = collections.defaultdict(float)
    assert 'warm' not in net.run(episodes('count', 2, False), 'cpu')[2]
    assert net.sums['n_adv'] == 0


def test_dev_read(tok, monkeypatch):
    """Own notes that say the value and a reader that reads the digit: right on own notes, chance on swapped, up to the
    cap on notes; the adapter is off for the read and the training mode comes back."""
    net = tiny(tok, 'sym', 2)
    net.train()
    states = []

    def says(net_, bank, b, ep):
        return {(w, 0): str(n) for w, n in enumerate(ep['count']['numbers'])}

    def reads(model, tk, p, cands, device):
        states.append(any(getattr(m, 'disable_adapters', False) for m in model.modules()))
        line = p.rsplit(C.NOTE, 1)[-1]
        return torch.tensor([0. if c == line[0] else -1. for c in cands])

    monkeypatch.setattr(ON, 'written', says)
    monkeypatch.setattr(E, 'cand_logprobs', reads)
    eps = episodes('count', 8)
    got = C.one_note_by_game(net, 'cpu', dict(count=eps), most=12)['count']
    assert got['own'] == 1. and got['swap'] < .75 and got['notes'] == 12 and got['chance'] == round(1 / 9, 4)
    assert got['ci'][0] > 0 and net.training and all(states) and len(states) == 24
    assert got['moved'] > .5 and got['moved_ci'][0] > 0


def test_prior_cannot_hide_a_note(tok, monkeypatch):
    """A model that says Safe whatever the note says, while the note moves its answer: the shares right are level, and
    moved still sees it; notes that tell it nothing move nothing."""
    net = tiny(tok, 'sym', 2)
    eps = episodes('trap', 12)
    shown = {}

    def says(net_, bank, b, ep):
        return {(w, k): ('T' if ep['trap']['traps'][s['place']] else 'S') for w, ag in enumerate(ep['agents']) if ag['write']
                for k, s in enumerate(ag['segments'])}

    def reads(model, tk, p, cands, device):
        note = p.rsplit(C.NOTE, 1)[-1][0]
        return torch.tensor([(5. if c == 'Safe' else 0.) + shown['on'] * (c[0] == note) for c in cands])

    monkeypatch.setattr(ON, 'written', says)
    monkeypatch.setattr(E, 'cand_logprobs', reads)
    shown['on'] = 1.
    got = C.one_note_by_game(net, 'cpu', dict(trap=eps))['trap']
    assert got['own'] == got['swap'] and got['ci'] == [0., 0.]
    assert got['moved'] > .2 and got['moved_ci'][0] > 0 and got['nats'] > 0
    shown['on'] = 0.
    got = C.one_note_by_game(net, 'cpu', dict(trap=eps))['trap']
    assert got['moved'] == 0. and C.leak(got) == 0.
    big = dict(nats=30., chance=1 / 9)
    assert C.leak(dict(own=.5, swap=.1, moved=.2, moved_ci=[.1, .3], **big)) == pytest.approx(.4)
    assert C.leak(dict(own=.5, swap=.5, moved=.3, moved_ci=[.1, .5], **big)) == .3
    assert C.leak(dict(own=.5, swap=.5, moved=-.3, moved_ci=[-.5, -.1], **big)) == 0.  # steered off the truth: no read
    assert C.leak(dict(own=.5, swap=.5, moved=.3, moved_ci=[-.1, .5], **big)) == 0.    # noise
    weak = C.leak(dict(own=.5, swap=.5, moved=.6, moved_ci=[.4, .8], nats=.1, chance=1 / 9))
    assert weak == pytest.approx(.6 * .1 / math.log(9))                                # steady, and too weak to change an answer
    assert C.leak(None) == 0.


def test_train_charges_the_games_read_above_noise(tok, tmp_path, monkeypatch):
    """A dev's one-note read: hot games only get the penalty, one dev above noise is enough, and the best weights count
    the leak: the same follower numbers score lower while the base model can read the note."""
    real_dev = C.sym_dev
    plan = itertools.chain([.8, .8], itertools.repeat(0.))   # count's leak at each dev

    def planned_dev(net, device, dev):
        return dict(real_dev(net, device, dev), distinct=3, letters_max=0., letters_text=0.,
                    letters_by_game=dict(count=0., trap=0.), dev_swap=1e3)

    def planned_right(net, device, by_game, readers=('follower', 'base'), bs=48):
        return {g: dict(follower_own=.9, follower_swap=.1, chance=.2, base_own=0., base_swap=0., base_ci=[.5, .6])
                for g in by_game}

    def planned_one(net, device, by_game, bs=48, most=None):
        v = next(plan)
        assert set(by_game) == {'count', 'trap'}
        return dict(count=dict(own=v + .1, swap=.1, ci=[v - .1, v + .1], moved=v, moved_ci=[v - .1, v + .1], nats=9., chance=.11, notes=9),
                    trap=dict(own=.5, swap=.5, ci=[-.1, .1], moved=0., moved_ci=[-.1, .1], nats=0., chance=.5, notes=9))

    weights = []
    monkeypatch.setattr(C, 'sym_dev', planned_dev)
    monkeypatch.setattr(C, 'right_by_game', planned_right)
    monkeypatch.setattr(C, 'one_note_by_game', planned_one)
    monkeypatch.setattr(C, 'RIGHT_EPS', 2)
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.one_note = True
    net.train()
    real = type(net).loss

    def loss(self, eps, device):
        weights.append((eps[0]['game'], self.adv))
        return real(self, eps, device)

    monkeypatch.setattr(type(net), 'loss', loss)
    out = tmp_path / 'r'
    out.mkdir()
    C.train(net, 'cpu', 8, 2, 0, games=('count', 'trap'), out=out, every=1, letters_from=0, lam0=2., gap0=1e-9, cut0=1e6,
            tau0=.3, beta_full=True, min_steps=100, gate_letters=.8, gate_distinct=2, gap_ref='max', gap_by='cut',
            protect=True, frozen_adv=.5)
    recs = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    devs = [r for r in recs if 'dev_nll' in r]
    assert devs[0]['base_hot'] == ['count'] and devs[0]['frozen_adv'] == .5
    assert devs[0]['one_note']['count']['own'] == pytest.approx(.9)
    assert devs[0]['score'] == pytest.approx(.4) and devs[2]['score'] == pytest.approx(.8)     # (0.8 - 0.8 + 0.8) / 2, 0.8
    assert 'base_hot' not in devs[2] and devs[2]['best']
    assert not any(a for g, a in weights if g == 'trap')
    assert any(a for g, a in weights if g == 'count')
    assert devs[0]['weights']['count'] == pytest.approx(.8) and devs[0]['weights']['trap'] == pytest.approx(.5)   # the leak, the least
    assert devs[0]['adv_games'] == dict(count=.5) and devs[1]['adv_games'] == dict(count=.75)    # grows while it stays read
    assert devs[2]['adv_games'] == dict(count=.75)                                               # holds once it does not
    assert devs[2]['weights']['count'] == pytest.approx(.5)


def test_no_game_may_collapse(tok, tmp_path, monkeypatch):
    """A dev with a game's follower under half its peak is not the best, however the score looks; under a quarter goes back
    to the best weights. A game that never got anywhere cannot fall."""
    real_dev = C.sym_dev
    trap = iter([.8, .3, .1, .8])                      # the trap follower's lead over swapped notes at each dev
    level = iter([.1, .9, .9, .95])                    # count's, which lifts the score

    def planned_dev(net, device, dev):
        return dict(real_dev(net, device, dev), distinct=3, letters_max=0., letters_text=0.,
                    letters_by_game=dict(count=0., trap=0., relay=0.), dev_swap=1e3)

    def planned_right(net, device, by_game, readers=('follower', 'base'), bs=48):
        if 'base' not in readers:
            return {g: dict(follower_own=0., follower_swap=0., chance=.2) for g in by_game}
        return dict(trap=dict(follower_own=next(trap), follower_swap=0., chance=.5, base_own=0., base_swap=0., base_ci=[0., 0.]),
                    count=dict(follower_own=next(level), follower_swap=0., chance=.2, base_own=0., base_swap=0., base_ci=[0., 0.]),
                    relay=dict(follower_own=.1, follower_swap=.1, chance=.2, base_own=0., base_swap=0., base_ci=[0., 0.]))

    def planned_one(net, device, by_game, bs=48, most=None):
        return dict(trap=dict(own=.5, swap=.5, ci=[0., 0.], moved=.3, moved_ci=[.2, .4], nats=9., chance=.5, notes=9))

    monkeypatch.setattr(C, 'sym_dev', planned_dev)
    monkeypatch.setattr(C, 'right_by_game', planned_right)
    monkeypatch.setattr(C, 'one_note_by_game', planned_one)
    monkeypatch.setattr(C, 'RIGHT_EPS', 2)
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.one_note = True
    net.train()
    out = tmp_path / 'r'
    out.mkdir()
    C.train(net, 'cpu', 4, 2, 0, games=('count', 'trap'), out=out, every=1, letters_from=0, lam0=2., gap0=1e-9, cut0=1e6,
            tau0=.3, beta_full=True, min_steps=100, gate_letters=.8, gate_distinct=2, gap_ref='max', gap_by='cut',
            protect=True, frozen_adv=.5)
    devs = [r for r in map(json.loads, (out / 'log.jsonl').read_text().splitlines()) if 'dev_nll' in r]
    assert devs[0]['best'] and devs[0]['fell'] == []
    assert devs[1]['fell'] == ['trap'] and 'best' not in devs[1] and 'back_to' not in devs[1]     # .3 of a peak of .8
    assert devs[2]['fell'] == ['trap'] and devs[2]['back_to'] == 1 and devs[2]['lr_scale'] == .5  # .1 of it
    assert devs[3]['fell'] == [] and devs[3]['best']
    assert [d['adv_games']['trap'] for d in devs] == [.5, .5, .25, .375]   # holds, halves when it went back, grows again
