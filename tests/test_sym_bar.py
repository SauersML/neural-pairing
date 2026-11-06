"""swarm/sym_bar.py on a tiny random Qwen3 sym net on the CPU: a reader's prompt is the stream the follower reads, with
the note tokens as text; swapped notes are the next episode's; no notes means no note lines; the forced choice is the
highest log-prob among a group's whole says; the examples come before the test turn."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402
from swarm import eavesdrop as E                        # noqa: E402
from swarm import sym_bar as B                          # noqa: E402
from swarm import sym_eaves as SE                       # noqa: E402

from test_crumbs import tiny                            # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def notes_of(net, game, n=3, split='test'):
    chunks = SE.banks(net, 'cpu', SE.episodes(game, n, split))
    seven = net.tok('7', add_special_tokens=False).input_ids[0]
    out = []
    for eps, bank in chunks:                          # a note of digits per writer's segment, different per episode
        bank['ids'][0][:] = seven
        got = [B.written(net, bank, b, ep) for b, ep in enumerate(eps)]
        for b, g in enumerate(got):
            for key in g:
                g[key] = g[key][:-1] + str(b)
        out.append((eps, got))
    return out


@pytest.mark.parametrize('game', ['count', 'trail'])
def test_prompt_is_the_followers_stream(tok, game):
    net = tiny(tok, 'sym', 2)
    (eps, notes), = notes_of(net, game)
    ep, own = eps[0], notes[0]
    a = next(i for i, ag in enumerate(ep['agents']) if not ag['write'])
    st = net.stream(ep, a)
    s, cut, _ = st['asks'][0]
    ids = list(st['ids'][:cut])
    pieces, last = [], 0
    for pos, w, slot in st['reads']:                   # the note tokens' places, m at a time
        if slot % net.m == 0:
            pieces += [tok.decode(ids[last:pos]), own[w, slot // net.m]]
            last = pos + net.m
    pieces.append(tok.decode(ids[last:]))
    assert ''.join(pieces) == B.prompt_text(tok, ep['agents'][a]['system'], B.turns(ep, a, own, s))
    assert C.NOTE not in B.prompt_text(tok, ep['agents'][a]['system'], B.turns(ep, a, None, s))


def test_forms_and_notes_shown(tok):
    net = tiny(tok, 'sym', 2)
    items = B.forms_of(notes_of(net, 'count'))
    assert [len(f) for _, f in items] == [5, 5, 5]
    assert items[0][1]['swap'] == items[1][1]['own'] and items[2][1]['swap'] == items[0][1]['own']
    e = items[0][1]['english_own'][0, 0]
    assert 'count the apples' in e and str(items[0][0]['count']['numbers'][0]) in e
    got = B.shown(items, 'count', n=4)
    assert len(got) == 2 and len(got[0]['notes']) == 3 and got[0]['truth'] == items[0][0]['count']
    assert got[0]['notes'][0]['note'] == items[0][1]['own'][0, 0] and 'apples' in got[0]['says'][0]
    assert B.chance_of([ep for ep, _ in items]) == pytest.approx(1 / 7, abs=1e-4)
    assert B.shares([[1., 0.], [], [1.]]) == .75


def test_right_answers_and_examples(tok):
    net = tiny(tok, 'sym', 2)
    model = net.model.get_base_model().eval()
    (eps, notes), = notes_of(net, 'count')
    (shot_eps, shot_notes), = notes_of(net, 'count', 2, 'train')
    ep, own = eps[0], notes[0]
    a = next(i for i, ag in enumerate(ep['agents']) if not ag['write'])
    s = next(i for i, sg in enumerate(ep['agents'][a]['segments']) if sg['say'] is not None)
    group = CG.choices(ep, a, s)[0]
    p = B.prompt_text(tok, ep['agents'][a]['system'], B.turns(ep, a, own, s))
    with net.model.disable_adapter(), torch.no_grad():
        got = B.right_answers(model, tok, ep, own, [], 'cpu')
        lp = E.cand_logprobs(model, tok, p, group, 'cpu').tolist()
        direct = []                                    # the same log-probs, one whole say at a time
        for c in group:
            pid, cid = tok(p, add_special_tokens=False).input_ids, tok(c, add_special_tokens=False).input_ids
            logp = model(input_ids=torch.tensor([pid + cid])).logits[0].float().log_softmax(-1)
            direct.append(sum(float(logp[len(pid) + j - 1, t]) for j, t in enumerate(cid)))
    assert lp == pytest.approx(direct, abs=.02)
    assert got == [float(lp[0] > max(lp[1:]))]
    shots = list(zip(shot_eps, shot_notes))
    chat = [t for e, n in shots for t in B.turns(e, a, n)] + B.turns(ep, a, own, s)
    assert sum(t['role'] == 'assistant' for t in chat) == 2 and chat[-1]['role'] == 'user'
    assert shot_notes[0][0, 0] in chat[0]['content']
    with net.model.disable_adapter(), torch.no_grad():
        items = B.forms_of([(eps, notes)])
        out = B.read_all(model, tok, items, B.forms_of([(shot_eps, shot_notes)]), B.FORMS, 2, 'cpu')
    assert set(out) == set(B.FORMS) | {'own_minus_swap', 'own_minus_swap_ci'}
    assert out['own_minus_swap'] == pytest.approx(out['own'] - out['swap'], abs=2e-4)
