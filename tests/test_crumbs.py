"""swarm/crumbs.py on a tiny random Qwen3 (the real Qwen3-0.6B tokenizer) on the CPU: the token stream follows the chat
template, each arm trains one step and answers greedily, the follower's loss reaches the crumb inputs, and a reader's
token reads only the crumbs left at its own place by earlier agents."""
import random
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402

ARMS = (('cut', 4), ('resid', 4), ('resid', 1), ('kv', 4), ('kv', 1), ('text', 4), ('tok', 4), ('tok', 2))


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def tiny(tok, arm, m):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok) + C.ALPHABET, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      attn_implementation='sdpa')
    return C.build('cpu', arm, m, model=Qwen3ForCausalLM(cfg).to(torch.float32), tok=tok, layers=(1, 2))


def trails(n, seed=0, test=False):
    rng = random.Random(seed)
    return [CG.episode('trail', rng, test) for _ in range(n)]


def test_stream_follows_chat_template(tok):
    net = tiny(tok, 'cut', 4)
    ep = trails(1)[0]
    st = net.stream(ep, 1)
    s, cut, _ = st['asks'][0]
    fol = ep['agents'][1]
    msgs = [dict(role='system', content=fol['system'])]
    for seg in fol['segments'][:s]:
        msgs += [dict(role='user', content=seg['user'])]
    # consecutive user turns render one after another, as in the stream
    want = tok.apply_chat_template(msgs + [dict(role='user', content=fol['segments'][s]['user'])], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    assert tok.decode(st['ids'][:cut]) == want
    say = tok.decode([i for i, t in zip(st['ids'], st['target']) if t])
    assert say == fol['segments'][s]['say'] + C.END


def test_writer_slots_and_text_notes(tok):
    ep = trails(1)[0]
    net = tiny(tok, 'kv', 4)
    st = net.stream(ep, 0)
    assert len(st['slots']) == 4 * CG.N_ROOMS and sum(k >= 0 for k in st['kind']) == 4 * CG.N_ROOMS
    assert [p for _, p in st['slots'][::4]] == ep['trail']['order']
    note = tiny(tok, 'text', 4).stream(ep, 1)
    text = tok.decode(note['ids'])
    for r in ep['trail']['visit']:
        assert f'A note left here: You enter the {ep["places"][r]}. You see a {ep["trail"]["objects"][r]}.' in text


@pytest.mark.parametrize('arm,m', ARMS)
def test_arm_trains_and_answers(tok, arm, m, tmp_path):
    net = tiny(tok, arm, m)
    if arm == 'tok':
        net.adv = 1.
    C.train(net, 'cpu', 1, 2, 0, games=('trail',))
    loss = net.loss(trails(2), 'cpu')
    assert torch.isfinite(loss)
    loss.backward()
    if arm in C.CRUMB_ARMS:
        assert net.crumb.grad is not None and net.crumb.grad.abs().sum() > 0
    net.zero_grad()
    out, rows = C.evaluate(net, 'cpu', n=4, n_choice=2, n_gen=2, seed=1, games=('trail',), bs=2, keep_rows=2)
    tf, gen = out['trail']['tf'], out['trail']['gen']
    crumbs = arm in C.CRUMB_ARMS
    assert set(tf) == ({'on', 'off', 'transplant'} if crumbs else {'on'}) | ({'soft', 'base', 'base_off'} if arm == 'tok' else set())
    assert set(gen) == ({'on', 'off', 'swap', 'room'} if crumbs else {'on'}) | ({'base'} if arm == 'tok' else set())
    for v in tf.values():
        assert v['nll'] > 0 and v['n'] == 4 and v['n_choice'] == 2 and 0 <= v['choice'] <= 1
    assert out['trail']['episodes']['on']['choice'][2:] == [None, None]
    assert rows and all('1/3' in r['says'] for r in rows)
    if crumbs:
        got = C.probe(net, 'cpu', n_fit=12, n_test=4, bs=4)
        assert 0 <= got['best'] <= 1 and got['chance'] == 1 / len(set(C.G.WORD_POOLS[1]))


def test_forced_choice_matches_scoring(tok):
    """The gold say's NLL equals the cross-entropy the trainer sums, and a forced choice counts per group."""
    net = tiny(tok, 'kv', 2)
    net.eval()
    eps = trails(2)
    with torch.no_grad():
        nll, choice = net.forced(eps, 'cpu')
        total, count, _ = net.run(eps, 'cpu')
    toks = [len(net._ids(ep['agents'][1]['segments'][-1]['say'])) + 1 for ep in eps]
    assert abs(sum(nll) - float(total)) < 1e-3 * float(total)
    assert count == sum(toks)
    assert all(c in (0, .25, .5, .75, 1) for c in choice)


def test_report_pairs_arms(tok, tmp_path):
    for arm, m in (('cut', 4), ('kv', 4)):
        net = tiny(tok, arm, m)
        out, _ = C.evaluate(net, 'cpu', n=4, n_choice=4, n_gen=0, seed=1, games=('trail',), bs=2)
        d = tmp_path / f'{arm}{m}'
        d.mkdir()
        (d / 'scores.json').write_text(C.json.dumps(dict(steps_run=1, test=out)))
    rep = C.report(tmp_path, boots=50)
    v = rep['vs_cut']['kv4']['trail']['nll']
    assert v['n'] == 4 and v['lo'] <= v['mean'] <= v['hi']
    assert set(rep['within']['kv4']['trail']) == {'on-off', 'on-transplant'}


@pytest.mark.parametrize('arm', C.CRUMB_ARMS)
def test_reader_sees_only_its_places(tok, arm):
    """Changing the crumbs at a room the follower never visits leaves its logits alone; masking its own rooms'
    crumbs changes them; so does changing a visited room's crumbs."""
    net = tiny(tok, arm, 2)
    if arm == 'resid':
        for r in net.readers:
            r.alpha.data.fill_(1.0)
    net.eval()
    eps = trails(2)
    with torch.no_grad():
        _, _, bank = net.run(eps, 'cpu', need_loss=False)
        x = net._pad([net.stream(ep, 1) for ep in eps], 'cpu')
        rows = torch.arange(2)

        def logits(b, on=True):
            return net._forward(rows, x, 1, b, 1, False, on)[0][:, -1]

        base = logits(bank)
        unseen = [r for r in range(CG.N_ROOMS) if r not in eps[0]['trail']['visit']][0]
        seen = eps[0]['trail']['visit'][0]
        name = dict(resid='h', kv='k', tok='y')[arm]

        def poke(place):
            b = dict(bank)
            sl = (bank['place'][0] == place).nonzero()[:, 0]
            new = []
            for t in bank[name]:
                t = t.clone()
                if name == 'y':
                    t[0, sl] = t[0, sl].roll(1, -1)
                elif name == 'h':
                    t[0, sl] += 5.0
                else:
                    t[0, :, sl] += 5.0
                new.append(t)
            b[name] = new
            return b

        assert torch.allclose(logits(poke(unseen)), base, atol=1e-5)
        assert not torch.allclose(logits(poke(seen))[0], base[0], atol=1e-4)
        assert torch.allclose(logits(poke(seen))[1], base[1], atol=1e-5)
        assert not torch.allclose(logits(bank, on=False), base, atol=1e-4)


@pytest.mark.parametrize('arm', ('resid', 'kv', 'text', 'tok'))
def test_every_game_runs(tok, arm):
    """Each crumb game trains, reads teacher-forced with its choices, and answers greedily (test values)."""
    net = tiny(tok, arm, 2)
    for game in CG.GAMES:
        rng = random.Random(5)
        eps = [CG.episode(game, rng, True) for _ in range(2)]
        loss = net.loss(eps, 'cpu')
        assert torch.isfinite(loss), game
        loss.backward()
        net.zero_grad()
        nll, _ = net.forced(eps, 'cpu', 'transplant' if arm != 'text' else 'on')
        assert all(v is not None and v > 0 for v in nll), game
        got = net.answer(eps, 'cpu')
        want = {(b, a, s) for b, ep in enumerate(eps) for a, ag in enumerate(ep['agents'])
                for s, sg in enumerate(ag['segments']) if sg['say'] is not None}
        assert set(got) == want, game


def test_tok_messages_and_adversary(tok, tmp_path):
    """The follower's stream holds m read positions per visited room; messages are one-hot; the adversary term is
    positive at a random start and its gradient reaches A and the writer's crumb inputs; the base mode reads with
    the adapter disabled; intercept and load round-trip."""
    net = tiny(tok, 'tok', 3)
    ep = trails(1)[0]
    st = net.stream(ep, 1)
    assert len(st['reads']) == 3 * CG.N_VISIT
    assert all(net.tok.decode(st['ids'][p]) == net.tok.pad_token for p, _, _ in st['reads'])
    net.adv = 1.
    net.adv_sum, net.adv_n = 0., 0
    total, count, bank = net.run(trails(4), 'cpu')
    y = bank['y'][0]
    assert torch.allclose(y.sum(-1)[bank['place'] >= 0], torch.ones(1))
    assert net.adv_n == 4 * (CG.N_VISIT + 1)
    loss = net.loss(trails(4), 'cpu')
    loss.backward()
    assert net.A.grad.abs().sum() > 0 and net.crumb.grad.abs().sum() > 0
    net.zero_grad()
    with torch.no_grad():                                 # LoRA B starts at zero: give the adapter an effect
        for n, prm in net.model.named_parameters():
            if 'lora_B' in n:
                prm.normal_(0, 0.05)
    net.eval()
    on, _ = net.forced(trails(2), 'cpu', 'on')
    base, _ = net.forced(trails(2), 'cpu', 'base')
    assert on != base
    got = C.intercept(net, 'cpu', n=3, others=1, bs=3)
    trail = [r for r in got['messages'] if r['game'] == 'trail']
    assert len(trail) == 2 * 3 * CG.N_ROOMS and all(len(r['tokens']) == 3 for r in trail)
    assert all(r['vocab_ids'] == [len(tok) + k for k in r['tokens']] for r in trail)
    (tmp_path / 'args.json').write_text(C.json.dumps(dict(arm='tok', m=3, adv=1.)))
