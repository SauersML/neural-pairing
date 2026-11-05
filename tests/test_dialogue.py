"""swarm/dialogue.py on a tiny random Qwen3 (the real Qwen3 tokenizer) on the CPU: the games' answers score 1 and are
among their candidates; the letter masks; the opening follows the chat template; soft and hard message tokens; a
training step reaches the LoRA (and the slots) and the letter penalty lowers the letter mass; the readers' gains train
the speakers; the chunked conversation gives the same answer
log-probability as the stream rebuilt in full; the dev numbers and the penalty schedule; and the whole pipeline (train
with privacy, eval, in-context) runs and writes its scores and transcripts."""
import json
import math
import os
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import dialogue as D                        # noqa: E402
from swarm import games as G                           # noqa: E402

TOK = os.environ.get('DIALOGUE_TEST_TOK', 'Qwen/Qwen3-0.6B')   # the Qwen3 tokenizer (the same for Qwen3-4B)


def fake_squad(test):
    return [dict(context=f'The {w} was built in {1800 + i} by the river {v}.', question=f'When was the {w} built?',
                 answers=[str(1800 + i), f'in {1800 + i}'])
            for i, (w, v) in enumerate(zip(G.WORD_POOLS[test][:20], G.WORD_POOLS[test][20:40]))]


def fake_hotpot(test):
    return [dict(question=f'Which river runs past the town where {w} was born?', answer=v, titles=[w, f'{w} town'],
                 pages=[f'{w} was born in Oldtown.', f'Oldtown lies on the river {v}.'])
            for w, v in zip(G.WORD_POOLS[test][:20], G.WORD_POOLS[test][20:40])]


@pytest.fixture(autouse=True)
def stub_data(monkeypatch):
    monkeypatch.setattr(D, 'squad_rows', fake_squad)
    monkeypatch.setattr(CG, 'hotpot_rows', fake_hotpot)


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK, local_files_only=True)


def tiny_model(tok, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      attn_implementation='sdpa')
    return Qwen3ForCausalLM(cfg).to(torch.float32)


def tiny(tok, channel='auto', m=4):
    net = D.build('cpu', tok_name=TOK, model=tiny_model(tok), rank=4, m=m)
    net.channel = channel
    return net


def eps_of(game, n, test=False, seed=0):
    rng = random.Random(seed)
    return [D.episode(game, rng, test) for _ in range(n)]


@pytest.mark.parametrize('game', D.GAMES)
def test_game_answers_score_one(game):
    for ep in eps_of(game, 20, test=True):
        for x in 'AB':
            assert ep['views'][x] and ep['answers'][x]
            assert D.score(ep, x, ep['answers'][x])['main'] == 1.0
            if 'cands' in ep:
                assert ep['answers'][x] in ep['cands'][x]
                wrong = [c for c in ep['cands'][x] if c != ep['answers'][x]]
                assert all(D.score(ep, x, c)['main'] < 1.0 for c in wrong)


def test_games_hide_the_answer_from_b():
    for ep in eps_of('twenty', 20):
        assert 'The target is' not in ep['views']['B'] and 'The target is' in ep['views']['A']
    for ep in eps_of('intersect', 20, test=True):
        assert set(ep['golds']['lists']['A']) & set(ep['golds']['lists']['B']) == set(ep['golds']['shared'])
        assert set(ep['golds']['shared']) <= set(G.WORD_POOLS[True])
    for ep in eps_of('schedule', 20):
        free = ep['golds']['free']
        both = [h for h, a, b in zip(D.HOURS, free['A'], free['B']) if a and b]
        assert ep['answers']['B'] == both[0]


def test_b_alone_cannot_find_the_target():
    """twenty: the target is not the list's most typical object (a leak in the first version: every other object
    was the target with one or two attributes changed)."""
    hits = 0
    eps = eps_of('twenty', 300)
    for ep in eps:
        objs = [tuple(line.split('. a ')[1].split()) for line in ep['views']['B'].split('\n') if '. a ' in line]
        overlap = [sum(sum(a == b for a, b in zip(o, q)) for q in objs) for o in objs]
        hits += overlap.index(max(overlap)) + 1 == ep['golds']['k']
    assert hits / len(eps) < 0.15



def test_letter_sets(tok):
    """Letters in any script and byte fragments are letters; digits, punctuation, symbols, emoji and whitespace are
    allowed and not letters; special tokens, control characters and rows past the tokenizer are masked."""
    V = len(tok) + 8
    allowed, letter = D.letter_sets(tok, V)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    for s in ('hello', ' the', 'Straße', ' мир', 'نعم'):
        assert all(letter[i] and allowed[i] for i in enc(s)), s
    for s in ('123', ' 7', '!', ' ->', '%', '\n', '  ', ' (', '🙂'):
        ids = enc(s)
        assert all(allowed[i] for i in ids), s
        if s != '🙂':                                    # an emoji may be split into byte fragments
            assert not any(letter[i] for i in ids), s
    zh = enc('中')
    assert all(letter[i] for i in zh)
    for t in tok.all_special_ids + [tok.convert_tokens_to_ids('<think>')]:
        assert not allowed[t]
    assert not allowed[len(tok):].any()
    ctrl = tok.convert_tokens_to_ids(tok.convert_ids_to_tokens(enc('\x01'))[0])
    assert not allowed[ctrl]
    frag = [i for i in range(V) if allowed[i] and letter[i]]
    assert 0 < int((allowed & ~letter).sum()) < int(allowed.sum()) and len(frag) > 0


def test_opening_follows_chat_template(tok):
    net = tiny(tok)
    ep = eps_of('schedule', 1)[0]
    got = tok.decode(net.prefix(ep, 'B', 'closed') + net.ids(D.SUFFIX))
    want = tok.apply_chat_template([dict(role='system', content=got.split('\n')[1].split('<|im_end|>')[0]),
                                    dict(role='user', content=ep['views']['B'])], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    assert got == want


def test_soft_and_hard_tokens(tok):
    """soft: softmax(logits / tau) over the allowed vocabulary times the embeddings; hard: the argmax token's row,
    never a masked token; a low tau brings soft to hard."""
    net = tiny(tok)
    net.train()
    E = net.lm.model.embed_tokens.weight
    h = torch.randn(2, 3, E.shape[1])
    soft, idx, p = net.write(h, hard=False)
    assert torch.allclose(p.sum(-1), torch.ones(2, 3)) and float(p[..., ~net.allowed].sum()) == 0
    assert torch.allclose(soft, p @ E, atol=1e-5)
    hard, idx2, none = net.write(h, hard=True)
    assert none is None and torch.equal(idx, idx2) and torch.equal(hard, E[idx]) and net.allowed[idx].all()
    net.tau = 0.01
    sharp, _, _ = net.write(h, hard=False)
    assert (sharp - hard).abs().max() < (soft - hard).abs().max()


def grads(net):
    lora = [p for n, p in net.model.named_parameters() if 'lora_' in n and '.default.' in n]
    return dict(slots=net.slots.grad is not None and net.slots.grad.abs().sum() > 0,
                lora=bool(lora) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora))


@pytest.mark.parametrize('channel', ['auto', 'jacobi', 'slots', 'kv'])
def test_training_step_reaches_the_writer(tok, channel):
    """The listener's loss reaches the writer through its message tokens (the slots channel: its slots too; kv: its
    slots' keys and values), and the message gradient is tracked."""
    net = tiny(tok, channel)
    net.train()
    net.track = True
    loss, parts, ids = net.loss(eps_of('twenty', 2), lam=1.0)
    loss.backward()
    assert torch.isfinite(loss) and {'ce_A', 'ce_B'} <= set(parts) and net.msg_grad() > 0
    assert grads(net) == dict(slots=channel in ('slots', 'kv'), lora=True)
    if channel != 'kv':
        assert ids.shape == (2, len(D.SPEAKERS), 4) and 0 <= parts['letters_soft'] <= 1
        assert {'letters_hard', 'ent', 'distinct', 'info'} <= set(parts)
        base, _, _ = net.loss(eps_of('schedule', 2))
        with_info, parts2, _ = net.loss(eps_of('schedule', 2), info=1.0)
        assert float(with_info) == pytest.approx(float(base) - parts2['info'], abs=1e-3)


def test_jacobi_converges_to_autoregressive(tok):
    """Hard messages: with m - 1 passes Jacobi decoding writes exactly what token-by-token decoding writes; with fewer
    passes the first passes + 1 tokens already agree."""
    net = tiny(tok, 'auto', m=5)
    net.eval()
    eps = eps_of('twenty', 3)
    with torch.no_grad():
        _, _, auto, _ = net.converse(eps, hard=True)
        net.channel, net.iters = 'jacobi', 4
        _, _, jac, _ = net.converse(eps, hard=True)
        net.iters = 1
        _, _, jac1, _ = net.converse(eps, hard=True)
    assert torch.equal(auto, jac)
    assert torch.equal(auto[:, 0, :2], jac1[:, 0, :2])


def test_jacobi_soft_last_pass_gradient(tok):
    """Soft messages under gradient: m - 1 passes with gradient on the last only give auto's soft vectors, and a loss
    on them reaches the speakers' LoRA."""
    net = tiny(tok, 'auto', m=5)
    net.eval()
    eps = eps_of('twenty', 2)
    _, sent_auto, _, _ = net.converse(eps, hard=False)
    net.channel, net.iters, net.grad_last = 'jacobi', 4, True
    _, sent_jac, _, _ = net.converse(eps, hard=False)
    for a, b in zip(sent_auto, sent_jac):
        assert torch.allclose(a.float(), b.float(), atol=1e-3)
    sum(v.float().pow(2).sum() for v in sent_jac).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in net.model.named_parameters() if 'lora_' in n)


def test_letter_penalty_lowers_letter_mass(tok):
    """Steps on the letter penalty alone move the softmax mass off the letters."""
    net = tiny(tok, 'slots')
    net.train()
    eps = eps_of('schedule', 2)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.05)
    masses = []
    for _ in range(6):
        S, sent, ids, probs = net.converse(eps)
        mass = (torch.stack(probs, 1) @ net.letter.float()).mean()
        masses.append(float(mass.detach()))
        mass.backward()
        opt.step()
        opt.zero_grad()
    assert all(b < a for a, b in zip(masses, masses[1:])), masses     # a tiny random model's logits barely move


def test_base_gain_trains_the_speakers(tok):
    """The untrained model's gain over its floor reaches the speakers through the soft messages, and the agents' LoRA
    stays trainable after the reader's adapter-off pass."""
    net = tiny(tok, 'slots')
    net.train()
    eps = eps_of('twenty', 2)
    loss, parts, ids = net.loss(eps, lam_p=1.0)
    assert 'gain_base' in parts
    S, sent, ids, _ = net.converse(eps)
    g = net.reader_gain(eps, sent)
    g.sum().backward()
    assert grads(net)['slots'] and g.shape == (2,)
    assert all(p.requires_grad for n, p in net.model.named_parameters() if '.default.' in n)


@pytest.mark.parametrize('channel', ['auto', 'jacobi', 'slots'])
def test_chunked_equals_full(tok, channel):
    """Hard messages: B's answer log-probability from the chunked conversation equals B's stream rebuilt in full from
    the token ids (the slots channel: B's own turns as its slots), and the reader's seat holds the same ids."""
    net = tiny(tok, channel)
    net.eval()
    eps = eps_of('intersect', 3)
    with torch.no_grad():
        S, sent, ids, _ = net.converse(eps, hard=True)
        tails = [net.answer_tail(ep, 'B') for ep in eps]
        h = S['B'].flush(tails)
        U, s = h.shape[1] - max(len(t) for t in tails), len(net.ids(D.SUFFIX))
        a, n = net.target_lp(h, [U + s] * len(eps), [t[s:] for t in tails])
        rows = net.rows(eps, ids, own='B', tails=tails)
        b, m = net.read_lp(rows, [t[s:] for t in tails])
    assert torch.equal(n, m)
    assert torch.allclose(a, b, atol=1e-4), (a, b)
    assert ids.shape == (3, len(D.SPEAKERS), 4) and net.allowed[ids].all()


def test_cut_and_transplant(tok):
    net = tiny(tok)
    net.eval()
    eps = eps_of('twenty', 3) + eps_of('schedule', 2)
    with torch.no_grad():
        on, sent, _ = D.answer_nats(net, eps, hard=True)
        same, _, _ = D.answer_nats(net, eps, 'transplant', sent, hard=True)
        cut, _, _ = D.answer_nats(net, eps, 'cut', hard=True)
        msgs, texts = D.conversations(net, eps)
        _, tr = D.conversations(net, eps, mode='transplant', donor=msgs)
    assert on['B'] == pytest.approx(same['B'], abs=1e-4) and on['B'] != pytest.approx(cut['B'], abs=1e-4)
    assert D.swap_order(eps) == [1, 2, 0, 4, 3] and len(tr['B']) == 5


def test_kv_channel_cut_and_transplant(tok):
    net = tiny(tok, 'kv')
    net.eval()
    eps = eps_of('twenty', 3)
    with torch.no_grad():
        on, sent, _ = D.answer_nats(net, eps)
        same, _, _ = D.answer_nats(net, eps, 'transplant', sent)
        cut, _, _ = D.answer_nats(net, eps, 'cut')
    assert on['B'] == pytest.approx(same['B'], abs=1e-4) and on['B'] != pytest.approx(cut['B'], abs=1e-4)
    d = D.dev_eval(net, eps)
    assert 'xgap_B' in d['twenty'] and 'soft_on' not in d['twenty'] and '_all' not in d


def test_dev_eval_reports(tok):
    net = tiny(tok, 'jacobi')
    dev = eps_of('twenty', 3) + eps_of('schedule', 3)
    d = D.dev_eval(net, dev, ('base',))
    r = d['twenty']
    for k in ('on', 'swap', 'soft_on', 'soft_swap', 'xgap_A', 'xgap_B', 'soft_xgap_A', 'soft_xgap_B', 'letters',
              'distinct', 'gain_base'):
        assert k in r, k
    assert len(r['examples']) == 3 and len(r['examples'][0]) == len(D.SPEAKERS) and 0 <= d['_all']['letters'] <= 1
    assert D.held(d, ['twenty', 'schedule']) == pytest.approx((r['xgap_B'] + d['schedule']['xgap_B']) / 2, abs=1e-3)
    light = D.dev_eval(net, dev, soft=False)
    assert 'soft_xgap_B' not in light['twenty'] and light['twenty']['xgap_B'] == pytest.approx(r['xgap_B'], abs=1e-3)


def test_letter_schedule():
    import argparse
    args = argparse.Namespace(free=300, lam0=0.1, lam_every=200, lam_max=1.0, letter_goal=0.01)
    lam, raised = 0.0, None
    seen = []
    for step, letters in ((0, 0.9), (200, 0.9), (300, 0.9), (400, 0.9), (500, 0.9), (600, 0.5), (700, 0.005),
                          (900, 0.005), (1000, 0.2), (1100, 0.2), (1300, 0.2), (1500, 0.2), (1700, 0.2)):
        lam, raised = D.letter_lam(step, lam, raised, letters, args)
        seen.append(lam)
    assert seen == [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.8, 1.0, 1.0]


def test_pipeline_writes_scores(tok, tmp_path, monkeypatch):
    shared, real_build = {}, D.build

    def build(device, base=D.BASE, rank=16, ckpt=None, tok_name=None, model=None, m=D.M):
        if 'net' not in shared:
            shared['net'] = real_build('cpu', tok_name=TOK, model=tiny_model(tok), rank=4, ckpt=ckpt, m=4)
        return shared['net']

    def load_lm(name, device):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
        t.padding_side = 'left'
        return tiny_model(tok, seed=3).eval(), t

    monkeypatch.setattr(D, 'build', build)
    monkeypatch.setattr(D, '_load_lm', load_lm)
    out, ckpt = tmp_path / 'out', tmp_path / 'ckpt'
    monkeypatch.setattr(sys, 'argv', ['dialogue.py', '--out', str(out), '--ckpt', str(ckpt), '--steps', '4',
                                      '--batch', '2', '--micro', '2', '--dev_every', '2', '--dev_n', '2', '--free', '1',
                                      '--priv_by', '2', '--priv_ramp', '1', '--lam_every', '1', '--log_every', '1', '--dev_big', '2', '--full_every', '4', '--channel', 'jacobi', '--ctrl', '--contrast', '0.5', '--bias_k', '10',
                                      '--iters', '2',
                                      '--n', '3', '--n_ic', '2', '--shots', '40', '--info', '0.1',
                                      '--device', 'cpu', '--tok', TOK])
    D.main()
    scores = json.loads((out / 'scores.json').read_text())
    assert set(scores) == set(D.GAMES)
    for g, s in scores.items():
        for k in ('trained', 'trained_cut', 'trained_transplant', 'base_reads', 'base_closed', 'oracle',
                  'ic_Qwen3-8B', 'ic_Qwen3-8B_closed', 'ic_AFM-4.5B'):
            assert k in s['means'], (g, k)
        assert 'trained' in s['forced'] and 'base_reads' in s['paired']
        assert {'trained', 'base_reads', 'ic_Qwen3-8B', 'ic_AFM-4.5B', 'forced_trained'} <= set(s['gains'])
        if g in ('schedule', 'intersect', 'twenty'):
            assert 'choice' in s['forced']['trained']
    assert (ckpt / 'talk.pt').exists() and (ckpt / 'adapter').exists()
    tr = json.loads((out / 'transcripts.json').read_text())
    assert [r['i'] for r in tr['test']['squad']] == [0, 1, 2]
    assert len(tr['test']['squad'][0]['ids']) == len(D.SPEAKERS) and len(tr['shots']['twenty']) >= D.SHOTS
    ex = json.loads((out / 'examples.json').read_text())
    assert len(ex['squad'][0]['msgs']) == len(D.SPEAKERS) and 'ic_AFM-4.5B' in ex['squad'][0]['readers']
    log = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    assert any(r.get('ctrl') or r.get('ctrl_wait') for r in log) and any('contrast' in r for r in log)
    assert any(r.get('done') for r in log) and [r['step'] for r in log if isinstance(r.get('dev'), dict)] == [0, 2, 4]
    devs = [r['dev'] for r in log if isinstance(r.get('dev'), dict)]
    assert [r['full'] for r in log if isinstance(r.get('dev'), dict)] == [True, False, True]
    assert devs[-1]['squad']['xgap_B_0'] == devs[0]['squad']['xgap_B']
    again = D.build('cpu', tok_name=TOK, model=tiny_model(tok), rank=4, ckpt=ckpt)
    assert again.channel == 'jacobi' and again.iters == 2 and again.m == 4


def fake_corpus():
    """A stand-in for swarm/corpus.py: one fill-in task (each agent predicts the partner's chunk) and one split-input
    task."""
    from types import SimpleNamespace

    def fill(rng, test):
        w = G.WORD_POOLS[test]
        chunks = {x: ' '.join(rng.sample(w, 12)) for x in 'AB'}
        return dict(game='fill_web', views={x: f'Your chunks: {chunks[x]}' for x in 'AB'},
                    answers=dict(A=chunks['B'], B=chunks['A']), weights=dict(A=1.0, B=1.0), golds={})

    def split(rng, test):
        a, b = rng.randrange(50), rng.randrange(50)
        return dict(game='split_sum', views=dict(A=f'You hold {a}.', B=f'You hold {b}. Name the sum.'),
                    answers=dict(A=str(a + b), B=str(a + b)), weights=dict(A=0.5, B=1.0), golds={})

    return SimpleNamespace(TASKS=dict(fill_web=fill, split_sum=split), FAMILY=dict(fill_web=1, split_sum=2),
                           MIX=dict(fill_web=0.6, split_sum=0.4), mix_of=lambda: dict(fill_web=0.6, split_sum=0.4),
                           score=lambda ep, x, text: dict(main=float(text.strip() == ep['answers'][x])))


def test_corpus_mixture(tok, monkeypatch):
    import argparse
    monkeypatch.setattr(D, 'CP', fake_corpus())
    args = argparse.Namespace(games='schedule,twenty', mix='corpus', old_share=0.05)
    w = D.mixture(args)
    assert w == pytest.approx(dict(schedule=0.025, twenty=0.025, fill_web=0.57, split_sum=0.38))
    assert [D.family(g) for g in w] == [0, 0, 1, 2]
    ep = D.test_set('fill_web', 1, 'dev')[0]
    assert ep['game'] == 'fill_web' and D.score(ep, 'A', ep['answers']['A'])['main'] == 1.0
    net = tiny(tok)
    d = D.dev_eval(net, D.test_set('fill_web', 2, 'dev') + D.test_set('split_sum', 2, 'dev'))
    assert -20 < d['fill_web']['on'] < 0           # nats per answer token, not per answer


def test_dev_pools_fill_sources(tok, monkeypatch):
    import argparse
    cp = fake_corpus()

    def wiki(rng, test):
        return dict(cp.TASKS['fill_web'](rng, test), game='fill_wiki')

    cp.TASKS['fill_wiki'], cp.FAMILY['fill_wiki'] = wiki, 1
    monkeypatch.setattr(D, 'CP', cp)
    games = ['squad', 'fill_web', 'fill_wiki', 'split_sum']
    a = argparse.Namespace(dev_big=5, dev_n=2, dev_n_corpus=3)
    assert [D.dev_size(g, games, a) for g in games] == [5, 3, 3, 3]
    d = D.dev_eval(tiny(tok), D.test_set('fill_web', 3, 'dev') + D.test_set('fill_wiki', 3, 'dev'))
    for k in ('on', 'xgap_B', 'soft_xgap_B'):
        assert d['fill'][k] == pytest.approx((d['fill_web'][k] + d['fill_wiki'][k]) / 2, abs=1e-3)


def test_letter_controller():
    import argparse
    a = argparse.Namespace(ctrl_up=1.5, ctrl_down=0.5, ctrl_keep=0.8, lam_max=50.0)
    assert D.letter_ctrl(0.01, 8.5, 10.0, a) == (pytest.approx(0.015), True)
    assert D.letter_ctrl(0.01, 7.9, 10.0, a) == (pytest.approx(0.005), False)
    assert D.pooled_gap({'squad': {'xgap_B': 2.0}, 'fill': {'xgap_B': 4.0}, 'fill_web': {'xgap_B': 100.0}}) == 3.0
    b = argparse.Namespace(ctrl_start=0.5)
    assert D.ctrl_ready(6.0, 10.0, b) and not D.ctrl_ready(4.0, 10.0, b) and not D.ctrl_ready(0.0, 0.0, b)


def test_contrast_charges_a_constant_conversation(tok):
    """Three copies of one episode: the conversations are the same, so the margin is 0 and the hinge charges its
    whole target; on different episodes the term reaches the speakers."""
    net = tiny(tok, 'jacobi')
    net.eval()
    ep = eps_of('twenty', 1)[0]
    base, _, _ = net.loss([ep] * 3)
    with_c, parts, _ = net.loss([ep] * 3, contrast=1.0)
    assert parts['contrast'] == pytest.approx(0.0, abs=1e-3)
    assert float(with_c) == pytest.approx(float(base) + 0.2, abs=1e-3)
    loss, parts, _ = net.loss(eps_of('twenty', 3), contrast=1.0)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in net.model.named_parameters() if 'lora_' in n)


def test_letter_bias_clears_letters_from_hard_messages(tok):
    net = tiny(tok, 'jacobi')
    net.eval()
    eps = eps_of('twenty', 2)
    net.bias = 1e4
    with torch.no_grad():
        _, _, ids, _ = net.converse(eps, hard=True)
    assert D.letter_frac(ids, net.letter) == 0.0


def test_contrast_gradient_stays_on_the_cross_entropy_scale(tok, monkeypatch):
    """The contrastive term works per answer token, as the cross-entropy does: on 20-token answers its gradient on the
    messages stays within a few times the cross-entropy's (28 summed the answer log-probability, about 20 times
    larger here, and its messages collapsed to one token in 160 steps). B's own turns stay its own in the negative:
    only the messages it received come from the other episode."""
    monkeypatch.setattr(D, 'CP', fake_corpus())
    net = tiny(tok, 'jacobi')
    net.train()                                         # soft messages (no dropout in the tiny model or the LoRA)
    eps = D.test_set('fill_web', 3, 'dev')
    norms = []
    for c in (0.0, 1.0):
        net.model.zero_grad(set_to_none=True)
        net.track = True
        loss, parts, _ = net.loss(eps, contrast=c)
        loss.backward()
        net.track = False
        norms.append(net.msg_grad())
    assert norms[1] < 3 * norms[0], norms
    sent = [torch.full((3, 2, 1), float(t)) + torch.arange(3.0)[:, None, None] for t in range(len(D.SPEAKERS))]
    mixed = D.swap_received(sent, [1, 2, 0])
    for t, spk in enumerate(D.SPEAKERS):
        want = [1, 2, 0] if spk == 'A' else [0, 1, 2]
        assert mixed[t][:, 0, 0].tolist() == [t + k for k in want], (t, spk)
