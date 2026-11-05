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

ARMS = (('cut', 4), ('resid', 4), ('resid', 1), ('kv', 4), ('kv', 1), ('text', 4), ('sym', 2))


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def tiny(tok, arm, m):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
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
    C.train(net, 'cpu', 1, 2, 0, games=('trail',))
    loss = net.loss(trails(2), 'cpu')
    assert torch.isfinite(loss)
    loss.backward()
    if arm in ('resid', 'kv'):
        assert net.crumb.grad is not None and net.crumb.grad.abs().sum() > 0
    net.zero_grad()
    out, rows = C.evaluate(net, 'cpu', n=4, n_choice=2, n_gen=2, seed=1, games=('trail',), bs=2, keep_rows=2)
    tf, gen = out['trail']['tf'], out['trail']['gen']
    crumbs = arm in C.CRUMB_ARMS
    assert set(tf) == ({'on', 'off', 'transplant'} if crumbs else {'on'}) | ({'soft', 'base', 'base_off'} if arm == 'sym' else set())
    assert set(gen) == ({'on', 'off', 'swap', 'room'} if crumbs else {'on'}) | ({'base'} if arm == 'sym' else set())
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
        name = dict(resid='h', kv='k', sym='e')[arm]

        def poke(place):
            b = dict(bank)
            sl = (bank['place'][0] == place).nonzero()[:, 0]
            new = []
            for t in bank[name]:
                t = t.clone()
                if name in ('h', 'e'):
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


@pytest.mark.parametrize('arm', ('resid', 'kv', 'text', 'sym'))
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


def lora_effect(net):
    with torch.no_grad():                                 # LoRA B starts at zero: give the adapters an effect
        for n, prm in net.model.named_parameters():
            if 'lora_B' in n:
                prm.normal_(0, 0.05)


def test_letter_table(tok):
    """Letters: any Unicode L* character, or bytes that are not complete UTF-8; digits, punctuation, symbols and
    spaces are allowed; special and added tokens and the rows past the tokenizer's are banned."""
    letter, banned = C.letter_table(tok, len(tok) + 64)
    one = tok.convert_tokens_to_ids
    for t in ('cat', 'Ġthe', 'ä¸Ń', 'ä'):            # a word, a spaced word, a CJK character, a lone UTF-8 lead byte
        assert letter[one(t)], t
    for t in ('7', '!', 'Ġ', '%', 'Ċ', '...'):
        assert not letter[one(t)] and not banned[one(t)], t
    for i in (one('<|im_end|>'), one('<think>'), tok.pad_token_id, len(tok), len(tok) + 63):
        assert banned[i]
    import unicodedata
    for i in range(0, len(tok) - 300, 997):              # the byte map agrees with the tokenizer's own decoding
        text = tok.decode([i])
        if '\ufffd' not in text:
            assert bool(letter[i]) == any(unicodedata.category(c).startswith('L') for c in text), (i, text)
    assert 0.5 < float(letter[~banned].float().mean()) < 0.99


def test_sym_notes_are_greedy_decoding(tok):
    """A hard note is the greedy decoding (special tokens masked) of the writer's tokens up to its place, the cue
    turn and the note so far, run as plain token ids; the bank sends those tokens' embedding rows."""
    net = tiny(tok, 'sym', 3)
    lora_effect(net)
    net.eval()
    eps = trails(2)
    E = net.model.get_input_embeddings().weight
    with torch.no_grad():
        _, _, bank = net.run(eps, 'cpu', need_loss=False)
        for b, ep in enumerate(eps):
            st = net.stream(ep, 0)
            cols = ((bank['agent'][b] == 0) & (bank['place'][b] >= 0)).nonzero()[:, 0]
            assert len(cols) == 3 * CG.N_ROOMS
            for k in range(CG.N_ROOMS):
                pre = st['ids'][:st['slots'][3 * k][0]] + net.cue
                got = []
                for _ in range(3):
                    lg = net.model(input_ids=torch.tensor([pre + got])).logits[0, -1].float()
                    got.append(int(lg.masked_fill(net.banned, float('-inf')).argmax()))
                ids = bank['ids'][0][b, cols[3 * k:3 * k + 3]]
                assert ids.tolist() == got, (b, k)
                assert torch.equal(bank['e'][0][b, cols[3 * k:3 * k + 3]], E[ids])


def test_sym_training_and_privacy(tok):
    """The writer's stream holds no note tokens and the follower's one line per visited room's note; in training
    the follower's loss reaches the notes it reads and nothing else; the letter, entropy and privacy terms are
    collected; the base model and the eavesdropper read with the shared adapter off; the eavesdropper's update
    leaves the shared adapter's gradients alone; intercept and note_stats decode the hard notes."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    ep = trails(1)[0]
    fol, wr = net.stream(ep, 1), net.stream(ep, 0)
    assert len(fol['reads']) == 2 * CG.N_VISIT and tok.decode(fol['ids']).count(C.NOTE) == CG.N_VISIT
    assert len(wr['slots']) == 2 * CG.N_ROOMS and all(k == -1 for k in wr['kind'])
    assert wr['ids'] == tiny(tok, 'cut', 2).stream(ep, 0)['ids']
    net.train()
    net.tau, net.lam, net.beta = 1., 1., 0.1
    eps = trails(2)
    total, count, bank = net.run(eps, 'cpu')
    g = torch.autograd.grad(total, bank['e'][0])[0].abs().sum(-1)
    real = bank['place'] >= 0
    assert not bool(net.banned[bank['ids'][0][real]].any())
    for b, e_ in enumerate(eps):
        seen = torch.tensor([p in e_['trail']['visit'] for p in bank['place'][b].tolist()]) & real[b]
        assert bool((g[b][seen] > 0).all()) and float(g[b][~seen].sum()) == 0
    net.loss(eps, 'cpu')
    s = net.sums
    assert s['n_tok'] == 2 * 2 * CG.N_ROOMS and 0 < float(s['let']) / s['n_tok'] < 1 and float(s['ent']) > 0
    net.adv, net.eaves_on = 1., True
    loss = net.loss(eps, 'cpu')
    assert net.sums['n_adv'] == net.sums['n_eav'] == 2 * (CG.N_VISIT + 1) and len(net.stash) == 1
    loss.backward()
    shared = [p for n, p in net.model.named_parameters() if '.default.' in n]
    eav = net.eaves_params()
    assert eav and all(p.requires_grad for p in shared) and not any(p.requires_grad for p in eav)
    assert all(p.grad is None for p in eav)
    grads = [None if p.grad is None else p.grad.clone() for p in shared]
    assert sum(float(q.abs().sum()) for q in grads if q is not None) > 0
    v = net.eaves_step()
    assert v > 0 and not net.stash and sum(float(p.grad.abs().sum()) for p in eav if p.grad is not None) > 0
    assert all((q is None and p.grad is None) or torch.equal(q, p.grad) for q, p in zip(grads, shared))
    assert all(p.requires_grad for p in shared) and not any(p.requires_grad for p in eav)
    net.zero_grad()
    net.eval()
    on, _ = net.forced(trails(2), 'cpu', 'on')
    base, _ = net.forced(trails(2), 'cpu', 'base')
    soft, _ = net.forced(trails(2), 'cpu', 'soft')
    assert on != base and on != soft
    got = C.intercept(net, 'cpu', n=3, others=1, bs=3)
    trail = [r for r in got['messages'] if r['game'] == 'trail']
    assert len(trail) == 2 * 3 * CG.N_ROOMS
    assert all(len(r['vocab_ids']) == 2 and r['text'] == tok.decode(r['vocab_ids']) for r in trail)
    stats = C.note_stats(net, 'cpu', trails(3))
    assert 0 <= stats['letters'] <= 1 and len(stats['examples']) == 2 * CG.N_ROOMS


def test_keytrail_is_pointer():
    """keytrail draws only trail episodes whose key room the follower never enters but can learn of from a later
    room's crumbs; it is a training variant, not one of the games."""
    rng = random.Random(3)
    assert {CG.case(CG.episode('keytrail', rng)) for _ in range(60)} == {'pointer'}
    assert 'keytrail' not in CG.GAMES


def test_corpus_games(tok, tmp_path, monkeypatch):
    """Corpus tasks as crumb games, on a made-up cache: fill-in puts each of A's parts at its own place and B writes
    them there; a split-input task uses one place; "corpus" expands to every cached task at corpus.mix_of()."""
    from swarm import corpus as CP
    monkeypatch.setattr(CP, 'CACHE', tmp_path)
    monkeypatch.setattr(CP, '_rows', {})
    words = 'the river bends past an old mill where children fish for trout in summer and skate in winter'.split()
    doc = ' '.join(f'{words[i % len(words)]}{i}' + ('.' if i % 9 == 8 else '') for i in range(400))   # no repeats
    for test in (False, True):
        CP.write_rows('fill_web', test, [dict(text=doc, part=0)] * 3)
        CP.write_rows('qa_squad', test, [dict(context='The mill was built in 1820 by the river.',
                                               question='When was the mill built?', answers=['1820'])] * 3)
    games, weights = C.expand_games('corpus,trail', {'trail': 2.})
    assert games == ('c_fill_web', 'c_qa_squad', 'trail') and weights['trail'] == 2.
    assert abs(weights['c_fill_web'] - 0.6 * 2) < 1e-9 and abs(weights['c_qa_squad'] - 0.4 * 2) < 1e-9
    rng = random.Random(0)
    fill, qa = CG.episode('c_fill_web', rng), CG.episode('c_qa_squad', rng)
    writer, reader = fill['agents']
    assert writer['write'] and not reader['write'] and len(fill['places']) == CP.CHUNKS
    chunks = [c.strip() for c in fill['corpus']['golds']['chunks']]
    assert [sg['place'] for sg in writer['segments']] == list(range(CP.CHUNKS))
    own = range(0, 2 * CP.CHUNKS, 2)
    assert [sg['say'] for sg in reader['segments'][1:]] == [f'({i + 1}) {chunks[i]}' for i in own]
    assert all(chunks[i] not in reader['segments'][0]['user'] for i in own)
    assert qa['places'] == ['the desk'] and qa['agents'][1]['segments'][0]['say'] == '1820'
    say = reader['segments'][1]['say']
    assert CG.score(fill, 1, 1, say) == 1. and CG.score(qa, 1, 0, '1820') == 1. and CG.score(qa, 1, 0, '1700') == 0.
    net = tiny(tok, 'kv', 2)
    loss = net.loss([fill, qa], 'cpu')
    assert torch.isfinite(loss)
    loss.backward()
    nll, _ = net.forced([fill, qa], 'cpu', 'transplant')
    assert all(v is not None and v > 0 for v in nll)
