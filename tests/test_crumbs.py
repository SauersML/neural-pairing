"""swarm/crumbs.py on a tiny random Qwen3 (the real Qwen3-0.6B tokenizer) on the CPU: the token stream follows the chat
template, each arm trains one step and answers greedily, the follower's loss reaches the crumb inputs, and a reader's
token reads only the crumbs left at its own place by earlier agents."""
import collections
import json
from pathlib import Path
import random
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
    # the exact rule: a byte fragment is a letter only if its bytes occur inside some letter's UTF-8 encoding; the
    # first rule (exact False) makes every fragment a letter
    first = C.letter_table(tok, len(tok) + 64, exact=False)[0]
    assert not letter[one('ðŁ')] and first[one('ðŁ')]       # bytes F0 9F: the start of every 4-byte emoji
    assert letter[one('Ð')] and first[one('Ð')]              # byte D0: the start of Cyrillic letters
    assert bool((letter <= first).all()) and int((first & ~letter).sum()) > 100
    assert C.fragment_letter(bytes([0xF0, 0x9F])) is False and C.fragment_letter(bytes([0xD0])) is True
    assert C.fragment_letter('7'.encode() + bytes([0xF0, 0x9F])) is False
    assert C.fragment_letter('a'.encode() + bytes([0xF0, 0x9F])) is True


def test_letters_by_context(tok):
    """The sym arm's letters depend on the bytes pending before a token: the emoji U+1F9F1 is the lead fragment
    F0 9F A7 and the lone continuation byte B1, letter-free together, while B1 after C3 ends U+00F1, a letter; the
    penalty takes each note's pending bytes from its argmax tokens so far, and letters_exact reads each note from its
    start the same way."""
    net = tiny(tok, 'sym', 4)
    one = tok.convert_tokens_to_ids
    lead, b1, c3 = one('ðŁ§'), one('±'), one('Ã')
    assert net.masks.walk([lead, b1])[0] == [False, False]
    assert net.masks.walk([c3, b1])[0] == [True, True]
    assert net.masks.walk([b1])[0] == [True]                    # nothing pending: a lone continuation byte is a letter
    assert tok.decode([lead, b1]) == '\U0001f9f1' and tok.decode([c3, b1]) == '\u00f1'
    assert net.masks.pending(b'', lead) == bytes([0xF0, 0x9F, 0xA7]) and net.masks.pending(bytes([0xF0, 0x9F, 0xA7]), b1) == b''
    assert bool(net.letter[b1]) and not bool(net.masks.mask(bytes([0xF0, 0x9F, 0xA7]))[b1])
    # the penalty: the letter mass after each note's own pending bytes
    V = net.letter.shape[0]
    p = torch.zeros(1, 2, V)
    # note 0 after F0 9F A7: B1 is free, a second lead breaks the open emoji (a letter); note 1 after C3: B1 ends
    # U+00F1 and 7 breaks it, both letters
    p[0, 0, b1], p[0, 0, lead] = .75, .25
    p[0, 1, b1], p[0, 1, one('7')] = .5, .5
    hard = torch.tensor([[b1, b1]])
    net.sums = collections.defaultdict(float)
    net.codes = collections.defaultdict(list)
    net._track(p, hard, torch.ones(1, 2, dtype=torch.bool), 1, [[bytes([0xF0, 0x9F, 0xA7]), bytes([0xC3])]])
    assert abs(float(net.sums['let']) - 1.25) < 1e-6 and net.sums['hard_let'] == 1 and net.sums['n_tok'] == 2
    # letters_exact: each m-token note read from its start
    eps = trails(2)
    ns = C.note_stats(net, 'cpu', eps)
    _, _, bank = net.run(eps, 'cpu', need_loss=False)
    ids, real = bank['ids'][0], bank['place'] >= 0
    want, n = 0, 0
    for b in range(len(eps)):
        got = ids[b][real[b]].tolist()
        assert len(got) % net.m == 0
        want, n = want + sum(sum(net.masks.walk(got[g:g + net.m])[0]) for g in range(0, len(got), net.m)), n + len(got)
    assert abs(ns['letters_exact'] - want / n) < 1e-9


def test_note_pass_one_matches_greedy(tok):
    """note_pass one (the argmax tokens decoded without gradient, then one pass with it) gives greedy's numbers
    (m passes feeding back the argmax tokens): the same note ids, soft tokens, letter mass, loss and gradients; soft
    (feeding back the soft tokens) sends different tokens; evaluation decodes the same notes in every mode."""
    net = tiny(tok, 'sym', 4)
    lora_effect(net)
    net.tau, net.lam, net.beta = .5, .3, .01
    eps = trails(2)
    params = [q for q in net.parameters() if q.requires_grad]
    got = {}
    for mode in ('greedy', 'one', 'soft'):
        net.note_pass = mode
        net.train()
        net.sums = collections.defaultdict(float)
        net.zero_grad()
        loss = net.loss(eps, 'cpu')
        loss.backward()
        _, _, bank = net.run(eps, 'cpu')
        net.eval()
        with torch.no_grad():
            _, _, hard = net.run(eps, 'cpu', need_loss=False)
        got[mode] = dict(loss=float(loss.detach()), let=float(net.sums['let'].detach()), e=bank['e'][0].detach(),
                         ids=bank['ids'][0], grads=[q.grad.clone() for q in params if q.grad is not None],
                         hard=hard['ids'][0])
    g, o, sf = got['greedy'], got['one'], got['soft']
    assert torch.equal(g['ids'], o['ids']) and torch.allclose(g['e'], o['e'], atol=1e-5)
    assert abs(g['loss'] - o['loss']) < 1e-5 and abs(g['let'] - o['let']) < 1e-4
    assert len(g['grads']) == len(o['grads']) > 0
    for a, b in zip(g['grads'], o['grads']):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(g['e'], sf['e'], atol=1e-5)
    assert torch.equal(g['hard'], o['hard']) and torch.equal(g['hard'], sf['hard'])


def test_note_warmup_detaches(tok, tmp_path):
    """The warm-up: with writer_base set the writers write with the adapter off (the base model's notes, the same as
    the whole model with its adapter disabled), without gradient, while the readers still train; train() sets it up to
    detach_until with the entropy terms off and the letter controller only taking the reference gap, then clears it."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.train()
    eps = trails(2)
    net.writer_base = True
    net.sums = collections.defaultdict(float)
    net.zero_grad()
    total, _, bank = net.run(eps, 'cpu')
    assert bank['e'][0].requires_grad is False and net.sums['n_tok'] > 0
    total.backward()
    assert any(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in net.parameters() if q.requires_grad)
    with torch.no_grad():
        _, _, mine = net.run(eps, 'cpu', need_loss=False)
        net.writer_base = False
        with net.model.disable_adapter():
            _, _, plain = net.run(eps, 'cpu', need_loss=False)
    assert torch.equal(mine['ids'][0], plain['ids'][0])
    net.detach_notes = True
    total, _, bank = net.run(eps, 'cpu')
    assert bank['e'][0].requires_grad is False
    net.detach_notes = False
    total, _, bank = net.run(eps, 'cpu')
    assert bank['e'][0].requires_grad is True
    C.train(net, 'cpu', 3, 2, 0, games=('trail',), out=tmp_path, every=1, letters_from=0, lam0=.01, beta_full=True,
            detach_until=2, gap_ref='max', min_steps=100)
    recs = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    devs = {r['step']: r for r in recs if 'dev_nll' in r}
    assert all('lam_next' not in devs[k] for k in (1, 2)) and 'lam_next' in devs[3]
    assert recs[-1]['step'] == 3 and recs[-1]['detached'] is False and recs[-1]['beta'] == C.BETA
    assert net.detach_notes is False and net.writer_base is False and net.lam > 0


def test_sym_hint_and_dev0(tok, tmp_path):
    """--note_hint: one line added to every writer's system prompt, none to the reader's; note_stats' three letter
    shares; dev0: a dev before the first step with no controller update; gap_ref max: the reference gap is the
    largest so far."""
    net = tiny(tok, 'sym', 2)
    ep = trails(1)[0]
    net.hint = 'Write your note to the follower in emoji, symbols and digits only, with no letters.'
    assert f'\n{net.hint}<|im_end|>' in tok.decode(net.stream(ep, 0)['ids'])
    assert net.hint not in tok.decode(net.stream(ep, 1)['ids'])
    ns = C.note_stats(net, 'cpu', trails(2))
    assert 0 <= ns['letters_exact'] <= 1 and 0 <= ns['letters'] <= 1 and 0 <= ns['letters_text'] <= 1
    C.train(net, 'cpu', 2, 2, 0, games=('trail',), out=tmp_path, every=1, letters_from=0, lam0=.01, gap_ref='max',
            dev0=True, min_steps=100)
    recs = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert recs[0]['step'] == 0 and recs[0]['examples'] and 'lam_next' not in recs[0] and 'letters_text' in recs[0]
    later = [r for r in recs if r['step'] > 0 and 'gap' in r]
    assert later[0]['gap_ref'] == pytest.approx(later[0]['gap'], abs=1e-3)
    assert later[1]['gap_ref'] == pytest.approx(max(later[0]['gap'], later[1]['gap']), abs=1e-3)
    assert later[1]['held'] == (later[1]['gap'] >= C.LAM_RULE[0] * later[0]['gap'] - 1e-4)


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


def test_kv_text_crumbs(tok):
    """kv crumbs taken from the writer's own user text: the slots are that text's tokens (all, or the last N), no
    slot tokens are added, re-rotated keys equal keys computed at the shifted positions, and the transplant of crumbs
    of different lengths moves the donor's crumbs to the place."""
    net = tiny(tok, 'kv', 2)
    ep = trails(1)[0]
    for source, n in (('sentence', None), ('last4', 4), ('last1', 1)):
        net.source = source
        st = net.stream(ep, 0)
        assert all(k == -1 for k in st['kind'])
        for s, sg in enumerate(ep['agents'][0]['segments']):
            got = [t for t, p in st['slots'] if p == sg['place']]
            text = tok.decode([st['ids'][t] for t in got])
            assert sg['user'].endswith(text) and (n is None and text == sg['user'] or len(got) == n)
    net.source = 'sentence'
    ids = torch.tensor([st['ids'][:20]])
    with torch.no_grad():
        k0 = net.model(ids, position_ids=torch.arange(20)[None] + 3, use_cache=True).past_key_values
        k1 = net.model(ids, position_ids=torch.arange(20)[None] + 40, use_cache=True).past_key_values
    k0, k1 = [k for k, *_ in k0], [k for k, *_ in k1]
    assert torch.allclose(net._rotate(k0[1], torch.full((1, 20), 37)), k1[1], atol=1e-4)
    eps = trails(3, seed=4)
    with torch.no_grad():
        _, _, bank = net.run(eps, 'cpu', need_loss=False)
        donor = {0: (eps[0]['trail']['visit'][0], 1, 2)}
        moved = C.transplant(bank, donor)
        pa = donor[0][0]
        src = ((bank['place'][1] == 2) & (bank['agent'][1] == 0)).nonzero()[:, 0]
        dst = ((moved['place'][0] == pa) & (moved['agent'][0] == 0)).nonzero()[:, 0]
        assert len(dst) == len(src) and torch.equal(moved['k'][0][0, :, dst], bank['k'][0][1, :, src])
        for rerope, remember in ((False, False), (True, False), (False, True), (True, True)):
            net.rerope, net.remember = rerope, remember
            nll, choice = net.forced(eps, 'cpu', 'on')
            assert all(v is not None and v > 0 for v in nll)
        texts = net.answer(eps, 'cpu', donor=donor)
        assert len(texts) == len(eps)


def test_sym_contrast_negatives(tok):
    """--con_negatives: 0 reads every other episode's notes in the pass as negatives (with 2 episodes the same one
    other as the default 1); more negatives give a larger term, one per follower row, whose gradient reaches every
    episode's notes; shifted(bank, k) gives the k-th next episode's crumbs."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.train()
    net.tau, net.lam, net.beta, net.contrast = 1., 0., 0., 1.

    def con(eps, k):
        net.con_negatives = k
        torch.manual_seed(0)
        net.sums = C.collections.defaultdict(float)
        _, _, bank = net.run(eps, 'cpu')
        return net.sums, bank

    two = trails(2, seed=3)
    a, _ = con(two, 1)
    b, _ = con(two, 0)
    assert a['n_con'] == b['n_con'] == 2 and float(a['con']) == pytest.approx(float(b['con']), rel=1e-5)
    three = trails(3, seed=3)
    one, _ = con(three, 1)
    every, bank = con(three, 0)
    assert one['n_con'] == every['n_con'] == 3 and float(every['con']) > float(one['con']) + 1e-6
    g = torch.autograd.grad(every['con'], bank['e'][0])[0].abs().sum(-1)
    assert all(float(g[b].sum()) > 0 for b in range(3))
    sh = C.shifted(bank, 2)
    assert torch.equal(sh['place'], bank['place'][[2, 0, 1]])
    net.con_negatives = 1


def test_sym_privacy_gate(tok, tmp_path):
    """The stricter privacy gate: gate_held devs in a row with the gap held, at least gate_distinct different notes
    among the dev's examples (logged as distinct) and letters under gate_letters; with gate_held 1 and gate_distinct 0
    it gates at the first held dev under the letter share, as before (the tiny model's notes are all letters)."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.train()

    def run(name, **kw):
        out = tmp_path / name
        out.mkdir()
        C.train(net, 'cpu', 3, 2, 0, games=('trail',), out=out, every=1, letters_from=0, lam0=0.5, gap0=1e-9,
                cut0=1e6, tau0=.3, beta_full=True, min_steps=100, **kw)
        return [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]

    recs = run('default')
    assert all(r['held'] for r in recs) and all(r['letters_text'] > C.PRIV_LETTERS for r in recs)
    assert not any('private' in r for r in recs)
    for r in recs:
        assert r['distinct'] == len({e.split(': ', 1)[1] for e in r['examples']})
    recs = run('one', gate_letters=1.01)
    assert [('private' in r) for r in recs] == [True, False, False]
    recs = run('two', gate_held=2, gate_letters=1.01)
    assert [('private' in r) for r in recs] == [False, True, False]
    recs = run('distinct', gate_held=2, gate_letters=1.01, gate_distinct=99)
    assert not any('private' in r for r in recs)


def test_sym_contrast_and_letter_rule(tok, tmp_path):
    """The contrastive term: one per follower row with notes, its gradient reaching both episodes' notes (its own and
    the next episode's, which it reads instead), none with --contrast 0; the letter weight: 0 up to letters_from,
    then LETTERS[1], times LAM_RULE[1] after each dev whose gap holds LAM_RULE[0] of the last pre-penalty one, else
    times LAM_RULE[2]; privacy only from a dev with the gap held and under PRIV_LETTERS letters."""
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    net.train()
    net.tau, net.lam, net.beta = 1., 0., 0.
    eps = trails(2)
    net.sums = C.collections.defaultdict(float)
    net.run(eps, 'cpu')
    assert net.sums['n_con'] == 0
    net.contrast = 1.
    net.sums = C.collections.defaultdict(float)
    _, _, bank = net.run(eps, 'cpu')
    s = net.sums
    assert s['n_con'] == 2 and float(s['con'].detach()) > 0
    g = torch.autograd.grad(s['con'], bank['e'][0])[0].abs().sum(-1)
    assert all(float(g[b].sum()) > 0 for b in range(2))
    assert torch.isfinite(net.loss(eps, 'cpu'))
    net.contrast = 1.
    C.train(net, 'cpu', 6, 2, 0, games=('trail',), out=tmp_path, every=1, letters_from=2, min_steps=100)
    recs = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert (tmp_path / 'pre.pt').exists() and all('lam_next' not in r for r in recs if r['step'] <= 2)
    gap0 = next(r['gap'] for r in recs if r['step'] == 2)
    lam = C.LETTERS[1]
    keep, up, down = C.LAM_RULE
    private = False
    for r in recs[2:]:
        held = gap0 > 0 and r['gap'] >= keep * gap0
        if gap0 <= 0:                                  # no positive reference yet: the weight stays
            gap0 = r['gap']
        elif not held:
            lam *= down
        elif not private or r['letters_text'] >= C.PRIV_LETTERS:
            lam *= up
        assert r['held'] == held and r['lam_next'] == pytest.approx(lam)
        if 'private' in r:
            assert held and r['letters_text'] < C.PRIV_LETTERS
        private = private or 'private' in r
    assert (tmp_path / 'gate.pt').exists() == any('private' in r for r in recs)
    out = tmp_path / 'cont'
    out.mkdir()
    C.train(net, 'cpu', 6, 2, 0, games=('trail',), out=out, every=1, letters_from=0, lam0=0.5, gap0=1e9,
            lam_rule=(2., .5), lost=3, tau0=.3, beta_full=True, min_steps=100)
    recs = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    assert [r['held'] for r in recs] == [False] * 3 and 'stop' in recs[-1] and not (out / 'pre.pt').exists()
    assert [r['lam_next'] for r in recs] == pytest.approx([.25, .125, .0625])
    out = tmp_path / 'neg'
    out.mkdir()
    C.train(net, 'cpu', 4, 2, 0, games=('trail',), out=out, every=1, letters_from=0, lam0=0.5, gap0=-1e9, cut0=-1e6,
            lost=2, tau0=.3, beta_full=True, min_steps=100)
    recs = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    assert len(recs) == 4 and not any(r['held'] or 'private' in r or 'stop' in r for r in recs)
    assert [r['lam_next'] for r in recs] == pytest.approx([.5] * 4)
    assert net.tau == pytest.approx(.3) and net.beta == C.BETA
    # a fixed reference cut: the gap is cut0 minus the hard dev NLL, whatever the dev's own cut does
    out = tmp_path / 'cut0'
    out.mkdir()
    C.train(net, 'cpu', 2, 2, 0, games=('trail',), out=out, every=1, letters_from=0, lam0=0.5, gap0=1., cut0=1e6,
            lam_rule=(2., .5), tau0=.3, beta_full=True, min_steps=100, lr=1e-5)
    recs = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    assert all(r['held'] and r['gap'] == pytest.approx(1e6 - r['dev_nll'], abs=1e-3) for r in recs)
    assert all(r['gap_now'] == pytest.approx(r['dev_cut'] - r['dev_nll'], abs=1e-3) for r in recs)
    second = 1. if 'private' in recs[0] and recs[1]['letters'] < C.PRIV_LETTERS else 2.
    assert [r['lam_next'] for r in recs] == pytest.approx([1., second])
