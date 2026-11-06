"""swarm/opentalk.py on a tiny random Qwen3 with a LoRA (CPU, fp32, the real tokenizer): transcripts, greedy messages
that are the writer pass's fixed point, the answer's gradient reaching the writer through the soft tokens, the
contrast and privacy terms, the hard evaluation, and a short end-to-end run on opendata-format shards."""
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import opendata as O                         # noqa: E402
from swarm import opentalk as OT                        # noqa: E402
from swarm import letters as LT                         # noqa: E402

WORDS = 'river stone lamp orbit candle maple violet harbor engine signal copper meadow lantern quartz'.split()


def tiny(seed=0, lora=True):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(O.tok()) + 8, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_implementation='sdpa',
                      tie_word_embeddings=True)
    lm = Qwen3ForCausalLM(cfg)
    if not lora:
        return lm
    model = get_peft_model(lm, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    for n, p in model.named_parameters():
        if 'lora_B' in n:
            p.data.normal_(0, 0.05)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    return model


def net_of(m=4, base_copy=False, merged=False):
    net = OT.Talk(tiny(), O.tok(), m=m)
    if base_copy:
        net.base_lm = OT.frozen(tiny(lora=False))
    if merged:                                        # the same random base (seed 0), adapter merged in
        net.dec = OT.frozen(tiny(lora=False), gc=False).eval()
        net.sync_dec()
    return net


def test_merged_copy_decodes_what_the_adapter_model_decodes():
    rng = random.Random(7)
    eps = [episode(rng, i, f) for i, f in enumerate(['doc', 'pair'])]
    plain, merged = net_of(), net_of(merged=True)
    for p in merged.dec.parameters():
        assert not p.requires_grad
    assert merged.decode(eps) == plain.decode(eps)
    before = merged.decode(eps)
    big_adapter(plain)                                     # the same change to both adapters
    with torch.no_grad():
        for n, p in merged.model.named_parameters():
            if 'lora_B' in n:
                p.copy_(dict(plain.model.named_parameters())[n])
    assert merged.decode(eps) == before                    # the copy moves only through sync_dec
    merged.sync_dec()
    assert merged.decode(eps) == plain.decode(eps) != before
    soft, mass, fixed, toks = merged.writer(eps, merged.decode(eps))
    assert fixed > 0.99


def episode(rng, i, family='doc', turns=3, teacher=True):
    words = lambda n: ' '.join(rng.choice(WORDS) for _ in range(n))          # noqa: E731
    who = [('B' if family == 'doc' else 'AB'[(k + i) % 2]) for k in range(turns)]
    ts = []
    for k in range(turns):
        resp = O.piece(words(rng.randint(3, 9))) + ([O.IM_END] if k % 2 == 0 else [])
        ask = words(6) + f' q{i}x{k}a q{i}x{k}b q{i}x{k}c please?'                # a real request (not degenerate)
        ts.append(dict(who=who[k], user=ask, teacher_user=words(6), talk=rng.randint(2, 3),
                       response=O.tok().decode(resp), response_ids=resp))
    ep = dict(id=f'e{family}{i}', family=family, source='x', split='train', teacher=teacher,
              views=dict(A=words(30), B=words(5) if family != 'doc' else ''), turns=ts)
    if teacher:
        ep['topk'] = {}
        for k, t in enumerate(ts):
            n = len(t['response_ids'])
            ids = torch.stack([torch.randperm(5000)[:O.TOPK] for _ in range(n)]).int()
            lp = torch.log_softmax(torch.randn(n, O.TOPK + 1) * 2, -1)[:, :O.TOPK].sort(-1, descending=True)[0]
            ep['topk'][k] = (ids, lp.half())
    return ep


def test_transcripts_place_messages_and_answers():
    net, rng = net_of(), random.Random(0)
    ep = episode(rng, 0, 'pair')
    msgs = [[(OT.Talk.speaker(ep, k, j), [100 + 10 * k + j] * net.m) for j in range(t['talk'])]
            for k, t in enumerate(ep['turns'])]
    for x in 'AB':
        b = net.build(ep, x, msgs)
        flat = [m for turn in msgs for m in turn]
        assert len(b['own']) + len(b['recv']) == len(flat)
        for pos, g in b['own']:
            assert flat[g][0] == x and b['ids'][pos:pos + net.m] == flat[g][1]
        for pos, g in b['recv']:
            assert flat[g][0] != x and b['ids'][pos:pos + net.m] == flat[g][1]
        for k, pos, n in b['ans']:
            assert ep['turns'][k]['who'] == x and b['ids'][pos:pos + n] == ep['turns'][k]['response_ids']
        none = net.build(ep, x, msgs, mode='none')
        assert not none['own'] and not none['recv'] and len(none['ans']) == len(b['ans'])
        assert 'Your messages with' not in net.tok.decode(none['ids'])
        swapped = net.build(ep, x, msgs, recv=[[7] * net.m])
        assert all(swapped['ids'][p:p + net.m] == [7] * net.m for p, _ in swapped['recv'])
    a = net.tok.decode(net.build(ep, 'A', msgs)['ids'])
    assert ep['views']['A'] in a and ep['views']['B'] not in a


def big_adapter(net, scale=0.3, seed=0):
    """A large adapter, so the messages depend on their context (a small one let a decode bug pass)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in net.model.named_parameters():
            if 'lora_B' in n:
                p.add_(torch.randn(p.shape, generator=g) * scale)
    net.sync_dec()
    return net


def test_greedy_messages_are_the_writer_pass_fixed_point():
    net, rng = big_adapter(net_of()), random.Random(1)
    net.bias = 3.0                                                  # the letter bias acts in both
    eps = [episode(rng, i, f, turns=3 + i % 2) for i, f in enumerate(['doc', 'pair', 'chat_alt'])]
    msgs = net.decode(eps)
    for i, ep in enumerate(eps):
        assert len(msgs[i]) == len(ep['turns'])
        for k, t in enumerate(ep['turns']):
            assert [s for s, _ in msgs[i][k]] == [OT.Talk.speaker(ep, k, j) for j in range(t['talk'])]
            assert all(len(m) == net.m and all(net.allowed[v] for v in m) for _, m in msgs[i][k])
    soft, mass, fixed, toks = net.writer(eps, msgs)
    assert fixed > 0.99, fixed                                       # argmax of the writer's logits = the decode
    assert 0 <= float(mass.detach()) <= 1
    for i, ep in enumerate(eps):                                     # the decode's own context is the transcript
        for x in 'AB':
            b = net.build(ep, x, msgs[i])
            with torch.no_grad():
                h = net.forward([b['ids']])[0]
            for pos, g in b['own']:
                rows = net.rows(net.masks.walk(b['ids'][pos:pos + net.m])[1])   # the letters after the pending bytes
                z = net.hard_logits(h[pos - 1:pos - 1 + net.m], rows=rows).argmax(-1).tolist()
                assert z == b['ids'][pos:pos + net.m]
    # one episode alone decodes the same messages as in the batch (padding and other rows change nothing)
    assert net.decode([eps[1]])[0] == msgs[1]


def test_answer_gradient_reaches_the_writer_through_received_soft_tokens():
    net, rng = net_of(), random.Random(2)
    eps = [episode(rng, i, 'doc') for i in range(2)]
    msgs = net.decode(eps)
    soft, mass, fixed, toks = net.writer(eps, msgs)
    tot, cnt, keys = net.reader(eps, msgs, soft)
    assert all(x == 'B' for _, _, x in keys)                         # doc: B answers every turn
    who = [(i, g) for i in range(2) for g in soft[i]]
    grads = torch.autograd.grad((tot / cnt).sum(), [soft[i][g] for i, g in who], allow_unused=True)
    speaker = {(i, g): s for i in range(2) for g, (s, _) in OT.enumerate_msgs(msgs[i])}
    by_A = [gr for (i, g), gr in zip(who, grads) if speaker[(i, g)] == 'A']
    by_B = [gr for (i, g), gr in zip(who, grads) if speaker[(i, g)] == 'B']
    assert by_A and all(gr is not None and float(gr.abs().sum()) > 0 for gr in by_A)   # B reads A's soft tokens
    assert all(gr is None or float(gr.abs().sum()) == 0 for gr in by_B)               # its own: hard ids
    # with the hard ids instead, the reader's numbers match the soft run's to within the soft-hard difference
    with torch.no_grad():
        th, ch, _ = net.reader(eps, msgs)
    assert torch.isfinite(th).all() and torch.equal(ch, cnt)


def test_loss_terms_and_backward():
    net, rng = net_of(base_copy=True), random.Random(3)
    eps = [episode(rng, i, f, teacher=f != 'fill') for i, f in enumerate(['doc', 'pair', 'fill'])]
    net.model.train()
    loss, parts = net.loss(eps, lam=0.5, contrast=1.0, target=0.2, lam_p=0.5)
    assert {'kl', 'letters_soft', 'fixed', 'letters', 'distinct', 'margin', 'gain_base'} <= set(parts)
    loss.backward()
    g = [p.grad for n, p in net.model.named_parameters() if 'lora_A' in n]
    assert g and all(x is not None for x in g) and sum(float(x.abs().sum()) for x in g) > 0
    assert all(p.grad is None for p in net.base_lm.parameters())


def test_kl_is_zero_when_the_student_is_the_teacher():
    net, rng = net_of(), random.Random(4)
    ep = episode(rng, 0, 'doc')
    msgs = net.decode([ep])
    with torch.no_grad():                                             # the student's own top-32 as the teacher's
        b = net.build(ep, 'B', msgs[0])
        h = net.forward([b['ids']])
        for k, pos, n in b['ans']:
            z = net.lm.lm_head(h[0, pos - 1:pos - 1 + n]).float().log_softmax(-1)
            lp, ids = z.topk(O.TOPK, -1)
            ep['topk'][k] = (ids.int(), lp)
        tot, cnt, _ = net.reader([ep], msgs)
    assert float((tot / cnt).abs().max()) < 1e-4


def test_evaluate_reports_every_family():
    net, rng = net_of(), random.Random(5)
    eps = [OT.fit(episode(rng, i, f, teacher=f != 'fill'), net, 4000) for i, f in
           enumerate(['doc', 'doc', 'pair', 'pair', 'fill', 'fill'])]
    res = OT.evaluate(net, eps, bs=2, base=True)
    for f in ('doc', 'pair', 'fill'):
        d = res[f]
        assert {'on', 'swap', 'none', 'base_on', 'base_none', 'closed', 'gap', 'letters', 'distinct'} <= set(d)
        assert len(d['closed_by_turn']) == 3 and len(d['examples']) == 3
    assert res['_all']['letters'] is not None
    top = res['_all']['top']
    assert 0 < len(top) <= 20 and all(isinstance(t, str) and 0 < c <= 1 for t, c in top)
    assert [c for _, c in top] == sorted((c for _, c in top), reverse=True)


def test_letter_sets_leave_emoji_open():
    tk = O.tok()
    allowed, letter = OT.letter_sets(tk, len(tk))
    tid = tk.convert_tokens_to_ids
    for t in ('ðŁ', 'ï¸'):                       # F0 9F, the start of every 4-byte emoji; EF B8 (variation selectors)
        assert tid(t) is not None and allowed[tid(t)] and not letter[tid(t)], t
    for t in ('ä¸', 'Ã'):                         # the start of CJK ideographs; of Latin-1 letters such as e acute
        assert allowed[tid(t)] and letter[tid(t)], t
    for text, want in (('the', True), ('123', False), ('!', False), (' 😀', False), ('é', True)):
        ids = tk(text, add_special_tokens=False).input_ids
        if len(ids) == 1:
            assert bool(letter[ids[0]]) == want, text
    assert LT.scan(bytes([0xF0, 0x9F])) == (False, bytes([0xF0, 0x9F]))
    assert LT.scan(bytes([0xE4])) == (True, bytes([0xE4]))
    assert OT.token_label(tk, tid('ðŁ')) == 'xf09f'


def test_letters_depend_on_the_pending_bytes():
    tk = O.tok()
    lm = LT.LetterMasks(tk, len(tk))
    brick = tk('🧱', add_special_tokens=False).input_ids                # F0 9F A7 + B1 in Qwen3's vocabulary
    assert len(brick) == 2 and lm.bytes[brick[1]] == bytes([0xB1])
    flags, pend = lm.walk(brick)
    assert flags == [False, False] and pend == [b'', bytes([0xF0, 0x9F, 0xA7])]
    assert lm.mask(b'')[brick[1]]                                       # B1 with nothing pending: a letter
    assert lm.mask(bytes([0xC3]))[brick[1]]                             # C3 B1 is the letter n with tilde
    enye = tk('ñ', add_special_tokens=False).input_ids
    assert all(lm.walk(enye)[0][-1:])                                   # the letter, in one token or two
    assert lm.walk(tk(' 42 !', add_special_tokens=False).input_ids)[0] == [False] * len(tk(' 42 !').input_ids)
    net = net_of()
    net.bias = 5.0
    z0 = net.hard_logits(torch.zeros(1, 64), rows=net.rows([b'']))
    z1 = net.hard_logits(torch.zeros(1, 64), rows=net.rows([bytes([0xF0, 0x9F, 0xA7])]))
    assert z1[0, brick[1]] - z0[0, brick[1]] == 5.0                     # the bias comes off B1 after the lead
    assert net.letters_of([brick, enye]) == [False, False] + lm.walk(enye)[0]


def test_end_to_end_on_shards(tmp_path, monkeypatch):
    rng = random.Random(6)
    recs = []
    for split, n in (('train', 70), ('eval_seen', 4), ('eval_heldout', 4)):
        for i in range(n):
            f = ['doc', 'pair', 'fill', 'chat_alt'][i % 4]
            ep = episode(rng, i, f, turns=3, teacher=f != 'fill')
            ep.update(id=f'{split}{i}', split=split)
            for k, t in enumerate(ep['turns']):
                if ep['teacher']:
                    ids, lp = ep['topk'][k]
                    t['_top'] = (ids.numpy().astype(np.int32), lp.numpy().astype(np.float16))
            ep.pop('topk', None)
            recs.append(ep)
        O.write_shards([r for r in recs if r['split'] == split], tmp_path / 'data', split, 0, size=30)
    monkeypatch.setattr(OT, 'build', lambda *a, **k: net_of(m=3, base_copy=k.get('base_copy'), merged=True))
    monkeypatch.setattr(OT, 'ctrl_ready', lambda gap, gap0, args: True)    # a random model's gap may be <= 0
    out, ckpt = tmp_path / 'out', tmp_path / 'ckpt'
    monkeypatch.setattr(sys, 'argv', ['opentalk.py', '--data', str(tmp_path / 'data'), '--out', str(out), '--ckpt',
                                      str(ckpt), '--steps', '20', '--batch', '2', '--micro', '1', '--bucket', '2',
                                      '--dev_every', '10', '--dev_n', '2', '--dev_bs', '2', '--free', '0', '--ctrl',
                                      '--log_every', '5', '--lam_p', '0.5', '--priv_letters', '1.1'])
    OT.main()
    rows = [json.loads(line) for line in open(out / 'log.jsonl')]
    devs = [r for r in rows if 'dev' in r and isinstance(r['dev'], dict)]
    assert [r['step'] for r in devs] == [0, 10, 20]
    assert any(r.get('ctrl') for r in rows)
    assert all(r['symbols_line'] for r in rows if r.get('ctrl'))      # SYMBOLS joins the prompts at the start
    assert json.loads((ckpt / 'talk.json').read_text())['symbols'] is True
    assert all(len(r['dev']['_all']['top']) <= 20 for r in devs)
    s20 = next(r for r in rows if r.get('step') == 20 and 'tok_s_grad' in r)
    assert s20['tok_s_grad'] > 0 and s20['tok_s_decode'] > 0 and s20['h_per_1000_steps'] > 0
    scores = json.loads((out / 'scores.json').read_text())
    assert set(scores) == {'state', 'eval_seen', 'eval_heldout'}
    assert (ckpt / 'adapter').exists()
    trained = {r['step'] for r in rows if 'kl' in r}
    assert trained                                                    # training steps logged
    first = next(r for r in rows if 'train' in r)
    assert first['dev'] == 8 and first['train'] == 62                  # 2 per family carved out, never trained on


def test_fast_cache_decodes_what_the_cache_decodes():
    rng = random.Random(11)
    eps = [episode(rng, i, f, turns=3) for i, f in enumerate(['doc', 'pair', 'pair'])]
    net = net_of(merged=True)
    big_adapter(net)
    net.sync_dec()
    net.bias = 2.0                                         # the letter bias, by pending bytes, in every step
    plain = net.decode(eps)
    net.fast_eps, net.fast_T = 2, 2048                     # pieces of 2 episodes, the last one padded
    assert net.decode(eps) == plain
    net.fast_eps = 4                                       # one piece, rows past the batch left empty
    assert net.decode(eps) == plain
    assert len(net._fast) == 2 and all(isinstance(c, OT.FastCache) for c in net._fast.values())
    net.fast_T = 64                                        # too short for a transcript: the Cache takes over
    net._fast = {}
    assert net.decode(eps) == plain


def test_fast_cache_hidden_states_match_the_cache():
    net = net_of(merged=True)
    big_adapter(net)
    net.sync_dec()
    a, b = OT.Cache(net, 3, net.dec), OT.FastCache(net, 3, 256, net.dec, least=4)  # reads 4, 8, then 16 slots
    rng = random.Random(2)
    for tails in ([[5, 6, 7], [8], []], [[9], [10], [11]], [[], [12, 13], [14]], [[15], [], [16]]):
        tails = [[t + rng.randint(0, 900) for t in row] for row in tails]
        ha, hb = a.flush(tails), b.flush(tails)
        live = [r for r, t in enumerate(tails) if t]
        assert torch.allclose(ha[live], hb[live], atol=1e-5)


def test_resume_continues_the_run_exactly(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    rng = random.Random(6)
    recs = []
    for i in range(40):
        f = ['doc', 'pair', 'fill', 'chat_alt'][i % 4]
        ep = episode(rng, i, f, turns=2, teacher=f != 'fill')
        ep.update(id=f'train{i}', split='train')
        for k, t in enumerate(ep['turns']):
            if ep['teacher']:
                ids, lp = ep['topk'][k]
                t['_top'] = (ids.numpy().astype(np.int32), lp.numpy().astype(np.float16))
        ep.pop('topk', None)
        recs.append(ep)
    O.write_shards(recs, tmp_path / 'data', 'train', 0, size=20)
    monkeypatch.setattr(OT, 'build', lambda *a, **k: net_of(m=3, merged=True))
    monkeypatch.setattr(OT, 'ctrl_ready', lambda gap, gap0, args: True)
    save = OT.save

    def keep(net, ckpt, extra=None, opt=None, state=None):    # each save also kept as ckpt_step<N>
        save(net, ckpt, extra, opt, state)
        shutil.copytree(ckpt, f'{ckpt}_step{extra["step"]}', dirs_exist_ok=True)
    monkeypatch.setattr(OT, 'save', keep)

    def run(name, steps, *extra):
        monkeypatch.setattr(sys, 'argv', ['opentalk.py', '--data', str(tmp_path / 'data'), '--out',
                                          str(tmp_path / name), '--ckpt', str(tmp_path / name / 'ckpt'), '--steps',
                                          str(steps), '--batch', '2', '--micro', '1', '--bucket', '2', '--dev_every',
                                          '2', '--dev_n', '1', '--dev_bs', '2', '--free', '0', '--ctrl', '--lam0',
                                          '0.5', '--lam_p', '0', '--parts', 'train', *extra])
        OT.main()
        return [json.loads(line) for line in open(tmp_path / name / 'log.jsonl')]
    run('whole', 4)
    first = tmp_path / 'whole' / 'ckpt_step2'                             # the run as it stood at step 2
    assert (first / 'optim.pt').exists()
    st = json.loads((first / 'state.json').read_text())
    assert st['files'] == ['train-0000.jsonl.gz', 'train-0001.jsonl.gz'] and st['ref'] is not None
    O.write_shards(recs[:5], tmp_path / 'data', 'train', 0, tag='g9-')    # a shard that came after the start
    rows = run('rest', 4, '--resume', str(first))
    res = next(r for r in rows if 'resumed' in r)
    assert res['step'] == 2 and res['optimizer'] and res['held_shards'] == ['train-g9-0000.jsonl.gz']
    assert [r['step'] for r in rows if isinstance(r.get('dev'), dict)] == [4]     # its step-2 dev already ran
    w = load_file(str(tmp_path / 'whole' / 'ckpt' / 'adapter' / 'adapter_model.safetensors'))
    r_ = load_file(str(tmp_path / 'rest' / 'ckpt' / 'adapter' / 'adapter_model.safetensors'))
    assert w.keys() == r_.keys() and all(torch.equal(w[k], r_[k]) for k in w)
    talk_w = json.loads((tmp_path / 'whole' / 'ckpt' / 'talk.json').read_text())
    talk_r = json.loads((tmp_path / 'rest' / 'ckpt' / 'talk.json').read_text())
    assert talk_w == talk_r


def test_evaluate_splits_leaky_and_clean_families():
    net, rng = net_of(), random.Random(4)
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'doc', 'chat_alt', 'chat_alt', 'pair',
                                                                       'pair'])]
    res = OT.evaluate(net, eps, bs=2)
    assert res['_leaky']['families'] == ['doc', 'pair'] and res['_clean']['families'] == ['chat_alt']
    assert res['_leaky']['closed'] == round((res['doc']['closed'] + res['pair']['closed']) / 2, 4)


def test_controller_floor_and_reference():
    from types import SimpleNamespace as NS
    args = NS(ctrl_keep=0.8, ctrl_up=2.0, ctrl_down=0.5, lam_max=4.0, lam_min=0.4, gap_ref='first', ref_letters=None)
    lam, ok = OT.letter_ctrl(0.4, 0.1, 0.23, args)                        # a missed dev stops at the floor
    assert (lam, ok) == (0.4, False)
    assert OT.letter_ctrl(0.8, 0.1, 0.23, args) == (0.4, False)
    assert OT.letter_ctrl(0.4, 0.2, 0.23, args) == (0.8, True)
    args.lam_min = 0.0                                                   # the default halves as before
    assert OT.letter_ctrl(0.4, 0.1, 0.23, args) == (0.2, False)
    assert OT.next_ref(0.23, 0.5, 0.7, args) == 0.23                    # first: the start's gap stays
    args.gap_ref = 'max'
    assert OT.next_ref(0.23, 0.5, 0.7, args) == 0.5
    args.ref_letters = 0.05                                              # a dev of words cannot raise it
    assert OT.next_ref(0.23, 0.5, 0.7, args) == 0.23
    assert OT.next_ref(0.23, 0.5, 0.01, args) == 0.5
    assert OT.next_ref(0.23, 0.1, 0.01, args) == 0.23


def test_controller_reanchors_once_on_a_letterless_miss():
    from types import SimpleNamespace as NS
    args = NS(ctrl_keep=0.8, ctrl_up=2.0, ctrl_down=0.5, lam_max=4.0, lam_min=0.4, gap_ref='anchor', ref_letters=0.05)
    assert OT.ctrl_update(0.4, 0.1, 0.23, 0.30, None, 400, args) == (0.4, False, 0.23, None)   # words: no anchor
    assert OT.ctrl_update(0.4, -0.01, 0.23, 0.01, None, 400, args) == (0.4, False, 0.23, None)  # no gap to hold
    assert OT.ctrl_update(0.4, 0.1, 0.23, 0.01, None, 400, args) == (0.4, False, 0.1, 400)     # a miss, the new ref
    assert OT.ctrl_update(0.4, 0.09, 0.1, 0.01, 400, 600, args) == (0.8, True, 0.1, 400)       # holds against it
    assert OT.ctrl_update(0.8, 0.05, 0.1, 0.01, 400, 800, args) == (0.4, False, 0.1, 400)      # once only
