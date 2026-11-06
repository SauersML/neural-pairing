"""swarm/opentalk.py on a tiny random Qwen3 with a LoRA (CPU, fp32, the real tokenizer): transcripts, greedy messages
that are the writer pass's fixed point, the answer's gradient reaching the writer through the soft tokens, the
contrast and privacy terms, the hard evaluation, and a short end-to-end run on opendata-format shards."""
import json
from collections import deque
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


def net_of(m=4, base_copy=False, merged=False, two=False):
    net = OT.Talk(tiny(), O.tok(), m=m)
    if two:
        net.read_model = tiny()
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
    assert mass.shape[0] % net.m == 0 and 0 <= float(mass.detach().mean()) <= 1
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
    monkeypatch.setattr(OT, 'build', lambda *a, **k: net_of(m=3, base_copy=k.get('base_copy'), merged=True,
                                                            two=k.get('two', False)))
    monkeypatch.setattr(OT, 'ctrl_ready', lambda gap, gap0, args: True)    # a random model's gap may be <= 0
    out, ckpt = tmp_path / 'out', tmp_path / 'ckpt'
    monkeypatch.setattr(sys, 'argv', ['opentalk.py', '--data', str(tmp_path / 'data'), '--out', str(out), '--ckpt',
                                      str(ckpt), '--steps', '20', '--batch', '2', '--micro', '1', '--bucket', '2',
                                      '--dev_every', '10', '--dev_n', '2', '--dev_bs', '2', '--free', '0', '--ctrl',
                                      '--log_every', '5', '--lam_p', '0.5', '--priv_letters', '1.1', '--readers',
                                      'x/Tiny-1.5'])
    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: tiny(3, lora=False))
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
    assert set(scores) == {'state', 'eval_seen', 'eval_heldout', 'eval_seen_tiny_1_5', 'eval_heldout_tiny_1_5'}
    assert 'tiny_1_5_on' in scores['eval_seen_tiny_1_5']['doc'] and 'base_none' in scores['eval_seen']['doc']
    assert all('base_gap' in r['dev']['_all'] and 'base_none' not in r['dev']['_all'] for r in devs)
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


def test_sqrt_letter_charge_prices_the_last_letters():
    net = net_of(m=32)
    share = lambda k: torch.cat([torch.ones(k), torch.zeros(32 - k)])    # noqa: E731
    assert float(net.letter_charge(share(8))) == pytest.approx(0.25)      # linear: the mean
    net.charge = 'sqrt'
    assert float(net.letter_charge(share(32))) == pytest.approx(1.0, abs=1e-3)
    assert float(net.letter_charge(share(8))) == pytest.approx(0.5, abs=1e-3)
    assert float(net.letter_charge(share(1))) == pytest.approx(0.1768, abs=3e-3)
    two = torch.cat([share(32), share(0)])                                # the mean over messages
    assert float(net.letter_charge(two)) == pytest.approx((1 + OT.CHARGE_EPS ** 0.5) / 2, abs=1e-3)
    zero = torch.zeros(32, requires_grad=True)                            # bounded at no letters
    net.letter_charge(zero).backward()
    assert torch.isfinite(zero.grad).all() and float(zero.grad.max()) < 1
    rng = random.Random(9)                                                # a real batch's mass, end to end
    eps = [episode(rng, i, 'doc') for i in range(2)]
    net2 = net_of(m=4)
    net2.charge = 'sqrt'
    loss, parts = net2.loss(eps, lam=0.5)
    loss.backward()
    assert 0 <= parts['letters_soft'] <= 1


def test_message_stats_counts_letters_and_filler():
    assert OT.is_filler('...\n') and OT.is_filler(' ...') and OT.is_filler('…\n') and OT.is_filler('  \n')
    assert OT.is_filler('…') and OT.is_filler('..') and OT.is_filler('\n')
    assert not OT.is_filler('.') and not OT.is_filler(' the') and not OT.is_filler('1') and not OT.is_filler('. a')
    net, rng = net_of(), random.Random(4)
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'doc', 'chat_alt', 'chat_alt'])]
    res = OT.evaluate(net, eps, bs=2)
    for k in ('letters_per_msg', 'no_letter_msgs', 'filler', 'none_on'):
        assert k in res['doc'] and k in res['_all']
    assert res['doc']['none_on'] == round(res['doc']['none'] - res['doc']['on'], 4)
    assert res['_all']['letters_per_msg'] == pytest.approx(res['_all']['letters'] * net.m, abs=1e-2)
    assert {'on', 'swap', 'none'} <= set(res['_all'])


def scripted_run(tmp_path, monkeypatch, devs, steps, *extra):
    """A short run whose devs return the scripted overall numbers in order (gap, letters, eff_tokens, base_gap)."""
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
    O.write_shards(recs, tmp_path / 'data', 'train', 0, size=40)
    monkeypatch.setattr(OT, 'build', lambda *a, **k: net_of(m=3, base_copy=k.get('base_copy'), merged=True,
                                                            two=k.get('two', False)))
    monkeypatch.setattr(OT, 'ctrl_ready', lambda gap, gap0, args: True)
    script = iter(devs)

    def fake(net, eps, bs=8, base=False, examples=3, readers=(), copies=()):
        gap, letters, eff, read, *bar = next(script)
        fams = {f: dict(gap=gap, gap_se=0.01, none=0.5, on=0.5 - gap, closed=gap) for f in ('doc', 'pair', 'fill', 'chat_alt')}
        for v in fams.values():
            v.update(copy_head_closed=bar[0]) if bar else None
        return dict(_all=dict(gap=gap, letters=letters, eff_tokens=eff, charge=letters ** 0.5, base_gap=read,
                              base_gap_se=0.01, closed=0.0, distinct=100), **fams)
    monkeypatch.setattr(OT, 'evaluate', fake)
    monkeypatch.setattr(sys, 'argv', ['opentalk.py', '--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'out'),
                                      '--ckpt', str(tmp_path / 'ckpt'), '--steps', str(steps), '--batch', '2',
                                      '--micro', '1', '--bucket', '2', '--dev_n', '1', '--dev_bs', '2', '--free', '0',
                                      '--ctrl', '--lam0', '0.5', '--lam_min', '0.5', '--parts', 'train', *extra])
    OT.main()
    return [json.loads(line) for line in open(tmp_path / 'out' / 'log.jsonl')]


def test_a_lost_dev_goes_back_to_the_best_weights(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .7, 100., 0.), (.35, .5, 100., 0.), (.01, .5, 100., 0.),
                                                (.3, .5, 100., 0.)], 6, '--dev_every', '2', '--protect', '1',
                        '--lam_p', '0')
    back = next(r for r in rows if r.get('rollback'))
    assert back['step'] == 4 and back['to'] == 2 and back['why'] == 'gap' and back['lr'] == pytest.approx(5e-5)
    assert [r['step'] for r in rows if r.get('new_best')] == [0, 2]
    assert next(r for r in rows if 'ends_on_best' in r)['ends_on_best'] == 2
    best = load_file(str(tmp_path / 'ckpt_best' / 'adapter' / 'adapter_model.safetensors'))
    end = load_file(str(tmp_path / 'ckpt' / 'adapter' / 'adapter_model.safetensors'))
    assert best.keys() == end.keys() and all(torch.equal(best[k], end[k]) for k in best)
    st = json.loads((tmp_path / 'ckpt' / 'state.json').read_text())
    assert st['rollbacks'] == 1 and st['lr_scale'] == 0.5 and st['best']['step'] == 2


def test_a_collapsed_dev_goes_back_and_skips_the_controller(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .7, 100., 0.), (.4, .01, 3., 0.)], 2, '--dev_every', '2',
                        '--protect', '1', '--lam_p', '0', '--gap_ref', 'anchor', '--ref_letters', '0.05')
    assert next(r for r in rows if r.get('rollback'))['why'] == 'collapsed'
    assert [r['step'] for r in rows if r.get('ctrl')] == [0]             # no controller record at the collapsed dev


def test_the_batches_call_one_early_dev(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .7, 100., 0.)] * 3, 3, '--dev_every', '100', '--protect', '1',
                        '--lam_p', '0', '--watch_steps', '1', '--watch_distinct', '1000000')
    assert len([r for r in rows if r.get('forced_dev')]) == 1             # quiet until the next scheduled dev
    assert [r['step'] for r in rows if isinstance(r.get('dev'), dict)] == [0, 1, 3]   # 3: the run's last, for its best


def test_privacy_starts_only_if_a_frozen_reader_reads(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .01, 100., 0.01), (.3, .01, 100., 0.2)], 2, '--dev_every', '2',
                        '--lam_p', '0.5')
    assert [r['step'] for r in rows if r.get('privacy_not_needed')] == [0]
    assert [r['step'] for r in rows if r.get('privacy_starts')] == [2]


def test_ramp_trouble_and_paired_se():
    from types import SimpleNamespace as NS
    assert OT.ramped(1.0, 0.5, 400, 400, 200) == 0.5 and OT.ramped(1.0, 0.5, 400, 500, 200) == 0.75
    assert OT.ramped(1.0, 0.5, 400, 900, 200) == 1.0 and OT.ramped(1.0, 0.5, None, 500, 200) == 1.0
    assert OT.ramped(1.0, 0.5, 400, 500, 0) == 1.0
    args = NS(collapse_tokens=8.0, gap_floor=0.05, lost_keep=0.5)
    assert OT.dev_trouble(0.3, 0.23, 100., args) == '' and OT.dev_trouble(0.3, 0.23, 7., args) == 'collapsed'
    assert OT.dev_trouble(0.1, 0.23, 100., args) == 'gap' and OT.dev_trouble(0.04, 0.05, 100., args) == 'gap'
    assert OT.dev_trouble(0.04, None, 100., args) == ''
    a, b = {0: (2., 2), 1: (6., 2), 2: (4., 2)}, {0: (0., 2), 1: (0., 2), 2: (0., 2)}
    assert OT.paired_se(a, b) == pytest.approx(0.5774, abs=1e-3) and OT.paired_se({0: (1., 1)}, {0: (1., 1)}) == 0.0


def test_frozen_readers_report_on_and_swapped():
    net, rng = net_of(), random.Random(4)
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'doc', 'chat_alt', 'chat_alt'])]
    res = OT.evaluate(net, eps, bs=2, base='read', readers=[('other', tiny(3, lora=False))])
    for k in ('base_on', 'base_swap', 'base_gap', 'base_gap_se', 'other_on', 'other_swap', 'other_gap', 'other_gap_se'):
        assert k in res['doc'], k
    assert 'base_none' not in res['doc'] and {'base_gap', 'base_gap_se', 'other_gap', 'gap_se'} <= set(res['_all'])
    assert res['doc']['other_gap'] == round(res['doc']['other_swap'] - res['doc']['other_on'], 4)


def test_target_charge_pays_only_above_the_target():
    net = net_of(m=32)
    net.charge, net.letter_target = 'target', 0.25
    share = lambda k: torch.cat([torch.ones(k), torch.zeros(32 - k)])    # noqa: E731
    assert float(net.letter_charge(share(8))) == 0 and float(net.letter_charge(share(4))) == 0
    assert float(net.letter_charge(share(16))) == pytest.approx(0.25)
    two = torch.cat([share(32), share(0)])                               # each message pays for its own letters
    assert float(net.letter_charge(two)) == pytest.approx(0.375)
    below = share(4).clone().requires_grad_(True)                        # below the target nothing pulls
    net.letter_charge(below).backward()
    assert float(below.grad.abs().sum()) == 0


def test_target_falls_only_while_messages_help_and_differ():
    from types import SimpleNamespace as NS
    args = NS(target_end=0.01, lam_min=0.5, lam_max=2.0, lam_rate=0.2, value_bar=0.15, min_msgs=0.5)
    assert OT.target_step(0.4, 1.0, 0.5, False, 100, args) == (0.4, 1.0)
    t, lam = OT.target_step(0.4, 1.0, 0.5, True, 1, args)
    assert t == pytest.approx(0.2) and lam == pytest.approx(1.02)
    assert OT.target_step(0.011, 1.99, 0.9, True, 1, args) == (0.01, 2.0)
    assert OT.target_step(0.4, 0.5, 0.1, True, 1, args)[1] == 0.5
    q = lambda *x: deque(x, maxlen=len(x))                               # noqa: E731
    v, m, d = q(.2, .2), q(.2, .2), q(.9, .9)
    assert OT.target_open(v, m, d, 0.2, args)
    assert not OT.target_open(deque([.2], maxlen=2), m, d, 0.2, args)    # the window is not full yet
    assert not OT.target_open(q(.1, .1), m, d, 0.2, args)                # no real help in training
    assert not OT.target_open(v, m, d, 0.1, args)                        # no real help at the last dev
    assert not OT.target_open(v, m, q(.25, .25), 0.2, args)              # the same messages everywhere
    assert not OT.target_open(v, q(.05, .05), d, 0.2, args)              # help that wrong messages give as well
    dev = dict(fill=dict(closed=.9), doc=dict(closed=.1), pair=dict(closed=.3), _all=dict(closed=.5))
    assert OT.value_outside_fill(dev) == pytest.approx(0.2)
    alive = dict(a=dict(gap=.1, gap_se=.01), b=dict(gap=.02, gap_se=.01), c=dict(gap=.1), _all=dict(gap=1, gap_se=.01))
    assert OT.families_alive(alive, 3) == 1


def test_family_norm_and_contrast_against_none():
    net, rng = net_of(m=3), random.Random(11)
    eps = [episode(rng, i, f, teacher=f != 'fill') for i, f in enumerate(['doc', 'pair', 'fill'])]
    base, _ = net.loss(eps)
    net.family_scale = dict(doc=2.0, pair=2.0, fill=2.0)
    halved, _ = net.loss(eps)
    assert float(halved) == pytest.approx(float(base) / 2, rel=1e-4)
    net.contrast_ref = 'none'

    def no_base():
        raise AssertionError('the hinge reads the adapter\'s own none')
    net.base = no_base
    loss, parts = net.loss(eps, contrast=1.0, target=0.2)
    assert {'value', 'value_nf', 'margin', 'margin_n', 'distinct_msgs'} <= set(parts)
    assert 0 < parts['distinct_msgs'] <= 1
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for n, p in net.model.named_parameters() if 'lora_A' in n) > 0


def test_a_target_run_goes_back_when_the_value_falls_far_from_its_peak():
    from types import SimpleNamespace as NS
    a = NS(collapse_tokens=8.0, min_alive=3, value_keep=0.6, gap_floor=0.1, lost_keep=0.5)
    assert OT.dev_trouble(.3, .2, 100., a, 8, .25, .48) == 'value'
    assert OT.dev_trouble(.3, .2, 100., a, 8, .30, .48) == ''         # keep is 0.6 here
    assert OT.dev_trouble(.3, .2, 100., a, 8, .25, None) == ''
    assert OT.dev_trouble(.3, .2, 100., a, 2, .48, .48) == 'dead'
    assert OT.dev_trouble(.3, .2, 5., a, 8, .48, .48) == 'collapsed'


def test_messages_cannot_carry_later_turns():
    net, rng = net_of(m=3), random.Random(12)
    for f in ('chat_alt', 'doc'):
        ep = episode(rng, 0, f, turns=3)
        later = dict(ep, turns=[dict(t) for t in ep['turns']])
        for t in later['turns'][1:]:
            t['user'] = 'Something else entirely, about twelve kinds of cheese.'
            t['response_ids'] = list(reversed(t['response_ids']))
        one, two = net.decode([ep])[0], net.decode([later])[0]
        assert one[0] == two[0]                                          # turn 0's messages never see turn 1
        b = net.build(ep, ep['turns'][1]['who'], one)
        first_later = min(pos for k, pos, _ in b['ans'] if k >= 1)       # and sit before any later answer
        assert all(pos < first_later for pos, g in b['recv'] if g < len(one[0]))


def target_args(*extra):
    return ('--dev_every', '2', '--protect', '1', '--lam_p', '0', '--letter_charge', 'target', '--contrast_ref', 'none',
            '--family_norm', '1', '--target0', '0.4', '--target_half', '2', '--target_window', '1', '--value_bar', '-10',
            '--min_msgs', '0', '--min_alive', '2', '--bias_k', '2', '--lam0', '1', '--lam_min', '0.5', '--log_every', '1',
            *extra)


def test_letter_levels_halve_from_the_start():
    assert [OT.level_of(x, 1.0) for x in (1.0, .6, .5, .3, .25, .1, .01)] == [0, 0, 1, 1, 2, 3, 6]


def test_a_run_back_at_a_level_uses_that_levels_file_and_keeps_its_pace(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'target_open', lambda *a: False)             # the target stays: all one level
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.), (.6, .4, 100., 0.), (.3, .4, 100., 0.), (.6, .4, 100., 0.)], 6,
                        *target_args())
    assert [r['step'] for r in rows if 'new_level_best' in r] == [0, 2]  # dev 6 only equals the best
    back = next(r for r in rows if r.get('rollback'))
    assert back['why'] == 'value' and back['to'] == 2 and back['half'] == 2   # under 80% of .6, at the same pace
    assert json.loads((tmp_path / 'ckpt_L0' / 'talk.json').read_text())['step'] == 2
    assert not (tmp_path / 'ckpt_L0.tmp').exists()


def test_the_80_percent_rule_reads_the_current_level_not_the_overall_best(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'level_of', lambda target, t0: 0 if target == t0 else 1)
    monkeypatch.setattr(OT, 'target_open', lambda *a: True)
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.), (.45, .3, 100., 0.), (.46, .3, 100., 0.), (.35, .3, 100., 0.)], 6,
                        *target_args())
    assert [(r['step'], r['new_level_best']) for r in rows if 'new_level_best' in r] == [(0, 0), (4, 1)]
    assert [r['step'] for r in rows if r.get('new_best')] == [0]         # .46 is under the overall best
    assert not any(r.get('rollback') and r['step'] < 6 for r in rows)    # .45 fell: not good, no file, no rollback
    back = next(r for r in rows if r.get('rollback'))
    assert back['why'] == 'value' and back['to'] == 4                    # under 80% of .46, not back to level 0's .5
    assert back['half'] == 2


def test_a_level_with_no_good_dev_goes_back_one_level_and_slows(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'level_of', lambda target, t0: 0 if target == t0 else 1)
    monkeypatch.setattr(OT, 'target_open', lambda *a: True)
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.), (.3, .3, 100., 0.), (.3, .3, 100., 0.)], 4, *target_args())
    back = next(r for r in rows if r.get('rollback'))
    assert back['why'] == 'level' and back['to'] == 0 and back['half'] == 4
    st = json.loads((tmp_path / 'ckpt' / 'state.json').read_text())
    assert list(st['levels']) == ['0'] and st['misses'] == 0


def test_a_dead_dev_at_a_new_level_goes_back_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'level_of', lambda target, t0: 0 if target == t0 else 1)
    monkeypatch.setattr(OT, 'target_open', lambda *a: True)
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.), (.0, .01, 100., 0.)], 2, *target_args())
    back = next(r for r in rows if r.get('rollback'))
    assert back['why'] == 'dead' and back['to'] == 0 and back['half'] == 4


def test_the_run_ends_on_the_lowest_letter_level_with_a_good_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'level_of', lambda target, t0: 0 if target == t0 else 1)
    monkeypatch.setattr(OT, 'target_open', lambda *a: True)
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.), (.45, .3, 100., 0.), (.46, .3, 100., 0.)], 4, *target_args())
    end = next(r for r in rows if 'ends_on_level' in r)
    assert end['ends_on_level'] == 1 and end['at'] == 4 and end['best'] == 0
    assert (tmp_path / 'ckpt_L0' / 'adapter').exists() and (tmp_path / 'ckpt_L1' / 'adapter').exists()
    assert json.loads((tmp_path / 'ckpt_best' / 'talk.json').read_text())['step'] == 0


def test_the_final_evaluation_reads_the_lowest_good_level_and_the_overall_best(tmp_path, monkeypatch):
    monkeypatch.setattr(OT, 'level_of', lambda target, t0: 0 if target == t0 else 1)
    monkeypatch.setattr(OT, 'target_open', lambda *a: True)
    real = O.Episodes
    monkeypatch.setattr(O, 'Episodes', lambda path, split, **k: real(path, 'train', **k))
    devs = [(.5, .4, 100., 0.), (.45, .3, 100., 0.), (.46, .3, 100., 0.)] + [(.2, .1, 100., 0.)] * 4
    scripted_run(tmp_path, monkeypatch, devs, 4, *target_args('--parts', 'train,eval', '--eval_n', '2'))
    scores = json.loads((tmp_path / 'out' / 'scores.json').read_text())
    assert {'eval_seen', 'eval_heldout', 'overall_best_eval_seen', 'overall_best_eval_heldout'} <= set(scores)
    assert scores['state']['level'] == '1' and scores['state']['final']['step'] == 4
    assert scores['state']['best']['step'] == 0


def test_a_dev_whose_score_fell_keeps_the_target_shut():
    from types import SimpleNamespace as NS
    a = NS(value_bar=.05, min_msgs=.5)
    full = deque([.5], maxlen=1)
    assert OT.target_open(full, deque([.5], maxlen=1), deque([.9], maxlen=1), .5, a, .4)
    assert not OT.target_open(full, deque([.5], maxlen=1), deque([.9], maxlen=1), .5, a, .4, True)


def test_a_lower_share_counts_as_fell_only_when_on_rose():
    assert not OT.score_fell(.4638, .5029, .1766, .184, .01)             # dev 1200: none improved, on did not rise
    assert OT.score_fell(.1595, .4819, .2632, .1875, .01)                # dev 1000: the messages really collapsed
    assert not OT.score_fell(.4, .5, .2, None, .01)                      # no earlier on to compare with
    assert not OT.score_fell(.495, .5, .3, .2, .01)                      # the share moved within noise


def test_the_gradient_probe_splits_the_writers_gradient(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0.)] * 3, 4, *target_args('--grad_probe', '2', '--target0', '0.01'))
    got = [r for r in rows if 'g_answer' in r]
    assert got and all(r['g_charge'] > 0 and r['g_answer'] > 0 for r in got)
    assert all(-1 <= r['g_cos'] <= 1 and 'lr_writer' in r for r in got)
    assert all(r['g_total'] ** 2 == pytest.approx(r['g_charge'] ** 2 + r['g_answer'] ** 2
                                                  + 2 * r['g_cos'] * r['g_charge'] * r['g_answer'], rel=.02) for r in got)


def test_the_warm_stage_saves_new_bests_and_ends_on_them(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .4, 100., 0.), (.4, .4, 100., 0.), (.2, .4, 100., 0.)], 4,
                        *target_args('--warm', '1', '--warm_reader', '1', '--warm_margin', '-1'))
    assert [r['step'] for r in rows if r.get('new_best')] == [0, 2]
    assert not any(r.get('rollback') for r in rows)                      # warm never rolls back
    assert next(r for r in rows if 'ends_on_level' in r)['at'] == 2


def test_two_adapters_keep_the_jobs_apart(tmp_path, monkeypatch):
    net, rng = net_of(m=3, two=True, merged=True), random.Random(19)
    net.chat, net.warm_reader = True, True
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'chat_alt'])]
    msgs = net.decode(eps)
    none = lambda: net.reader(eps, msgs, mode='none')[0].tolist()                         # noqa: E731
    before, was = none(), net.decode(eps)
    big_adapter(net)                                                     # the writer's adapter moves the messages only
    assert none() == pytest.approx(before)
    assert net.decode(eps) != was
    moved = net.decode(eps)
    with torch.no_grad():
        for n, p in net.read_model.named_parameters():
            if 'lora_B' in n:
                p.add_(0.3)
    assert none() != pytest.approx(before)                               # the reader's moves the reading only
    assert net.decode(eps) == moved
    with net.base():                                                     # the untrained reader is the reader's adapter off
        off = none()
    plain = OT.frozen(tiny(lora=False))
    assert off == pytest.approx(net.reader(eps, msgs, mode='none', lm=plain)[0].tolist(), rel=0.01)
    grads = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)    # noqa: E731
    loss, _ = net.warm_loss(eps)
    loss.backward()
    assert grads(net.model) > 0 and grads(net.read_model) > 0
    net.model.zero_grad()
    net.warm_reader = False
    net.read_model.zero_grad()
    loss, _ = net.warm_loss(eps)
    loss.backward()
    assert grads(net.model) > 0 and grads(net.read_model) == 0           # the copy's loss never reaches the reader
    net.model.zero_grad()
    loss, _ = net.loss(eps)
    loss.backward()
    assert grads(net.model) > 0 and grads(net.read_model) > 0            # KL reaches the writer through the soft tokens
    OT.save(net, tmp_path / 'c')
    assert (tmp_path / 'c' / 'adapter_reader' / 'adapter_model.safetensors').exists()


def test_a_two_adapter_run_resumes_from_a_one_adapter_checkpoint(tmp_path, monkeypatch):
    first = tmp_path / 'one'
    first.mkdir()
    (first / 'ckpt_dir').mkdir()
    one = net_of(m=3, merged=True)
    opt = torch.optim.AdamW([p for p in one.model.parameters() if p.requires_grad], lr=1e-4)
    OT.save(one, first / 'ckpt_dir', dict(step=2, lam=1.0), opt, dict(stage='warm', dev_value=.4, copy_value=.5, target=.4))
    rows = scripted_run(tmp_path, monkeypatch, [(.5, .4, 100., 0., .4), (.5, .4, 100., 0., .4)], 5,
                        *target_args('--two', '1', '--warm', '1', '--warm_reader', '1', '--warm_margin', '1',
                                     '--resume', str(first / 'ckpt_dir')))
    starts = next(r for r in rows if 'resumed' in r)
    assert starts['optimizer'] is False                                  # two groups cannot take a one-group state
    assert next(r for r in rows if r.get('new_best'))['step'] == 2       # the rollback point exists from the first step
    assert any(r.get('warm_done') for r in rows)
    assert (tmp_path / 'ckpt' / 'adapter_reader').exists() and (tmp_path / 'ckpt_best' / 'adapter_reader').exists()


def chat_net(m=4):
    net = net_of(m=m)
    net.chat = True
    return net


def test_chat_format_is_a_plain_chat():
    net, rng = chat_net(), random.Random(13)
    ep = episode(rng, 0, 'pair')
    ep['turns'][0].update(talk=2, who='A')
    ep['turns'][1].update(talk=3, who='B')
    ep['turns'][2].update(talk=2, who='A')
    msgs = [[(OT.Talk.speaker(ep, k, j), [100 + 10 * k + j] * net.m) for j in range(t['talk'])]
            for k, t in enumerate(ep['turns'])]
    b = net.build(ep, 'A', msgs)
    text = net.tok.decode(b['ids'])
    assert 'system' not in text and 'agent' not in text.lower() and 'Your messages' not in text
    assert text.startswith('<|im_start|>user\n' + ep['views']['A'] + '\n\n' + ep['turns'][0]['user'] + '<|im_end|>\n')
    head = net.ids(OT.ANSWER_HEAD)
    flat = [m for turn in msgs for m in turn]
    for pos, g in b['own']:                                              # own messages are assistant turns
        assert b['ids'][pos - len(head):pos] == head and b['ids'][pos:pos + net.m] == flat[g][1]
        assert b['ids'][pos + net.m:pos + net.m + 2] == net.ids(OT.TURN_END)
    for pos, g in b['recv']:                                             # the partner's are user turns
        h = net.ids(OT.USER_HEAD)
        assert b['ids'][pos - len(h):pos] == h and b['ids'][pos:pos + net.m] == flat[g][1]
    for k, pos, n in b['ans']:
        assert b['ids'][pos - len(head):pos] == head and b['ids'][pos:pos + n] == ep['turns'][k]['response_ids']
    y = net.build(ep, 'B', msgs)                                         # a partner's view opens its first user turn
    assert net.tok.decode(y['ids']).startswith('<|im_start|>user\n' + ep['views']['B'] + '\n\n')
    assert net.tok.decode(y['ids']).count(ep['turns'][0]['user']) == 0
    none = net.build(ep, 'A', msgs, mode='none')                         # none: the answerer's own turns alone
    assert not none['own'] and not none['recv'] and len(none['ans']) == 2
    plain = net.ids(f"<|im_start|>user\n{ep['views']['A']}\n\n", cache=False) + \
        net.ids(ep['turns'][0]['user'] + OT.TURN_END, cache=False) + head
    assert none['ids'][:len(plain)] == plain


def test_chat_messages_are_the_writer_pass_fixed_point_and_keep_the_order():
    net, rng = big_adapter(chat_net()), random.Random(14)
    net.bias = 3.0
    eps = [episode(rng, i, f, turns=3 + i % 2) for i, f in enumerate(['doc', 'pair', 'chat_alt'])]
    msgs = net.decode(eps)
    _, mass, fixed, _ = net.writer(eps, msgs)
    assert fixed > 0.99, fixed
    assert net.decode([eps[1]])[0] == msgs[1]
    later = dict(eps[1], turns=[dict(t) for t in eps[1]['turns']])
    for t in later['turns'][1:]:
        t['user'] = 'Something else entirely, about twelve kinds of cheese.'
        t['response_ids'] = list(reversed(t['response_ids']))
    assert net.decode([later])[0][0] == msgs[1][0]
    loss, parts = net.loss(eps, lam=0.5, contrast=1.0, target=0.2)
    loss.backward()
    assert parts['fixed'] > 0.99


def test_copy_messages_take_only_what_the_partner_holds_by_then():
    net, rng = chat_net(m=4), random.Random(15)
    ep = OT.fit(episode(rng, 0, 'pair'), net, 4000)
    for t, talk, who in zip(ep['turns'], (2, 4, 2), 'ABA'):
        t.update(talk=talk, who=who)
    head = net.copy_messages(ep, 'head')
    assert [s for s, _ in head[1]] == ['B', 'A', 'B', 'A']
    assert head[0][0][1] == net.ids(ep['turns'][0]['user'], cache=False)[:4]   # the asker sends its request's head
    b0 = net.held(ep, 'B', 0)
    assert head[0][1] == ('B', b0[:4])
    a1 = net.held(ep, 'A', 1)
    assert head[1][1][1] == a1[:4] and head[1][3][1] == a1[4:8]
    assert not set(net.ids(ep['turns'][2]['user'], cache=False)) - set(net.held(ep, 'A', 2))     # its own request
    later = net.ids(ep['turns'][2]['user'], cache=False)
    assert not any(later[i:i + 4] == w for _, w in sum(head[:2], []) for i in range(len(later) - 3))
    whole = net.copy_messages(ep, 'whole')
    assert whole[0][1][1] == b0 and whole[1][3][1] == []
    long = list(range(1000, 1100))
    picks = OT.best_windows(long, [1050, 1051, 1090], 8, 2)
    assert 1050 in picks[0] and 1051 in picks[0] and 1090 in picks[1] and len(picks[0]) == 8
    assert OT.best_windows([1, 2], [1], 8, 2) == [[1, 2], []]


def test_a_dev_reports_the_copy_rows_and_fit_caps_the_talk():
    net, rng = chat_net(m=3), random.Random(16)
    net.talk_max = 2
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'doc', 'chat_alt', 'chat_alt'])]
    assert all(t['talk'] == 2 for e in eps for t in e['turns'])
    d = OT.evaluate(net, eps, bs=2, copies=('head', 'best'))
    for f in ('doc', 'chat_alt'):
        assert {'copy_head', 'copy_best', 'copy_head_closed', 'copy_best_closed'} <= set(d[f])
    assert OT.value_outside_fill(d, 'copy_head_closed') is not None
    from types import SimpleNamespace as NS
    args = NS(value_bar=0.05, min_msgs=0.5)
    q = lambda *x: deque(x, maxlen=len(x))                               # noqa: E731
    v, m, dd = q(.2, .2), q(.2, .2), q(.9, .9)
    assert OT.target_open(v, m, dd, 0.2, args, copy_value=0.1)
    assert not OT.target_open(v, m, dd, 0.2, args, copy_value=0.2)       # the writer has to beat a plain copy
    assert not OT.target_open(v, m, dd, None, args)                      # shut until a dev has measured it


def test_a_chat_target_run_trains_with_no_letters_pull_while_shut(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .9, 100., 0.)] * 2, 2, '--dev_every', '2', '--protect', '1',
                        '--lam_p', '0', '--letter_charge', 'target', '--contrast_ref', 'none', '--family_norm', '1',
                        '--format', 'chat', '--talk_max', '2', '--target0', '1', '--target_window', '1',
                        '--value_bar', '0.05', '--min_alive', '2', '--bias_k', '0', '--lam0', '1', '--dev_copy', '1',
                        '--symbols_line', '0', '--log_every', '1')
    steps = [r for r in rows if 'target' in r and 'kl' in r]
    assert steps and all(r['target'] == 1.0 for r in steps)              # no copy measured in the fake: dev value decides
    st = json.loads((tmp_path / 'ckpt' / 'talk.json').read_text())
    assert st['bias'] == 0.0 and st['symbols'] is False


def test_a_resumed_part_can_start_with_a_dev_and_never_goes_back_from_it(tmp_path, monkeypatch):
    common = ('--protect', '1', '--lam_p', '0', '--letter_charge', 'target', '--contrast_ref', 'none', '--family_norm', '1',
              '--format', 'chat', '--target0', '1', '--min_alive', '2', '--bias_k', '0', '--lam0', '1', '--dev_every', '2')
    scripted_run(tmp_path, monkeypatch, [(.3, .9, 100., 0.), (.3, .9, 100., 0.)], 2, *common)
    (tmp_path / 'out' / 'log.jsonl').rename(tmp_path / 'first.jsonl')
    rows = scripted_run(tmp_path, monkeypatch, [(.0, .9, 100., 0.), (.3, .9, 100., 0.)], 4, *common, '--resume',
                        str(tmp_path / 'ckpt'), '--dev_first', '1')
    assert [r['step'] for r in rows if isinstance(r.get('dev'), dict)] == [2, 4]
    assert not any(r.get('rollback') for r in rows)                      # dead at its first dev: nothing to go back from
    assert not any(r.get('new_best') for r in rows)                      # .3 only equals the first part's best


def test_warm_targets_are_exactly_m_allowed_tokens_and_the_writer_learns_them():
    net, rng = chat_net(m=4), random.Random(17)
    big_adapter(net)
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'pair', 'chat_alt', 'fill'])]
    msgs = net.copy_messages(eps[1], 'target')
    assert all(len(m) == 4 and all(net.allowed[v] for v in m) for turn in msgs for _, m in turn)
    nl = net.ids('\n')[0]
    short = dict(eps[1], views=dict(A='', B=''))                         # nothing to send: all padding
    first = net.copy_messages(short, 'target')[0]
    assert first[1][1] == [nl] * 4
    best = net.copy_messages(eps[1], 'best')
    assert all(p[1][:len(b[1])] == b[1][:4] for tp, tb in zip(msgs, best) for p, b in zip(tp, tb) if b[1])
    loss, parts = net.warm_loss(eps)
    assert {'ce', 'acc', 'distinct', 'letters'} <= set(parts) and parts['ce'] > 0
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for n, p in net.model.named_parameters() if 'lora_A' in n) > 0
    hs, G, toks = net.own_hidden(eps[:1], [net.copy_messages(eps[0], 'target')])
    assert hs.shape[0] == len(toks) == len(G) * 4
    net.model.zero_grad()
    net.warm_reader = True
    both, parts = net.warm_loss(eps)
    assert 'kl' in parts and 0 <= parts['acc_first'] <= 1 and float(both) > parts['ce']
    both.backward()


def test_dev_closed_shares_measure_against_the_better_none():
    net, rng = chat_net(m=3), random.Random(18)
    eps = [OT.fit(episode(rng, i, f), net, 4000) for i, f in enumerate(['doc', 'doc', 'chat_alt', 'chat_alt'])]
    d = OT.evaluate(net, eps, bs=2, base=True, copies=('head',))
    for f in ('doc', 'chat_alt'):
        ref = min(d[f]['none'], d[f]['base_none'])
        assert d[f]['closed_fair'] == pytest.approx(O.closed_share(ref, d[f]['on']), abs=1e-3)
        assert d[f]['copy_head_fair'] == pytest.approx(O.closed_share(ref, d[f]['copy_head']), abs=1e-3)


def test_the_writer_warm_starts_until_it_nears_the_copy_then_trains_in_full(tmp_path, monkeypatch):
    rows = scripted_run(tmp_path, monkeypatch, [(.3, .9, 100., 0., .5), (.49, .9, 100., 0., .5), (.5, .9, 100., 0., .5)], 4,
                        '--dev_every', '2', '--protect', '1', '--lam_p', '0', '--letter_charge', 'target',
                        '--contrast_ref', 'none', '--family_norm', '1', '--format', 'chat', '--target0', '1',
                        '--min_alive', '2', '--bias_k', '0', '--lam0', '1', '--dev_copy', '1', '--symbols_line', '0',
                        '--warm', '1', '--log_every', '1')
    assert [r['step'] for r in rows if r.get('warm_done')] == [2]         # 0.49 is within 0.02 of the head copy's 0.5
    steps = [r for r in rows if 'stage' in r]
    assert [r['stage'] for r in steps] == ['warm', 'warm', 'full', 'full']
    assert 'ce' in steps[0] and 'ce' not in steps[-1] and 'kl' in steps[-1]
    assert not any(r.get('rollback') for r in rows)
    assert [r['step'] for r in rows if r.get('new_best')] == [0, 2, 4]   # new bests are saved in the warm stage too
    assert json.loads((tmp_path / 'ckpt' / 'state.json').read_text())['stage'] == 'full'
