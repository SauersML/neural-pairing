"""swarm/crumb_open.py on a tiny random Qwen3 (the real Qwen3-0.6B tokenizer) on the CPU, with a stand-in for
swarm/opendata.py's records and cached teacher rows: records become walker visits and a reader conversation with a
teacher row on every answer token, the KL loss trains and reaches the crumbs (the same with gradient checkpointing),
a teacher equal to the reader gives KL 0, a turn without a teacher gives the cross-entropy, and the evaluation reports
every mode, turn, visit count and group of families; training goes round the records pass after pass."""
import json
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_open as CO                      # noqa: E402
from swarm import crumbs as C                           # noqa: E402

from test_crumbs import tiny                            # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def record(tok, i, family, split='eval_seen'):
    words = ' '.join(f'w{i}x{j}' for j in range(60))
    if family == 'chat_alt':
        views, whos = dict(A='', B=''), 'ABABA'
    else:
        views, whos = dict(A='\n'.join(f'line {j}: {words}' for j in range(4)), B='my own notes'), 'BBB'
    turns = []
    for k, w in enumerate(whos):
        resp = f'answer {k} of record {i}: ' + ' '.join(f'r{k}y{j}' for j in range(6))
        turns.append(dict(who=w, user=f'request {k} from {w}', teacher_user=f'everything {k}', response=resp,
                          response_ids=tok(resp, add_special_tokens=False).input_ids))
    return dict(id=f'r{i}', source='src' + str(i % 2), family=family, split=split, kind='doc', views=views,
                teacher=family != 'fill', turns=turns)


def attach(net, tok, rec, k=32):
    """rec['topk'] as opendata.Episodes(teacher=True) gives it: per teacher turn the model's own top-k (adapter off)."""
    if rec['teacher']:
        rec['topk'] = {}
        for i, t in enumerate(rec['turns']):
            ids = tok(t['teacher_user'], add_special_tokens=False).input_ids + list(t['response_ids'])
            with torch.no_grad(), net.model.disable_adapter():
                z = net.model(input_ids=torch.tensor([ids])).logits[0].float().log_softmax(-1)
            lp, top = z[len(ids) - len(t['response_ids']) - 1:len(ids) - 1].topk(k, -1)
            rec['topk'][i] = (top.int(), lp.half())
    return rec


def stand_in(recs):
    def episodes(path, split, seed=0, teacher=True):
        return iter([r for r in recs if r['split'] == split])
    return types.SimpleNamespace(Episodes=episodes)


def test_episode_sides_and_teacher_rows(tok, monkeypatch):
    net = tiny(tok, 'kv', 2)
    doc, chat = attach(net, tok, record(tok, 0, 'doc')), attach(net, tok, record(tok, 1, 'chat_alt'))
    monkeypatch.setattr(CO, 'OD', stand_in([doc, chat]))
    ep = CO.episode(net, doc, 64, 3)
    walker, reader = ep['agents']
    assert ep['open']['visits'] == len(walker['segments']) == 3 and all(s['say'] is None for s in walker['segments'])
    assert [s['user'].split('\n')[-1] for s in reader['segments']] == ['request 0 from B', 'request 1 from B',
                                                                        'request 2 from B']
    assert reader['segments'][0]['user'].startswith('my own notes')
    st = net.stream(ep, 1)
    n_say = sum(len(t['response_ids']) for t in doc['turns'])
    assert sum(r >= 0 for r in st['tk']) == n_say == len(st['tk_ids'])
    assert st['tk_ids'].dtype == torch.long and torch.equal(st['tk_ids'][:3], doc['topk'][0][0][:3].long())
    assert sum(st['target']) == n_say + 3                    # each answer's closing <|im_end|> keeps cross-entropy
    assert ep['open']['tokens'] == len(net.stream(ep, 0)['ids']) + len(st['ids'])
    # chat_alt: the walker's material is A's turns (request and answer), the reader answers B's turns
    ep2 = CO.episode(net, chat, 400, 8)
    text = '\n'.join(s['user'] for s in ep2['agents'][0]['segments'])
    assert 'request 0 from A' in text and 'answer 2 of record 1' in text and 'request 1 from B' not in text
    assert len(ep2['agents'][1]['segments']) == 2
    # answers that end with the teacher's own <|im_end|>: that token keeps its teacher row
    ended = record(tok, 7, 'doc')
    for t in ended['turns']:
        t['response_ids'] = t['response_ids'] + [net.end]
    ep3 = CO.episode(net, attach(net, tok, ended), 64, 3)
    st3 = net.stream(ep3, 1)
    n_end = sum(len(t['response_ids']) for t in ended['turns'])
    assert sum(r >= 0 for r in st3['tk']) == n_end == len(st3['tk_ids']) == sum(st3['target'])
    assert torch.equal(st3['tk_ids'][len(ended['turns'][0]['response_ids']) - 1], ended['topk'][0][0][-1].long())
    # a teacher turn without its cached row is skipped
    del doc['topk'][1]
    assert CO.episode(net, doc, 64, 3) is None


def test_kl_trains_and_matches(tok, monkeypatch):
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    recs = [attach(net, tok, record(tok, i, 'doc')) for i in range(2)]
    eps = [CO.episode(net, r, 64, 3) for r in recs]
    loss = net.loss(eps, 'cpu')
    loss.backward()
    assert torch.isfinite(loss) and net.crumb.grad is not None and net.crumb.grad.abs().sum() > 0
    net.zero_grad(set_to_none=True)
    # a teacher equal to the reader itself (crumbs on, its whole distribution as the top-k) gives KL 0
    with torch.no_grad():
        got = CO.kl_pass(net, eps, 'cpu', ('on', 'cut'))
        _, _, bank = net.run(eps, 'cpu', need_loss=False)
        for b, ep in enumerate(eps):
            x = net._pad([net.stream(ep, 1)], 'cpu')
            net.grab = True
            net._forward(torch.tensor([b]), x, 1, bank, 1, False)
            h, net.grab, net.last = net.last, False, None
            sel = x['target'][:, 1:].nonzero()
            z = net.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float().log_softmax(-1)
            z = z[x['tk'][sel[:, 0], sel[:, 1] + 1] >= 0]
            k0 = 0
            for s in ep['agents'][1]['segments']:
                n = len(s['say_ids'])
                lp, ids = z[k0:k0 + n].topk(z.shape[-1], -1)
                s['teacher'] = (ids, lp)
                k0 += n
        same = CO.kl_pass(net, eps, 'cpu', ('on', 'cut'))
    for d in same['on']:
        for total, n in d.values():
            assert n > 0 and abs(total / n) < 1e-4
    assert all(v[0] > 0 for d in got['cut'] for v in d.values())
    # a turn without a teacher (fill-in) is the cross-entropy of the real text
    fill = record(tok, 5, 'fill')
    ep = CO.episode(net, fill, 64, 3)
    with torch.no_grad():
        kl = CO.kl_pass(net, [ep], 'cpu', ('on',))['on'][0]
        _, _, bank = net.run([ep], 'cpu', need_loss=False)
        st = net.stream(ep, 1)
        lps = net._say_logprobs(torch.tensor([0]), [st], 1, bank, True, 'cpu')[0]
        x = net._pad([st], 'cpu')
        net.grab = True
        net._forward(torch.tensor([0]), x, 1, bank, 1, False)
        h, net.grab, net.last = net.last, False, None
        sel = x['target'][:, 1:].nonzero()
        z = net.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float().log_softmax(-1)
        end = -z.gather(1, x['ids'][sel[:, 0], sel[:, 1] + 1][:, None])[:, 0]
        seg = x['seg'][sel[:, 0], sel[:, 1] + 1]
        last = x['ids'][sel[:, 0], sel[:, 1] + 1] == net.end
    for s, (total, n) in kl.items():
        # _say_logprobs sums the say tokens and the closing <|im_end|>; the KL leaves <|im_end|> out
        assert n == len(ep['agents'][1]['segments'][s]['say_ids'])
        assert abs(total - (-lps[s] - float(end[(seg == s) & last].sum()))) < 1e-3


def test_checkpointing_matches(tok):
    """The kv arm with gradient checkpointing: the same loss and crumb gradient as without."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    got = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                          attn_implementation='sdpa')
        net = C.build('cpu', 'kv', 2, model=Qwen3ForCausalLM(cfg).to(torch.float32), tok=tok, layers=(1, 2),
                      ckpt=ckpt)
        net.train()
        recs = [attach(net, tok, record(tok, i, 'doc')) for i in range(2)]
        eps = [CO.episode(net, r, 64, 3) for r in recs]
        loss = net.loss(eps, 'cpu')
        loss.backward()
        lora = next(p for n, p in net.model.named_parameters() if 'lora_A' in n and 'layers.0.' in n)
        got.append((float(loss.detach()), net.crumb.grad.clone(), lora.grad.clone()))
    (l0, g0, a0), (l1, g1, a1) = got
    assert abs(l0 - l1) < 1e-5 and g0.abs().sum() > 0
    assert torch.allclose(g0, g1, atol=1e-6, rtol=1e-4) and torch.allclose(a0, a1, atol=1e-6, rtol=1e-4)


def test_evaluate_open_reports(tok, monkeypatch):
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    recs = [attach(net, tok, record(tok, i, f, s)) for i, (f, s) in enumerate([
        ('doc', 'eval_seen'), ('chat_alt', 'eval_seen'), ('doc', 'eval_seen'), ('fill', 'eval_heldout'),
        ('doc', 'eval_heldout')])]
    monkeypatch.setattr(CO, 'OD', stand_in(recs))
    res = CO.evaluate_open(net, 'cpu', 'unused', 10, 64, 4, bs=3)
    seen = res['eval_seen']
    assert seen['episodes'] == 3 and res['eval_heldout']['episodes'] == 2
    for k in ('on', 'cut', 'transplant', 'zero_on', 'zero_cut', 'closed_on', 'closed_transplant', 'closed_zero'):
        assert k in seen and seen[k] is not None
    assert seen['tokens'] > 0 and set(seen['by_turn']) == {1, 2, 3}
    assert {1, 'all'} <= set(seen['by_visit']) <= set(CO.VISITS_AT) | {'all'}
    assert abs(seen['by_visit']['all']['on'] - seen['on']) < 1e-9
    assert set(seen['by_source']) == {'src0', 'src1'} and set(seen['by_family']) == {'doc', 'chat_alt'}
    # by_group: the families pooled (doc simulated, chat_alt real), each as summarize() gives it
    groups = seen['by_group']
    assert set(groups) == {'real', 'simulated'}
    assert [groups[g]['episodes'] for g in ('real', 'simulated')] == [1, 2]
    from swarm import opendata                         # the two groups split opendata's families
    real, sim = set(CO.GROUPS['real']), set(CO.GROUPS['simulated'])
    assert real | sim == set(opendata.MIX) and not real & sim
    for g, f in (('simulated', 'doc'), ('real', 'chat_alt')):
        assert abs(groups[g]['closed_on'] - seen['by_family'][f]['closed_on']) < 1e-9
        assert set(groups[g]['by_turn']) and 'all' in groups[g]['by_visit'] and 'by_source' not in groups[g]
    json.dumps(res)
    one = CO.score_batch(net, [CO.episode(net, recs[0], 64, 4)], 'cpu', zero=False)[0]
    assert one['visits'] == 4 and set(one['by_visit']) == {1, 2}
    assert one['turns']['on'] != one['turns']['cut']
    got = CO.summarize([one])
    assert set(got['by_visit_same4']) == {1, 2, 4, 'all'}
    assert abs(got['by_visit'][4]['on'] - got['on']) < 1e-9 and abs(got['by_visit'][1]['cut'] - got['cut']) < 1e-9
    # dev: the KL with the crumbs on and cut over the dev episodes
    on, more = CO.dev_kl(net, [CO.episode(net, r, 64, 4) for r in recs[:3]], 'cpu')
    assert abs(on - seen['on']) < 1e-4 and abs(more['dev_cut'] - seen['cut']) < 1e-3


def test_passes_go_round():
    """passes(): the training records pass after pass, each new pass in its own order, the skipped ids and the ones
    that do not fit left out of every pass; a pass that gives nothing ends it."""
    recs = [dict(id=f'r{i}') for i in range(5)]
    first = iter(recs)
    next(first)                                         # the dev drew r0 already
    order = lambda n: iter(recs[::-1] if n % 2 == 0 else recs)
    new = []
    got = CO.passes(first, order, {'r0'}, lambda r: None if r['id'] == 'r2' else r['id'], new.append)
    assert [next(got) for _ in range(9)] == ['r1', 'r3', 'r4', 'r4', 'r3', 'r1', 'r1', 'r3', 'r4']
    assert new == [2, 3]
    assert list(CO.passes(iter(recs), order, {'r0'}, lambda r: None)) == []


def test_rerope_reaches_the_reader(tok):
    """--rerope: the slot keys re-rotated next to the reader change what it reads (on differs from the walker's own
    positions; cut is the same) and the loss still reaches the slots."""
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    recs = [attach(net, tok, record(tok, i, 'doc')) for i in range(2)]
    eps = [CO.episode(net, r, 64, 3) for r in recs]
    got = {}
    for rerope in (False, True):
        net.rerope = rerope
        with torch.no_grad():
            got[rerope] = CO.kl_pass(net, eps, 'cpu', ('on', 'cut'))
    tot = lambda g: sum(a for d in g for a, _ in d.values())
    assert abs(tot(got[False]['cut']) - tot(got[True]['cut'])) < 1e-6
    assert abs(tot(got[False]['on']) - tot(got[True]['on'])) > 1e-3 * tot(got[False]['on'])
    loss = net.loss(eps, 'cpu')
    loss.backward()
    assert net.crumb.grad is not None and net.crumb.grad.abs().sum() > 0


def test_even_and_mix_sources(tok):
    """--source evenN keeps N evenly spaced tokens of each walker part (the positions crumb_select's even selection
    keeps of the full copy) and adds no tokens; mixN keeps the same and adds the m slot tokens after each visit; both
    train (the loss reaches the adapter, and for mix the slots), and the reader reads fewer crumbs than the full copy
    (on differs, cut is the same)."""
    from swarm import crumb_select as CS
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    recs = [attach(net, tok, record(tok, i, 'doc')) for i in range(2)]
    eps = [CO.episode(net, r, 64, 3) for r in recs]
    net.source = 'sentence'
    full = [net.stream(ep, 0) for ep in eps]
    for ep, st in zip(eps, full):
        runs = CS.runs_of(net, ep)
        assert len(runs) == 3 and all(b - a > 8 for a, b in runs)
        keep = CS.even_keep(runs, len(st['slots']), 8)
        want = [t for (t, _), k in zip(st['slots'], keep.tolist()) if k]
        net.source = 'even8'
        ev = net.stream(ep, 0)
        assert [t for t, _ in ev['slots']] == want and ev['ids'] == st['ids'] and set(ev['kind']) == {-1}
        net.source = 'mix8'
        mx = net.stream(ep, 0)
        slot_tokens = [t for t, k in enumerate(mx['kind']) if k >= 0]
        assert len(slot_tokens) == 3 * net.m and len(mx['ids']) == len(st['ids']) + 3 * net.m
        assert len(mx['slots']) == 3 * (8 + net.m) and {t for t, _ in mx['slots']} >= set(slot_tokens)
        net.source = 'sentence'
    got = {}
    for source in ('sentence', 'even8', 'mix8'):
        net.source = source
        with torch.no_grad():
            got[source] = CO.kl_pass(net, eps, 'cpu', ('on', 'cut'))
        net.zero_grad(set_to_none=True)
        loss = net.loss(eps, 'cpu')
        loss.backward()
        lora = [p.grad for n, p in net.named_parameters() if 'lora_' in n and p.grad is not None]
        assert torch.isfinite(loss) and lora and sum(float(g.abs().sum()) for g in lora) > 0
        crumb = net.crumb.grad
        assert (crumb is not None and crumb.abs().sum() > 0) == (source == 'mix8')
    tot = lambda g: sum(a for d in g for a, _ in d.values())
    assert abs(tot(got['sentence']['cut']) - tot(got['even8']['cut'])) < 1e-5
    assert abs(tot(got['sentence']['on']) - tot(got['even8']['on'])) > 1e-4 * tot(got['sentence']['on'])
    assert C.kv_source('mix48') == 'mix48' and C.kv_source('even64') == 'even64'
    for bad in ('even', 'mix0', 'evens', 'last-1'):
        with pytest.raises(ValueError):
            C.kv_source(bad)
