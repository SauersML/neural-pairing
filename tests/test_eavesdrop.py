"""swarm/eavesdrop.py on tiny random Qwen3 models (the real Qwen3-0.6B tokenizer) on the CPU, with crumbs.py's tok arm:
the intercepted messages are the writers' alphabet ids in the rooms the follower visits; the untrained reader (base
mode) reads the same messages with the adapter off and scores differently from the trained one; forced choice puts the
gold object first when the model is made to prefer it; the vector form puts A's rows in the message slots; and the
whole script (two runs, every part) writes its JSON with the table.

tiny_net and tiny_lm also stand in for the real models under eavesdrop.py --tiny.
Usage: python -m pytest -q tests/test_eavesdrop.py   (from the repository root)
"""
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import eavesdrop as E                        # noqa: E402

_TOK = {}


def tokenizer():
    if 'tok' not in _TOK:
        from transformers import AutoTokenizer
        _TOK['tok'] = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    return _TOK['tok']


def tiny_model(seed=0):
    """A tiny random Qwen3 with 72 vocabulary rows past the tokenizer (room for the 64-token alphabet)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tokenizer()) + 72, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      attn_implementation='sdpa')
    return Qwen3ForCausalLM(cfg).to(torch.float32)


def tiny_lm():
    return tiny_model().eval(), tokenizer()


def tiny_net(run=None):
    """crumbs.py's tok arm (m = 2) on a tiny model, its LoRA made non-zero."""
    from swarm import crumbs as C
    net = C.build('cpu', 'tok', 2, model=tiny_model(), tok=tokenizer(), layers=(1, 2))
    torch.manual_seed(1)
    for n, p in net.model.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.2)
    return net.eval()


def trails(n, seed=0):
    rng = random.Random(seed)
    return [CG.episode('trail', rng, True) for _ in range(n)]


def test_untrained_reader_reads_the_same_messages():
    """Base mode reads the trained writers' bank with the adapter off: the same messages, other scores."""
    net = tiny_net()
    eps = trails(3)
    with torch.no_grad():
        res = E.reader(net, eps, 'cpu')
        bank = net.run(eps, 'cpu', need_loss=False)[2]
        with net.model.disable_adapter():
            plain = net.run(eps, 'cpu', need_loss=False)[2]
    assert set(res) == {'trained', 'untrained'} and res['trained']['n_rooms'] == 3 * CG.N_VISIT
    assert res['trained']['say_nll'] != res['untrained']['say_nll']
    assert not torch.equal(bank['y'][0], plain['y'][0]), 'the writers\' messages come from the trained adapter'


def test_choose_and_name_find_a_preferred_object():
    """A model trained a few steps to answer one object for every message prefers it in forced choice."""
    model, tok = tiny_lm()
    msgs = [dict(tokens=[1, 2], object='lantern', objects=['lantern', 'violin', 'kettle'], note='You see a lantern.')]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ids, vec = E.encode(tok, E.prompt(msgs[0], 'alien'))
    a = tok(E.answer('lantern'), add_special_tokens=False).input_ids
    for _ in range(100):
        loss = -E.target_logprobs(model, [(ids + a, vec + [-1] * len(a), len(a))], tok.pad_token_id, 'cpu').mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    items = [(E.prompt(m, 'alien'), m['object'], m['objects']) for m in msgs]
    (right, nll), = E.choose(model, tok, items, 'cpu')
    assert right == 1.0 and nll < 0.5
    assert E.name(model, tok, items, 'cpu') == [1.0]


def test_vector_slots_hold_A_rows():
    model, tok = tiny_lm()
    A = torch.randn(64, model.config.hidden_size)
    ids, vec = E.encode(tok, E.prompt(dict(tokens=[5, 9, 5]), 'vectors'))
    assert sorted(v for v in vec if v >= 0) == [5, 5, 9]
    emb = E.embed(model, torch.tensor([ids]), torch.tensor([vec]), A)[0]
    slots = [i for i, v in enumerate(vec) if v >= 0]
    assert torch.allclose(emb[slots], A[[5, 9, 5]])
    plain = model.get_input_embeddings()(torch.tensor([ids]))[0]
    other = [i for i, v in enumerate(vec) if v < 0]
    assert torch.equal(emb[other], plain[other])


def test_intercept_matches_rooms():
    net = tiny_net()
    eps = trails(4)
    got = E.intercept(net, eps, 'cpu', 'test', rooms='visited')
    assert len(got) == 4 * CG.N_VISIT
    for m in got:
        ep = eps[m['episode']]
        assert m['room'] in ep['trail']['visit'] and m['object'] == ep['trail']['objects'][m['room']]
        assert len(m['tokens']) == net.m and m['note'].startswith(f'You enter the {ep["places"][m["room"]]}.')


def test_script_writes_its_table(tmp_path, monkeypatch):
    out = tmp_path / 'eaves.json'
    monkeypatch.setattr(sys, 'argv', ['eavesdrop.py', '--runs', 'adv=x,noadv=y', '--out', str(out), '--tiny',
                                      '--device', 'cpu', '--models', 'a,b', '--sizes', '4,8', '--steps', '3',
                                      '--n_reader', '4', '--n_train', '6'])
    E.main()
    res = json.loads(out.read_text())
    assert set(res['runs']) == {'adv', 'noadv'}
    adv = res['runs']['adv']
    assert set(adv['reader']) == {'trained', 'untrained'} and set(adv['incontext']) == {'a', 'b'}
    assert set(adv['crack']) == set(E.FORMS) and set(adv['crack']['alien']) == {'4', '8'}
    assert 'incontext' not in res['runs']['noadv'] and 'crack' in res['runs']['noadv']
    who = [r['who'] for r in res['table']]
    assert who[0] == 'chance' and 'adv: trained reader' in who and 'noadv: untrained reader' in who
    for r in res['table']:
        assert r['choice'] is None or 0 <= r['choice'] <= 1
