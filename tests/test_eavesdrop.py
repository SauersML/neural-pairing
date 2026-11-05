"""swarm/eavesdrop.py on tiny random Qwen3 models (the real Qwen3-0.6B tokenizer) on the CPU, with dummy runs whose
messages are letterless vocabulary tokens: a crumbs.py trail run (messages.json) and a dialogue.py run
(transcripts.json, scores.json). The items keep the follower's visited test rooms and the episode's objects as
candidates; dialogue's episodes are rebuilt from their seeds; forced choice and the greedy score find an answer the
model was made to prefer; the untrained reader (crumbs.py's base mode) scores differently from the trained one; and
the whole script writes its JSON with the table.

tiny_lm and tiny_load also stand in for the real models under eavesdrop.py --tiny.
Usage: python -m pytest -q tests/test_eavesdrop.py   (from the repository root)
"""
import importlib.util
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import eavesdrop as E                        # noqa: E402

_TOK = {}
GAMES = ('schedule', 'intersect', 'twenty')             # dialogue games that need no downloaded data
DIALOGUE = importlib.util.find_spec('swarm.dialogue') is not None    # dialogue.py lands separately


def tokenizer():
    if 'tok' not in _TOK:
        from transformers import AutoTokenizer
        _TOK['tok'] = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    return _TOK['tok']


def tiny_model(seed=0):
    """A tiny random Qwen3 with 72 vocabulary rows past the tokenizer (crumbs.py's tok arm needs spare rows)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tokenizer()) + 72, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      attn_implementation='sdpa')
    return Qwen3ForCausalLM(cfg).to(torch.float32)


def tiny_lm():
    return tiny_model().eval(), tokenizer()


def tiny_load(run=None):
    """crumbs.py's sym arm (m = 2) on a tiny model, its LoRA made non-zero."""
    from swarm import crumbs as C
    net = C.build('cpu', 'sym', 2, model=tiny_model(), tok=tokenizer(), layers=(1, 2))
    torch.manual_seed(1)
    for n, p in net.model.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.2)
    return net.eval()


def letterless():
    """Vocabulary ids whose text has no letters in any script."""
    tok = tokenizer()
    ids = sorted(set(tok('0 1 2 3 4 5 6 7 8 9 !? ... ### 123 456 789 :) ;; == ++ -- ** // %% $$ 🙂 🚀 ✨ 🎉',
                         add_special_tokens=False).input_ids))
    assert all(not any(ch.isalpha() for ch in tok.decode([i])) for i in ids)
    return ids


def trail_run(root, n=6, m=4):
    """A dummy crumbs.py run folder: messages.json with n training and n test trail episodes."""
    rng, pool = random.Random(0), letterless()
    eps, msgs = {}, []
    for split in ('train', 'test'):
        r = random.Random(f'intercept-trail-{split}')
        eps[split] = [CG.episode('trail', r, split == 'test') for _ in range(n)]
        for b, ep in enumerate(eps[split]):
            t = ep['trail']
            for sg in ep['agents'][0]['segments']:
                p = sg['place']
                msgs.append(dict(split=split, game='trail', episode=b, writer=0, place=p, room=ep['places'][p],
                                 vocab_ids=[rng.choice(pool) for _ in range(m)], object=t['objects'][p],
                                 note=f'A note left here: {sg["user"]}', follower_visits=p in t['visit']))
    root.mkdir(parents=True, exist_ok=True)
    (root / 'messages.json').write_text(json.dumps(dict(messages=msgs, trail_episodes=eps)))
    return root


def dialogue_run(root, n=4, m=3):
    """A dummy dialogue.py run folder: transcripts.json (n test and n shots episodes per game) and scores.json."""
    rng, pool = random.Random(1), letterless()
    rows = {key: {g: [dict(i=i, ids=[[rng.choice(pool) for _ in range(m)] for _ in range(4)]) for i in range(n)]
                  for g in GAMES} for key in ('test', 'shots')}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'transcripts.json').write_text(json.dumps(rows))
    keys = ('trained', 'base_reads', 'base_closed', 'oracle')
    (root / 'scores.json').write_text(json.dumps({g: dict(n=n, means={k: 0.5 for k in keys},
                                                          forced={k: dict(choice=0.25) for k in keys}) for g in GAMES}))
    return root


def gate_run(root):
    """A dummy crumbs.py run folder holding only gate.pt (tiny_load's weights, its LoRA zeroed)."""
    net = tiny_load()
    w = {k: (torch.zeros_like(v) if 'lora_B' in k else v) for k, v in net.state_dict().items()}
    root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weights=w), root / 'gate.pt')
    return root


def test_items(tmp_path):
    tok = tokenizer()
    train, test, eps = E.trail_items(json.loads((trail_run(tmp_path / 't') / 'messages.json').read_text()), tok, 4)
    assert len(eps) == 4 and len(test) == 4 * CG.N_VISIT and len(train) == 6 * CG.N_ROOMS
    for it in test:
        assert it['answer'] in it['cands'] and len(it['cands']) == CG.N_ROOMS
        assert not any(ch.isalpha() for ch in it['turns'][0][1])
        assert it['english'].startswith('You enter the ') and it['answer'] in it['english']
    if not DIALOGUE:
        return
    dtrain, dtest = E.dialogue_items(dialogue_run(tmp_path / 'd'), tok, 3)
    assert len(dtest) == 3 * len(GAMES) and len(dtrain) == 4 * len(GAMES)
    for it in dtest:
        assert [s for s, _ in it['turns']] == ['B', 'A', 'B', 'A'] and it['answer'] in it['cands']
        assert E.score_text(it, it['answer']) == 1.0
    p = E.prompt(dtest[0], 'messages', dtrain[:2])
    assert p.startswith(E.HEAD) and p.count('Messages:\n') == 3 and p.endswith('Answer:')
    assert dtest[0]['english'] in E.prompt(dtest[0], 'english')


def test_forced_and_greedy_find_a_preferred_answer():
    """A model trained a few steps to answer one object prefers it in forced choice and says it."""
    model, tok = tiny_lm()
    it = dict(source='trail', game='trail', view='', turns=[('explorer', '12 !?')], english='You see a lantern.',
              answer='lantern', cands=['lantern', 'violin', 'kettle'], ep=None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        loss = -E.target_logprobs(model, tok, [(E.prompt(it, 'messages'), E.cont('lantern'))], 'cpu').mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    (right, nll), = E.forced(model, tok, [E.prompt(it, 'messages')], [it], 'cpu')
    assert right == 1.0 and nll < 0.5
    assert E.score_text(it, E.greedy(model, tok, [E.prompt(it, 'messages')], 'cpu', new=4)[0]) == 1.0


def test_cached_candidates_match_full_rows():
    """Scoring candidates from the prompt's repeated cache gives the same log-probs as whole padded rows."""
    model, tok = tiny_lm()
    p = E.prompt(dict(view='You are in the attic.', turns=[('explorer', '12 !?')], english=''), 'messages')
    conts = [E.cont(c) for c in ('lantern', 'violin case', 'number 11', 'cushion and skate')]
    with torch.no_grad():
        full = E.target_logprobs(model, tok, [(p, c) for c in conts], 'cpu')
        cached = E.cand_logprobs(model, tok, p, conts, 'cpu')
    assert torch.allclose(full, cached, atol=1e-4), (full, cached)


def test_untrained_reader_differs():
    net = tiny_load()
    rng = random.Random(3)
    res = E.reader(net, [CG.episode('trail', rng, True) for _ in range(2)], 'cpu')
    assert set(res) == {'trained', 'untrained'} and res['trained']['n'] == 2 * CG.N_VISIT
    assert res['trained']['say_nll'] != res['untrained']['say_nll']


def test_script_writes_its_table(tmp_path, monkeypatch):
    out = tmp_path / 'eaves.json'
    monkeypatch.setattr(sys, 'argv', [
        'eavesdrop.py', '--out', str(out), '--tiny', '--device', 'cpu', '--models', 'a,b', '--sizes', '4,8',
        '--steps', '2', '--n_test', '2', '--n_intercept', '3',
        '--trail', f'tok4={trail_run(tmp_path / "t")},gate=' + str(gate_run(tmp_path / 'g')) + '@gate'] +
        (['--dialogue', f'd7={dialogue_run(tmp_path / "d")}'] if DIALOGUE else []))
    E.main()
    res = json.loads(out.read_text())
    assert set(res['runs']) == {'tok4', 'gate'} | ({'d7'} if DIALOGUE else set())
    gate = res['runs']['gate']
    assert set(gate['reader']) == {'trained', 'untrained'} and set(gate['crack']['messages']) == {'4', '8'}
    assert gate['n_test'] == 2 * CG.N_VISIT and (tmp_path / 'messages_gate.json').exists()
    tok4 = res['runs']['tok4']
    assert set(tok4['reader']) == {'trained', 'untrained'}
    for r in res['runs'].values():
        assert set(r['incontext']) == {'a', 'b'} and set(r['incontext']['a']) == set(E.FORMS)
        assert set(r['crack']) == set(E.FORMS) and set(r['crack']['messages']) == {'4', '8'}
    who = [r['who'] for r in res['table']]
    assert 'tok4: chance' in who and 'tok4: untrained reader' in who
    if DIALOGUE:
        d7 = res['runs']['d7']
        assert set(d7['own']) == set(GAMES) and set(d7['crack']['english']['8']['games']) == set(GAMES)
        assert 'd7/twenty: base_reads (dialogue.py)' in who
    for r in res['table']:
        assert r['choice'] is None or 0 <= r['choice'] <= 1
