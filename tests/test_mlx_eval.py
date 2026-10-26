"""The MLX evaluator (mlx_eval.py) against the torch pair on tiny random models of both architectures, with LoRA and
open bridges: logits, gates, attention records, greedy generation and evaluate.run_episodes agree under link, cut and
shuffle; evalfree's matched-donor records and payload_ctl's table agree; code scoring runs real subprocesses under the
MLX backend; and every command line (evalfree, transcripts, readmap, payload) runs on an exported checkpoint.
"""
import gc
import json
import sys

import pytest

mx = pytest.importorskip('mlx.core')
import numpy as np
from peft import LoraConfig, get_peft_model
import torch

import evaluate as E
import evalfree as F
import link
import mlx_eval as V
import mlx_pair as M
import payload_ctl
import play_tasks
import session as S


@pytest.fixture(autouse=True)
def maxnew(monkeypatch):
    monkeypatch.setenv('NL_MAXNEW', '3')


@pytest.fixture(scope='module')
def pairs():
    """The torch pair, the MLX pair with the same weights (mp) and its evaluator (vp), and a batch of three duplex and
    three dialogue episodes."""
    mx.random.seed(17)
    mx.set_memory_limit(2 * 2**30)
    mx.set_cache_limit(16 * 2**20)
    models, toks, mm = {}, {}, {}
    for x in M.SIDES:
        toks[x], m = link.load(x)
        mm[x] = M.make_model(m.config.to_dict())
        mm[x].load_weights([(k, M.from_torch(v)) for k, v in m.state_dict().items()
                            if k != 'lm_head.weight' or not m.config.tie_word_embeddings])
        M.add_lora(mm[x], 4)
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=list(M.TARGETS)))
        for name, mod in m.get_base_model().named_modules():
            if hasattr(mod, 'lora_A'):
                torch.nn.init.normal_(mod.lora_B['default'].weight, std=.006)
                ml = dict(mm[x].named_modules())[name]
                ml.lora_A = M.from_torch(mod.lora_A['default'].weight)
                ml.lora_B = M.from_torch(mod.lora_B['default'].weight)
        models[x] = m.eval()
    xb, sizes = (1, 2, 4), {x: [.3, .6, .9] for x in M.SIDES}
    pair = link.Pair(models, toks, xb, sizes, heads=4, head_dim=32, gate_norm=True)
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.uniform_(-.6, .6)
            side.null.normal_(0, .3)
            side.age.normal_(0, .4)
            for g in side.gate:
                g.weight.normal_(0, .08)
                g.bias.normal_(0, .2)
    mp = M.Pair(mm, xb, sizes, heads=4, head_dim=32, gate_norm=True)
    for x in M.SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in pair.bridge[x].state_dict().items()])
    mx.eval(mp.parameters())
    system = {x: 'You are ' + x + '.' for x in M.SIDES}
    duplex = [dict(game='plain', meta={}, mode='duplex', system=system,
                   task={'quill': 'Remember the number 42.' if i == 0 else 'Blue.',
                         'ember': 'What is the number?' if i == 1 else 'Name a city in Italy.'},
                   reply={'quill': ('Yes.', .2), 'ember': ('42.', 1.)}) for i in range(3)]
    dialogue = [dict(game='chat', meta={'sender': 'quill'}, mode='dialogue', system=system,
                     task={'quill': 'Ask a question.' if i else 'Which number?', 'ember': None},
                     turns=[('quill', 'Blue?', .4), ('ember', 'Green.', 1.), ('quill', 'Okay.', .6)],
                     **({'link_off_after': 1} if i == 1 else {})) for i in range(3)]
    b = S.collate([S.build(e, toks) for e in duplex + dialogue], {x: toks[x].pad_token_id for x in M.SIDES})
    return pair, mp, V.Pair(mp, toks), toks, b, dict(duplex=duplex, dialogue=dialogue)


@pytest.mark.parametrize('mode', [None, 'zero', 'shuffle'])
@torch.no_grad()
def test_joint_and_greedy_parity(pairs, mode):
    pair, _, vp, _, b, eps = pairs
    h, g = pair.joint(b, mode)
    mh, mg = vp.joint(b, mode)
    for x in M.SIDES:
        tl = pair.token_logits(x, h[x], b[x]['idx']).numpy()
        ml = vp.token_logits(x, mh[x], b[x]['idx']).numpy()
        np.testing.assert_allclose(ml, tl, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(mg[x], g[x], atol=2e-6, rtol=2e-5)
    for label, es in eps.items():
        key = 'text' if label == 'duplex' else 'turns'
        a = S.generate(pair, es, mode=mode, max_turn=3, keep_tokens=True)
        z = S.generate(vp, es, mode=mode, max_turn=3, keep_tokens=True)
        for aa, zz in zip(a, z):
            assert aa['tokens'] == zz['tokens'], (label, aa['tokens'], zz['tokens'])
            assert aa['sampled'] == zz['sampled']
            assert aa[key] == zz[key]
        a = E.run_episodes(pair, es, mode=mode, bs=3)
        z = E.run_episodes(vp, es, mode=mode, bs=3)
        assert [r['ok'] for r in a] == [r['ok'] for r in z]
        assert [r[key] for r in a] == [r[key] for r in z]


@pytest.mark.parametrize('mode', [None, 'shuffle'])
@torch.no_grad()
def test_attention_records_parity(pairs, mode):
    pair, _, vp, _, b, _ = pairs
    pair.record, vp.record = {}, {}
    try:
        pair.joint(b, mode)
        vp.joint(b, mode)
        for key in pair.record:
            a = pair.record[key][0]['attn'].mean(1, keepdim=True).numpy()
            np.testing.assert_allclose(vp.record[key][0]['attn'].numpy(), a, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(vp.record[key][0]['gate'], pair.record[key][0]['gate'], atol=2e-6, rtol=2e-5)
    finally:
        pair.record = vp.record = None


@torch.no_grad()
def test_evalfree_and_payload_parity(pairs, write_data, tmp_path):
    pair, _, vp, toks, _, _ = pairs
    replies = write_data(tmp_path)
    plans, _ = F.make_plans(replies, n=2, coread_data=tmp_path / 'coread.json', games=['relay_word', 'co_talk'])
    for game, ps in plans.items():
        assert list(F.evaluate_game(pair, ps, bs=2)) == list(F.evaluate_game(vp, ps, bs=2)), game
        eps = [p['episode'] for p in ps]
        if game == 'co_talk':
            for mode in (None, 'zero', 'shuffle'):
                a = S.generate(pair, eps, mode=mode, max_turn=3, keep_tokens=True)
                z = S.generate(vp, eps, mode=mode, max_turn=3, keep_tokens=True)
                for ep, aa, zz in zip(eps, a, z):
                    assert len(zz['turns']) == len(ep['turns'])
                    assert aa['tokens'] == zz['tokens']
                    assert all(all(flags[:-1]) for flags in zz['sampled'])
        else:
            a, z = payload_ctl.table(pair, toks, eps, bs=2), V.payload_table(vp, toks, eps, bs=2)
            for aa, zz in zip(a, z):
                for key in aa:
                    if key.endswith('_ce'):
                        np.testing.assert_allclose(aa[key], zz[key], atol=2e-5, rtol=2e-5)
                    else:
                        assert aa[key] == zz[key]


def test_code_scoring_under_the_mlx_backend():
    original = play_tasks.subprocess
    with V.backend():
        assert play_tasks.run_tests('def f(): return 3', ['assert f() == 3']) == 1.
        assert play_tasks.run_tests('def f(): return 3', ['assert f() == 4']) == 0.
        assert play_tasks.run_tests('raise SystemExit(0)', ['assert False']) == 0.
    assert play_tasks.subprocess is original


def test_command_lines_on_an_exported_checkpoint(pairs, write_data, tmp_path, monkeypatch):
    mp = pairs[1]
    root = tmp_path
    write_data(root)
    checkpoint = root / 'step_tiny'
    M.export_checkpoint(mp, checkpoint, {})
    common = ['--memory_gib', '1.5', '--run', str(checkpoint), '--replies', str(root / 'replies.json')]
    for command, extra in [
            ('evalfree', ['--n', '2', '--bs', '2', '--games', 'relay_word', 'co_talk',
                          '--coread_data', str(root / 'coread.json'), '--out_dir=' + str(root / 'free')]),
            ('transcripts', ['--n', '3', '--bs', '3', '--coread_data=' + str(root / 'coread.json'),
                             '--out', str(root / 'transcripts.txt')]),
            ('readmap', ['--n', '2', '--game', 'relay_word', '--coread_data', str(root / 'coread.json'),
                         '--out', str(root / 'readmap.json')]),
            ('payload', ['--n', '2', '--bs', '2', '--out', str(root / 'payload.json')])]:
        monkeypatch.setattr(sys, 'argv', ['mlx_eval.py', command, *common, *extra])
        V.main()
        gc.collect()
        mx.clear_cache()
    manifest = json.loads((root / 'free/manifest.json').read_text())
    assert manifest['complete'] and manifest['device'] == 'mlx'
    assert (root / 'transcripts.txt').stat().st_size > 0
    talks = [r for r in json.loads((root / 'transcripts.json').read_text()) if r['game'] == 'co_talk']
    assert len(talks) == 3
    for r in talks:
        assert set(r['conditions']) == {'link', 'cut', 'shuffle', 'crossed'}
        assert all(len(r['conditions'][c]['turns']) == len(r['target']) for c in ('link', 'cut', 'shuffle'))
        runs = r['conditions']['crossed']['receiver_runs']
        assert runs and all(len(run['turns']) == len(r['target']) for run in runs.values())
    maps = json.loads((root / 'readmap.json').read_text())
    assert len(maps) == 2
    for r in maps:
        for blk, rows in r['attn'].items():
            assert len(rows) == len(r['payload_label_index'])
            for i, row in enumerate(rows):
                assert len(row) == len(r['partner_pieces'])
                assert abs(sum(row) + r['null'][blk][i] - 1.) < .003
    payload = json.loads((root / 'payload.json').read_text())
    assert payload and all(set(r) == {'game', 'episode', 'receiver', 'n_tokens', 'donor', 'link_ce', 'cut_ce',
                                      'shuffle_ce'} for r in payload)
    assert json.loads(F.summarize(root / 'free', 'tiny_mlx', results_dir=root).read_text())['rows']
