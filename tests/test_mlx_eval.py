"""No-mock torch/MLX evaluation parity on both real NL_TINY architectures.

MPD_MEM_GIB=3 HF_HUB_OFFLINE=1 /Users/user/mpd-data/venv/bin/python tests/test_mlx_eval.py
"""
import os
os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MLX_ENABLE_TF32'] = '0'
os.environ['NL_MAXNEW'] = '3'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import json
import numpy as np
import torch
import mlx.core as mx
from peft import LoraConfig, get_peft_model
import link
import session as S
import evaluate as E
import evalfree as F
import mlx_pair as M
import mlx_eval as V

torch.set_num_threads(2)
torch.manual_seed(17)
mx.random.seed(17)
mx.set_memory_limit(2 * 2**30)
mx.set_cache_limit(16 * 2**20)


def main():
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
    mp = M.Pair(mm, xb, sizes, heads=4, head_dim=32, gate_norm=True, checkpoint=False)
    for x in M.SIDES:
        mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in pair.bridge[x].state_dict().items()])
    mx.eval(mp.parameters())
    vp = V.Pair(mp, toks)
    system = {x: 'You are ' + x + '.' for x in M.SIDES}
    duplex = [dict(game='plain', meta={}, mode='duplex', system=system,
                   task={'quill': 'Remember the number 42.' if i == 0 else 'Blue.',
                         'ember': 'What is the number?' if i == 1 else 'Name a city in Italy.'},
                   reply={'quill': ('Yes.', .2), 'ember': ('42.', 1.)}) for i in range(3)]
    dialogue = [dict(game='chat', meta={'sender': 'quill'}, mode='dialogue', system=system,
                     task={'quill': 'Ask a question.' if i else 'Which number?', 'ember': None},
                     turns=[('quill', 'Blue?', .4), ('ember', 'Green.', 1.), ('quill', 'Okay.', .6)],
                     **({'link_off_after': 1} if i == 1 else {})) for i in range(3)]
    built = [S.build(e, toks) for e in duplex + dialogue]
    b = S.collate(built, {x: toks[x].pad_token_id for x in M.SIDES})
    results = {}
    with torch.no_grad():
        for mode in (None, 'zero', 'shuffle'):
            h, g = pair.joint(b, mode)
            mh, mg = vp.joint(b, mode)
            for x in M.SIDES:
                tl = pair.token_logits(x, h[x], b[x]['idx']).numpy()
                ml = vp.token_logits(x, mh[x], b[x]['idx']).numpy()
                err = float(np.max(np.abs(tl - ml)))
                results[f'logits/{mode}/{x}'] = err
                np.testing.assert_allclose(ml, tl, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(mg[x], g[x], atol=2e-6, rtol=2e-5)
            for label, eps in (('duplex', duplex), ('dialogue', dialogue)):
                a = S.generate(pair, eps, mode=mode, max_turn=3, keep_tokens=True)
                z = S.generate(vp, eps, mode=mode, max_turn=3, keep_tokens=True)
                for aa, zz in zip(a, z):
                    assert aa['tokens'] == zz['tokens'], (label, mode, aa['tokens'], zz['tokens'])
                    assert aa['sampled'] == zz['sampled']
                    key = 'text' if label == 'duplex' else 'turns'
                    assert aa[key] == zz[key]
                # Explicitly exercise unchanged evaluate.run_episodes and scores.
                a = E.run_episodes(pair, eps, mode=mode, bs=3)
                z = E.run_episodes(vp, eps, mode=mode, bs=3)
                assert [r['ok'] for r in a] == [r['ok'] for r in z]
                assert [r[key] for r in a] == [r[key] for r in z]
                results[f'greedy/{label}/{mode}'] = 'token-identical'
                print(label, mode, 'token-identical', flush=True)
        for mode in (None, 'shuffle'):
            pair.record, vp.record = {}, {}
            pair.joint(b, mode)
            vp.joint(b, mode)
            for key in pair.record:
                a = pair.record[key][0]['attn'].mean(1, keepdim=True).numpy()
                z = vp.record[key][0]['attn'].numpy()
                results[f'attention/{mode}/{key}'] = float(np.max(np.abs(a - z)))
                np.testing.assert_allclose(z, a, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(vp.record[key][0]['gate'], pair.record[key][0]['gate'], atol=2e-6, rtol=2e-5)
            pair.record = vp.record = None
    # Real episode builders, donor selection, payload scorers, and co_talk.
    # The fixture is a small corpus, not a mocked model or tokenizer.
    import tempfile
    from test_evalfree import fixtures
    import payload_ctl as P
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        root = Path(tmp)
        replies = fixtures(root)
        plans, _ = F.make_plans(replies, n=2, coread_data=root / 'coread.json',
                                games=['relay_word', 'co_talk'])
        for game, ps in plans.items():
            a = list(F.evaluate_game(pair, ps, bs=2))
            z = list(F.evaluate_game(vp, ps, bs=2))
            assert a == z, ('matched-donor evaluation', game)
            if game == 'co_talk':
                eps = [p['episode'] for p in ps]
                for mode in (None, 'zero', 'shuffle'):
                    a = S.generate(pair, eps, mode=mode, max_turn=3, keep_tokens=True)
                    z = S.generate(vp, eps, mode=mode, max_turn=3, keep_tokens=True)
                    for ep, aa, zz in zip(eps, a, z):
                        assert len(zz['turns']) == len(ep['turns'])
                        assert aa['tokens'] == zz['tokens']
                        assert all(all(flags[:-1]) for flags in zz['sampled'])
            else:
                eps = [p['episode'] for p in ps]
                a, z = P.table(pair, toks, eps, bs=2), V.payload_table(vp, toks, eps, bs=2)
                for aa, zz in zip(a, z):
                    for key in aa:
                        if key.endswith('_ce'):
                            np.testing.assert_allclose(aa[key], zz[key], atol=2e-5, rtol=2e-5)
                        else:
                            assert aa[key] == zz[key]
            results['integration/' + game] = 'matched-donor records identical; payload/full-dialogue parity'
            print('integration', game, 'passed', flush=True)
    import play_tasks
    original_processes = play_tasks.subprocess
    with V.backend():
        assert play_tasks.run_tests('def f(): return 3', ['assert f() == 3']) == 1.
        assert play_tasks.run_tests('def f(): return 3', ['assert f() == 4']) == 0.
        assert play_tasks.run_tests('raise SystemExit(0)', ['assert False']) == 0.
    assert play_tasks.subprocess is original_processes
    results['code_scoring'] = 'real subprocess pass/fail/early-exit checks passed'
    # Execute every actual CLI against a genuine exported tiny checkpoint,
    # including complete transcript files and the original JSON serializers.
    import gc
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixtures(root)
        data = json.loads((root / 'coread.json').read_text())
        # payload_ctl draws unscored co_instruct too, before transcripts filters.
        data['test'].append(dict(task='instruct', src='instruct', instruction='Say hello.', response='Hello.'))
        data['weights']['instruct'] = 1
        (root / 'coread.json').write_text(json.dumps(data))
        checkpoint = root / 'step_tiny'
        M.export_checkpoint(mp, checkpoint, {})
        common = ['--memory_gib', '1.5', '--run', str(checkpoint), '--replies', str(root / 'replies.json')]
        old_argv = sys.argv
        try:
            for command, extra in [
                ('evalfree', ['--n', '2', '--bs', '2', '--games', 'relay_word', 'co_talk',
                              '--coread_data', str(root / 'coread.json'), '--out_dir=' + str(root / 'free')]),
                ('transcripts', ['--n', '3', '--bs', '3', '--coread_data=' + str(root / 'coread.json'),
                                 '--out', str(root / 'transcripts.txt')]),
                ('readmap', ['--n', '2', '--game', 'relay_word', '--coread_data', str(root / 'coread.json'),
                             '--out', str(root / 'readmap.json')]),
                ('payload', ['--n', '2', '--bs', '2', '--out', str(root / 'payload.json')]),
            ]:
                sys.argv = ['mlx_eval.py', command, *common, *extra]
                V.main()
                gc.collect()
                mx.clear_cache()
                print('CLI', command, 'passed', flush=True)
        finally:
            sys.argv = old_argv
        manifest = json.loads((root / 'free/manifest.json').read_text())
        assert manifest['complete'] and manifest['device'] == 'mlx'
        assert (root / 'transcripts.txt').stat().st_size > 0
        transcripts = json.loads((root / 'transcripts.json').read_text())
        talks = [r for r in transcripts if r['game'] == 'co_talk']
        assert len(talks) == 3
        for r in talks:
            assert set(r['conditions']) == {'link', 'cut', 'shuffle'}
            assert all(len(c['turns']) == len(r['target']) for c in r['conditions'].values())
        maps = json.loads((root / 'readmap.json').read_text())
        assert len(maps) == 2
        for r in maps:
            for blk, rows in r['attn'].items():
                assert len(rows) == len(r['payload_label_index'])
                for i, row in enumerate(rows):
                    assert len(row) == len(r['partner_pieces'])
                    assert abs(sum(row) + r['null'][blk][i] - 1.) < .003
        payload = json.loads((root / 'payload.json').read_text())
        assert payload and all(set(r) == {'game', 'episode', 'receiver', 'n_tokens',
                                          'link_ce', 'cut_ce', 'shuffle_ce'} for r in payload)
        summary = F.summarize(root / 'free', 'tiny_mlx', results_dir=root)
        assert json.loads(summary.read_text())['rows']
        results['CLI'] = 'all four real entry points, schemas, full co_talk, and betting summary passed'
    print(json.dumps(results, indent=2), flush=True)
    print('MLX peak GiB', mx.get_peak_memory() / 2**30, flush=True)


if __name__ == '__main__':
    main()
