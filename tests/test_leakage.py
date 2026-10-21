"""Leakage regression checks; run with NL_TINY=1 NL_DEVICE=cpu."""
import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pairing'))
import coread as C
import games as G
from link import SIDES


def replies():
    return {x: {'own': {q: 'An ordinary answer.' for sp in ('train', 'test') for q in G.POOLS['questions'][sp]},
                'steer': []} for x in SIDES}


class EpisodeLeakage(unittest.TestCase):
    def test_named_game_filters_mixed_source_tasks(self):
        items = [dict(src='mixed', task='qa', context='Ana lives in Lima.', question='Where is Ana?', answer='Lima'),
                 dict(src='mixed', task='instruct', instruction='Say hello.', response='Hello.')]
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump(dict(weights={'mixed': 1}, train=items, test=[]), f); f.flush()
            co = C.Coread(replies(), 'train', seed=1, path=f.name)
            for game in ('co_qa', 'co_instruct'):
                for _ in range(20):
                    self.assertEqual(co.episode(game)['game'], game)

    def test_merge_sum_unseen_operands(self):
        train = G.Games(replies(), 'train', seed=3)
        test = G.Games(replies(), 'test', seed=3)
        for _ in range(100):
            self.assertTrue(all(0 <= v < 50 for v in train.episode('merge_sum')['meta']['values'].values()))
            self.assertTrue(all(50 <= v < 100 for v in test.episode('merge_sum')['meta']['values'].values()))

    def test_swap_never_copies_own_text(self):
        items = [dict(src='small', task='passage', kind='prose', text=t)
                 for t in ('The red fox lives near the old green bridge.',
                           'The blue whale swims across the cold deep ocean.')]
        # Duplicate rows must not defeat text-level distinctness.
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump(dict(weights={'small': 1}, train=items + items[:1], test=[]), f); f.flush()
            co = C.Coread(replies(), 'train', seed=3, path=f.name)
            for _ in range(100):
                ep = co.episode('co_swap')
                self.assertNotEqual(ep['meta']['item']['text'], ep['meta']['other']['text'])
                for x in SIDES:
                    target = ep['reply'][x][0]
                    for a, b in ep['meta']['payload'][x]:
                        self.assertNotIn(target[a:b], ep['task'][x])

    def test_test_split_filters_repeated_content(self):
        train = [dict(src='qa', task='qa', context='The hidden city is Lima.', question='Where is Ana?', answer='Lima'),
                 dict(src='opus', task='translate', source='Please open the green door.', target='Ouvrez la porte verte.',
                      source_lang='English', target_lang='French')]
        contaminated = [dict(train[0], question='Which city is hidden?'),
                        dict(train[1], source=train[1]['target'], target=train[1]['source'],
                             source_lang='French', target_lang='English')]
        fresh = dict(train[0], context='Ana flew to Lima yesterday.', question='Where did Ana fly?')
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump(dict(weights={'qa': 1, 'opus': 1}, train=train, test=contaminated + [fresh]), f); f.flush()
            co = C.Coread(replies(), 'test', path=f.name)
            self.assertEqual([it for rows in co.by_src.values() for it in rows], [fresh])

    def test_code_split_rejects_identical_program(self):
        item = dict(src='mbpp', task='code_write', problem='Add the values. It must pass: assert f(1, 2) == 3',
                    code='def f(a, b): return a + b', tests=['assert f(1, 2) == 3'])
        duplicate = dict(item, problem='Return the total. It must pass: assert f(2, 3) == 5')
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump(dict(weights={'mbpp': 1}, train=[item], test=[duplicate]), f); f.flush()
            co = C.Coread(replies(), 'test', path=f.name)
            self.assertEqual(co.by_src, {})


class CodeRewardLeakage(unittest.TestCase):
    def test_clean_exit_does_not_pass_an_unexecuted_assertion(self):
        from play_tasks import run_tests
        for code in ('raise SystemExit(0)', 'import os; os._exit(0)'):
            with self.subTest(code=code):
                self.assertEqual(run_tests(code, ['assert False']), 0.)
        self.assertEqual(run_tests('def f(x): return x + 1',
                                   ['assert f(1) == 2', 'assert f(1) == 3']), .5)


class ShuffleLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import link
        from peft import LoraConfig, get_peft_model
        cls.torch = torch
        cls.toks, models = {}, {}
        for x in SIDES:
            cls.toks[x], model = link.load(x)
            models[x] = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
                                                       target_modules=['q_proj', 'v_proj'])).eval()
        torch.manual_seed(48)
        cls.pair = link.Pair(models, cls.toks, (1, 2, 4), heads=2, head_dim=16).eval()
        for bridge in cls.pair.bridge.values():
            bridge.alpha.data.fill_(1.)
        cls.eps = [dict(mode='duplex', system={x: f'You are {x}.' for x in SIDES},
                        task={'quill': f'Report the secret {i}.', 'ember': f'The secret is purple velvet {i}.'},
                        reply={x: ('Short answer.', 1.) for x in SIDES}) for i in range(3)]

    def batch(self, eps):
        import session as S
        return S.collate([S.build(ep, self.toks) for ep in eps],
                         {x: self.toks[x].pad_token_id for x in SIDES})

    def test_shuffle_has_no_path_from_original_partner(self):
        import session as S
        changed = copy.deepcopy(self.eps)
        changed[0]['task']['ember'] = 'The secret is orange metal 0.'
        with self.torch.no_grad():
            a, _ = self.pair.joint(self.batch(self.eps), mode='shuffle')
            b, _ = self.pair.joint(self.batch(changed), mode='shuffle')
            self.assertTrue(self.torch.equal(a['quill'][0], b['quill'][0]))
            # The changed original partner still reaches its assigned shuffled receiver.
            self.assertGreater(float((a['quill'][1] - b['quill'][1]).abs().max()), 1e-5)
            ga = S.generate(self.pair, self.eps, max_turn=3, force=True, keep=True, mode='shuffle')
            gb = S.generate(self.pair, changed, max_turn=3, force=True, keep=True, mode='shuffle')
            self.assertTrue(self.torch.equal(ga[0]['h']['quill'], gb[0]['h']['quill']))

    def test_shuffle_rejects_singletons(self):
        import session as S
        with self.torch.no_grad():
            with self.assertRaisesRegex(ValueError, 'at least two'):
                self.pair.joint(self.batch(self.eps[:1]), mode='shuffle')
            with self.assertRaisesRegex(ValueError, 'at least two'):
                S.generate(self.pair, self.eps[:1], max_turn=1, mode='shuffle')


if __name__ == '__main__':
    unittest.main()
