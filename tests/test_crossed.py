"""Whole-dialogue isolation on real tiny Qwen/AFM models, with active LoRA/link."""
import copy
import os
from pathlib import Path
import sys
import unittest

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))

import torch
from peft import LoraConfig, get_peft_model
import coread as C
import crossed as X
import evalfree as F
import evaluate as E
import games as G
import link
import session as S


def dialogue(game, opener, i):
    other = link.partner(opener)
    if game == 'co_play':
        secrets = dict(zip(link.SIDES, ({'word': w} for w in
                       (('fox', 'otter'), ('walrus', 'badger'), ('tiger', 'raven'))[i])))
        texts = {x: secrets[link.partner(x)]['word'] for x in link.SIDES}
        system = {x: C.play_system('guess', x, opener, secrets) for x in link.SIDES}
        meta = dict(play='guess', opener=opener, secrets=secrets,
                    payload={x: [(0, len(texts[x]))] for x in link.SIDES})
    else:
        city = ('Lima', 'Oslo', 'Suva')[i]
        item = dict(src='test', task='qa', context=f'Ada lives in {city}.', question='Where does Ada live?', answer=city)
        texts = {opener: city, other: 'A detail.'}
        system = {x: C.talk_system(item, x, opener) for x in link.SIDES}
        meta = dict(asker=opener, holder=other, item=item, payload={opener: [(0, len(city))]})
    return dict(game=game, mode='dialogue', system=system, task={x: None for x in link.SIDES},
                turns=[(x, texts[x], 1.) for x in (opener, other) * 3],
                meta=dict(meta, sender=other, receiver=opener), pause_turns={1: 1, 3: 2})


class Crossed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        toks, models = {}, {}
        for x in link.SIDES:
            toks[x], model = link.load(x)
            model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj']))
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    torch.nn.init.normal_(p, std=.03)
            models[x] = model.eval()
        cls.pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
        for side in cls.pair.bridge.values():
            side.alpha.data.fill_(.9)

    def test_whole_generation_is_bitwise_isolated(self):
        for game in ('co_talk', 'co_play'):
            for opener in link.SIDES:
                ep, donor = [dialogue(game, opener, i) for i in (0, 1)]
                for x in F.payloads(ep):
                    with self.subTest(game=game, opener=opener, receiver=x):
                        y = link.partner(x)
                        changed = copy.deepcopy(ep)
                        changed['system'][y] = 'A different and much longer private prompt. ' * 20
                        changed['task'][y] = 'The secret is a violet elephant.'
                        # Gold/replay/forced fields must never enter the crossed run.
                        changed['turns'] = [(sp, 'FORBIDDEN SECRET', wt) for sp, _, wt in ep['turns']]
                        changed['forced'] = list(range(6))
                        changed['turn_tokens'] = [[99]] * 6
                        a = X.generate(self.pair, ep, donor, x, max_turn=3, keep=True)
                        b = X.generate(self.pair, changed, donor, x, max_turn=3, keep=True)
                        self.assertEqual(len(a['turns']), 6)
                        for key in ('turns', 'tokens', 'sampled', 'gates', 'gate_turn'):
                            self.assertEqual(a[key], b[key], key)
                        for side in link.SIDES:
                            self.assertTrue(torch.equal(a['h'][side], b['h'][side]))
                            self.assertTrue(torch.equal(a['p'][side], b['p'][side]))
                        # Nonvacuity: the donor's prompt actually affects the receiver.
                        new_donor = copy.deepcopy(donor)
                        new_donor['system'][y] = 'A private donor context about violet elephants.'
                        c = X.generate(self.pair, ep, new_donor, x, max_turn=3, keep=True)
                        n = min(len(a['h'][x]), len(c['h'][x]))
                        self.assertGreater(float((a['h'][x][:n] - c['h'][x][:n]).abs().max()), 1e-4)
                        print('PASS bitwise whole six-turn isolation and active donor:', game, opener, x, flush=True)

    def test_sampled_isolation(self):
        ep, donor = [dialogue('co_play', 'quill', i) for i in (0, 1)]
        changed = copy.deepcopy(ep)
        changed['system']['quill'] = 'Different real opener secret.'
        a = X.generate(self.pair, ep, donor, 'ember', 2, temperature=.8, seed=19, keep=True)
        b = X.generate(self.pair, changed, donor, 'ember', 2, temperature=.8, seed=19, keep=True)
        self.assertEqual(a['tokens'], b['tokens'])
        self.assertTrue(torch.equal(a['h']['ember'], b['h']['ember']))

    def test_matching_and_plan_guards(self):
        ep, donor = [dialogue('co_play', 'quill', i) for i in (0, 1)]
        self.assertTrue(F.matched(ep, donor))
        for bad in (ep, dialogue('co_play', 'ember', 1), dialogue('co_talk', 'quill', 1)):
            self.assertFalse(F.matched(ep, bad))
        bad = copy.deepcopy(donor)
        bad['meta']['play'] = 'rhyme'
        self.assertFalse(F.matched(ep, bad))
        bad = copy.deepcopy(donor)
        bad['meta']['secrets']['quill'] = ep['meta']['secrets']['quill']
        self.assertFalse(F.matched(ep, bad))
        with self.assertRaisesRegex(ValueError, 'explicit matched donor'):
            E.run_episodes(self.pair, [ep], mode='crossed')
        with self.assertRaisesRegex(ValueError, 'matched donor'):
            X.generate(self.pair, ep, ep, 'quill', 1)

    def test_fresh_play_donors_do_not_require_a_second_stored_conversation(self):
        # Real game constructor and split pools; donors need no reference text.
        replies = {x: dict(own={}, steer=[]) for x in link.SIDES}
        source = G.Games(replies, split='test', seed=13)
        ep = dialogue('co_play', 'ember', 0)
        before = copy.deepcopy(ep)
        donor, attempts = F.choose_donor(ep, source)
        self.assertTrue(F.matched(ep, donor))
        self.assertGreaterEqual(attempts, 1)
        self.assertEqual(donor['meta']['split'], 'test')
        self.assertEqual(donor['meta']['opener'], 'ember')
        for x in link.SIDES:
            self.assertIn(donor['meta']['secrets'][x]['word'], G.POOLS['words']['test'])
            self.assertEqual(donor['system'][x], C.play_system('guess', x, 'ember', donor['meta']['secrets']))
        self.assertTrue(all(not text for _, text, _ in donor['turns']))
        self.assertEqual(ep, before)
        repeat, _ = F.choose_donor(ep, G.Games(replies, split='test', seed=13))
        self.assertEqual(donor, repeat)

    def test_entrypoints_preserve_legacy_and_keep_separate_traces(self):
        old = os.environ.get('NL_MAXNEW')
        os.environ['NL_MAXNEW'] = '2'
        try:
            for game in ('co_talk', 'co_play'):
                ep, donor = [dialogue(game, 'quill', i) for i in (0, 1)]
                original = copy.deepcopy([ep, donor])
                plans = [dict(episode=ep, donor=donor, episode_id='0', index=0, attempts=1)]
                legacy = list(F.evaluate_game(self.pair, plans))
                added = list(F.evaluate_game(self.pair, plans, crossed=True))
                for a, b in zip(legacy, added):
                    self.assertEqual(a, dict(b, conditions={k: v for k, v in b['conditions'].items() if k != 'crossed'}))
                    self.assertEqual(len(b['conditions']['crossed']['turns']), 6)
                a = F.summary_rows(legacy, [game])
                b = F.summary_rows(added, [game])
                for aa, bb in zip(a, b):
                    for key, value in aa.items():
                        self.assertEqual(value, {k: v for k, v in bb[key].items() if k != 'crossed'}
                                         if key in ('mean', 'capped_n') else bb[key])
                result = E.run_episodes(self.pair, [ep], mode='crossed', donors=[donor])[0]
                self.assertEqual(set(result['receiver_runs']), set(F.payloads(ep)))
                self.assertNotIn('turns', result, 'independent traces must not be spliced')
                for x, r in result['receiver_runs'].items():
                    self.assertEqual(result['each'][x], F.receiver_scores(ep, r)[x])
                    record = next(r for r in added if r['receiver'] == x)
                    self.assertEqual(r['tokens'], record['conditions']['crossed']['tokens'])
                # The old entry point remains exactly session.generate + old scorer.
                for mode in (None, 'zero', 'shuffle', 'random'):
                    actual = E.run_episodes(self.pair, [ep, donor], mode=mode, bs=2)
                    raw = S.generate(self.pair, [ep, donor], max_turn=2, mode=mode,
                                     gen=torch.Generator().manual_seed(0))
                    expected = [dict(game=e['game'], meta=e['meta'], **r, **E.score(e, r)) for e, r in zip((ep, donor), raw)]
                    self.assertEqual(actual, expected)
                self.assertEqual([ep, donor], original)
        finally:
            if old is None:
                os.environ.pop('NL_MAXNEW')
            else:
                os.environ['NL_MAXNEW'] = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
