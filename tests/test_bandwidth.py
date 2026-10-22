"""Actual tiny Qwen/AFM forwards, real replies and tokenizers; no mocked paths.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python tests/test_bandwidth.py
"""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))

import torch
import bandwidth as B
import games as G
import link
import payload_ctl
import session as S


class Bandwidth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.replies = json.loads(Path(os.environ.get('COLDSTART_REPLIES',
            '/Users/user/mpd-data/neural-pairing/replies.json')).read_text())
        cls.toks, models = {}, {}
        for x in link.SIDES:
            cls.toks[x], models[x] = link.load(x)
            models[x].eval()
        torch.manual_seed(13)
        cls.pair = link.Pair(models, cls.toks, (1, 2, 4), heads=2, head_dim=16).eval()
        cls.items = B.plans(cls.replies, 4)

    def setUp(self):
        with torch.no_grad():
            for side in self.pair.bridge.values():
                side.alpha.fill_(.3)

    def test_alpha_zero_exact_bits_all_pauses(self):
        with torch.no_grad():
            for side in self.pair.bridge.values():
                side.alpha.zero_()
        for k in B.PAUSES:
            rows = B.measure(self.pair, self.toks, self.items['relay_digits/4'], k, bs=2)
            for r in rows:
                self.assertEqual(r['link_logp'], r['shuffle_logp'])
                self.assertEqual(r['link_logp'], r['cut_logp'])
                self.assertEqual(r['bits'], 0.)
                self.assertEqual(r['cut_bits'], 0.)
                self.assertEqual(r['bits_per_tick'], 0.)

    def test_per_token_logp_matches_payload_table(self):
        worst = 0.
        for cell in ('relay_digits/1', 'relay_digits/16', 'relay_word', 'relay_number'):
            for pause in (0, 4):
                rows = B.measure(self.pair, self.toks, self.items[cell], pause, bs=2, logit_chunk=3)
                for row, it in zip(rows, self.items[cell]):
                    eps = [dict(it[key], pause=pause) for key in ('episode', 'donor')]
                    reference = next(r for r in payload_ctl.table(self.pair, self.toks, eps, bs=2)
                                     if r['episode'] == 0)
                    for cond in ('link', 'shuffle', 'cut'):
                        error = abs(-sum(row[cond + '_logp']) / row['n_tokens'] - reference[cond + '_ce'])
                        worst = max(worst, error)
                        self.assertLess(error, 3e-6)
                    self.assertAlmostEqual(row['bits'], (reference['shuffle_ce'] - reference['link_ce']) *
                                           reference['n_tokens'] / math.log(2), delta=8e-5)
                    # Independently inspect every token, not just its mean.
                    built = [S.build(ep, self.toks) for ep in eps]
                    batch = S.collate(built, {x: self.toks[x].pad_token_id for x in link.SIDES})
                    x = row['receiver']
                    selected = [built[0][x]['labels'][j] for j in row['payload_label_indices']]
                    with torch.no_grad():
                        for cond, mode in payload_ctl.CONDS:
                            h, _ = self.pair.joint(batch, mode=mode)
                            idx = torch.tensor([p for p, _, _ in selected])
                            targets = torch.tensor([t for _, t, _ in selected])
                            logits = self.pair.token_logits(x, h[x], idx)
                            lp = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
                            torch.testing.assert_close(lp, torch.tensor(row[cond + '_logp']), atol=3e-6, rtol=0)
        print('maximum payload_ctl mean-CE difference:', worst, flush=True)

    def test_same_value_donors_zero_and_exchangeable_mean(self):
        for pause in (0, 8):
            identical = copy.deepcopy(self.items['relay_digits/4'])
            for it in identical:
                it['donor'] = copy.deepcopy(it['episode'])
            rows = B.measure(self.pair, self.toks, identical, pause)
            self.assertTrue(all(r['bits'] == 0 for r in rows))
            # Same secret, two different sender contexts. Score both assignments
            # with identical receiver streams: exchangeability cancels the mean,
            # although neither individual nuisance contrast must be zero.
            pairs = []
            for it in identical[:2]:
                a = copy.deepcopy(it)
                s = a['episode']['meta']['sender']
                a['donor']['task'][s] += ' Send it carefully and silently.'
                b = dict(a, item=a['item'] + 2, episode=a['donor'], donor=a['episode'])
                pairs += [a, b]
            changed = B.measure(self.pair, self.toks, pairs, pause)
            self.assertAlmostEqual(sum(r['bits'] for r in changed), 0., delta=1e-5)

    def test_draws_balanced_exact_lengths_and_exclusions(self):
        self.assertEqual(self.items, B.plans(self.replies, 4))
        for cell, items in self.items.items():
            self.assertEqual([it['episode']['meta']['receiver'] for it in items], list(link.SIDES) * 2)
            for it in items:
                for ep in (it['episode'], it['donor']):
                    self.assertEqual(ep['meta']['receiver'], it['episode']['meta']['receiver'])
                    if it['length']:
                        self.assertEqual(len(ep['meta']['value'].split()), it['length'])
                    elif ep['game'] == 'relay_word':
                        self.assertIn(ep['meta']['value'], G.POOLS['words']['test'])
                        self.assertNotIn(ep['meta']['value'], G.POOLS['words']['train'])
                if cell != 'relay_word':
                    self.assertEqual(it['holdout'], 'unverified_training_overlap')
        exclude = {game: [it[key]['meta']['value'] for cell, items in self.items.items()
                          for it in items for key in ('episode', 'donor') if it[key]['game'] == game and it['length'] != 1]
                   for game in ('relay_number', 'relay_digits')}
        fresh = B.plans(self.replies, 4, seen_values=exclude, require_unseen=True)
        for items in fresh.values():
            for it in items:
                for ep in (it['episode'], it['donor']):
                    self.assertNotIn(ep['meta']['value'], exclude.get(ep['game'], []))
        with self.assertRaisesRegex(ValueError, 'no unseen values'):
            B.plans(self.replies, 4, seen_values={'relay_digits': list(range(10))})
        with self.assertRaisesRegex(ValueError, 'complete'):
            B.plans(self.replies, 4, require_unseen=True)
        certified = B.plans(self.replies, 4, require_unseen=True, numeric_split_verified=True)
        self.assertEqual(certified['relay_digits/1'][0]['source_entropy_bits'], 0.)
        self.assertEqual(B.numeric_support(1), 1)
        self.assertEqual(B.numeric_support(2), sum(G.held_out_digits(str(i)) for i in range(100)))
        self.assertEqual(B.numeric_support(None), sum(G.held_out_digits(str(i)) for i in range(10000)))

    def test_batch_invariance_and_tick_denominators(self):
        items = self.items['relay_digits/12']
        base = B.measure(self.pair, self.toks, items, bs=1)
        batched = B.measure(self.pair, self.toks, items, bs=3)
        paused = B.measure(self.pair, self.toks, items, pause=8, bs=2)
        for a, b, p in zip(base, batched, paused):
            self.assertEqual(a['donor_value'], b['donor_value'])
            self.assertAlmostEqual(a['bits'], b['bits'], delta=5e-5)
            self.assertEqual(p['sender_exposure_ticks'], a['sender_exposure_ticks'] + 8)
            self.assertEqual(p['tick_exchanges'], a['tick_exchanges'] + 8 * len(self.pair.xblocks))
            self.assertEqual(a['sender_exposure_ticks'], max(a['payload_prediction_ticks']) + 2)
            self.assertEqual(a['bits_per_tick'], a['bits'] / a['sender_exposure_ticks'])
            self.assertAlmostEqual(a['bits_per_exchange'], a['bits_per_tick'] / len(self.pair.xblocks), places=14)

    def test_intervals_and_invalid_input(self):
        self.assertEqual(B.interval(range(6))['median_ci'], [0., 5.])
        self.assertEqual(B.interval(range(5))['median_ci'], [None, None])
        r = B.interval([-1000, 0, 1000], clip_bits=2)
        self.assertEqual(r['mean'], 0.)
        self.assertEqual(r['clipped_items'], 2)
        self.assertEqual(r['clipped_mean'], 0.)
        for bad in (0, 3, 5, True):
            with self.assertRaises(ValueError):
                B.plans(self.replies, bad)
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                B.measure(self.pair, self.toks, self.items['relay_word'], pause=bad)
        wrong = copy.deepcopy(self.items['relay_word'])
        wrong[0]['donor'] = wrong[1]['episode']
        with self.assertRaisesRegex(ValueError, 'same game and role'):
            B.measure(self.pair, self.toks, wrong)

    def test_real_free_greedy_and_report_serialization(self):
        items = self.items['relay_digits/1']
        gen = B.greedy(self.pair, items, pause=2, n=2, bs=2, max_turn=2)
        self.assertEqual(len(gen), 6)
        self.assertEqual({r['condition'] for r in gen}, {'link', 'shuffle', 'cut'})
        self.assertTrue(all(0 <= r['per_digit'] <= 1 for r in gen))
        rows = B.measure(self.pair, self.toks, items)
        summary = B.summarize(rows)
        self.assertEqual(summary['mean_bits'], sum(r['bits'] for r in rows) / len(rows))
        json.dumps(summary, allow_nan=False)
        table = B.text_table({'relay_digits/1': {0: dict(summary=summary)}})
        self.assertIn('bits/exchange', table)
        self.assertIn('CAPPED', table)


if __name__ == '__main__':
    unittest.main(verbosity=2)
