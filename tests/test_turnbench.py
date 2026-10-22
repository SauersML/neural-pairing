"""Real tiny Qwen/AFM benchmark tests; no mocked model or scoring path.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python tests/test_turnbench.py
COLDSTART_REPLIES may override the existing local replies file.
TURNBENCH_CHECKPOINT may reuse the three-step demo checkpoint.
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
import games as G
import link
import payload_ctl
import session as S
import turnbench as T


class Turnbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.replies = json.loads(Path(os.environ.get('COLDSTART_REPLIES',
                                  '/Users/user/mpd-data/neural-pairing/replies.json')).read_text())
        if os.environ.get('TURNBENCH_CHECKPOINT'):
            from evaluate import load_pair
            cls.pair, cls.toks = load_pair(os.environ['TURNBENCH_CHECKPOINT'])
        else:
            from peft import LoraConfig, get_peft_model
            cls.toks, models = {}, {}
            for x in link.SIDES:
                cls.toks[x], m = link.load(x)
                models[x] = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'])).eval()
            cls.pair = link.Pair(models, cls.toks, (1, 2, 4), heads=2, head_dim=16).eval()
            # A genuine optimizer update opens the zero-initialized link.
            ep = T.episodes(cls.replies, T.secrets(8))['duplex'][0][0]
            b = S.collate([S.build(ep, cls.toks)], {x: cls.toks[x].pad_token_id for x in link.SIDES})
            opt = torch.optim.AdamW(cls.pair.bridge.parameters(), lr=.003)
            h, _ = cls.pair.joint(b)
            r = ep['meta']['receiver']
            loss = torch.nn.functional.cross_entropy(cls.pair.token_logits(r, h[r], b[r]['idx']), b[r]['lab'])
            loss.backward(); opt.step(); opt.zero_grad()
        cls.pair.eval()
        cls.items = T.secrets(16)
        cls.plans = T.episodes(cls.replies, cls.items, pauses=(0, 1, 4, 8))

    def test_common_secrets_balanced_and_distinct_donors(self):
        for kind in T.KINDS:
            for r in link.SIDES:
                self.assertEqual(sum(it['kind'] == kind and it['receiver'] == r for it in self.items), 2)
        for plans in self.plans.values():
            for it, (ep, donor) in zip(self.items, plans):
                self.assertEqual(ep['meta']['value'], it['value'])
                self.assertEqual(ep['meta']['receiver'], it['receiver'])
                self.assertEqual(donor['meta']['value'], it['donor_value'])
                self.assertNotEqual(ep['meta']['value'], donor['meta']['value'])
                if it['kind'] != 'numbers':
                    self.assertIn(it['value'], G.POOLS[it['kind']]['test'])
                    self.assertNotIn(it['value'], G.POOLS[it['kind']]['train'])
                else:
                    self.assertRegex(it['value'], r'^[1-9][0-9]{3}$')
                    self.assertEqual(it['holdout'], 'unverified_training_overlap')
        self.assertEqual(self.items, T.secrets(16))
        seen = [it['value'] for it in self.items if it['kind'] == 'numbers']
        for it in T.secrets(128, seen_numbers=seen):
            if it['kind'] == 'numbers':
                self.assertNotIn(it['value'], seen)
                self.assertNotIn(it['donor_value'], seen)

    def test_text_leak_checks_both_tokenizers_and_payload_ownership(self):
        for plans in self.plans.values():
            for ep, donor in plans:
                for e in (ep, donor):
                    built = T.assert_no_text_leak(e, self.toks)
                    r = e['meta']['receiver']
                    labs = S.payload_labels(e, self.toks)
                    self.assertEqual(list(labs), [r])
                    decoded = self.toks[r].decode([built[r]['labels'][j][1] for j in labs[r]]).strip()
                    self.assertEqual(decoded, e['meta']['value'])
                    # An independent raw-ID check up to the first payload label.
                    pos = built[r]['labels'][labs[r][0]][0]
                    for tok in self.toks.values():
                        ids = S.ids(tok, self.toks[r].decode(built[r]['ids'][:pos + 1]))
                        for prefix in ('', ' '):
                            self.assertFalse(T.contains(ids, S.ids(tok, prefix + e['meta']['value'])))
                    if e['mode'] == 'dialogue':
                        self.assertEqual(e['turns'][0], (e['meta']['holder'], T.NEUTRAL, 0.))
                    if e['mode'] == 'pen':
                        self.assertEqual(S.pen_author(e, 3), r)
        ep = copy.deepcopy(self.plans['dialogue'][0][0])
        ep['turns'][0] = (ep['meta']['holder'], ep['meta']['value'], 0.)
        with self.assertRaisesRegex(ValueError, 'secret in'):
            T.assert_no_text_leak(ep, self.toks)

    def test_zero_pause_bitwise_episodes_timelines_and_measured_rows(self):
        for name in ('duplex', 'dialogue'):
            self.assertEqual(self.plans[name], self.plans[name + '+0'])
            for a, b in zip(self.plans[name], self.plans[name + '+0']):
                for ea, eb in zip(a, b):
                    self.assertEqual(S.build(ea, self.toks), S.build(eb, self.toks))
            base = T.measure(self.pair, self.toks, self.plans[name], bs=2)
            zero = T.measure(self.pair, self.toks, self.plans[name + '+0'], bs=2)
            self.assertEqual(base, zero)
            self.assertEqual(T.summarize(base), T.summarize(zero))

    def test_all_structures_real_scoring_costs_and_pause_increments(self):
        for name, plans in self.plans.items():
            rows = T.measure(self.pair, self.toks, plans[:8], bs=2)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(math.isfinite(r[c]) for r in rows for c in ('link_ce', 'shuffle_ce', 'cut_ce')))
            for row, (ep, _) in zip(rows, plans):
                built = S.build(ep, self.toks)
                self.assertEqual(row['forward_tokens'], sum(len(built[x]['ids']) for x in link.SIDES))
                self.assertGreater(row['ticks'], 0)
                self.assertGreaterEqual(row['ticks'], row['payload_ticks'])
                self.assertGreaterEqual(row['evaluation_padded_tokens_share'], 0)
            if '+' in name:
                base, k = name.split('+'); k = int(k)
                for (ep, _), (plain, _) in zip(plans, self.plans[base]):
                    a = T.costs(ep, S.build(ep, self.toks), self.toks)
                    b = T.costs(plain, S.build(plain, self.toks), self.toks)
                    self.assertEqual(a['ticks'] - b['ticks'], k)
                    self.assertEqual(a['forward_tokens'] - b['forward_tokens'], 2 * k)

    def test_explicit_shuffle_equals_existing_payload_table(self):
        # Independent pair composition must match both directions of native
        # cyclic shuffle on exactly two episodes, including pen/dialogue timing.
        for name in ('duplex', 'duplex+4', 'dialogue', 'dialogue+4', 'pen'):
            for ep, donor in self.plans[name][:2]:
                expected = payload_ctl.table(self.pair, self.toks, [ep, donor], bs=2)
                expected = next(r for r in expected if r['episode'] == 0)
                actual = T.measure(self.pair, self.toks, [(ep, donor)], bs=1)[0]
                for key in ('link_ce', 'cut_ce', 'shuffle_ce'):
                    self.assertAlmostEqual(actual[key], expected[key], delta=3e-5)
        # Donor assignment survives changes in batching, even singleton tails.
        plans = self.plans['pen']
        a = T.measure(self.pair, self.toks, plans, bs=1)
        b = T.measure(self.pair, self.toks, plans, bs=3)
        for x, y in zip(a, b):
            self.assertEqual(x['donor_value'], y['donor_value'])
            for key in ('link_ce', 'cut_ce', 'shuffle_ce'):
                self.assertAlmostEqual(x[key], y[key], delta=3e-5)

    def test_full_default_pool_token_audit(self):
        # 'open' is in the held-out pool AND the dialogue system prompt. It
        # must be excluded before drawing the common set, never per structure.
        self.assertIn('open', G.POOLS['words']['test'])
        self.assertIn('open', T.carrier_text())
        items = T.secrets(128)
        for it in items:
            self.assertNotIn(it['value'].casefold(), T.carrier_text().casefold())
            self.assertNotIn(it['donor_value'].casefold(), T.carrier_text().casefold())
        for plans in T.episodes(self.replies, items).values():
            for ep, donor in plans:
                T.assert_no_text_leak(ep, self.toks)
                T.assert_no_text_leak(donor, self.toks)

    def test_statistics_and_validation(self):
        self.assertEqual(T.comparison([1.] * 6)['p'], 1 / 64)
        self.assertEqual(T.comparison([-1.] * 6)['p'], 1.)
        self.assertEqual(T.comparison([0.] * 6)['n'], 0)
        self.assertEqual(T.comparison([0.] * 6)['p'], 1.)
        self.assertEqual(T.comparison(list(range(6)))['median_ci95'], [0., 5.])
        self.assertEqual(T.comparison([1.] * 5)['median_ci95'], [None, None])
        for n in (0, 7, 9, True):
            with self.assertRaises(ValueError):
                T.secrets(n)
        for pause in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                T.episode(G.Games(self.replies), self.items[0], 'duplex', pause)


if __name__ == '__main__':
    unittest.main(verbosity=2)
