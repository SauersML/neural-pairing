"""Run directly: MPD_MEM_GIB=6 /Users/user/mpd-data/venv/bin/python tests/test_stats.py."""
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
from stats import _bets, betting_ci, median_ci, sign_test


class Statistics(unittest.TestCase):
    def test_known_sign_tests(self):
        self.assertEqual(sign_test([]), dict(p=1., positive=0, negative=0, ties=0))
        self.assertEqual(sign_test([1] * 5 + [0] * 3)['p'], 0.0625)
        self.assertEqual(sign_test([-1] * 6)['p'], 0.03125)
        self.assertEqual(sign_test([-1, 1])['p'], 1.)
        self.assertEqual(sign_test([2, 2, 2], null=2)['ties'], 3)

    def test_exact_median_indices(self):
        self.assertEqual(median_ci(range(6)), (0., 5.))  # coverage 1-2/64
        self.assertEqual(median_ci(range(10)), (1., 8.))  # coverage 1-22/1024
        self.assertEqual(median_ci(range(5)), (-math.inf, math.inf))
        self.assertEqual(median_ci([], bounds=(0, 1)), (0, 1))
        self.assertEqual(median_ci([0.4] * 20), (0.4, 0.4))

    def test_betting_known_cases(self):
        self.assertEqual(betting_ci([]), (0., 1.))
        self.assertEqual(betting_ci([0.5]), (0., 1.))
        a, b = betting_ci([0.] * 100)
        c, d = betting_ci([1.] * 100)
        self.assertEqual(a, 0.)
        self.assertEqual(d, 1.)
        self.assertAlmostEqual(b, 1 - c)
        self.assertLess(b, .1)
        a, b = betting_ci([.25] * 100)
        self.assertLess(a, .25)
        self.assertGreater(b, .25)
        self.assertLess(b - a, .2)
        xs = [0, 1, .3, .4, .2] * 10
        original = betting_ci(xs)
        scaled = betting_ci([2 * x - 1 for x in xs], bounds=(-1, 1))
        for a, b in zip(original, scaled):
            self.assertAlmostEqual(2 * a - 1, b)
        self.assertEqual(_bets([.1, .2, .3], .05)[:2], _bets([.1, .9, .8], .05)[:2])
        # Independent direct product verifies terminal one-sided threshold.
        bets = _bets(xs, .05)
        for m, positive in zip(original, (True, False)):
            cap = .5 / (m if positive else 1 - m)
            capital = math.prod(1 + (1 if positive else -1) * min(b, cap) * (x - m)
                                for b, x in zip(bets, xs))
            self.assertAlmostEqual(capital, 40., places=8)

    def test_invalid_inputs(self):
        for xs in ([float('nan')], [float('inf')], [-.01], [1.01]):
            with self.assertRaises(ValueError):
                betting_ci(xs)
        for alpha in (0, 1, -1):
            with self.assertRaises(ValueError):
                betting_ci([0], alpha)
        with self.assertRaises(ValueError):
            betting_ci([0], bounds=(1, 1))

    def test_exact_small_bernoulli_coverage(self):
        for p in (.1, .5, .9):
            coverage = 0.
            for xs in itertools.product((0, 1), repeat=8):
                a, b = betting_ci(xs)
                if a <= p <= b:
                    coverage += p ** sum(xs) * (1 - p) ** (8 - sum(xs))
            self.assertGreaterEqual(coverage, .95 - 1e-12)
            print('exact Bernoulli', p, 'n=8 coverage', round(coverage, 4), flush=True)

    def test_quick_coverage_simulation(self):
        rng = random.Random(481)
        distributions = [('bernoulli', lambda: float(rng.random() < .2), .2, 0.),
                         ('uniform', rng.random, .5, .5),
                         ('beta_skew', lambda: rng.betavariate(1, 5), 1 / 6, 1 - .5 ** .2),
                         ('atom_mixture', lambda: 0. if rng.random() < .8 else rng.random(), .1, 0.)]
        for name, draw, mean, median in distributions:
            covered = med_covered = 0
            reps = 200
            for _ in range(reps):
                xs = [draw() for _ in range(48)]
                a, b = betting_ci(xs)
                c, d = median_ci(xs, bounds=(0, 1))
                covered += a <= mean <= b
                med_covered += c <= median <= d
            print(name, 'mean coverage', covered / reps, 'median coverage', med_covered / reps, flush=True)
            # A deterministic smoke check, not a Monte Carlo proof of coverage.
            self.assertGreaterEqual(covered / reps, .91)
            self.assertGreaterEqual(med_covered / reps, .91)


if __name__ == '__main__':
    unittest.main()
