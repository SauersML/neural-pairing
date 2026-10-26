"""stats.py: exact sign tests and median intervals on known cases; the betting confidence interval on known cases, its
symmetry, scaling and terminal capital, its input validation, its exact coverage on every Bernoulli sample of size 8,
and its simulated coverage (with the median interval's) on four distributions.
"""
import itertools
import math
import random

import pytest

from stats import _bets, betting_ci, median_ci, sign_test


def test_known_sign_tests():
    assert sign_test([]) == dict(p=1., positive=0, negative=0, ties=0)
    assert sign_test([1] * 5 + [0] * 3)['p'] == 0.0625
    assert sign_test([-1] * 6)['p'] == 0.03125
    assert sign_test([-1, 1])['p'] == 1.
    assert sign_test([2, 2, 2], null=2)['ties'] == 3


def test_exact_median_indices():
    assert median_ci(range(6)) == (0., 5.)              # coverage 1 - 2/64
    assert median_ci(range(10)) == (1., 8.)             # coverage 1 - 22/1024
    assert median_ci(range(5)) == (-math.inf, math.inf)
    assert median_ci([], bounds=(0, 1)) == (0, 1)
    assert median_ci([0.4] * 20) == (0.4, 0.4)


def test_betting_known_cases():
    assert betting_ci([]) == (0., 1.)
    assert betting_ci([0.5]) == (0., 1.)
    a, b = betting_ci([0.] * 100)
    c, d = betting_ci([1.] * 100)
    assert a == 0. and d == 1.
    assert round(b - (1 - c), 7) == 0
    assert b < .1
    a, b = betting_ci([.25] * 100)
    assert a < .25 < b and b - a < .2
    xs = [0, 1, .3, .4, .2] * 10
    original = betting_ci(xs)
    scaled = betting_ci([2 * x - 1 for x in xs], bounds=(-1, 1))
    for a, b in zip(original, scaled):
        assert round(2 * a - 1 - b, 7) == 0
    assert _bets([.1, .2, .3], .05)[:2] == _bets([.1, .9, .8], .05)[:2]
    # At each end of the interval, a direct product of the bets reaches the one-sided threshold.
    bets = _bets(xs, .05)
    for m, positive in zip(original, (True, False)):
        cap = .5 / (m if positive else 1 - m)
        capital = math.prod(1 + (1 if positive else -1) * min(b, cap) * (x - m) for b, x in zip(bets, xs))
        assert round(capital - 40., 8) == 0


@pytest.mark.parametrize('xs', [[float('nan')], [float('inf')], [-.01], [1.01]])
def test_invalid_samples(xs):
    with pytest.raises(ValueError):
        betting_ci(xs)


def test_invalid_levels_and_bounds():
    for alpha in (0, 1, -1):
        with pytest.raises(ValueError):
            betting_ci([0], alpha)
    with pytest.raises(ValueError):
        betting_ci([0], bounds=(1, 1))


@pytest.mark.parametrize('p', [.1, .5, .9])
def test_exact_small_bernoulli_coverage(p):
    coverage = 0.
    for xs in itertools.product((0, 1), repeat=8):
        a, b = betting_ci(xs)
        if a <= p <= b:
            coverage += p ** sum(xs) * (1 - p) ** (8 - sum(xs))
    assert coverage >= .95 - 1e-12


def test_simulated_coverage():
    """A deterministic smoke check of coverage, not a Monte Carlo proof."""
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
        assert covered / reps >= .91 and med_covered / reps >= .91, name
