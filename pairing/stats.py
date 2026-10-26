"""Distribution-free confidence intervals and tests (no normality or symmetry assumption).

betting_ci: a confidence interval for the mean of bounded values from fixed-n hedged capital with predictable plug-in
bets (Waudby-Smith and Ramdas 2024, JRSS B 86(1), doi:10.1093/jrsssb/qkad009, equations 24-27;
https://arxiv.org/html/2010.09686#S4.SS4). The values must be independent, identically distributed and inside bounds
known in advance; n is fixed before the outcomes are seen, and the values are used in their given order.
sign_test: the exact two-sided sign test, of equal probabilities for positive and negative values (a different
hypothesis from a zero mean).
median_ci: the exact order-statistic interval for a median.

Usage: from stats import betting_ci, sign_test; lo, hi = betting_ci(diffs, bounds=(-1, 1)); p = sign_test(diffs)['p']
"""
import math


def _alpha(alpha):
    """ValueError unless 0 < alpha < 1."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')


def _values(values):
    """values as a list of floats; ValueError for a value that is not finite."""
    xs = [float(x) for x in values]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError('observations must be finite')
    return xs


def _bets(xs, alpha):
    """The bets of equation 27; the bet at step t uses only the observations before t."""
    total, residual, variance = 0.0, 0.25, 0.25
    bets = []
    for t, x in enumerate(xs, 1):
        bets.append(math.sqrt(2 * math.log(2 / alpha) / (len(xs) * variance)))
        total += x
        mean = (0.5 + total) / (t + 1)
        residual += (x - mean) ** 2
        variance = residual / (t + 1)
    return bets


def betting_ci(values, alpha=0.05, bounds=(0.0, 1.0)):
    """(lower, upper): the level 1 - alpha confidence interval for the mean of values, which lie in bounds; lower >
    upper denotes an empty set, and no values give bounds itself.

    The values are rescaled to [0, 1] (theta = c = 1/2). A mean m is rejected when either one-sided capital reaches
    2/alpha; the bets use the regularized mean and variance of equation 26 and the fixed horizon of equation 27. Only
    the terminal capital is used, so repeated calls with growing n are not covered. Each one-sided capital is monotone
    in m, so bisection finds each endpoint, which is rounded outward.
    """
    _alpha(alpha)
    lo, hi = map(float, bounds)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError('bounds must be finite and strictly increasing')
    raw = _values(values)
    if any(x < lo or x > hi for x in raw):
        raise ValueError('observation outside specified bounds')
    if not raw:
        return lo, hi
    xs = [(x - lo) / (hi - lo) for x in raw]
    bets = _bets(xs, alpha)
    threshold = math.log(2 / alpha)

    def log_capital(m, positive):
        cap = 0.5 / (m if positive else 1 - m) if (m if positive else 1 - m) else math.inf
        sign = 1 if positive else -1
        return math.fsum(math.log1p(sign * min(bet, cap) * (x - m)) for x, bet in zip(xs, bets))

    lower, upper = 0.0, 1.0
    if log_capital(0.0, True) >= threshold:
        a, b = 0.0, 1.0
        for _ in range(48):
            m = (a + b) / 2
            if log_capital(m, True) >= threshold:
                a = m
            else:
                b = m
        lower = a                     # outward endpoint, never exclude a grid sliver
    if log_capital(1.0, False) >= threshold:
        a, b = 0.0, 1.0
        for _ in range(48):
            m = (a + b) / 2
            if log_capital(m, False) >= threshold:
                b = m
            else:
                a = m
        upper = b
    return lo + (hi - lo) * lower, lo + (hi - lo) * upper


def sign_test(values, null=0.0):
    """dict(p, positive, negative, ties): the exact two-sided binomial sign test of values against null. Values equal
    to null are dropped and the smaller tail is doubled. The null hypothesis is that, given a value other than null,
    either sign has probability 1/2; with atoms this differs from null being a median. p is 1 when every value equals
    null.
    """
    xs = _values(values)
    if not math.isfinite(null):
        raise ValueError('null must be finite')
    positive, negative = sum(x > null for x in xs), sum(x < null for x in xs)
    n = positive + negative
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return dict(p=min(1.0, 2 * tail / (1 << n)), positive=positive, negative=negative,
                ties=len(xs) - n)


def median_ci(values, alpha=0.05, bounds=(-math.inf, math.inf)):
    """(X_(k), X_(n-k+1)): the exact equal-tail order-statistic interval for a population median, with the largest k
    such that P[Bin(n, 1/2) < k] <= alpha/2; k = 0 gives the support bounds (unbounded by default). Coverage is at
    least 1 - alpha, also for discrete distributions and ties.
    """
    _alpha(alpha)
    xs = sorted(_values(values))
    lo, hi = bounds
    if math.isnan(lo) or math.isnan(hi) or lo > hi or any(x < lo or x > hi for x in xs):
        raise ValueError('invalid support bounds')
    n, tail, k = len(xs), 0, 0
    for j in range(1, n // 2 + 1):
        tail += math.comb(n, j - 1)
        if tail / (1 << n) > alpha / 2:
            break
        k = j
    return (xs[k - 1], xs[n - k]) if k else (lo, hi)
