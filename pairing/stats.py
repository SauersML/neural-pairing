"""Small distribution-free inference tools; no normality or symmetry assumption.

betting_ci: fixed-n hedged capital with predictable plug-in bets, Waudby-Smith
& Ramdas (2024), JRSS B 86(1), doi:10.1093/jrsssb/qkad009, equations 24--27.
https://arxiv.org/html/2010.09686#S4.SS4
The sample must be independent, identically distributed and bounded by the
specified bounds. Choose n before observing outcomes; do not sort the sample.
Sign tests concern positive versus negative probability, NOT the mean.
"""
import math


def _alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')


def _values(values):
    xs = [float(x) for x in values]
    if not all(math.isfinite(x) for x in xs):
        raise ValueError('observations must be finite')
    return xs


def _bets(xs, alpha):
    """Equation 27, using ONLY observations strictly before the current bet."""
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
    """Closed conservative numerical inversion of terminal hedged capital.

    Normalize to [0,1]; theta=c=1/2. Reject m if either one-sided capital
    reaches 2/alpha. Bets use the regularized mean/variance in eq. 26 and the
    fixed horizon in eq. 27. Only TERMINAL capital is used (no optional stopping
    across repeated calls with different n). Each one-sided capital is monotone
    in m, so bisection brackets the whole set without grid discretization loss.
    Return (lower, upper); lower > upper denotes an empty confidence set.
    Empty data returns the full known range. Bounds must be known in advance.
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
    """Exact two-sided binomial sign test; drop exact ties, double smaller tail.

    H0: conditional on a non-tie, either sign has probability 1/2. This does
    not test a zero mean, and with atoms it need not test 'null is a median'.
    Returns p=1 for no non-ties, and the three counts for transparent reporting.
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
    """Exact equal-tail binomial order-statistic interval for a population median.

    Choose largest k with P[Bin(n,1/2) < k] <= alpha/2 and return
    [X_(k), X_(n-k+1)]. k=0 uses the known support (unbounded by default).
    Coverage is at least 1-alpha, including discrete distributions and ties.
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
