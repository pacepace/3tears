"""Check core.py against the package's own functions on random lattice samples (bp-decide's src)."""
import math
import random
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import core  # noqa: E402
from threetears.evals.analysis import stats as s  # noqa: E402

rng = random.Random(20261118)
worst_e = worst_l = 0.0
agree_dec = total_dec = 0
for rep in range(300):
    n = rng.randint(2, 30)
    shift = rng.choice([-2, -1, 0, 0, 1, 2])
    a, b = [], []
    for _ in range(n):
        x = rng.randint(0, 3)
        y = min(3, max(0, x + rng.choice([-1, 0, 0, 1, 1]) + (shift if rng.random() < 0.3 else 0)))
        a.append(F(x, 3)); b.append(F(y, 3))
    idx = [int((y - x) * 3) + 3 for x, y in zip(a, b)]
    counts = np.bincount(idx, minlength=7).astype(float)
    up, down = core.log_up_down(counts)
    ref = s.bet_evidence(a, b, value_range=(0, 1))
    worst_e = max(worst_e, abs(math.exp(up) / ref[0] - 1), abs(math.exp(down) / ref[1] - 1))
    lo = core.cs_lower(counts[None, :])[0]
    ci = s.bounded_difference_interval(a, b, paired=True, value_range=(0, 1), confidence=0.9, mixture=True, current=True)
    worst_l = max(worst_l, abs(lo - ci[0]))
    # The running-maximum decision of today's tilted bet, path by path.
    path = np.cumsum(np.eye(7)[idx], axis=0)
    pu, pd = core.log_up_down(path)
    tilted = np.logaddexp(pu + math.log(0.9), pd + math.log(0.1))
    hit = np.nonzero(tilted >= math.log(20))[0]
    mine = None if hit.size == 0 else (int(hit[0]) + 1, bool(pu[hit[0]] + math.log(0.9) >= pd[hit[0]] + math.log(0.1)))
    theirs = s.bet_decided_at(a, b, value_range=(0, 1), toward=0.9)
    total_dec += 1
    agree_dec += mine == (None if theirs is None else theirs[:2])
print(f"evidence: worst relative gap {worst_e:.2e}; CS lower end: worst absolute gap {worst_l:.2e}; "
      f"decision point agrees on {agree_dec} of {total_dec}")
