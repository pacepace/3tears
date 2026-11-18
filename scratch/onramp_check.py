"""Reproduce the held-out-1 primary through the package's own functions (bp-decide's src)."""

from fractions import Fraction
from threetears.evals.analysis import stats as s

TALLY = [
    (0, 3),
    (2, 3),
    (3, 3),
    (3, 3),
    (2, 3),
    (1, 2),
    (3, 3),
    (2, 1),
    (3, 3),
    (3, 3),
    (3, 3),
    (1, 2),
    (1, 2),
    (0, 1),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 2),
    (2, 2),
    (0, 3),
]
a = [Fraction(c, 3) for c, _ in TALLY]
b = [Fraction(t, 3) for _, t in TALLY]
d = [y - x for x, y in zip(a, b)]
print(
    "n",
    len(d),
    "mean",
    float(sum(d) / len(d)),
    "wins",
    sum(x > 0 for x in d),
    "ties",
    sum(x == 0 for x in d),
    "losses",
    sum(x < 0 for x in d),
)
print("p toward 0.9", s.bounded_separation_p(a, b, paired=True, value_range=(0, 1), mixture=True, toward=0.9))
print("evidence (up, down)", s.bet_evidence(a, b, value_range=(0, 1)))
for conf in (0.9, 0.8):
    print(
        "CS current",
        conf,
        s.bounded_difference_interval(
            a, b, paired=True, value_range=(0, 1), confidence=conf, mixture=True, current=True
        ),
    )
print("CS running", s.bounded_difference_interval(a, b, paired=True, value_range=(0, 1), confidence=0.9, mixture=True))
