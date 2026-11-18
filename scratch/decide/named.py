"""The named cases of the report, read through the package's own functions (bp-decide's src), with the candidate
recommendation statistics beside them, and the prior sensitivity of the probability that the arm is better."""
from __future__ import annotations

import math
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import core  # noqa: E402
from sim import draw  # noqa: E402
from threetears.evals.analysis import stats as s  # noqa: E402

ONRAMP = [(0,3),(2,3),(3,3),(3,3),(2,3),(1,2),(3,3),(2,1),(3,3),(3,3),(3,3),(1,2),(1,2),(0,1),(3,3),(3,3),(3,3),(3,3),(0,3),(1,3),(2,3),(3,2),(2,2),(0,3)]


def tallies_from_draw(model: str, effect: float, n: int, seed: int) -> list[tuple[int, int]]:
    """A seeded on-ramp-model run as per-case (control, arm) tallies out of 3."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.6, size=n)
    sa = np.clip(base + rng.normal(0, 0.1, size=n), 0, 1)
    sb = np.clip(base + effect + rng.normal(0, 0.1, size=n), 0, 1)
    a = (rng.random((n, 3)) < sa[:, None]).sum(-1)
    b = (rng.random((n, 3)) < sb[:, None]).sum(-1)
    return list(zip(a.tolist(), b.tolist()))


CASES = {
    "all wins, 8 cases (pass/fail)": ([F(0)] * 8, [F(1)] * 8, None),
    "+0.3, 8 cases (k=3: 8 vs 15 of 24)": (
        [F(x, 3) for x in (0, 1, 1, 0, 2, 1, 2, 1)], [F(x, 3) for x in (3, 3, 2, 1, 2, 1, 1, 2)], None),
    "on-ramp held-out-1, 24 apps": ([F(c, 3) for c, _ in ONRAMP], [F(t, 3) for _, t in ONRAMP], None),
    "no effect, 24 cases": (None, None, ("onramp", 0.0, 24, 24)),
    "small regression -0.05, 100 cases": (None, None, ("onramp", -0.05, 100, 100)),
    "gate, no worse than 5 points, 40 cases": (None, None, ("onramp", 0.0, 40, 40)),
    "gate, no worse than 5 points, 100 cases": (None, None, ("onramp", 0.0, 100, 140)),
}


def mc_dp(diffs: np.ndarray, a: float, base: str, draws: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Exact Monte Carlo of the Dirichlet-process posterior of the mean difference: weights Dir(1,...,1, a) over the
    observed differences and one prior atom, the atom's value itself the mean of a draw from the base (for a
    Dirichlet process the prior part is a DP(a, base) whose mean is drawn here by stick-breaking)."""
    n = diffs.size
    g = rng.gamma(1.0, size=(draws, n))
    if a > 0:
        ga = rng.gamma(a, size=(draws, 1))
        # The prior part's mean: a DP(a, base)'s mean, by 60 sticks.
        sticks = rng.beta(1.0, a, size=(draws, 60))
        rem = np.cumprod(np.hstack([np.ones((draws, 1)), 1 - sticks[:, :-1]]), axis=1)
        w = sticks * rem
        atoms = rng.uniform(-1, 1, size=(draws, 60)) if base == "uniform" else rng.choice([-1.0, 1.0], size=(draws, 60))
        prior_mean = (w * atoms).sum(1) / w.sum(1)
        total = g.sum(1, keepdims=True) + ga
        delta = (g @ diffs[:, None])[:, 0] / total[:, 0] + (ga[:, 0] / total[:, 0]) * prior_mean
    else:
        delta = (g @ diffs[:, None])[:, 0] / g.sum(1)
    return float((delta > 0).mean()), float(np.maximum(-delta, 0).mean()), float(np.maximum(delta, 0).mean()), float(np.quantile(delta, 0.05))


def skeptical(diffs: np.ndarray, tau: float) -> float:
    """P(delta > 0) under a normal prior N(0, tau^2) and a normal likelihood with the sample's sd (floored at the
    sd of one case in n moving the full width, so a sample with no spread still has some)."""
    n = diffs.size
    sd = max(float(diffs.std(ddof=1)), 1.0 / math.sqrt(n))
    prec = 1 / tau**2 + n / sd**2
    mu = (n / sd**2) * diffs.mean() / prec
    return float(core.norm_cdf(mu * math.sqrt(prec)))


def main() -> None:
    rng = np.random.default_rng(20261118)
    for name, (a, b, spec) in CASES.items():
        if spec is not None:
            model, effect, n, seed = spec
            # The first seed from the stated one whose estimate is within 1.5 points of the true effect: a typical draw,
            # not a lucky or unlucky one.
            while True:
                t = tallies_from_draw(model, effect, n, seed)
                if abs(sum(y - x for x, y in t) / (3 * n) - effect) <= 0.015:
                    break
                seed += 1
            name = f"{name} (seed {seed})"
            a = [F(x, 3) for x, _ in t]
            b = [F(y, 3) for _, y in t]
        diffs = np.array([float(y - x) for x, y in zip(a, b)])
        idx = np.rint(diffs * 3).astype(int) + 3
        counts = np.bincount(idx, minlength=7).astype(float)
        up, down = s.bet_evidence(a, b, value_range=(0, 1))
        p_tilt = s.bounded_separation_p(a, b, paired=True, value_range=(0, 1), mixture=True, toward=0.9)
        p_gate = s.bounded_separation_p(a, b, paired=True, value_range=(0, 1), mixture=True, toward=0.1)
        ci = s.bounded_difference_interval(a, b, paired=True, value_range=(0, 1), confidence=0.9, mixture=True, current=True)
        dec = s.bet_decided_at(a, b, value_range=(0, 1), toward=0.9)
        # The evidence that the arm is no worse than 5 points below the control: the upward mixture against -0.05.
        m5 = (1 - 0.05) / 2
        up_m5 = math.exp(float(core._mix(counts @ core._up_table(m5).T)))
        mu1, sd1 = core.dp_posterior(counts, a=1.0)
        print(f"== {name}")
        print(f"   n {len(diffs)}  control {sum(a)*3 if spec or len(a)!=8 or name.startswith('+') else sum(a)} arm-sum  "
              f"wins {int((diffs>0).sum())} ties {int((diffs==0).sum())} losses {int((diffs<0).sum())}  "
              f"estimate {100*diffs.mean():+.1f} pts  interval90 ({100*ci[0]:+.1f}, {100*ci[1]:+.1f})")
        if spec is not None:
            print(f"   tallies (control, arm) of 3: {[(int(x*3), int(y*3)) for x, y in zip(a, b)]}")
        print(f"   E_up {up:.3g}  E_down {down:.3g}  tilted-better p {p_tilt:.4f}  gate-lean p {p_gate:.4f}  decided {dec}"
              f"  E_up vs -5pts {up_m5:.3g}")
        print(f"   P(better), normal approx DP a=1 uniform base: {float(core.p_better(mu1, sd1)):.4f};"
              f" exp. loss switch {100*float(core.expected_loss_switch(mu1, sd1)):.2f} pts,"
              f" keep {100*float(core.expected_loss_switch(-mu1, sd1)):.2f} pts")
        for a_, base in ((0.0, "-"), (1.0, "uniform"), (2.0, "uniform"), (2.0, "ends"), (8.0, "uniform")):
            pb, els, elk, q05 = mc_dp(diffs, a_, base, 40000, rng)
            print(f"   MC DP a={a_:g} base={base:8s}: P(better) {pb:.4f}  E-loss switch {100*els:.2f}  keep {100*elk:.2f}  5% quantile {100*q05:+.1f}")
        lap = rng.dirichlet(counts + 1.0, size=40000) @ core.VALUES
        print(f"   lattice Dirichlet(1 each of 7): P(better) {(lap>0).mean():.4f}")
        for tau in (0.05, 0.1, 0.2):
            print(f"   skeptical normal prior sd {tau:g}: P(better) {skeptical(diffs, tau):.4f}")
        print(f"   E_up/(E_up+E_down) {up/(up+down):.4f}  proof needs E_up {(20 - 0.1 * down) / 0.9:.1f}")
        lo_gate = s.bounded_difference_interval(a, b, paired=True, value_range=(0, 1), confidence=0.9, mixture=True)
        print(f"   running-intersection interval90 {lo_gate}")


if __name__ == "__main__" and "--path" not in sys.argv:
    main()


def path_max() -> None:
    """The most the evidence for the arm reached over the cases in the order given, which the advice reads."""
    for name, (a, b, spec) in CASES.items():
        if spec is not None:
            model, effect, n, seed = spec
            while True:
                t = tallies_from_draw(model, effect, n, seed)
                if abs(sum(y - x for x, y in t) / (3 * n) - effect) <= 0.015:
                    break
                seed += 1
            a = [F(x, 3) for x, _ in t]
            b = [F(y, 3) for _, y in t]
        idx = [int((y - x) * 3) + 3 for x, y in zip(a, b)]
        path = np.cumsum(np.eye(7)[idx], axis=0)
        up, down = core.log_up_down(path)
        m5 = core._mix(path @ core._up_table((1 - 0.05) / 2).T)
        print(f"   no worse than -5 pts: reached at most {math.exp(m5.max()):.3g} (at case {int(m5.argmax()) + 1}); now {math.exp(m5[-1]):.3g}")
        print(f"{name}: evidence for the arm reached at most {math.exp(up.max()):.3g} (at case {int(up.argmax()) + 1}); now {math.exp(up[-1]):.3g}; against now {math.exp(down[-1]):.3g}")


if __name__ == "__main__" and "--path" in sys.argv:
    path_max()
