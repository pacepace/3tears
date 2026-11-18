"""Decision quality of candidate "should I switch?" rules, under one look and under a peeker who looks after every
case and acts on the first "switch". Seeded; pure numpy over core.py (checked against the package by check_core.py).

Usage: python sim.py [reps] > out.txt
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import core  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4000
NMAX = 100
LOOKS = (8, 24, 100)
EFFECTS = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
FINE = tuple(round(x, 3) for x in np.arange(-0.30, 0.3001, 0.025))


def draw(model: str, effect: float, rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
    """Lattice indices (reps, n) of the paired per-case differences, i = 3 + 3 * diff."""
    if model == "passfail":
        base = rng.uniform(0.2, 0.7, size=(reps, n))
        a = rng.random((reps, n)) < base
        b = rng.random((reps, n)) < np.clip(base + effect, 0, 1)
        return 3 + 3 * (b.astype(int) - a.astype(int))
    if model == "onramp":
        base = rng.uniform(0.1, 0.6, size=(reps, n))
        sa = np.clip(base + rng.normal(0, 0.1, size=(reps, n)), 0, 1)
        sb = np.clip(base + effect + rng.normal(0, 0.1, size=(reps, n)), 0, 1)
        a = (rng.random((reps, n, 3)) < sa[..., None]).sum(-1)
        b = (rng.random((reps, n, 3)) < sb[..., None]).sum(-1)
        return 3 + (b - a)
    if model == "extreme":
        # The most spread a mean-delta difference on [-1, 1] can have: every case a full win or a full loss.
        up = rng.random((reps, n)) < (1 + effect) / 2
        return np.where(up, 6, 0)
    if model == "skewed":
        # A graded difference whose mean is delta but which is not symmetric: a third of the range gained on most
        # cases, the whole range lost on the rest (q / 3 - (1 - q) = delta).
        q = (3 * effect + 3) / 4
        return np.where(rng.random((reps, n)) < q, 4, 0)
    raise ValueError(model)


def true_effect(model: str, effect: float) -> float:
    rng = np.random.default_rng(99)
    idx = draw(model, effect, rng, 400, 500)
    return float(((idx - 3) / 3).mean())


def rules(idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per look (reps, NMAX): whether each rule says switch; plus the estimate and evidence for the three-way split."""
    onehot = np.eye(7)[idx]
    counts = np.cumsum(onehot, axis=1)
    n = np.arange(1, idx.shape[1] + 1)[None, :]
    est = (counts @ core.VALUES) / n
    up, down = core.log_up_down(counts)
    tilted = np.logaddexp(up + math.log(0.9), down + math.log(0.1))
    crossed = tilted >= math.log(20)
    first = np.where(crossed.any(1), crossed.argmax(1), idx.shape[1])
    up_side = np.take_along_axis(
        up + math.log(0.9) >= down + math.log(0.1), np.minimum(first, idx.shape[1] - 1)[:, None], 1
    )[:, 0]
    decided_up = (np.arange(idx.shape[1])[None, :] >= first[:, None]) & up_side[:, None]
    out = {
        "est": est,
        "E_up": np.exp(up),
        "E_down": np.exp(down),
        "today": decided_up,
        "leader": est > 0,
        "ev3": up >= math.log(3),
        "ev10": up >= math.log(10),
        "ev2": up >= math.log(2),
    }
    # The advice read off the evidence's path, as the decision is: the evidence for the arm has reached the bar at some
    # count so far, the arm is still ahead now, and the evidence against it has not reached the bar now.
    for c in (3, 10):
        reached = np.maximum.accumulate(up, axis=1) >= math.log(c)
        out[f"ev{c}s"] = reached & (est > 0) & (down < math.log(c))
    # The evidence that the arm is no worse than a margin below the control: the upward mixture against -margin.
    for margin in (0.05, 0.10):
        m = (1.0 - margin) / 2.0
        out[f"ev3m{int(margin * 100)}"] = (core.mix(counts @ core.up_table(m).T) >= math.log(3)) & (est > 0)
    for margin in (0.0, 0.05, 0.10):
        m = (1.0 - margin) / 2.0
        hu = core.mix(counts @ core.up_table(m).T)
        hd = core.mix(counts @ core.down_table(m).T)
        excluded = np.logaddexp(hu, hd) + math.log(0.5) >= math.log(10)
        out[f"cs{int(margin * 100)}"] = excluded & (est > 0)
    mu, sd = core.dp_posterior(counts, a=1.0)
    pb = core.p_better(mu, sd)
    out["p90"] = pb >= 0.9
    out["p80"] = pb >= 0.8
    out["el1"] = core.expected_loss_switch(mu, sd) <= 0.01
    for key in list(out):
        if key not in ("est", "E_up", "E_down"):
            out[key][:, 0] = False  # no rule reads a single case
    return out


RULES = (
    "today",
    "leader",
    "ev2",
    "ev3",
    "ev10",
    "ev3s",
    "ev10s",
    "ev3m5",
    "ev3m10",
    "cs0",
    "cs5",
    "cs10",
    "p80",
    "p90",
    "el1",
)


def cell(model: str, effect: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = draw(model, effect, rng, REPS, NMAX)
    r = rules(idx)
    res = {}
    for name in RULES:
        says = r[name]
        for look in LOOKS:
            once = says[:, look - 1].mean()
            ever = says[:, :look].any(1).mean()
            res[(name, look)] = (once, ever)
    # three-way split of the proposed rule at one look: switch / ahead but under the bar / keep (level or behind)
    for c, key in ((3, "ev3s"), (10, "ev10s")):
        for look in LOOKS:
            sw = r[key][:, look - 1]
            ahead = (r["est"][:, look - 1] > 0) & ~sw
            res[("split", c, look)] = (sw.mean(), ahead.mean(), 1 - sw.mean() - ahead.mean())
    res["null_ever_up3"] = (r["E_up"][:, 1:] >= 3).any(1).mean()
    return res


def regret(delta: float, p_switch: float, cost: float = 0.0) -> float:
    """Expected regret in points: the best action's value less the chosen one's (switch = delta - cost, keep = 0)."""
    best = max(delta - cost, 0.0)
    return 100 * (best - p_switch * (delta - cost))


def main() -> None:
    print(f"reps {REPS} a cell; looks after every case from 2 to {NMAX}; seeds 1000 + cell index")
    models = ("passfail", "onramp", "extreme", "skewed")
    results = {}
    k = 0
    for model in models:
        for effect in sorted(set(EFFECTS) | set(FINE)):
            if model == "skewed" and effect > 0.33:
                continue
            k += 1
            results[(model, effect)] = (true_effect(model, effect), cell(model, effect, 1000 + k))
            print(f"# done {model} {effect:+.3f}", file=sys.stderr, flush=True)
    import pickle

    with open(__file__.rsplit("/", 1)[0] + f"/results_{REPS}.pkl", "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
