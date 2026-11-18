"""Tables of the report from sim.py's pickled results."""
from __future__ import annotations

import math
import pickle
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sim import EFFECTS, LOOKS, RULES, regret  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
res = pickle.load(open(__file__.rsplit("/", 1)[0] + f"/results_{REPS}.pkl", "rb"))

NAMES = {
    "today": "today: act only when decided",
    "leader": "always switch to the leader",
    "ev3": "evidence for v2 >= 3 (moderate)",
    "ev10": "evidence for v2 >= 10 (strong)",
    "ev2": "evidence for v2 >= 2",
    "ev3s": "ADVICE, easy to undo: evidence for v2 has reached 3, still ahead",
    "ev10s": "ADVICE, hard to undo: evidence for v2 has reached 10, still ahead",
    "ev3m5": "evidence v2 is no worse than -5 pts >= 3",
    "ev3m10": "evidence v2 is no worse than -10 pts >= 3",
    "cs0": "worst plausible case >= 0",
    "cs5": "worst plausible case >= -5 pts",
    "cs10": "worst plausible case >= -10 pts",
    "p80": "P(better) >= 0.8 (DP a=1)",
    "p90": "P(better) >= 0.9 (DP a=1)",
    "el1": "expected loss of switching <= 1 pt",
}


def cp(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95%, through the package's own public implementation."""
    from threetears.evals.analysis.stats import exact_rate_interval

    held = exact_rate_interval(k, n)
    assert held is not None
    return held


def fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    for model in ("passfail", "onramp", "extreme", "skewed"):
        print(f"\n### {model}\n")
        for protocol, slot in (("one look at n", 0), ("every look to n, act on the first switch", 1)):
            print(f"\n**{model}, {protocol}**: share told to switch, by true effect (realised effect in brackets), "
                  "and expected regret in points at cost 0\n")
            heads = [f"{e:+.1f} [{res[(model, e)][0]:+.3f}]" for e in EFFECTS if (model, e) in res]
            for look in LOOKS:
                print(f"\nn = {look}\n")
                print("| rule | " + " | ".join(heads) + " | worst expected regret (pts) over -0.3..+0.3 |")
                print("|---|" + "---|" * (len(heads) + 1))
                for rule in RULES:
                    cells = []
                    for e in EFFECTS:
                        if (model, e) not in res:
                            continue
                        delta, r = res[(model, e)]
                        p = r[(rule, look)][slot]
                        cells.append(f"{fmt(p)} / {regret(delta, p):.1f}")
                    worst = max(
                        (regret(res[k][0], res[k][1][(rule, look)][slot]), k[1]) for k in res if k[0] == model
                    )
                    print(f"| {NAMES[rule]} | " + " | ".join(cells) + f" | {worst[0]:.1f} at {worst[1]:+.3f} |")
    print("\n### Three-way split of the advice (read off the path) at one look: switch / ahead but below the bar / level or behind\n")
    for model in ("passfail", "onramp"):
        for c in (3, 10):
            print(f"\n{model}, bar {c}\n")
            print("| n | " + " | ".join(f"{e:+.1f}" for e in EFFECTS) + " |")
            print("|---|" + "---|" * len(EFFECTS))
            for look in LOOKS:
                print(f"| {look} | " + " | ".join(
                    "{:.2f} / {:.2f} / {:.2f}".format(*res[(model, e)][1][("split", c, look)]) for e in EFFECTS) + " |")
    print("\n### Told to switch to an arm that is no better: the most over every effect <= 0, every look to 100 [CP 95%]\n")
    print("| rule | " + " | ".join(("passfail", "onramp", "extreme", "skewed")) + " |")
    print("|---|---|---|---|---|")
    for rule in RULES:
        row = []
        for model in ("passfail", "onramp", "extreme", "skewed"):
            worst = max((res[k][1][(rule, 100)][1], k[1]) for k in res if k[0] == model and k[1] <= 0.0)
            kk = round(worst[0] * REPS)
            lo, hi = cp(kk, REPS)
            row.append(f"{worst[0]:.3f} [{lo:.3f}, {hi:.3f}] at {worst[1]:+.3f}")
        print(f"| {NAMES[rule]} | " + " | ".join(row) + " |")


def summary(cost: float = 0.0) -> None:
    """Worst expected regret over every model and every effect on the fine grid, split by the side it falls on."""
    print(f"\n### Worst expected regret in points over all four data models and every effect from -0.3 to +0.3, switching cost {100 * cost:g} pts\n")
    print("Each cell: worst when the arm is worse (a wrong switch) / worst when the arm is better (a missed gain); one look at n, then every look to n acting on the first switch.\n")
    print("| rule | n=8 one look | n=8 every look | n=24 one look | n=24 every look | n=100 one look | n=100 every look |")
    print("|---|---|---|---|---|---|---|")
    for rule in RULES:
        cells = []
        for look in LOOKS:
            for slot in (0, 1):
                down = max(regret(res[k][0], res[k][1][(rule, look)][slot], cost) for k in res if res[k][0] < cost)
                up = max(regret(res[k][0], res[k][1][(rule, look)][slot], cost) for k in res if res[k][0] >= cost)
                cells.append(f"{down:.1f} / {up:.1f}")
        print(f"| {NAMES[rule]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
    summary(0.0)
    summary(0.05)
