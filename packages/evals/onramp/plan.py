"""A comparison's plan, committed before its run: the arms, the apps, ``k``, the tiers, the check and the test.

A plan is a TOML file under ``plans/`` (``plans/held-out-1.toml`` is the first). ``--plan PATH`` runs exactly what it
states: its apps, its ``k``, each of its tiers into a directory of its own with that tier's cap, and each of its arms
from the wheel built at that arm's commit. A live run on a plan is refused unless the plan is committed and unchanged
since, its apps are named (no placeholder is left), and everything the command line says agrees with it. The run
records the plan's path and the commit it was last changed in (``run.json``), so a reader can check that the plan was
written before the run.

A plan names its arms by commit, never by a wheel path: the wheel an arm runs is the one built from its commit
(``runs/wheels/<ref>/``), and a wheel passed on the command line for an arm must be that one.
"""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reading import CHECKS

#: The repository the plans and the arms' commits live in.
REPO = Path(__file__).resolve().parents[3]
#: Where the plans are kept.
PLANS = Path(__file__).resolve().parent / "plans"
#: How an app the coordinator has not yet named is written in a plan's list of apps.
PLACEHOLDER = "PLACEHOLDER"


@dataclass(frozen=True)
class PlannedArm:
    """One arm of a plan: its name and the commit its wheel is built from (the control first)."""

    name: str
    ref: str


@dataclass(frozen=True)
class Continues:
    """The earlier run a plan continues as one comparison: its run (its directory's name under the runs) and the plan
    commit that run recorded."""

    run: str
    plan_commit: str


@dataclass(frozen=True)
class Plan:
    """A plan as its file states it (see the module).

    :ivar tiers: The cap of each tier, in dollars, by tier name, in the order the tiers run.
    """

    path: Path
    name: str
    question: str
    arms: tuple[PlannedArm, ...]
    apps: tuple[str, ...]
    k: int
    tiers: Mapping[str, float]
    primary: str
    secondary: tuple[str, ...]
    #: The files under ``onramp/`` its apps are read from: ``apps.jsonl`` (the development set) unless it says.
    registries: tuple[str, ...] = ("apps.jsonl",)
    #: The earlier run this plan's run continues as one comparison, or ``None`` for a comparison of its own.
    continues: Continues | None = None

    @property
    def cap_usd(self) -> float:
        """The most the whole plan may spend: every tier at its cap."""
        return round(sum(self.tiers.values()), 2)

    def placeholders(self) -> list[str]:
        """The apps the plan has not named yet."""
        return [app for app in self.apps if app.startswith(PLACEHOLDER)]


def load_plan(path: Path) -> Plan:
    """The plan in ``path``.

    :raises ValueError: A file that is not a plan: a field missing or of the wrong shape, a check the harness does not
        grade, fewer than two arms, or a tier with no positive cap.
    """
    raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))  # Any: TOML
    try:
        arms = tuple(PlannedArm(str(arm["name"]), str(arm["ref"])) for arm in raw["arms"])
        tiers = {str(name): float(tier["cap_usd"]) for name, tier in raw["tiers"].items()}
        plan = Plan(
            path=path,
            name=str(raw["name"]),
            question=str(raw["question"]),
            arms=arms,
            apps=tuple(str(app) for app in raw["apps"]),
            k=int(raw["k"]),
            tiers=tiers,
            primary=str(raw["check"]["primary"]),
            secondary=tuple(str(check) for check in raw["check"]["secondary"]),
            registries=tuple(str(name) for name in raw.get("cases", ["apps.jsonl"])),
            continues=(
                Continues(str(raw["continues"]["run"]), str(raw["continues"]["plan_commit"]))
                if "continues" in raw
                else None
            ),
        )
    except (KeyError, TypeError) as unshaped:
        raise ValueError(f"{path} is not a plan: {unshaped!r} is missing or misshapen") from unshaped
    if len(plan.arms) < 2 or len({arm.name for arm in plan.arms}) != len(plan.arms):
        raise ValueError(f"{path} is not a plan: it names two or more arms, each once")
    if unknown := [check for check in (plan.primary, *plan.secondary) if check not in CHECKS]:
        raise ValueError(f"{path} is not a plan: {', '.join(unknown)} is no check the harness grades")
    if any("/" in name or not name.endswith(".jsonl") for name in plan.registries):
        raise ValueError(f"{path} is not a plan: its cases are files beside apps.jsonl, named by file name")
    if plan.k < 1 or not plan.apps or any(cap <= 0 for cap in plan.tiers.values()) or not plan.tiers:
        raise ValueError(f"{path} is not a plan: k, the apps and every tier's cap must be stated and positive")
    return plan


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


def commit_of(ref: str) -> str | None:
    """The full commit ``ref`` names in this repository, or ``None`` when it names none."""
    done = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return done.stdout.strip() if done.returncode == 0 else None


def head_commit() -> str | None:
    """The commit checked out here, which a run's code, apps and twins are as of; ``None`` outside a repository."""
    done = _git("rev-parse", "--verify", "--quiet", "HEAD")
    return done.stdout.strip() if done.returncode == 0 else None


#: What a live run on a plan must have committed: the harness, its apps, twins and transcripts, and the package.
GUARDED = "packages/evals"


def uncommitted(path: str = GUARDED) -> list[str]:
    """Every change under ``path`` that is not committed (``git status --porcelain``), one line each; ignored files,
    the runs among them, are none. Outside a repository, one line saying so."""
    done = _git("status", "--porcelain", "--", path)
    if done.returncode != 0:
        return [f"not a repository ({done.stderr.strip()})"]
    return [line for line in done.stdout.splitlines() if line.strip()]


def committed_at(path: Path) -> str | None:
    """The commit ``path`` was last changed in, when the file is committed and unchanged since; else ``None``."""
    try:
        relative = str(path.resolve().relative_to(REPO))
    except ValueError:  # a file outside the repository has no commit here
        return None
    if _git("ls-files", "--error-unmatch", "--", relative).returncode != 0:
        return None
    if _git("status", "--porcelain", "--", relative).stdout.strip():
        return None
    done = _git("log", "-1", "--format=%H", "--", relative)
    return done.stdout.strip() or None


def wheel_dir(workdir: Path, ref: str) -> Path:
    """Where the wheel built from ``ref`` is kept: ``scripts/onramp-agent.sh`` builds the control's there too."""
    return workdir / "wheels" / ref


def built_wheel(workdir: Path, ref: str) -> Path | None:
    """The wheel already built from ``ref``, or ``None``."""
    found = sorted(wheel_dir(workdir, ref).glob("3tears_evals-*.whl"))
    return found[0] if found else None


def build_wheel(workdir: Path, ref: str) -> Path:
    """The wheel of ``3tears-evals`` as it stood at ``ref``, built once and reused: as ``scripts/onramp-agent.sh``
    builds the control's.

    :raises subprocess.CalledProcessError: ``git archive`` or ``uv build`` failed.
    """
    if (wheel := built_wheel(workdir, ref)) is not None:
        return wheel
    out = wheel_dir(workdir, ref)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as source:
        archive = Path(source) / "evals.tar"
        subprocess.run(["git", "-C", str(REPO), "archive", "--output", str(archive), ref, "packages/evals"], check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(source, filter="data")
        subprocess.run(
            ["uv", "build", "--quiet", "--wheel", "--out-dir", str(out), str(Path(source) / "packages" / "evals")],
            check=True,
        )
    wheel = built_wheel(workdir, ref)
    if wheel is None:
        raise subprocess.CalledProcessError(1, ["uv", "build"], output=f"no wheel landed in {out}")
    return wheel


def plan_refusal(
    plan: Plan,
    *,
    commit: str | None,
    changed: Sequence[str],
    wheels: Mapping[str, str],
    apps: Sequence[str],
    k: int | None,
    tier: str | None,
    cap_usd: float | None,
    known_apps: Sequence[str],
    workdir: Path,
) -> str | None:
    """Why a live run may not start on ``plan`` with what the command line says, or ``None`` when it may.

    The plan must be committed and unchanged since (``commit``: :func:`committed_at` of it), and so must everything a
    run reads under :data:`GUARDED` (``changed``: :func:`uncommitted`), so the head commit ``run.json`` records is the
    code, apps and twins that ran; the plan must name every app (no placeholder), and only apps the harness has; and the
    command line must agree with it: any ``--apps``, ``--k``
    or ``--tier`` given is the plan's, ``--cap`` is the plan's total, and every ``--arm`` given is an arm of the plan
    whose wheel is the one built from that arm's commit (``runs/wheels/<ref>/``, by the plan's ref or one naming the
    same commit).
    """
    if commit is None:
        return f"the plan {plan.path} is not committed, or has changed since its last commit: commit it before the run"
    if changed:
        listed = "; ".join(changed[:5]) + (f"; and {len(changed) - 5} more" if len(changed) > 5 else "")
        return f"{GUARDED} has changes not committed ({listed}): commit them, so run.json's head commit is what ran"
    if placeholders := plan.placeholders():
        return f"the plan names {len(placeholders)} app(s) still as placeholders: name them, and commit the plan"
    if unknown := [app for app in plan.apps if app not in known_apps]:
        return f"the plan names apps the harness does not have ({', '.join(unknown)}): add them to apps.jsonl first"
    if apps and sorted(apps) != sorted(plan.apps):
        return f"--apps names {', '.join(apps)}, and the plan {', '.join(plan.apps)}: run the plan's apps"
    if k is not None and k != plan.k:
        return f"--k is {k}, and the plan's k is {plan.k}"
    if tier is not None and [tier] != list(plan.tiers):
        return f"--tier is {tier}, and the plan runs every tier it names: {', '.join(plan.tiers)}"
    if cap_usd is None or round(cap_usd, 2) != plan.cap_usd:
        return f"a live run on the plan states its cap as the plan's total: pass --cap {plan.cap_usd:.2f}"
    planned = {arm.name: arm for arm in plan.arms}
    for name, wheel in wheels.items():
        if name not in planned:
            return f"arm {name} is not an arm of the plan ({', '.join(planned)})"
        ref = planned[name].ref
        held = Path(wheel).resolve().parent
        built_from = held.name
        same = built_from == ref or (commit_of(ref) is not None and commit_of(built_from) == commit_of(ref))
        if held != wheel_dir(workdir, built_from).resolve() or not same:
            return f"arm {name}'s wheel {wheel} is not the one built from the plan's {ref} ({wheel_dir(workdir, ref)})"
    return None


def continuation_refusal(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> str | None:  # Any: run.json
    """Why the run ``later`` describes may not continue the run ``earlier`` describes as one comparison, or ``None``
    when it may. Both are ``run.json``'s shape (a planned run's too, before it runs, as ``campaign.run_record`` builds
    it).

    The two must have the same arms, by name and commit and in the same order (the control first, so the direction
    is the same); the same tier and model; the same ``k``; the same primary check; and the same question (which sets
    the lean). The level is the package's, read by one interpreter. No app may be in both: a case is read once.
    """
    said = []
    arms = [(name, arm.get("commit")) for name, arm in earlier.get("arms", {}).items()]
    if arms != [(name, arm.get("commit")) for name, arm in later.get("arms", {}).items()]:
        said.append(
            f"the arms differ ({arms} against {[(n, a.get('commit')) for n, a in later.get('arms', {}).items()]})"
        )
    models = {name: tier.get("model") for name, tier in earlier.get("tiers", {}).items()}
    if models != {name: tier.get("model") for name, tier in later.get("tiers", {}).items()}:
        said.append(f"the tiers or their models differ ({models} against the later run's)")
    for key in ("k", "primary", "question"):
        if earlier.get(key) != later.get(key):
            said.append(f"{key} differs ({earlier.get(key)!r} against {later.get(key)!r})")
    if repeated := sorted(set(earlier.get("apps", [])) & set(later.get("apps", []))):
        said.append(f"an app is read once in a comparison, and {', '.join(repeated)} is in both runs")
    return "; ".join(said) or None
