"""The scope-divergence movement rule, checked against data with a known truth (#601).

A scope divergence sets a lever's movement on an end-to-end measure beside its movement on the part under
test (``total_ms`` beside ``tool_ms``), and grades each movement against its own noise: ``improved`` or
``regressed`` only when the difference of the two levels' means clears TWO standard errors of that
difference, the SE propagated from each level's SEM; ``flat`` otherwise. The bundle publishes
``se_of_delta`` and ``direction`` on every :class:`~threetears.evals.analysis.MeasureMovement`.

Two blocks:

- ``TestTheBundleAppliesTheRule`` pins the published movements to the rule on seeded sweeps: ``se_of_delta``
  is ``sqrt(SEM_a² + SEM_b²)`` over each level's observations, and ``direction`` is flat exactly when
  ``|delta| < 2·se_of_delta``.
- ``TestNoMovementIsReadAsFlat`` runs that rule on levels with no true difference. "Two standard errors" is
  the conventional bar the bundle's own comment invokes for a difference of means, so a reader takes a
  non-flat direction as a 5%-level call.
- ``TestTheLensSaysTheWholeAndThePartDisagree`` runs the lens itself: a divergence is published where the
  whole's direction and the part's differ, which is a claim that the whole moved by something the part does
  not account for.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis import MeasureMovement, assemble_context_bundle
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, standard_error_of_mean
from threetears.evals.contracts import EvalCampaign, EvalResult, EvalRun, LatencyMetrics
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import at_most

#: The multiple of the difference's SE a movement must clear, as the bundle states it.
_SE_MULTIPLE = 2.0


def _batch(chunk_tokens: int) -> EvalRun:
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )


def _sweep_movements(
    rng: random.Random, *, n_cases: int, whole_shift: float
) -> tuple[list[tuple[str, str, MeasureMovement]], dict[tuple[str, str], list[float]]]:
    """A chunk-width sweep at one repeat per case, with seeded wall-clock and tool time per observation.

    ``total_ms`` (the whole) shifts by ``whole_shift`` at the wide level; ``tool_ms`` (the part) does not
    move. Every observation is independent normal noise around its level.

    Returns:
        Every movement of ``total_ms`` or ``tool_ms`` the bundle's divergences publish, with the two levels it
        runs from and to, and each ``(measure, level)``'s observations as written.
    """
    profile = toyhost_profile()
    batches = (_batch(256), _batch(1024))
    written: dict[tuple[str, str], list[float]] = {}
    results: dict[str, list[EvalResult]] = {}
    for level, batch in enumerate(batches):
        members = []
        for result in toyhost_measurements(batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            if result.test_case_id not in TOYHOST_DOCUMENTS[:n_cases] or result.k_iteration != 1:
                continue
            total = round(2000.0 + whole_shift * level + rng.gauss(0.0, 40.0), 3)
            tool = round(300.0 + rng.gauss(0.0, 20.0), 3)
            label = ("256", "1024")[level]
            written.setdefault(("total_ms", label), []).append(total)
            written.setdefault(("tool_ms", label), []).append(tool)
            members.append(result.model_copy(update={"latency": LatencyMetrics(total_ms=total, tool_ms=tool)}))
        results[batch.id] = members
    campaign = EvalCampaign(
        id="7c2d9e41-5b3a-4f68-9d17-8e4a2c6b0f35",
        scope_id=TOYHOST_SCOPE,
        name="divergence",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(list(batches), results), profile=profile)
    assert all(
        divergence.end_to_end.direction != divergence.subsystem.direction for divergence in bundle.scope_divergences
    ), "the lens publishes a divergence exactly where the whole's direction and the part's differ"
    movements = [
        (divergence.level_a, divergence.level_b, movement)
        for divergence in bundle.scope_divergences
        if divergence.lever == "chunk_tokens"
        for movement in (divergence.end_to_end, divergence.subsystem, *divergence.whole_components)
        if movement.name in ("total_ms", "tool_ms")
    ]
    return movements, written


def _rule_direction(delta: float, se: float | None, higher_is_better: bool) -> str:
    if se is None or delta == 0.0 or abs(delta) < _SE_MULTIPLE * se:
        return "flat"
    return "improved" if (delta > 0) == higher_is_better else "regressed"


class TestTheBundleAppliesTheRule:
    @pytest.mark.parametrize("seed", ["a", "b", "c"])
    def test_each_published_movement_is_the_rule(self, seed: str) -> None:
        movements, written = _sweep_movements(random.Random(f"divergence-rule-{seed}"), n_cases=6, whole_shift=400.0)
        assert {movement.name for _, _, movement in movements} == {"total_ms", "tool_ms"}, (
            "the sweep must publish a divergence carrying both seeded measures"
        )
        for level_a, level_b, movement in movements:
            before, after = written[(movement.name, level_a)], written[(movement.name, level_b)]
            sem_before, sem_after = standard_error_of_mean(before), standard_error_of_mean(after)
            assert sem_before is not None and sem_after is not None
            assert movement.se_of_delta == pytest.approx(math.sqrt(sem_before**2 + sem_after**2), rel=1e-9)
            assert movement.delta == pytest.approx(sum(after) / len(after) - sum(before) / len(before), rel=1e-9)
            # Wall-clock: lower is better.
            assert movement.direction == _rule_direction(movement.delta, movement.se_of_delta, False), movement.name
        whole = next(movement for _, _, movement in movements if movement.name == "total_ms")
        assert whole.direction != "flat", "the seeded whole moves far past its noise"


def _false_direction_rate(n_per_level: int, replicates: int, seed: str) -> float:
    """The rule's non-flat share over two levels of ``n_per_level`` independent observations with equal means."""
    rng = random.Random(seed)
    calls = 0
    for _ in range(replicates):
        narrow = [rng.gauss(0.0, 1.0) for _ in range(n_per_level)]
        wide = [rng.gauss(0.0, 1.0) for _ in range(n_per_level)]
        sem_narrow, sem_wide = standard_error_of_mean(narrow), standard_error_of_mean(wide)
        assert sem_narrow is not None and sem_wide is not None
        delta = sum(wide) / n_per_level - sum(narrow) / n_per_level
        calls += _rule_direction(delta, math.sqrt(sem_narrow**2 + sem_wide**2), True) != "flat"
    return calls / replicates


class TestNoMovementIsReadAsFlat:
    """With no true difference between the levels, a movement reads non-flat at most α of the time."""

    def test_with_many_observations_the_bar_is_the_conventional_one(self) -> None:
        """At 30 observations a level two SEs is close to the large-sample 95% bar (t on ~58 df: 0.050).

        6,000 replicates: SE at α is 0.0028, so the bound is 0.061.
        """
        rate = _false_direction_rate(30, 6000, "divergence-null-30")
        assert rate <= at_most(SIGNIFICANCE_ALPHA, 6000), f"false direction rate {rate:.4f}"

    @pytest.mark.parametrize(
        ("n_per_level", "replicates"),
        # The replicates put each measured rate at least 3.5 of its own SEs above the 4-SE bound: 6,000 give
        # a bound of 0.061 (measured 0.106 and 0.082); 24,000 give 0.0556 (measured 0.062).
        [(3, 6000), (5, 6000), (10, 24000)],
    )
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#601 finding: the movement rule reads a fixed 2 SE where the difference of two small samples needs "
            "Student's t (about 2.78 at 3 observations a level, 2.31 at 5, 2.10 at 10). With independent "
            "observations and no true difference, measured non-flat rate 0.106 (3 a level), 0.082 (5), 0.062 "
            "(10) against nominal 0.05. (Where the levels share cases with a large between-case spread the same "
            "rule is instead very conservative: the SEMs carry the case spread a paired difference would cancel.)"
        ),
    )
    def test_with_few_observations_the_bar_holds_alpha(self, n_per_level: int, replicates: int) -> None:
        rate = _false_direction_rate(n_per_level, replicates, f"divergence-null-{n_per_level}")
        assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates), (
            f"{n_per_level} a level: false direction rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )


def _lens_fires_rate(n_per_level: int, part_shift: float, replicates: int, seed: str) -> float:
    """How often the lens publishes a divergence when the part carries ALL of the whole's movement.

    Each observation's whole is its part plus a remainder; the lever shifts the part by ``part_shift`` SDs and
    leaves the remainder alone, so the whole moves by exactly what the part moves and there is no divergence
    to find. The lens grades each movement on its own 2-SE bar and publishes when the two directions differ.
    """
    rng = random.Random(seed)
    fires = 0
    for _ in range(replicates):
        directions = []
        parts = [[rng.gauss(part_shift * level, 1.0) for _ in range(n_per_level)] for level in (0, 1)]
        remainders = [[rng.gauss(0.0, 1.0) for _ in range(n_per_level)] for _ in (0, 1)]
        wholes = [
            [p + r for p, r in zip(part, rest, strict=True)] for part, rest in zip(parts, remainders, strict=True)
        ]
        for before, after in (wholes, parts):
            sem_before, sem_after = standard_error_of_mean(before), standard_error_of_mean(after)
            assert sem_before is not None and sem_after is not None
            delta = sum(after) / n_per_level - sum(before) / n_per_level
            directions.append(_rule_direction(delta, math.sqrt(sem_before**2 + sem_after**2), False))
        fires += directions[0] != directions[1]
    return fires / replicates


class TestTheLensSaysTheWholeAndThePartDisagree:
    """With the part carrying all of the whole's movement, the lens publishes a divergence at most α of the time.

    The lens compares two verdicts, each made alone — "the whole moved" beside "the part did not" — which is
    the difference between a significant and a non-significant result, not a test of the difference (Gelman &
    Stern 2006). The test that answers the lens's question is one on the remainder's own movement.
    """

    #: 3,000 replicates: SE at α is 0.0040, so the bound is 0.066.
    REPLICATES = 3000

    @pytest.mark.parametrize(("n_per_level", "part_shift"), [(5, 0.0), (5, 1.5), (10, 1.0)])
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#601 finding: the scope-divergence lens publishes a divergence whenever the whole's and the part's "
            "separately graded directions differ. With the part carrying all of the whole's movement (no divergence "
            "exists), measured rate 0.11 (5 a level, no movement), 0.33 (5 a level, part moving 1.5 SD) and 0.33 "
            "(10 a level, 1 SD) against nominal 0.05: the whole's noise is larger, so it often reads flat where "
            "the part reads moved."
        ),
    )
    def test_no_divergence_is_published_beyond_alpha(self, n_per_level: int, part_shift: float) -> None:
        rate = _lens_fires_rate(n_per_level, part_shift, self.REPLICATES, f"lens-{n_per_level}-{part_shift}")
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_per_level} a level, part shift {part_shift}: divergence published {rate:.4f} against α"
        )
