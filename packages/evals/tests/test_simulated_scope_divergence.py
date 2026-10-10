"""The scope-divergence lens and the test it grades movements with, checked against data with a known truth (#601).

A scope divergence sets a lever's movement on an end-to-end measure beside its movement on the part under
test (``total_ms`` beside ``tool_ms``). It is published only where the DIFFERENCE between the two movements
separates: each case's remainder — its whole minus its part, per-case means — is tested between the two
levels by :func:`~threetears.evals.analysis.stats.level_difference` (paired over shared cases, Welch's
otherwise, read against Student's t), and the lever's tests are Holm-corrected together. Each movement it
shows is graded by the same test, in the #592 vocabulary: ``improved``, ``regressed``, ``not_separated``,
``equivalent`` (only against a declared margin) or ``untested``.

Three blocks:

- ``TestTheBundleAppliesTheRule`` pins the published movements and the divergence's own p to
  ``level_difference`` over the per-case values a seeded sweep wrote.
- ``TestNoMovementIsReadAsMoved`` runs that test on levels with no true difference: it separates at most α
  of the time at 3, 5 and 10 cases a level, and where the levels share cases with a large between-case
  spread it holds α instead of collapsing toward zero.
- ``TestNoDivergenceIsPublishedBeyondAlpha`` runs the lens's own test with the part carrying all of the
  whole's movement, so there is no divergence to find.
"""

from __future__ import annotations

import random

import pytest

from threetears.evals.analysis import MeasureMovement, ScopeDivergence, assemble_context_bundle
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, holm_adjust, level_difference
from threetears.evals.kernel import EvalCampaign
from threetears.evals.schema import EvalResult, EvalRun, LatencyMetrics
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import at_most, within

#: Written values: ``(measure, level label) -> {case: value}``, one observation per case.
_Written = dict[tuple[str, str], dict[str, float]]


def _batch(chunk_tokens: int) -> EvalRun:
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )


def _sweep(rng: random.Random, *, n_cases: int, whole_shift: float) -> tuple[list[ScopeDivergence], _Written]:
    """A chunk-width sweep at one repeat per case, with seeded wall-clock and tool time per observation.

    ``total_ms`` (the whole) shifts by ``whole_shift`` at the wide level; ``tool_ms`` (the part) does not
    move. Every observation is independent normal noise around its level, and both levels run the same cases.

    Returns:
        The chunk-width divergences the bundle publishes, and each ``(measure, level)``'s values as written.
    """
    profile = toyhost_profile()
    batches = (_batch(256), _batch(1024))
    written: _Written = {}
    results: dict[str, list[EvalResult]] = {}
    for level, batch in enumerate(batches):
        members = []
        label = ("256", "1024")[level]
        for result in toyhost_measurements(batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            if result.test_case_id not in TOYHOST_DOCUMENTS[:n_cases] or result.k_iteration != 1:
                continue
            total = round(2000.0 + whole_shift * level + rng.gauss(0.0, 40.0), 3)
            tool = round(300.0 + rng.gauss(0.0, 20.0), 3)
            written.setdefault(("total_ms", label), {})[result.test_case_id] = total
            written.setdefault(("tool_ms", label), {})[result.test_case_id] = tool
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
    return [divergence for divergence in bundle.scope_divergences if divergence.lever == "chunk_tokens"], written


def _rule_direction(values_a: dict[str, float], values_b: dict[str, float], *, higher_is_better: bool) -> str:
    """The direction ``level_difference`` gives a measure that declares no margin."""
    tested = level_difference(values_a, values_b)
    if tested.separated is None:
        return "untested"
    if not tested.separated:
        return "not_separated"
    assert tested.delta is not None
    return "improved" if (tested.delta > 0) == higher_is_better else "regressed"


class TestTheBundleAppliesTheRule:
    @pytest.mark.parametrize("seed", ["a", "b", "c"])
    def test_each_published_movement_is_the_rule(self, seed: str) -> None:
        divergences, written = _sweep(random.Random(f"divergence-rule-{seed}"), n_cases=6, whole_shift=400.0)
        (divergence,) = [d for d in divergences if (d.end_to_end.name, d.subsystem.name) == ("total_ms", "tool_ms")]
        movement: MeasureMovement
        for movement in (divergence.end_to_end, divergence.subsystem):
            # Deltas run from level_a to level_b, which are in name order: "1024" before "256".
            before, after = written[(movement.name, divergence.level_a)], written[(movement.name, divergence.level_b)]
            tested = level_difference(before, after)
            assert movement.test == "paired", "both levels ran the same cases"
            assert (movement.n_a, movement.n_b) == (6, 6)
            assert movement.delta == pytest.approx(sum(after.values()) / 6 - sum(before.values()) / 6, rel=1e-9)
            assert movement.se_of_delta == pytest.approx(tested.se, rel=1e-9)
            # Wall-clock: lower is better.
            assert movement.direction == _rule_direction(before, after, higher_is_better=False), movement.name
        assert divergence.end_to_end.direction == "improved", "the seeded whole falls far past its noise, 1024 → 256"

    @pytest.mark.parametrize("seed", ["a", "b", "c"])
    def test_the_divergence_is_the_test_of_the_remainder(self, seed: str) -> None:
        divergences, written = _sweep(random.Random(f"divergence-rule-{seed}"), n_cases=6, whole_shift=400.0)
        remainders = {
            label: {
                case: written[("total_ms", label)][case] - written[("tool_ms", label)][case]
                for case in written[("total_ms", label)]
            }
            for label in ("256", "1024")
        }
        (divergence,) = [d for d in divergences if (d.end_to_end.name, d.subsystem.name) == ("total_ms", "tool_ms")]
        tested = level_difference(remainders[divergence.level_a], remainders[divergence.level_b])
        assert (divergence.test, divergence.n_cases_a, divergence.n_cases_b) == ("paired", 6, 6)
        assert divergence.p_raw == pytest.approx(tested.p_value, rel=1e-6)
        assert divergence.p_raw <= divergence.p_adjusted < SIGNIFICANCE_ALPHA
        family = [d.p_raw for d in divergences]
        if len(family) == divergence.family_size:
            assert divergence.p_adjusted == pytest.approx(holm_adjust(family)[family.index(divergence.p_raw)])


def _false_separation_rate(
    n_per_level: int, replicates: int, seed: str, *, shared_cases: bool, between_case_sd: float
) -> float:
    """``level_difference``'s separated share over two levels with equal means.

    Each case has a level drawn with ``between_case_sd``, and each observation adds unit noise. With
    ``shared_cases`` both levels observe the same cases (one draw of each case's level); without, each level
    draws its own cases.
    """
    rng = random.Random(seed)
    calls = 0
    for _ in range(replicates):
        cases = [rng.gauss(0.0, between_case_sd) for _ in range(n_per_level)]
        narrow = {f"c{i}": case + rng.gauss(0.0, 1.0) for i, case in enumerate(cases)}
        if not shared_cases:
            cases = [rng.gauss(0.0, between_case_sd) for _ in range(n_per_level)]
        prefix = "c" if shared_cases else "d"
        wide = {f"{prefix}{i}": case + rng.gauss(0.0, 1.0) for i, case in enumerate(cases)}
        calls += level_difference(narrow, wide).separated is True
    return calls / replicates


class TestNoMovementIsReadAsMoved:
    """With no true difference between the levels, a movement separates at most α of the time."""

    @pytest.mark.parametrize(
        ("n_per_level", "replicates"),
        # The old fixed 2-SE rule measured 0.106, 0.082 and 0.062 here (#601); Welch's t holds α. 6,000
        # replicates give a bound of 0.061, 24,000 a bound of 0.0556.
        [(3, 6000), (5, 6000), (10, 24000), (30, 6000)],
    )
    def test_independent_levels_hold_alpha(self, n_per_level: int, replicates: int) -> None:
        rate = _false_separation_rate(
            n_per_level, replicates, f"divergence-null-{n_per_level}", shared_cases=False, between_case_sd=0.0
        )
        assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates), (
            f"{n_per_level} a level: false separation rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    @pytest.mark.parametrize("n_per_level", [3, 5, 10])
    def test_levels_sharing_spread_out_cases_hold_alpha_exactly(self, n_per_level: int) -> None:
        """Paired over shared cases, the case spread cancels: the rate is α, not the ~0.4% the 2-SE rule gave.

        The paired t-test is exact for normal differences, so the rate is within Monte-Carlo error of α on
        both sides. 6,000 replicates: SE at α is 0.0028, so the band is 0.039 to 0.061.
        """
        rate = _false_separation_rate(
            n_per_level, 6000, f"divergence-paired-null-{n_per_level}", shared_cases=True, between_case_sd=5.0
        )
        assert within(rate, SIGNIFICANCE_ALPHA, 6000), f"{n_per_level} shared cases: separation rate {rate:.4f}"


def _lens_fires_rate(n_per_level: int, part_shift: float, replicates: int, seed: str, *, shared_cases: bool) -> float:
    """How often the lens's test separates when the part carries ALL of the whole's movement.

    Each observation's whole is its part plus a remainder; the lever shifts the part by ``part_shift`` SDs and
    leaves the remainder alone, so the whole moves by exactly what the part moves and there is no divergence
    to find. The lens tests each case's whole minus its part between the levels (``level_difference``, pinned
    to the bundle above) and publishes only where that separates — after a Holm correction that can only
    make it rarer.
    """
    rng = random.Random(seed)
    fires = 0
    for _ in range(replicates):
        remainders = []
        for level in (0, 1):
            prefix = "c" if shared_cases else f"l{level}-"
            parts = {f"{prefix}{i}": rng.gauss(part_shift * level, 1.0) for i in range(n_per_level)}
            wholes = {case: part + rng.gauss(0.0, 1.0) for case, part in parts.items()}
            remainders.append({case: wholes[case] - parts[case] for case in parts})
        fires += level_difference(remainders[0], remainders[1]).separated is True
    return fires / replicates


class TestNoDivergenceIsPublishedBeyondAlpha:
    """With the part carrying all of the whole's movement, the lens publishes a divergence at most α of the time.

    The old lens compared two verdicts, each made alone — "the whole moved" beside "the part did not" — which
    is the difference between a significant and a non-significant result, not a test of the difference
    (Gelman & Stern 2006). It measured 0.11 (5 a level, no movement), 0.33 (5 a level, part moving 1.5 SD)
    and 0.33 (10 a level, 1 SD) here. The lens now tests the remainder's own movement.
    """

    #: 3,000 replicates: SE at α is 0.0040, so the bound is 0.066.
    REPLICATES = 3000

    @pytest.mark.parametrize("shared_cases", [False, True])
    @pytest.mark.parametrize(("n_per_level", "part_shift"), [(5, 0.0), (5, 1.5), (10, 1.0)])
    def test_no_divergence_is_published_beyond_alpha(
        self, n_per_level: int, part_shift: float, shared_cases: bool
    ) -> None:
        rate = _lens_fires_rate(
            n_per_level,
            part_shift,
            self.REPLICATES,
            f"lens-{n_per_level}-{part_shift}-{shared_cases}",
            shared_cases=shared_cases,
        )
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_per_level} a level, part shift {part_shift}: divergence published {rate:.4f} against α"
        )

    def test_a_whole_moving_without_its_part_is_still_found(self) -> None:
        """The lens's purpose survives: a sweep whose whole moves 400 ms past an unmoved part is published."""
        for seed in ("a", "b", "c"):
            divergences, _ = _sweep(random.Random(f"divergence-power-{seed}"), n_cases=6, whole_shift=400.0)
            assert any((d.end_to_end.name, d.subsystem.name) == ("total_ms", "tool_ms") for d in divergences), seed

    def test_a_sweep_with_nothing_moving_publishes_nothing_here(self) -> None:
        divergences, _ = _sweep(random.Random("divergence-quiet"), n_cases=12, whole_shift=0.0)
        assert not [d for d in divergences if (d.end_to_end.name, d.subsystem.name) == ("total_ms", "tool_ms")]


class TestTheCarrierIsNamedOnlyWhenShown:
    """``carried_by`` names one component as carrying a whole-run swing; on the largest delta alone it named
    one of two components that moved alike every time the whole moved."""

    REPLICATES = 400

    @staticmethod
    def _named_share(seed: str, llm_shift: float, tool_shift: float) -> tuple[float, int, dict[str, int]]:
        from fractions import Fraction

        from threetears.evals.analysis import component_carrier, measure_movement
        from threetears.evals.kernel.metrics import describe_measure

        registry = toyhost_profile().measures
        rng = random.Random(seed)
        named: dict[str, int] = {}
        moved = 0
        for _ in range(TestTheCarrierIsNamedOnlyWhenShown.REPLICATES):
            levels: list[dict[str, dict[str, Fraction]]] = []
            base = [(rng.gauss(1500.0, 300.0), rng.gauss(500.0, 150.0)) for _ in range(8)]
            for shifted in (False, True):
                llm = {
                    f"c{i}": Fraction(round(b + (llm_shift if shifted else 0.0) + rng.gauss(0.0, 80.0), 3))
                    for i, (b, _) in enumerate(base)
                }
                tool = {
                    f"c{i}": Fraction(round(t + (tool_shift if shifted else 0.0) + rng.gauss(0.0, 80.0), 3))
                    for i, (_, t) in enumerate(base)
                }
                levels.append({"llm_ms": llm, "tool_ms": tool, "total_ms": {c: llm[c] + tool[c] for c in llm}})
            at_a, at_b = levels
            whole = measure_movement(describe_measure("total_ms", registry), at_a["total_ms"], at_b["total_ms"])
            parts = [measure_movement(describe_measure(n, registry), at_a[n], at_b[n]) for n in ("llm_ms", "tool_ms")]
            if whole.direction in ("improved", "regressed"):
                moved += 1
                carrier = component_carrier(whole, parts, at_a, at_b)
                if carrier is not None:
                    named[carrier.name] = named.get(carrier.name, 0) + 1
        return sum(named.values()) / TestTheCarrierIsNamedOnlyWhenShown.REPLICATES, moved, named

    def test_two_components_moved_alike_name_no_carrier_beyond_alpha(self) -> None:
        """Both parts slower by 300 ms, 8 paired cases: the whole moves in every replicate. Measured 0.048 (1.00 on the largest delta alone).

        400 replicates: SE at α is 0.0109, so the bound is 0.094."""
        rate, moved, _ = self._named_share("carrier-alike", 300.0, 300.0)
        assert moved == self.REPLICATES
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), f"named a carrier in {rate:.3f}"

    def test_the_part_that_moved_is_named(self) -> None:
        """Model time slower by 600 ms, tool time unmoved, 8 paired cases. Measured 1.0; held to 0.90 less 4 SE."""
        rate, _, named = self._named_share("carrier-power", 600.0, 0.0)
        assert rate >= 0.90 - 4 * (0.9 * 0.1 / self.REPLICATES) ** 0.5, f"named the moving part in only {rate:.3f}"
        assert set(named) == {"llm_ms"}
