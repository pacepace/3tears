"""The one opt-in model: a beta-binomial pass^k at a depth an arm never reached, labelled a model on every surface.

Off unless the host asks for it (``HostProfile.model_pass_hat_k``). When on, each frontier point that scored no case
at the subject's ``k`` carries a model's pass^k there, ``basis: "model"`` with what it assumes; nothing ranks, gates
or bars on it, the page puts it in a table of its own with its assumption, the writer reads it labelled, and a memo
comparing arms on it is dropped, as any comparison at an unreached depth is.

Arm A ran each of four cases twice, passing 2, 1, 0 and 2 of them: shares 1, 1/2, 0, 1, so mean 5/8, sample variance
11/48, and a mean ``1/n`` of 1/2. The per-case chance's variance is ``(11/48 - (5/8)(3/8)(1/2)) / (1 - 1/2) = 43/192``;
the beta's ``α + β = (15/64) / (43/192) - 1 = 2/43``, so ``α = 5/172`` and ``β = 3/172``; pass^3 is
``α(α+1)(α+2) / ((α+β)(α+β+1)(α+β+2))``.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from threetears.evals.analysis import build_campaign_report, build_code_only_report
from threetears.evals.analysis.campaign_bundle import writer_view
from threetears.evals.analysis.depth import MODEL_ASSUMES, model_pass_hat_k
from threetears.evals.analysis.lens_text import frontier_text
from threetears.evals.analysis.report.memo_style import MemoStyle
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock
from threetears.evals.analysis.style import ArmName, lint_prose
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.depth_support import ONCE, PROFILE, TWICE, TwoArms

#: A host that asks for the model.
_MODELLED = replace(PROFILE, model_pass_hat_k=True)

_ALPHA, _TOTAL = Fraction(5, 172), Fraction(2, 43)
#: A's model pass^3, worked from the beta's shapes.
_A_PASS_3 = float(_ALPHA * (_ALPHA + 1) * (_ALPHA + 2) / (_TOTAL * (_TOTAL + 1) * (_TOTAL + 2)))


def _a_attempts() -> dict[str, list[bool]]:
    return {case: list(attempts) for case, attempts in TWICE.items()}


def test_the_model_reads_pass_hat_k_off_a_beta_fitted_by_the_method_of_moments() -> None:
    read = model_pass_hat_k(_a_attempts(), 3)

    assert read is not None
    assert read.basis == "model" and read.assumes == MODEL_ASSUMES
    assert (read.alpha, read.beta) == (pytest.approx(5 / 172), pytest.approx(3 / 172))
    assert read.pass_hat_k == pytest.approx(_A_PASS_3)
    assert read.cases == 4
    # At depth 1 the beta's first moment is the mean share.
    one = model_pass_hat_k(_a_attempts(), 1)
    assert one is not None and one.pass_hat_k == pytest.approx(5 / 8)


def test_cases_that_spread_no_more_than_their_noise_are_read_at_the_mean_chance() -> None:
    # Every case passed one of two: no spread between cases at all, so every case is read at 1/2 and pass^3 is 1/8.
    read = model_pass_hat_k({f"tc{index}": [True, False] for index in range(4)}, 3)

    assert read is not None and (read.alpha, read.beta) == (None, None)
    assert read.pass_hat_k == pytest.approx(1 / 8)


def test_the_model_reads_nothing_it_cannot_fit() -> None:
    # One case, or every case run once: the spread between cases cannot be told from each case's noise.
    assert model_pass_hat_k({"tc1": [True, True]}, 3) is None
    assert model_pass_hat_k({case: list(attempts) for case, attempts in ONCE.items()}, 2) is None
    # Every attempt passed: pass^k is 1 at any depth, and no shape is fitted.
    every = model_pass_hat_k({"tc1": [True, True], "tc2": [True, True]}, 5)
    assert every is not None and every.pass_hat_k == 1.0 and every.alpha is None
    with pytest.raises(ValueError):
        model_pass_hat_k(_a_attempts(), 0)


def test_the_model_is_off_unless_the_host_asks_for_it() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=3, k_b=3)

    assert all(point.model_pass_hat_k is None for point in arms.bundle.frontier.subjects[0].points)
    assert not PROFILE.model_pass_hat_k


def test_a_host_that_asks_reads_a_model_at_each_unreached_depth_and_nothing_ranks_on_it() -> None:
    plain = TwoArms(TWICE, ONCE, k_a=3, k_b=3)
    modelled = TwoArms(TWICE, ONCE, k_a=3, k_b=3, profile=_MODELLED)
    a, b = modelled.point(modelled.key_a), modelled.point(modelled.key_b)

    # Both planned 3; A reached 2 and B 1, so pass^3 is unmeasured for both.
    assert a.pass_hat_k is None and b.pass_hat_k is None
    assert a.model_pass_hat_k is not None and a.model_pass_hat_k.k == 3
    assert a.model_pass_hat_k.pass_hat_k == pytest.approx(_A_PASS_3)
    # B ran every case once: the model cannot fit it, and says nothing.
    assert b.model_pass_hat_k is None
    # Every other field of every point, and the subject's verdict, reads as it does with the model off.
    for key in (modelled.key_a, modelled.key_b):
        assert modelled.point(key).model_dump(exclude={"model_pass_hat_k"}) == plain.point(key).model_dump(
            exclude={"model_pass_hat_k"}
        )
    on, off = modelled.bundle.frontier.subjects[0], plain.bundle.frontier.subjects[0]
    assert on.model_dump(exclude={"points"}) == off.model_dump(exclude={"points"})


def test_the_page_puts_the_model_in_a_table_of_its_own_labelled_with_what_it_assumes() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=3, k_b=3, profile=_MODELLED)
    read = arms.readings()
    report = build_code_only_report(
        arms.bundle, assembled_at="2026-10-24T00:00:00+00:00", campaign_name="depth", arm_labels=read.arm_labels()
    )

    page = build_campaign_report(arms.campaign, read, {"character": report}, None)

    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "model_pass_k"]
    assert [{key: cell.value for key, cell in row.items()} for row in table.rows] == [
        {"aspect": "character", "arm": "A", "depth": 3, "model": round(100 * _A_PASS_3, 1)}
    ]
    said = f"These pass^k figures are {MODEL_ASSUMES}. No arm ran a case that deep, so nothing ranks on them."
    assert said in [block.text for block in page.blocks if isinstance(block, DisclosureBlock)]
    assert lint_prose(said) == []


def test_the_writer_reads_the_model_labelled_and_the_text_says_it_is_one() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=3, k_b=3, profile=_MODELLED)
    bundle = campaign_bundle(arms.bundle, arms=arms.arms, readings=arms.readings().readings)

    [aspect] = writer_view(bundle)["aspects"]

    assert aspect["depth"]["pass_k"]["model"] == {
        "A": {"k": 3, "pass_hat_k": pytest.approx(_A_PASS_3, rel=1e-3), "basis": "model", "assumes": MODEL_ASSUMES}
    }
    assert MODEL_ASSUMES in frontier_text(arms.bundle.frontier)


def test_a_memo_comparing_arms_on_the_model_is_dropped_as_any_unreached_comparison_is() -> None:
    arms = TwoArms(TWICE, ONCE, k_a=3, k_b=3, profile=_MODELLED)
    style = MemoStyle(bundle=arms.bundle, names={arms.key_a: ArmName(label="A"), arms.key_b: ArmName(label="B")})

    assert style.compares_unreached("On the model, A beats B at pass^3.")
