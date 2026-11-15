"""A run whose results nothing scored joins its arm's test setup, rather than a setup of its own.

The judge's served model, temperature, sent settings and configs are read off a run's SCORES. A judged run that
scored nothing (every report it should have judged went unwritten) has none to read them off, so each reads empty: an
input none of its results exercised. Such an input shaped none of the run's observations, so the run takes the level
its arm's runs read and pools with them, as an unscored result inside a scored run already does. Where its arm's runs
read two levels, there is no one setup to join and it stays apart; where nothing of its arm was scored, it takes the
level every scored run shares, so an arm that wrote nothing still shares a setup with the arm it is read against.

Mutations that turn this file red: an idle run left undecided on the judge (a cell of its own beside its arm's); an
idle run given one of two levels its arm read; the class of a scored run moved by an idle run beside it; an arm with
no scored run left in a setup no other arm shares.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.schema import RubricScore
from threetears.evals.schema.models import EvalResult, EvalRun
from packages.evals.tests.factories import RECORDED_REQUEST_SETTINGS
from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_NARROW,
    TOYHOST_WIDE,
    toyhost_aspect_design,
    toyhost_campaign_of_one_aspect,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_GRADER_VERSION,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The extractor held to every seat, so its runs have a judge: the judged variant of the toy host.
_PROFILE = toyhost_profile(every_seat=True)

#: The judge every scored run names, and the model a provider may answer it with.
_PIN = "vendor/judge-1"


def _run(level: int, tag: str, *, served: str | None) -> tuple[EvalRun, list[EvalResult]]:
    """One judged batch of the arm at ``level``, and its results: each scored, as ``served`` answered, or, with
    ``served`` None, none of them scored."""
    batch = toyhost_batch(
        chunk_tokens=level,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        grader_version=TOYHOST_GRADER_VERSION,
    )
    batch = batch.model_copy(
        update={
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{batch.id}|{tag}")),
            "judge_model": _PIN,
            "judge_temperature": 0.0,
        }
    )
    results = toyhost_measurements(
        batch,
        profile=_PROFILE,
        cost_usd=0.02,
        total_ms=900.0,
        field_accuracy=0.8,
        layout_fidelity=None if served is None else 3,
    )
    if served is not None:
        results = [
            result.model_copy(
                update={
                    "rubric_scores": [
                        score.model_copy(
                            update={
                                "served_model": served,
                                "judge_temperature": 0.0,
                                "judge_sent_settings": RECORDED_REQUEST_SETTINGS,
                            }
                        )
                        for score in result.rubric_scores
                    ]
                }
            )
            for result in results
        ]
    assert all(isinstance(score, RubricScore) for result in results for score in result.rubric_scores)
    return batch, results


def _bundle(runs: Sequence[tuple[EvalRun, list[EvalResult]]]) -> AnalysisContextBundle:
    campaign = toyhost_campaign_of_one_aspect(
        id="7c1e4b20-5a9d-4f38-9e62-0b3d8f1a2c54",
        scope_id=TOYHOST_SCOPE,
        name="unscored runs",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[run.id for run, _ in runs],
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    storage = ToyhostStorage([run for run, _ in runs], {run.id: results for run, results in runs}, ())
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _class_of(bundle: AnalysisContextBundle, run: tuple[EvalRun, list[EvalResult]]) -> str:
    """The one apparatus class every observation of ``run`` sits in."""
    ids = {result.id for result in run[1]}
    (found,) = {cell.apparatus_class_id for cell in bundle.cells for oid in cell.observation_ids if oid in ids}
    return found


def _unknown(bundle: AnalysisContextBundle, class_id: str) -> set[str]:
    return {dim for cell in bundle.cells if cell.apparatus_class_id == class_id for dim in cell.unknown_dimensions}


def test_a_run_its_judge_scored_nothing_of_joins_its_arm_s_setup_and_moves_no_scored_run() -> None:
    scored = _run(TOYHOST_NARROW, "a1", served=_PIN)
    idle = _run(TOYHOST_NARROW, "a2", served=None)
    other = _run(TOYHOST_WIDE, "b1", served=_PIN)
    assert not any(result.rubric_scores for result in idle[1]), "the precondition: nothing of the idle run is scored"

    bundle = _bundle([scored, idle, other])
    without = _bundle([scored, other])

    assert _class_of(bundle, idle) == _class_of(bundle, scored)
    assert "judge_model" not in _unknown(bundle, _class_of(bundle, idle))
    narrow = [cell for cell in bundle.cells if cell.apparatus_class_id == _class_of(bundle, scored)]
    assert len({cell.variant_key for cell in narrow}) == 2, "both arms read under the one setup"
    # The scored runs' setups are the ones they read without the idle run beside them.
    assert (_class_of(bundle, scored), _class_of(bundle, other)) == (
        _class_of(without, scored),
        _class_of(without, other),
    )


def test_a_run_its_judge_scored_nothing_of_stays_apart_when_its_arm_read_two_judges() -> None:
    first = _run(TOYHOST_NARROW, "a1", served=_PIN)
    second = _run(TOYHOST_NARROW, "a3", served="vendor/judge-2")
    idle = _run(TOYHOST_NARROW, "a2", served=None)

    bundle = _bundle([first, second, idle, _run(TOYHOST_WIDE, "b1", served=_PIN)])

    assert len({_class_of(bundle, run) for run in (first, second, idle)}) == 3
    assert "judge_model" in _unknown(bundle, _class_of(bundle, idle))


def test_an_arm_its_judge_scored_nothing_of_shares_the_setup_of_the_arm_it_is_read_against() -> None:
    idle = _run(TOYHOST_NARROW, "a2", served=None)
    other = _run(TOYHOST_WIDE, "b1", served=_PIN)

    bundle = _bundle([idle, other])

    assert _class_of(bundle, idle) == _class_of(bundle, other)
