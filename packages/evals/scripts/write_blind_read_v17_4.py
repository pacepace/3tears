"""Write ``tests/fixtures/blind_read_preferences/v17_4.json``: blind reads closed and stored by a durable 17.4 build.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

**It must run at ``be8de80cb``** (``evals/main`` before #1236, the last commit whose build writes durable 17.4 and
stores each preference's p and verdict as its close computed them). Run at any later commit it writes what that build
stores, which is not a 17.4 document, so it refuses unless ``DURABLE_SCHEMA_MINOR`` is 4 and ``BlindPreference`` has no
``shares`` field. From a checkout of this repo::

    git worktree add --detach /tmp/wt-174 be8de80cb
    cp packages/evals/scripts/write_blind_read_v17_4.py /tmp/wt-174/packages/evals/scripts/
    PYTHONPATH=/tmp/wt-174:$(ls -d /tmp/wt-174/packages/*/src /tmp/wt-174/packages/agent/*/src | paste -sd:) \\
        .venv/bin/python /tmp/wt-174/packages/evals/scripts/write_blind_read_v17_4.py \\
        packages/evals/tests/fixtures/blind_read_preferences/v17_4.json
    git worktree remove --force /tmp/wt-174

``PYTHONPATH`` puts the old checkout's sources ahead of the current tree's editable installs; the script checks that the
``threetears.evals`` it imported is the old checkout's and refuses otherwise.

Every path a 17.4 close stores a preference by is covered: a read every rater finished (the exact sign test on one
answer per item, the bounded test on several raters' shares), a read closed early (the anytime test), skipped items on
each test, a model reviewer's preference under the model that served it (two served models read apart), and each
outcome: improved, regressed, not separated, and too few items to test. Nothing calls a model: every answer is written
here, the model's as a reviewer's stored answers are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT))

from packages.evals.tests.factories import make_eval_result  # noqa: E402
from threetears import evals  # noqa: E402
from threetears.evals.analysis.blind_read import close_blind_read, draw_blind_read  # noqa: E402
from threetears.evals.schema import EvalResult, JudgeEvidence, RubricDim, RubricScore, label_key_of  # noqa: E402
from threetears.evals.schema.blind_read import (  # noqa: E402
    BlindAnswer,
    BlindArm,
    BlindPreference,
    BlindRead,
    BlindVerdict,
)
from threetears.evals.schema.versioning import DURABLE_SCHEMA_MINOR, DURABLE_SCHEMA_VERSION  # noqa: E402

GROUNDED = RubricDim(name="answer.grounded", description="Every fact is in the policy.", scale="pass_fail")
JUDGE_MODEL = "acme/judge-1"
REVIEWER = "other/reviewer-9"
CONTROL_RUN, CANDIDATE_RUN = "run-ctrl", "run-cand"
CREATED_AT, ANSWERED_AT, CLOSED_AT = (
    "2026-11-16T09:00:00+00:00",
    "2026-11-16T10:00:00+00:00",
    "2026-11-16T11:00:00+00:00",
)


def _campaign(cases: int) -> tuple[list[EvalResult], dict[str, JudgeEvidence]]:
    """``cases`` cases, each arm's answer its own text, every answer passed by the judge."""
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(cases):
        for run, text in ((CONTROL_RUN, f"old reply to {case}"), (CANDIDATE_RUN, f"new reply to {case}")):
            read = JudgeEvidence(case_material=f"Customer question number {case}.", artifact=text)
            key = label_key_of(read, GROUNDED)
            result = make_eval_result(
                id=f"res-{run}-{case}",
                eval_run_id=run,
                test_case_id=f"case-{case:03d}",
                k_iteration=1,
                rubric_scores=[
                    RubricScore(
                        dim=GROUNDED.name,
                        score=1,
                        scale="pass_fail",
                        axis="capability",
                        served_model=JUDGE_MODEL,
                        judge_temperature=0.0,
                        output_fingerprint=key.output_fingerprint,
                        criterion_fingerprint=key.criterion_fingerprint,
                    )
                ],
            )
            results.append(result)
            evidence[result.id] = read
    return results, evidence


def _answer(read: BlindRead, position: int, rater: str, prefer: str | None, **kwargs: Any) -> BlindAnswer:
    """``rater``'s answer to the item at ``position``: every output passed, and ``prefer`` the arm preferred (``tie``
    for a tie, ``skip`` for a skipped item, None for a preference left blank)."""
    item = read.items[position]
    if prefer == "skip":
        return BlindAnswer(item=item.item, rater=rater, answered_at=ANSWERED_AT, skipped=True, **kwargs)
    slot = (
        None if prefer is None else "tie" if prefer == "tie" else next(o.slot for o in item.outputs if o.arm == prefer)
    )
    return BlindAnswer(
        item=item.item,
        rater=rater,
        answered_at=ANSWERED_AT,
        verdicts=[BlindVerdict(slot=output.slot, dim=GROUNDED.name, score=1) for output in item.outputs],
        prefer=slot,
        **kwargs,
    )


def _read(
    name: str,
    cases: int,
    answers: list[tuple[str, list[str | None], dict[str, Any]]],
    *,
    closed_early: bool = False,
) -> dict[str, Any]:
    """One read drawn over ``cases`` cases, answered as ``answers`` says (each rater's preference per item from the
    top, with any fields its answers carry), closed, and written as a 17.4 build stores it."""
    results, evidence = _campaign(cases)
    arms = [
        BlindArm(key="v1", role="control", run_ids=[CONTROL_RUN]),
        BlindArm(key="v2", role="candidate", run_ids=[CANDIDATE_RUN]),
    ]
    read = draw_blind_read(
        results, evidence, [GROUNDED], arms, scope_id="scope", created_by="agent:test", name=name, created_at=CREATED_AT
    )
    given = [
        _answer(read, position, rater, prefer, **fields)
        for rater, prefers, fields in answers
        for position, prefer in enumerate(prefers)
        if prefer != "unanswered"
    ]
    closed = close_blind_read(
        read.model_copy(update={"answers": given}), results, closed_by="agent:test", closed_at=CLOSED_AT
    ).read
    assert closed.closed_early is closed_early, name
    assert closed.report is not None and closed.report.preferences, name
    return closed.to_dict()


def _served(model: str) -> dict[str, Any]:
    return {"made_by": "model", "reviewer_served_model": model}


def documents() -> list[dict[str, Any]]:
    """Every read, in a fixed order."""
    v2, v1, tie, skip, gap = "v2", "v1", "tie", "skip", "unanswered"
    return [
        # The exact sign test on a finished read: separated toward the candidate, with a skip and a tie.
        _read("sign, improved, one skipped", 16, [("owner", [v2] * 12 + [v1, tie, skip, v2], {})]),
        # Separated toward the control.
        _read("sign, regressed", 12, [("owner", [v1] * 11 + [v2], {})]),
        # Not told apart.
        _read("sign, not separated", 12, [("owner", [v2] * 7 + [v1] * 5, {})]),
        # Too few non-tie items to test: four, beside two ties and two skips.
        _read("sign, too few", 8, [("owner", [v2] * 4 + [tie, tie, skip, skip], {})]),
        # The bounded test on two raters' shares, one of whom skipped an item.
        _read(
            "shares, two raters, one skipped",
            16,
            [("ann", [v2] * 16, {}), ("bo", [v2] * 10 + [skip] + [v2] * 3 + [v1] * 2, {})],
        ),
        # Not told apart on shares.
        _read(
            "shares, two raters, not separated", 12, [("ann", [v2] * 7 + [v1] * 5, {}), ("bo", [v2] * 6 + [v1] * 6, {})]
        ),
        # Closed early: the anytime test, separated, with a skip.
        _read(
            "anytime, closed early, improved",
            30,
            [("ann", [v2] * 20 + [skip] + [v2] * 3 + [gap] * 6, {})],
            closed_early=True,
        ),
        # Closed early and not separated, with a skip.
        _read(
            "anytime, closed early, not separated",
            16,
            [("ann", [v2] * 7 + [v1, skip, v2] + [gap] * 6, {})],
            closed_early=True,
        ),
        # Closed early with too few items to test.
        _read("anytime, closed early, too few", 16, [("ann", [v2, v2, skip, v1] + [gap] * 12, {})], closed_early=True),
        # A model reviewer, every answer served by one model, and a person beside it on the same read.
        _read(
            "a reviewer under its served model, and a person",
            12,
            [(REVIEWER, [v2] * 10 + [skip, v1], _served("other/a-1")), ("owner", [v1] * 10 + [v2, tie], {})],
        ),
        # A model reviewer whose answers two models served: one preference per served model, read early.
        _read(
            "a reviewer under two served models",
            24,
            [
                (REVIEWER, [v2] * 11 + [gap] * 13, _served("other/a-1")),
                (REVIEWER, [gap] * 12 + [v1] * 12, _served("other/b-2")),
            ],
            closed_early=True,
        ),
    ]


def main(out: Path) -> None:
    """Write every document to ``out`` (sorted keys, one-space indent, one trailing newline).

    :param out: The fixture file to write.
    :ptype out: Path
    """
    source = Path(evals.__file__).resolve()
    if not source.is_relative_to(CHECKOUT):
        raise SystemExit(f"threetears.evals was imported from {source}, not from {CHECKOUT}: set PYTHONPATH")
    if (DURABLE_SCHEMA_VERSION, DURABLE_SCHEMA_MINOR) != (17, 4) or "shares" in BlindPreference.model_fields:
        raise SystemExit(
            f"this build writes durable {DURABLE_SCHEMA_VERSION}.{DURABLE_SCHEMA_MINOR}; run it at be8de80cb (17.4)"
        )
    out.write_text(json.dumps(documents(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_blind_read_v17_4.py <out.json>")
    main(Path(sys.argv[1]).resolve())
