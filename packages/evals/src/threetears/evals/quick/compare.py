"""``compare``: run two or more candidates over one case list, with one as the control, and report which separated.

The second rung of adopting the engine. :func:`~threetears.evals.quick.run_eval` measures one candidate;
a newcomer's next question is whether a changed prompt or a different model does better, and that is a
campaign with a control. Everything here is the engine's own path, composed:

- **Each arm is one** :func:`~threetears.evals.quick.run_eval` **run**, labelled by its arm name, all into
  one host and one scope. The arm name is the run's candidate model, so it is what the variant key is
  built from: two arms with different names are two variants, over one content-addressed case set.
- **The campaign declares its design** — one axis, the candidate-model lever, at a level per arm; a
  controlled stimulus, since every arm saw the same cases; a commissioned apparatus, since the runs were
  launched for it; and the repeats each arm ran — through
  :func:`~threetears.evals.analysis.create_campaign`, which gates the declaration against the host's
  vocabulary as it gates any other.
- **The control is designated from the control arm's run** through
  :func:`~threetears.evals.analysis.set_campaign_control`, which reads the variant key that run's
  observations carry and writes it to the declaration's ``control``. No key is computed here.
- **The report is the campaign's report** (:func:`~threetears.evals.analysis.campaign_report`): with no
  analysis generated, the code-only report, whose contrasts table tests every other arm against the
  control on every reading, Holm-corrected, and says which separated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis import (
    Report,
    campaign_report,
    create_campaign,
    report_markdown,
    set_campaign_control,
)
from threetears.evals.contracts import DEFAULT_LAUNCH_K_RUNS, CassetteMode
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.ops.summary import EvalSummary
from threetears.evals.quick.one_call import Candidate, ExpectedLabel, Scorer, callable_host, run_eval
from threetears.evals.quick.tools import Tool, ToolUsingCandidate

#: Who a :func:`compare` campaign and its control are recorded as created by, unless the caller says.
COMPARE_CREATED_BY = "compare"


@dataclass(frozen=True)
class Comparison:
    """What :func:`compare` ran and what its campaign's report says.

    Attributes:
        campaign_id: The campaign holding every arm's run.
        scope_id: The scope the campaign and its runs are stored in.
        name: The campaign's name, which titles its report.
        control: The arm every other arm is tested against.
        arms: Each arm's run summary, by arm name, in the order the arms were given.
        report: The campaign's report, as :func:`~threetears.evals.analysis.campaign_report` read it.
        host: The host the runs and the campaign are stored in, for reading them further.
    """

    campaign_id: str
    scope_id: str
    name: str
    control: str
    arms: dict[str, EvalSummary]
    report: Report
    host: EvalHost

    def render(self) -> str:
        """The report as Markdown: the arms, the contrasts against the control and their verdicts, the charts."""
        return report_markdown(self.report)


def _refuse_unusable_arms(candidates: Mapping[str, Candidate | ToolUsingCandidate], control: str) -> None:
    if isinstance(candidates, str) or not isinstance(candidates, Mapping):
        raise ValueError("compare needs its candidates as a mapping of arm name to candidate")
    if len(candidates) < 2:
        raise ValueError(f"compare needs at least two candidates to compare, and was given {len(candidates)}")
    if blank := [repr(arm) for arm in candidates if not isinstance(arm, str) or not arm.strip()]:
        raise ValueError(f"an arm's name labels its run and its variant, and {', '.join(blank)} is blank")
    if control not in candidates:
        raise ValueError(f"control {control!r} names no arm; the arms are {', '.join(map(repr, candidates))}")


async def compare(
    cases: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Candidate | ToolUsingCandidate],
    scorers: Sequence[Scorer] = (),
    *,
    control: str,
    scope_id: str,
    expected: ExpectedLabel | None = None,
    host: EvalHost | None = None,
    k: int = DEFAULT_LAUNCH_K_RUNS,
    name: str | None = None,
    created_by: str = COMPARE_CREATED_BY,
    tools: Mapping[str, Tool] | None = None,
    cassette_mode: CassetteMode = "off",
    cassette_corpus_id: str | None = None,
) -> Comparison:
    """Run each candidate over every case ``k`` times as one arm, test every arm against ``control``, and report.

    Args:
        cases: The cases, each a JSON object, the same for every arm.
        candidates: The arms, by name. Each name labels its run and is the level its arm is declared at,
            so it is what the report calls the arm (``model=<name>``).
        scorers: The grades, as :func:`~threetears.evals.quick.run_eval` takes them.
        control: The arm every other arm is tested against.
        scope_id: The scope every run and the campaign are stored in.
        expected: Declares every candidate a classifier, as :func:`~threetears.evals.quick.run_eval` takes it.
        host: Where to run and store: ``None`` builds one :func:`~threetears.evals.quick.callable_host` over
            the scorers for every arm. A host of the caller's own is held to what ``run_eval`` holds it to.
        k: Repeats per case, per arm.
        name: The campaign's name, which titles its report; ``None`` names it by its arms, control first.
        created_by: Who the campaign and its control are recorded as created by.
        tools: The tools every arm's candidate calls, as :func:`~threetears.evals.quick.run_eval` takes them.
        cassette_mode: Every arm's cassette mode, as :func:`~threetears.evals.quick.run_eval` takes it. Replay
            is what makes the arms comparable when the tools' answers vary: every arm is served the one
            capture ``cassette_corpus_id`` names, so no difference between them is a difference in what
            their tools said.
        cassette_corpus_id: The capture every arm replays, made over the same cases in ``host`` and ``scope_id``.

    Returns:
        The comparison: every arm's summary, the campaign's id and its report.

    Raises:
        ValueError: Fewer than two candidates, a blank arm name, a ``control`` that names no arm, or
            anything :func:`~threetears.evals.quick.run_eval` refuses.
        ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
    """
    _refuse_unusable_arms(candidates, control)
    if host is None:
        host = callable_host(scorers)
    arms: dict[str, EvalSummary] = {}
    for arm, candidate in candidates.items():
        arms[arm] = await run_eval(
            cases,
            candidate,
            scorers,
            scope_id=scope_id,
            expected=expected,
            host=host,
            k=k,
            model=arm,
            tools=tools,
            cassette_mode=cassette_mode,
            cassette_corpus_id=cassette_corpus_id,
        )
    ordered = [control, *(arm for arm in candidates if arm != control)]
    title = name or " vs ".join(ordered)
    campaign = create_campaign(
        host.storage,
        {
            "name": title,
            "subject_id": title,
            "behavior": "classify" if expected is not None else "score",
            "run_ids": [summary.run_id for summary in arms.values()],
            "declared_design": {
                "axes": [
                    {
                        "axis_id": CANDIDATE_MODEL_LEVER,
                        "values": [{"content": arm, "display": arm} for arm in ordered],
                        "rationale": f"does any arm do better than {control}",
                    }
                ],
                "controls": {"stimulus": "controlled", "apparatus": "commissioned"},
                "intended_repetitions": k,
            },
        },
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
    )
    set_campaign_control(
        host.storage, campaign.id, scope_id, arms[control].run_id, set_by=created_by, profile=host.profile
    )
    return Comparison(
        campaign_id=campaign.id,
        scope_id=scope_id,
        name=title,
        control=control,
        arms=arms,
        report=campaign_report(host, campaign.id, scope_id),
        host=host,
    )


__all__ = ["COMPARE_CREATED_BY", "Comparison", "compare"]
