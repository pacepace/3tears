"""One comparison, continued: a quick-path comparison run again over more cases runs only the cases added.

A newcomer adds cases to a comparison and runs it again, as often as they like, with nothing declared (owner ruling,
2026-11-17). Each paired contrast on a declared range is read by the mixture bet over the cases in the order they were
added, and that bet keeps its error over every reading only while it is ONE process: a case already run keeps the
values it ran with, and a reading after more cases is the same capital read later. Running every case again on each
run would draw a new process each time, and the readings of several processes do not share one bound.

So an :class:`~threetears.evals.quick.contract.Eval` comparison run into a store that holds a comparison it filed
before under the same name, in the same scope, continues it when nothing but the cases' tail has changed: the stored
results of every case already run are read again as they ran, only the cases appended after them run, and the new
campaign holds every arm's runs of both. Whether it is the same comparison is read from stored facts only
(:class:`EarlierComparison`):

- the eval's name and scope, which find the comparison (the newest campaign the quick path filed under that name);
- each arm, by the name, model and levers its run was launched under, and by the variant key and measurement context
  its new run is stamped with (:func:`~threetears.evals.analysis.bundle.observations.variant_key_of_run`,
  ``EvalRun.context_key``), which hold the graders' and the judge's definitions;
- the repeats per case, the graders in their order, the guardrails' margins, the question and its margin;
- every case already run, at its place: the case set's address (``_template_id``) over the cases first run, and each
  stored case's content, label and name at its place, so a case removed, moved or edited is seen.

Anything else starts a NEW comparison, which runs every case and says so in one sentence naming what changed
(:func:`new_comparison_sentence`). A comparison continued says so too (:func:`continued_sentence`), and says how to start
afresh: a new name, or another scope or store. An arm is known by its name, never by its code, so a runner changed
under the same name continues the comparison; the sentence says that as well.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis.bundle.observations import variant_key_of_run
from threetears.evals.analysis.campaigns import list_campaigns
from threetears.evals.kernel.candidate_kind import QUICK_PATH_KINDS
from threetears.evals.kernel.host import EvalHost
from threetears.evals.run.reads import list_results
from threetears.evals.schema import EvalRun

#: Who files the quick path's campaigns, as :data:`~threetears.evals.quick.result.CREATED_BY` records it (read here,
#: not imported, because the result module imports this one).
_FILED_BY = "eval"


@dataclass(frozen=True)
class EarlierArm:
    """One arm of a stored comparison: how its runs were launched, and its runs in the order they ran.

    :ivar launched_as: The arm's name as its runs carry it (``None`` for an arm named by its model), its candidate
        model, and its levers, as JSON.
    :ivar run_ids: Its runs, oldest first: the comparison's first launch, then each continuation's.
    :ivar variant_key: The variant key its results are stamped with.
    :ivar context_key: Its first run's measurement context key.
    :ivar declared: What its first run froze of the graders' declarations (``EvalRun.declared_measures`` and
        ``declared_margins``), as JSON.
    """

    launched_as: tuple[str | None, str, str]
    run_ids: tuple[str, ...]
    variant_key: str | None
    context_key: str | None
    declared: str


@dataclass(frozen=True)
class EarlierComparison:
    """The comparison the quick path filed last under one name in one scope, as its stored campaign and runs hold it.

    :ivar campaign_id: Its campaign.
    :ivar declared_at: When its design and control were fixed (its campaign's ``declared_at``), which a continuation
        keeps: nothing a continuation may change is part of the design.
    :ivar template_id: The case set every run of it ran.
    :ivar first_cases: How many cases its first launch ran: the case set's address is over these.
    :ivar case_ids: Every case it ran, in the order they were first run.
    :ivar k: Repeats per case.
    :ivar control: The control arm's ``launched_as``.
    :ivar arms: Each arm, keyed by ``launched_as``.
    :ivar design: What its campaign declared that a continuation must declare alike: the graders in order, the
        guardrails' margins, latency under test and the question, as JSON.
    :ivar margin: The question's margin, a share of 0 to 1; ``None`` declares none.
    """

    campaign_id: str
    declared_at: str | None
    template_id: str
    first_cases: int
    case_ids: tuple[str, ...]
    k: int
    control: tuple[str | None, str, str]
    arms: Mapping[tuple[str | None, str, str], EarlierArm]
    design: str
    margin: float | None


def launched_as(arm: str | None, model: str, levers: Mapping[str, Any] | None) -> tuple[str | None, str, str]:
    """How an arm is launched, as the identity a continuation compares: its name, its model and its levers."""
    return arm, model, canonical_json(dict(levers or {}))


def canonical_json(value: Any) -> str:
    """``value`` as sorted, compact JSON: the one spelling two stored facts are compared in."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def design_facts(
    measure_order: Sequence[str],
    guardrail_margins: Mapping[str, float],
    measure_latency: bool,
    question: str | None,
) -> str:
    """What a comparison declares that a continuation must declare alike, as JSON: graders in order, guardrails'
    margins, latency under test and the question. Its margin is compared beside it (:func:`same_margin`)."""
    return canonical_json(
        {
            "measure_order": list(dict.fromkeys(measure_order)),
            "guardrail_margins": dict(sorted(guardrail_margins.items())),
            "measure_latency": bool(measure_latency),
            "question": question or "",
        }
    )


def same_margin(stored: float | None, given: float | None) -> bool:
    """Whether a comparison's stored margin (its share of 0 to 1, read back from the points its campaign stores) is the
    one ``given`` now: both none, or one share to within float rounding."""
    if stored is None or given is None:
        return stored is given
    return math.isclose(stored, given, rel_tol=1e-9, abs_tol=1e-12)


def earlier_comparison(
    host: EvalHost, scope_id: str, name: str, launched_as_run: Callable[[EvalRun], tuple[str | None, str, str]]
) -> EarlierComparison | None:
    """The comparison the quick path filed last under ``name`` in ``scope_id``, or ``None`` where there is none to
    continue: none filed, one of a single arm, or one whose runs are not one quick-path case set run alike. Reads the
    store.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param name: The eval's name, which its campaign's subject is.
    :ptype name: str
    :param launched_as_run: How a stored run's arm was launched (:func:`launched_as` of its arm's name, model and
        levers), read by the module that writes a run's arm name
        (:func:`~threetears.evals.quick.one_call.launched_as_run`).
    :ptype launched_as_run: Callable[[EvalRun], tuple[str | None, str, str]]
    :return: The comparison, or ``None``.
    :rtype: EarlierComparison | None
    """
    filed = [
        campaign
        for campaign in list_campaigns(host.storage, scope_id, subject_id=name, archived=False)
        if campaign.created_by == _FILED_BY and len(campaign.aspects) == 1
    ]
    if not filed:
        return None
    campaign = max(filed, key=lambda each: (each.created_at, each.id))
    (aspect,) = campaign.aspects
    runs = [run for run_id in aspect.run_ids if (run := host.storage.load_eval_run(run_id, scope_id)) is not None]
    if len(runs) != len(aspect.run_ids) or not runs:
        return None
    runs.sort(key=lambda run: (run.created_at, run.id))
    if (
        len({run.template_id for run in runs}) != 1
        or runs[0].template_id is None
        or len({run.k_runs for run in runs}) != 1
        or any(run.candidate_kind not in QUICK_PATH_KINDS for run in runs)
    ):
        return None
    grouped: dict[tuple[str | None, str, str], list[EvalRun]] = {}
    for run in runs:
        grouped.setdefault(launched_as_run(run), []).append(run)
    if len(grouped) < 2:
        return None
    by_arm: dict[tuple[str | None, str, str], EarlierArm] = {}
    case_lists: list[tuple[str, ...]] = []
    for key, held in grouped.items():
        variants = {variant_key_of_run(list_results(host.storage, run.id, scope_id)) for run in held}
        if len(variants) != 1:
            return None
        first = held[0]
        by_arm[key] = EarlierArm(
            launched_as=key,
            run_ids=tuple(run.id for run in held),
            variant_key=next(iter(variants)),
            context_key=first.context_key,
            declared=canonical_json(
                {
                    "measures": {
                        name: declared.model_dump(mode="json") for name, declared in first.declared_measures.items()
                    },
                    "margins": first.declared_margins,
                }
            ),
        )
        case_lists.append(tuple(case_id for run in held for case_id in run.test_case_ids))
    # Every arm ran every case, in one order: a comparison whose arms ran other cases is not one to extend.
    if len(set(case_lists)) != 1:
        return None
    control_variant = next((arm.variant_key for arm in campaign.arms if arm.role == "control"), None)
    control = next((key for key, arm in by_arm.items() if arm.variant_key == control_variant), None)
    if control is None:
        return None
    design = aspect.design
    return EarlierComparison(
        campaign_id=campaign.id,
        declared_at=campaign.declared_at,
        template_id=runs[0].template_id,
        first_cases=len(grouped[control][0].test_case_ids),
        case_ids=case_lists[0],
        k=runs[0].k_runs,
        control=control,
        arms=by_arm,
        design=design_facts(
            design.measure_order if design is not None else (),
            {entry.dimension: entry.margin for entry in design.guardrail_margins} if design is not None else {},
            design.measure_latency if design is not None else False,
            campaign.question or None,
        ),
        margin=None if aspect.margin is None else aspect.margin / 100,
    )


#: What a result's footer calls a store kept in memory for one run (``store=":memory:"``): it is gone when the run ends.
MEMORY_WHERE = "memory (this process only)"

#: What the print says under a comparison's first line in place of
#: :data:`~threetears.evals.analysis.bundle.comparisons.REREAD_SENTENCE` when its runs were kept in memory: nothing is
#: there to continue, so running it again is a new comparison. The one owner of the sentence.
UNKEPT_SENTENCE = (
    "Kept in memory, so running it again re-runs every case as a new comparison: to add cases and read it again "
    "with its error kept, keep its runs in a store (store= a file)."
)


def continued_sentence(already: int, added: int) -> str:
    """What a continued comparison's print says of how it ran: the cases read again as they ran, the cases that ran
    now, and how to start a new comparison instead. The one owner of the sentence."""
    ran = f"{added} new case{'s' if added != 1 else ''} ran" if added else "no case was added, so nothing ran"
    return (
        f"Continues the comparison run before under this name: its {already} case{'s' if already != 1 else ''} already "
        f"run are read as they ran, and {ran}. An arm is known by its name, not its code: to compare a changed runner, "
        "or to start afresh, give the eval a new name, or run it in another scope or store."
    )


def new_comparison_sentence(changed: str) -> str:
    """What a comparison's print says when it could not continue the one run before under its name: what changed, and
    that every case ran. The one owner of the sentence.

    :param changed: What changed, as a clause (``"the arms changed"``, ``"case 3 was edited, moved or removed"``).
    :ptype changed: str
    """
    return (
        f"A new comparison: {changed} since the comparison run before under this name, so every case ran and the "
        "earlier readings are not continued."
    )


__all__ = [
    "MEMORY_WHERE",
    "UNKEPT_SENTENCE",
    "EarlierComparison",
    "canonical_json",
    "continued_sentence",
    "design_facts",
    "earlier_comparison",
    "launched_as",
    "new_comparison_sentence",
    "same_margin",
]
