"""A model as a blind read's reader: its one call per item, planned, admitted and made, and its reply read.

The one reading both doors make. The quick path's ``Judge(reviewer=)`` hands a client and its eval's cost cap
(:mod:`threetears.evals.quick.blind`); a host's ``blind_read_review`` action admits the calls against its out-of-run
cap and makes them in a job (:mod:`threetears.evals.ops.blind_reads`). Each item is one call in a fresh context:
the sheet's criteria, the case material and the outputs in their drawn slots, with a strict reply shape, and nothing
of any other item. The model is never shown an arm, a run, a result or a judge score, since the sheet holds none.

What the reading produces is answers and a :class:`~threetears.evals.schema.blind_read.BlindReview`; it writes
nothing to a read. Which door may keep a model's answers on a read is the door's rule.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Mapping
from typing import Any, Final

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.kernel.text_block import under
from threetears.evals.schema.blind_read import (
    BlindAnswer,
    BlindPreferenceAnswer,
    BlindReview,
    BlindSheet,
    BlindVerdict,
)
from threetears.evals.schema.completion import CompletionProtocolDefect, PricedCompletion
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.models import SCALES, RubricScale

log = get_logger(__name__)

#: What a label is read as, on a pass/fail dimension.
_PASS_FAIL: Final = {"pass": 1, "fail": 0, "1": 1, "0": 0, "true": 1, "false": 0}


def answer_score(label: str, scale: RubricScale) -> int | None:
    """The score a written answer gives on ``scale``: pass or fail (or 1, 0, true, false) on pass/fail, 1 to 5 on the
    ordinal scale; None for anything else. A labels file and a model's reply are read by it alike.

    :param label: The answer, trimmed and lower-cased by the caller.
    :param scale: The criterion's scale.
    :return: The score, or None where the answer is not one the scale holds.
    """
    if scale == "pass_fail":
        return _PASS_FAIL.get(label)
    return int(label) if label.isdigit() and 1 <= int(label) <= 5 else None


_REVIEW_SYSTEM: Final = (
    "You are reviewing outputs against stated criteria. You are shown what each output answers, then one output or "
    "two outputs labelled A and B in a random order. Judge each output on each criterion from what is shown and "
    "nothing else. Reply with one JSON object and no other text: "
    '{"verdicts": {"<slot>": {"<criterion>": <answer>}}, "prefer": "A" | "B" | "tie" | null, "reason": "<one '
    'sentence>"}. Give every slot and criterion shown. Set "prefer" only where two outputs are shown and you are '
    "asked which is better; otherwise null."
)


def _review_prompt(sheet: BlindSheet, index: int) -> str:
    item = sheet.items[index]
    asked = [criterion for criterion in sheet.criteria if criterion.dim in item.ask.verdicts]
    lines = ["# Criteria"]
    for criterion in asked:
        answers = " or ".join(
            f'"{level.value}"' + (f" ({level.meaning})" if level.meaning else "") for level in criterion.levels
        )
        lines += under(f"{criterion.dim}: {criterion.question} Answer {answers}.", "  ", first="- ")
    lines += ["", "# What the output answers", item.case_material]
    for output in item.outputs:
        lines += ["", f"# Output {output.slot}", output.text]
    if item.ask.preference:
        lines += ["", "Then say which output is better: A, B, or tie."]
    return "\n".join(lines)


def _reply_format(sheet: BlindSheet, index: int) -> dict[str, Any]:
    """The strict reply shape one item asks for: a verdict per criterion per output, the preference, and a reason.
    Every answer is a string enum, the one form every provider's strict mode enforces."""
    item = sheet.items[index]
    levels = {criterion.dim: [level.value for level in criterion.levels] for criterion in sheet.criteria}
    per_output = {
        "type": "object",
        "properties": {dim: {"type": "string", "enum": levels[dim]} for dim in item.ask.verdicts},
        "required": list(item.ask.verdicts),
        "additionalProperties": False,
    }
    slots = [output.slot for output in item.outputs]
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "object",
                "properties": dict.fromkeys(slots, per_output),
                "required": slots,
                "additionalProperties": False,
            },
            "prefer": {"type": "string", "enum": ["A", "B", "tie"] if item.ask.preference else ["none"]},
            "reason": {"type": "string"},
        },
        "required": ["verdicts", "prefer", "reason"],
        "additionalProperties": False,
    }
    return strict_response_format("blind_review", schema)


class _ReplyUnusable(ValueError):
    """A reviewer's reply that is not the answer its item asked for."""


def _answer_of(sheet: BlindSheet, index: int, content: str) -> tuple[list[BlindVerdict], BlindPreferenceAnswer | None]:
    """The verdicts and preference one reply gives for the item at ``index``.

    :raises _ReplyUnusable: The reply is not the JSON object asked for, misses a slot or criterion, or gives an
        answer off its scale.
    """
    item = sheet.items[index]
    scales = {criterion.dim: criterion.scale for criterion in sheet.criteria}
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as unparsed:
        raise _ReplyUnusable(f"its reply is not JSON ({unparsed.msg})") from unparsed
    given = reply.get("verdicts") if isinstance(reply, dict) else None
    if not isinstance(given, dict):
        raise _ReplyUnusable('its reply carries no "verdicts" object')
    reason = str(reply.get("reason") or "")
    verdicts: list[BlindVerdict] = []
    for output in item.outputs:
        of_slot = given.get(output.slot)
        for dim in item.ask.verdicts:
            raw = of_slot.get(dim) if isinstance(of_slot, dict) else None
            score = None if raw is None else answer_score(str(raw).strip().lower(), scales[dim])
            if score is None:
                levels = " or ".join(SCALES[scales[dim]].levels)
                raise _ReplyUnusable(f"it gave {raw!r} for output {output.slot} on {dim}, whose answer is {levels}")
            verdicts.append(BlindVerdict(slot=output.slot, dim=dim, score=score, reason=reason))
    if not item.ask.preference:
        return verdicts, None
    choices: dict[str, BlindPreferenceAnswer] = {"A": "A", "B": "B", "tie": "tie"}
    prefer = choices.get(str(reply.get("prefer")))
    if prefer is None:
        raise _ReplyUnusable(f"it preferred {reply.get('prefer')!r}, and a preference is A, B or tie")
    return verdicts, prefer


#: The ledger purpose a model reviewer's calls are written under: measurement spend, on its own line.
REVIEWER_PURPOSE: Final = "reviewer"


def plan_review(sheet: BlindSheet, items: Collection[str] | None = None) -> dict[str, PlannedCall]:
    """The calls a model's reading of ``sheet`` makes: one per item, by item, in the drawn order.

    The planner half: its caller admits these under its budget before any is made, and :func:`answer_sheet` makes
    only what was admitted.

    :param sheet: The blinded view of a read.
    :param items: The items to plan; None for every item.
    :return: Each item's call: the prompt pair and the strict reply shape it asks for.
    """
    return {
        item.item: PlannedCall(
            system=_REVIEW_SYSTEM, user=_review_prompt(sheet, index), response_format=_reply_format(sheet, index)
        )
        for index, item in enumerate(sheet.items)
        if items is None or item.item in items
    }


async def answer_sheet(
    sheet: BlindSheet,
    client: PricedCompletion,
    budget: OutOfRunBudget,
    admitted: Mapping[str, AdmittedCall],
    *,
    reused: int = 0,
    stop: Callable[[float, int], str | None] | None = None,
) -> tuple[list[BlindAnswer], BlindReview]:
    """Have a model answer ``sheet``, one call per item, each with nothing of any other item in it.

    The model is shown exactly what a person is: the sheet's criteria, the case material and the outputs in their
    drawn slots. Each answer is a model's, named by the client's model, and carries the model that served it. Both
    doors read a sheet this way: the quick path's ``Judge(reviewer=)`` and a host's ``blind_read_review`` job.

    **Spend.** Every call was admitted through ``budget`` by this function's caller (:func:`plan_review` plans them)
    and is written to the out-of-run ledger as it is made, under the purpose ``reviewer``: what it cost to check the
    judge, never a candidate's cost. A caller that holds the calls to more than their admission (the quick path's cost
    cap over everything its eval spends) passes ``stop``: asked before each call with what the reading's priced calls
    have cost so far and how many went unpriced, it returns why the reading stops there, or None to go on.

    **A reply that cannot be used** (not the JSON asked for, a verdict missing or off its scale) leaves its item not
    answered, as a skip, with the reason on the reading; the rest are asked. A completion that breaks the protocol
    stops the reading, said the same way. Either way what was answered is returned.

    :param sheet: The blinded view of a read.
    :param client: The reviewer's client, which names its model and prices its calls where it can.
    :param budget: The ledger the calls are admitted against and written to.
    :param admitted: Each item's admitted call, by item: an item with none is not asked. An item answered before is
        left out by the caller.
    :param reused: How many items the caller read from answers recorded earlier, for the reading to state.
    :param stop: Why the reading stops before the next call, from the priced spend so far and the unpriced calls;
        None asks every admitted call.
    :return: An answer per item asked, in order (a skip where the reply could not be used), and the reading.
    """
    model = client.model_name
    answers: list[BlindAnswer] = []
    unanswered: list[str] = []
    served: set[str] = set()
    priced: list[float] = []
    calls = unpriced = 0
    stopped: str | None = None
    for index, item in enumerate(sheet.items):
        call = admitted.get(item.item)
        if call is None:
            continue
        if stop is not None and (stopped := stop(math.fsum(priced), unpriced)) is not None:
            break
        try:
            recorded = await budget.generate(client, call)
        except CompletionProtocolDefect as broken:
            # The call was made and its row written; the reading says what stopped it, and what was answered stands.
            contain(
                broken, log=log, what=f"a model reviewer's reading of a blind read stopped: {broken}", level="error"
            )
            stopped = f"the reviewer's completion broke the protocol ({broken})"
            calls += 1
            unpriced += 1
            break
        calls += 1
        if recorded.spend.cost_usd is None:
            unpriced += 1
        else:
            priced.append(recorded.spend.cost_usd)
        which = getattr(recorded.result, "served_model", None)
        which = which if isinstance(which, str) and which.strip() else None
        if which is not None:
            served.add(which)
        try:
            verdicts, prefer = _answer_of(sheet, index, str(getattr(recorded.result, "content", "")))
        except _ReplyUnusable as unusable:
            unanswered.append(f"case {index + 1}: {unusable}")
            answers.append(BlindAnswer(item=item.item, rater=model, made_by="model", skipped=True))
            continue
        answers.append(
            BlindAnswer(
                item=item.item,
                rater=model,
                made_by="model",
                reviewer_served_model=which,
                verdicts=verdicts,
                prefer=prefer,
            )
        )
    review = BlindReview(
        reviewer=model,
        served_models=sorted(served),
        asked=calls,
        reused=reused,
        calls=calls,
        cost_usd=math.fsum(priced) if priced else None,
        unpriced_calls=unpriced,
        unanswered=unanswered,
        stopped=stopped,
    )
    return answers, review


__all__ = ["REVIEWER_PURPOSE", "answer_score", "answer_sheet", "plan_review"]
