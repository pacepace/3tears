"""How a campaign stored before aspects existed reads as a campaign of one aspect.

A v15 campaign is one template, one bag of runs and one optional design that carries its own questions and its
control. The campaign of aspects asks ONE question of its arms and measures it on one or more aspects, each with its
own template, runs, margin and design, and it keeps a decision rule's clauses as floors. :func:`upgrade_campaign_v15`
is the pure dict step between the two: it reads the stored v15 document and returns the document the next shape
validates, and it is not registered in any upgrade chain yet.

**What the step keeps, and what it cannot.** The template, the runs and the design move into the one aspect, labelled
with the campaign's behavior. The control key, its lever map and its predicate move verbatim onto the campaign's
control arm: relocated, never recomputed, as a step's rules require of every stored key. The live questions' texts
become the campaign's one question and their typed rules become floors. A question's id, its lineage (``supersedes``)
and its retirement do not survive: the next shape asks one question, which has none of them, so a retired or
superseded question is dropped with its rule and a live one's id goes with it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.kernel.declaration import aware_instant

__all__ = ["QUESTION_SEPARATOR", "UPGRADED_CONTROL_LABEL", "upgrade_campaign_v15"]

#: The label a v15 campaign's declared control carries as an arm. A v15 design names its control by key alone.
UPGRADED_CONTROL_LABEL = "control"

#: What joins the texts of a v15 campaign's live questions into the one question the campaign asks.
QUESTION_SEPARATOR = "\n\n"

#: The design fields that belong to the campaign as a whole once it has aspects, so an aspect's design drops them.
_MOVED_TO_THE_CAMPAIGN = frozenset(
    {"questions", "control", "control_levers", "control_identity_version", "declared_at", "declared_by"}
)


def upgrade_campaign_v15(document: Mapping[str, Any]) -> dict[str, Any]:
    """A stored v15 ``eval_campaign`` as the campaign of aspects it reads as: one aspect, its arms, its floors.

    Pure: no store, host, clock or randomness, and ``document`` is not modified. The fields every campaign is
    addressed by (``id``, ``doc_type``, ``schema_version``, ``scope_id``, ``subject_id``, ``behavior``,
    ``archived``, ``created_at``) and every field below not named here pass through untouched; ``schema_version`` is
    the chain's to set, never a step's.

    - **One aspect**, labelled with ``behavior`` (each ``/`` in it read as ``-``, since an aspect label is the first
      part of a floor's key and takes no ``/``): the campaign's ``template_id`` and ``run_ids``, ``stratum`` None,
      and its design — the stored design less its questions, its control and when and by whom it was declared, plus
      ``merit_axes``: the live questions' axes, each once in the order first named; empty when a live question
      names none (an unscoped question asks about every axis, so the joined one does too); None when no question is
      live, since a design that asks nothing confirms nothing. ``declared_at`` is the design's. With no stored design
      the aspect has none, and the campaign declares nothing.
    - **The margin** is the ``max_points`` of the floor that reads every pass rate (below), so the margin a drop is
      tested against and the floor a candidate is warned on are one number; None when there is no such floor.
    - **The arms**: the declared control alone, labelled :data:`UPGRADED_CONTROL_LABEL`, carrying the stored key,
      lever map and predicate as they are; none when the design declares no control.
    - **The question**: the live questions' texts in declaration order, joined by :data:`QUESTION_SEPARATOR`; empty
      when none is live. A question is live when it is neither retired nor superseded by another.
    - **The floors**: every clause of every live question's rule, on the one aspect, in order. Two clauses on the
      same reading of the same kind are one floor: the tighter ``max_points`` of two rate drops, since dropping the
      tighter would loosen what its author asked to be warned of, and one cost floor.
    - **When it was declared**: the latest live question's ``asked_at``, since the campaign's question is asked in full
      only once its last part was, and a reading confirms only what was asked before the evidence existed. Never the
      design's ``declared_at``, which every later write of the design re-stamps (naming the control after runs joined,
      among them), so reading it would turn a pre-registered question post hoc. The design's time stands only when no
      question is live. ``declared_by`` is the design's. None and empty with no design. **The call** is None: no v15
      campaign recorded one.

    :param document: The stored v15 campaign, as the store returned it.
    :ptype document: Mapping[str, Any]
    :return: A new document in the campaign-of-aspects shape, at the version it was read at.
    :rtype: dict[str, Any]
    :raises ValueError: A live question's ``asked_at`` is not an ISO-8601 time with an offset, so which was asked last
        cannot be said (:func:`~threetears.evals.kernel.declaration.aware_instant`, as the pre-registration read
        refuses it).
    """
    upgraded = copy.deepcopy(dict(document))
    template_id = upgraded.pop("template_id", None)
    run_ids = upgraded.pop("run_ids", [])
    design = upgraded.pop("declared_design", None)
    # A label is the first part of a floor's key, which `/` separates, so a behavior holding one is spelled with `-`.
    label = upgraded["behavior"].replace("/", "-")
    aspect: dict[str, Any] = {
        "label": label,
        "template_id": template_id,
        "stratum": None,
        "margin": None,
        "design": None,
        "run_ids": run_ids,
        "declared_at": None,
    }
    upgraded.update(question="", arms=[], aspects=[aspect], floors=[], declared_at=None, declared_by="", call=None)
    if design is None:
        return upgraded

    live = _live_questions(design.get("questions") or [])
    clauses = [clause for question in live for clause in (question.get("decision") or {}).get("clauses") or []]
    aspect_design = {key: value for key, value in design.items() if key not in _MOVED_TO_THE_CAMPAIGN}
    aspect_design["merit_axes"] = _joined_merit_axes(live) if live else None
    floors = _floor_clauses(clauses)
    aspect.update(margin=_margin(floors), design=aspect_design, declared_at=design.get("declared_at"))
    upgraded["question"] = QUESTION_SEPARATOR.join(question["text"] for question in live)
    upgraded["floors"] = [{"aspect": label, "clause": clause} for clause in floors]
    upgraded["declared_at"] = _last_asked(live) if live else design.get("declared_at")
    upgraded["declared_by"] = design.get("declared_by", "")
    if design.get("control") is not None:
        upgraded["arms"] = [
            {
                "label": UPGRADED_CONTROL_LABEL,
                "role": "control",
                "variant_key": design["control"],
                "levers": design.get("control_levers"),
                "identity_version": design.get("control_identity_version"),
            }
        ]
    return upgraded


def _live_questions(questions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The stored questions neither retired nor superseded, in declaration order."""
    superseded = {question.get("supersedes") for question in questions if question.get("supersedes")}
    return [
        question for question in questions if question.get("retired_at") is None and question["id"] not in superseded
    ]


def _joined_merit_axes(live: Sequence[Mapping[str, Any]]) -> list[str]:
    """The axes the joined question asks about: each live question's, once, or none when one of them names none."""
    joined: list[str] = []
    for question in live:
        axes = question.get("merit_axes") or []
        if not axes:
            return []
        joined.extend(axis for axis in axes if axis not in joined)
    return joined


def _margin(floors: Sequence[Mapping[str, Any]]) -> float | None:
    """The ``max_points`` of the floor reading every pass rate, or None."""
    return next(
        (
            float(clause["max_points"])
            for clause in floors
            if clause.get("kind", "rate_drop") == "rate_drop" and clause.get("reading") is None
        ),
        None,
    )


def _floor_clauses(clauses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The clauses as floors: one per kind and reading, the tighter of two rate drops, in first-named order."""
    kept: dict[tuple[str, str | None], dict[str, Any]] = {}
    for clause in clauses:
        key = (clause.get("kind", "rate_drop"), clause.get("reading"))
        held = kept.get(key)
        if held is None:
            kept[key] = dict(clause)
        elif key[0] == "rate_drop" and clause["max_points"] < held["max_points"]:
            held["max_points"] = clause["max_points"]
    return list(kept.values())


def _last_asked(live: Sequence[Mapping[str, Any]]) -> str:
    """The ``asked_at`` of the live question asked last, as stored."""
    last = max(live, key=lambda question: aware_instant(question["asked_at"], f"question {question['id']}"))
    return str(last["asked_at"])
