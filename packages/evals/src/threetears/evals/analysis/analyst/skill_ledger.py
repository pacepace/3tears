"""The skill ledger: an entry for every analyst skill this package ships, and the basis it shipped on.

The skill's files are the source of truth for what an analyst and an editor read; ``skill/PROMOTIONS.toml`` beside
them records why each version of them shipped. It is package data, so it ships in the wheel with the skill, and it is
not part of the skill: the skill is its Markdown files alone (:func:`~threetears.evals.analysis.analyst.skill.load_skill`),
so writing an entry never moves the digest it records.

An entry is keyed by the skill's digest (:attr:`AnalystSkill.digest
<threetears.evals.analysis.analyst.skill.AnalystSkill.digest>`), with the analyst's and the editor's system text
digests beside it, so a stored report's recorded digests trace back to the entry they shipped under. Its basis is
``measured``, when a campaign's promotion read decided it (the campaign, the arm promoted, the writer model and the
held-out case set), or ``unmeasured``, with the reason it shipped without one. An unmeasured entry is an honest
record, not a fault: with few writer cases, most edits cannot yet be measured.

:func:`skill_record` is the one reader. It refuses a skill with no entry, naming its digests and the entry to add.
"""

from __future__ import annotations

import datetime
import tomllib
from functools import cache
from importlib.resources import files
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from threetears.evals.analysis.analyst.skill import AnalystSkill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.models import CaseSetRef

#: The ledger's file, beside the skill's files.
_LEDGER = "PROMOTIONS.toml"

#: A sha256, as every digest the ledger records is written.
_DIGEST = r"^[0-9a-f]{64}$"


class _Shipped(BaseModel):
    """What every entry records: which skill, the texts it gives each role, and when."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str = Field(pattern=_DIGEST, description="The skill's digest: every file's path and text.")
    analyst: str = Field(pattern=_DIGEST, description="The analyst's system text's digest, with no note withheld.")
    editor: str = Field(pattern=_DIGEST, description="The editor's system text's digest.")
    date: datetime.date = Field(description="The day the entry was recorded.")


class MeasuredSkill(_Shipped):
    """A skill shipped on a campaign's decision: the promotion read answered ``promote`` for the arm running it."""

    basis: Literal["measured"]
    campaign: str = Field(min_length=1, description="The campaign the decision was read over.")
    scope: str = Field(min_length=1, description="The scope the campaign lives in.")
    candidate: str = Field(pattern=_DIGEST, description="The promoted arm's variant key.")
    decision: Literal["promote"] = Field(description="What the promotion read answered; only a promotion ships.")
    writer_model: str = Field(min_length=1, description="The model the promoted arm wrote with.")
    case_set: CaseSetRef = Field(description="The held-out case set the decision was read on.")


class UnmeasuredSkill(_Shipped):
    """A skill shipped without a campaign's decision, and why."""

    basis: Literal["unmeasured"]
    reason: str = Field(min_length=10, description="Why it shipped without a measurement.")


#: One entry of the ledger: a skill shipped on a decision, or without one.
ShippedSkill = Annotated[MeasuredSkill | UnmeasuredSkill, Field(discriminator="basis")]


class _Ledger(BaseModel):
    """Every entry, in the order recorded; one per skill digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shipped: tuple[ShippedSkill, ...]

    @model_validator(mode="after")
    def _one_entry_per_skill(self) -> _Ledger:
        """Refuse a digest recorded twice, which would give one skill two bases."""
        digests = [entry.skill for entry in self.shipped]
        if twice := sorted({digest for digest in digests if digests.count(digest) > 1}):
            raise ValueError(f"the ledger records skill {', '.join(twice)} more than once")
        return self


@cache
def _ledger() -> _Ledger:
    """The ledger this build ships, read once."""
    text = files("threetears.evals.analysis.analyst").joinpath("skill", _LEDGER).read_text(encoding="utf-8")
    try:
        return _Ledger.model_validate(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ValidationError) as unread:
        raise ValidationFailedError(
            f"the skill ledger skill/{_LEDGER} does not read: {unread}",
            remedy=f"correct skill/{_LEDGER}, as its header describes each entry",
        ) from unread


def skill_record(skill: AnalystSkill) -> ShippedSkill:
    """The ledger's entry for a skill: the basis it shipped on.

    :param skill: The skill, as a build ships it or as an arm runs it.
    :ptype skill: AnalystSkill
    :return: Its entry.
    :rtype: ShippedSkill
    :raises ValidationFailedError: The ledger does not read; it has no entry for the skill; or its entry records
        analyst or editor texts the skill does not give.
    """
    analyst, editor = skill.analyst().digest, skill.editor().digest
    entry = next((each for each in _ledger().shipped if each.skill == skill.digest), None)
    if entry is None:
        raise ValidationFailedError(
            f"the skill ledger has no entry for skill {skill.digest}: every skill a build ships records the basis it "
            "shipped on",
            remedy=(
                f"append an entry to skill/{_LEDGER}: [[shipped]] with skill = {skill.digest!r}, analyst = "
                f"{analyst!r}, editor = {editor!r}, date = the day, and basis = 'measured' with the promotion read's "
                "campaign, scope, candidate, decision, writer_model and case_set, or basis = 'unmeasured' with the "
                "reason it ships without a measurement"
            ),
        )
    wrong = [
        f"{role} {recorded} where the skill gives {given}"
        for role, recorded, given in (("analyst", entry.analyst, analyst), ("editor", entry.editor, editor))
        if recorded != given
    ]
    if wrong:
        raise ValidationFailedError(
            f"the skill ledger's entry for skill {skill.digest} records " + "; ".join(wrong),
            remedy=f"copy the skill's own digests into its entry in skill/{_LEDGER}",
        )
    return entry


__all__ = ["ShippedSkill", "skill_record"]
