"""The engine's judge wording is pinned: a release cannot change a judge's identity without saying so in its notes.

A judge's recorded identity is a fingerprint of the wording it was sent, not a counter
(:func:`~threetears.evals.schema.models.builtin_judge_config_id`, and
:func:`~threetears.evals.schema.models.configured_judge_config_id` for the reply-format suffix a configured judge is
sent). So a release that rewords a built-in prompt, or the suffix, moves no version number and refuses no stored
document, while every result judged before it becomes another judge to agreement, judge profiles, drift and evidence
tiers. ``schema/versioning.py`` makes that a breaking fact a release's ``MIGRATIONS.toml`` notes MUST carry; this
file is what makes the policy hold. It pins every fingerprint the engine writes by value, twice: as the tree computes
it today, and as the newest final release shipped it.

**When this goes red**, one of these happened:

- **A fingerprint moved.** The engine's wording changed, on purpose or as a side effect of an edit to the prompt
  text, its JSON-format contract, a scale's wording or the sourced-facts rule. Re-pin ``JUDGE_IDENTITIES`` to the
  values the failure prints, and put a note in ``MIGRATIONS.toml``'s ``unreleased`` section that carries
  ``JUDGE_WORDING_CHANGE_SENTENCE`` verbatim and says which wording moved. The note stays required for as long as
  the section is unreleased: the rule compares today's pins with the released ones, not with the previous commit.
- **The unreleased section moved to a new ``since`` tag.** A release was cut, and its pins are now the ones a
  later wording change is measured from. Copy ``JUDGE_IDENTITIES`` as that tag shipped them into
  ``RELEASED_JUDGE_IDENTITIES`` and set ``RELEASED_AT`` to the tag.
- **A fingerprint was added or removed** (a new built-in prompt, a new scale, a new judged artifact). Pin it in
  both tables as the new tree computes it; a prompt no release shipped needs no note.
"""

from __future__ import annotations

from typing import Any, get_args

from packages.evals.tests.script_loader import load_script
from threetears.evals.run.judge_service import builtin_judge_config_ids, judge_config_ids_for
from threetears.evals.schema.models import (
    JUDGE_WORDING_CHANGE_SENTENCE,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    JudgedArtifact,
    RubricDim,
    RubricScale,
    judge_config_of,
)

surface = load_script("public_surface.py")

#: Every judge fingerprint the engine writes, as this tree computes it. Keyed ``builtin:<prompt>`` for a built-in
#: prompt (an axis's, or the rubric dimension prompt of a scale for an artifact) and ``suffix:<scale>:<artifact>`` for
#: the reply-format suffix a configured judge of that scale is sent; the value is the 12 hex characters recorded after
#: ``builtin:`` or ``@``.
JUDGE_IDENTITIES: dict[str, str] = {
    "builtin:transcript-axis": "f886726e63d2",
    "builtin:outcome-axis": "5fd0040e2c22",
    "builtin:ordinal:transcript": "501c023ca46f",
    "builtin:pass_fail:transcript": "6a15a72a00e0",
    "builtin:ordinal:document": "acec207171e0",
    "builtin:pass_fail:document": "c2362452be2f",
    "suffix:ordinal:transcript": "48f825101269",
    "suffix:pass_fail:transcript": "77f5d3e70b47",
    "suffix:ordinal:document": "c31b0cac0f26",
    "suffix:pass_fail:document": "469f6f5d4bdb",
}

#: The final release the table below was taken at: the ``since`` of ``MIGRATIONS.toml``'s ``unreleased`` section.
RELEASED_AT = "v0.65.0"

#: Every judge fingerprint as :data:`RELEASED_AT` shipped it, the baseline a wording change is announced against.
#: v0.65.0 shipped no 3tears-evals, so it shipped no judge and no fingerprint can have moved since it.
RELEASED_JUDGE_IDENTITIES: dict[str, str] = {}

#: A configured judge's id, which the suffix fingerprints are read off.
_CONFIG = "pinned-config"


def _fingerprint(recorded: str) -> str:
    """The 12 hex characters a recorded judge identity carries after ``builtin:`` or ``@``."""
    return recorded.rsplit("@", 1)[1] if judge_config_of(recorded) is not None else recorded.split(":", 1)[1]


def judge_identities_today() -> dict[str, str]:
    """Every judge fingerprint the engine writes, through the functions a result's identities are recorded by."""
    found: dict[str, str] = {}
    for artifact in (JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT):
        rubric = [
            RubricDim(name=f"pinned.{scale}", description="a dim", scale=scale) for scale in get_args(RubricScale)
        ]
        builtin = builtin_judge_config_ids(rubric, judged_artifact=artifact)
        configured = judge_config_ids_for(rubric, {dim.name: _CONFIG for dim in rubric}, judged_artifact=artifact)
        if TRANSCRIPT_DIM_ID in builtin:
            found["builtin:transcript-axis"] = _fingerprint(builtin[TRANSCRIPT_DIM_ID])
            found["builtin:outcome-axis"] = _fingerprint(builtin[OUTCOME_DIM_ID])
        for dim in rubric:
            found[f"builtin:{dim.scale}:{artifact.value}"] = _fingerprint(builtin[dim.name])
            found[f"suffix:{dim.scale}:{artifact.value}"] = _fingerprint(configured[dim.name])
    return found


def moved_without_a_note(released: dict[str, str], today: dict[str, str], notes: list[str]) -> list[str]:
    """The fingerprints a release shipped that moved since, when no note carries the sentence; else none.

    Args:
        released: The fingerprints the newest final release shipped.
        today: The fingerprints this tree writes.
        notes: The unreleased section's ``notes``.

    Returns:
        The moved keys, sorted, while no note carries :data:`JUDGE_WORDING_CHANGE_SENTENCE`.
    """
    moved = sorted(key for key, value in released.items() if key in today and today[key] != value)
    if any(JUDGE_WORDING_CHANGE_SENTENCE in note for note in notes):
        return []
    return moved


def _unreleased() -> Any:
    newest = surface.load_migrations().releases[0]
    assert newest.version == "unreleased", "MIGRATIONS.toml's first section is the release being prepared"
    return newest


# --- the real tree -----------------------------------------------------------------------------------------------


def test_every_judge_fingerprint_is_pinned() -> None:
    """A wording the engine sends a judge cannot change without this table changing with it."""
    today = judge_identities_today()

    assert today == JUDGE_IDENTITIES, (
        "a judge's wording moved, so every result judged before this change is another judge. Re-pin "
        f"JUDGE_IDENTITIES to {today}, and add a note to MIGRATIONS.toml's `unreleased` section that carries "
        f"JUDGE_WORDING_CHANGE_SENTENCE verbatim ({JUDGE_WORDING_CHANGE_SENTENCE!r}) and names the wording that moved."
    )


def test_the_released_pins_are_the_unreleased_sections_baseline() -> None:
    """The released table follows the release the notes are written against, so a cut release rolls it forward."""
    since = _unreleased().since

    assert since == RELEASED_AT, (
        f"MIGRATIONS.toml's `unreleased` section is since {since}, and RELEASED_JUDGE_IDENTITIES was taken at "
        f"{RELEASED_AT}: copy JUDGE_IDENTITIES as {since} shipped them into RELEASED_JUDGE_IDENTITIES and set "
        f"RELEASED_AT to {since!r}."
    )


def test_a_judge_wording_change_since_the_last_release_is_in_the_notes() -> None:
    moved = moved_without_a_note(RELEASED_JUDGE_IDENTITIES, JUDGE_IDENTITIES, _unreleased().notes)

    assert not moved, (
        f"the judge fingerprints {moved} moved since {RELEASED_AT}, so results judged before this release are "
        "another judge for agreement, judge profiles, drift and evidence tiers. Add a note to MIGRATIONS.toml's "
        f"`unreleased` section that carries {JUDGE_WORDING_CHANGE_SENTENCE!r} verbatim and says which wording moved."
    )


def test_every_judge_the_engine_can_record_has_a_pin() -> None:
    """Each axis, and each scale on each artifact a judge reads, for the built-in prompt and for the suffix."""
    artifacts = (JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT)
    expected = {"builtin:transcript-axis", "builtin:outcome-axis"} | {
        f"{prompt}:{scale}:{artifact.value}"
        for prompt in ("builtin", "suffix")
        for scale in get_args(RubricScale)
        for artifact in artifacts
    }

    assert set(JUDGE_IDENTITIES) == expected
    assert set(JudgedArtifact) - set(artifacts) == {JudgedArtifact.UNJUDGED}, "a new judged artifact needs its pins"


# --- the rule, on made-up tables ---------------------------------------------------------------------------------


def test_a_moved_fingerprint_with_no_note_is_named() -> None:
    """The rule above, while the real released table is empty and so can never fire on the real tree."""
    released = {"builtin:transcript-axis": "aaaaaaaaaaaa", "suffix:ordinal:document": "bbbbbbbbbbbb"}
    today = {"builtin:transcript-axis": "cccccccccccc", "suffix:ordinal:document": "bbbbbbbbbbbb"}

    assert moved_without_a_note(released, today, ["an unrelated note"]) == ["builtin:transcript-axis"]
    assert moved_without_a_note(released, today, [f"{JUDGE_WORDING_CHANGE_SENTENCE} The transcript axis."]) == []
    assert moved_without_a_note(released, released, []) == []


def test_a_prompt_no_release_shipped_needs_no_note() -> None:
    assert moved_without_a_note({}, {"builtin:transcript-axis": "cccccccccccc"}, []) == []
