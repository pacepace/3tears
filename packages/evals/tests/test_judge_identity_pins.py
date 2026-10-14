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
- **The released table is not the one the release shipped.** The baseline is read from the tree at ``RELEASED_AT``
  (this file's ``JUDGE_IDENTITIES`` as the release carried it, which its own pin held to what it computed), never
  taken on trust from the typed copy: re-pin ``RELEASED_JUDGE_IDENTITIES`` to the values the failure prints. Until
  a final release ships 3tears-evals there is no released judge to read, and the typed table, empty, stands.
- **A fingerprint was added or removed** (a new built-in prompt, a new scale, a new judged artifact). Pin it in
  both tables as the new tree computes it; a prompt no release shipped needs no note.
"""

from __future__ import annotations

import ast
from pathlib import Path
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
    "builtin:transcript-axis": "fdfa47da36a8",
    "builtin:outcome-axis": "576896bef534",
    "builtin:ordinal:transcript": "61371c63c7dd",
    "builtin:pass_fail:transcript": "6a15a72a00e0",
    "builtin:ordinal:document": "24be20140704",
    "builtin:pass_fail:document": "c2362452be2f",
    "suffix:ordinal:transcript": "244c6e35eec0",
    "suffix:pass_fail:transcript": "77f5d3e70b47",
    "suffix:ordinal:document": "31a65e926680",
    "suffix:pass_fail:document": "469f6f5d4bdb",
}

#: The final release the table below was taken at: the ``since`` of ``MIGRATIONS.toml``'s ``unreleased`` section.
RELEASED_AT = "v0.68.0"

#: Every judge fingerprint as :data:`RELEASED_AT` shipped it, the baseline a wording change is announced against.
#: v0.68.0 shipped no 3tears-evals, so it shipped no judge and no fingerprint can have moved since it. A copy for
#: the reader: what a wording change is measured from is read from the release's tree
#: (:func:`released_judge_identities`), and this table is held to it.
RELEASED_JUDGE_IDENTITIES: dict[str, str] = {}

#: This file, as a path inside the repository: where a released tree's pins are read.
PINS_PATH = "packages/evals/tests/test_judge_identity_pins.py"

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


def judge_table_in(source: str) -> dict[str, str] | None:
    """The ``JUDGE_IDENTITIES`` literal a copy of this file assigns, read as written rather than imported.

    The released tree's own pin held that literal equal to what the tree computed, so reading it is reading the
    fingerprints the release wrote, without installing the release.

    Args:
        source: A copy of this file.

    Returns:
        The table, or None when the copy assigns no ``JUDGE_IDENTITIES`` literal.
    """
    for node in ast.parse(source).body:
        if "JUDGE_IDENTITIES" in surface.assigned_names(node) and isinstance(node, ast.AnnAssign) and node.value:
            table = ast.literal_eval(node.value)
            return table if isinstance(table, dict) else None
    return None


def released_judge_identities(tag: str) -> dict[str, str] | None:
    """Every judge fingerprint the release ``tag`` shipped, computed from its tree, never typed.

    Args:
        tag: A final release tag (:data:`RELEASED_AT`).

    Returns:
        The table, or None when the release shipped no 3tears-evals and so no judge.

    Raises:
        AssertionError: The checkout lacks ``tag``, which says nothing about what it shipped; or the release shipped
            the package without a readable pin table, so what it wrote cannot be told.
    """
    assert surface.tag_exists(tag), (
        f"{tag} is not a tag in this checkout, so the judge fingerprints it shipped cannot be read: fetch the tags "
        "(git fetch --tags, or fetch-depth: 0 in CI)."
    )
    if not surface.package_at(tag):
        return None
    source = surface.source_at(tag, PINS_PATH)
    table = judge_table_in(source) if source is not None else None
    assert table is not None, (
        f"{tag} shipped 3tears-evals, and its tree holds no JUDGE_IDENTITIES literal at {PINS_PATH}, so the "
        "fingerprints it wrote cannot be read from it. Every release since this file was added carries it: check "
        "that RELEASED_AT names the final release MIGRATIONS.toml's `unreleased` section follows."
    )
    return table


def released_table_disagreement(tag: str, computed: dict[str, str] | None, typed: dict[str, str]) -> str | None:
    """Why the typed released table is not the one ``tag`` shipped, or None when it is.

    Args:
        tag: The release the tables are taken at.
        computed: What :func:`released_judge_identities` read from its tree; None when it shipped no judge.
        typed: :data:`RELEASED_JUDGE_IDENTITIES`.

    Returns:
        The failure, with its remedy, or None.
    """
    if computed is None:
        if not typed:
            return None
        return (
            f"{tag} shipped no 3tears-evals, so it shipped no judge, and RELEASED_JUDGE_IDENTITIES names {typed}: "
            "empty it, or set RELEASED_AT to the final release that shipped those fingerprints."
        )
    if typed == computed:
        return None
    return (
        f"RELEASED_JUDGE_IDENTITIES is not what {tag} shipped: its tree pins {computed}. Re-pin "
        "RELEASED_JUDGE_IDENTITIES to that table; a wording change since it is measured from it either way."
    )


def released_baseline() -> dict[str, str]:
    """What a wording change is measured from: :data:`RELEASED_AT`'s tree, or, while that release shipped no
    judge, the typed table, which is then empty."""
    computed = released_judge_identities(RELEASED_AT)
    return RELEASED_JUDGE_IDENTITIES if computed is None else computed


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


def test_the_released_table_is_the_one_the_release_shipped() -> None:
    """Computed from the tree at :data:`RELEASED_AT`, so editing ``RELEASED_AT`` alone cannot satisfy the rule.

    RELEASED_AT is v0.68.0 today, which shipped no 3tears-evals: there is no released judge to read, and the test
    passes on the typed table, which must then be empty. The first final release that ships the package carries
    this file, and from then on the typed table must equal what its tree pins.
    """
    disagreement = released_table_disagreement(
        RELEASED_AT, released_judge_identities(RELEASED_AT), RELEASED_JUDGE_IDENTITIES
    )

    assert disagreement is None, disagreement


def test_a_judge_wording_change_since_the_last_release_is_in_the_notes() -> None:
    moved = moved_without_a_note(released_baseline(), JUDGE_IDENTITIES, _unreleased().notes)

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


def test_a_typed_table_that_disagrees_with_a_released_tree_is_named() -> None:
    """Once a release ships the package, its tree decides; before then only an empty typed table stands."""
    shipped = {"builtin:transcript-axis": "aaaaaaaaaaaa"}

    assert released_table_disagreement("v1.0.0", shipped, shipped) is None
    stale = released_table_disagreement("v1.0.0", shipped, {})
    assert stale is not None and "aaaaaaaaaaaa" in stale and "Re-pin RELEASED_JUDGE_IDENTITIES" in stale
    assert released_table_disagreement("v1.0.0", shipped, {"builtin:transcript-axis": "bbbbbbbbbbbb"}) is not None
    assert released_table_disagreement("v0.65.0", None, {}) is None
    assert released_table_disagreement("v0.65.0", None, shipped) is not None


def test_a_released_copy_of_this_file_gives_back_its_pinned_table() -> None:
    """The reader a released tree goes through, run over this tree's copy: it reads the table as typed."""
    assert judge_table_in(Path(__file__).read_text(encoding="utf-8")) == JUDGE_IDENTITIES
    assert judge_table_in("RELEASED_AT = 'v0.65.0'\n") is None
