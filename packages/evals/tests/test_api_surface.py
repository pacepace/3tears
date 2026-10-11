"""The public surface is held to a checked-in snapshot, and a name never leaves it unrecorded.

``api/surface.json`` is every public root's ``__all__``, one entry per ``root.name`` with its kind and signature,
written by ``scripts/generate_api_snapshot.py`` over the same walk that renders ``docs/reference.md``. This test
compares the installed package with it:

* a name added, or a kind or signature changed, fails until the snapshot is regenerated, so every change to the
  surface is a diff a reviewer sees;
* a name that left a root (moved, renamed or removed) fails until ``MIGRATIONS.toml``'s newest section records
  its new home (or ``removed``) and why, and the generator refuses to write a snapshot without it. The release
  notes are printed from that record (``scripts/release_notes.py``); there is no compatibility shim;
* a name exported from two roots fails unless it is one object and ``api/reexports.toml`` says why the second
  root needs it; two objects under one name always fail.

The rest of the file holds each of those rules on small, made-up surfaces, so a rule that stopped firing is seen
even while the real surface happens to be clean.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")
generator = load_script("generate_api_snapshot.py")
notes = load_script("release_notes.py")

Entry = surface.SurfaceEntry


def _release(*changes: dict[str, Any], version: str = "unreleased") -> Any:
    return surface.Release.model_validate({"version": version, "change": list(changes)})


def _change(source: str, to: str, *names: str, **renamed: str) -> dict[str, Any]:
    return {"from": source, "to": to, "names": list(names), "renamed": renamed, "why": "because"}


A = "threetears.evals.alpha"
B = "threetears.evals.beta"


# --- the real package --------------------------------------------------------------------------------------------


def test_the_installed_surface_is_the_snapshot_and_its_record() -> None:
    findings = surface.all_findings(
        surface.installed_surface(), surface.load_snapshot(), surface.load_migrations(), surface.load_reexports()
    )
    assert not findings, "\n\n".join(f.problem for f in findings)


def test_the_committed_snapshot_is_byte_for_byte_what_the_generator_writes() -> None:
    """A hand edit, or a reformat, of the generated file is caught as well as a stale one."""
    committed = surface.SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert committed == surface.render_snapshot(surface.installed_surface()), (
        f"packages/evals/api/surface.json is not what the generator writes: run `{surface.REGENERATE_SNAPSHOT}`."
    )


def test_no_signature_names_a_module_below_a_root() -> None:
    """A module under a root is internal and may move; a move must not read as a change to the surface."""
    below_a_root = re.compile(r"threetears\.evals\.\w+\.\w+")
    leaking = {name: e.signature for name, e in surface.installed_surface().items() if below_a_root.search(e.signature)}
    assert not leaking


def test_a_name_belongs_to_the_root_holding_its_defining_module_not_the_first_that_exports_it() -> None:
    assert surface.home_root("threetears.evals.kernel.host", "VALIDATED_SLOTS") == "threetears.evals.kernel.host"
    assert surface.home_root("threetears.evals.ops", "EvalSummary") == "threetears.evals.analysis"


def test_the_snapshot_records_what_a_caller_depends_on() -> None:
    installed = surface.installed_surface()
    store = installed["threetears.evals.schema.DocumentStore"]
    assert store.kind == "protocol"
    assert "get(self, doc_id: str, scope_id: str)" in store.signature
    host = installed["threetears.evals.kernel.host.EvalHost"]
    assert host.kind == "dataclass"
    assert host.signature.startswith("EvalHost(profile: HostProfile, storage: EvalStorage")
    # A constant's value is data, not a contract: a schema version moves on every bump.
    assert installed["threetears.evals.schema.CORE_SCHEMA_VERSION"] == Entry("constant (int)", "")


# --- the snapshot ------------------------------------------------------------------------------------------------


def test_an_added_name_fails_until_the_snapshot_is_regenerated() -> None:
    findings = surface.snapshot_findings({f"{A}.new": Entry("function", "new()")}, {}, _release())
    assert [(f.regenerating_fixes, "is new" in f.problem) for f in findings] == [(True, True)]


def test_a_changed_signature_fails_and_shows_both() -> None:
    findings = surface.snapshot_findings(
        {f"{A}.f": Entry("function", "f(x: int)")}, {f"{A}.f": Entry("function", "f()")}, _release()
    )
    [finding] = findings
    assert finding.regenerating_fixes
    assert "was function `f()`, now function `f(x: int)`" in finding.problem


def test_a_name_that_left_unrecorded_is_not_fixed_by_regenerating() -> None:
    findings = surface.snapshot_findings({}, {f"{A}.gone": Entry("function", "gone()")}, _release(version="9.9.9"))
    [finding] = findings
    assert not finding.regenerating_fixes
    assert f"{A}.gone left the public surface" in finding.problem
    assert "(9.9.9)" in finding.problem


def test_a_name_that_left_and_is_recorded_needs_only_a_regeneration() -> None:
    newest = _release(_change(A, B, "moved"))
    findings = surface.snapshot_findings({f"{B}.moved": Entry("class", "")}, {f"{A}.moved": Entry("class", "")}, newest)
    assert [f.regenerating_fixes for f in findings] == [True, True]
    assert f"now {B}.moved" in findings[0].problem


def test_a_removal_recorded_only_in_an_older_section_does_not_count() -> None:
    migrations = surface.Migrations.model_validate(
        {
            "release": [
                {"version": "2", "change": [_change(A, B, "other")]},
                {"version": "1", "change": [_change(A, "removed", "gone")]},
            ]
        }
    )
    findings = surface.snapshot_findings({}, {f"{A}.gone": Entry("class", "")}, migrations.releases[0])
    assert [f.regenerating_fixes for f in findings] == [False]


def test_the_generator_refuses_to_write_while_a_removal_is_unrecorded(tmp_path: Path) -> None:
    path = tmp_path / "surface.json"
    blocked = [surface.Finding("x left the public surface", False), surface.Finding("y is new", True)]
    assert generator.write_snapshot({}, blocked, path) is False
    assert not path.exists()
    installed = {f"{A}.y": Entry("function", "y()")}
    assert generator.write_snapshot(installed, [surface.Finding("y is new", True)], path) is True
    assert surface.load_snapshot(path) == installed


def test_a_snapshot_of_another_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "surface.json"
    path.write_text('{"format": 0, "names": {}}', encoding="utf-8")
    with pytest.raises(SystemExit, match="regenerate it"):
        surface.load_snapshot(path)


# --- the newest record against the package -----------------------------------------------------------------------


def test_a_record_of_a_name_still_exported_is_refused() -> None:
    installed = {f"{A}.kept": Entry("class", ""), f"{B}.kept": Entry("class", "")}
    [finding] = surface.record_findings(installed, _release(_change(A, B, "kept")))
    assert "still exported" in finding.problem and not finding.regenerating_fixes


def test_a_record_naming_a_home_that_does_not_export_the_name_is_refused() -> None:
    [finding] = surface.record_findings({}, _release(_change(A, B, "lost")))
    assert f"{B}.lost, which is not exported" in finding.problem


def test_a_removal_and_a_rename_are_recorded_against_their_own_names() -> None:
    newest = _release(_change(A, "removed", "gone"), _change(A, A, Old="New"))
    assert surface.record_findings({f"{A}.New": Entry("class", "")}, newest) == []
    assert [(m.old, m.new) for m in newest.migrations()] == [(f"{A}.gone", "removed"), (f"{A}.Old", f"{A}.New")]


# --- names exported twice ----------------------------------------------------------------------------------------


class _Thing:
    pass


def _reexports(*entries: dict[str, Any]) -> Any:
    return surface.ReExports.model_validate({"reexport": list(entries)})


def test_a_name_exported_twice_undeclared_fails_naming_its_home() -> None:
    exports = {"Thing": [(B, _Thing), (A, _Thing)]}
    [finding] = surface.duplicate_findings(exports, {"Thing": A}, _reexports())
    assert f"its home is {A}" in finding.problem and not finding.regenerating_fixes


def test_a_declared_reexport_passes() -> None:
    exports = {"Thing": [(A, _Thing), (B, _Thing)]}
    declared = _reexports({"home": A, "also": [B], "names": ["Thing"], "why": "B hands it back"})
    assert surface.duplicate_findings(exports, {"Thing": A}, declared) == []


def test_a_declaration_with_the_wrong_home_or_roots_fails() -> None:
    exports = {"Thing": [(A, _Thing), (B, _Thing)]}
    declared = _reexports({"home": B, "also": [A], "names": ["Thing"], "why": "B hands it back"})
    [finding] = surface.duplicate_findings(exports, {"Thing": A}, declared)
    assert "correct the declaration" in finding.problem


def test_two_objects_under_one_name_fail_even_when_declared() -> None:
    exports = {"Thing": [(A, _Thing), (B, int)]}
    declared = _reexports({"home": A, "also": [B], "names": ["Thing"], "why": "B hands it back"})
    [finding] = surface.duplicate_findings(exports, {"Thing": A}, declared)
    assert "different objects" in finding.problem and "Rename one" in finding.problem


def test_a_declaration_for_a_name_exported_once_is_stale() -> None:
    declared = _reexports({"home": A, "also": [B], "names": ["Thing"], "why": "B hands it back"})
    [finding] = surface.duplicate_findings({"Thing": [(A, _Thing)]}, {}, declared)
    assert "drop the declaration" in finding.problem


# --- the hand-written records are strict -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (_change(A, B), "names nothing"),
        (_change(A, "removed", Old="New"), "has no new name"),
        ({**_change(A, B, "X"), "renamed": {"X": "Y"}}, "both `names` and `renamed`"),
        (_change(A, A, "X"), "to itself"),
        (_change("alpha", B, "X"), "write the root in full"),
        ({**_change(A, B, "X"), "becuase": "typo"}, "Extra inputs are not permitted"),
        ({**_change(A, B, "X"), "why": ""}, "at least 1 character"),
    ],
)
def test_a_malformed_change_is_refused(change: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        _release(change)


def test_a_name_recorded_twice_in_one_release_is_refused() -> None:
    with pytest.raises(ValidationError, match="twice"):
        _release(_change(A, B, "X"), _change(A, "removed", "X"))


def test_two_sections_for_one_version_are_refused() -> None:
    with pytest.raises(ValidationError, match="more than one section"):
        surface.Migrations.model_validate({"release": [{"version": "1"}, {"version": "1"}]})


# --- release notes -----------------------------------------------------------------------------------------------


def _notes(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = notes.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_the_release_notes_carry_the_old_to_new_map(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _notes(capsys, "v0.66.0-evaladopt.6")
    assert code == 0
    assert out.startswith("## 3tears-evals 0.66.0-evaladopt.6")
    assert "Every name that left a public root between" in out
    assert "- `threetears.evals.quick.MeasureSummary` → `threetears.evals.quick.RunMeasureSummary`" in out
    assert "- `threetears.evals.analysis.TwoPillarDisclosure` → removed" in out
    assert "- `threetears.evals.contracts.ActorPolicy` → `threetears.evals.schema.ActorPolicy`" in out
    assert "- `threetears.evals.contracts.host.EvalHost` → `threetears.evals.kernel.host.EvalHost`" in out
    assert "### Other breaking changes" in out


def test_the_release_notes_of_a_tag_are_what_it_shipped_and_the_next_release_holds_the_rest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, tagged, _ = _notes(capsys, "v0.66.0-evaladopt.7")
    assert code == 0 and tagged.startswith("## 3tears-evals 0.66.0-evaladopt.7")
    assert "`threetears.evals.run.EvalJobTimeout` → `threetears.evals.kernel.EvalJobTimeout`" in tagged
    assert "NonPublicImport" not in tagged and "Report version 7" not in tagged
    code, following, _ = _notes(capsys, "v0.66.0-evaladopt.8")
    assert code == 0 and following.startswith("## 3tears-evals 0.66.0-evaladopt.8")
    assert "`threetears.evals.testing.NonPublicImport` → `threetears.evals.testing.BoundaryFinding`" in following
    assert "Report version 7" in following and "EvalJobTimeout" not in following


def test_the_release_notes_show_a_rename_and_a_removal() -> None:
    release = _release(_change(A, A, Cell="ChartValue"), _change(B, "removed", "gone"))
    text = "\n".join(notes.render(release))
    assert f"- `{A}.Cell` → `{A}.ChartValue`" in text
    assert f"### `{B}` → removed" in text and f"- `{B}.gone` → removed" in text


def test_release_notes_for_an_unknown_version_name_the_ones_there_are(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _notes(capsys, "v0.0.0")
    assert code == 1
    assert "no section for 'v0.0.0'" in err and "0.66.0-evaladopt.6" in err


def test_the_scripts_share_one_walk() -> None:
    """The generators read the shared module, not a copy of it."""
    reference = load_script("generate_reference.py")
    assert isinstance(reference, ModuleType)
    assert reference.describe is surface.describe
    assert generator.installed_surface is surface.installed_surface


# --- a section against the two releases it sits between ----------------------------------------------------------


def _section(version: str) -> Any:
    release = surface.load_migrations().release(version)
    assert release is not None
    return release


def test_the_departures_are_what_a_root_stopped_exporting() -> None:
    before = {A: {"kept", "moved"}, B: {"gone"}}
    after = {A: {"kept"}, "threetears.evals.gamma": {"moved"}}
    assert surface.departures(before, after) == {f"{A}.moved", f"{B}.gone"}


def test_the_0_66_0_evaladopt_6_section_records_exactly_what_left_between_its_tags() -> None:
    release = _section("0.66.0-evaladopt.6")
    assert release.since == "v0.66.0-evaladopt.5"
    assert surface.history_findings(release) == []
    assert len(release.migrations()) == len(
        surface.departures(surface.exports_at("v0.66.0-evaladopt.5"), surface.exports_at("v0.66.0-evaladopt.6"))
    )


def test_a_section_missing_a_departure_is_refused() -> None:
    release = _section("0.66.0-evaladopt.6")
    short = release.model_copy(update={"changes": release.changes[1:]})
    findings = surface.history_findings(short)
    assert findings and all("does not record it" in f.problem for f in findings)


def test_a_section_recording_a_name_that_never_left_is_refused() -> None:
    release = _section("0.66.0-evaladopt.6")
    extra = surface.NameChange.model_validate(_change("threetears.evals.run", "removed", "launch_run"))
    padded = release.model_copy(update={"changes": [*release.changes, extra]})
    [finding] = surface.history_findings(padded)
    assert "threetears.evals.run.launch_run, which did not leave its root" in finding.problem


def test_a_section_sending_a_name_where_that_release_does_not_export_it_is_refused() -> None:
    wrong = _release(_change("threetears.evals.contracts", "threetears.evals.vega", "EvalRun"))
    wrong = wrong.model_copy(update={"version": "0.66.0-evaladopt.6", "since": "v0.66.0-evaladopt.5"})
    findings = surface.history_findings(wrong)
    assert any(
        "to threetears.evals.vega.EvalRun, which v0.66.0-evaladopt.6 does not export" in f.problem for f in findings
    )


def test_a_checkout_without_the_earlier_tag_cannot_pass_the_check() -> None:
    release = _release(_change(A, "removed", "gone")).model_copy(update={"since": "v0.0.0-never-tagged"})
    [finding] = surface.history_findings(release)
    assert "fetch the tags" in finding.problem and not finding.regenerating_fixes


def test_the_0_66_0_evaladopt_7_section_records_exactly_what_left_between_its_tags() -> None:
    release = _section("0.66.0-evaladopt.7")
    assert release.since == "v0.66.0-evaladopt.6"
    assert surface.history_findings(release) == []
    assert len(release.migrations()) == len(
        surface.departures(surface.exports_at("v0.66.0-evaladopt.6"), surface.exports_at("v0.66.0-evaladopt.7"))
    )


def test_the_newest_section_is_checked_against_its_tag_or_the_working_tree() -> None:
    """The newest section is "unreleased" between releases and named for its version once a release is cut."""
    release = surface.load_migrations().releases[0]
    assert release.since is not None
    assert surface.history_findings(release) == []


# --- a tagged release has its own section ------------------------------------------------------------------------


def _labelled(*sections: tuple[str, str | None]) -> Any:
    return surface.Migrations.model_validate(
        {"release": [{"version": version, "since": since} for version, since in sections]}
    )


def test_the_newest_tag_behind_head_has_its_own_section() -> None:
    """Tagging a release whose section still reads "unreleased" leaves the next release's work in it."""
    assert surface.labelling_findings(surface.load_migrations(), surface.newest_tag()) == []


def test_a_tag_shipped_with_its_section_still_unreleased_is_refused() -> None:
    migrations = _labelled(("unreleased", "v2.0.0"), ("2.0.0", "v1.0.0"))
    unlabelled, stale_since = surface.labelling_findings(migrations, "v3.0.0")
    assert "v3.0.0 is tagged, and MIGRATIONS.toml has no section for it" in unlabelled.problem
    assert 'version = "3.0.0"' in unlabelled.problem and 'since = "v3.0.0"' in unlabelled.problem
    assert "the unreleased section is since v2.0.0" in stale_since.problem
    assert not unlabelled.regenerating_fixes and not stale_since.regenerating_fixes


def test_the_unreleased_section_starts_at_the_newest_tag() -> None:
    migrations = _labelled(("unreleased", "v2.0.0"), ("3.0.0", "v2.0.0"), ("2.0.0", "v1.0.0"))
    [finding] = surface.labelling_findings(migrations, "v3.0.0")
    assert (
        "the unreleased section is since v2.0.0, but v3.0.0 is the newest release this tree follows" in finding.problem
    )


def test_a_relabelled_section_and_a_fresh_unreleased_one_pass() -> None:
    assert surface.labelling_findings(_labelled(("unreleased", "v3.0.0"), ("3.0.0", "v2.0.0")), "v3.0.0") == []


def test_the_tagged_commit_itself_passes_with_its_section_newest() -> None:
    """Relabelling happens before the tag, so the tagged tree's newest section is its own version."""
    assert surface.labelling_findings(_labelled(("3.0.0", "v2.0.0")), "v3.0.0") == []


#: The tags of a family whose pre-releases were cut on the branch carrying their work, then released on main.
_PRE_RELEASES = ["v0.66.0-evaladopt.1", "v0.66.0-evaladopt.9", "v0.66.0-evaladopt.10"]


def test_the_evaladopt_flow_follows_the_newest_pre_release_its_branch_carries() -> None:
    """Pre-releases are tagged on the branch carrying the work, so the branch follows its own, in release order."""
    every = ["v0.64.0", "v0.65.0", *_PRE_RELEASES]
    assert surface.newest_release_tag(every, every, "0.66.0") == "v0.66.0-evaladopt.10"
    migrations = _labelled(("unreleased", "v0.66.0-evaladopt.10"), ("0.66.0-evaladopt.10", "v0.66.0-evaladopt.9"))
    assert surface.labelling_findings(migrations, surface.newest_release_tag(every, every, "0.66.0")) == []


def test_a_branch_that_has_not_merged_a_pre_release_does_not_follow_it() -> None:
    """develop, before the work a pre-release was cut from reaches it, still follows the last final release."""
    every = ["v0.64.0", "v0.65.0", *_PRE_RELEASES]
    assert surface.newest_release_tag(["v0.64.0"], every, "0.66.0") == "v0.65.0"


def test_the_main_tag_flow_follows_a_release_tagged_where_develop_cannot_reach_it() -> None:
    """A release is tagged on main and develop never merges main back: develop follows it all the same.

    Read by reachability, develop's newest tag stayed the release before, so the section it had renamed for the
    release and the unreleased section since it read as wrong the moment the release was tagged.
    """
    every = ["v0.64.0", "v0.65.0", *_PRE_RELEASES, "v0.66.0"]
    develop_reaches = ["v0.64.0", "v0.65.0", *_PRE_RELEASES]
    for prepared in ("0.66.0", "0.67.0"):  # before, and after, develop bumps to the next version
        tag = surface.newest_release_tag(develop_reaches, every, prepared)
        assert tag == "v0.66.0"
        assert surface.labelling_findings(_labelled(("unreleased", "v0.66.0"), ("0.66.0", "v0.65.0")), tag) == []
    # A final release above the version a tree prepares has not been shipped past by it.
    assert surface.newest_release_tag(["v0.64.0"], ["v0.64.0", "v0.65.0", "v0.66.0"], "0.65.0") == "v0.65.0"


def test_a_final_release_sorts_above_its_pre_releases_and_a_non_release_tag_is_not_one() -> None:
    every = ["v0.66.0", "v0.66.0-evaladopt.10", "nightly", "v0.66", "release-0.67.0"]
    assert surface.newest_release_tag(every, every, "0.66.0") == "v0.66.0"
    assert surface.newest_release_tag([], [], "0.66.0") is None


def test_a_checkout_without_tags_cannot_pass_the_labelling_check() -> None:
    [finding] = surface.labelling_findings(_labelled(("unreleased", "v1.0.0")), None)
    assert "fetch the tags" in finding.problem


def test_each_tagged_section_is_checked_against_its_own_tags() -> None:
    """Every section with a `since` is held to what left between that tag and its own (the working tree only for
    the unreleased one), so relabelling a section moves which surfaces it is read against."""
    for release in surface.load_migrations().releases:
        if release.since is None:
            continue
        assert surface.history_findings(release) == [], release.version
