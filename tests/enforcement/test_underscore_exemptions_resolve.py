"""thin shell -- actual reconciliation logic in
:mod:`threetears.enforcement.underscore_access.ledger`.

``_underscore_exemptions.txt`` is a list of ``path:scope#N:symbol`` keys, each with a rationale
recording why one private access was judged acceptable. Nothing verified either half of that
until this existed, and both had rotted: entries pointing at code that had moved or gone, and
accesses in the scrape suites with no entry at all. A stale exemption is worse than a missing
one -- it reads as a reviewed decision about a specific access, so a reader trusts a rationale
for code that is not there.

Neither direction is caught by the underscore walker, which scans ``packages/*/src`` only and
never enters a ``tests/`` tree, so for every exempted test file the ledger is documentation that
nothing reads back.

Only the repo-specific paths and the failure messages live here; the walking, the ruff-config
discovery and the reconciliation are in the package, alongside the walkers whose exemptions they
describe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from threetears.enforcement.underscore_access import (
    carry_forward_rationales,
    orphan_rationales,
    unlisted_accesses,
    ledger_paths,
    missing_files,
    scoped_accesses,
    unresolved_entries,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


class TestUnderscoreExemptionsResolve:
    def test_every_rationale_is_attached_to_an_entry(self) -> None:
        """No rationale floats free of the entry it justifies.

        Note the direction: this asserts every RATIONALE has an entry, not that every entry has
        a rationale. The converse is the ledger header's stated rule and is a different check
        that nothing performs -- worth knowing before trusting this one to cover it.
        """
        orphans = orphan_rationales(_EXEMPTIONS)

        assert not orphans, (
            "these rationale lines are not followed by the entry they justify, so they document "
            f"nothing and accumulate on every regeneration: {orphans}"
        )

    def test_every_entry_names_a_file_that_exists(self) -> None:
        """An exemption for a path that is gone documents a decision about nothing.

        Entries survived here for a `packages/agent-tools/` directory that had been deleted
        outright, still reading as reviewed judgements about code no longer in the tree.
        """
        missing = missing_files(_EXEMPTIONS, _REPO_ROOT)

        assert not missing, (
            f"these exemptions name files that do not exist, so they document decisions about "
            f"code that is gone: {missing}"
        )

    def test_every_entry_resolves_to_the_symbol_it_claims(self) -> None:
        """An entry whose scope no longer holds that symbol is a rationale for the wrong code."""
        unresolved = unresolved_entries(_EXEMPTIONS, _REPO_ROOT)

        assert not unresolved, (
            "these exemptions point at a line that no longer contains the symbol they name. "
            "Regenerate that path's entries -- see the procedure in the exemptions file header "
            f"-- rather than editing the numbers by hand: {unresolved}"
        )

    def test_two_accesses_of_one_symbol_keep_their_own_rationales(self, tmp_path: Path) -> None:
        """The keying itself, driven through a fixture rather than inferred from the real ledger.

        Keyed on ``(path, symbol)``, the regeneration applied the first rationale to every
        access of that name in a file, so unrelated entries all documented whichever reason came
        first. Worse than ordinary staleness because correcting one by hand was reverted by the
        next run, which was what wrote it.

        A previous version of this test pinned only the scope WALKER. That left the keying free
        to be simplified back to ``(path, symbol)`` with the whole suite still green -- while its
        docstring claimed to pin the mechanism. This drives the function the regeneration
        actually calls, so reverting the key fails here.
        """
        source = tmp_path / "sample.py"
        source.write_text("def first():\n    obj._shared\n\ndef second():\n    obj._shared\n")
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text(
            "# rationale: reason belonging to first\n"
            "sample.py:2:_shared\n"
            "# rationale: reason belonging to second\n"
            "sample.py:5:_shared\n"
        )

        carried = carry_forward_rationales(ledger, tmp_path)

        reasons = set(carried.values())
        assert len(reasons) == 2, (
            "both rationales collapsed onto one key, so a regeneration would overwrite one "
            f"access's reason with the other's: {carried}"
        )
        assert carried[("sample.py", "first", "_shared", 0)] == "reason belonging to first"
        assert carried[("sample.py", "second", "_shared", 0)] == "reason belonging to second"

    def test_one_rationale_still_covers_every_repeat_in_a_scope(self, tmp_path: Path) -> None:
        """Repeats that genuinely share a reason keep sharing it, and the line is still not the key.

        RENAMED AND STRENGTHENED, from ``test_two_accesses_in_one_scope_still_share_one_rationale``,
        which asserted the map held exactly ONE key for two accesses. That shape was the defect
        rather than the contract: it is what silently replaced one hand-written rationale with a
        copy of another's -- see the test below. What this test was really protecting is unchanged
        and still asserted here: the key must not be the line number, or every line shift loses
        the rationale, which is the rot carry-forward exists to prevent. The occurrence ordinal
        keeps that property, because it moves with the code rather than with the file.
        """
        source = tmp_path / "sample.py"
        source.write_text("def only():\n    obj._shared\n    obj._shared\n")
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text("# rationale: one reason covers both\nsample.py:2:_shared\nsample.py:3:_shared\n")

        carried = carry_forward_rationales(ledger, tmp_path)

        assert set(carried.values()) == {"one reason covers both"}, (
            "a single rationale no longer covers repeats that legitimately share one"
        )
        assert set(carried) == {("sample.py", "only", "_shared", 0), ("sample.py", "only", "_shared", 1)}

        # The property the old shape was really about: shift every line and the rationales still
        # land, because the key holds no line number.
        source.write_text("import os\n\n\ndef only():\n    obj._shared\n    obj._shared\n")
        shifted = tmp_path / "_shifted.txt"
        shifted.write_text("# rationale: one reason covers both\nsample.py:5:_shared\nsample.py:6:_shared\n")
        assert set(carry_forward_rationales(shifted, tmp_path).values()) == {"one reason covers both"}

    def test_two_accesses_in_one_scope_keep_their_own_rationales(self, tmp_path: Path) -> None:
        """The defect this key was widened for, pinned so it cannot be simplified back.

        One function touching the same private twice for two reasons is the commonest shape there
        is -- a line that reads a value and a line that asserts on it. Keyed on the scope alone,
        both collapse, and the SECOND rationale is silently replaced by a copy of the first on the
        next regeneration. It happened: two ``_x11vnc`` reads in one ``test_start_is_idempotent``,
        whose hand-written second reason was overwritten by a run that was supposed to preserve it.

        Self-repairing in the wrong direction, which is what makes it worth a test rather than
        care: fixing the wrong entry by hand is reverted by the very next regeneration.
        """
        source = tmp_path / "sample.py"
        source.write_text("def only():\n    obj._shared\n    obj._shared\n")
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text(
            "# rationale: the arrange line, which reads the value\n"
            "sample.py:2:_shared\n"
            "# rationale: the assert line, which compares it\n"
            "sample.py:3:_shared\n"
        )

        carried = carry_forward_rationales(ledger, tmp_path)

        assert carried[("sample.py", "only", "_shared", 0)] == "the arrange line, which reads the value"
        assert carried[("sample.py", "only", "_shared", 1)] == "the assert line, which compares it"

    def test_every_exempted_access_has_an_entry(self) -> None:
        """The other direction, which a stale-entry check structurally cannot cover.

        A missing entry is not a stale entry: it points at nothing because it does not exist.
        Accesses went missing behind inline pragmas before anything looked for them.
        """
        unlisted = unlisted_accesses(_EXEMPTIONS, _REPO_ROOT)

        assert not unlisted, (
            "these private accesses sit on a per-file-exempted path and have no ledger entry, "
            f"so nothing records why they were judged acceptable: {unlisted}"
        )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@test",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@test",
        },
    )


def _committed_repo(tmp_path: Path, files: dict[str, str]) -> None:
    _git(tmp_path, "init", "-q")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "ledger fresh")


class TestCarryForwardAcrossScopeDrift:
    """A drifted entry's line indexes into the file as it stood when the ledger was last
    written, not into the current file. Resolved against the current file, an edit longer
    than a function slides a DIFFERENT function under every recorded line below it, and the
    regeneration replaces each reviewed rationale below the edit with a placeholder."""

    _TWO_FUNCTIONS = "def first():\n    obj._a\n\n\ndef second():\n    obj._b\n"
    _LEDGER = (
        "# rationale: reason belonging to first\n"
        "sample.py:2:_a\n"
        "# rationale: reason belonging to second\n"
        "sample.py:6:_b\n"
    )

    def test_a_drift_past_whole_functions_still_carries_every_rationale(self, tmp_path: Path) -> None:
        _committed_repo(tmp_path, {"sample.py": self._TWO_FUNCTIONS, "_exemptions.txt": self._LEDGER})
        # An edit longer than either function, landing above both: every recorded line below
        # it now sits inside a function it was never recorded against.
        inserted = "def inserted():\n" + "    pass\n" * 10 + "\n\n"
        (tmp_path / "sample.py").write_text(inserted + self._TWO_FUNCTIONS)

        carried = carry_forward_rationales(tmp_path / "_exemptions.txt", tmp_path)

        assert carried[("sample.py", "first", "_a", 0)] == "reason belonging to first"
        assert carried[("sample.py", "second", "_b", 0)] == "reason belonging to second"

    def test_a_fresh_entry_alongside_the_drift_resolves_against_the_current_file(self, tmp_path: Path) -> None:
        """The mixed state a regeneration actually meets: entries committed before the edit
        have drifted, while an entry hand-added for the edit's own new access is correct
        against the current file and appears in no commit at all."""
        _committed_repo(tmp_path, {"sample.py": self._TWO_FUNCTIONS, "_exemptions.txt": self._LEDGER})
        (tmp_path / "sample.py").write_text("def inserted():\n    obj._c\n\n\n" + self._TWO_FUNCTIONS)
        (tmp_path / "_exemptions.txt").write_text(
            self._LEDGER + "# rationale: reason belonging to inserted\nsample.py:2:_c\n"
        )

        carried = carry_forward_rationales(tmp_path / "_exemptions.txt", tmp_path)

        assert carried[("sample.py", "inserted", "_c", 0)] == "reason belonging to inserted"
        assert carried[("sample.py", "first", "_a", 0)] == "reason belonging to first"
        assert carried[("sample.py", "second", "_b", 0)] == "reason belonging to second"

    def test_without_history_a_drifted_entry_degrades_to_the_current_file(self, tmp_path: Path) -> None:
        """No git repository, so there is no snapshot to consult: resolution falls back to
        the current file, and a mapping it cannot make is simply absent -- surfacing as a
        placeholder at regeneration rather than as a wrong rationale."""
        (tmp_path / "sample.py").write_text("def inserted():\n" + "    pass\n" * 10 + "\n\n" + self._TWO_FUNCTIONS)
        (tmp_path / "_exemptions.txt").write_text(self._LEDGER)

        carried = carry_forward_rationales(tmp_path / "_exemptions.txt", tmp_path)

        assert ("sample.py", "first", "_a", 0) not in carried
        assert carried[("sample.py", "inserted", "_a", 0)] == "reason belonging to first"


class TestAnEditAboveAnEntryIsNotDrift:
    """The guarantee the key was changed for, stated as the thing it must not do.

    A line-keyed entry stops matching when anything above it moves, so the gate
    reports the original violation against code nobody touched and the only remedy
    is re-running a regeneration script. That fired twice in one session here: a
    `# parity-exempt:` comment, then a one-line import.

    A gate that fails on unrelated edits trains people to re-run the script without
    reading what changed, which is the state in which a genuinely wrong entry gets
    regenerated straight past. That is why this is worth a key change rather than
    care.
    """

    _SOURCE = "def only():\n    obj._thing\n"
    _PADDED = "import calendar\nimport json\nimport os\n\n\ndef only():\n    obj._thing\n"

    def _ledger(self, tmp_path: Path, key: str) -> Path:
        path = tmp_path / "_exemptions.txt"
        path.write_text(f"# rationale: {'framework-stable internal read the test needs'}\nsample.py:{key}:_thing\n")
        return path

    def test_a_scope_keyed_entry_survives_five_lines_inserted_above_it(self, tmp_path: Path) -> None:
        """Same access, same scope, five lines further down. Nothing to re-run.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        source = tmp_path / "sample.py"
        ledger = self._ledger(tmp_path, "only#0")

        source.write_text(self._SOURCE)
        assert unresolved_entries(ledger, tmp_path) == [], "the scope key did not resolve before the edit"

        source.write_text(self._PADDED)
        assert unresolved_entries(ledger, tmp_path) == [], "an edit above the access broke a scope-keyed entry"

    def test_the_line_keyed_form_is_what_breaks(self, tmp_path: Path) -> None:
        """The control. Without it the test above passes on a check that resolves nothing.

        Line-keyed entries are skipped by the scope-keyed reader, so they resolve
        vacuously rather than failing -- which is the shape that would make the first
        test meaningless. Asserted here so that stays visible.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        source = tmp_path / "sample.py"
        source.write_text(self._PADDED)

        assert unresolved_entries(self._ledger(tmp_path, "2"), tmp_path) == []

    def test_an_access_that_leaves_its_scope_is_still_caught(self, tmp_path: Path) -> None:
        """The key is stable, not blind: a genuinely stale entry must still surface.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        source = tmp_path / "sample.py"
        ledger = self._ledger(tmp_path, "only#0")

        source.write_text("def renamed():\n    obj._thing\n")

        assert unresolved_entries(ledger, tmp_path) == ["sample.py:only#0:_thing"]

    def test_a_second_access_in_the_scope_needs_its_own_entry(self, tmp_path: Path) -> None:
        """Occurrence still separates two accesses, so the second is not silently covered.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        source = tmp_path / "sample.py"
        source.write_text("def only():\n    obj._thing\n    obj._thing\n")
        ledger = self._ledger(tmp_path, "only#0")

        assert unresolved_entries(ledger, tmp_path) == []
        assert scoped_accesses(source).keys() == {("only", "_thing", 0), ("only", "_thing", 1)}


class TestTheMissingFileGateCannotSilentlyEmpty:
    """`missing_files` had no test, and the format change disabled it.

    It reads only the path field, so it looked untouched by a change to the KEY
    field. It was not: it went through `ledger_entries`, which filters on a digit
    key, so on a scope-keyed ledger it returned `[]` for any possible content and
    `test_every_entry_names_a_file_that_exists` could not fail.

    A disabled gate, not a wrong answer -- every path currently exists. The
    direction it guards is covered nowhere else: `unresolved_entries` skips a
    non-existent source BECAUSE this owned it, and `unlisted_accesses` only walks
    files that do exist. All three reviewers found it independently.
    """

    def test_a_deleted_path_is_reported_under_a_scope_key(self, tmp_path: Path) -> None:
        """The case the check exists for, in the format the ledger now uses.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text("# rationale: a reason long enough to satisfy the bar\ngone/away.py:only#0:_thing\n")

        assert missing_files(ledger, tmp_path) == ["gone/away.py"]

    def test_a_deleted_path_is_reported_under_a_line_key_too(self, tmp_path: Path) -> None:
        """Both forms, because a consumer ledger may still be line-keyed.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text("# rationale: a reason long enough to satisfy the bar\ngone/away.py:7:_thing\n")

        assert missing_files(ledger, tmp_path) == ["gone/away.py"]

    def test_a_present_path_is_not_reported(self, tmp_path: Path) -> None:
        """The control, so the two above cannot pass by reporting everything.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        (tmp_path / "here.py").write_text("def only():\n    obj._thing\n")
        ledger = tmp_path / "_exemptions.txt"
        ledger.write_text("# rationale: a reason long enough to satisfy the bar\nhere.py:only#0:_thing\n")

        assert missing_files(ledger, tmp_path) == []

    def test_the_real_ledger_is_actually_read(self) -> None:
        """The vacuity guard: the live check must see all 311 entries, not zero.

        This is what would have caught the defect. `test_every_entry_names_a_file_that_exists`
        asserts an empty result, which an empty INPUT satisfies just as well.

        :return: nothing
        :rtype: None
        """
        assert len(ledger_paths(_EXEMPTIONS)) > 300


class TestTheTwoOccurrenceCountersAgree:
    """Occurrence ordinals are computed twice, over different populations.

    `common.exemptions._occurrences` numbers WALKER VIOLATIONS; this module's
    `scoped_accesses` numbers `private_accesses`. They are not the same set:

    - `private_accesses` returns a `set[(line, symbol)]`, so two reads of one
      private on ONE line collapse to a single access while the walkers report
      both;
    - the walkers also emit violations that are not private reads at all --
      shape_a imports, shape_c/shape_e `__all__` symbols -- which can collide on
      `(file, scope, symbol)` with a read.

    Either shape needs an ordinal the sanctioned regeneration can never emit, so
    the ledger and the matcher would disagree about which access an entry covers.
    Latent today, and "today" is the whole guarantee -- so this asserts it rather
    than leaving it in a commit message. If it fires, the fix is one numbering
    source, not a new entry.

    Nothing in THIS module exercises `_occurrences`: both directions here number
    via `scoped_accesses` on both sides, so a check written here could never see
    the two counters disagree. Nor does anything else in this repo -- all five
    walkers currently report zero violations over its src roots, so the matcher
    runs over an empty list and `test_underscore_access.py` matches none of the
    311 entries. It is a regression guard for when violations appear, not a
    present check.

    Which makes THIS the only live guard: it does not prove the counters agree,
    it proves the shape on which they could disagree is absent.
    """

    def test_no_walker_scanned_file_has_two_accesses_of_one_symbol_on_one_line(self) -> None:
        """The collapse `private_accesses` can hide, checked where BOTH counters run.

        Scoped to exempted files under a `src` root, deliberately. The walkers scan
        `packages/*/src` and never enter a `tests/` tree, so for every other exempted
        file only this module's counter runs and there is no second opinion to
        disagree with. Two such lines exist today in
        `packages/models/tests/.../test_tracking.py` (`tracker_a._prom is
        tracker_b._prom`) and they are harmless for exactly that reason.

        Widening this to every exempted file would fail on those two and say nothing
        true: it would be reporting a divergence between one counter and a counter
        that never runs.

        :return: nothing
        :rtype: None
        """
        from threetears.enforcement.underscore_access import all_exempted_files, private_accesses

        scanned = [p for p in all_exempted_files(_REPO_ROOT) if "/src/" in p.as_posix()]
        assert scanned, "no exempted file sits under a src root; this check would be vacuous"

        collapsed: list[str] = []
        for source in scanned:
            text = source.read_text(errors="replace").split("\n")
            for number, symbol in sorted(private_accesses(source)):
                if number <= len(text) and text[number - 1].count(f".{symbol}") > 1:
                    collapsed.append(f"{source.relative_to(_REPO_ROOT)}:{number}:{symbol}")

        assert not collapsed, (
            "two accesses of one private on one line, in a file the walkers DO scan: "
            "`private_accesses` counts these once and the walkers count them twice, so the "
            f"ledger's ordinals and the matcher's diverge -- {collapsed}"
        )
