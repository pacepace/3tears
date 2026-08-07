"""thin shell -- actual reconciliation logic in
:mod:`threetears.enforcement.underscore_access.ledger`.

``_underscore_exemptions.txt`` is a list of ``path:line:symbol`` triples, each with a rationale
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
    missing_files,
    orphan_rationales,
    unlisted_accesses,
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
        """A triple whose line no longer holds that symbol is a rationale for the wrong code."""
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
