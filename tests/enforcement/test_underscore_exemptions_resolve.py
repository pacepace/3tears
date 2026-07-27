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

from pathlib import Path

from threetears.enforcement.underscore_access import (
    enclosing_scopes,
    ledger_entries,
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

    def test_the_regeneration_can_tell_two_accesses_of_one_symbol_apart(self) -> None:
        """A rationale describes one ACCESS, and a file may touch a symbol for several reasons.

        Keyed on ``(path, symbol)``, the regeneration carried the FIRST rationale onto every
        access of that name in the file: three tests reaching for the same helper all documented
        whichever reason came first. Worse than ordinary staleness, because it was not
        correctable -- writing the right one by hand was reverted by the next run, which was
        what wrote the wrong one.

        Asserted on the MECHANISM rather than on the ledger's contents, and that distinction is
        the point. Scanning the file cannot detect the collapse: it destroys the evidence, so
        after it happens there is exactly one rationale where there should be several, and
        nothing left to compare. A version of this test that scanned the ledger passed against
        the reintroduced bug.
        """
        source = _REPO_ROOT / "packages" / "scrape" / "tests" / "test_tool.py"
        scopes = enclosing_scopes(source)
        assert scopes, "no scopes resolved; the keying would degrade to symbol-only everywhere"

        # The real entries this was found on: three accesses of one symbol, in three tests.
        render_once = sorted(
            line
            for path, line, symbol in ledger_entries(_EXEMPTIONS)
            if path.endswith("scrape/tests/test_tool.py") and symbol == "_render_once"
        )
        assert len(render_once) > 1, (
            "this file no longer has multiple accesses of that symbol, so it no longer "
            "exercises the case; point the assertion at another multi-access symbol"
        )
        distinct = {scopes.get(line, "") for line in render_once}
        assert len(distinct) == len(render_once), (
            f"two accesses resolve to the same scope, so the regeneration cannot keep their "
            f"rationales apart: lines {render_once} map to {sorted(distinct)}"
        )

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
