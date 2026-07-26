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
