"""the ledger's attribute-read counter, on a file that is not valid utf-8.

The counter reads every file through the shared parse cache, which refuses invalid utf-8, and
falls back to a lenient decode for those. Without the fallback such a file would read as having
no private accesses at all, and its ledger entries would all report as stale.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import private_accesses


class TestPrivateAccesses:
    def test_a_private_read_is_found_in_valid_source(self, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text("def f(other):\n    return other._state\n")
        assert private_accesses(source) == {(2, "_state")}

    def test_a_file_that_is_not_valid_utf8_is_still_read(self, tmp_path: Path) -> None:
        source = tmp_path / "latin1.py"
        source.write_bytes(b"# caf\xe9\ndef f(other):\n    return other._state\n")
        assert private_accesses(source) == {(3, "_state")}
