"""the ledger's import-keyed entries: owner ruling 1's record of a confinement module's own test.

A ledger entry for such a test names an IMPORT, not an attribute read, so the reconciler must
resolve it against the file's private import bindings or report every one of them as stale. And
the numbering has to be the one the private-binding gate uses to match the entry, or an entry
covers a different import than the one its rationale was written for.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import import_bindings, unresolved_entries


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestImportBindings:
    def test_every_private_segment_of_every_import_is_one_binding_in_line_order(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path / "test_x.py",
            "import os\n"
            "from pkg._vendor_internals import sock\n"
            "from pkg import _vendor_internals, public\n"
            "from pkg._vendor_internals import _RETRIES\n"
            "def test_late() -> None:\n"
            "    import pkg._vendor_internals.deep\n",
        )

        assert import_bindings(source) == {
            ("<module>", "_vendor_internals", 0): 2,
            ("<module>", "_vendor_internals", 1): 3,
            ("<module>", "_vendor_internals", 2): 4,
            ("<module>", "_RETRIES", 0): 4,
            ("test_late", "_vendor_internals", 0): 6,
        }

    def test_a_file_with_no_private_import_has_none(self, tmp_path: Path) -> None:
        source = _write(tmp_path / "test_y.py", "import os\nfrom pkg import public\nvalue = obj._attribute\n")

        assert import_bindings(source) == {}


class TestUnresolvedEntriesReadImports:
    def test_an_import_keyed_entry_resolves_while_the_import_exists(self, tmp_path: Path) -> None:
        _write(tmp_path / "tests" / "test_vendor_internals.py", "from pkg._vendor_internals import sock\n")
        ledger = _write(
            tmp_path / "_exemptions.txt",
            "# rationale: pins the vendor members\ntests/test_vendor_internals.py:<module>#0:_vendor_internals\n",
        )

        assert unresolved_entries(ledger, tmp_path) == []

    def test_an_import_keyed_entry_is_stale_once_the_import_is_gone(self, tmp_path: Path) -> None:
        _write(tmp_path / "tests" / "test_vendor_internals.py", "from pkg.public import sock\n")
        ledger = _write(
            tmp_path / "_exemptions.txt",
            "# rationale: pins the vendor members\ntests/test_vendor_internals.py:<module>#0:_vendor_internals\n",
        )

        assert unresolved_entries(ledger, tmp_path) == ["tests/test_vendor_internals.py:<module>#0:_vendor_internals"]
