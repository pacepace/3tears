"""3tears' own key sources, which a consumer repo's no_whole_reads walk takes as declared without walking 3tears, are
each declared in 3tears' source: annotated ``KeyColumn`` / ``KeyColumns`` / ``KeyValues`` / ``DeclaredKey`` where
3tears defines it. A rename of one leaves its old name here and turns this red, so the list cannot outlive a rename
and no consumer's verdict rests on a name 3tears no longer declares."""

from __future__ import annotations

from tests.enforcement.no_whole_reads_support import NO_WHOLE_READS_CONFIG
from threetears.enforcement.common.ast_helpers import iter_python_files, parse_python_path
from threetears.enforcement.no_whole_reads.keys import THREETEARS_KEY_SOURCES, declared_key_sources


def test_every_3tears_key_source_a_consumer_takes_as_declared_is_declared_in_3tears() -> None:
    trees = [parse_python_path(path) for root in NO_WHOLE_READS_CONFIG.src_roots for path in iter_python_files(root)]
    declared = declared_key_sources(trees, base=frozenset()).names
    assert THREETEARS_KEY_SOURCES <= declared, sorted(THREETEARS_KEY_SOURCES - declared)
