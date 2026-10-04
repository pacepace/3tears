"""The definition seed loads a corpus for one scope and seeds that scope alone.

Corpus files carry no scope — it is server-owned, like ``id`` — so the scope arrives at load, and
occupancy is asked of the scope being seeded. A corpus holding a definition of another scope is
refused at construction: the seed reads one scope, so such a definition's slot would never read
occupied and every boot would write it again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threetears.evals.run.definition_seed import SeedCorpus, load_seed_corpus, seed_eval_definitions

from packages.evals.tests.factories import make_template
from packages.evals.tests.factories import memory_storage


def _corpus_dir(tmp_path: Path, template: dict[str, object]) -> Path:
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "probe.json").write_text(json.dumps(template), encoding="utf-8")
    return tmp_path


def test_a_corpus_is_built_in_the_scope_it_is_loaded_for(tmp_path: Path) -> None:
    corpus = load_seed_corpus(
        _corpus_dir(tmp_path, {"name": "probe", "intent": "i", "candidate_kind": "test-kind"}), "scope-a"
    )

    assert corpus.scope_id == "scope-a"
    assert [t.scope_id for t in corpus.templates] == ["scope-a"]


def test_a_file_naming_its_own_scope_does_not_load(tmp_path: Path) -> None:
    """The scope is the host's to choose at load; a file that names one is refused rather than obeyed."""
    with pytest.raises(ValueError, match="probe.json"):
        load_seed_corpus(
            _corpus_dir(
                tmp_path, {"name": "probe", "intent": "i", "candidate_kind": "test-kind", "scope_id": "scope-b"}
            ),
            "scope-a",
        )


def test_seeding_writes_the_corpus_scope_and_asks_occupancy_there(tmp_path: Path) -> None:
    directory = _corpus_dir(tmp_path, {"name": "probe", "intent": "i", "candidate_kind": "test-kind"})
    storage, _ = memory_storage()

    first = seed_eval_definitions(storage, load_seed_corpus(directory, "scope-a"))
    again = seed_eval_definitions(storage, load_seed_corpus(directory, "scope-a"))
    other = seed_eval_definitions(storage, load_seed_corpus(directory, "scope-b"))

    assert first.created["eval_template"] == 1
    assert again.created["eval_template"] == 0 and again.skipped["eval_template"] == 1
    assert other.created["eval_template"] == 1, "a slot filled in one scope is empty in another"
    assert [t.name for t in storage.query_templates("scope-a")] == ["probe"]
    assert [t.name for t in storage.query_templates("scope-b")] == ["probe"]


def test_a_corpus_holding_another_scope_s_definition_is_refused() -> None:
    stray = make_template(scope_id="scope-b")

    with pytest.raises(ValueError, match=stray.id):
        SeedCorpus(scope_id="scope-a", templates=(make_template(scope_id="scope-a"), stray))


def test_a_corpus_of_its_own_scope_constructs() -> None:
    """The positive control for the refusal above, on the same shape."""
    corpus = SeedCorpus(scope_id="scope-a", templates=(make_template(scope_id="scope-a"),))

    assert len(corpus.templates) == 1
