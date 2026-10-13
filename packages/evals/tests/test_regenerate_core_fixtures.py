"""The core fixture regenerator's two decisions: whether it may write the current corpus at all, and which files.

``scripts/regenerate_core_fixtures.py`` is the one documented path that rewrites
``tests/fixtures/core_documents/v{CORE_SCHEMA_VERSION}/``, so the rules that keep it from rewriting the stored past
are read here by driving its functions directly, never by running it over the real corpus.
"""

from __future__ import annotations

from packages.evals.tests.script_loader import load_script

regenerate = load_script("regenerate_core_fixtures.py")


# --- a released corpus is never regenerated ----------------------------------------------------------------------


def test_a_corpus_a_tagged_tree_holds_is_refused_by_name() -> None:
    """Read against a tag whose tree holds ``v14/``: the refusal names the tag and the major that is the remedy."""
    refusal = regenerate.released_corpus_refusal(14, "v0.66.0-evaladopt.22")

    assert refusal is not None
    assert "v0.66.0-evaladopt.22 shipped" in refusal and "core_documents/v14/" in refusal
    assert "v15/" in refusal and "CORE_SCHEMA_VERSION to 15" in refusal


def test_a_release_that_shipped_no_corpus_lets_it_be_written() -> None:
    """v0.65.0 shipped no 3tears-evals, so no store it wrote holds a core document of any version."""
    assert regenerate.released_corpus_refusal(14, "v0.65.0") is None


def test_a_checkout_without_release_tags_refuses_rather_than_guessing() -> None:
    refusal = regenerate.released_corpus_refusal(14, None)

    assert refusal is not None and "git fetch --tags" in refusal
