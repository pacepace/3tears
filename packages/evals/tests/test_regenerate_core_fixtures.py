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


# --- which fixtures are kept: a fresh draw only, spelled as before -----------------------------------------------

_KEPT = {
    "id": "0199d3a2-6c1e-7a4b-9f00-1c2d3e4f5a6b",
    "created_at": "2026-10-13T05:04:01.438077+00:00",
    "label": "a fixed value",
    "tags": ["a", "b"],
}


def _sampled(**changes: object) -> dict[str, object]:
    return {**_KEPT, **changes}


def test_a_fresh_clock_reading_and_a_fresh_id_are_the_same_document() -> None:
    sampled = _sampled(id="0199d3a2-9999-7fff-8abc-000000000001", created_at="2026-10-14T23:59:59.000001+00:00")

    assert regenerate.same_document(_KEPT, sampled)


def test_a_timestamp_respelled_is_written() -> None:
    """The offset spelled ``Z``, or the fraction dropped: what the build writes changed, not what it drew."""
    assert not regenerate.same_document(_KEPT, _sampled(created_at="2026-10-13T05:04:01.438077Z"))
    assert not regenerate.same_document(_KEPT, _sampled(created_at="2026-10-13T05:04:01+00:00"))
    assert not regenerate.same_document(_KEPT, _sampled(created_at="2026-10-13T05:04:01.438+00:00"))


def test_a_uuid_from_another_generator_is_written() -> None:
    """uuid4 and uuid7 mask to the same hex; the version digit tells them apart."""
    assert not regenerate.same_document(_KEPT, _sampled(id="5f0e3b1c-2d4a-4b6c-8d9e-0f1a2b3c4d5e"))


def test_any_other_difference_is_written() -> None:
    assert not regenerate.same_document(_KEPT, _sampled(label="another fixed value"))
    assert not regenerate.same_document(_KEPT, _sampled(tags=["a"]))
    assert not regenerate.same_document(_KEPT, {key: value for key, value in _KEPT.items() if key != "label"})
    assert not regenerate.same_document(_KEPT, _sampled(id="2026-10-13T05:04:01.438077+00:00"))
