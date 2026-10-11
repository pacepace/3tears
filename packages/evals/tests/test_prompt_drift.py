"""A host can see that a stored copy of a package prompt predates the package's own, before it pays for it.

Each catalogue prompt is store-master: a host seeds an empty slot from the shipped default once, and never
overwrites a stored one. So a stored copy seeded before the package changed its default (``RubricDim.scale`` becoming
required, say) keeps being sent, and every paid call it steers can fail. ``seed_digest`` names a default's content;
a host records it when it seeds, and ``prompt_drift`` reads a stored copy against the shipped default:

- **current** — the stored copy is the shipped default;
- **stale** — it is a default the package has moved past: untouched since it was seeded, or with no record of its
  seeding at all, and not the shipped one. It names the remedy;
- **customized** — an operator changed it since it was seeded, or the host says it is deliberately its own. Never
  reported as drift, so a host that tunes a prompt is not nagged.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from threetears.evals.kernel import SEED_PROMPTS, PromptDrift, SeedTemplate, prompt_drift, seed_digest

_ENTRY = next(entry for entry in SEED_PROMPTS if entry.kind == "template")


def _shipped() -> SeedTemplate:
    module, constant = _ENTRY.seed_ref
    seed = getattr(importlib.import_module(module), constant)
    assert isinstance(seed, SeedTemplate)
    return seed


def _edited(template: SeedTemplate, text: str) -> SeedTemplate:
    first, *rest = template.sections
    return dataclasses.replace(template, sections=(dataclasses.replace(first, content_template=text), *rest))


@pytest.mark.parametrize("entry", SEED_PROMPTS, ids=lambda entry: entry.prompt_type)
def test_every_catalogue_prompt_has_a_digest_of_its_shipped_default(entry) -> None:
    module, constant = entry.seed_ref
    seed = getattr(importlib.import_module(module), constant)
    digest = seed_digest(seed)
    assert len(digest) == 64 and digest == seed_digest(seed)


def test_a_digest_names_content_and_not_the_revision_number() -> None:
    shipped = _shipped()
    assert seed_digest(dataclasses.replace(shipped, version=shipped.version + 1)) == seed_digest(shipped)
    assert seed_digest(_edited(shipped, "another rule")) != seed_digest(shipped)
    assert seed_digest(SeedTemplate.from_dict(shipped.to_dict())) == seed_digest(shipped), (
        "a stored payload reads alike"
    )


def test_a_stored_copy_of_the_shipped_default_is_current() -> None:
    drift = prompt_drift(_ENTRY, shipped=_shipped(), stored=_shipped(), seeded_digest=seed_digest(_shipped()))
    assert isinstance(drift, PromptDrift) and drift.state == "current" and not drift.stale


def test_a_copy_seeded_from_an_older_default_and_untouched_is_stale_and_names_the_remedy() -> None:
    older = _edited(_shipped(), "the rule before scale was required")
    drift = prompt_drift(_ENTRY, shipped=_shipped(), stored=older, seeded_digest=seed_digest(older))

    assert drift.state == "stale" and drift.stale
    assert drift.prompt_type == _ENTRY.prompt_type
    assert (drift.shipped_digest, drift.stored_digest) == (seed_digest(_shipped()), seed_digest(older))
    assert "reseed" in drift.remedy and "customized" in drift.remedy


def test_a_copy_with_no_record_of_its_seeding_that_is_not_the_shipped_default_reads_stale() -> None:
    """Without a record, an old default and a tuned copy look alike; the host says which with ``customized``."""
    drift = prompt_drift(_ENTRY, shipped=_shipped(), stored=_edited(_shipped(), "x"), seeded_digest=None)
    assert drift.state == "stale"


def test_a_copy_an_operator_changed_since_it_was_seeded_is_customized_not_stale() -> None:
    older = _edited(_shipped(), "the old default")
    tuned = _edited(_shipped(), "the operator's own wording")
    drift = prompt_drift(_ENTRY, shipped=_shipped(), stored=tuned, seeded_digest=seed_digest(older))
    assert drift.state == "customized" and not drift.stale


def test_a_copy_the_host_declares_its_own_is_never_stale() -> None:
    drift = prompt_drift(
        _ENTRY, shipped=_shipped(), stored=_edited(_shipped(), "ours"), seeded_digest=None, customized=True
    )
    assert drift.state == "customized" and not drift.stale


def test_a_text_prompt_is_read_the_same_way() -> None:
    entry = next(entry for entry in SEED_PROMPTS if entry.kind == "text")
    assert prompt_drift(entry, shipped="new text", stored="new text", seeded_digest=None).state == "current"
    assert prompt_drift(entry, shipped="new text", stored="old text", seeded_digest=seed_digest("old text")).stale


def test_a_stored_copy_of_another_kind_than_its_entry_is_refused() -> None:
    with pytest.raises(TypeError, match="template"):
        prompt_drift(_ENTRY, shipped=_shipped(), stored="text", seeded_digest=None)
