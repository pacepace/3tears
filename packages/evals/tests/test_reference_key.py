"""A case's reference key: its shape, the identity it is part of, and the case material a judge reads it as.

Pinned here:

- **The key refuses** a key with no entity, a name or alias two entities fold alike to (naming both), and any key it
  does not declare, at every level.
- **A case without a key is stored as before**, with no ``reference`` in its document; a case with one round-trips.
- **The key is identity**: a case without one digests and is seeded under the id it had before keys existed (pinned
  from ``a1cf245e9``); adding a key, or editing one alias, moves both; writing the same key's entities, aliases or
  facts in another order moves neither, since a key's parts are sets.
- **Drift and reset follow it**: a corpus case whose key changed reads ``stale``, its reset writes a new revision
  holding the new key under a new id, and a live case carried into that revision keeps its key.
- **The case material** a kind appends for its judges is one fixed text, pinned whole.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.kernel import ConflictError, EvalStorage, reference_material
from threetears.evals.run import (
    SeedCorpus,
    SeedTemplateCase,
    definition_drift,
    load_seed_corpus,
    reset_definitions,
    seed_eval_definitions,
    template_case_content_digest,
)
from threetears.evals.schema import EvalTemplate, EvalTestCase, ReferenceKey
from packages.evals.tests.clock_support import frozen_clock
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.run import TOY_EXTRACTOR_KIND, toyhost_template


@pytest.fixture(autouse=True)
def _frozen_clock() -> Iterator[None]:
    """Nothing here asserts on the time, so every stamp the code under test makes reads a frozen clock."""
    with frozen_clock():
        yield


SUBJECT = toyhost_template().name
CASE_KEY = f"template_case/{SUBJECT}/hurried-buyer"

#: What a case of the toy host's template that carries no key digests to, and the id the seed gives it, at
#: ``a1cf245e9``, the build before keys existed: a case without a key must keep both.
DIGEST_BEFORE_KEYS = "865c5fada1ca5ba247459b9a9407708e51b1e9963159b3f96920607689194084"
SEEDED_ID_BEFORE_KEYS = "490aee8c-6404-5664-b80e-e16dbe493048"


def _key(**changes: Any) -> dict[str, Any]:
    """A key as a corpus file states it: two core entities with aliases and facts, and one acceptable other."""
    key: dict[str, Any] = {
        "core": [
            {
                "name": "Northwind Trading",
                "aliases": ["Northwind", "NW Trading"],
                "facts": [
                    {"claim": "Founded in 1975.", "source_url": "https://example.org/northwind"},
                    {"claim": "Based in Seattle.", "source_url": "https://example.org/northwind/about"},
                ],
            },
            {"name": "Contoso", "aliases": ["Contoso Ltd"]},
        ],
        "acceptable": [{"name": "Fabrikam"}],
    }
    return {**key, **changes}


def _case(**changes: Any) -> SeedTemplateCase:
    fields: dict[str, Any] = {
        "template": SUBJECT,
        "name": "hurried-buyer",
        "variation_params": {"invoice": "INV-7"},
        "host_payload": {"persona": "a buyer in a hurry"},
    }
    return SeedTemplateCase(**{**fields, **changes})


def _corpus(*cases: SeedTemplateCase) -> SeedCorpus:
    return SeedCorpus(scope_id=TOYHOST_SCOPE, templates=(toyhost_template(),), template_cases=cases)


def _subject(storage: EvalStorage) -> EvalTemplate:
    template = storage.load_template_by_name(SUBJECT, TOYHOST_SCOPE)
    assert template is not None
    return template


def _cases_on(storage: EvalStorage, template: EvalTemplate) -> list[EvalTestCase]:
    return sorted(storage.query_test_cases(TOYHOST_SCOPE, template_id=template.id), key=lambda c: c.name or "")


def _states(storage: EvalStorage, corpus: SeedCorpus) -> dict[str, str]:
    drift = definition_drift(toyhost_host(storage=storage), corpus)
    return {reading.key: reading.state for reading in drift.readings}


# --- 1. what the key refuses -----------------------------------------------------------------------------------------


def test_a_key_naming_no_entity_is_refused() -> None:
    with pytest.raises(ValidationError, match="names at least one entity"):
        ReferenceKey()
    with pytest.raises(ValidationError, match="names at least one entity"):
        ReferenceKey.model_validate({"type": "entity_set", "core": [], "acceptable": []})


def test_a_key_with_only_acceptable_entities_or_only_core_is_a_key() -> None:
    assert [e.name for e in ReferenceKey.model_validate({"acceptable": [{"name": "Fabrikam"}]}).entities] == [
        "Fabrikam"
    ]
    assert [e.name for e in ReferenceKey.model_validate({"core": [{"name": "Contoso"}]}).entities] == ["Contoso"]


@pytest.mark.parametrize(
    ("core", "acceptable", "said"),
    [
        ([{"name": "Contoso"}, {"name": "CONTOSO"}], [], "'contoso' names both"),
        ([{"name": "Contoso"}], [{"name": "Fabrikam", "aliases": ["contoso"]}], "'contoso' names both"),
        ([{"name": "Contoso", "aliases": ["Big   Co"]}], [{"name": "Fabrikam", "aliases": ["big co"]}], "'big co'"),
        # NFKC folds the ligature into the two letters it stands for.
        ([{"name": "ﬁne goods"}], [{"name": "Fine Goods"}], "'fine goods' names both"),
    ],
)
def test_a_name_or_alias_two_entities_fold_alike_to_is_refused_naming_both(
    core: list[dict[str, Any]], acceptable: list[dict[str, Any]], said: str
) -> None:
    with pytest.raises(ValidationError) as refused:
        ReferenceKey.model_validate({"core": core, "acceptable": acceptable})

    message = str(refused.value)
    assert said in message, message
    names = [entity["name"] for entity in (*core, *acceptable)]
    assert all(repr(name) in message for name in names), f"both entities are named: {message}"


def test_an_entity_whose_alias_folds_to_its_own_name_is_one_entity() -> None:
    key = ReferenceKey.model_validate({"core": [{"name": "Contoso", "aliases": ["CONTOSO"]}]})

    assert [e.name for e in key.entities] == ["Contoso"]


@pytest.mark.parametrize(
    "document",
    [
        {**_key(), "exhaustive": True},
        _key(acceptable=[{"name": "Fabrikam", "id": "e1"}]),
        _key(acceptable=[{"name": "Fabrikam", "facts": [{"claim": "c", "source_url": "u", "checked": True}]}]),
    ],
    ids=["on the key", "on an entity", "on a fact"],
)
def test_a_key_it_does_not_declare_is_refused_at_every_level(document: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs are not permitted"):
        ReferenceKey.model_validate(document)


def test_a_blank_name_claim_or_source_and_another_type_are_refused() -> None:
    for document in (
        {"core": [{"name": ""}]},
        {"core": [{"name": "Contoso", "facts": [{"claim": "", "source_url": "u"}]}]},
        {"core": [{"name": "Contoso", "facts": [{"claim": "c", "source_url": ""}]}]},
        {"type": "ranked_list", "core": [{"name": "Contoso"}]},
    ):
        with pytest.raises(ValidationError):
            ReferenceKey.model_validate(document)


# --- 2. how a case stores it -----------------------------------------------------------------------------------------


def test_a_case_without_a_key_stores_no_reference_and_one_with_a_key_round_trips() -> None:
    template = toyhost_template()
    plain = _case().test_case(template)
    keyed = _case(reference=_key()).test_case(template)

    assert "reference" not in plain.to_dict() and plain.reference is None
    stored = keyed.to_dict()
    assert stored["reference"]["type"] == "entity_set"
    read = EvalTestCase.from_dict(stored)
    assert read == keyed and read.reference == ReferenceKey.model_validate(_key())
    assert EvalTestCase.from_dict(json.loads(json.dumps(stored))).to_dict() == stored, "a JSON store's trip too"


def test_a_corpus_file_carries_a_key_into_its_case(tmp_path: Path) -> None:
    (tmp_path / "templates").mkdir()
    (tmp_path / "template_cases").mkdir()
    (tmp_path / "templates" / "probe.json").write_text(
        json.dumps({"name": "probe", "intent": "i", "candidate_kind": TOY_EXTRACTOR_KIND}), encoding="utf-8"
    )
    (tmp_path / "template_cases" / "hurried-buyer.json").write_text(
        json.dumps({"template": "probe", "variation_params": {"invoice": "INV-7"}, "reference": _key()}),
        encoding="utf-8",
    )

    (case,) = load_seed_corpus(tmp_path, TOYHOST_SCOPE).template_cases

    assert case.reference == ReferenceKey.model_validate(_key())


# --- 3. identity -----------------------------------------------------------------------------------------------------


def test_a_case_without_a_key_keeps_the_digest_and_seeded_id_it_had_before_keys() -> None:
    case = _case()

    assert template_case_content_digest(case) == DIGEST_BEFORE_KEYS
    seeded = case.test_case(toyhost_template())
    assert (seeded.content_hash, seeded.id) == (DIGEST_BEFORE_KEYS, SEEDED_ID_BEFORE_KEYS)


def test_adding_a_key_or_editing_one_alias_moves_the_digest_and_the_id() -> None:
    template = toyhost_template()
    plain = _case().test_case(template)
    keyed = _case(reference=_key()).test_case(template)
    edited_key = _key()
    edited_key["core"][1]["aliases"] = ["Contoso Limited"]
    edited = _case(reference=edited_key).test_case(template)

    assert len({plain.content_hash, keyed.content_hash, edited.content_hash}) == 3
    assert len({plain.id, keyed.id, edited.id}) == 3


def test_writing_a_key_s_entities_aliases_or_facts_in_another_order_moves_neither() -> None:
    """A key's parts are sets: the key keeps them in one order, so another order is the same key and the same case."""
    template = toyhost_template()
    written = _key()
    reordered = _key(
        core=[
            {"name": "Contoso", "aliases": ["Contoso Ltd"]},
            {
                "name": "Northwind Trading",
                "aliases": ["NW Trading", "Northwind"],
                "facts": [
                    {"claim": "Based in Seattle.", "source_url": "https://example.org/northwind/about"},
                    {"claim": "Founded in 1975.", "source_url": "https://example.org/northwind"},
                ],
            },
        ]
    )
    assert [e["name"] for e in reordered["core"]] != [e["name"] for e in written["core"]], "the fixture reorders"

    first = _case(reference=written).test_case(template)
    second = _case(reference=reordered).test_case(template)

    assert (first.content_hash, first.id) == (second.content_hash, second.id)
    assert first.to_dict()["reference"] == second.to_dict()["reference"]
    assert first.reference is not None and [e.name for e in first.reference.core] == ["Contoso", "Northwind Trading"]


def test_a_host_writing_its_own_case_sets_the_content_hash_the_seed_gives_it() -> None:
    template = toyhost_template()
    seeded = _case(reference=_key()).test_case(template)
    written = EvalTestCase(
        scope_id=TOYHOST_SCOPE,
        template_id=template.id,
        variation_params={"invoice": "INV-7"},
        host_payload={"persona": "a buyer in a hurry"},
        reference=ReferenceKey.model_validate(_key()),
    )

    assert template_case_content_digest(written) == seeded.content_hash


# --- 4. drift and reset ----------------------------------------------------------------------------------------------


def test_a_corpus_case_whose_key_changed_reads_stale_and_its_reset_writes_a_new_revision_and_case() -> None:
    storage, _ = memory_storage()
    seed_eval_definitions(toyhost_host(storage=storage), _corpus(_case(reference=_key())))
    first = _subject(storage)
    (before,) = _cases_on(storage, first)
    moved_key = _key(acceptable=[{"name": "Fabrikam"}, {"name": "Litware"}])
    moved = _corpus(_case(reference=moved_key))

    assert _states(storage, _corpus(_case(reference=_key())))[CASE_KEY] == "current"
    assert _states(storage, moved)[CASE_KEY] == "stale"
    reset = reset_definitions(toyhost_host(storage=storage), moved, [CASE_KEY])

    assert set(_states(storage, moved).values()) == {"current"}, reset.after.render()
    second = _subject(storage)
    assert second.revision == first.revision + 1
    (after,) = _cases_on(storage, second)
    assert after.reference == ReferenceKey.model_validate(moved_key)
    assert after.id != before.id and after.content_hash != before.content_hash
    (kept,) = _cases_on(storage, first)
    assert kept == before, "the superseded revision keeps the case it ran, key and all"


def test_a_live_case_carried_into_the_next_revision_keeps_its_key() -> None:
    storage, _ = memory_storage()
    seed_eval_definitions(toyhost_host(storage=storage), _corpus(_case()))
    first = _subject(storage)
    operators = EvalTestCase(
        scope_id=TOYHOST_SCOPE,
        template_id=first.id,
        variation_params={"invoice": "INV-ops"},
        host_payload={"persona": "an operator's buyer"},
        reference=ReferenceKey.model_validate(_key()),
        name="ops",
    )
    storage.save_test_case(operators)
    grown = _corpus(_case(), _case(name="second", variation_params={"invoice": "INV-8"}))
    stale = [f"template_case/{SUBJECT}/second"]

    with pytest.raises(ConflictError, match="carry_live_cases"):
        reset_definitions(toyhost_host(storage=storage), grown, stale)
    reset_definitions(toyhost_host(storage=storage), grown, stale, carry_live_cases=True)

    (carried,) = [c for c in _cases_on(storage, _subject(storage)) if c.carried_from == operators.id]
    assert carried.reference == operators.reference


# --- 5. the case material --------------------------------------------------------------------------------------------

#: The whole of what a judge reads for :func:`_key`: the non-exhaustive line, the core, then the acceptable others,
#: each entity sorted by its folded name, its aliases and facts in their own one order.
GOLDEN_MATERIAL = """\
Reference key. It may not list every entity that counts: an entity the answer names outside it is unverified, not wrong.
Required (core):
  - Contoso (also called: Contoso Ltd)
  - Northwind Trading (also called: Northwind; NW Trading)
    - Based in Seattle. (source: https://example.org/northwind/about)
    - Founded in 1975. (source: https://example.org/northwind)
Also acceptable:
  - Fabrikam"""


def test_the_case_material_is_one_fixed_text() -> None:
    assert reference_material(ReferenceKey.model_validate(_key())) == GOLDEN_MATERIAL


def test_a_key_with_no_core_says_so_in_its_material() -> None:
    material = reference_material(ReferenceKey.model_validate({"acceptable": [{"name": "Fabrikam"}]}))

    assert material.splitlines()[1:] == ["Required (core): none.", "Also acceptable:", "  - Fabrikam"]
