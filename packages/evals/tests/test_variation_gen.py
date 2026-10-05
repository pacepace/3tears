"""Unit tests for the eval variation generator."""

from __future__ import annotations

import random

import pytest

from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import ActorPolicy, ConversationSpec, EvalTemplate, EvalTestCase, VariationAxis
from threetears.evals.gen.variation_gen import generate_variations

# =============================================================================
# Fake storage — captures saves, supplies queries
# =============================================================================


# parity-with: threetears.evals.gen.variation_gen.EvalTestCaseStore
class _FakeStorage:
    def __init__(self, existing: list[EvalTestCase] | None = None):
        self.test_cases: list[EvalTestCase] = list(existing or [])
        self.saved: list[EvalTestCase] = []

    def query_test_cases(self, scope_id: str, /, *, template_id: str | None = None) -> list[EvalTestCase]:
        return [
            tc
            for tc in self.test_cases
            if tc.scope_id == scope_id and (template_id is None or tc.template_id == template_id)
        ]

    def save_test_case(self, test_case: EvalTestCase, /) -> None:
        self.saved.append(test_case)
        self.test_cases.append(test_case)


# parity-with: threetears.evals.contracts.provider.VariationLLM
class _FakeLLM:
    """Returns a sequence of canned responses; records each call's args.

    ``generate`` accepts ``response_format`` to match the real generator client
    (the axis generator routes through json_object mode); the value is
    recorded so a test can assert the caller opted in.
    """

    model_name = "writer-model"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.response_formats: list = []

    async def generate(self, *, system: str, user: str, response_format=None):
        self.calls.append((system, user))
        self.response_formats.append(response_format)
        if not self._responses:
            raise AssertionError("FakeLLM exhausted")

        class _R:
            content = self._responses.pop(0)

        return _R()

    def __getattr__(self, name):
        raise AttributeError(name)


def _template(*axes, name="t", scope_id="uni") -> EvalTemplate:
    return EvalTemplate(
        scope_id="uni-1",
        name=name,
        intent="test",
        variation_axes=list(axes),
        conversation=ConversationSpec(actors=[ActorPolicy(id="x", policy="p", intent="i")]),
        candidate_kind="test-kind",
    )


# =============================================================================
# enum generator
# =============================================================================


async def test_enum_axis_yields_every_value_once_within_cap():
    template = _template(VariationAxis(name="tone", generator="enum", values=["casual", "cocky", "uncertain"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, n_variations=10, storage=storage, scope_id="uni")).cases
    assert len(out) == 3
    tones = sorted(tc.variation_params["tone"] for tc in out)
    assert tones == ["casual", "cocky", "uncertain"]


async def test_enum_axis_persists_each_unique_test_case():
    template = _template(VariationAxis(name="tone", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases
    assert len(out) == 2
    assert len(storage.saved) == 2


async def test_existing_test_case_reused_not_duplicated():
    """A second call with the same template+scope returns the same test cases."""
    template = _template(VariationAxis(name="tone", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    out1 = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases
    assert len(storage.saved) == 2

    out2 = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases
    assert len(storage.saved) == 2  # nothing new saved
    # Same test case objects returned (same id).
    out1_ids = {tc.id for tc in out1}
    out2_ids = {tc.id for tc in out2}
    assert out1_ids == out2_ids


# =============================================================================
# sample generator
# =============================================================================


async def test_sample_axis_chooses_n_distinct_values():
    template = _template(VariationAxis(name="x", generator="sample", values=["a", "b", "c", "d", "e"]))
    storage = _FakeStorage()
    rng = random.Random(42)
    out = (await generate_variations(template, n_variations=3, storage=storage, scope_id="u", rng=rng)).cases
    values = [tc.variation_params["x"] for tc in out]
    assert len(values) == 3
    assert len(set(values)) == 3
    assert all(v in {"a", "b", "c", "d", "e"} for v in values)


async def test_sample_axis_handles_n_larger_than_values():
    template = _template(VariationAxis(name="x", generator="sample", values=["a", "b"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, 10, storage=storage, scope_id="u")).cases
    # Only 2 distinct values available.
    assert len(out) == 2


async def test_an_axis_with_no_values_refuses_rather_than_generating_nothing():
    template = _template(VariationAxis(name="x", generator="sample", values=[]))
    storage = _FakeStorage()
    with pytest.raises(ValidationFailedError, match="'x'"):
        await generate_variations(template, 5, storage=storage, scope_id="u")
    assert storage.saved == []


async def test_an_empty_axis_does_not_fall_back_to_the_templates_stored_cases():
    """The fallback froze a case set nobody asked for under a run that said it generated one."""
    template = _template(VariationAxis(name="x", generator="sample", values=[]))
    storage = _FakeStorage()
    storage.saved.append(EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "old"}))
    with pytest.raises(ValidationFailedError):
        await generate_variations(template, 5, storage=storage, scope_id="u")


# =============================================================================
# llm generator
# =============================================================================


async def test_llm_axis_parses_json_object_response():
    template = _template(VariationAxis(name="category_pair", generator="llm", description="Category pairings"))
    storage = _FakeStorage()
    llm = _FakeLLM(['{"values": ["kitchen + garden", "toys + stationery", "tools + linen"]}'])
    out = (await generate_variations(template, 3, storage=storage, scope_id="u", llm=llm)).cases
    pairs = [tc.variation_params["category_pair"] for tc in out]
    assert sorted(pairs) == ["kitchen + garden", "tools + linen", "toys + stationery"]


async def test_llm_axis_uses_json_object_mode():
    """The axis generator opts into json_object structured output like its siblings."""
    from threetears.evals.contracts.provider import JSON_OBJECT_RESPONSE_FORMAT

    template = _template(VariationAxis(name="x", generator="llm"))
    storage = _FakeStorage()
    llm = _FakeLLM(['{"values": ["a"]}'])
    await generate_variations(template, 1, storage=storage, scope_id="u", llm=llm)
    assert llm.response_formats[0] == JSON_OBJECT_RESPONSE_FORMAT


async def test_llm_axis_strips_code_fences_and_commentary():
    template = _template(VariationAxis(name="x", generator="llm"))
    storage = _FakeStorage()
    llm = _FakeLLM(['Sure! Here are three: ```json\n{"values": ["a", "b", "c"]}\n```'])
    out = (await generate_variations(template, 3, storage=storage, scope_id="u", llm=llm)).cases
    assert sorted(tc.variation_params["x"] for tc in out) == ["a", "b", "c"]


async def test_llm_axis_dedupes_against_existing():
    """LLM-generated values that match existing test cases are excluded from output.

    Existing test cases are *reused* only when the new generation re-discovers
    their variation_params. LLM axes strip existing values from the response,
    so the existing test case isn't re-produced — its row remains in storage
    but isn't returned by this call. This matches the specified behaviour: "generate
    that many new EvalTestCases for this template (deduplicating against
    existing test cases; persisting each unique one)".
    """
    template = _template(VariationAxis(name="x", generator="llm"))
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "stale"})
    storage = _FakeStorage(existing=[existing])
    # LLM returns one stale + two novel
    llm = _FakeLLM(['{"values": ["stale", "novel1", "novel2"]}'])
    out = (await generate_variations(template, 2, storage=storage, scope_id="u", llm=llm)).cases
    new_values = sorted(tc.variation_params["x"] for tc in out)
    # The stale value is filtered out of the LLM response; only the novel
    # two appear in this generation's output.
    assert new_values == ["novel1", "novel2"]
    # Both novel ones were freshly persisted; the existing 'stale' row is
    # untouched (still in storage from before).
    saved_values = sorted(tc.variation_params["x"] for tc in storage.saved)
    assert saved_values == ["novel1", "novel2"]


async def test_enum_axis_with_existing_reuses_matching_test_cases():
    """When an enum axis produces a combo matching an existing test case, the existing case is returned (not re-saved)."""
    template = _template(VariationAxis(name="x", generator="enum", values=["a", "b"]))
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "a"})
    storage = _FakeStorage(existing=[existing])
    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases
    by_x = {tc.variation_params["x"]: tc for tc in out}
    # 'a' is the existing test case (reused, same id).
    assert by_x["a"].id == existing.id
    # 'b' is freshly persisted.
    assert len(storage.saved) == 1
    assert storage.saved[0].variation_params == {"x": "b"}


async def test_llm_axis_dedupes_passed_to_prompt():
    """The existing values block is rendered into the prompt for the LLM to see."""
    template = _template(VariationAxis(name="x", generator="llm"))
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "Stale Category"})
    storage = _FakeStorage(existing=[existing])
    llm = _FakeLLM(['{"values": ["novel"]}'])
    await generate_variations(template, 1, storage=storage, scope_id="u", llm=llm)

    _system, user = llm.calls[0]
    assert "Stale Category" in user


async def test_an_llm_axis_whose_response_is_unparseable_refuses():
    template = _template(VariationAxis(name="x", generator="llm"))
    storage = _FakeStorage()
    llm = _FakeLLM(["I'm sorry but I cannot help with that request."])
    with pytest.raises(ValidationFailedError, match="produced no values"):
        await generate_variations(template, 3, storage=storage, scope_id="u", llm=llm)


async def test_an_llm_axis_whose_object_has_no_values_key_refuses():
    """A parseable JSON object that carries no 'values' array refuses by name, not a crash."""
    template = _template(VariationAxis(name="x", generator="llm"))
    storage = _FakeStorage()
    llm = _FakeLLM(['{"result": ["a", "b"]}'])  # valid object, wrong key
    with pytest.raises(ValidationFailedError, match="produced no values"):
        await generate_variations(template, 3, storage=storage, scope_id="u", llm=llm)


async def test_llm_axis_requires_llm_client():
    template = _template(VariationAxis(name="x", generator="llm"))
    storage = _FakeStorage()
    with pytest.raises(ValueError, match="llm client"):
        await generate_variations(template, 1, storage=storage, scope_id="u", llm=None)


# =============================================================================
# Multi-axis Cartesian product
# =============================================================================


async def test_multi_axis_combines_via_cartesian_product():
    template = _template(
        VariationAxis(name="tone", generator="enum", values=["casual", "cocky"]),
        VariationAxis(name="depth", generator="enum", values=["short", "long"]),
    )
    storage = _FakeStorage()
    out = (await generate_variations(template, n_variations=10, storage=storage, scope_id="u")).cases
    assert len(out) == 4
    combos = sorted((tc.variation_params["tone"], tc.variation_params["depth"]) for tc in out)
    assert combos == [
        ("casual", "long"),
        ("casual", "short"),
        ("cocky", "long"),
        ("cocky", "short"),
    ]


async def test_multi_axis_truncates_when_product_exceeds_n_variations():
    template = _template(
        VariationAxis(name="a", generator="enum", values=["1", "2", "3"]),
        VariationAxis(name="b", generator="enum", values=["x", "y", "z"]),
    )
    storage = _FakeStorage()
    rng = random.Random(0)
    out = (await generate_variations(template, n_variations=4, storage=storage, scope_id="u", rng=rng)).cases
    assert len(out) == 4
    # All 4 combos must be distinct.
    combos = {(tc.variation_params["a"], tc.variation_params["b"]) for tc in out}
    assert len(combos) == 4


# =============================================================================
# Preview mode
# =============================================================================


async def test_preview_mode_does_not_persist():
    template = _template(VariationAxis(name="x", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, 5, storage=storage, scope_id="u", preview=True)).cases
    assert len(out) == 2
    assert storage.saved == []


async def test_preview_mode_does_not_dedup_against_storage():
    template = _template(VariationAxis(name="x", generator="enum", values=["a"]))
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "a"})
    storage = _FakeStorage(existing=[existing])
    out = (await generate_variations(template, 1, storage=storage, scope_id="u", preview=True)).cases
    # Preview generates a fresh test case (different id from existing).
    assert len(out) == 1
    assert out[0].id != existing.id


# =============================================================================
# Edge cases
# =============================================================================


async def test_n_variations_zero_raises():
    template = _template(VariationAxis(name="x", generator="enum", values=["a"]))
    with pytest.raises(ValueError, match="positive"):
        await generate_variations(template, 0, storage=_FakeStorage(), scope_id="u")


async def test_the_dedup_key_is_stable_under_key_order():
    """A stored case written in another key order is the same case, so it is reused, not re-minted."""
    template = _template(
        VariationAxis(name="x", generator="enum", values=["1"]),
        VariationAxis(name="y", generator="enum", values=["2"]),
    )
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"y": "2", "x": "1"})
    storage = _FakeStorage(existing=[existing])

    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases

    assert [tc.id for tc in out] == [existing.id]
    assert storage.saved == []


async def test_the_dedup_key_distinguishes_different_values():
    """A stored case with a different value blocks nothing: the new combination is persisted."""
    template = _template(VariationAxis(name="x", generator="enum", values=["2"]))
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params={"x": "1"})
    storage = _FakeStorage(existing=[existing])

    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases

    assert [tc.variation_params for tc in out] == [{"x": "2"}]
    assert [tc.variation_params for tc in storage.saved] == [{"x": "2"}]


# =============================================================================
# Generated-artifact identity
# =============================================================================


async def test_generated_cases_are_content_hashed():
    """A generated artifact carries a pinned identity, or 'the same cell' can drift.

    Without it, a regenerated variation set silently becomes different inputs
    wearing the old test-case ids, and every downstream comparison is confounded.
    """
    from threetears.evals.contracts.identity import compute_content_hash

    template = _template(VariationAxis(name="tone", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases

    assert len(out) == 2
    for tc in out:
        assert tc.content_hash == compute_content_hash(tc.variation_params)
        assert tc.content_hash is not None


async def test_distinct_variations_get_distinct_content_hashes():
    template = _template(VariationAxis(name="tone", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases
    assert len({tc.content_hash for tc in out}) == 2


async def test_dedup_and_content_identity_agree():
    """Two cases the generator considers duplicates must hash identically.

    These are two independent notions of 'same generated artifact' — the
    generation-time dedup key and the persisted content hash. If they ever
    disagreed, generation would either mint a second case that hashes the same
    as the first, or reuse a case whose content hash says it is different.
    """
    from threetears.evals.contracts.identity import compute_content_hash

    params_a = {"tone": "casual", "category": "garden"}
    params_b = {"category": "garden", "tone": "casual"}
    template = _template(
        VariationAxis(name="tone", generator="enum", values=["casual"]),
        VariationAxis(name="category", generator="enum", values=["garden"]),
    )
    existing = EvalTestCase(template_id=template.id, scope_id="u", variation_params=params_b)
    storage = _FakeStorage(existing=[existing])

    out = (await generate_variations(template, 5, storage=storage, scope_id="u")).cases

    # The generator treats the stored case as a duplicate of what it generated...
    assert [tc.id for tc in out] == [existing.id] and storage.saved == []
    # ...and the persisted content hash says the same about the two spellings.
    assert compute_content_hash(params_a) == compute_content_hash(params_b)


async def test_a_case_with_no_generated_content_carries_no_content_hash():
    """A template with no variation axes generates nothing to pin.

    ``None`` here means 'no generated content', which is why an empty params
    dict must not be given the digest of ``{}`` — that would assert a
    provenance the case does not have.
    """
    from threetears.evals.contracts.identity import compute_content_hash

    assert compute_content_hash({}) is None
    assert EvalTestCase(scope_id="u", template_id="t").content_hash is None


# =============================================================================
# The counts a launch records
# =============================================================================


async def test_a_short_generation_says_it_ran_short():
    """Two enum values cannot make five cases; the counts say so rather than the run looking whole."""
    template = _template(VariationAxis(name="x", generator="enum", values=["a", "b"]))
    counts = (await generate_variations(template, 5, storage=_FakeStorage(), scope_id="u")).counts
    assert (counts.requested, counts.kept, counts.reused) == (5, 2, 0)
    assert counts.short


async def test_a_full_generation_is_not_short_and_counts_what_it_reused():
    template = _template(VariationAxis(name="x", generator="enum", values=["a", "b"]))
    storage = _FakeStorage()
    await generate_variations(template, 2, storage=storage, scope_id="u")
    counts = (await generate_variations(template, 2, storage=storage, scope_id="u")).counts
    assert (counts.requested, counts.kept, counts.reused) == (2, 2, 2)
    assert not counts.short


async def test_the_counts_name_the_model_that_wrote_the_llm_axis():
    """Read off the client that made the calls, so the run cannot record a writer other than the one called."""
    template = _template(VariationAxis(name="x", generator="llm"))
    counts = (
        await generate_variations(
            template, 1, storage=_FakeStorage(), scope_id="u", llm=_FakeLLM(['{"values": ["a"]}'])
        )
    ).counts
    assert counts.variation_model == "writer-model"


async def test_the_counts_name_no_writer_when_no_axis_is_llm():
    template = _template(VariationAxis(name="x", generator="enum", values=["a"]))
    counts = (await generate_variations(template, 1, storage=_FakeStorage(), scope_id="u")).counts
    assert counts.variation_model is None


async def test_a_client_for_a_template_no_model_writes_is_refused():
    template = _template(VariationAxis(name="x", generator="enum", values=["a"]))
    with pytest.raises(ValueError, match="no llm-generated axis"):
        await generate_variations(template, 1, storage=_FakeStorage(), scope_id="u", llm=_FakeLLM([]))
