"""A finding's title names only arms its readings or trials are taken at (#829).

The instance, from a production memo (2026-10): a finding titled "GPT-6 Luna Pro and Ling 3.1 Flash quadrupled
evaluation cost" whose readings were taken at two other arms only, so the reader was shown no reading of Ling's cost.
The generator reads which arms a title names off the names the bundle gives each arm
(:mod:`threetears.evals.analysis.arm_mentions`) and refuses a title naming one its finding cites nothing of, naming
the arm, through the one repair round.

The matcher tests pin both directions: the names a writer plausibly writes are found, and a name that is a piece of
another arm's ("luna" inside "luna-pro"), a neighbouring version, a shared level, a bare word or a bare number name
nothing. The generation tests drive :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy
campaign with its two arms' level displays renamed to model ids, which leaves every key and cell as assembled.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    FixturedCompletion,
    alias_at,
    cell_at,
    memo_payload,
)
from threetears.evals.analysis.arm_mentions import arm_mentions, arm_vocabulary
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.kernel.campaign import EvalAnalysis, VariantIndexEntry
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema import SweepableValue
from threetears.evals.schema.models import utc_now_iso

#: The production title, verbatim.
_TITLE_829 = "GPT-6 Luna Pro and Ling 3.1 Flash quadrupled evaluation cost"

# --- the matcher --------------------------------------------------------------------------------------------------


def _entry(**levers: Any) -> VariantIndexEntry:
    named = {name: SweepableValue.of(value, display=str(value)) for name, value in levers.items()}
    return VariantIndexEntry(variant_key=compute_variant_key(named), levers=named)


#: Three arms on a model sweep, as the production campaign's: two whose names share their first words, and a third.
_LUNA, _LUNA_PRO, _LING = _ARMS = [
    _entry(candidate_kind="assistant", model=model, effort="high")
    for model in ("acme/gpt-6-luna", "acme/gpt-6-luna-pro", "other/ling-3.1-flash")
]


def _named(title: str, entries: list[VariantIndexEntry] = _ARMS) -> list[tuple[str, frozenset[str]]]:
    return [(mention.written, mention.name.variants) for mention in arm_mentions(title, arm_vocabulary(entries))]


def _arm(entry: VariantIndexEntry) -> frozenset[str]:
    return frozenset({entry.variant_key})


def test_the_production_title_names_luna_pro_and_ling_and_never_also_luna() -> None:
    assert _named(_TITLE_829) == [("GPT-6 Luna Pro", _arm(_LUNA_PRO)), ("Ling 3.1 Flash", _arm(_LING))]


def test_a_name_that_is_the_head_of_another_names_its_own_arm_where_written_alone() -> None:
    assert _named("GPT-6 Luna is cheaper than GPT-6 Luna-Pro.") == [
        ("GPT-6 Luna", _arm(_LUNA)),
        ("GPT-6 Luna-Pro", _arm(_LUNA_PRO)),
    ]


@pytest.mark.parametrize("written", ["Ling 3.1 Flash", "ling-3.1-flash", "LING 3.1 FLASH", "ling_3.1_flash"])
def test_case_and_separators_do_not_hide_a_name(written: str) -> None:
    assert _named(f"{written} is slower") == [(written, _arm(_LING))]


def test_a_provider_qualified_id_is_named_whole_and_by_its_own_name() -> None:
    assert _named("other/ling-3.1-flash is slower") == [("other/ling-3.1-flash", _arm(_LING))]
    assert _named("Ling 3.1 Flash is slower") == [("Ling 3.1 Flash", _arm(_LING))]


@pytest.mark.parametrize(
    "title",
    [
        "Ling 3.11 Flash is slower",  # a neighbouring version
        "GPT-6.1 Luna is cheaper",  # a neighbouring version, glued
        "A gpt-6-luna-pro-based pipeline",  # a name glued to more name
        "Luna Pro is dearer",  # a shortened name: missed, not guessed
        "High effort costs more",  # a level every arm shares
    ],
)
def test_what_is_not_an_arm_s_name_names_no_arm(title: str) -> None:
    assert _named(title) == []


def test_a_bare_word_or_a_bare_number_off_the_model_axis_names_no_arm() -> None:
    arms = [
        _entry(model="acme/gpt-6", effort=effort, budget=budget) for effort, budget in (("high", 16384), ("low", 12288))
    ]
    assert _named("High effort at 16384 tokens costs more than low effort at 12288", arms) == []


def test_a_level_with_a_letter_and_a_digit_off_the_model_axis_is_a_name() -> None:
    terse, plain = arms = [_entry(model="acme/gpt-6", preset=preset) for preset in ("terse-v2", "plain-v1")]
    assert _named("Terse v2 answers faster", arms) == [("Terse v2", _arm(terse))]


def test_a_model_named_by_one_bare_word_is_still_a_name() -> None:
    luna, ling = arms = [_entry(model=model) for model in ("luna", "ling")]
    assert _named("Luna costs more than Ling", arms) == [("Luna", _arm(luna)), ("Ling", _arm(ling))]


#: ASCII hyphen, U+2010 hyphen, U+2011 non-breaking hyphen, U+2013 en dash, U+2014 em dash.
_DASHES = ["-", "\u2010", "\u2011", "\u2013", "\u2014"]


@pytest.mark.parametrize("dash", _DASHES, ids=["hyphen-minus", "u2010", "u2011", "en-dash", "em-dash"])
def test_a_name_glued_to_more_name_by_any_dash_is_not_a_match(dash: str) -> None:
    gpt5, o3 = arms = [_entry(model=model) for model in ("acme/gpt-5", "acme/o3")]
    assert _named(f"GPT{dash}5{dash}mini is cheaper", arms) == []
    assert _named(f"O3{dash}mini is slower", arms) == []
    assert _named(f"GPT{dash}5 is cheaper than O3", arms) == [(f"GPT{dash}5", _arm(gpt5)), ("O3", _arm(o3))]


def test_a_name_keeps_its_own_spelling_where_case_folding_would_change_it() -> None:
    strasse, plain = arms = [_entry(model="acme/gpt-6", preset=preset) for preset in ("straße-2", "plain-v1")]
    assert _named("Straße-2 answers faster", arms) == [("Straße-2", _arm(strasse))]
    assert _named("STRASSE-2 answers faster", arms) == []


def test_a_quantity_with_a_short_unit_off_the_model_axis_names_no_arm() -> None:
    arms = [
        _entry(model="acme/gpt-6", boost=boost, window=window, cache=cache, wait=wait)
        for boost, window, cache, wait in (("4x", "8k", "2.5 GB", "100ms"), ("2x", "16k", "10 GB", "250ms"))
    ]
    assert _named("A 4x rise in cost under 8k tokens, with 2.5 GB of cache and 100ms waits", arms) == []
    assert _named("2x cuts cost at 16k, 10 GB and 250ms", arms) == []


def test_a_quantity_with_a_longer_unit_names_its_arm_only_written_as_one_token() -> None:
    wide, _narrow = arms = [
        _entry(model="acme/gpt-6", chunk=chunk, top_k=top_k) for chunk, top_k in (("1024tok", "8chunks"), ("256tok", "4chunks"))
    ]
    assert _named("The 1024tok chunk is the slower one", arms) == [("1024tok", _arm(wide))]
    assert _named("A 1024 tok chunk is slower", arms) == []
    assert _named("Retrieving 8 chunks costs more", arms) == []


@pytest.mark.parametrize("level", ["o3", "gpt-5", "claude-sonnet-4", "fanout-2", "v2", "x4", "8k-ctx"])
def test_a_name_with_a_letter_and_a_digit_off_the_model_axis_is_still_a_name(level: str) -> None:
    named, plain = arms = [_entry(model="acme/gpt-6", preset=preset) for preset in (level, "plain-v1")]
    assert _named(f"{level} answers faster", arms) == [(level, _arm(named))]


def test_an_arm_whose_levels_are_unavailable_offers_no_name() -> None:
    lost = VariantIndexEntry(variant_key="f" * 64, levers={}, levels_unavailable="keyed by an older predicate")
    assert _named("GPT-6 Luna costs more", [lost, _LING]) == []


# --- through the generator ----------------------------------------------------------------------------------------


class _MemoSequence(FixturedClient):
    """Hands back one prepared memo per call, in order: the first output, then its repair."""

    def __init__(self, *bodies: str) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        self.completion = FixturedCompletion(content=self.bodies[len(self.calls)])
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


#: What the toy arms are renamed to: the narrow width becomes the model whose name heads the other's.
_NARROW_NAME, _WIDE_NAME = "acme/gpt-6-luna", "acme/gpt-6-luna-pro"


def _bundle() -> AnalysisContextBundle:
    """The toy bundle, its narrow arm's level displayed as :data:`_NARROW_NAME` and its wide one's as :data:`_WIDE_NAME`.

    A level's display is not part of its identity, so every key, cell and alias stays as assembled.
    """
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    displays = {
        variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)): _NARROW_NAME,
        variant_of_cell_ref(cell_at(bundle, TOYHOST_WIDE)): _WIDE_NAME,
    }
    index = [
        entry.model_copy(
            update={
                "levers": entry.levers
                | {TOYHOST_AXIS: entry.levers[TOYHOST_AXIS].model_copy(update={"display": displays[entry.variant_key]})}
            }
        )
        for entry in bundle.variant_index
    ]
    return bundle.model_copy(update={"variant_index": index})


def _memo(bundle: AnalysisContextBundle, title: str, *, at: tuple[int, ...], trials: tuple[str, ...] = ()) -> str:
    """The toy memo plus a cost finding titled ``title``, its readings at the widths ``at`` and naming ``trials``."""
    payload = memo_payload(bundle)
    finding = payload["findings"][0]
    cost = finding | {
        "title": title,
        "evidence": [
            {"cell": alias_at(bundle, level), "measure_id": "production_replicating_cost", "reading": "measure"}
            for level in at
        ],
        "trials": list(trials),
        "confidence": "medium",
    }
    payload["findings"] = [finding, cost]
    return json.dumps(payload)


async def _generate(bundle: AnalysisContextBundle, client: FixturedClient) -> EvalAnalysis:
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


async def _accepted(bundle: AnalysisContextBundle, memo: str) -> EvalAnalysis:
    client = FixturedClient(memo)
    analysis = await _generate(bundle, client)
    assert len(client.calls) == 1, "accepted first time"
    return analysis


class TestATitleNamingAnArmItsFindingCitesNothingOfIsRefused:
    async def test_it_is_refused_naming_the_arm_as_written_its_levels_and_its_cell(self) -> None:
        bundle = _bundle()
        memo = _memo(bundle, "GPT-6 Luna Pro costs four times as much per document", at=(TOYHOST_NARROW,))
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, FixturedClient(memo))
        message = str(refused.value)
        assert 'findings[1].title names "GPT-6 Luna Pro"' in message
        assert f"{TOYHOST_AXIS}={_WIDE_NAME} at cell(s) {cell_at(bundle, TOYHOST_WIDE)}" in message
        assert "none of that finding's evidence rows or trials is taken at that arm" in message
        assert "A title claims no more than its evidence (rule 9)" in message
        assert f"{TOYHOST_AXIS}={_NARROW_NAME} " not in message, (
            "the cited arm, whose name heads the other's, is not named"
        )

    async def test_the_repair_is_told_the_arm_s_cell_by_its_alias_and_the_corrected_memo_is_stored(self) -> None:
        bundle = _bundle()
        title = "GPT-6 Luna Pro costs four times as much per document"
        client = _MemoSequence(
            _memo(bundle, title, at=(TOYHOST_NARROW,)), _memo(bundle, title, at=(TOYHOST_NARROW, TOYHOST_WIDE))
        )

        analysis = await _generate(bundle, client)

        refusal = analysis.generation.repaired_refusal
        assert refusal is not None
        assert f"at cell(s) {alias_at(bundle, TOYHOST_WIDE)})" in refusal, "named in the writer's own alias"
        assert len(client.calls) == 2 and refusal in client.calls[1]["user"]
        assert analysis.document.findings[1].title == title


class TestATitleBreakingBothTitleChecksIsToldOfBothInItsOneRepair:
    async def test_the_one_refusal_names_the_null_and_the_arm_and_the_repaired_memo_is_stored(self) -> None:
        bundle = _bundle()
        both = "GPT-6 Luna Pro does not change cost per document"
        fixed = "GPT-6 Luna is the cheaper arm"
        client = _MemoSequence(_memo(bundle, both, at=(TOYHOST_NARROW,)), _memo(bundle, fixed, at=(TOYHOST_NARROW,)))

        analysis = await _generate(bundle, client)

        refusal = analysis.generation.repaired_refusal
        assert refusal is not None
        assert 'findings[1].title says "' in refusal and "states that arms are alike" in refusal
        assert 'findings[1].title names "GPT-6 Luna Pro"' in refusal
        assert "drop that arm from the title" in refusal
        assert len(client.calls) == 2 and refusal in client.calls[1]["user"]
        assert analysis.document.findings[1].title == fixed


class TestATitleNamingOnlyCitedArmsIsAccepted:
    async def test_a_name_heading_another_arm_s_names_only_its_own(self) -> None:
        bundle = _bundle()
        await _accepted(bundle, _memo(bundle, "GPT-6 Luna Pro costs four times as much", at=(TOYHOST_WIDE,)))
        await _accepted(bundle, _memo(bundle, "GPT-6 Luna is the cheaper arm", at=(TOYHOST_NARROW,)))

    async def test_an_arm_the_finding_names_a_trial_of_is_cited(self) -> None:
        bundle = _bundle()
        wide = variant_of_cell_ref(cell_at(bundle, TOYHOST_WIDE))
        trial = next(trial.trial_id for trial in bundle.trials if trial.arm == wide)
        memo = _memo(bundle, "GPT-6 Luna Pro lost a document to a timeout", at=(TOYHOST_NARROW,), trials=(trial,))
        await _accepted(bundle, memo)

    async def test_a_title_naming_no_arm_is_not_read_for_one(self) -> None:
        bundle = _bundle()
        await _accepted(bundle, _memo(bundle, "The wider chunk costs more per document", at=(TOYHOST_NARROW,)))
