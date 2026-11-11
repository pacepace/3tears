"""An arm is named by what tells it from the other arms of its report — and by nothing else (#567).

A seven-run assistant campaign whose runs shared every lever and differed only in their rig printed every
lever it carried, ``name=value`` joined, as each arm's name: about 1,700 characters, in sixty places,
the levers of other kinds included (``classifier_prompt=(not a classifier run)``). A reader needs the
difference, so the rule is:

- an arm is named by the levers on which the report's arms differ;
- one that differs on none is named by its candidate kind and model, plus its rig where an arm was
  measured under more than one;
- a lever that does not apply to the arm's kind (the engine's "not a run of this kind" level) is never
  named;
- every level is cut alike, wherever the name is printed — and two arms the cut would name alike are
  told apart by their digests, so no two arms ever print the same name;
- no report dumps an arm's whole lever set: it names the arm by its difference, and the bundle beside it carries the
  rest;
- where a chart has no room for an arm's name it prints the arm's short name (GENERATOR.md §6.1): the campaign's
  label, else the control's role, else only what differs from the control, the model first and trimmed of the prefix
  it shares with the control's; never longer than the arm's own name, and two arms whose short names would read alike
  are each named in full.

The first half pins the rule on the naming functions themselves; the second builds code-only reports through the toy
host and reads their tables in both outputs, Markdown and HTML, since a name is only as consistent as the one surface
that spells it differently.
"""

from __future__ import annotations

import html
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from threetears.evals.analysis import (
    Report,
    assemble_context_bundle,
    build_code_only_report,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.campaigns import campaign_arms, set_campaign_control
from threetears.evals.analysis.arms import (
    ELISION,
    LABEL_LEVEL_CHARS,
    NO_LEVER_MOVED,
    arm_label,
    arm_names,
    elide_level,
    short_digest,
    writer_arms,
)
from threetears.evals.analysis.report import ChartBlock, TableBlock
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.host import HostProfile, KindContract
from threetears.evals.schema import SweepableValue
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.models import EvalRun
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design, toyhost_campaign_of_one_aspect
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
    toyhost_observation,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.g2_support import g2_report
from packages.evals.tests.report_support import toy_campaign_host

# =============================================================================
# The rule, on the naming functions
# =============================================================================


def _value(content: Any, display: str | None = None) -> SweepableValue:
    return SweepableValue.of(content, display=display if display is not None else str(content))


def _entry(**levers: SweepableValue) -> VariantIndexEntry:
    """An index entry over ``levers``; a dotted lever name is spelled with ``__`` here."""
    named = {name.replace("__", "."): value for name, value in levers.items()}
    return VariantIndexEntry(variant_key=compute_variant_key(named), levers=named)


#: What an assistant arm of #567 shared with every other: a long description, directives, a model and its kind.
_SHARED: dict[str, SweepableValue] = {
    "candidate_kind": _value("assistant"),
    "model": _value("openai/gpt-5-mini"),
    "backstory": _value("A support agent who " + "walks each customer through the fix one step at a time " * 20),
    "directives": _value("Answer in the customer's language; never guess an order number."),
}


class TestAnArmIsNamedByWhatTellsItApart:
    def test_the_only_arm_of_a_report_is_named_by_its_kind_and_model(self) -> None:
        """One arm differs from nothing, so no lever names it — its kind and model say what ran."""
        (entry,) = index = [_entry(**_SHARED)]
        assert arm_names(index) == {entry.variant_key: "candidate_kind=assistant, model=openai/gpt-5-mini"}

    def test_a_described_arm_with_neither_a_difference_nor_a_model_says_it_moved_nothing(self) -> None:
        (entry,) = index = [_entry(temperature=_value(0.2))]
        assert arm_names(index) == {entry.variant_key: NO_LEVER_MOVED}

    def test_arms_that_differ_on_one_lever_are_named_by_that_lever_alone(self) -> None:
        cool, warm = index = [_entry(**_SHARED, temperature=_value(t)) for t in (0.2, 0.7)]
        assert arm_names(index) == {cool.variant_key: "temperature=0.2", warm.variant_key: "temperature=0.7"}

    def test_an_axis_one_arm_lacks_still_tells_the_arms_apart(self) -> None:
        """Absence is a level: the control of a one-knob sweep is named by the model it shares."""
        control, swept = index = [_entry(**_SHARED), _entry(**_SHARED, temperature=_value(0.7))]
        names = arm_names(index)
        assert names[swept.variant_key] == "temperature=0.7"
        assert names[control.variant_key] == "candidate_kind=assistant, model=openai/gpt-5-mini"


class TestALeverThatDoesNotApplyIsNeverNamed:
    def _mixed(self, *prompts: str) -> tuple[VariantIndexEntry, list[VariantIndexEntry]]:
        """An assistant arm and one classifier arm per prompt, each carrying the other kind's levers as the
        engine resolves them: at that kind's "not a run of this kind" level."""
        assistant = _entry(**_SHARED, classifier__prompt=SweepableValue.not_this_kind("classifier"))
        not_assistant = SweepableValue.not_this_kind("assistant")
        classifiers = [
            _entry(
                candidate_kind=_value("classifier"),
                model=_value("openai/gpt-5-mini"),
                backstory=not_assistant,
                directives=not_assistant,
                classifier__prompt=_value(prompt),
            )
            for prompt in prompts
        ]
        return assistant, classifiers

    def test_across_kinds_the_kind_names_the_arm_and_nothing_inapplicable_does(self) -> None:
        """An assistant arm and a classifier arm differ by kind; each one's own levers do not set it apart
        from an arm they do not apply to."""
        assistant, (classifier,) = self._mixed("Label the request.")
        names = arm_names([assistant, classifier])
        assert names == {
            assistant.variant_key: "candidate_kind=assistant",
            classifier.variant_key: "candidate_kind=classifier",
        }

    def test_a_lever_that_differs_among_the_arms_it_applies_to_names_only_those(self) -> None:
        assistant, classifiers = self._mixed("Label the request.", "Label the request in one word.")
        names = arm_names([assistant, *classifiers])
        assert names[assistant.variant_key] == "candidate_kind=assistant"
        assert sorted(names[arm.variant_key] for arm in classifiers) == [
            "candidate_kind=classifier, classifier.prompt=Label the request in one word.",
            "candidate_kind=classifier, classifier.prompt=Label the request.",
        ]

    def test_the_writer_is_never_handed_a_lever_that_does_not_apply(self) -> None:
        assistant, (classifier,) = self._mixed("Label the request.")
        view = writer_arms([assistant, classifier])
        assert view["shared_levels"] == {"model": "openai/gpt-5-mini"}
        levels = {arm["variant_key"]: arm["levels"] for arm in view["arms"]}  # type: ignore[attr-defined]
        assert set(levels[assistant.variant_key]) == {"backstory", "candidate_kind", "directives"}
        assert set(levels[classifier.variant_key]) == {"candidate_kind", "classifier.prompt"}
        assert "(not a" not in repr(view)

    def test_arms_that_differ_only_where_a_lever_does_not_apply_are_still_told_apart(self) -> None:
        """No kind lever here, so nothing the rule may name separates them: each is named by its model, and
        the digest says the names shown do not tell them apart."""
        bare = _entry(model=_value("m"), extractor__style=SweepableValue.not_this_kind("extractor"))
        styled = _entry(model=_value("m"), extractor__style=_value("terse"))
        assert arm_names([bare, styled]) == {
            arm.variant_key: f"model=m (arm {short_digest(arm.variant_key)})" for arm in (bare, styled)
        }

    def test_a_host_value_that_only_reads_like_the_level_is_named_like_any_value(self) -> None:
        """Recognised by its content hash: a display alone proves nothing about what the level is."""
        lookalike = _value(None, display="(not a classifier run)")
        assert lookalike.not_of_kind is None
        (entry, other) = index = [_entry(prompt=lookalike), _entry(prompt=_value("Label the request."))]
        assert arm_names(index)[entry.variant_key] == "prompt=(not a classifier run)"
        assert arm_names(index)[other.variant_key] == "prompt=Label the request."

    def test_the_level_names_its_kind_and_is_the_one_a_kind_contract_mints(self) -> None:
        level = SweepableValue.not_this_kind("classifier")
        assert (level.not_of_kind, level.display) == ("classifier", "(not a classifier run)")
        another_kinds_run = toyhost_observation().model_copy(update={"candidate_kind": "something-else"})
        minted = TOY_EXTRACTOR_CONTRACT.levels(another_kinds_run)
        assert minted and {value.not_of_kind for value in minted.values()} == {TOY_EXTRACTOR_KIND}


#: Two long values that part only in the middle — exactly the characters a cut drops.
_PARTING_IN_THE_MIDDLE = tuple(
    "Open with the station ident, then " + word + " the request line for a full hour before the news"
    for word in ("read", "skip")
)


class TestNoTwoArmsShareAName:
    def test_a_long_level_is_cut_in_the_middle_on_one_line(self) -> None:
        text = "first line\n" + "x" * 100 + " the end of it"
        cut = elide_level(text)
        assert len(cut) == LABEL_LEVEL_CHARS and ELISION in cut and "\n" not in cut
        assert cut.startswith("first line x") and cut.endswith("the end of it")
        assert elide_level("short  value\n") == "short value"

    def test_arms_the_cut_would_name_alike_are_told_apart_by_their_digests(self) -> None:
        first, second = index = [_entry(**_SHARED, opening=_value(text)) for text in _PARTING_IN_THE_MIDDLE]
        assert elide_level(_PARTING_IN_THE_MIDDLE[0]) == elide_level(_PARTING_IN_THE_MIDDLE[1])
        names = arm_names(index)
        assert names[first.variant_key] != names[second.variant_key]
        for entry in (first, second):
            assert names[entry.variant_key] == (
                f"opening={elide_level(_PARTING_IN_THE_MIDDLE[0])} (arm {short_digest(entry.variant_key)})"
            )

    def test_two_values_a_host_displayed_alike_are_told_apart_too(self) -> None:
        index = [_entry(prompt=_value(text, display="a prompt")) for text in ("one", "two")]
        names = arm_names(index)
        assert len(set(names.values())) == 2 and all(
            name.startswith("prompt=a prompt (arm ") for name in names.values()
        )

    def test_arms_the_cut_keeps_apart_carry_no_digest(self) -> None:
        """The digest marks levels that do not tell arms apart, so it appears only where they do not."""
        texts = [_PARTING_IN_THE_MIDDLE[0] + suffix for suffix in (" — rev 1", " — rev 2")]
        index = [_entry(**_SHARED, opening=_value(text)) for text in texts]
        names = arm_names(index)
        assert sorted(names.values()) == sorted(f"opening={elide_level(text)}" for text in texts)
        assert not any("(arm " in name for name in names.values())

    def test_a_cell_adds_its_rig_and_an_arm_the_index_lacks_says_so(self) -> None:
        (entry,) = index = [_entry(**_SHARED)]
        names = arm_names(index)
        assert arm_label(entry.variant_key, names, rig="abc") == f"{names[entry.variant_key]} @ rig abc"
        assert arm_label("f" * 64, names).startswith(f"unplaced ({'f' * 12})")


# =============================================================================
# Every block of a report, in all three outputs
# =============================================================================


class _Summarizer(BaseModel):
    """A second kind's one overlay, so every toy-extractor run carries a lever that does not apply to it."""

    style: str = Field(default="terse", title="style", description="how terse the summary reads")


def _profile_with_a_second_kind() -> HostProfile:
    return replace(
        toyhost_profile(), kinds=(TOY_EXTRACTOR_CONTRACT, KindContract("toy-summarizer", overlays=_Summarizer))
    )


def _aspect_page(host: Any, campaign: Any) -> Report:
    """The campaign's one aspect's code-only page, naming each arm by its variant, as the aspect page's builder does
    when no campaign labels are passed to it: the names these tests pin."""
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    return build_code_only_report(bundle, assembled_at="2026-10-18T00:00:00+00:00", campaign_name=campaign.name)


def _code_only_report(batches: list[EvalRun], profile: HostProfile) -> Report:
    """The code-only report of a toy-host campaign over ``batches``, measured alike."""
    host = toyhost_host(profile=profile)
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    campaign = toyhost_campaign_of_one_aspect(
        id="7b0e2c55-5d0f-4c8e-9a3c-1f2e3d4c5b6a",
        scope_id=TOYHOST_SCOPE,
        name="arm labels",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    return _aspect_page(host, campaign)


def _batch(*, reviewer_pool: str = "pool-a", instructions: str | None = None) -> EvalRun:
    batch = toyhost_batch(
        chunk_tokens=256,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool=reviewer_pool,
        batch_label=instructions or reviewer_pool,
    )
    return batch if instructions is None else batch.model_copy(update={"overlays": {"instructions": instructions}})


def _table(report: Report, name: str) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def _arm_column(report: Report, name: str, column: str = "arm") -> list[str]:
    """The named table's arm column, as served, without the control's marker."""
    return [str(row[column]).removeprefix("▸ ") for row in _table(report, name).values()]


def _assert_every_output_prints(report: Report, names: set[str]) -> None:
    """Each name the scoreboard prints reaches Markdown and HTML spelled alike, and no other arm name does."""
    markdown, page = report_markdown(report), report_html(report)
    assert set(_arm_column(report, "scoreboard")) == names
    for name in names:
        assert f"| {name} |" in markdown or f"| ▸ {name} |" in markdown, name
        # A cell of 40 characters or more wraps (``class="wrap"``), as every long cell of a table does; a hyphenated
        # name is held whole, as one word (``span.id``), its spelling as it is.
        spelled = [
            html.escape(name),
            *(f'<span class="{held}">{html.escape(name)}</span>' for held in ("id", "id long") if "-" in name),
        ]
        cells = [
            f"{td}{marker}{said}</td>"
            for td in ("<td>", '<td class="wrap">')
            for marker in ("", "▸ ")
            for said in spelled
        ]
        assert any(cell in page for cell in cells), name


@pytest.fixture(scope="module")
def rig_only() -> Report:
    """#567's shape: two runs at identical levers under two rigs, with another kind's lever on each."""
    profile = _profile_with_a_second_kind()
    return _code_only_report([_batch(reviewer_pool="pool-a"), _batch(reviewer_pool="pool-b")], profile)


class TestARigOnlyCampaignIsNamedByItsModel:
    # The page reads the name's levers as words, never ``axis=level``.
    _ARM = "extractor-v2"

    def test_the_arm_is_named_by_its_kind_and_model(self, rig_only: Report) -> None:
        assert _arm_column(rig_only, "scoreboard") == [self._ARM]

    def test_every_output_prints_the_same_short_name(self, rig_only: Report) -> None:
        _assert_every_output_prints(rig_only, {self._ARM})

    def test_the_other_kinds_lever_is_printed_nowhere(self, rig_only: Report) -> None:
        for output in (report_markdown(rig_only), report_html(rig_only)):
            assert "toy-summarizer" not in output

    def test_no_output_dumps_the_levers_the_arms_share(self, rig_only: Report) -> None:
        for output in (report_markdown(rig_only), report_html(rig_only)):
            assert "chunk_tokens=256tok" not in output
            assert "retriever_top_k=3" not in output


@pytest.fixture(scope="module")
def one_lever() -> Report:
    """The toy campaign: two arms that differ on chunk width alone."""
    host, campaign = toy_campaign_host()
    return _aspect_page(host, campaign)


@pytest.fixture(scope="module")
def cut_alike() -> Report:
    """Two arms whose one difference is in the middle of a long value — the characters a cut drops."""
    return _code_only_report([_batch(instructions=text) for text in _PARTING_IN_THE_MIDDLE], toyhost_profile())


class TestArmsDifferingOnOneLeverAreNamedByIt:
    def test_every_block_names_each_arm_by_that_lever_alone(self, one_lever: Report) -> None:
        _assert_every_output_prints(one_lever, {"256tok", "1024tok"})

    def test_the_cases_table_heads_each_arm_s_column_by_the_same_name(self, one_lever: Report) -> None:
        headers = {column.header.removeprefix("▸ ") for column in _table(one_lever, "cases").columns}
        assert {"256tok", "1024tok"} <= headers


class TestTheCutNeverMergesTwoArms:
    def test_the_two_arms_are_named_apart_in_every_output(self, cut_alike: Report) -> None:
        arms = _arm_column(cut_alike, "scoreboard")
        assert len(arms) == 2 and len(set(arms)) == 2
        assert all(arm.startswith("extractor.instructions=") and ELISION in arm and " (arm " in arm for arm in arms)
        _assert_every_output_prints(cut_alike, set(arms))


# --- a campaign's derived arms are labelled as the page names them -------------------------------------------------


def test_a_derived_arm_is_labelled_by_the_page_s_word_and_a_taken_one_gets_a_number() -> None:
    """The toy campaign names no arm, so both chunk widths are derived: each labelled ``1024tok``, never
    ``chunk_tokens=1024tok``. Naming the 256-token control ``1024tok`` takes that word, so the derived arm reads
    ``1024tok (2)``."""
    host, campaign = toy_campaign_host()
    derived = campaign_arms(host.storage, campaign, profile=host.profile)
    assert sorted(arm.label for arm in derived) == ["1024tok", "256tok"]
    assert all(arm.role == "candidate" for arm in derived)

    named = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, campaign.run_ids[0], set_by="t", profile=host.profile
    )
    assert named.control is not None
    held = named.model_copy(update={"arms": [named.control.model_copy(update={"label": "1024tok"})]})
    labels = [arm.label for arm in campaign_arms(host.storage, held, profile=host.profile)]
    assert labels[0] == "1024tok" and labels[1:] == ["1024tok (2)"]


# --- an arm's short name, where a chart has no room for its name ---------------------------------------------------

#: The control's model, and two candidates' sharing its prefix: one trimmed to ``flash``, one whose separator differs,
#: so both trim to ``flash`` too.
_CONTROL_MODEL, _FLASH, _FLASH_UNDERSCORED = "Meridian-4.1", "Meridian-4.1-flash", "Meridian-4.1_flash"


def _modelled_page(arms: list[tuple[str, int]], labels: dict[int, str] | None = None) -> tuple[Report, list[str]]:
    """The code-only page of a toy campaign with one arm per ``(model, chunk_tokens)``, the first its control, and the
    campaign's ``labels`` by arm position; with each arm's variant key, in that order."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    batches = [
        toyhost_batch(
            chunk_tokens=chunk, retriever_top_k=3, extraction_schema="v1", ocr_engine_version="tess-5.3.1", model=model
        ).model_copy(update={"candidate_model": model})
        for model, chunk in arms
    ]
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    campaign = toyhost_campaign_of_one_aspect(
        id="5f1d7c2a-8e3b-4a6d-9c0f-2b7e4d1a6c3e",
        scope_id=TOYHOST_SCOPE,
        name="short names",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    campaign = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, batches[0].id, set_by="t", profile=profile
    )
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    by_levels = {
        (entry.levers["model"].display, entry.levers["chunk_tokens"].display): entry.variant_key
        for entry in bundle.variant_index
    }
    keys = [by_levels[(model, f"{chunk}tok")] for model, chunk in arms]
    given = {keys[position]: label for position, label in (labels or {}).items()}
    report = build_code_only_report(
        bundle, assembled_at="2026-10-18T00:00:00+00:00", campaign_name=campaign.name, arm_labels=given or None
    )
    return report, keys


def _short_names(report: Report) -> dict[str, str | None]:
    """Each arm's name on the outcome bar, to the short name it carries for a tight place."""
    (chart,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.chart == "outcome_bar"]
    return {bar.arm: bar.short_name for bar in chart.bars}


class TestAnArmsShortNameIsWhatSetsItApart:
    def test_the_control_is_its_role_and_a_candidate_only_what_differs_with_the_shared_model_prefix_trimmed(
        self,
    ) -> None:
        """GENERATOR.md §6.1: against ``Meridian-4.1`` at 256 tokens, ``Meridian-4.1-flash`` at 1024 is ``flash ·
        chunk tokens 1024tok``, the model first; at 256 it differs in its model alone, so is ``flash``."""
        report, keys = _modelled_page([(_CONTROL_MODEL, 256), (_FLASH, 1024), (_FLASH, 256)])
        shorts = _short_names(report)
        control, wide, narrow = (
            next(name for name in shorts if f"model {model}" in name and f"{chunk}tok" in name)
            for model, chunk in ((_CONTROL_MODEL, 256), (_FLASH, 1024), (_FLASH, 256))
        )
        assert shorts == {control: "control", wide: "flash · chunk tokens 1024tok", narrow: "flash"}
        assert all(len(short or name) <= len(name) for name, short in shorts.items())
        assert len(keys) == len(shorts)

    def test_a_campaign_label_names_the_arm_in_tight_places_too(self) -> None:
        """An alias wins: the control labelled ``Production`` is ``Production``, a candidate labelled ``fast`` is
        ``fast``, and the unlabelled candidate is still named by what differs."""
        report, _ = _modelled_page(
            [(_CONTROL_MODEL, 256), (_FLASH, 1024), (_FLASH, 256)], labels={0: "Production", 2: "fast"}
        )
        shorts = _short_names(report)
        assert shorts["Production"] == "Production" and shorts["fast"] == "fast"
        (other,) = [name for name in shorts if name not in ("Production", "fast")]
        assert shorts[other] == "flash · chunk tokens 1024tok"

    def test_two_arms_whose_short_names_would_read_alike_are_each_named_in_full(self) -> None:
        """``Meridian-4.1-flash`` and ``Meridian-4.1_flash``, both at 1024 tokens, both trim to ``flash · chunk tokens
        1024tok``: each is named in full, so the two never read alike; the control keeps its role."""
        report, _ = _modelled_page([(_CONTROL_MODEL, 256), (_FLASH, 1024), (_FLASH_UNDERSCORED, 1024)])
        shorts = _short_names(report)
        candidates = {name: short for name, short in shorts.items() if short != "control"}
        assert len(candidates) == 2 and all(short == name for name, short in candidates.items())
        assert len(set(shorts.values())) == len(shorts)

    async def test_an_arm_s_own_name_wins_where_it_is_already_short(self) -> None:
        """The quick path names its arms ``v4.1-flash`` and ``v4-flash``: no short name is longer than the arm's own,
        so ``v4-flash`` is ``v4-flash``, never ``candidate v4-flash``."""
        shorts = _short_names(await g2_report())
        assert shorts["v4-flash"] == "v4-flash"
        assert all(len(short or name) <= len(name) for name, short in shorts.items())
