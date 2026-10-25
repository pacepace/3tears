"""The ``Report`` — the one page every door into a campaign's report renders, and the evidence it was computed from.

**One page, result first.** A report is an ordered list of blocks in its sections: the first screen (the title — the
campaign's name —, the question declared quoted, each criterion set after the readings it judges, and the subhead under
it, the arms line, the verdict line, the scoreboard and the caveats under it that change how the verdict is read, in normal weight; every other
qualification is a step down, in the sections), the floors clause by clause where any is declared, what it
means (the memo's findings, each a table code built and one sentence, then the contrasts against the control), the
cases, the trials (the census in a line, then a row and a record per failed or lost trial, the passed ones one closed
table), what was not tested, the methods (a table, then the engine's disclosures folded), and the glossary
of every term of art the page linked. Every count, rate, interval and cost in it is code's, computed from the
evidence bundle the report carries (:attr:`Report.bundle`). Every string it prints is held to
:mod:`~threetears.evals.analysis.style`.

**Who wrote what is part of the shape.** A ``text`` block holds what the analysis's author wrote, and nothing else;
what code says — the verdict, the basis, a table's caption, a reason something is not drawn — is a ``disclosure``.
A code-only report (no analysis was generated) holds no ``text`` block at all, and its model refuses one, so a
reader never takes code's computation for an analyst's judgement, or the reverse.

**One chart, and the intervals drawn inside the tables.** The page's one standalone ``chart`` block is the outcome
bar (one stacked bar per arm: passed, failed and lost by cause, every segment's count written beside it). Every
estimate's uncertainty is drawn where the estimate is read: a :class:`TableCell` holding a number may carry its
interval (``low``, ``high``) and a declared ``margin``, and HTML draws them beside the number as a small glyph on one
scale for the column (:mod:`~threetears.evals.analysis.report.svg`), the figures written in the row as text, so
Markdown loses nothing. A ``chart`` block of interval rows (a dot and its interval per row) remains drawable, carrying
its values so Markdown prints them as a table; the builder draws none, since the tables carry the intervals. A cell's
interval is drawn only for an arm with :data:`~threetears.evals.analysis.report.verdict.MIN_CASES` usable cases
and an interval computed.

**The bundle is the machine form.** There is no published JSON of the report: a program reads ``bundle.json``,
which :meth:`Report.write` puts beside the page, and the page names its fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.kernel.authored import Recommendation
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import round_floats
from threetears.evals.analysis.report.blocks import (
    FINDING_ROLES,
    FIRST_SCREEN,
    FOLDED_SECTIONS,
    SECTION_TITLES,
    VERDICT_OUTCOMES,
    BarOutcome,
    CellValue,
    ChartBlock,
    ChartKind,
    ContrastOutcome,
    CostQualityPoint,
    DisclosureBlock,
    DisclosureSource,
    Fact,
    GuardrailOutcome,
    IntervalRow,
    OutcomeBar,
    OutcomeSegment,
    ReportBasis,
    ReportBlock,
    ReportSection,
    TableBlock,
    TableCell,
    TableColumn,
    TextBlock,
    TextRole,
    TrialBlock,
    TrialDetail,
    Verdict,
    VerdictKind,
    VerdictReason,
    fold_anchor,
    fold_ids,
    interval_text,
    interval_value_text,
    reading_anchor,
    table_row,
)
from threetears.evals.analysis.report.serialize_html import campaign_report_html, report_html
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown, report_markdown


#: Each of a campaign's own sections' heading, three words or fewer; a first screen's carries none
#: (:attr:`CampaignReport.section_titles`).
_CAMPAIGN_SECTION_TITLES: dict[str, str] = {
    "answer": "",
    "warnings": "",
    "scorecard": "Scorecard",
    "findings": "Findings",
    "next": "Next steps",
    "provenance": "Design & provenance",
}


class ReportSource(EvalDocumentModel):
    """What the report is a report of, and how that analysis was generated — or that none was."""

    analysis_id: str | None = Field(
        default=None, min_length=1, description="The analysis the report renders; None on a code-only report."
    )
    campaign_id: str = Field(min_length=1, description="The campaign.")
    campaign_name: str | None = Field(default=None, min_length=1, description="The campaign's name, where known.")
    scope_id: str = Field(min_length=1, description="The scope the campaign lives in.")
    behavior: str = Field(min_length=1, description="The behaviour under analysis.")
    generated_at: str = Field(
        min_length=1,
        description="When the analysis was generated (ISO-8601); on a code-only report, when the evidence was assembled.",
    )
    generator_model: str | None = Field(
        default=None, min_length=1, description="The model that wrote the analysis; None on a code-only report."
    )
    bundle_fingerprint: str = Field(min_length=1, description="The fingerprint of the bundle the report renders.")


#: The files :meth:`Report.write` writes, by role.
ReportFile = Literal["html", "bundle", "markdown"]


class Report(EvalDocumentModel):
    """A campaign's report: one page, the bundle it was computed from, and the typed verdicts it states."""

    basis: ReportBasis = Field(description="`analysis` over a generated memo, `code_only` over the evidence alone.")
    title: str = Field(
        min_length=1,
        description=(
            "The title line, a name: the campaign's, with what it compares where no question was declared "
            "(`invoices: does v2 do better than v1?`)."
        ),
    )
    finding_count: int = Field(ge=0, description="How many findings the memo holds.")
    source: ReportSource = Field(description="What this is a report of.")
    blocks: list[ReportBlock] = Field(description="The report, in reading order.")
    verdicts: list[Verdict] = Field(
        default_factory=list,
        description=(
            "Every verdict code reaches, typed — the guardrails, then each bar on each cell, then the contrasts "
            "against the control — the source each printed verdict is rendered from."
        ),
    )
    bundle: AnalysisContextBundle = Field(
        exclude=True, description="The evidence every number here was computed from, written beside the page."
    )

    @model_validator(mode="after")
    def _positions_point_at_findings(self) -> Self:
        """Refuse a block linked to a finding the report does not hold.

        :raises ValueError: A block's ``finding`` or ``rests_on`` names a position past ``finding_count``.
        """
        for index, block in enumerate(self.blocks):
            named = ([block.finding] if block.finding is not None else []) + list(block.rests_on)
            if stray := sorted({position for position in named if not 0 <= position < self.finding_count}):
                raise ValueError(
                    f"block {index} ({block.kind}) is linked to finding position(s) {stray}, and the report holds "
                    f"{self.finding_count} finding(s)"
                )
        return self

    @model_validator(mode="after")
    def _every_link_lands(self) -> Self:
        """Refuse one id carried twice, by cells or blocks, and a cell linking to an id nothing on the page carries.

        A finding's trace links a figure only where the page prints the same figure; a finding whose trace reaches
        nothing says so in words instead (:data:`~threetears.evals.analysis.report.words.UNTRACED`).

        :raises ValueError: An id is carried twice, or a link names one nothing carries.
        """
        anchors = [
            *(cell.anchor for cell in _cells(self.blocks) if cell.anchor is not None),
            *(block.anchor for block in self.blocks if block.anchor is not None),
        ]
        if twice := sorted({anchor for anchor in anchors if anchors.count(anchor) > 1}):
            raise ValueError(f"the ids {twice} are carried more than once, and an id names one place on the page")
        places = set(anchors)
        if stray := sorted({cell.ref for cell in _cells(self.blocks) if cell.ref is not None} - places):
            raise ValueError(f"cells link to {stray}, which nothing on the page carries")
        return self

    @model_validator(mode="after")
    def _every_block_sits_where_the_page_lays_it_out(self) -> Self:
        """Refuse a block in a section an aspect's page has no place for: a campaign's own, or the glossary, which the
        serializers write from the terms the page links. A block no serializer lays out would vanish from the page.

        :raises ValueError: A block sits in a campaign's own section or in the glossary.
        """
        if stray := sorted(
            {
                block.section
                for block in self.blocks
                if block.section in _CAMPAIGN_SECTION_TITLES or block.section == "glossary"
            }
        ):
            raise ValueError(f"an aspect's page lays out no section {', '.join(stray)}, and blocks sit in it")
        return self

    @model_validator(mode="after")
    def _the_basis_matches_what_the_report_holds(self) -> Self:
        """Refuse a report whose basis disagrees with its source or its blocks.

        :raises ValueError: An analysis report names no analysis or model, or a code-only report names one or holds a
            finding or an author's words.
        """
        source = self.source
        if self.basis == "analysis":
            if source.analysis_id is None or source.generator_model is None:
                raise ValueError("an analysis report names its analysis_id and generator_model in its source")
            return self
        if source.analysis_id is not None or source.generator_model is not None:
            raise ValueError("a code-only report renders no analysis, so it names no analysis_id or generator_model")
        if self.finding_count:
            raise ValueError(f"a code-only report has no findings, and this one counts {self.finding_count}")
        if authored := [index for index, block in enumerate(self.blocks) if isinstance(block, TextBlock)]:
            raise ValueError(f"a code-only report holds no author's words, and blocks {authored} are text blocks")
        return self

    @property
    def section_titles(self) -> dict[str, str]:
        """Each section's heading on this page: :data:`SECTION_TITLES`, except that ``contrasts`` is headed
        ``Readings`` where the page contrasts nothing — a one-arm page, whose section holds only the arm's readings
        over all its cases, by stratum and by label — so no heading promises a comparison the page does not make."""
        titles = dict(SECTION_TITLES)
        readings_only = all(
            isinstance(block, TableBlock) and block.name in ("readings", "strata", "labels")
            for block in self.blocks
            if block.section == "contrasts"
        )
        if readings_only:
            titles["contrasts"] = "Readings"
        return titles

    @property
    def terms(self) -> list[str]:
        """Every glossary term the page links, in the order first linked — what its glossary defines."""
        linked: list[str] = []
        for block in self.blocks:
            found = (
                [column.term for column in block.columns]
                if isinstance(block, TableBlock)
                else [block.term]
                if isinstance(block, DisclosureBlock)
                else []
            )
            for term in found:
                if term is not None and term not in linked:
                    linked.append(term)
        return linked

    def write(self, out_dir: Path | str) -> dict[ReportFile, Path]:
        """Write the report as one self-contained page, its Markdown, and the bundle it was computed from.

        ``report.html`` needs nothing beside it — no script file, no stylesheet file, no image; its one script, the kit's
        chart drawing, is inlined in it; it loads the kit's fonts
        from Google Fonts where it can reach them and reads the same in the system's faces where it cannot — and
        ``bundle.json`` is the bundle with its ``schema_version``, so the fingerprint the page names is the one a reader
        computes from the file (``AnalysisContextBundle.from_json(...).fingerprint(like=named)``, in the named one's version: an
        unprefixed one was taken before ``v2:``). ``report.md`` is the same page as Markdown.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role.
        :rtype: dict[ReportFile, Path]
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[ReportFile, Path] = {
            "html": directory / "report.html",
            "bundle": directory / "bundle.json",
            "markdown": directory / "report.md",
        }
        files["html"].write_text(report_html(self), encoding="utf-8")
        files["bundle"].write_text(self.bundle.to_json(indent=2) + "\n", encoding="utf-8")
        files["markdown"].write_text(report_markdown(self), encoding="utf-8")
        return files


#: The shape of a campaign page's JSON (:meth:`CampaignReport.json_document`), raised whenever a reader of the old
#: shape would misread the new one. 3: each aspect's `question_after_runs` is replaced by `late_criteria`. 4: each
#: reading and pair in `readings` states both arms' depth and what limits its interval. 5: the page's model gains a
#: campaign's own sections, a cell's `flag` and the `cost_quality` chart, and the JSON what the page reads across
#: aspects as it lands; one unreleased shape holds them all.
CAMPAIGN_JSON_SHAPE = 5


class CampaignAspectFold(EvalDocumentModel):
    """One aspect's fold of a campaign's page: the writer's summary of the aspect, then that aspect's own page."""

    aspect: str = Field(min_length=1, description="The aspect's label: the fold's heading.")
    shows: str = Field(
        min_length=1,
        description="What code read on the aspect, in a few words beside the heading (`v2 shown worse`).",
    )
    summary: list[ReportBlock] = Field(
        default_factory=list,
        description=(
            "The writer's summary of the aspect (what changed, how sure, what it means), as text blocks; empty "
            "with no memo, or none for this aspect."
        ),
    )
    report: Report = Field(
        description=(
            "The aspect's own page (`build_code_only_report` over its bundle), laid out in the fold below the summary "
            "without its title, its question, its broken floors or its verdict line (:attr:`head`)."
        )
    )

    @property
    def head(self) -> list[ReportBlock]:
        """The aspect page's first screen as the fold lays it out: its arms line, scoreboard and caveats, without its
        question, when it was set and its broken floors, which the campaign's first screen carries once, or its verdict
        line, which reads the aspect alone and could contradict, or pick an arm against, the campaign's table. An aspect
        of one arm keeps its verdict line: it counts that arm's cases passed and missed, and has no other arm to pick
        against."""
        one_arm = len(self.report.bundle.scoreboard) == 1
        dropped = {"floor", "basis", "timing"} if one_arm else {"floor", "basis", "timing", "verdict"}
        return [
            block
            for block in self.report.blocks
            if block.section in FIRST_SCREEN
            and not (isinstance(block, DisclosureBlock) and block.section == "header" and block.source in dropped)
        ]


class CampaignReport(EvalDocumentModel):
    """A campaign's page: the decision first, then what code read across its aspects, the writer's findings and next
    steps, how the campaign was set up, then one fold per aspect.

    **The first screen, in order** (:data:`FIRST_SCREEN`: the ``header``, ``answer``, ``warnings`` and ``scoreboard``
    sections), under the title: the question, or the line saying none was asked; the writer's summary across every
    aspect; the writer's recommendation and its whys, or a line saying none was written; what warns, one plain line
    per arm for the bars it misses, the floors it breaks and the guardrails it is not shown safe on; each report-wide
    caveat once; the person's call, or that none is recorded, and any broken floor it overrides. Nothing stands above
    the title.

    **Then, each under its heading** (:attr:`headed_sections`), in the order
    :func:`~threetears.evals.analysis.report.campaign.page.build_campaign_report` describes: the Scorecard, the memo's
    Findings and Next steps, and Design & provenance. Then :attr:`aspects`, one fold each, in the campaign's order,
    and after them how the page was made.

    **Who wrote what is part of the shape**, as on an aspect's page: a ``text`` block holds what the writer wrote and
    nothing else, and a code-only campaign page (no memo) holds none.
    """

    basis: ReportBasis = Field(description="`analysis` with a memo over the campaign, `code_only` without one.")
    title: str = Field(min_length=1, description="The title line: the campaign's name.")
    campaign_id: str = Field(min_length=1, description="The campaign.")
    campaign_name: str | None = Field(default=None, description="The campaign's name, where known.")
    scope_id: str = Field(min_length=1, description="The scope the campaign lives in.")
    analysis_id: str | None = Field(
        default=None, description="The stored analysis whose memo the page lays out; None on a code-only page."
    )
    finding_count: int = Field(ge=0, description="How many findings the memo holds.")
    blocks: list[ReportBlock] = Field(
        description="Every block but the folds': the first screen, then each headed section's, in reading order."
    )
    aspects: list[CampaignAspectFold] = Field(description="One fold per aspect, in the campaign's order.")
    readings: CampaignReadings = Field(description="What code read off the campaign: every row and every floor.")
    call: CampaignCall | None = Field(default=None, description="The person's call; None while none is recorded.")
    recommendation: Recommendation | None = Field(
        default=None, description="The writer's recommendation, as the memo holds it; None with no memo."
    )
    recommended_variant_key: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The variant key of the arm the recommendation names, as the analysis recorded it when it was saved "
            "(`EvalAnalysis.recommended_variant_key`): what a reader matches a call's `variant_key` to, whatever "
            "later relabels the arm. Set exactly when the page holds a recommendation."
        ),
    )

    @model_validator(mode="after")
    def _the_recommendation_is_keyed(self) -> Self:
        """Refuse a recommendation without its arm's key, and a key with no recommendation to name.

        :raises ValueError: One of ``recommendation`` and ``recommended_variant_key`` is set without the other.
        """
        if (self.recommendation is None) != (self.recommended_variant_key is None):
            raise ValueError(
                "recommended_variant_key is the key of the arm the recommendation names: set exactly when the page "
                "holds a recommendation"
            )
        return self

    @model_validator(mode="after")
    def _every_link_lands_in_a_fold(self) -> Self:
        """Refuse a cell linking anywhere but an aspect's fold, by the id of the section around it
        (``aspect-<fold id>``): a fold is closed, and a link landing inside one is lost where a browser does not open
        it, so a trace lands on the fold's heading.

        :raises ValueError: A link names something other than a fold.
        """
        places = {fold_anchor(fold_id) for fold_id in self.fold_ids}
        if stray := sorted({cell.ref for cell in _cells(self.blocks) if cell.ref is not None} - places):
            raise ValueError(f"cells link to {stray}, which is no aspect's fold")
        return self

    @model_validator(mode="after")
    def _every_block_sits_where_the_page_lays_it_out(self) -> Self:
        """Refuse a block in the glossary, which the serializers write from the terms the page links: a block there
        would vanish from the page. Every other section is on the first screen or headed (:attr:`section_titles`).

        :raises ValueError: A block, or an aspect's summary block, sits in the glossary.
        """
        blocks = [*self.blocks, *(block for fold in self.aspects for block in fold.summary)]
        if any(block.section == "glossary" for block in blocks):
            raise ValueError("the glossary is written from the terms the page links, and blocks sit in it")
        return self

    @model_validator(mode="after")
    def _the_basis_matches_what_the_page_holds(self) -> Self:
        """Refuse a code-only page holding a finding or a writer's words, and a block linked to a finding not held.

        :raises ValueError: A code-only page counts a finding or holds a text block, or a block names a finding
            position past ``finding_count``.
        """
        blocks = [*self.blocks, *(block for fold in self.aspects for block in fold.summary)]
        for index, block in enumerate(blocks):
            named = ([block.finding] if block.finding is not None else []) + list(block.rests_on)
            if stray := sorted({position for position in named if not 0 <= position < self.finding_count}):
                raise ValueError(
                    f"block {index} ({block.kind}) is linked to finding position(s) {stray}, and the page holds "
                    f"{self.finding_count} finding(s)"
                )
        if self.basis == "code_only":
            if analysed := [fold.aspect for fold in self.aspects if fold.report.basis != "code_only"]:
                raise ValueError(
                    f"a code-only page folds code-only aspect pages, and {', '.join(analysed)} renders an analysis"
                )
            if self.analysis_id is not None:
                raise ValueError(f"a code-only page lays out no analysis, and this one names {self.analysis_id}")
            if self.recommendation is not None:
                raise ValueError("a code-only page has no memo, so it holds no recommendation")
            if self.finding_count:
                raise ValueError(f"a code-only page has no findings, and this one counts {self.finding_count}")
            if authored := [index for index, block in enumerate(blocks) if isinstance(block, TextBlock)]:
                raise ValueError(f"a code-only page holds no writer's words, and blocks {authored} are text blocks")
        return self

    @property
    def section_titles(self) -> dict[str, str]:
        """Each section's heading on a campaign's page, in its reading order (:data:`ReportSection`'s): an aspect
        page's (:data:`SECTION_TITLES`) with the campaign's own among them (``answer``, ``warnings``, ``scorecard``,
        ``findings``, ``next`` and ``provenance``), each three words or fewer, the first screen's headed by none."""
        titles = {**SECTION_TITLES, **_CAMPAIGN_SECTION_TITLES}
        return {section: titles[section] for section in get_args(ReportSection)}

    @property
    def headed_sections(self) -> tuple[list[str], list[str]]:
        """The headed sections holding a block, in reading order (:attr:`section_titles`), split at the aspect folds:
        those up to Design & provenance read before the folds, what the page says of how it was made (cases, trials,
        what was not tested, the methods) after them. The first screen and the glossary are laid out apart.

        :return: ``(before, after)``, each a list of section ids.
        :rtype: tuple[list[str], list[str]]
        """
        present = [
            section
            for section in self.section_titles
            if section not in FIRST_SCREEN
            and section != "glossary"
            and any(block.section == section for block in self.blocks)
        ]
        order = list(self.section_titles)
        fold_at = order.index("provenance")
        return (
            [section for section in present if order.index(section) <= fold_at],
            [section for section in present if order.index(section) > fold_at],
        )

    @property
    def terms(self) -> list[str]:
        """Every glossary term the page links, in the order first linked: its own first screen's, then each aspect's
        fold's — what its one glossary defines."""
        linked: list[str] = []
        for block in self.blocks:
            found = (
                [column.term for column in block.columns]
                if isinstance(block, TableBlock)
                else [block.term]
                if isinstance(block, DisclosureBlock)
                else []
            )
            for term in found:
                # Two columns of one table can link one term: it is defined once.
                if term is not None and term not in linked:
                    linked.append(term)
        for fold in self.aspects:
            linked += [term for term in fold.report.terms if term not in linked]
        return linked

    @property
    def verdicts(self) -> list[Verdict]:
        """Every verdict code reaches on the campaign, typed: each aspect's page's, aspect by aspect in the campaign's
        order, each naming its aspect — what a gate reads."""
        return [verdict for fold in self.aspects for verdict in fold.report.verdicts]

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the page as one self-contained HTML file, its Markdown, its JSON, and each aspect's bundle.

        ``report.html`` needs nothing beside it; ``report.md`` is the same page as Markdown; ``campaign.json`` is the
        page's machine form (:meth:`json_document`), naming each aspect's bundle by its fingerprint; and each aspect's
        bundle is written whole as ``bundle-<fold id>.json`` (:attr:`fold_ids`), so the fingerprint the page names is
        the one a reader computes from the file.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role: ``html``, ``markdown``, ``json``, and ``bundle:<aspect>`` per aspect.
        :rtype: dict[str, Path]
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {
            "html": directory / "report.html",
            "markdown": directory / "report.md",
            "json": directory / "campaign.json",
        }
        files["html"].write_text(campaign_report_html(self), encoding="utf-8")
        files["markdown"].write_text(campaign_report_markdown(self), encoding="utf-8")
        files["json"].write_text(json.dumps(self.json_document(), indent=2) + "\n", encoding="utf-8")
        for fold, fold_id in zip(self.aspects, self.fold_ids, strict=True):
            path = directory / f"bundle-{fold_id}.json"
            path.write_text(fold.report.bundle.to_json(indent=2) + "\n", encoding="utf-8")
            files[f"bundle:{fold.aspect}"] = path
        return files

    @property
    def fold_ids(self) -> list[str]:
        """Each aspect's fold as the id its section and every id inside it are named by: the label in lower case,
        each run of other characters one hyphen, numbered where two would read alike."""
        return fold_ids([fold.aspect for fold in self.aspects])

    def json_document(self) -> dict[str, object]:
        """The page's machine form, tagged ``kind: campaign`` at shape :data:`CAMPAIGN_JSON_SHAPE`: code's readings,
        the writer's recommendation (its arm's label and variant key, and how sure) where there is a memo, the call,
        and each aspect's bundle by its fingerprint and shape, with each criterion set after its readings were visible
        and when (``late_criteria``, empty where none was), since its readings read the same either way.

        Every float in it is served to twelve significant figures (:func:`~threetears.evals.schema.hashing.round_floats`):
        a figure code computed carries bits past that which two Python builds compute differently, so one campaign
        serves one JSON across builds, except, rarely, a value that falls exactly on a 12th-figure rounding boundary; a declared value (a
        margin, a bar, alpha) has fewer figures and is unchanged.
        The page's models keep their figures as computed.

        :return: A JSON-ready mapping.
        :rtype: dict[str, object]
        """
        recommendation = self.recommendation
        document: dict[str, object] = {
            "kind": "campaign",
            "shape": CAMPAIGN_JSON_SHAPE,
            "campaign_id": self.campaign_id,
            "basis": self.basis,
            "analysis_id": self.analysis_id,
            "recommendation": None
            if recommendation is None
            else {
                "arm": recommendation.arm,
                "variant_key": self.recommended_variant_key,
                "confidence": recommendation.confidence,
            },
            "readings": self.readings.to_dict(),
            "call": self.call.to_dict() if self.call is not None else None,
            "aspects": [
                {
                    "aspect": fold.aspect,
                    "bundle_fingerprint": fold.report.source.bundle_fingerprint,
                    "bundle_schema_version": fold.report.bundle.schema_version,
                    "late_criteria": [late.to_dict() for late in fold.report.bundle.reading_scope.late_criteria],
                }
                for fold in self.aspects
            ],
        }
        served: dict[str, object] = round_floats(document)
        return served


def _cells(blocks: Sequence[ReportBlock]) -> list[TableCell]:
    """Every cell of every table among ``blocks``, in reading order."""
    return [cell for block in blocks if isinstance(block, TableBlock) for row in block.rows for cell in row.values()]


__all__ = [
    "CAMPAIGN_JSON_SHAPE",
    "FINDING_ROLES",
    "FIRST_SCREEN",
    "FOLDED_SECTIONS",
    "SECTION_TITLES",
    "VERDICT_OUTCOMES",
    "BarOutcome",
    "CampaignAspectFold",
    "CampaignReport",
    "CellValue",
    "ChartBlock",
    "ChartKind",
    "ContrastOutcome",
    "CostQualityPoint",
    "DisclosureBlock",
    "DisclosureSource",
    "Fact",
    "GuardrailOutcome",
    "IntervalRow",
    "OutcomeBar",
    "OutcomeSegment",
    "Report",
    "ReportBasis",
    "ReportBlock",
    "ReportFile",
    "ReportSection",
    "ReportSource",
    "TableBlock",
    "TableCell",
    "TableColumn",
    "TextBlock",
    "TextRole",
    "TrialBlock",
    "TrialDetail",
    "Verdict",
    "VerdictKind",
    "VerdictReason",
    "fold_anchor",
    "fold_ids",
    "interval_text",
    "interval_value_text",
    "reading_anchor",
    "table_row",
]
