"""The one report-document page builder: a document and the report input it was written from, to the page.

:func:`build` lays the page out in its fixed order; what each part says is the document's, and every part no analyst
writes is code's. It reads the document as written: nothing here reads ``story``, and a slot's word, a claim's weight
and a block's kind each set only how that one part looks (its title, its type size, what it draws), never which parts
the page has or their order.

**Part by part:**

- *Masthead*: code's, as the planned page's (:func:`~threetears.evals.analysis.report_plan_facts.plan_facts`), its
  keys written out. *Headline*, *lede*, *flag*: the document's prose, every figure key written out with each arm's name
  and tag (:func:`~threetears.evals.analysis.report.figure_keys.render`).
- *Confidence*: the form in code's words, then each caveat in the document's order: its label (code's for a caveat
  code assigns, from :data:`~threetears.evals.analysis.report.doc_page.words.CAVEAT_LABELS`; the analyst's for an
  ``analyst:`` one), then the analyst's clause, or code's checked line where code proved it immaterial
  (:func:`~threetears.evals.analysis.report_decision.caveat_materiality`).
- *Decision*: each entry in the document's order, titled by its slot word alone.
- *Options*: every arm, with its settings, served model and cost, the arms ``recommends`` names marked
  (:func:`~threetears.evals.analysis.report.plan_page.tables.arms_table`).
- *Claims*: in the document's order, each with the labels of the caveats the document lands beside it (as the
  confidence labels them), so a caveat landed on a claim is one its reader and the editor see. Each evidence block is
  completed and drawn by the code ``block_render`` uses:
  :func:`~threetears.evals.analysis.report.block_spec.complete_chart`, then
  :func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart` under
  :func:`~threetears.evals.analysis.report.block_spec.block_look` (the arms in the input's order, nothing
  recommended), so a block on the page is the block the analyst saw drawn, value for value.
- *Explore*: a table of the document's prose rows. *Next steps*: after the claims.
- *Appendix*: the arms key, every table the input holds
  (:func:`~threetears.evals.analysis.report.plan_page.full_results.every_table`), what was left out and why, Method
  (each term with the labels of the caveats landed on it), the editor's findings left open, and the made line.

**Strict, or reading.** Generation builds strict: a figure key that does not resolve, a block that does not complete
or draw, a caveat code neither assigns nor proves immaterial where the document says it checked, or an arm
recommended that the input does not hold, is refused (:class:`~threetears.evals.analysis.errors.UnresolvableReference`,
naming the field path). A stored document is read with ``strict=False``, and degrades instead (plan §2.3): a key that
no longer resolves prints ``—``; a block that no longer draws as its kind prints as a table of its values, else as its
caption alone, each with code's line saying so; and the page carries one note to regenerate. A document that no longer
validates never reaches here: its stored record is refused by its version first
(:class:`~threetears.evals.kernel.report_doc.EvalReportDoc`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.doc_page.model import (
    DocCaveatLine,
    DocClaim,
    DocConfidence,
    DocEvidence,
    DocFinding,
    DocNext,
    DocOmitted,
    DocPage,
    DocTerm,
)
from threetears.evals.analysis.report.doc_page.words import (
    CAVEAT_LABELS,
    CLAIM_WORD,
    DEGRADED_BLOCK,
    DEGRADED_PAGE,
    DEGRADED_TABLE,
    EXPLORE_HEADERS,
    EXPLORE_TITLE,
    PAGE_WORD,
    SET_APART,
    SEVERITY_WORDS,
    SLOT_TITLES,
    UNRESOLVED,
)
from threetears.evals.analysis.report.figure_keys import NamedKind, format_figure, render, resolve
from threetears.evals.analysis.report.md_parts import one_line
from threetears.evals.analysis.report.plan_page.full_results import every_table
from threetears.evals.analysis.report.plan_page.look import PageLook
from threetears.evals.analysis.report.plan_page.model import (
    PLANNED_SECTION,
    PageChart,
    PageMasthead,
    PageSlot,
)
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.tables import arms_key, arms_table
from threetears.evals.analysis.report.plan_page.words import CONFIDENCE_WORDS, REFERENCE_TEMPLATE
from threetears.evals.analysis.report_decision import (
    Caveat,
    ImpliedCaveat,
    Materiality,
    caveat_materiality,
    every_caveat,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms, plan_facts
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.numbers import format_number
from threetears.evals.kernel.report_doc import DocCaveat, Evidence, ReportDoc, ReportFinding, is_analyst_caveat
from threetears.evals.kernel.report_plan import ChartSpec, PlannedChart

#: The module's entry point, ``build``, with its first production caller: the analyst's ``report_check``
#: (``ops/analyst.py``), which builds a draft's page strict. The writing loop's editor and the report read call it too.
__all__ = ["build"]

#: A figure key in the document's prose, as the figure keys' grammar spells one: ``{path|filter}``.
_KEY = re.compile(r"\{[^{}]*\}")

#: The kind of block a block that no longer draws as its own falls back to: a table of its values.
_TABLE = "table"

#: The caveat kinds whose id names something by a reader name, and which kind of name it is.
_READER_NAMED: Mapping[str, NamedKind] = {
    "unsettled": "aspect",
    "point_estimate": "criterion",
    "borderline": "criterion",
}

#: How the explore table orders its rows.
_EXPLORE_ORDER = "as the report lists them"


@dataclass
class _Reader:
    """One page's reading of its document: strict, refusing what does not resolve, or not, noting each path that
    degraded."""

    report_input: ReportInput
    arm_names: Mapping[str, str]
    strict: bool
    degraded: list[str] = field(default_factory=list)

    def refuse(self, path: str, why: str) -> None:
        """Refuse in strict mode; note the path otherwise.

        :raises UnresolvableReference: Strict.
        """
        if self.strict:
            raise UnresolvableReference(f"{path} {why}")
        self.degraded.append(path)

    def written(self, text: str, path: str) -> str:
        """Prose with its figure keys written out; reading, a key that no longer resolves prints ``—``."""
        try:
            return render(text, self.report_input, self.arm_names)
        except UnresolvableReference as refused:
            self.refuse(path, f"does not resolve on this input: {refused}")
        return _KEY.sub(lambda key: self._one_key(key.group(0)), text)

    def _one_key(self, key: str) -> str:
        try:
            return render(key, self.report_input, self.arm_names)
        except UnresolvableReference:
            # NOSILENT: reading a stored document only (strict refused above); the page prints the key as
            # unresolved and carries the one note to regenerate, which `written` recorded for this field.
            return UNRESOLVED


def _masthead(reader: _Reader) -> PageMasthead:
    masthead = plan_facts(reader.report_input).masthead
    return PageMasthead(
        kind=masthead.kind,
        system=masthead.system,
        counts=reader.written(masthead.counts, "masthead.counts"),
        run_date=reader.written(masthead.run_date, "masthead.run_date"),
        synthetic=masthead.synthetic,
    )


def _code_label(caveat: Caveat | ImpliedCaveat, look: PageLook) -> str:
    """A caveat code assigns, labelled by its kind, naming what its id names as a reader is shown it."""
    rest = caveat.id.partition(":")[2]
    named = _READER_NAMED.get(caveat.kind)
    if named is not None:
        name = look.reader(named, rest)
    elif caveat.kind == "late":
        name = humanize("dimension", rest.partition(":")[0])
    else:
        name = humanize("dimension", rest.rsplit(":", 1)[-1])
    label = CAVEAT_LABELS.get(caveat.kind, humanize("dimension", caveat.kind)).format(name=name)
    return label[:1].upper() + label[1:]


@dataclass(frozen=True)
class _Labels:
    """Each caveat's label wherever the document names it: code's for a caveat code assigns, else the analyst's from
    the confidence that declares it."""

    reader: _Reader
    look: PageLook
    code: Mapping[str, Caveat | ImpliedCaveat]
    declared: Mapping[str, str]

    def label(self, caveat_id: str, path: str) -> str:
        """One caveat's label; strict refuses a caveat neither code's nor the analyst's, reading prints ``—``.

        :raises UnresolvableReference: Strict, and the caveat is neither one code assigns nor a declared analyst one.
        """
        assigned = self.code.get(caveat_id)
        if assigned is not None:
            return _code_label(assigned, self.look)
        own = self.declared.get(caveat_id)
        if is_analyst_caveat(caveat_id) and own is not None:
            return own
        self.reader.refuse(path, f"names caveat {caveat_id!r}, which code does not assign on this input")
        return own or UNRESOLVED

    def landed(self, caveat_ids: Sequence[str], path: str) -> list[str]:
        """The labels of the caveats a claim or a Method item names, in its order."""
        return [self.label(caveat_id, f"{path}[{index}]") for index, caveat_id in enumerate(caveat_ids)]


def _labels(doc: ReportDoc, reader: _Reader, look: PageLook) -> _Labels:
    return _Labels(
        reader=reader,
        look=look,
        code={caveat.id: caveat for caveat in every_caveat(reader.report_input)},
        declared={caveat.id: caveat.label for caveat in doc.confidence.caveats if caveat.label is not None},
    )


def _caveat(
    caveat: DocCaveat, path: str, reader: _Reader, labels: _Labels, materiality: Mapping[str, Materiality]
) -> DocCaveatLine:
    label = labels.label(caveat.id, f"{path}.id")
    if caveat.material_if is not None:
        return DocCaveatLine(label=label, text=reader.written(caveat.material_if, f"{path}.material_if"), checked=False)
    held = materiality.get(caveat.id)
    if held is None or held.checked is None:
        reader.refuse(f"{path}.checked", "is checked immaterial, which code does not prove on this input")
        return DocCaveatLine(label=label, text=UNRESOLVED, checked=True)
    return DocCaveatLine(label=label, text=held.checked, checked=True)


def _confidence(doc: ReportDoc, reader: _Reader, labels: _Labels) -> DocConfidence:
    materiality = caveat_materiality(reader.report_input)
    return DocConfidence(
        form=doc.confidence.form,
        words=CONFIDENCE_WORDS[doc.confidence.form],
        caveats=[
            _caveat(caveat, f"confidence.caveats[{index}]", reader, labels, materiality)
            for index, caveat in enumerate(doc.confidence.caveats)
        ],
    )


def _planned(kind: str, evidence: Evidence, *, reference: bool) -> PlannedChart:
    return PlannedChart.model_validate(
        {
            "kind": kind,
            "title": "",
            "query": evidence.query,
            "reference": [{"kind": evidence.reference}] if reference and evidence.reference is not None else [],
            "emphasis": evidence.emphasis,
        }
    )


def _drawn(
    planned: PlannedChart, path: str, reader: _Reader, look: PageLook, title: str
) -> tuple[ChartSpec, PageChart]:
    """A block completed and drawn, as ``block_render`` draws it.

    :raises UnresolvableReference: It does not complete, or the page cannot draw it from the input.
    """
    spec, faults = complete_chart(planned, reader.report_input, path)
    if faults:
        raise UnresolvableReference("; ".join(f"{at} {why}" for at, why in faults))
    return spec, resolve_chart(spec, reader.report_input, look, title="", fallback_title=title)


def _evidence(evidence: Evidence, path: str, reader: _Reader, look: PageLook) -> DocEvidence:
    caption = reader.written(evidence.caption, f"{path}.caption")
    title = one_line(caption)
    degraded = ""
    drawn: tuple[ChartSpec, PageChart] | None = None
    try:
        drawn = _drawn(_planned(evidence.block, evidence, reference=True), path, reader, look, title)
    except UnresolvableReference as refused:
        reader.refuse(path, f"is a {evidence.block!r} block the page cannot draw on this input: {refused}")
    if drawn is None:
        try:
            drawn = _drawn(_planned(_TABLE, evidence, reference=False), path, reader, look, title)
            degraded = DEGRADED_TABLE
        except UnresolvableReference:
            # NOSILENT: reading a stored document only (strict refused above); the block prints its caption alone
            # under code's line saying so, and the page carries the one note to regenerate.
            degraded = DEGRADED_BLOCK
    if drawn is None:
        return DocEvidence(
            block=evidence.block,
            chart=None,
            caption=caption,
            level="",
            unit="",
            domain="",
            reference="",
            degraded=degraded,
        )
    spec, chart = drawn
    return DocEvidence(
        block=evidence.block,
        chart=chart,
        caption=caption,
        level=spec.level,
        unit=spec.unit,
        domain=" to ".join(format_number(end) for end in spec.domain),
        reference=_reference(spec, reader.report_input),
        degraded=degraded,
    )


def _reference(spec: ChartSpec, report_input: ReportInput) -> str:
    """A block's reference line and the figure it sits at; empty for none."""
    if not spec.reference:
        return ""
    line = spec.reference[0]
    return REFERENCE_TEMPLATE.format(
        kind=humanize("dimension", line.kind).capitalize(), value=format_figure(resolve(line.key, report_input))
    )


def _explore(doc: ReportDoc, reader: _Reader) -> TableBlock | None:
    if not doc.explore:
        return None
    rows = [
        {key: TableCell(value=reader.written(getattr(row, key), f"explore[{index}].{key}")) for key in EXPLORE_HEADERS}
        for index, row in enumerate(doc.explore)
    ]
    return TableBlock(
        section=PLANNED_SECTION,
        name="explore",
        title=EXPLORE_TITLE,
        columns=[TableColumn(key=key, header=header) for key, header in EXPLORE_HEADERS.items()],
        rows=rows,
        order=_EXPLORE_ORDER,
        total_rows=len(rows),
    )


def _findings(findings: Sequence[ReportFinding], places: Mapping[str, int]) -> list[DocFinding]:
    shown = []
    for finding in findings:
        place = places.get(finding.claim)
        shown.append(
            DocFinding(
                about=CLAIM_WORD.format(place=place) if place is not None else PAGE_WORD,
                claim_place=place,
                severity=SEVERITY_WORDS[finding.severity],
                finding=finding.finding,
            )
        )
    return shown


def build(
    doc: ReportDoc,
    report_input: ReportInput,
    *,
    strict: bool,
    findings: Sequence[ReportFinding] = (),
    made: str = "",
) -> DocPage:
    """Build the page a report document draws (this module's account).

    :param doc: The document.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from, which its keys and blocks read.
    :ptype report_input: ReportInput
    :param strict: Refuse what does not resolve or draw (generation); False reads a stored document, degrading.
    :ptype strict: bool
    :param findings: The editor's findings left unresolved, which Method prints; none by default.
    :ptype findings: Sequence[ReportFinding]
    :param made: How the page was made, in one line, for its foot; empty for none.
    :ptype made: str
    :return: The page.
    :rtype: DocPage
    :raises UnresolvableReference: Strict, and a figure key does not resolve, a block does not complete or draw, a
        caveat checked immaterial is not one code proves so, a caveat is neither code's nor the analyst's, or an arm
        recommended is not the input's; each naming its field path.
    """
    arms = fact_arms(report_input)
    reader = _Reader(report_input=report_input, arm_names={arm.id: arm.name for arm in arms}, strict=strict)
    look = block_look(report_input, arms)
    held = {arm.id for arm in report_input.arms}
    for index, arm in enumerate(doc.recommends):
        if arm not in held:
            reader.refuse(f"recommends[{index}]", f"names arm {arm!r}, which the input does not hold")
    labels = _labels(doc, reader, look)
    claims = [
        DocClaim(
            id=claim.id,
            place=place,
            text=reader.written(claim.text, f"claims[{place - 1}].text"),
            weight=claim.weight,
            evidence=[
                _evidence(evidence, f"claims[{place - 1}].evidence[{index}]", reader, look)
                for index, evidence in enumerate(claim.evidence)
            ],
            caveats=labels.landed(claim.caveats, f"claims[{place - 1}].caveats"),
        )
        for place, claim in enumerate(doc.claims, start=1)
    ]
    page = DocPage(
        title=report_input.meta.campaign,
        masthead=_masthead(reader),
        headline=reader.written(doc.headline, "headline"),
        lede=reader.written(doc.lede, "lede"),
        flag=reader.written(doc.flag, "flag") if doc.flag is not None else "",
        confidence=_confidence(doc, reader, labels),
        decision=[
            PageSlot(
                key=entry.slot,
                title=SLOT_TITLES[entry.slot],
                text=reader.written(entry.text, f"decision[{index}].text"),
                highlighted=entry.slot in SET_APART,
            )
            for index, entry in enumerate(doc.decision)
        ],
        options=arms_table(report_input, look, doc.recommends),
        recommended_rows=[index for index, arm in enumerate(report_input.arms) if arm.id in doc.recommends],
        claims=claims,
        explore=_explore(doc, reader),
        next=[
            DocNext(
                text=reader.written(step.text, f"next[{index}].text"),
                settles=reader.written(step.settles, f"next[{index}].settles"),
            )
            for index, step in enumerate(doc.next)
        ],
        arms_key=arms_key(report_input, look),
        full_results=every_table(report_input, look),
        omitted=[
            DocOmitted(
                what=reader.written(item.what, f"omitted[{index}].what"),
                why=reader.written(item.why, f"omitted[{index}].why"),
            )
            for index, item in enumerate(doc.omitted)
        ],
        method=[
            DocTerm(
                term=reader.written(item.term, f"method[{index}].term"),
                text=reader.written(item.text, f"method[{index}].text"),
                caveats=labels.landed(item.caveats, f"method[{index}].caveats"),
            )
            for index, item in enumerate(doc.method)
        ],
        findings=_findings(findings, {claim.id: claim.place for claim in claims}),
        made=made,
    )
    if reader.degraded:
        return page.model_copy(update={"degraded": DEGRADED_PAGE})
    return page
