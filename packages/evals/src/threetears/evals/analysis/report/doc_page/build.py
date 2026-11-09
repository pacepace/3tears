"""The one report-document page builder: a document and the report input it was written from, to the page.

:func:`build` lays the page out in its fixed order; what each part says is the document's, and every part no analyst
writes is code's. It reads the document as written: nothing here reads ``story``, and a slot's word, a claim's weight
and a block's kind each set only how that one part looks (its title, its type size, what it draws), never which parts
the page has or their order.

**Part by part:**

- *Masthead*: code's (GENERATOR §2.1: the campaign's kind, the system under test, the counts and the run date, counts
  and date as figure keys where the input holds the figure), its keys written out. *Draft*: a page its editor's review did not clear says so first, under the masthead, in code's
  one line (:func:`~threetears.evals.analysis.report.doc_page.model.draft_line`), its open findings listed in the
  appendix; the caller says which a page is (``draft``), from the stored document's
  :attr:`~threetears.evals.kernel.report_doc.EvalReportDoc.draft`. *Headline*, *lede*, *flag*: the document's prose, every figure key written out with each arm's name
  and tag (:func:`~threetears.evals.analysis.report.figure_keys.render`).
- *Confidence*: the form in code's words, then each caveat in the document's order: its label (code's for a caveat
  code assigns, from :data:`~threetears.evals.analysis.report.doc_page.words.CAVEAT_LABELS`; the analyst's for an
  ``analyst:`` one), then the analyst's clause, or code's checked line where code proved it immaterial
  (:func:`~threetears.evals.analysis.report_decision.caveat_materiality`).
- *Decision*: each entry in the document's order, titled by its slot word alone.
- *Options*: every arm, with its settings, served model and cost, the arms ``recommends`` names marked
  (:func:`~threetears.evals.analysis.report.parts.tables.arms_table`).
- *Claims*: in the document's order, each with the labels of the caveats the document lands beside it (as the
  confidence labels them), each distinct label once, so a caveat landed on a claim is one its reader and the editor
  see; a label several share says how many and the arms their records name. Each evidence block is
  completed and drawn by the code ``block_render`` uses:
  :func:`~threetears.evals.analysis.report.block_spec.complete_chart`, then
  :func:`~threetears.evals.analysis.report.parts.resolve.resolve_chart` under
  :func:`~threetears.evals.analysis.report.block_spec.block_look` (the arms in the input's order, nothing
  recommended), so a block on the page is the block the analyst saw drawn, value for value.
- *Explore*: a table of the document's prose rows. *Next steps*: after the claims.
- *Appendix*: the arms key, every table the input holds
  (:func:`~threetears.evals.analysis.report.parts.full_results.every_table`), what was left out and why, Method
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
    DocCheck,
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
    CAVEAT_REPEATED,
    CAVEAT_REPEATED_ON,
    CLAIM_WORD,
    DEGRADED_BLOCK,
    DEGRADED_PAGE,
    DEGRADED_TABLE,
    EXPLORE_HEADERS,
    EXPLORE_TITLE,
    LATE_NAMED,
    PAGE_WORD,
    SET_APART,
    SEVERITY_WORDS,
    SLOT_TITLES,
    UNRESOLVED,
)
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    NamedKind,
    format_figure,
    input_check_name,
    is_figure_key,
    render,
    resolve,
    written_arm,
)
from threetears.evals.analysis.report.md_parts import one_line
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.look import PageLook
from threetears.evals.analysis.report.parts.model import (
    WRITER_SECTION,
    PageChart,
    PageMasthead,
    PageSlot,
)
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report.parts.tables import arms_key, arms_table
from threetears.evals.analysis.report.parts.words import CONFIDENCE_WORDS, REFERENCE_TEMPLATE
from threetears.evals.analysis.report_decision import (
    Caveat,
    ImpliedCaveat,
    Materiality,
    caveat_materiality,
    every_caveat,
    late_subject,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.kernel.report_doc import DocCaveat, Evidence, ReportDoc, ReportFinding, is_analyst_caveat
from threetears.evals.kernel.chart_query import ChartSpec, PlannedChart

#: The entry points, with their callers: :func:`build`, for the writing loop's editor (the page text); the store gate
#: (``analysis/report_doc_store.py``), the one checker, which builds a draft's page strict for the analyst's
#: ``report_check`` and the loop alike; and the report writer (``actions/report_writing.py``, the page it stores).
#: :func:`written_prose`, for a surface that prints one piece of a stored document's prose without its page: the
#: campaign listing's headline (``analysis/report_doc_store.py``'s :func:`stored_headline`).
__all__ = ["build", "written_prose"]

#: A figure key in the document's prose, as the figure keys' grammar spells one: ``{path|filter}``.
_KEY = re.compile(r"\{[^{}]*\}")

#: A figure key that prints a caveat's own words (``{caveats.latency}``), and the caveat it prints.
_CAVEAT_KEY = re.compile(r"\{caveats\.([^{}|]+)(?:\|[^{}]*)?\}")

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
    degraded; each key written in the page's forms."""

    report_input: ReportInput
    arm_names: Mapping[str, str]
    strict: bool
    forms: PageForms
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
            return render(text, self.report_input, self.arm_names, forms=self.forms)
        except UnresolvableReference as refused:
            self.refuse(path, f"does not resolve on this input: {refused}")
        return _KEY.sub(lambda key: self._one_key(key.group(0)), text)

    def _one_key(self, key: str) -> str:
        try:
            return render(key, self.report_input, self.arm_names, forms=self.forms)
        except UnresolvableReference:
            # NOSILENT: reading a stored document only (strict refused above); the page prints the key as
            # unresolved and carries the one note to regenerate, which `written` recorded for this field.
            return UNRESOLVED


def written_prose(text: str, report_input: ReportInput) -> str:
    """One piece of a stored document's prose as its page prints it: every figure key written out, each arm by its
    name and tag (:func:`~threetears.evals.analysis.report.figure_keys.render`), by the page's own reading rule, so a
    key that no longer resolves prints ``—``, never its braces.

    :param text: The prose, as the document holds it (its keys unresolved).
    :ptype text: str
    :param report_input: The report input of the snapshot the document was written from.
    :ptype report_input: ReportInput
    :return: The prose with every key written out.
    :rtype: str
    """
    # One piece of prose is read off its page, in the forms its input's page prints each figure in.
    reader = _Reader(
        report_input=report_input,
        arm_names={arm.id: arm.name for arm in fact_arms(report_input)},
        strict=False,
        forms=PageForms.of(report_input),
    )
    return reader.written(text, "prose")


#: The masthead's kind, by what the campaign is (GENERATOR §2.1).
_KIND_LABELS = {
    "change": "Change · control vs candidates",
    "choice": "Choice · candidates compared",
    "measurement": "Measurement · one arm",
}


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        resolve(key, report_input)
    except UnresolvableReference:
        return False
    return True


def _first_resolving(keys: Sequence[str], report_input: ReportInput) -> str | None:
    return next((key for key in keys if _resolves(key, report_input)), None)


def _noun(noun: str, count: object) -> str:
    """``noun`` as a count of ``count`` takes it: singular for one, plural (an ``s``) for none or more than one."""
    return noun if count == 1 else f"{noun}s"


def _keyed_noun(key: str, noun: str, report_input: ReportInput) -> str:
    """``noun`` as the count ``key`` resolves to takes it (:func:`_noun`)."""
    return _noun(noun, resolve(key, report_input).value)


def _masthead_counts(report_input: ReportInput, kind: str) -> str:
    """The masthead's counts, as figure keys where they resolve: the arms, then the tasks or aspects, then the cases;
    each noun singular where its count is one (``vs 1 candidate``, ``× 1 repeat``)."""
    counts: list[str] = []
    if kind == "change":
        name = _first_resolving(("control.name", "control.id"), report_input)
        candidates = _keyed_noun("n_candidates", "candidate", report_input)
        counts.append(f"{{{name}}} vs {{n_candidates}} {candidates}" if name else f"{{n_candidates}} {candidates}")
    elif kind == "choice":
        counts.append(f"{len(report_input.arms)} arms")
    if report_input.tasks:
        counts.append(f"{len(report_input.tasks)} {_noun('task', len(report_input.tasks))}")
    elif len(report_input.aspects) > 1:
        counts.append(f"{len(report_input.aspects)} aspects")
    elif aspect := _first_resolving(("aspect.name", "aspect.id"), report_input):
        counts.append(f"{{{aspect}}}")
    if _resolves("cases", report_input) and _resolves("repeats", report_input):
        cases = _keyed_noun("cases", "case", report_input)
        counts.append(f"{{cases}} {cases} × {{repeats}} {_keyed_noun('repeats', 'repeat', report_input)}")
    return " · ".join(counts)


def _masthead(reader: _Reader) -> PageMasthead:
    """GENERATOR §2.1: kind · system under test · counts · run date, each key written out."""
    report_input = reader.report_input
    meta = report_input.meta
    if len({arm.id for arm in report_input.arms}) <= 1:
        kind = "measurement"
    else:
        kind = "change" if any(arm.role == "control" for arm in report_input.arms) else "choice"
    # As the editor and the judge read it (doc_page_text), so a recorded review keeps its key: a reader's page dates it
    # and names each part once where it prints it (masthead_parts).
    return PageMasthead(
        kind=_KIND_LABELS[kind],
        system=meta.behavior or meta.campaign,
        counts=reader.written(_masthead_counts(report_input, kind), "masthead.counts"),
        run_date=reader.written("{meta.run_date}" if meta.run_date else "", "masthead.run_date"),
        synthetic=bool(meta.synthetic),
    )


def _late_name(rest: str, look: PageLook, report_input: ReportInput) -> str:
    """A late criterion as its label names it: what was set, and, where the page holds more than one aspect or another
    late criterion of its kind, the aspect and the measure its id names, so each line says which one it is."""
    criterion, _, name = rest.partition(":")
    word = humanize("dimension", criterion)
    kin = sum(late.criterion == criterion for late in report_input.caveats.late_criteria)
    if not name or (len(report_input.aspects) < 2 and kin < 2):
        return word
    late = next(
        (late for late in report_input.caveats.late_criteria if (late.criterion, late.name or "") == (criterion, name)),
        None,
    )
    aspect, measure = late_subject(report_input, late) if late is not None else (None, None)
    measure_kind: NamedKind = "judged" if measure in report_input.judged else "measure"
    if aspect is not None and measure is not None:
        return LATE_NAMED["both"].format(
            criterion=word, aspect=look.reader("aspect", aspect), measure=look.reader(measure_kind, measure)
        )
    if aspect is not None:
        return LATE_NAMED["aspect"].format(criterion=word, aspect=look.reader("aspect", aspect))
    return LATE_NAMED["measure"].format(
        criterion=word,
        measure=look.reader(measure_kind, measure) if measure is not None else look.reader("criterion", name),
    )


def _code_label(caveat: Caveat | ImpliedCaveat, look: PageLook, report_input: ReportInput) -> str:
    """A caveat code assigns, labelled by its kind, naming what its id names as a reader is shown it."""
    rest = caveat.id.partition(":")[2]
    named = _READER_NAMED.get(caveat.kind)
    if named is not None:
        name = look.reader(named, rest)
    elif caveat.kind == "late":
        name = _late_name(rest, look, report_input)
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
            return _code_label(assigned, self.look, self.reader.report_input)
        own = self.declared.get(caveat_id)
        if is_analyst_caveat(caveat_id) and own is not None:
            # The analyst's label is the document's prose, so its figure keys are written out as every other field's.
            return self.reader.written(own, path)
        self.reader.refuse(path, f"names caveat {caveat_id!r}, which code does not assign on this input")
        return self.reader.written(own, path) if own is not None else UNRESOLVED

    def landed(self, caveat_ids: Sequence[str], path: str, *, printed: str = "") -> list[str]:
        """The labels of the caveats a claim or a Method item names, in its order, each distinct label once.

        A label several of them share (two short runs, which code labels alike) is said once, with how many carry it
        and what tells them apart where their records say it: the arms they name, as the page writes each arm. A
        caveat whose own words the item's text prints by its key (``printed``: ``{caveats.latency}``) is read, so a
        caveat neither code's nor the analyst's is still refused, and not labelled again beside the words that say it.
        """
        said = set(_CAVEAT_KEY.findall(printed))
        shared: dict[str, list[str]] = {}
        for index, caveat_id in enumerate(caveat_ids):
            label = self.label(caveat_id, f"{path}[{index}]")
            if caveat_id not in said:
                shared.setdefault(label, []).append(caveat_id)
        return [self._said(label, held) for label, held in shared.items()]

    def _said(self, label: str, caveat_ids: Sequence[str]) -> str:
        if len(caveat_ids) == 1:
            return label
        named = [arm for caveat_id in caveat_ids for arm in self._arms_named(caveat_id)]
        arms = [self.look.arm(arm) for arm in dict.fromkeys(named) if arm in self.look.arms]
        count = format_figure(
            Figure(path="count", kind="number", value=float(len(caveat_ids)), scale="count"),
            ("words",),
            forms=self.reader.forms,
        )
        if not arms:
            return CAVEAT_REPEATED.format(label=label, count=count)
        written = ", ".join(written_arm(arm.name, arm.tag) for arm in arms)
        return CAVEAT_REPEATED_ON.format(label=label, count=count, arms=written)

    def _arms_named(self, caveat_id: str) -> list[str]:
        """The arms a code caveat's own record names: a short run's arm, an implied caveat's arms; none otherwise."""
        assigned = self.code.get(caveat_id)
        if isinstance(assigned, ImpliedCaveat):
            return list(assigned.arms)
        if assigned is not None and assigned.kind == "short_run":
            runs = self.reader.report_input.caveats.short_runs
            return [run.arm for run in runs if f"short_run:{run.run}" == caveat_id and run.arm is not None]
        return []


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
        # The axis's ends are what the chart declares it drawn over, each as the page prints a declared value.
        domain=" to ".join(reader.forms.declared(end).text for end in spec.domain),
        reference=_reference(spec, reader.report_input, reader.forms),
        degraded=degraded,
    )


def _reference(spec: ChartSpec, report_input: ReportInput, forms: PageForms) -> str:
    """A block's reference line and the figure it sits at, in the page's forms; empty for none."""
    if not spec.reference:
        return ""
    line = spec.reference[0]
    return REFERENCE_TEMPLATE.format(
        kind=humanize("dimension", line.kind).capitalize(),
        value=format_figure(resolve(line.key, report_input), forms=forms),
    )


def _explore(doc: ReportDoc, reader: _Reader) -> TableBlock | None:
    if not doc.explore:
        return None
    rows = [
        {key: TableCell(value=reader.written(getattr(row, key), f"explore[{index}].{key}")) for key in EXPLORE_HEADERS}
        for index, row in enumerate(doc.explore)
    ]
    return TableBlock(
        section=WRITER_SECTION,
        name="explore",
        title=EXPLORE_TITLE,
        columns=[TableColumn(key=key, header=header) for key, header in EXPLORE_HEADERS.items()],
        rows=rows,
        order=_EXPLORE_ORDER,
        total_rows=len(rows),
    )


#: An engine key in an editor's finding, as the editor quoted it: a dotted figure path, in backticks or not, and the
#: value it was read at where the editor wrote one (``settle.margin.<aspect>.<arm>.cases_needed_80 = 62``).
_FINDING_KEY = re.compile(
    r"`?(?P<key>[a-z_][\w-]*(?:\.[\w<>*-]+)+)`?(?:\s*=\s*`?(?P<value>-?\d[\d.,]*\d|-?\d)`?)?(?![\w-])"
)

#: A figure key's last word as a sentence says it, where the key names no figure on this input (an editor's
#: placeholder, ``<arm>``): the settle sizes by what they buy, any other by its words.
_STATISTIC_WORDS = {
    "cases_needed": "the cases that settle it half the time",
    "cases_needed_80": "the cases that settle it four times in five",
    "cases_needed_two_sided": "the cases a two-sided interval needs half the time",
    "cases_needed_two_sided_80": "the cases a two-sided interval needs four times in five",
    "trials_needed": "the trials that settle it half the time",
    "trials_needed_80": "the trials that settle it four times in five",
    "n": "the count",
}

#: A figure key's words that name its place in the grammar, not what it is about.
_PLACE_WORDS = frozenset({"overall", "criteria", "margin", "settle", "measure", "compare", "cost", "reading", "task"})


def _key_words(path: str) -> str:
    """A figure key that names no figure on this input, in words: what it measures and of what, never its path."""
    *subject, statistic = path.split(".")
    named = [
        "an arm" if word.startswith("<") else measure_name(word)[:1].lower() + measure_name(word)[1:]
        for word in subject[1:]
        if word not in _PLACE_WORDS and word != "*"
    ]
    said = _STATISTIC_WORDS.get(statistic, statistic.replace("_", " "))
    return f"{said} ({', '.join(named)})" if named else said


#: A claim's id in an editor's finding (``c2``), which the page names by its place.
_CLAIM_ID = re.compile(r"\b(?P<id>c\d+)\b")


def _finding_words(text: str, reader: _Reader, places: Mapping[str, int]) -> str:
    """An editor's finding as a page prints it: each figure key it quotes written out as the figure it names (its
    value, as every key on the page prints), each claim id as the claim's place (``claim 2``), so no finding prints
    the engine's vocabulary. A key that does not resolve on this input is left as the editor wrote it."""

    def figure(match: re.Match[str]) -> str:
        key = match.group("key").rstrip(".")
        tail = match.group(0)[len(match.group(0).rstrip(".")) :]
        try:
            return render(f"{{{key}}}", reader.report_input, reader.arm_names, forms=reader.forms) + tail
        except UnresolvableReference:
            # NOSILENT: a figure key's form that names no figure here (an editor's placeholder) reads in words; any
            # other dotted word (a version, a model id) is the editor's and stands.
            return _key_words(key) + tail if is_figure_key(key) else match.group(0)

    def claim(match: re.Match[str]) -> str:
        place = places.get(match.group("id"))
        if place is None:
            return match.group(0)
        # ``Claim 2`` where it opens a sentence, ``claim 2`` inside one.
        opens = (
            not match.string[: match.start()].rstrip(" `") or match.string[: match.start()].rstrip(" `")[-1] in ".!?"
        )
        word = CLAIM_WORD.format(place=place)
        return word if opens else word[:1].lower() + word[1:]

    return _CLAIM_ID.sub(claim, _FINDING_KEY.sub(figure, text))


def _findings(findings: Sequence[ReportFinding], places: Mapping[str, int], reader: _Reader) -> list[DocFinding]:
    shown = []
    for finding in findings:
        place = places.get(finding.claim)
        shown.append(
            DocFinding(
                about=CLAIM_WORD.format(place=place) if place is not None else PAGE_WORD,
                claim_place=place,
                severity=SEVERITY_WORDS[finding.severity],
                # As the editor wrote it, which the judge reads (doc_page_text); a reader's page prints its words.
                finding=finding.finding,
                words=_finding_words(finding.finding, reader, places),
            )
        )
    return shown


def _checks(report_input: ReportInput) -> list[DocCheck]:
    """Each goal check the input reads, by the name the page's tables print (:func:`input_check_name`) and the
    expression it stands for, in the checks' sorted order; none named by its expression itself, which has nothing to
    define."""
    named = []
    for key in report_input.measures:
        check = goal_check_of(key)
        name = input_check_name(key, report_input)
        if check is not None and name is not None and name != f"`{check}`":
            named.append(DocCheck(name=name, expression=check))
    return sorted(named, key=lambda held: held.expression)


def build(
    doc: ReportDoc,
    report_input: ReportInput,
    *,
    strict: bool,
    findings: Sequence[ReportFinding] = (),
    made: str = "",
    draft: bool = False,
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
    :param draft: Whether the page is a draft, its editor's review not cleared, which its first line then says.
    :ptype draft: bool
    :return: The page.
    :rtype: DocPage
    :raises UnresolvableReference: Strict, and a figure key does not resolve, a block does not complete or draw, a
        caveat checked immaterial is not one code proves so, a caveat is neither code's nor the analyst's, or an arm
        recommended is not the input's; each naming its field path.
    """
    arms = fact_arms(report_input)
    # Every figure the page states, its tables' and its prose keys', is in one form per measure, decided once here for
    # the whole page.
    forms = PageForms.of(report_input)
    reader = _Reader(
        report_input=report_input, arm_names={arm.id: arm.name for arm in arms}, strict=strict, forms=forms
    )
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
        # The page's name is the campaign's: what it compares is the masthead's, so the name never stacks.
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
        options=arms_table(report_input, look, doc.recommends, forms=forms),
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
        arms_key=arms_key(report_input, look, forms=forms),
        full_results=every_table(report_input, look, forms=forms),
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
                caveats=labels.landed(item.caveats, f"method[{index}].caveats", printed=item.text),
            )
            for index, item in enumerate(doc.method)
        ],
        findings=_findings(findings, {claim.id: claim.place for claim in claims}, reader),
        made=made,
        draft=draft,
        checks=_checks(report_input),
    )
    if reader.degraded:
        return page.model_copy(update={"degraded": DEGRADED_PAGE})
    return page
