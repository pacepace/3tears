"""``Claims``: the factual claims in each answer, listed and checked by the engine's own claims grader.

A :class:`~threetears.evals.quick.judged.Judge` scores an answer as a whole. Listed in an eval's ``grader``, a
:class:`Claims` instead has a model list the factual claims each answer makes and check each one, through the claim
criterion the engine grades in its judge phase (:class:`~threetears.evals.schema.ClaimCriterion`). Its readings rank
arms on their own, never in a composite and never in a case's pass:

- **False claims per answer** (``<name>_false_per_unit``) and **the share of answers holding one**
  (``<name>_false_unit_share``): claims the case's own evidence contradicts, or, for a claim no case evidence covers
  that the grader doubts, a quote from the web. Lower is better.
- **The grounded share** (``<name>_grounded_share``), with ``grounded_in``: of the specific claims an answer makes,
  the share its own documents support. Higher is better.

What the grader reads, each the eval's to say:

- **The answer, cut into segments.** By default one segment, the whole answer, which is one unit (one answer). A
  ``segments`` callable cuts a structured answer into several (a list of entries, each a unit), as text or as
  :class:`~threetears.evals.schema.Segment` values for a unit spanning several pieces, a piece outside every unit, or
  the host's own tags.
- **The case's evidence**, the same for every arm: the documents ``case_documents`` reads off each case (a field
  holding a source, or a callable rendering one). It decides truth, and the candidate's own documents never do.
- **The candidate's own documents**, which it returns on :attr:`~threetears.evals.quick.answer.Answer.documents`:
  what it was given or fetched. ``grounded_in="run"`` reads them for the grounded share.
- **The web**, with ``search``: a port the grader checks a doubted claim on, within a budget (``web``), whose
  searches are recorded as the run's spend.
- **The run's own record**: what the run did beside its answer (an order log, a playlist), returned on
  :attr:`~threetears.evals.quick.answer.Answer.session`, read off a case field (``session="field"``), or rendered by
  a callable (``session=``) from the case and the :class:`~threetears.evals.quick.answer.Answer`. No grader, judge or
  prompt sees it and nothing in the engine reads it: only your own readings do.

**Readings of your own** (``readers``, each a :class:`ClaimReading`) count what the engine's readings cannot: a
claim the session record contradicts, a claim about an entity your tags name. Each reads a frozen
:class:`~threetears.evals.kernel.host.ClaimView` of one cell (its claims, each with its scope, verdict and segment,
the segments with your tags, and the session record) and returns a number, or ``None`` where the cell holds nothing
it counts. It ranks arms as the criterion's own readings do, under its name (``answer.claims_<reading>``).

Four short examples, each importing from the quick path's root while these names are provisional
(``from threetears.evals.quick import Claims, ClaimReading, Document``). The first grades each answer's claims against its case's
``source``::

    claims = Claims(model, case_documents="source")
    result = await Eval(cases, describe, [claims]).run()

The second ranks an arm that looked its answer up above one that recalled it, by what each read::

    async def look_up(case):
        return Answer(case["source"], documents=[Document(case["source"], title="archive")])

    async def recall(case):
        return Answer(remembered(case), documents=())  # it read nothing, and says so

    grounded = Claims(model, case_documents="source", grounded_in="run")
    result = await Eval(cases, {"recall": recall, "look-up": look_up}, [grounded], name="landmarks").run()

The third counts a reading of its own: claims naming an item the run's own order log does not hold::

    async def assistant(case):
        reply, log = await take_order(case)
        return Answer(reply, session={"ordered": log})  # beside the answer, never graded as part of it

    def unordered(view):
        \"\"\"World claims naming an item the order log does not hold.\"\"\"
        held = (view.session or {}).get("ordered", ())
        return sum(1 for claim in view.claims if claim.scope == "world" and not any(i in claim.text for i in held))

    claims = Claims(model, case_documents="menu", readers={"unordered": ClaimReading(unordered, better="lower")})

The fourth reads each cell's claims back after a run, with their verdicts and readings::

    for cell in result.claim_cells():
        if cell.view is not None:  # None where the criterion lost the cell
            print(cell.arm, cell.case, cell.readings, [claim.verdict for claim in cell.view.claims])

**The judge's client is required.** The model is an id, a :class:`~threetears.evals.providers.Model`, or any
completion client that names its model, read as :class:`~threetears.evals.quick.judged.Judge` reads one, and lent to
the run as a judge's is. An eval graded by a ``Judge`` and ``Claims`` grades both on one judge model, through
**the Judge's client**: the two must name one model, and this grader's own client is then never called, whatever
settings it was built with.

**Nothing to check against is refused.** With no ``case_documents``, no ``search`` and no ``grounded_in``, every
reading would read 0 whatever the answers say. With ``grounded_in="run"`` alone (a retrieval answer graded on its
own sources), the grounded share reads what the answers say and nothing decides truth: the result says its false
claims were not checked.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from threetears.evals.kernel.claims.derived import claim_view
from threetears.evals.kernel.claims.readings import claim_reading_names, claim_readings
from threetears.evals.kernel.host import ClaimView, CompletionClients, CompletionRole
from threetears.evals.kernel.metrics import CLAIMS_FAMILY, MetricDescriptor
from threetears.evals.schema import (
    ClaimCriterion,
    ClaimEvidence,
    CompletionClient,
    EvidenceDocument,
    EvidenceProvenance,
    PageBudget,
    RepairBudget,
    Segment,
    WebBudget,
    WebSearch,
)
from threetears.evals.schema.models import CLAIM_DOCUMENT_BYTES
from threetears.evals.quick.answer import Answer, Document
from threetears.evals.quick.judged import DIMENSION_CONTEXT, BorrowedJudgeClient, answer_text, judge_client

if TYPE_CHECKING:
    from threetears.evals.schema import EvalResult, EvalTrace

#: How the default cut is named in each cell's evidence: the whole answer, one unit.
_WHOLE_ANSWER = "whole-answer-v1"

#: How much of a cell's documents each verify step reads by default: up to 8,000 characters of one document and
#: 32,000 in all, in passages of 1,200 cut around what a claim names.
_DEFAULT_PAGES = PageBudget(per_document_chars=8000, total_chars=32000, window_chars=1200)

#: How much a step asks again by default before the cell is the criterion's loss: two rounds, eight items.
_DEFAULT_REPAIRS = RepairBudget(rounds=2, max_items_per_cell=8)

#: How far the web check goes by default once a search is given: two searches for each doubted claim, five claims a
#: cell, the first three results of each, 6,000 characters of a page read in windows of 1,500.
_DEFAULT_WEB = WebBudget(
    max_queries_per_claim=2, max_checked_per_cell=5, results_per_query=3, page_chars=6000, window_chars=1500
)

#: The provenances ``grounded_in`` names.
_PROVENANCES: tuple[EvidenceProvenance, ...] = ("case", "run")

#: Each better end a reading of your own may declare, as its measure states it (``higher_is_better``).
_BETTER_ENDS: Mapping[str, bool] = MappingProxyType({"lower": False, "higher": True})


@dataclass(frozen=True, slots=True)
class ClaimReading:
    """PROVISIONAL. A reading of your own off each cell's claims, declared as ``Claims(readers={name: ...})`` takes it:
    ``ClaimReading(unordered, better="lower")``.

    Everything after ``read`` is keyword-only, so it grows by adding declarations, as ``grader(...)`` does.

    :ivar read: Handed a frozen :class:`~threetears.evals.kernel.host.ClaimView` of one cell, it returns a number, or
        ``None`` where the cell holds nothing it counts (no reading, never a zero). The first line of its docstring
        describes the reading.
    :ivar better: Which end of it is better: ``"lower"`` or ``"higher"``. It ranks arms on that end.
    :ivar bounds: The lowest and highest value it can read, low first (a share is ``(0, 1)``); ``None`` for a count,
        which never reads below zero.
    :ivar revision: The revision of ``read``: raise it whenever what ``read`` counts changes while the rest stays as it
        is (it reads a field the view gains, or counts a claim it did not), so runs read under the two never pool.
    """

    read: Callable[[ClaimView], float | None]
    better: Literal["lower", "higher"] = field(kw_only=True)
    bounds: tuple[float, float] | None = field(default=None, kw_only=True)
    revision: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        """Refuse a reading no measure could carry.

        :raises ValueError: A ``read`` that is not callable; a better end other than ``lower`` and ``higher``; bounds
            that are not two numbers, low first; or a revision that is not a non-negative int.
        """
        if not callable(self.read):
            raise ValueError(f"ClaimReading(read) is a function of a cell's view, not {self.read!r}")
        if self.better not in _BETTER_ENDS:
            raise ValueError(f"ClaimReading(better=...) is 'lower' or 'higher', not {self.better!r}")
        bounds = self.bounds
        if bounds is not None and not (
            isinstance(bounds, tuple)
            and len(bounds) == 2
            and all(isinstance(end, int | float) and not isinstance(end, bool) for end in bounds)
            and bounds[0] < bounds[1]
        ):
            raise ValueError(f"ClaimReading(bounds=...) is two numbers, low first, not {bounds!r}")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError(f"ClaimReading(revision=...) is a non-negative int, not {self.revision!r}")

    @property
    def declaration(self) -> tuple[object, ...]:
        """What it declares, for telling two declarations of one reading apart: its better end, bounds, revision and
        ``read`` by module, qualified name and code, so a function written twice alike reads as one."""
        return (
            self.better,
            self.bounds,
            self.revision,
            getattr(self.read, "__module__", None),
            _named(self.read),
            getattr(self.read, "__code__", None) or self.read,
        )


class Claims:
    """The factual claims in each answer, listed and checked by the engine: a grader in an eval's ``grader`` list.

    :ivar client: The completion client the claims grader calls, lent to the run and never closed by it.
    :ivar model: The model ``client`` calls: the run's judge pin.
    :ivar criterion: The claim criterion the eval's template declares, as the engine grades it.
    :ivar unit: What one unit is called on a page (``answer``, ``entry``).
    :ivar segmentation: How each answer is cut, by name, as each cell's evidence records it.
    :ivar search: The web search a doubted claim is checked on, or ``None`` for no web.
    :ivar readings: PROVISIONAL. Each reading of your own, by its stored name (``answer.claims_unordered``), as
        declared. Empty when none is declared.
    :ivar derived: PROVISIONAL. The measure each reading of your own is, as the host built over this eval declares it
        (derived from :attr:`criterion`, in the ``claims`` family), in the order given.
    """

    def __init__(
        self,
        model: str | CompletionClient,
        name: str = "claims",
        *,
        unit: str = "answer",
        segments: Callable[[Mapping[str, Any], Any], Sequence[Segment | str]] | None = None,
        segmentation: str | None = None,
        case_documents: str | Callable[[Mapping[str, Any]], Sequence[Document | str]] | None = None,
        session: str | Callable[[Mapping[str, Any], Answer], Mapping[str, Any] | None] | None = None,
        readers: Mapping[str, ClaimReading] | None = None,
        grounded_in: str | Iterable[str] | None = None,
        guidance: str = "",
        search: WebSearch | None = None,
        web: WebBudget | None = None,
        pages: PageBudget = _DEFAULT_PAGES,
        repairs: RepairBudget = _DEFAULT_REPAIRS,
    ) -> None:
        """Declare the claims grader.

        :param model: What lists and checks the claims: a model id, a :class:`~threetears.evals.providers.Model`, or
            any completion client that names its model (``model_name``).
        :param name: The criterion's name, under which its readings are keyed; a bare one is stored under
            ``answer.`` (``claims`` is ``answer.claims``), as a judge's bare dimension is.
        :param unit: What one unit is called on a page: ``answer`` when each answer is one, ``entry`` for a list of
            entries. Presentation only: it changes nothing the run measures.
        :param segments: Cuts an answer into pieces: ``(case, answer) -> [...]``, each piece its text (a unit of its
            own, numbered in order) or a :class:`~threetears.evals.schema.Segment` (one unit spanning several pieces, a
            piece in no unit, the host's own tags). Return one kind or the other, not both. ``None`` grades the whole
            answer as one unit.
        :param segmentation: The cut's name, recorded on every cell; change it whenever the cut changes, since it
            sets every per-unit denominator. ``None`` names the default cut, or ``segments`` by its qualified name.
        :param case_documents: The case's evidence: a field of the case holding a document's text, a list of texts,
            or mappings with ``text`` (and ``title``, ``url``, ``id``, ``overrides``), or a callable rendering a case's
            documents. It decides truth for every arm alike. A document naming the ids of others in ``overrides``
            outranks them where the two disagree (:attr:`~threetears.evals.quick.answer.Document.overrides`). ``None``: the case holds none, so only the web (with ``search``) can find a
            claim false.
        :param session: PROVISIONAL. Where each cell's record of the run (an order log, a play order) is read from, for
            the readings of your own; it is stored with the cell's evidence as given. ``None``: what the runner
            returned on ``Answer.session``, or no record. A field name: the case's field holding a JSON object, the
            same for every arm. A callable ``(case, answer) -> {...}``: handed the :class:`Answer` the runner returned
            (a plain return as ``Answer(value)``), so it reads ``answer.value`` and ``answer.session``; ``None`` where
            a cell has none. No grader, judge or prompt sees it and the engine reads none of it: only ``readers`` do.
        :param readers: PROVISIONAL. Readings of your own, each a :class:`ClaimReading` by name. Each is stored as
            ``<name>_<reading>`` (``answer.claims_unordered``) and ranks arms as the criterion's own readings do.
            Launch reads each on two synthetic cells first and refuses one that raises there; one that raises on a
            graded cell leaves that cell's readings of your own unread, saying so. ``None``: none.
        :param grounded_in: Which documents a claim's support by reads as grounded: ``"run"``, what the candidate
            returned on ``Answer.documents``; ``"case"``, the case's own, for a candidate shown them (a summarizer's
            source). ``None``: no grounded share.
        :param guidance: Your notes on what counts as a claim in your domain and how to read one, with examples; sent
            with every prompt the grader is asked.
        :param search: A web search port (:class:`~threetears.evals.schema.WebSearch`) a claim no case evidence
            covers, that the grader doubts, is checked on; a recorded one
            (:class:`~threetears.evals.testing.RecordedWebSearch`) replays offline. ``None``: no web.
        :param web: How far the web check may go in one cell; ``None`` takes a generic budget (two searches for each
            doubted claim, five claims a cell) when ``search`` is given. Needs ``search``.
        :param pages: How much of a cell's documents each verify step reads.
        :param repairs: How much each step asks again before the cell is the criterion's loss.
        :raises ValueError: A client naming no model; a name, unit or guidance that is not text; a ``grounded_in``
            naming another provenance; a web budget with no search; a ``case_documents`` that is neither a field
            name nor a callable; nothing to check a claim against (no ``case_documents``, no ``search`` and no
            ``grounded_in``), whose every reading would read 0; ``grounded_in="case"`` with no ``case_documents``;
            a ``session`` that is neither a field name nor a callable; a reading of your own whose name is no
            identifier or is one of the criterion's own readings, or that is not a :class:`ClaimReading`; or a
            criterion the engine refuses.
        """
        self.client, self.model = judge_client(model, grader="Claims")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Claims(name=...) names the readings it keys, a non-blank string, not {name!r}")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Claims(unit=...) is what a page calls one unit, a non-blank string, not {unit!r}")
        if web is not None and search is None:
            raise ValueError(
                "Claims(web=...) bounds the web check, and no search was given to check on; pass search= a WebSearch "
                "port (a RecordedWebSearch replays one offline), or leave web out"
            )
        if case_documents is not None and not callable(case_documents):
            if not isinstance(case_documents, str) or not case_documents.strip():
                raise ValueError(
                    "Claims(case_documents=...) names the case's field holding its evidence, or is a callable "
                    f"rendering a case's documents, not {case_documents!r}"
                )
        if case_documents is None and search is None and grounded_in is None:
            raise ValueError(
                "Claims has nothing to check a claim against, so every reading would read 0 whatever the answers say: "
                "give it case_documents= (the case's own evidence, which decides truth), search= (a web search port, "
                "which checks a doubted claim), or grounded_in='run' (the documents each answer says it read)"
            )
        provenances = _provenances(grounded_in)
        if provenances is not None and "case" in provenances and case_documents is None:
            raise ValueError(
                "Claims(grounded_in='case') reads the case's own documents, and no case_documents= names them, so "
                "its grounded share would read 0 for every arm: name the case's field or render its documents"
            )
        if segments is not None and not callable(segments):
            raise ValueError(f"Claims(segments=...) is a callable (case, answer) -> pieces, not {segments!r}")
        if session is not None and not callable(session) and (not isinstance(session, str) or not session.strip()):
            raise ValueError(
                "Claims(session=...) names the case's field holding the run's own record, or is a callable "
                f"(case, answer) rendering it, not {session!r}"
            )
        self.unit = unit
        self.search = search
        #: Whether anything decides a claim's truth: the case's documents or the web. Without either, only the
        #: grounded share reads what the answers say.
        self.checks_truth = case_documents is not None or search is not None
        self._segments = segments
        self._case_documents = case_documents
        self._session = session
        self.segmentation = segmentation if segmentation is not None else _cut_name(segments)
        if not self.segmentation.strip():
            raise ValueError("Claims(segmentation=...) names the cut, and this one is blank")
        try:
            self.criterion = ClaimCriterion(
                name=_dim_name(name),
                guidance=guidance,
                grounded_in=provenances,
                web=(web or _DEFAULT_WEB) if search is not None else None,
                pages=pages,
                repairs=repairs,
                judge="builtin",
            )
        except ValidationError as refused:
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
            raise ValueError(f"Claims refused: {reasons}") from refused
        if readers is not None and not isinstance(readers, Mapping):
            raise ValueError(f"Claims(readers=...) maps each reading's name to a ClaimReading, not {readers!r}")
        declared = [_derived(self.criterion.name, key, given) for key, given in (readers or {}).items()]
        self.readings: Mapping[str, ClaimReading] = MappingProxyType(
            {descriptor.name: reading for descriptor, reading in declared}
        )
        self.derived: tuple[MetricDescriptor, ...] = tuple(descriptor for descriptor, _reading in declared)

    @property
    def declared_readings(self) -> tuple[tuple[object, ...], ...]:
        """PROVISIONAL. Every reading of your own as declared, by stored name: what tells two ``Claims`` of one name
        reading alike from two reading apart, whichever objects they are."""
        return tuple((name, reading.declaration) for name, reading in self.readings.items())

    def __repr__(self) -> str:
        """The grader, by model and criterion."""
        return f"Claims({self.model!r}, {self.criterion.name!r})"

    @property
    def name(self) -> str:
        """The criterion's stored name (``answer.claims``)."""
        return self.criterion.name

    @property
    def reads_run_documents(self) -> bool:
        """Whether its grounded share reads the candidate's own documents, which each answer must then declare."""
        return self.criterion.grounded_in is not None and "run" in self.criterion.grounded_in

    def clients(self) -> CompletionClients:
        """The host client factory a run graded by it is built with: its client, lent, for the judge role alone.

        :return: A factory answering the judge role on :attr:`model` with :attr:`client`.
        """

        def lend(role: CompletionRole, model: str | None, *, temperature: float | None = None) -> BorrowedJudgeClient:
            if role != "judge":
                raise ValueError(f"an eval graded by Claims calls no model in the {role!r} role but its grader's")
            if model is not None and model != self.model:
                raise ValueError(f"the run asked for judge model {model!r}, and Claims calls {self.model!r}")
            return BorrowedJudgeClient(self.client, model_name=self.model, temperature=temperature)

        return lend

    def evidence(
        self,
        case: Mapping[str, Any],
        answer: Any,
        documents: Sequence[Document] | None,
        *,
        returned: Answer | None = None,
    ) -> ClaimEvidence:
        """What a cell's claims are listed from and checked against: the answer cut, the case's documents and the
        candidate's own.

        :param case: The case, as the runner was handed it.
        :param answer: The answer, as the graders read it.
        :param documents: What the candidate returned on ``Answer.documents``; ``None`` where it said nothing.
        :param returned: The :class:`Answer` the runner returned, for its ``session``; ``None`` for a plain return.
        :return: The evidence.
        :raises ValueError: The cut, the case's documents or the session record raised, or rendered nothing usable;
            the case's field is missing or holds no document; the session record is no JSON object, or is read off a
            case field while the runner returned one too; or the documents run past what one cell may hold even cut
            down.
        """
        segments = self._cut(case, answer)
        held = [*self._case_held(case)]
        session = self._session_record(case, returned if returned is not None else Answer(answer))
        stored: tuple[EvidenceDocument, ...] | None = None
        if documents is not None or held:
            stored = (*_fitted(held, "case"), *_fitted([*(documents or ())], "run"))
        try:
            return ClaimEvidence(segmentation=self.segmentation, segments=segments, documents=stored, session=session)
        except ValidationError as refused:
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
            raise ValueError(f"the claims grader's evidence was refused: {reasons}") from refused

    def read(self, view: ClaimView) -> dict[str, float | bool]:
        """PROVISIONAL. Every reading of your own off one cell's view, by stored name: the claim reader the host built
        over this eval registers for :attr:`criterion` (:data:`~threetears.evals.kernel.host.ClaimReader`).

        :param view: The cell's claims, segments and session record, frozen.
        :return: Each reading whose function returned a value; one returning ``None`` is left out, as a reading the
            cell does not carry.
        :raises ValueError: A reading's function raised, naming the reading and the function.
        """
        readings: dict[str, float | bool] = {}
        for name, reading in self.readings.items():
            try:
                value = reading.read(view)
            # prawduct:ok-broad-except — a reading's function is the caller's code: raised again, naming the reading
            except Exception as raised:
                raise ValueError(
                    f"the reading {name!r} ({_named(reading.read)}) raised {type(raised).__name__}: {raised}"
                ) from raised
            if value is not None:
                readings[name] = value
        return readings

    def _session_record(self, case: Mapping[str, Any], returned: Answer) -> dict[str, Any] | None:
        """The run's own record for this cell, as ``session`` reads it; ``None`` where it names none or holds none.

        :raises ValueError: The case lacks the field, or the runner returned a record beside one read off the case;
            the callable raised; or what it gave is no mapping.
        """
        source = self._session
        if source is None:
            held = returned.session
        elif isinstance(source, str):
            if source not in case:
                raise ValueError(f"the claims grader reads the session record from {source!r}, which this case lacks")
            if returned.session is not None:
                raise ValueError(
                    f"the claims grader reads the session record from the case's {source!r}, and the runner returned "
                    "one on Answer.session too; read one: drop session=, or render both with a session= callable"
                )
            held = case[source]
        else:
            try:
                held = source(case, returned)
            # prawduct:ok-broad-except — session is the caller's code: what it raises is named, the case with it
            except Exception as raised:
                raise ValueError(f"the claims grader's session raised {type(raised).__name__}: {raised}") from raised
        if held is None:
            return None
        if not isinstance(held, Mapping):
            raise ValueError(
                f"the claims grader's session record is {held!r}; it is a JSON object (a mapping), or None for none"
            )
        return dict(held)

    def _cut(self, case: Mapping[str, Any], answer: Any) -> tuple[Segment, ...]:
        """The answer as its segments: the whole of it as one unit, or as ``segments`` cuts it."""
        if self._segments is None:
            return (Segment(text=answer_text(answer), unit=0),)
        try:
            pieces = self._segments(case, answer)
        # prawduct:ok-broad-except — segments is the caller's code: what it raises is named, the case with it
        except Exception as raised:
            raise ValueError(f"the claims grader's segments raised {type(raised).__name__}: {raised}") from raised
        if isinstance(pieces, str | bytes) or not isinstance(pieces, Sequence) or not pieces:
            raise ValueError(
                f"the claims grader's segments gave {pieces!r}; it returns the answer's pieces, at least one"
            )
        texts = [piece for piece in pieces if isinstance(piece, str)]
        if texts and len(texts) != len(pieces):
            raise ValueError(
                "the claims grader's segments returned text and Segment values together; return text alone (each a "
                "unit, in order) or Segment values alone (each stating its unit)"
            )
        if texts:
            try:
                return tuple(Segment(text=text, unit=index) for index, text in enumerate(texts))
            except ValidationError as refused:
                raise ValueError("the claims grader's segments returned a blank piece") from refused
        if not all(isinstance(piece, Segment) for piece in pieces):
            raise ValueError(f"the claims grader's segments returned {pieces!r}; each piece is text or a Segment")
        return tuple(piece for piece in pieces if isinstance(piece, Segment))

    def _case_held(self, case: Mapping[str, Any]) -> list[Document]:
        """The case's own documents, as ``case_documents`` reads them; none where it names none."""
        source = self._case_documents
        if source is None:
            return []
        if isinstance(source, str):
            if source not in case:
                raise ValueError(f"the claims grader reads the case's evidence from {source!r}, which this case lacks")
            return _documents_in(case[source], source)
        try:
            rendered = source(case)
        # prawduct:ok-broad-except — case_documents is the caller's code: what it raises is named, the case with it
        except Exception as raised:
            raise ValueError(f"the claims grader's case_documents raised {type(raised).__name__}: {raised}") from raised
        return _documents_in(rendered, "case_documents")


@dataclass(frozen=True, slots=True)
class ClaimCell:
    """PROVISIONAL. One cell's claims under a claim criterion, read back from the store: what
    :meth:`~threetears.evals.quick.contract.Result.claim_cells` returns, one per cell the criterion graded.

    A read-only copy: the claims are the frozen view a reading of your own is handed, never the stored grade, so the
    stored shapes may move under it. It grows by adding fields.

    :ivar arm: The arm, by its key in the result's ``arms``.
    :ivar case: The case's name, as every other line names it: its own ``id``, else its place in the list from 1.
    :ivar repeat: Which repeat of the case, from 1.
    :ivar criterion: The criterion's stored name (``answer.claims``).
    :ivar readings: Each reading the cell carries, by stored name: the criterion's own, then your own (``readers``).
        A reading whose population leaves the cell out is absent, never zero; a lost cell carries only
        ``<criterion>_lost``.
    :ivar view: Its claims, each with its scope, verdict, excerpt and segment, its segments and the session record;
        ``None`` where no grade is stored (the criterion lost the cell, or its trace is not yet written).
    :ivar lost: Why the criterion lost the cell (``cut_short``, ``cost_cap``, ...), ``trace_pending`` while its trace's
        write is not yet known (no loss), or ``None`` for a graded cell.
    :ivar unread: Why your own readings of this cell are unread (a reading's function raised on it), or ``None``.
    """

    arm: str
    case: str
    repeat: int
    criterion: str
    readings: Mapping[str, float]
    view: ClaimView | None
    lost: str | None = None
    unread: str | None = None


def claim_cell(arm: str, case: str, result: EvalResult, trace: EvalTrace | None, criterion: str) -> ClaimCell | None:
    """One stored cell's claims under ``criterion``, as :class:`ClaimCell` reads them; ``None`` for a cell the criterion
    never graded (one left out before its claims phase).

    :param arm: The arm, by its key.
    :param case: The case's name.
    :param result: The stored result.
    :param trace: Its stored trace, or ``None`` for one that stored none.
    :param criterion: The criterion's stored name.
    :return: The cell, or ``None``.
    """
    tally = result.claim_tallies.get(criterion)
    lost = result.claim_failures.get(criterion)
    outcome = tally if tally is not None else lost
    if outcome is None:
        return None
    grade = trace.claim_grades.get(criterion) if trace is not None else None
    evidence = trace.claim_evidence if trace is not None else None
    return ClaimCell(
        arm=arm,
        case=case,
        repeat=result.k_iteration,
        criterion=criterion,
        readings=MappingProxyType(claim_readings(criterion, outcome)),
        view=claim_view(grade, evidence) if grade is not None and evidence is not None and tally is not None else None,
        lost=lost,
        unread=tally.derived_unread if tally is not None else None,
    )


def unchecked_truth(criterion: str) -> str:
    """What a result says of a criterion that nothing decided the truth of: its false claims were not checked.

    :param criterion: The criterion's stored name.
    :return: The sentence.
    """
    return (
        f"False claims on {criterion} were not checked: no case evidence or web search decides a claim's truth, so "
        "they read 0 whatever the answers say; its grounded share reads what the answers say."
    )


def _derived(criterion: str, key: object, given: object) -> tuple[MetricDescriptor, ClaimReading]:
    """One reading of your own as the measure the host declares, beside its declaration.

    :raises ValueError: A name that is no identifier, or is one of the criterion's own readings; a declaration that is
        not a :class:`ClaimReading`; or a measure the engine refuses.
    """
    if not isinstance(key, str) or not key.isidentifier():
        raise ValueError(
            f"Claims(readers=...) names each reading by an identifier (unordered, session_contradictions), not {key!r}"
        )
    name = f"{criterion}_{key}"
    if name in claim_reading_names(criterion, grounded=True):
        raise ValueError(
            f"Claims(readers=...) names a reading {key!r}, and {name!r} is one of the criterion's own readings: name it "
            "for what it counts"
        )
    if not isinstance(given, ClaimReading):
        raise ValueError(
            f"Claims(readers={{{key!r}: ...}}) is a ClaimReading(read, better='lower' or 'higher'), not {given!r}"
        )
    read = given.read
    doc = inspect.getdoc(read) if inspect.isfunction(read) or inspect.ismethod(read) else None
    words = " ".join(key.replace("_", " ").split())
    try:
        descriptor = MetricDescriptor(
            name=name,
            reader_name=f"{words[:1].upper()}{words[1:]}",
            data_type="numeric",
            family=CLAIMS_FAMILY,
            transferability_class="judge_mediated",
            attribution_scope="end_to_end",
            description=doc.splitlines()[0]
            if doc
            else f"The {words} {_named(read)} reads off each cell's claims under {criterion}.",
            derived_from=criterion,
            higher_is_better=_BETTER_ENDS[given.better],
            nonnegative=given.bounds is None,
            value_range=None if given.bounds is None else (float(given.bounds[0]), float(given.bounds[1])),
            definition_revision=given.revision,
        )
    except ValidationError as refused:
        reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
        raise ValueError(f"Claims(readers={{{key!r}: ...}}) refused: {reasons}") from refused
    return descriptor, given


def _named(read: object) -> str:
    """A reading's function, by its qualified name."""
    named = getattr(read, "__qualname__", None)
    return named if isinstance(named, str) else type(read).__qualname__


def _dim_name(name: str) -> str:
    """``name`` as a stored criterion name: as given when it carries a context, under ``answer.`` when bare."""
    return name if "." in name else f"{DIMENSION_CONTEXT}.{name}"


def _cut_name(segments: Callable[..., object] | None) -> str:
    """The default name of a cut: the whole answer's, or a callable's qualified name."""
    if segments is None:
        return _WHOLE_ANSWER
    named = getattr(segments, "__qualname__", None) or getattr(segments, "__name__", None)
    return named if isinstance(named, str) and named.strip() else type(segments).__qualname__


def _provenances(grounded_in: str | Iterable[str] | None) -> tuple[EvidenceProvenance, ...] | None:
    """``grounded_in`` as the criterion stores it.

    :raises ValueError: A provenance other than ``case`` and ``run``, or none named.
    """
    if grounded_in is None:
        return None
    named = (grounded_in,) if isinstance(grounded_in, str) else tuple(grounded_in)
    if unknown := [given for given in named if given not in _PROVENANCES]:
        raise ValueError(
            f"Claims(grounded_in=...) names 'run' (the candidate's own documents) or 'case' (the case's), not "
            f"{', '.join(map(repr, unknown))}"
        )
    return tuple(given for given in _PROVENANCES if given in named)


def _documents_in(value: object, where: str) -> list[Document]:
    """The documents ``value`` holds: a text, a :class:`Document`, a mapping with ``text``, or a list of those.

    :raises ValueError: Anything else, or none at all.
    """
    listed = list(value) if isinstance(value, list | tuple) else [value]
    documents: list[Document] = []
    for given in listed:
        if isinstance(given, Document):
            documents.append(given)
        elif isinstance(given, str):
            documents.append(Document(given))
        elif isinstance(given, Mapping) and isinstance(given.get("text"), str):
            documents.append(
                Document(
                    given["text"],
                    title=given.get("title"),
                    url=given.get("url"),
                    id=given.get("id"),
                    overrides=given.get("overrides", ()),
                )
            )
        else:
            raise ValueError(
                f"the case's evidence in {where!r} holds {given!r}; a document is its text, or a mapping with 'text' "
                "(and 'title', 'url', 'id', 'overrides')"
            )
    if not documents:
        raise ValueError(f"the case's evidence in {where!r} holds no document")
    return documents


def _fitted(documents: list[Document], provenance: EvidenceProvenance) -> tuple[EvidenceDocument, ...]:
    """``documents`` as stored evidence, cut to fit their half of what one cell's documents may hold.

    The case's documents and the candidate's own each get half, apart, so what an arm fetched never cuts the case's
    evidence: truth reads the same case documents whatever an arm read. A cut document says so in its title. Each is
    cut to an equal share of the room its titles and addresses leave, and the share halves until they fit.

    :raises ValueError: Their titles and addresses alone, every text cut to one character, run past the half: no cut
        of the texts could fit them, so the cell is left out by name.
    """
    if not documents:
        return ()
    share = CLAIM_DOCUMENT_BYTES // 2
    whole = _stored(documents, [document.text for document in documents], provenance)
    if _bytes(whole) <= share:
        return whole
    fixed = _bytes(_stored(documents, [document.text[:1] for document in documents], provenance))
    if fixed > share:
        raise ValueError(
            f"the {len(documents)} {provenance} document(s) take {fixed} bytes beside their text, each cut to one "
            f"character, and they may hold {share}: give fewer documents, or shorter titles and addresses"
        )
    room = (share - fixed) // len(documents)
    while room >= 1:
        cut = _stored(documents, [_cut_bytes(document.text, room) for document in documents], provenance)
        if _bytes(cut) <= share:
            return cut
        room //= 2
    raise ValueError(f"the {provenance} documents could not be cut to fit the {share} bytes they may hold")


def _stored(
    documents: Sequence[Document], texts: Sequence[str], provenance: EvidenceProvenance
) -> tuple[EvidenceDocument, ...]:
    """``documents`` as stored evidence with ``texts``, each one cut saying so in its title, under the id it was given or
    numbered by its place, with the documents it overrides.

    :raises ValueError: A document the stored shape refuses (an override on a ``run`` document).
    """
    stored: list[EvidenceDocument] = []
    for index, (document, text) in enumerate(zip(documents, texts, strict=True)):
        title = document.title or document.url or f"{provenance} document {index + 1}"
        if text != document.text:
            title = f"{title} (cut to its first {len(text)} of {len(document.text)} characters)"
        try:
            stored.append(
                EvidenceDocument(
                    id=document.id or f"{provenance}-{index + 1}",
                    provenance=provenance,
                    title=title,
                    url=document.url,
                    text=text,
                    overrides=tuple(document.overrides),
                )
            )
        except ValidationError as refused:
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
            raise ValueError(
                f"the claims grader's {provenance} document {index + 1} was refused: {reasons}"
            ) from refused
    return tuple(stored)


def _bytes(stored: Sequence[EvidenceDocument]) -> int:
    """How many bytes ``stored`` take as the cell's cap counts them: UTF-8 JSON, compact."""
    dumped = [document.model_dump(mode="json") for document in stored]
    return len(json.dumps(dumped, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _cut_bytes(text: str, room: int) -> str:
    """``text`` cut to at most ``room`` bytes of UTF-8 JSON, at a character boundary; a cut text keeps one character."""
    if len(json.dumps(text, ensure_ascii=False).encode("utf-8")) <= room:
        return text
    kept = text.encode("utf-8")[: max(1, room // 2)].decode("utf-8", errors="ignore")
    return kept if kept.strip() else text[:1]


__all__ = ["ClaimCell", "ClaimReading", "Claims", "claim_cell", "unchecked_truth"]
