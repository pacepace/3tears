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

Two short examples. The first grades each answer's claims against its case's ``source``::

    claims = Claims(model, case_documents="source")
    result = await Eval(cases, describe, [claims]).run()

The second ranks an arm that looked its answer up above one that recalled it, by what each read::

    async def look_up(case):
        return Answer(case["source"], documents=[Document(case["source"], title="archive")])

    async def recall(case):
        return Answer(remembered(case), documents=())  # it read nothing, and says so

    grounded = Claims(model, case_documents="source", grounded_in="run")
    result = await Eval(cases, {"recall": recall, "look-up": look_up}, [grounded], name="landmarks").run()

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

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from threetears.evals.kernel.host import CompletionClients, CompletionRole
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
from threetears.evals.quick.answer import Document
from threetears.evals.quick.judged import DIMENSION_CONTEXT, BorrowedJudgeClient, answer_text, judge_client

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


class Claims:
    """The factual claims in each answer, listed and checked by the engine: a grader in an eval's ``grader`` list.

    :ivar client: The completion client the claims grader calls, lent to the run and never closed by it.
    :ivar model: The model ``client`` calls: the run's judge pin.
    :ivar criterion: The claim criterion the eval's template declares, as the engine grades it.
    :ivar unit: What one unit is called on a page (``answer``, ``entry``).
    :ivar segmentation: How each answer is cut, by name, as each cell's evidence records it.
    :ivar search: The web search a doubted claim is checked on, or ``None`` for no web.
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
            or mappings with ``text`` (and ``title``, ``url``), or a callable rendering a case's documents. It decides
            truth for every arm alike. ``None``: the case holds none, so only the web (with ``search``) can find a
            claim false.
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
            or a criterion the engine refuses.
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
        self.unit = unit
        self.search = search
        #: Whether anything decides a claim's truth: the case's documents or the web. Without either, only the
        #: grounded share reads what the answers say.
        self.checks_truth = case_documents is not None or search is not None
        self._segments = segments
        self._case_documents = case_documents
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

    def evidence(self, case: Mapping[str, Any], answer: Any, documents: Sequence[Document] | None) -> ClaimEvidence:
        """What a cell's claims are listed from and checked against: the answer cut, the case's documents and the
        candidate's own.

        :param case: The case, as the runner was handed it.
        :param answer: The answer, as the graders read it.
        :param documents: What the candidate returned on ``Answer.documents``; ``None`` where it said nothing.
        :return: The evidence.
        :raises ValueError: The cut or the case's documents raised, or rendered nothing usable; the case's field is
            missing or holds no document; or the documents run past what one cell may hold even cut down.
        """
        segments = self._cut(case, answer)
        held = [*self._case_held(case)]
        if documents is None and not held:
            return ClaimEvidence(segmentation=self.segmentation, segments=segments, documents=None)
        case_documents = _fitted(held, "case")
        run_documents = _fitted([*(documents or ())], "run")
        try:
            return ClaimEvidence(
                segmentation=self.segmentation, segments=segments, documents=(*case_documents, *run_documents)
            )
        except ValidationError as refused:
            reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
            raise ValueError(f"the claims grader's evidence was refused: {reasons}") from refused

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


def unchecked_truth(criterion: str) -> str:
    """What a result says of a criterion that nothing decided the truth of: its false claims were not checked.

    :param criterion: The criterion's stored name.
    :return: The sentence.
    """
    return (
        f"False claims on {criterion} were not checked: no case evidence or web search decides a claim's truth, so "
        "they read 0 whatever the answers say; its grounded share reads what the answers say."
    )


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
            documents.append(Document(given["text"], title=given.get("title"), url=given.get("url")))
        else:
            raise ValueError(
                f"the case's evidence in {where!r} holds {given!r}; a document is its text, or a mapping with 'text' "
                "(and 'title', 'url')"
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
    """``documents`` as stored evidence with ``texts``, each one cut saying so in its title."""
    stored: list[EvidenceDocument] = []
    for index, (document, text) in enumerate(zip(documents, texts, strict=True)):
        title = document.title or document.url or f"{provenance} document {index + 1}"
        if text != document.text:
            title = f"{title} (cut to its first {len(text)} of {len(document.text)} characters)"
        stored.append(
            EvidenceDocument(
                id=f"{provenance}-{index + 1}", provenance=provenance, title=title, url=document.url, text=text
            )
        )
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


__all__ = ["Claims", "unchecked_truth"]
