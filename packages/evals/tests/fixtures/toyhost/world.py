"""The toy host's world — one dimension in every registrable quadrant, over a real seeding path.

Its dimensions are invoice-extraction vocabulary the engine has never seen. What matters is not the domain but the **shapes** — one per row of the obligations table the kit
derives from, because a row with no fixture is a row nobody has run:

* two ``representable`` dimensions with machine reads — a run sets them and the extractor sees them;
* one ``representable`` dimension whose read is a **labeled** claim, so a round trip over it proves
  the plumbing and nothing about the world;
* one ``judge_only`` — a run sets it and the extractor never sees it, because the adjudicated
  vendor template is what the answer key is graded against rather than something shown to the
  subject. Legitimate, and the quadrant a template presuming it should be warned about;
* one ``witnessed`` — the extractor sees it and **no run controls it**. This is the quadrant no
  existing eval design names, and a fixture that cannot produce it proves nothing about the
  dangerous case. Its read is deliberately ``async``, because a witnessed dimension is typically
  live state behind a service, and one call path has to cover both kinds of host;
* one ``triggered(turn)`` — armed by a seed and brought into being by a condition the host can
  fire, so the kit can complete arm → fire → read;
* one ``triggered(event)`` — armed the same way and brought into being by something happening in the
  pipeline rather than by the clock: a payment hold the seed stages, which takes effect when the
  extraction is posted. The toy kind fires it at run time through its cell's world session, so a run —
  not only the kit — exercises a trigger, and the result records the firing;
* one ``triggered(human)`` — armed the same way and fired only by a person, which is an answer
  about what is representable rather than a failed declaration.

The fourth capability combination — neither seedable nor perceivable — has no fixture because it
cannot be registered. ``test_world_registry.py`` proves that refusal by trying it.

**Two dimensions are coupled, the way production state usually is, and the registry says so.** A
born-digital invoice (an e-invoice rendered straight to PDF) was never scanned, so the page reader measures
it clean whatever scan quality was asked for: ``scan_quality`` holds as seeded only beside a paper template.
A host in that position declares two things rather than having the coupling found as a defect. Its
**base world** is the world every conformance check starts from — here a paper template, because this
world starts on an e-invoice, where scan quality cannot move at all. Its **coherence handle** answers
whether it holds a composed world as stated, so the kit never seeds a skewed scan beside an e-invoice
and then reports the page reader's honest "clean" as a seed that did not take, or as one dimension
clobbering another. A coupling narrows which values a check draws and never switches a check off: a
coherence handle that left ``scan_quality`` no value but "clean" over the base world would fail its checks.

**The world is real, not a shape.** ``seed`` and ``read`` move an actual object, and
``subject_view`` renders from that same object, so perception A/B and round-trip are answering
questions about behaviour rather than about a dict somebody typed to match. A hand-assembled
fixture tests the reader, not the writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from threetears.evals.contracts.host import Triggered, WorldDimension, WorldRegistry

#: What the extractor is shown, per surface. ``perceived_by`` on a dimension names one of these.
_DOCUMENT_SURFACE = "document_header"
_OPERATOR_SURFACE = "operator_context"

#: What supplies each dimension. Two of them, so a subject built with one and not the other places
#: the same world differently — which is the run-time algebra's whole point, and cannot be exercised
#: by a fixture whose every dimension hangs off one thing.
_DOCUMENT_CARRIER = "page_reader"
_OPERATOR_CARRIER = "console"

#: Vendor templates whose invoices are born digital — rendered straight to PDF, never printed or scanned.
_BORN_DIGITAL_TEMPLATES = frozenset({"peppol-einvoice"})

#: The event trigger's condition: the extraction being posted to the payables ledger.
PAYMENT_HOLD_CONDITION = "extraction_posted"

#: The host's identity of the event each triggered dimension's seed handle arms — what the seed handle returns,
#: and so what a firing of the seed's own event names.
PAYMENT_HOLD_EVENT = "posting-hold"
OPERATOR_REVIEW_EVENT = "operator-review"
SUPERVISOR_REVIEW_EVENT = "supervisor-review"

#: The world every conformance check composes over: a paper template, so a scan has a quality to vary.
_BASE_WORLD = {"vendor_template": "acme-2019"}


#: Expressions a toy-host scenario presumes, one per class the vocabulary check tells apart.
#:
#: Every one is invoice vocabulary the engine has never seen, for the reason the dimensions are:
#: a check proven only over names the engine ships would prove that the engine agrees with itself.
TOY_RESOLVABLE_EXPRESSIONS: tuple[str, ...] = (
    'state.document_language == "de"',
    "state.operator_corrections.length >= 1",
    'any(it == "reprice" for it in state.operator_corrections)',
    "state.ingest_backlog > 10",
)
"""Paths a run can resolve — a plain dimension, a synthetic length, a path read through a
generator binding, and a witnessed dimension no run seeds. The last is deliberate: a *goal check*
may legitimately read state the experiment does not control, and a vocabulary check that refused
it would be answering the authoring gate's question instead of its own."""

TOY_JUDGE_ONLY_EXPRESSION = 'state.vendor_template == "acme-2019"'
"""A path onto the judge-only quadrant. It resolves: the vocabulary exists, the subject simply
never sees it. Whether a *precondition* should presume such a dimension is a different question,
asked where preconditions and goal checks are still told apart."""

TOY_UNRESOLVABLE_EXPRESSION = 'state.documnet_language == "de"'
"""The founding incident's postcondition half, one transposition wide. Evaluated, it resolves to
missing, every comparison against it is false, and the extractor is scored down for a language
nobody set. Resolved statically, it is a typo somebody can fix before a run costs anything."""


@dataclass(frozen=True)
class ToyWorldFaults:
    """Defects a real host acquires, switchable so every ``failed`` verdict has a fixture.

    Not test-only contrivances: each is a thing that happens to living code, and each is what the
    corresponding conformance check exists to catch. A kit whose failure paths are never exercised
    is a kit that reports ``passed`` for the same reason a broken smoke alarm is quiet.

    Every default is False, so the toy host is sound unless a test asks for one specific defect —
    and asking for one is how a test asserts the check fires in the direction it claims.
    """

    seeding_vendor_template_does_nothing: bool = False
    """The seeder wired to nothing: the write path was refactored away and the declaration stayed.

    Round-trip is the check that catches it, and it is the founding defect in its purest form —
    an instantiation nobody verified took.
    """

    document_header_drops_language: bool = False
    """The renderer deleted: nothing on the subject's surface carries the dimension any more.

    Perception A/B is the check that survives this refactor. Without it the declaration goes on
    saying "supported" indefinitely, because an optional read cannot fail.
    """

    operator_context_reads_the_processing_shift: bool = False
    """The subject perceives state no dimension declares — the founding incident's own shape.

    ``processing_shift`` is undeclared world state. A renderer that reaches it widens the variance
    of every run in a campaign, and only ambient isolation can find it without spending a run.
    """

    operator_context_shows_vendor_template: bool = False
    """A judge-only dimension leaking onto a subject surface: the answer key printed beside the work.

    The operator console starts naming the adjudicated template — a debugging aid left in. Nothing it
    was declared to carry stops moving, so perception A/B stays green; perception stillness is the
    check that catches it, because ``vendor_template`` is perceived by no surface at all.
    """

    document_header_shows_payment_hold: bool = False
    """A surface showing a dimension its ``perceived_by`` does not name.

    ``payment_hold`` is declared on the operator console; the document header starts appending it
    too. The console still moves with it (A/B green), so only perception stillness sees the header
    moving for a dimension that never declared it.
    """

    seeding_scan_quality_resets_language: bool = False
    """Two dimensions sharing underlying state, so composing their preconditions is a lie.

    Independence is the check that catches it, and it finds the incident nobody has had yet:
    nothing today composes preconditions, so nothing today would notice.
    """

    arming_payment_hold_names_no_event: bool = False
    """A triggered seed handle that arms its event and returns nothing to name it by.

    The value still arrives, so every other check stays green; only the round trip's arming half sees
    that no cell could tell the seed's firing from one the world made of its own on the dimension.
    """


@dataclass
class ToyWorld:
    """The mutable state the toy host's handles read and write.

    A plain object rather than a dict so a seeder that writes the wrong dimension fails loudly
    instead of inventing a key.
    """

    faults: ToyWorldFaults = ToyWorldFaults()
    """Which of :class:`ToyWorldFaults` this world is suffering from."""

    document_language: str = "en"
    scan_quality: str = "clean"
    vendor_template: str = "peppol-einvoice"
    """Most of this host's traffic is e-invoices, so a fresh world starts on a born-digital template."""
    handwriting_present: bool = False
    """Whether an annotator judged the invoice to bear handwriting. Read back as the annotator's
    claim rather than as a measurement, which is what ``evidence="labeled"`` declares."""

    ingest_backlog: int = 0
    """No run seeds this — it is whatever the ingestion queue happened to hold."""

    operator_corrections: list[str] = field(default_factory=list)
    """Corrections in force. Arrives on a turn: arming stages it, the condition applies it."""

    armed_corrections: list[str] = field(default_factory=list)
    """What the next ``operator_reviews_extraction`` will apply. Staged, not yet in the world."""

    payment_hold: str = "released"
    """Whether the invoice's payment is held. Arrives on an event: arming stages it, posting applies it."""

    armed_payment_hold: str = "released"
    """What the next ``extraction_posted`` will apply. Staged, not yet in the world."""

    supervisor_signoff: str = "pending"
    """Only a supervisor moves this off ``pending``, which is why no unattended run can
    instantiate it."""

    processing_shift: str = "day"
    """**Undeclared world state**, and deliberately so: this is what ambient perturbation moves.
    A subject that perceives it is perceiving something no dimension here speaks for."""

    @property
    def measured_scan_quality(self) -> str:
        """What the page reader measures: a born-digital page has no scan, so it reads clean whatever was seeded."""
        return "clean" if self.vendor_template in _BORN_DIGITAL_TEMPLATES else self.scan_quality

    def render_subject_view(
        self, *, surfaces: tuple[str, ...] = (_DOCUMENT_SURFACE, _OPERATOR_SURFACE)
    ) -> dict[str, str]:
        """What a subject with these surfaces attached perceives of this world.

        The registry's ``subject_view`` handle. Parameterized by attached surfaces because what a
        subject faces depends on what that subject has attached, and conformance varies a
        dimension against the surface its own ``perceived_by`` names.

        Args:
            surfaces: The surfaces attached for this subject.

        Returns:
            ``{surface: rendered text}`` for each attached surface. ``vendor_template`` and
            ``supervisor_signoff`` appear in neither, which is what makes them ``judge_only``.
            For ``vendor_template`` that is checked: the ``perception_stillness`` conformance check
            moves it and requires both surfaces to hold still. ``supervisor_signoff`` only a person
            can move, so the same check records it ``unavailable`` (``not_instantiable_unattended``)
            and its judge-only claim stays a declaration.
        """
        rendered: dict[str, str] = {}
        if _DOCUMENT_SURFACE in surfaces:
            parts = [f"scan={self.measured_scan_quality}", f"handwriting={self.handwriting_present}"]
            if not self.faults.document_header_drops_language:
                parts.insert(0, f"language={self.document_language}")
            if self.faults.document_header_shows_payment_hold:
                parts.append(f"hold={self.payment_hold}")
            rendered[_DOCUMENT_SURFACE] = " ".join(parts)
        if _OPERATOR_SURFACE in surfaces:
            text = (
                f"{self.ingest_backlog} documents queued ahead of this one; "
                f"{len(self.operator_corrections)} corrections in force; payment {self.payment_hold}"
            )
            if self.faults.operator_context_reads_the_processing_shift:
                text += f"; shift={self.processing_shift}"
            if self.faults.operator_context_shows_vendor_template:
                text += f"; template={self.vendor_template}"
            rendered[_OPERATOR_SURFACE] = text
        return rendered


def toyhost_world(
    *,
    optional_capabilities: bool = True,
    # The default is one shared instance, which is safe here and only here: ToyWorldFaults is a
    # frozen dataclass of bools, so no caller can mutate it into the next test's starting state.
    faults: ToyWorldFaults = ToyWorldFaults(),  # noqa: B008 — immutable, see above
) -> tuple[WorldRegistry, ToyWorld]:
    """A world registry bound to a fresh :class:`ToyWorld`.

    Built through the real registration path, so every refusal in
    :class:`~threetears.evals.contracts.host.world.WorldRegistry` runs over this fixture on every construction.
    A fresh state each call, so a test that seeds one cannot leak into the next through a
    module-level singleton.

    Args:
        optional_capabilities: When False, register the same dimensions with none of the optional
            host capabilities — no way to perturb the witnessed dimension, no way to fire the
            triggered one, no ambient perturbation. This is the *hosts-without* half of the
            obligations table, and it is a real shape rather than a degraded one:
            samsung-frame-art-loader will never be able to perturb its own television. What it
            must produce is ``unavailable`` records, never silence.
        faults: Which defects this world is suffering from. See :class:`ToyWorldFaults`.

    Returns:
        The registry and the world its handles move.
    """
    state = ToyWorld(faults=faults)

    def seed_scan_quality(value: str) -> None:
        state.scan_quality = value
        if state.faults.seeding_scan_quality_resets_language:
            state.document_language = "en"

    async def seed_vendor_template(value: str) -> None:
        """Async on purpose, and a WRITE — the read side alone proves half the claim.

        The contract rests on a caller being unable to tell a local call from one crossing a
        service boundary. A fixture whose only awaitable is a read leaves every seeding path
        proven synchronous, and the seeding path is the one this contract exists to hold.
        """
        if not state.faults.seeding_vendor_template_does_nothing:
            state.vendor_template = value

    async def read_ingest_backlog() -> int:
        """Async on purpose — a witnessed dimension is usually live state behind a service."""
        return state.ingest_backlog

    async def fire_operator_review_async() -> None:
        """The third awaitable shape: a FIRE handle across a boundary, which a real trigger is."""
        state.operator_corrections = list(state.armed_corrections)

    def fire_extraction_posted() -> None:
        """The event trigger's fire handle: posting the extraction applies whatever hold was staged."""
        state.payment_hold = state.armed_payment_hold

    def arm_operator_corrections(value: list[str]) -> str:
        state.armed_corrections = list(value)
        return OPERATOR_REVIEW_EVENT

    def arm_supervisor_signoff(value: str) -> str:
        state.supervisor_signoff = value
        return SUPERVISOR_REVIEW_EVENT

    def arm_payment_hold(value: str) -> str | None:
        state.armed_payment_hold = value
        return None if state.faults.arming_payment_hold_names_no_event else PAYMENT_HOLD_EVENT

    def perturb_processing_shift() -> None:
        state.processing_shift = "night" if state.processing_shift == "day" else "day"

    def holds(world: dict[str, Any]) -> list[str]:
        """The registry's coherence handle: why this host would not hold a composed world as stated.

        Called with every dimension's value. Returns one reason per coupling the world trips, and nothing
        when the host holds it — the same answer the page reader gives, stated before anything is seeded.
        """
        template, scan = world.get("vendor_template"), world.get("scan_quality")
        if template in _BORN_DIGITAL_TEMPLATES and scan not in (None, "clean"):
            return [f"a {template} invoice is born digital, so the page reader measures it clean, never {scan}"]
        return []

    bindings: dict[str, Any] = {
        "toy.subject_view": state.render_subject_view,
        "toy.seed_language": lambda value: setattr(state, "document_language", value),
        "toy.read_language": lambda: state.document_language,
        "toy.seed_scan_quality": seed_scan_quality,
        "toy.read_scan_quality": lambda: state.measured_scan_quality,
        "toy.seed_vendor_template": seed_vendor_template,
        "toy.read_vendor_template": lambda: state.vendor_template,
        "toy.seed_handwriting_present": lambda value: setattr(state, "handwriting_present", value),
        "toy.read_handwriting_present": lambda: state.handwriting_present,
        "toy.read_ingest_backlog": read_ingest_backlog,
        "toy.arm_operator_corrections": arm_operator_corrections,
        "toy.read_operator_corrections": lambda: list(state.operator_corrections),
        "toy.arm_supervisor_signoff": arm_supervisor_signoff,
        "toy.read_supervisor_signoff": lambda: state.supervisor_signoff,
        "toy.arm_payment_hold": arm_payment_hold,
        "toy.read_payment_hold": lambda: state.payment_hold,
        "toy.holds": holds,
    }
    if optional_capabilities:
        bindings["toy.perturb_ingest_backlog"] = lambda value: setattr(state, "ingest_backlog", value)
        bindings["toy.fire_operator_review"] = fire_operator_review_async
        bindings["toy.fire_extraction_posted"] = fire_extraction_posted
        bindings["toy.perturb_processing_shift"] = perturb_processing_shift

    registry = WorldRegistry(
        (
            WorldDimension(
                name="document_language",
                carrier=_DOCUMENT_CARRIER,
                schema={"type": "string", "enum": ["en", "de", "fr"]},
                matters=(
                    "Extraction scenarios presume the document is in a language the field labels "
                    "are written in; a German invoice against English label heuristics measures "
                    "the heuristics, not the extractor."
                ),
                seed="toy.seed_language",
                read="toy.read_language",
                perceived_by=(_DOCUMENT_SURFACE,),
            ),
            WorldDimension(
                name="scan_quality",
                carrier=_DOCUMENT_CARRIER,
                schema={"type": "string", "enum": ["clean", "skewed", "faint"]},
                matters=(
                    "A scenario probing OCR recovery presumes a degraded scan, and running it "
                    "against a clean one scores the extractor on a problem it was never given."
                ),
                seed="toy.seed_scan_quality",
                read="toy.read_scan_quality",
                perceived_by=(_DOCUMENT_SURFACE,),
            ),
            WorldDimension(
                name="handwriting_present",
                carrier=_DOCUMENT_CARRIER,
                schema={"type": "boolean"},
                matters=(
                    "Handwritten annotation is what a scenario probing manual-override handling "
                    "presumes, and whether a page carries any is an annotator's judgement rather "
                    "than anything the pipeline measures."
                ),
                seed="toy.seed_handwriting_present",
                read="toy.read_handwriting_present",
                evidence="labeled",
                perceived_by=(_DOCUMENT_SURFACE,),
            ),
            WorldDimension(
                name="vendor_template",
                carrier=_DOCUMENT_CARRIER,
                schema={"type": "string", "enum": ["peppol-einvoice", "acme-2019", "globex-2021"]},
                matters=(
                    "The adjudicated template is what a goal check grades the extracted fields "
                    "against. The extractor is shown pixels and never the label, which is the "
                    "point: telling it the answer would measure nothing."
                ),
                seed="toy.seed_vendor_template",
                read="toy.read_vendor_template",
                # No perceived_by — this is the judge-only quadrant, deliberately.
            ),
            WorldDimension(
                name="ingest_backlog",
                carrier=_OPERATOR_CARRIER,
                schema={"type": "integer", "minimum": 0},
                matters=(
                    "The extractor is told how much work is queued behind it and demonstrably "
                    "trades thoroughness for speed when that number is high — so a campaign that "
                    "does not control it is varying its own stimulus without saying so."
                ),
                # No seed handle: the ingestion queue is whatever it is. Witnessed, not broken.
                read="toy.read_ingest_backlog",
                perturb="toy.perturb_ingest_backlog" if optional_capabilities else None,
                perceived_by=(_OPERATOR_SURFACE,),
            ),
            WorldDimension(
                name="operator_corrections",
                carrier=_OPERATOR_CARRIER,
                schema={"type": "array", "items": {"type": "string"}},
                matters=(
                    "A scenario probing whether the extractor defers to a human correction "
                    "presumes the correction arrives mid-run, and one applied before the first "
                    "turn measures obedience to a starting state instead."
                ),
                seed="toy.arm_operator_corrections",
                read="toy.read_operator_corrections",
                perceived_by=(_OPERATOR_SURFACE,),
                when=Triggered(
                    kind="turn",
                    condition="operator_reviews_extraction",
                    fire="toy.fire_operator_review" if optional_capabilities else None,
                ),
            ),
            WorldDimension(
                name="payment_hold",
                carrier=_OPERATOR_CARRIER,
                schema={"type": "string", "enum": ["released", "held"]},
                matters=(
                    "A scenario probing whether the extractor flags a held payment presumes the hold takes "
                    "effect when the extraction is posted, as it does in production; one in force from the "
                    "first turn measures reading a status line instead."
                ),
                seed="toy.arm_payment_hold",
                read="toy.read_payment_hold",
                perceived_by=(_OPERATOR_SURFACE,),
                when=Triggered(
                    kind="event",
                    condition=PAYMENT_HOLD_CONDITION,
                    fire="toy.fire_extraction_posted" if optional_capabilities else None,
                ),
            ),
            WorldDimension(
                name="supervisor_signoff",
                carrier=_OPERATOR_CARRIER,
                schema={"type": "string", "enum": ["pending", "approved", "rejected"]},
                matters=(
                    "Whether a supervisor cleared the batch is what a goal check reads to decide "
                    "if the pipeline escalated correctly, and no automation can produce it — "
                    "which is a fact about the scenario rather than a hole in the rig."
                ),
                seed="toy.arm_supervisor_signoff",
                read="toy.read_supervisor_signoff",
                # No perceived_by: the extractor is never told the batch was signed off.
                when=Triggered(kind="human", condition="supervisor_reviews_batch"),
            ),
        ),
        bindings=bindings,
        subject_view="toy.subject_view",
        perturb_ambient="toy.perturb_processing_shift" if optional_capabilities else None,
        base_world=_BASE_WORLD,
        coherence="toy.holds",
    )
    return registry, state


__all__ = [
    "OPERATOR_REVIEW_EVENT",
    "PAYMENT_HOLD_CONDITION",
    "PAYMENT_HOLD_EVENT",
    "SUPERVISOR_REVIEW_EVENT",
    "TOY_JUDGE_ONLY_EXPRESSION",
    "TOY_RESOLVABLE_EXPRESSIONS",
    "TOY_UNRESOLVABLE_EXPRESSION",
    "ToyWorld",
    "ToyWorldFaults",
    "toyhost_world",
]
