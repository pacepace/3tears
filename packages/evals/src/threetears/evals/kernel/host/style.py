"""How a host's output reads — a bounded contract, deliberately with no prose in it.

**The hard rule: no free-text style prose ever reaches the generator prompt.** This repo has
proven that prose becomes instruction — a "tone" sentence handed to a model is an instruction it
follows, and a host that can write one can steer the analysis rather than style it.

That is enforced structurally rather than promised: :attr:`StyleProfile.tone_register` is an
**enum**, and the words it maps to are engine-owned prompt fragments. The host picks from a list; it
never writes the instruction. :func:`prompt_fragment` is the only function that turns any of this
into prompt text, and it reads exactly that one enum.

**Nothing about how a report looks is declared here.** The report is one page the package draws itself
(:mod:`threetears.evals.analysis.report`), legible in light and dark through its own stylesheet, with no
chart library and no host palette or font; a host that wants its own pictures reads the evidence bundle the
page was computed from. A locale, units, a date format and a length budget are not slots here either: no
renderer and no number formatter reads one, and a field that claims to change formatting while changing
nothing is worse than no field.

**Style never touches substance.** The memo shape, ordering, gating semantics, truth gates,
posture model, caveat attachment and the materiality mechanism are engine-owned and identical for
every consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: The registers a host may pick from. Engine-owned and closed — the point of the enum is that a
#: host cannot write its own.
ToneRegister = Literal["neutral", "executive", "technical"]


#: How many categorical colour slots a chart may assign with validated separation.
#:
#: Part of eval's chart vocabulary: a chart intent names colour SLOTS, never colours. Slots 1-4 are the ones a
#: chart may rely on to tell levels apart; past :data:`SERIES_SLOTS` a chart recycles from slot 1.
VALIDATED_SLOTS = 4

#: How many categorical colour slots a chart has before it recycles — the width of the vocabulary.
SERIES_SLOTS = 8

#: The engine-owned prompt fragment each register maps to. **This mapping is the reason
#: ``tone_register`` can be safe:** the host picks a key and the engine supplies the words, so no
#: host sentence ever reaches a model. Editing these is an engine change, reviewed as one.
_TONE_FRAGMENTS: dict[ToneRegister, str] = {
    "neutral": "Write plainly. State what the evidence supports and no more.",
    "executive": "Lead with the decision. Keep supporting detail to what changes that decision.",
    "technical": "Prefer precise mechanism over summary. Name the measure behind every claim.",
}


@dataclass(frozen=True)
class StyleProfile:
    """One host's bounded presentation contract.

    Every field is an engine-owned enum. There is deliberately **no** string field a host can fill with
    instructions, and no field that nothing reads.
    """

    tone_register: ToneRegister = "neutral"
    """Which engine-owned register to write in. The host picks; the engine supplies the words."""


def prompt_fragment(style: StyleProfile) -> str:
    """The only text this module contributes to a generator prompt.

    One function, reading one enum, so "does host style reach the prompt" has a single call site
    to inspect rather than being a property of how carefully each caller behaved.

    Args:
        style: The host's style profile.

    Returns:
        The engine-owned fragment for the profile's register. Never host-supplied text.
    """
    return _TONE_FRAGMENTS[style.tone_register]


__all__ = [
    "SERIES_SLOTS",
    "VALIDATED_SLOTS",
    "StyleProfile",
    "ToneRegister",
    "prompt_fragment",
]
