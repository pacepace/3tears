"""How a host's output looks — a bounded contract, deliberately with no prose in it.

**The hard rule: no free-text style prose ever reaches the generator prompt.** This repo has
proven that prose becomes instruction — a "tone" sentence handed to a model is an instruction it
follows, and a host that can write one can steer the analysis rather than style it.

That is enforced structurally rather than promised:

1. :attr:`StyleProfile.tone_register` is an **enum**, and the words it maps to are engine-owned
   prompt fragments. The host picks from a list; it never writes the instruction.
2. **Neither non-enum field can carry an instruction to a model.** ``locale`` is a ``str``, so it
   is validated against a BCP47 pattern at construction. ``vega_config`` is an arbitrary nested
   dict of host strings and is deliberately *not* pattern-checked — it is a Vega-Lite theme and
   constraining its shape here would fork the Vega schema. What holds instead is structural:
   :func:`prompt_fragment` is the only function in this module that returns prompt text, it takes
   no argument but the register, and :func:`assert_no_style_text` proves no value from either
   field reaches a given prompt. A field that cannot reach the prompt builder does not need its
   contents constrained.
3. **No field here is read by the prompt builder.** :func:`prompt_fragment` is the single
   function that turns any of this into prompt text, and it reads exactly one enum.
4. :func:`assert_no_style_text` asserts a prompt carries no token traceable to a host free-text
   field, with **descriptor prose and case text** the deliberate exceptions — those are
   *substance*, the domain words riding in the data, not style. ``tests/test_host_contract.py``
   holds the narrow half (the one function turning style into prompt text reads an engine-owned
   table); no test in this repository yet runs it over an assembled generator prompt. It is a
   TEST and not a production gate on purpose: a
   Vega-Lite value is routinely an ordinary English word ("right", "center", "none"), so a
   substring scan over an assembled prompt has an unbounded false-positive rate against arbitrary
   host config, and gating a billed generation on it would discard real analyses. Point 3 is what
   production actually rests on.

**Style never touches substance.** The memo shape, ordering, gating semantics, truth gates,
posture model, caveat attachment and the materiality mechanism are engine-owned and identical for
every consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

#: The registers a host may pick from. Engine-owned and closed — the point of the enum is that a
#: host cannot write its own.
ToneRegister = Literal["neutral", "executive", "technical"]

#: A BCP47 language tag, shape-checked. ``locale`` is the module's one non-enum field, so without
#: this it is a bare ``str`` in a module whose headline promise is that it has no free text — and
#: a promise with one untyped hole is the shape a reviewer stops checking. The pattern is
#: deliberately narrow: two or three letters, an optional script, an optional region. It rejects
#: a sentence, which is the threat, and it is not a registry lookup, which would be a dependency.
_BCP47_RE = re.compile(r"^[a-z]{2,3}(-[A-Z][a-z]{3})?(-([A-Z]{2}|\d{3}))?$")


class StyleError(ValueError):
    """A style profile contradicts the bounded contract this module promises."""


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

    Every field is either an engine-owned enum or a value the renderer consumes. There is
    deliberately **no** string field a host can fill with instructions.
    """

    tone_register: ToneRegister = "neutral"
    """Which engine-owned register to write in. The host picks; the engine supplies the words."""

    locale: str = "en-US"
    """BCP47 tag driving number, date and list formatting in the renderer. Shape-checked."""

    vega_config: dict[str, Any] = field(default_factory=dict)
    """Vega-Lite theme and palette. Consumed by the chart renderer; never serialised into a prompt."""

    def __post_init__(self) -> None:
        """Refuse a locale that is not a language tag.

        Raises:
            StyleError: ``locale`` does not have the shape of a BCP47 tag — which is how a
                sentence would get into the one field here that is not an enum.
        """
        if not _BCP47_RE.match(self.locale):
            raise StyleError(f"locale {self.locale!r} is not a BCP47 tag — this contract carries no free text")


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


def assert_no_style_text(prompt: str, style: StyleProfile) -> None:
    """Raise when a host-supplied style **string** appears verbatim in ``prompt``.

    The structural half of the no-style-text-in-prompts promise, as a callable rather than a habit. ``vega_config`` is an
    arbitrary nested dict, so "no style text reaches the prompt" cannot be enforced by typing it —
    it is enforced by walking whatever the host actually supplied and proving none of it is there.

    **Scoped to the two fields a host can fill freely**, which is what makes it safe to point at a
    whole assembled prompt rather than only at :func:`prompt_fragment`'s output: ``locale``, which
    is merely shape-checked, and ``vega_config``, which could not be typed. The closed enums are
    out — see :func:`_host_supplied_strings` for why matching them reports the engine's own words
    as a leak.

    Args:
        prompt: The assembled prompt text to check.
        style: The profile whose values must not appear in it.

    Raises:
        StyleError: A host-supplied style value is present in the prompt, naming the value.
    """
    leaked = [value for value in _host_supplied_strings(style) if value and value in prompt]
    if leaked:
        raise StyleError(f"host style values reached the prompt: {sorted(leaked)} — this contract carries no free text")


def _host_supplied_strings(style: StyleProfile) -> list[str]:
    """Every string a host WROTE into ``style``, at any depth of ``vega_config``.

    Args:
        style: The profile to walk.

    Returns:
        The host-supplied strings — ``locale`` and every string in ``vega_config``, which is
        exactly the pair of fields this module's promise is scoped to: the one that is only
        shape-checked, and the one that could not be typed at all.

        ``tone_register`` is deliberately absent: it is a CLOSED engine-owned enum, so a host picks
        a key from a list and cannot hold a sentence in it, and a value that cannot carry an
        instruction is not what this check is for. The engine-owned tone fragment is absent for
        the opposite reason — it is the one thing that is *supposed* to reach a prompt.

        ``vega_config``'s dict KEYS are absent too, and this one is worth being exact about
        because they used to be walked. A key in a Vega-Lite theme is a SLOT NAME out of Vega's
        own schema — ``background``, ``range``, ``category`` — not something the host composed,
        and all three of those occur in the engine's generator prompt as ordinary English. Walking
        them makes a host reporting Vega's vocabulary indistinguishable from a host leaking its
        own words, and it fails on the first realistic theme. The residual, said plainly rather
        than left implied: a host that invented a key out of free text and a serializer that
        emitted keys without values would slip past. No serializer does that — anything writing a
        config into text writes its values — and a check that cries leak on every real theme is
        one nobody keeps.
    """
    found = [style.locale]

    def walk(node: Any) -> None:
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(style.vega_config)
    return found


__all__ = [
    "StyleError",
    "StyleProfile",
    "ToneRegister",
    "assert_no_style_text",
    "prompt_fragment",
]
