"""Message templates for the app's notifications, with the plural forms English needs.

A template is text with fields in braces:

- ``{name}`` is replaced by the value ``name`` (a number is written with thousands commas: ``1,204``);
- ``{name|one|other}`` picks a form by the number ``name``: ``one`` when it is exactly 1, ``other`` for anything else;
- ``{name|zero|one|other}`` adds a form for exactly 0;
- inside a form, ``#`` stands for the number itself, written as above: ``{count|one file|# files}``.

A field whose name has no value is left in the text exactly as written, braces and all, so a missing value is caught
in review rather than sent as a blank. There are no nested braces; a literal brace is written ``{{`` or ``}}``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_FIELD = re.compile(r"\{(.+)\}")
_OPEN, _CLOSE = "\x00", "\x01"


class TemplateError(ValueError):
    """A plural field with too few or too many forms, or a plural form chosen by a value that is not a number."""


def number(value: object) -> str:
    """A value as a message writes it: whole numbers with thousands commas, anything else as it is."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


def choose(value: object, forms: list[str]) -> str:
    """The plural form for ``value`` among ``forms`` (``[one, other]`` or ``[zero, one, other]``)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TemplateError(f"a plural form needs a number, not {value!r}")
    if len(forms) == 2:
        zero, (one, other) = None, forms
    elif len(forms) == 3:
        zero, one, other = forms
    else:
        raise TemplateError(f"a plural field takes two or three forms, not {len(forms)}")
    if value == 0 and zero is not None:
        return zero
    return one if value == 1 else other


def render(template: str, values: Mapping[str, object]) -> str:
    """``template`` with its fields filled from ``values``."""

    def field(match: re.Match[str]) -> str:
        name, *forms = match.group(1).split("|")
        name = name.strip()
        if name not in values:
            return match.group(0)
        if not forms:
            return number(values[name])
        return choose(values[name], forms).replace("#", number(values[name]))

    text = template.replace("{{", _OPEN).replace("}}", _CLOSE)
    return _FIELD.sub(field, text).replace(_OPEN, "{").replace(_CLOSE, "}")
