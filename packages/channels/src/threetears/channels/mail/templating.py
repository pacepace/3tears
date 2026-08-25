"""Substituting per-recipient values into a message the product authored.

**Products supply the content; this package owes a consistent way to fill it in.** What
that costs is small and the failure modes it prevents are not: a placeholder with no
value silently renders `Hello , open` into a live inbox, and a value carrying CRLF turns
a subject line into a second `Bcc` header.

**Deliberately not a template language, and the alternative was weighed.** The estate
already renders sandboxed Jinja templates (`threetears.agent.wake.webhook_adapter` uses
`jinja2.sandbox.SandboxedEnvironment` to turn a webhook payload into a task prompt), so
reusing it was the obvious first answer. Two things ruled it out. Adding `jinja2` to
`3tears-channels` changes this package's dependency metadata for every consumer that
only wants Slack; and a bulk-mail substitution surface actively wants to be weaker than
a template language -- there is no legitimate reason for an authored survey invitation
to evaluate an expression, and every reason for the thing filling in a respondent's name
to be incapable of it. A product that genuinely needs conditionals renders its own body
with whatever engine it likes and hands the result over as `body_text`.

Syntax, in full:

- ``{name}`` is replaced by ``values["name"]``. Names are ``[A-Za-z_][A-Za-z0-9_]*``.
- ``{{`` and ``}}`` produce a literal brace.
- Anything else, including a lone ``{`` that is not a well-formed placeholder, is left
  exactly as written.
- Substitution is ONE PASS: a value containing ``{other}`` lands as those characters and
  is never resolved again.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass

from threetears.channels.mail.message import EmailMessage

__all__ = [
    "EmailTemplate",
    "TemplateRenderError",
]


class TemplateRenderError(ValueError):
    """A template could not be rendered into a message that is safe to send.

    Every case it covers is a product bug that would otherwise reach a real inbox
    looking like a delivery problem: a missing value, a value of the wrong type, or a
    value that would break out of the header it was substituted into.
    """


#: One pass over the template: a literal brace pair, or a well-formed placeholder.
#: Anything else is not matched at all and therefore survives verbatim.
_TOKEN = re.compile(r"\{\{|\}\}|\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")

#: Characters that end a header and begin the next one. A value carrying any of these
#: in a header position is a header-injection attempt whatever its intent.
_HEADER_BREAKERS = ("\r", "\n", "\x00")


def _substitute(text: str, values: Mapping[str, str], *, escape_html: bool, is_header: bool, where: str) -> str:
    """Fill `text`'s placeholders from `values`, refusing anything unsafe.

    :param text: template text to render
    :ptype text: str
    :param values: substitution values, all of which must be strings
    :ptype values: Mapping[str, str]
    :param escape_html: whether each substituted value is HTML-escaped; true for the
        HTML part only, so the plain-text part keeps the characters as written
    :ptype escape_html: bool
    :param is_header: whether this text lands in a header, where a value carrying a
        line break would append headers of its own
    :ptype is_header: bool
    :param where: name of the field being rendered, for the error message
    :ptype where: str
    :return: rendered text
    :rtype: str
    :raises TemplateRenderError: a placeholder has no value, a value is not a string,
        or a value would break out of a header
    """

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group("name")
        if name not in values:
            raise TemplateRenderError(f"{where}: no value supplied for placeholder {name!r}")
        value = values[name]
        if not isinstance(value, str):
            raise TemplateRenderError(
                f"{where}: value for placeholder {name!r} is {type(value).__name__}, not a string. "
                "Format it at the call site, where the intended representation is known."
            )
        if is_header and any(breaker in value for breaker in _HEADER_BREAKERS):
            raise TemplateRenderError(
                f"{where}: value for placeholder {name!r} contains a line break, which would append "
                "headers of its own to the message."
            )
        return html.escape(value) if escape_html else value

    return _TOKEN.sub(_replace, text)


def _placeholders(text: str | None) -> set[str]:
    """Return every placeholder name `text` refers to.

    :param text: template text, or ``None`` for an absent part
    :ptype text: str | None
    :return: placeholder names
    :rtype: set[str]
    """
    if text is None:
        return set()
    return {match.group("name") for match in _TOKEN.finditer(text) if match.group("name") is not None}


@dataclass(frozen=True, slots=True)
class EmailTemplate:
    """An authored message with per-recipient placeholders still in it.

    :ivar subject: subject template; a header, so its values may not carry line breaks
    :ivar body_text: plain-text body template, always required
    :ivar body_html: optional HTML body template; its values are HTML-escaped
    :ivar list_unsubscribe_url: unsubscribe URL template, substituted per recipient
        because it carries that recipient's token
    :ivar list_unsubscribe_mailto: unsubscribe address template
    """

    subject: str
    body_text: str
    body_html: str | None = None
    list_unsubscribe_url: str | None = None
    list_unsubscribe_mailto: str | None = None

    def __post_init__(self) -> None:
        """Refuse an HTML part with no plain-text alternative.

        :return: nothing
        :rtype: None
        :raises TemplateRenderError: `body_html` is set and `body_text` is empty
        """
        if self.body_html is not None and not self.body_text:
            raise TemplateRenderError(
                "an HTML body needs a plain-text alternative: a recipient reading in plain text "
                "and every spam filter scoring the message both read the text part."
            )

    def placeholders(self) -> frozenset[str]:
        """Every placeholder name this template refers to.

        Exposed so a product can validate an authored template against the value set it
        will actually have, at authoring time, rather than discovering the mismatch on
        the send that fails.

        :return: placeholder names across every part
        :rtype: frozenset[str]
        """
        names: set[str] = set()
        for text in (self.subject, self.body_text, self.body_html):
            names |= _placeholders(text)
        for text in (self.list_unsubscribe_url, self.list_unsubscribe_mailto):
            names |= _placeholders(text)
        return frozenset(names)

    def render(self, *, to: str, values: Mapping[str, str]) -> EmailMessage:
        """Render this template for one recipient.

        :param to: recipient address
        :ptype to: str
        :param values: substitution values for this recipient
        :ptype values: Mapping[str, str]
        :return: a message ready to hand to a transport
        :rtype: EmailMessage
        :raises TemplateRenderError: a placeholder has no value, a value is not a
            string, or a value would break out of a header
        """
        return EmailMessage(
            to=to,
            subject=_substitute(self.subject, values, escape_html=False, is_header=True, where="subject"),
            body_text=_substitute(self.body_text, values, escape_html=False, is_header=False, where="body_text"),
            body_html=(
                None
                if self.body_html is None
                else _substitute(self.body_html, values, escape_html=True, is_header=False, where="body_html")
            ),
            list_unsubscribe_url=(
                None
                if self.list_unsubscribe_url is None
                else _substitute(
                    self.list_unsubscribe_url,
                    values,
                    escape_html=False,
                    is_header=True,
                    where="list_unsubscribe_url",
                )
            ),
            list_unsubscribe_mailto=(
                None
                if self.list_unsubscribe_mailto is None
                else _substitute(
                    self.list_unsubscribe_mailto,
                    values,
                    escape_html=False,
                    is_header=True,
                    where="list_unsubscribe_mailto",
                )
            ),
        )
