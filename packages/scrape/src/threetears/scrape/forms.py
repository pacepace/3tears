"""Deterministic HTML form parsing + postback-body serialization (no browser, no LLM).

Many server-side portals return their data only after a form POST: a plain GET yields the search
form, and the rows come back when the form's hidden + default fields are posted together with a submit
control. The classic case is ASP.NET WebForms -- a GET returns only the ``__VIEWSTATE`` /
``__EVENTVALIDATION`` search form -- but the shape is general (any server-rendered search form).

A browser scrape can drive such a form by clicking; this module is the deterministic, **browser-free**
counterpart. It parses a form into its POST target, its default field values (the WHATWG "successful
controls" a browser would submit for the untouched form), and its triggers, and serializes a postback
body. The caller composes it with an HTTP client (e.g. an :mod:`threetears.scrape` driver /
``threetears.core`` http client) to replay the POST, and with an extractor to read the result -- no
browser needed for the classic stateless case.

**Two kinds of trigger, because pages use both.** A form may submit via a named submit control
(``name=value``) or via ``__doPostBack(target, argument)`` on an anchor or script-driven element, which
sets the ``__EVENTTARGET`` / ``__EVENTARGUMENT`` hidden fields and submits the same form. The second
kind carries no ``name=value``, so a parser that only collects submit controls reports such a form as
having no trigger and a replay concludes -- wrongly -- that it cannot be driven without a browser. Both
are parsed (:attr:`HtmlForm.submit_controls`, :attr:`HtmlForm.event_triggers`) and both are serializable
(:func:`build_form_post`); which one produces data stays the caller's policy.

Domain-agnostic by design (the ``scrape`` charter -- no hardcoded framework or field meaning):
``require_field`` is a general knob (pass ``"__VIEWSTATE"`` to select an ASP.NET WebForms form), and
:func:`build_form_post` takes caller ``overrides`` so search criteria / option choices stay the
caller's policy, never assumed here. Pure and side-effect-free -- parsing only, no network, no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from bs4.element import Tag

__all__ = ["EventTrigger", "FormControl", "HtmlForm", "build_form_post", "parse_form"]

#: ``__doPostBack('<target>', '<argument>')`` as written in an ``href``/``onclick`` handler. Both
#: arguments are matched with either quote style; the target is the control that raised the event.
_DO_POSTBACK_RE = re.compile(r"""__doPostBack\(\s*['"]([^'"]*)['"]\s*,\s*['"]([^'"]*)['"]\s*\)""")

#: Attributes that can carry a ``__doPostBack`` call. Read from the PARSED tag, never the raw markup:
#: a served page writes the call HTML-escaped (``__doPostBack(&#39;x&#39;,&#39;&#39;)``), so a regex
#: over the bytes silently finds nothing while the browser sees a working trigger.
_EVENT_ATTRS = ("href", "onclick", "onchange")

#: The hidden fields an ASP.NET postback carries the raised event in. A browser sets these from
#: ``__doPostBack`` before submitting; a browser-free replay writes them directly.
EVENT_TARGET_FIELD = "__EVENTTARGET"
EVENT_ARGUMENT_FIELD = "__EVENTARGUMENT"

#: ``<input>`` types that TRIGGER a submission rather than carry a data value (a browser submits the
#: activated one's ``name=value``). ``image`` submits like ``submit`` (plus click coordinates a
#: deterministic replay omits).
_SUBMIT_INPUT_TYPES = frozenset({"submit", "image"})

#: ``<input>`` types excluded from the default "successful controls" body -- the submit-like triggers
#: (added explicitly by the caller, one at a time) and ``reset`` (never submitted).
_NON_DATA_INPUT_TYPES = frozenset({"submit", "image", "reset", "button"})


@dataclass(frozen=True)
class FormControl:
    """One named form control -- the ``name=value`` a submission carries for it.

    :param name: the control's ``name`` attribute (a control with no ``name`` is never "successful")
    :param value: the value submitted for it (a submit control's label/value; a field's default value)
    """

    name: str
    value: str


@dataclass(frozen=True)
class EventTrigger:
    """A postback raised by ``__doPostBack(target, argument)`` rather than by a named submit control.

    The second way a server-rendered form submits, and on many portals the ONLY way: the control is an
    anchor or a script-driven element carrying no ``name=value``, so it never appears among
    :attr:`HtmlForm.submit_controls`, and a replay that only knows about submit controls concludes the
    form has no trigger at all. What the browser actually does is set the two event fields and submit
    the same form, which needs no browser -- :func:`build_form_post` writes them directly.

    :param target: the raising control's identifier -- the value posted as ``__EVENTTARGET``
    :param argument: the event's argument -- posted as ``__EVENTARGUMENT`` (commonly empty; carries
        e.g. a page number for a pager control)
    :param label: the trigger element's visible text, whitespace-collapsed (``""`` when it has none) --
        how a caller tells "Search" from "Clear" without hardcoding control-name conventions
    """

    target: str
    argument: str = ""
    label: str = ""


@dataclass(frozen=True)
class HtmlForm:
    """A parsed HTML form reduced to what a browser-free postback replay needs.

    :param action_url: the absolute POST target (the form ``action`` resolved against the page URL; the
        page URL itself when ``action`` is empty/absent -- a self-posting form, the ASP.NET default)
    :param fields: every successful NON-submit control at its default value -- hidden inputs (the
        ``__VIEWSTATE`` family), text-like input values, each ``<select>``'s selected option, checked
        checkboxes/radios, and ``<textarea>`` text -- the base of a postback body
    :param submit_controls: the form's named submit controls, in document order (the triggers a caller
        chooses among; an unnamed submit is omitted -- it carries no distinguishing ``name=value``)
    :param event_triggers: the form's ``__doPostBack`` triggers, in document order, de-duplicated by
        ``(target, argument)`` -- the triggers available when ``submit_controls`` is empty or does not
        contain the one that produces data
    """

    action_url: str
    fields: dict[str, str]
    submit_controls: tuple[FormControl, ...]
    event_triggers: tuple[EventTrigger, ...] = ()


def _attr_str(tag: Tag, name: str, default: str = "") -> str:
    """A tag attribute as a single string (bs4 returns a multi-valued attribute as a list)."""
    value = tag.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return value if isinstance(value, str) else default


def _collapsed_text(tag: Tag) -> str:
    """The tag's text, whitespace-collapsed (a ``<button>``/``<option>`` label with no value attr)."""
    return " ".join(tag.get_text().split())


def _default_fields(form: Tag) -> dict[str, str]:
    """The form's successful non-submit controls at their default values (WHATWG submission, untouched).

    Mirrors what a browser would submit for the form with nothing changed: hidden and text-like inputs
    post their ``value`` (empty when unset); a checkbox/radio posts only when ``checked``; a
    ``<select>`` posts its selected option (the first when none is marked -- the browser default); a
    ``<textarea>`` posts its text. Submit/image/reset/button controls are excluded (the caller adds one
    submit). Unnamed controls are skipped -- a control with no ``name`` is never successful.
    """
    fields: dict[str, str] = {}
    for inp in form.find_all("input"):
        name = _attr_str(inp, "name")
        if not name:
            continue
        input_type = _attr_str(inp, "type", "text").strip().lower()
        if input_type in _NON_DATA_INPUT_TYPES:
            continue
        if input_type in ("checkbox", "radio"):
            if inp.has_attr("checked"):
                fields[name] = _attr_str(inp, "value", "on")
            continue
        fields[name] = _attr_str(inp, "value")
    for select in form.find_all("select"):
        name = _attr_str(select, "name")
        if not name:
            continue
        options = select.find_all("option")
        chosen = next((o for o in options if o.has_attr("selected")), options[0] if options else None)
        if chosen is not None:
            fields[name] = _attr_str(chosen, "value") if chosen.has_attr("value") else _collapsed_text(chosen)
    for textarea in form.find_all("textarea"):
        name = _attr_str(textarea, "name")
        if not name:
            continue
        fields[name] = textarea.get_text()
    return fields


def _submit_controls(form: Tag) -> tuple[FormControl, ...]:
    """The form's NAMED submit controls, in document order (``<input type=submit|image>``, ``<button>``).

    A ``<button>`` with no explicit ``type`` defaults to a submit button (HTML), so it is included; its
    submitted value is its ``value`` attribute, else its text. Unnamed submits are dropped -- they carry
    no ``name=value`` and so cannot be the distinguishing trigger a replay activates.
    """
    controls: list[FormControl] = []
    for inp in form.find_all("input"):
        if _attr_str(inp, "type").strip().lower() in _SUBMIT_INPUT_TYPES:
            name = _attr_str(inp, "name")
            if name:
                controls.append(FormControl(name=name, value=_attr_str(inp, "value")))
    for button in form.find_all("button"):
        if _attr_str(button, "type", "submit").strip().lower() != "submit":
            continue
        name = _attr_str(button, "name")
        if name:
            value = _attr_str(button, "value") if button.has_attr("value") else _collapsed_text(button)
            controls.append(FormControl(name=name, value=value))
    return tuple(controls)


def _event_triggers(form: Tag) -> tuple[EventTrigger, ...]:
    """The form's ``__doPostBack`` triggers, in document order, de-duplicated by ``(target, argument)``.

    Reads the call out of each element's PARSED attribute value rather than the raw markup, because a
    served page escapes the quotes (``__doPostBack(&#39;x&#39;,&#39;&#39;)``) -- scanning bytes finds
    nothing on exactly the pages this exists for. A call with an empty target is skipped: it posts no
    identifiable event, so it cannot be a distinguishing trigger.
    """
    seen: dict[tuple[str, str], EventTrigger] = {}
    for tag in form.find_all(True):
        if not isinstance(tag, Tag):
            continue
        for attr in _EVENT_ATTRS:
            value = tag.get(attr)
            if not isinstance(value, str):
                continue
            for target, argument in _DO_POSTBACK_RE.findall(value):
                if target and (target, argument) not in seen:
                    seen[(target, argument)] = EventTrigger(
                        target=target, argument=argument, label=_collapsed_text(tag)
                    )
    return tuple(seen.values())


def _find_form(soup: BeautifulSoup, require_field: str | None) -> Tag | None:
    """The first ``<form>`` (or the first carrying a control named ``require_field``), or ``None``."""
    for form in soup.find_all("form"):
        if not isinstance(form, Tag):
            continue
        if require_field is None or form.find("input", attrs={"name": require_field}) is not None:
            return form
    return None


def parse_form(html: str | bytes, *, base_url: str, require_field: str | None = None) -> HtmlForm | None:
    """Parse an HTML form into its POST target, default field values, and submit controls.

    Parses with the stdlib ``html.parser`` (no lxml C-extension required -- the ``scrape`` dependency
    philosophy). Returns the FIRST ``<form>``, or -- when ``require_field`` is set -- the first form
    carrying a control with that ``name`` (pass ``"__VIEWSTATE"`` to select an ASP.NET WebForms form);
    ``None`` when no matching form exists. ``fields`` are the successful controls a browser would submit
    for the untouched form; ``submit_controls`` are the named triggers the caller chooses among.

    :param html: the form page markup (bytes or str)
    :param base_url: the URL the page was fetched from (resolves a relative/empty form ``action``)
    :param require_field: when set, select the first form containing a control with this ``name``
    :return: the parsed form, or ``None`` when no matching form is present
    """
    soup = BeautifulSoup(html, "html.parser")
    form = _find_form(soup, require_field)
    if form is None:
        return None
    action = _attr_str(form, "action")
    action_url = urljoin(base_url, action) if action else base_url
    return HtmlForm(
        action_url=action_url,
        fields=_default_fields(form),
        submit_controls=_submit_controls(form),
        event_triggers=_event_triggers(form),
    )


def build_form_post(
    form: HtmlForm,
    *,
    submit: FormControl | None = None,
    event: EventTrigger | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Serialize a form's postback body: default fields, ONE chosen trigger, then caller overrides.

    The base is the form's default ``fields``. Exactly one trigger is expected, in whichever of the two
    forms the page uses: ``submit`` adds that control's ``name=value`` (a browser posts only the
    activated submit, not every one), while ``event`` writes ``__EVENTTARGET`` / ``__EVENTARGUMENT`` --
    what the browser's ``__doPostBack`` does before submitting the same form. Passing both is allowed
    but is rarely what a page does; the caller owns that choice. ``overrides`` set/replace specific
    fields (search criteria, an option choice) and are applied LAST so they always win -- including over
    the event fields, so a caller can still drive them directly. Returns a fresh dict; ``form`` is never
    mutated.

    :param form: the parsed form
    :param submit: the submit control to activate (``None`` posts no submit ``name=value``)
    :param event: the ``__doPostBack`` trigger to raise (``None`` writes no event fields)
    :param overrides: field values to set/replace, applied last (``None`` for the untouched defaults)
    :return: the postback body as a ``name -> value`` mapping
    """
    body = dict(form.fields)
    if submit is not None:
        body[submit.name] = submit.value
    if event is not None:
        body[EVENT_TARGET_FIELD] = event.target
        body[EVENT_ARGUMENT_FIELD] = event.argument
    if overrides:
        body.update(overrides)
    return body
