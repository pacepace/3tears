"""Tests for deterministic HTML form parsing + postback serialization (``threetears.scrape.forms``).

Pure and hermetic -- every test feeds markup and asserts on the parsed form / serialized body, no
network and no browser. Covers the ASP.NET WebForms archetype (a ``__VIEWSTATE`` form) and the general
mechanics (default "successful controls", submit-control listing, action resolution, override policy).
"""

from __future__ import annotations

from threetears.scrape.forms import EventTrigger, FormControl, HtmlForm, build_form_post, parse_form

_PAGE_URL = "https://portal.example.gov/lobby/Directory.aspx"

#: A WebForms page whose ONLY triggers are ``__doPostBack`` anchors -- no named submit control exists.
#: The quotes are HTML-escaped exactly as a real portal serves them, which is the point of the fixture:
#: the call is only visible once the markup is parsed, never in the raw bytes.
_EVENT_DRIVEN_FORM = b"""<html><body>
<form method="post" action="LobbyistSearch.aspx" id="aspnetForm">
  <input type="hidden" name="__VIEWSTATE" value="/wEPDwUJODY" />
  <input type="hidden" name="__EVENTTARGET" value="" />
  <input type="hidden" name="__EVENTARGUMENT" value="" />
  <input type="text" name="ctl00$MainContent$txtName" value="" />
  <a href="javascript:__doPostBack(&#39;ctl00$MainContent$SearchButton&#39;,&#39;&#39;)">Search</a>
  <a href="javascript:__doPostBack(&#39;ctl00$MainContent$ExportButton&#39;,&#39;&#39;)">Export Selected Year</a>
  <a href="javascript:__doPostBack(&#39;ctl00$MainContent$Pager&#39;,&#39;2&#39;)">2</a>
  <a href="javascript:__doPostBack(&#39;ctl00$MainContent$SearchButton&#39;,&#39;&#39;)">Search again</a>
  <a href="https://elsewhere.example/help">Help</a>
</form>
</body></html>"""

_ASPNET_FORM = b"""<html><body>
<form method="post" action="./Directory.aspx" id="aspnetForm">
  <input type="hidden" name="__VIEWSTATE" value="/wEPDwUKLTETC1" />
  <input type="hidden" name="__VIEWSTATEGENERATOR" value="CA0B0334" />
  <input type="hidden" name="__EVENTVALIDATION" value="/wEdAAxQ" />
  <input type="hidden" name="__EVENTTARGET" value="" />
  <input type="text" name="ctl00$txtName" value="" />
  <input type="checkbox" name="ctl00$activeOnly" value="1" checked />
  <input type="checkbox" name="ctl00$includeInactive" value="1" />
  <select name="ctl00$ddlYear">
    <option value="2024">2024</option>
    <option value="2025" selected="selected">2025</option>
  </select>
  <textarea name="ctl00$notes">hi</textarea>
  <input type="submit" name="ctl00$btnShowAll" value="Show All" />
  <button name="ctl00$btnSearch">Search</button>
  <input type="reset" name="ctl00$btnReset" value="Reset" />
</form>
</body></html>"""


def test_parse_serializes_successful_controls_at_defaults() -> None:
    form = parse_form(_ASPNET_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    # Hidden fields ride; text posts its (empty) value; a checked box posts its value, an unchecked
    # one is absent; the select posts its SELECTED option (not the first); the textarea posts its text.
    assert form.fields["__VIEWSTATE"] == "/wEPDwUKLTETC1"
    assert form.fields["__EVENTVALIDATION"] == "/wEdAAxQ"
    assert form.fields["ctl00$txtName"] == ""
    assert form.fields["ctl00$activeOnly"] == "1"
    assert "ctl00$includeInactive" not in form.fields  # unchecked → not successful
    assert form.fields["ctl00$ddlYear"] == "2025"
    assert form.fields["ctl00$notes"] == "hi"
    # Submit/reset controls are NOT default fields.
    assert "ctl00$btnShowAll" not in form.fields
    assert "ctl00$btnReset" not in form.fields


def test_parse_lists_named_submit_controls_only() -> None:
    form = parse_form(_ASPNET_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    names = {c.name: c.value for c in form.submit_controls}
    # <input type=submit> and a default-type <button> are submits; <input type=reset> is not.
    assert names == {"ctl00$btnShowAll": "Show All", "ctl00$btnSearch": "Search"}


def test_parse_resolves_relative_and_empty_action() -> None:
    form = parse_form(_ASPNET_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert form.action_url == "https://portal.example.gov/lobby/Directory.aspx"
    empty = parse_form(_ASPNET_FORM.replace(b'action="./Directory.aspx"', b'action=""'), base_url=_PAGE_URL)
    assert empty is not None
    assert empty.action_url == _PAGE_URL  # a self-posting form


def test_parse_returns_none_when_required_field_absent() -> None:
    assert (
        parse_form(b"<html><form><input name='q'></form></html>", base_url=_PAGE_URL, require_field="__VIEWSTATE")
        is None
    )


def test_parse_returns_none_when_no_form() -> None:
    assert parse_form(b"<html><body><p>no form here</p></body></html>", base_url=_PAGE_URL) is None


def test_parse_first_form_when_no_require_field() -> None:
    html = b"<html><form action='/a'><input name='x' value='1'></form><form action='/b'></form></html>"
    form = parse_form(html, base_url="https://h.example/p")
    assert form is not None
    assert form.action_url == "https://h.example/a"
    assert form.fields == {"x": "1"}


def test_build_form_post_defaults_plus_submit() -> None:
    form = parse_form(_ASPNET_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    body = build_form_post(form, submit=FormControl(name="ctl00$btnShowAll", value="Show All"))
    assert body["__VIEWSTATE"] == "/wEPDwUKLTETC1"
    assert body["ctl00$btnShowAll"] == "Show All"  # the activated submit is included
    assert "ctl00$btnSearch" not in body  # the other submit is NOT
    # The source form is unchanged (fresh dict returned).
    assert "ctl00$btnShowAll" not in form.fields


def test_build_form_post_overrides_win_last() -> None:
    form = HtmlForm(action_url="https://h/x", fields={"q": "", "year": "2025"}, submit_controls=())
    body = build_form_post(form, overrides={"q": "acme", "year": "2024"})
    assert body == {"q": "acme", "year": "2024"}


def test_build_form_post_no_submit_no_overrides_is_the_defaults() -> None:
    form = HtmlForm(action_url="https://h/x", fields={"__VIEWSTATE": "v"}, submit_controls=())
    assert build_form_post(form) == {"__VIEWSTATE": "v"}


# --- __doPostBack event triggers -------------------------------------------------


def test_parse_finds_event_triggers_when_no_submit_control_exists() -> None:
    """A form driven only by ``__doPostBack`` anchors still reports triggers.

    Without this the form parses to zero triggers and a browser-free replay concludes the page cannot
    be driven at all -- when in fact the browser is doing nothing more than setting two hidden fields.
    """
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert form.submit_controls == ()  # nothing carries a name=value
    assert [(t.target, t.argument) for t in form.event_triggers] == [
        ("ctl00$MainContent$SearchButton", ""),
        ("ctl00$MainContent$ExportButton", ""),
        ("ctl00$MainContent$Pager", "2"),
    ]


def test_event_triggers_are_deduplicated_but_keep_distinct_arguments() -> None:
    """Two anchors raising the same event are one trigger; the same control with a different
    argument is NOT -- that is how a pager names its pages."""
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    targets = [t.target for t in form.event_triggers]
    assert targets.count("ctl00$MainContent$SearchButton") == 1  # the repeated anchor collapsed
    pager = [t for t in form.event_triggers if t.target == "ctl00$MainContent$Pager"]
    assert pager[0].argument == "2"


def test_event_trigger_carries_its_label_for_caller_choice() -> None:
    """The visible text is what lets a caller pick "Search" over "Export" without hardcoding
    control-name conventions, which vary per portal."""
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    labels = {t.target: t.label for t in form.event_triggers}
    assert labels["ctl00$MainContent$SearchButton"] == "Search"
    assert labels["ctl00$MainContent$ExportButton"] == "Export Selected Year"


def test_plain_links_are_not_event_triggers() -> None:
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert all("elsewhere.example" not in t.target for t in form.event_triggers)


def test_event_trigger_read_from_onclick_as_well_as_href() -> None:
    html = b"""<html><form><input type="hidden" name="__VIEWSTATE" value="v" />
      <input type="button" onclick="__doPostBack('ctl00$Go','')" value="Go" /></form></html>"""
    form = parse_form(html, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert [t.target for t in form.event_triggers] == ["ctl00$Go"]


def test_event_trigger_with_empty_target_is_skipped() -> None:
    """An empty target posts no identifiable event, so it cannot be a distinguishing trigger."""
    html = b"""<html><form><input type="hidden" name="__VIEWSTATE" value="v" />
      <a href="javascript:__doPostBack('','')">nothing</a></form></html>"""
    form = parse_form(html, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert form.event_triggers == ()


def test_build_form_post_writes_the_event_fields() -> None:
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    search = next(t for t in form.event_triggers if t.label == "Search")
    body = build_form_post(form, event=search)
    assert body["__EVENTTARGET"] == "ctl00$MainContent$SearchButton"
    assert body["__EVENTARGUMENT"] == ""
    assert body["__VIEWSTATE"] == "/wEPDwUJODY"  # the form state still rides


def test_build_form_post_event_argument_rides_for_a_pager() -> None:
    form = parse_form(_EVENT_DRIVEN_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    pager = next(t for t in form.event_triggers if t.target.endswith("Pager"))
    assert build_form_post(form, event=pager)["__EVENTARGUMENT"] == "2"


def test_build_form_post_overrides_still_win_over_the_event_fields() -> None:
    form = HtmlForm(action_url="https://h/x", fields={}, submit_controls=())
    body = build_form_post(form, event=EventTrigger(target="a", argument="1"), overrides={"__EVENTARGUMENT": "9"})
    assert body == {"__EVENTTARGET": "a", "__EVENTARGUMENT": "9"}


def test_form_with_no_event_triggers_reports_an_empty_tuple() -> None:
    """The pre-existing submit-control archetype is untouched — it simply has no event triggers."""
    form = parse_form(_ASPNET_FORM, base_url=_PAGE_URL, require_field="__VIEWSTATE")
    assert form is not None
    assert form.event_triggers == ()
    assert len(form.submit_controls) == 2
