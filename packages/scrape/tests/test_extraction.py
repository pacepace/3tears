"""Unit tests for threetears.scrape.extraction -- structural validation and
candidate generation (mocking approach mirrors tests/unit/test_query_agent_matching.py's
create_chat_model mocking pattern).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from threetears.scrape.extraction import (
    DiscoverySchemaResult,
    RowValidationResult,
    ValidationResult,
    _CandidateStrategy,
    _CandidateStrategyList,
    _DiscoveredCandidate,
    _DiscoveredCandidateList,
    _DiscoveredFieldProposal,
    _DiscoveredRowCandidate,
    _DiscoveredRowCandidateList,
    _normalize_numeric_text,
    _RegexCandidateStrategy,
    _RegexCandidateStrategyList,
    _RowCandidateStrategy,
    _RowCandidateStrategyList,
    discover_candidates,
    discover_row_candidates,
    generate_candidates,
    generate_regex_candidates,
    generate_regex_row_candidates,
    generate_row_candidates,
    html_to_text,
    strip_boilerplate,
    validate_candidate,
    validate_regex_candidate,
    validate_regex_row_candidate,
    validate_row_candidate,
)

_PAGE_HTML = """
<html><body>
    <table class="warn-notices">
        <tr><td class="employer">Acme Corp</td><td class="count">42</td></tr>
    </table>
    <div class="empty-field"></div>
</body></html>
"""

_ROWS_PAGE_HTML = """
<html><body>
<table>
  <thead><tr><th>Employer</th><th>Count</th><th>County</th></tr></thead>
  <tbody>
    <tr><td class="employer">Dejana    Truck</td><td class="count">1,234</td><td class="county">Baltimore</td></tr>
    <tr><td class="employer">ZeniMax</td><td class="count">2.5K</td><td class="county">Cockeysville</td></tr>
    <tr><td class="employer"></td><td class="count">bad</td><td class="county">Spacer Row</td></tr>
  </tbody>
</table>
</body></html>
"""


def _fake_structured_model(result=None, *, side_effect=None):
    ainvoke_mock = AsyncMock(return_value=result, side_effect=side_effect)
    structured = SimpleNamespace(ainvoke=ainvoke_mock)
    return SimpleNamespace(with_structured_output=lambda schema, **kwargs: structured), ainvoke_mock


# ===========================================================================
# strip_boilerplate
# ===========================================================================


class TestStripBoilerplate:
    def test_removes_script_style_nav_header_footer_and_comments(self):
        html = (
            "<html><body>"
            "<script>var x = 1;</script>"
            "<style>.x{color:red}</style>"
            "<nav>Site nav</nav>"
            "<header>Site header</header>"
            "<!-- a comment -->"
            "<main>Real content</main>"
            "<footer>Site footer</footer>"
            "</body></html>"
        )
        stripped = strip_boilerplate(html)
        assert "Real content" in stripped
        for gone in ("var x = 1", "color:red", "Site nav", "Site header", "Site footer", "a comment"):
            assert gone not in stripped

    def test_content_with_no_boilerplate_is_unaffected(self):
        html = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"
        assert "Acme Corp" in strip_boilerplate(html)


# ===========================================================================
# validate_candidate
# ===========================================================================


class TestValidateCandidate:
    def test_valid_strategy_extracts_all_fields(self):
        strategy = {"employer": "td.employer", "affected_count": "td.count"}
        schema = {"employer": str, "affected_count": int}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is True
        assert result.extracted == {"employer": "Acme Corp", "affected_count": 42}
        assert result.errors == []

    def test_missing_selector_for_required_field(self):
        strategy = {"employer": "td.employer"}
        schema = {"employer": str, "affected_count": int}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is False
        assert any("no selector proposed" in e for e in result.errors)
        assert "affected_count" not in result.extracted

    def test_selector_matches_nothing(self):
        strategy = {"employer": "td.does-not-exist", "affected_count": "td.count"}
        schema = {"employer": str, "affected_count": int}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is False
        assert any("matched nothing" in e for e in result.errors)

    def test_selector_matches_empty_element(self):
        strategy = {"employer": "div.empty-field"}
        schema = {"employer": str}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is False
        assert any("empty element" in e for e in result.errors)

    def test_type_mismatch_fails_validation(self):
        strategy = {"affected_count": "td.employer"}  # "Acme Corp" doesn't parse as int
        schema = {"affected_count": int}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is False
        assert any("does not parse as int" in e for e in result.errors)

    def test_str_field_never_needs_coercion(self):
        strategy = {"employer": "td.employer"}
        schema = {"employer": str}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is True
        assert result.extracted["employer"] == "Acme Corp"

    def test_validation_result_is_frozen_dataclass_defaults(self):
        result = ValidationResult(valid=True)
        assert result.extracted == {}
        assert result.errors == []

    def test_syntactically_invalid_selector_degrades_gracefully_not_a_crash(self):
        # Real bug, live-discovered via schema-discovery mode (real Maryland WARN page,
        # 2026-07-15): an LLM-proposed selector is not guaranteed to be valid CSS -- a jQuery-ism
        # like "td:eq(0)" raised an uncaught soupsieve.util.SelectorSyntaxError here, crashing
        # every eval_loop.py candidate-generation path that calls this with LLM-proposed selectors.
        # Fixed at the source (_select_one_safe) -- this is the regression test for that fix.
        strategy = {"employer": "td:eq(0)"}
        schema = {"employer": str}
        result = validate_candidate(_PAGE_HTML, strategy, schema)
        assert result.valid is False
        assert any("matched nothing" in e for e in result.errors)


# ===========================================================================
# generate_candidates
# ===========================================================================


class TestGenerateCandidates:
    async def test_success_returns_selector_dicts(self):
        parsed = _CandidateStrategyList(
            candidates=[
                _CandidateStrategy(selectors={"employer": "td.employer"}),
                _CandidateStrategy(selectors={"employer": ".employer-name"}),
            ]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_candidates(_PAGE_HTML, {"employer": str}, n=2, api_key="k")
        assert candidates == [{"employer": "td.employer"}, {"employer": ".employer-name"}]
        assert ainvoke_mock.await_count == 1

    async def test_retries_before_succeeding(self):
        parsed = _CandidateStrategyList(candidates=[_CandidateStrategy(selectors={"employer": "td.employer"})])
        fake_model, ainvoke_mock = _fake_structured_model(side_effect=[RuntimeError("transient"), parsed])
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            candidates = await generate_candidates(_PAGE_HTML, {"employer": str}, api_key="k")
        assert ainvoke_mock.await_count == 2
        assert candidates == [{"employer": "td.employer"}]

    async def test_total_failure_returns_empty_list_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            candidates = await generate_candidates(_PAGE_HTML, {"employer": str}, api_key="k")
        assert candidates == []

    async def test_empty_candidate_list_from_llm_returns_empty(self):
        parsed = _CandidateStrategyList(candidates=[])
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_candidates(_PAGE_HTML, {"employer": str}, api_key="k")
        assert candidates == []


# ===========================================================================
# _normalize_numeric_text (the user's ask: "1M" and "1K" on different sites
# must normalize to the same magnitude)
# ===========================================================================


class TestNormalizeNumericText:
    def test_plain_digits_pass_through(self):
        assert _normalize_numeric_text("42") == "42"

    def test_strips_thousands_separator_commas(self):
        assert _normalize_numeric_text("1,234") == "1234"

    def test_strips_currency_symbol(self):
        assert _normalize_numeric_text("$1,234") == "1234"

    def test_expands_k_suffix(self):
        assert _normalize_numeric_text("2.5K") == "2500"

    def test_expands_m_suffix(self):
        assert _normalize_numeric_text("1M") == "1000000"

    def test_expands_b_suffix_lowercase(self):
        assert _normalize_numeric_text("1.2b") == "1200000000"

    def test_non_numeric_text_passes_through_unparsed(self):
        # Not this function's job to validate -- int()/float() downstream
        # raises on genuinely non-numeric text; normalization is a no-op.
        assert _normalize_numeric_text("bad") == "bad"

    def test_takes_leading_number_before_trailing_annotation(self):
        # Live finding (Maryland's real Total Employees column, 2026-07-14):
        # a genuine count with a trailing annotation the source site itself put there.
        assert _normalize_numeric_text("5 (Remote workers in MD)") == "5"

    def test_takes_leading_number_with_suffix_before_trailing_annotation(self):
        assert _normalize_numeric_text("2K (estimated)") == "2000"


# ===========================================================================
# validate_row_candidate
# ===========================================================================


class TestValidateRowCandidate:
    def test_valid_strategy_extracts_every_full_row(self):
        strategy = {
            "row_selector": "tbody tr",
            "field_selectors": {"employer": "td.employer", "affected_count": "td.count", "county": "td.county"},
        }
        schema = {"employer": str, "affected_count": int, "county": str}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, schema)
        assert result.valid is True
        assert result.total_rows_matched == 3  # row_selector matched all 3, including the bad one
        # Only the 2 fully-parseable rows make it into `rows` -- the spacer/malformed
        # row (empty employer, "bad" count) is excluded, not fatal to the candidate.
        assert result.records == [
            {"employer": "Dejana Truck", "affected_count": 1234, "county": "Baltimore"},
            {"employer": "ZeniMax", "affected_count": 2500, "county": "Cockeysville"},
        ]

    def test_whitespace_collapsed_within_a_row(self):
        strategy = {"row_selector": "tbody tr", "field_selectors": {"employer": "td.employer"}}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, {"employer": str})
        assert result.records[0]["employer"] == "Dejana Truck"  # not "Dejana    Truck"

    def test_no_row_selector_is_invalid(self):
        result = validate_row_candidate(_ROWS_PAGE_HTML, {}, {"employer": str})
        assert result.valid is False
        assert result.total_rows_matched == 0
        assert any("no row_selector proposed" in e for e in result.errors)

    def test_row_selector_matches_nothing_is_invalid(self):
        strategy = {"row_selector": ".does-not-exist", "field_selectors": {"employer": "td.employer"}}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, {"employer": str})
        assert result.valid is False
        assert result.total_rows_matched == 0
        assert result.records == []

    def test_syntactically_invalid_row_selector_degrades_gracefully_not_a_crash(self):
        # Regression test for the same live-discovered bug as TestValidateCandidate's own --
        # an invalid row_selector must degrade like "matched nothing," never crash.
        strategy = {"row_selector": "tr:eq(0)", "field_selectors": {"employer": "td.employer"}}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, {"employer": str})
        assert result.valid is False
        assert result.total_rows_matched == 0

    def test_syntactically_invalid_field_selector_degrades_gracefully_not_a_crash(self):
        strategy = {"row_selector": "tbody tr", "field_selectors": {"employer": "td:eq(0)"}}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, {"employer": str})
        assert result.valid is False
        assert result.records == []

    def test_every_row_failing_is_invalid(self):
        strategy = {"row_selector": "tbody tr", "field_selectors": {"employer": ".nope"}}
        result = validate_row_candidate(_ROWS_PAGE_HTML, strategy, {"employer": str})
        assert result.valid is False
        assert result.total_rows_matched == 3
        assert result.records == []
        assert len(result.errors) == 3  # one "matched nothing" per row

    def test_row_error_list_is_capped(self):
        many_rows_html = "<table><tbody>" + "<tr><td>x</td></tr>" * 50 + "</tbody></table>"
        strategy = {"row_selector": "tbody tr", "field_selectors": {"missing": ".nope"}}
        result = validate_row_candidate(many_rows_html, strategy, {"missing": str})
        assert result.total_rows_matched == 50
        assert len(result.errors) == 20  # _MAX_ROW_ERRORS, not 50

    def test_row_validation_result_is_frozen_dataclass_defaults(self):
        result = RowValidationResult(valid=False)
        assert result.records == []
        assert result.total_rows_matched == 0
        assert result.errors == []


# ===========================================================================
# generate_row_candidates
# ===========================================================================


class TestGenerateRowCandidates:
    async def test_success_returns_row_and_field_selector_dicts(self):
        parsed = _RowCandidateStrategyList(
            candidates=[
                _RowCandidateStrategy(row_selector="tbody tr", field_selectors={"employer": "td.employer"}),
            ]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_row_candidates(_ROWS_PAGE_HTML, {"employer": str}, api_key="k")
        assert candidates == [{"row_selector": "tbody tr", "field_selectors": {"employer": "td.employer"}}]
        assert ainvoke_mock.await_count == 1

    async def test_total_failure_returns_empty_list_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            candidates = await generate_row_candidates(_ROWS_PAGE_HTML, {"employer": str}, api_key="k")
        assert candidates == []


# ===========================================================================
# html_to_text
# ===========================================================================


class TestHtmlToText:
    def test_extracts_text_with_one_rendered_line_per_output_line(self):
        html = "<html><body><p>Acme Corp</p><p>County: Oakland</p></body></html>"
        text = html_to_text(html)
        assert text == "Acme Corp\nCounty: Oakland"

    def test_boilerplate_is_stripped_first(self):
        html = (
            "<html><body>"
            "<nav>Site nav</nav>"
            "<script>var x = 1;</script>"
            "<p>Real content</p>"
            "<footer>Site footer</footer>"
            "</body></html>"
        )
        text = html_to_text(html)
        assert text == "Real content"

    def test_now_empty_lines_between_block_tags_are_dropped(self):
        html = "<html><body><div></div><p>First</p><div>   </div><p>Second</p></body></html>"
        text = html_to_text(html)
        assert text == "First\nSecond"

    def test_line_breaks_are_preserved_not_collapsed_to_a_single_space(self):
        # Unlike per-field whitespace normalization -- a text-block page's
        # own paragraph/record boundaries are real structure, not noise.
        html = "<html><body><p>Line one</p><p>Line two</p><p>Line three</p></body></html>"
        text = html_to_text(html)
        assert text.count("\n") == 2

    def test_inline_br_still_produces_separate_lines(self):
        html = "<html><body><p>Employer Name<br>County: Oakland</p></body></html>"
        text = html_to_text(html)
        assert text == "Employer Name\nCounty: Oakland"


# ===========================================================================
# validate_regex_candidate
# ===========================================================================

_TEXT_PAGE = "Acme Corp\nCounty: Oakland\nAffected: 42"


class TestValidateRegexCandidate:
    def test_valid_pattern_extracts_all_fields(self):
        pattern = r"(?P<employer>[^\n]+)\nCounty: (?P<county>[^\n]+)\nAffected: (?P<affected_count>\d+)"
        schema = {"employer": str, "county": str, "affected_count": int}
        result = validate_regex_candidate(_TEXT_PAGE, pattern, schema)
        assert result.valid is True
        assert result.extracted == {"employer": "Acme Corp", "county": "Oakland", "affected_count": 42}
        assert result.errors == []

    def test_invalid_regex_syntax_is_reported_not_raised(self):
        result = validate_regex_candidate(_TEXT_PAGE, r"(?P<employer>[unterminated", {"employer": str})
        assert result.valid is False
        assert any("invalid regex" in e for e in result.errors)

    def test_pattern_matching_nothing_is_invalid(self):
        result = validate_regex_candidate(_TEXT_PAGE, r"(?P<employer>NoSuchCompany)", {"employer": str})
        assert result.valid is False
        assert any("matched nothing" in e for e in result.errors)

    def test_missing_named_group_for_a_schema_field_is_invalid(self):
        pattern = r"(?P<employer>[^\n]+)"
        result = validate_regex_candidate(_TEXT_PAGE, pattern, {"employer": str, "county": str})
        assert result.valid is False
        assert any("county" in e and "no named group" in e for e in result.errors)

    def test_named_group_matching_empty_text_is_invalid(self):
        pattern = r"(?P<employer>)\nCounty: (?P<county>[^\n]+)"
        result = validate_regex_candidate(_TEXT_PAGE, pattern, {"employer": str, "county": str})
        assert result.valid is False
        assert any("matched empty text" in e for e in result.errors)

    def test_type_mismatch_fails_validation(self):
        pattern = r"(?P<affected_count>Acme Corp)"
        result = validate_regex_candidate(_TEXT_PAGE, pattern, {"affected_count": int})
        assert result.valid is False
        assert any("does not parse as int" in e for e in result.errors)

    def test_whitespace_around_the_captured_value_is_normalized(self):
        text = "Acme Corp\nCounty:   Oakland  \n"
        pattern = r"(?P<employer>[^\n]+)\nCounty:(?P<county>.+)"
        result = validate_regex_candidate(text, pattern, {"employer": str, "county": str})
        assert result.extracted["county"] == "Oakland"


# ===========================================================================
# generate_regex_candidates
# ===========================================================================


class TestGenerateRegexCandidates:
    async def test_success_returns_pattern_strings(self):
        parsed = _RegexCandidateStrategyList(
            candidates=[
                _RegexCandidateStrategy(pattern=r"(?P<employer>[^\n]+)"),
                _RegexCandidateStrategy(pattern=r"(?P<employer>.+)\nCounty:.+"),
            ]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_regex_candidates(_TEXT_PAGE, {"employer": str}, n=2, api_key="k")
        assert candidates == [r"(?P<employer>[^\n]+)", r"(?P<employer>.+)\nCounty:.+"]
        assert ainvoke_mock.await_count == 1

    async def test_total_failure_returns_empty_list_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            candidates = await generate_regex_candidates(_TEXT_PAGE, {"employer": str}, api_key="k")
        assert candidates == []

    async def test_empty_candidate_list_from_llm_returns_empty(self):
        parsed = _RegexCandidateStrategyList(candidates=[])
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_regex_candidates(_TEXT_PAGE, {"employer": str}, api_key="k")
        assert candidates == []


# ===========================================================================
# validate_regex_row_candidate
# ===========================================================================

_TEXT_ROWS_PAGE = (
    "Acme Corp\nCounty: Oakland\nAffected: 42\n\n"
    "Widgets Inc\nCounty: Wayne\nAffected: 7\n\n"
    "Bad Record\nCounty: \nAffected: not-a-number"
)


class TestValidateRegexRowCandidate:
    _PATTERN = r"(?P<employer>[^\n]+)\nCounty: (?P<county>[^\n]*)\nAffected: (?P<affected_count>[^\n]+)"

    def test_valid_pattern_extracts_every_full_record(self):
        schema = {"employer": str, "county": str, "affected_count": int}
        result = validate_regex_row_candidate(_TEXT_ROWS_PAGE, self._PATTERN, schema)
        assert result.valid is True
        assert result.total_rows_matched == 3  # matched all 3, including the bad one
        assert result.records == [
            {"employer": "Acme Corp", "county": "Oakland", "affected_count": 42},
            {"employer": "Widgets Inc", "county": "Wayne", "affected_count": 7},
        ]

    def test_invalid_regex_syntax_is_reported_not_raised(self):
        result = validate_regex_row_candidate(_TEXT_ROWS_PAGE, r"(?P<employer>[unterminated", {"employer": str})
        assert result.valid is False
        assert any("invalid regex" in e for e in result.errors)

    def test_pattern_matching_nothing_is_invalid(self):
        result = validate_regex_row_candidate(_TEXT_ROWS_PAGE, r"(?P<employer>NoSuchCompany)", {"employer": str})
        assert result.valid is False
        assert result.total_rows_matched == 0
        assert result.records == []

    def test_every_record_failing_is_invalid(self):
        schema = {"employer": str, "county": str, "affected_count": int}
        result = validate_regex_row_candidate(_TEXT_ROWS_PAGE, self._PATTERN, schema)
        assert len(result.errors) >= 1  # the "Bad Record" entry's empty county / non-numeric count

    def test_row_error_list_is_capped(self):
        many_records = "\n\n".join(f"Employer {i}\nCounty: \nAffected: bad" for i in range(50))
        result = validate_regex_row_candidate(many_records, self._PATTERN, {"employer": str, "county": str})
        assert result.total_rows_matched == 50
        assert len(result.errors) == 20  # _MAX_ROW_ERRORS, not 50


# ===========================================================================
# generate_regex_row_candidates
# ===========================================================================


class TestGenerateRegexRowCandidates:
    async def test_success_returns_pattern_strings(self):
        parsed = _RegexCandidateStrategyList(
            candidates=[_RegexCandidateStrategy(pattern=r"(?P<employer>[^\n]+)\nCounty: (?P<county>[^\n]+)")]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            candidates = await generate_regex_row_candidates(_TEXT_ROWS_PAGE, {"employer": str}, api_key="k")
        assert candidates == [r"(?P<employer>[^\n]+)\nCounty: (?P<county>[^\n]+)"]
        assert ainvoke_mock.await_count == 1

    async def test_total_failure_returns_empty_list_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            candidates = await generate_regex_row_candidates(_TEXT_ROWS_PAGE, {"employer": str}, api_key="k")
        assert candidates == []


# ===========================================================================
# discover_candidates
# ===========================================================================


class TestDiscoverCandidates:
    async def test_discovers_and_validates_real_fields(self):
        parsed = _DiscoveredCandidateList(
            candidates=[
                _DiscoveredCandidate(
                    fields=[
                        _DiscoveredFieldProposal(
                            name="employer", type_name="str", selector="td.employer", sample_value_hint="Acme Corp"
                        ),
                        _DiscoveredFieldProposal(
                            name="count", type_name="int", selector="td.count", sample_value_hint="42"
                        ),
                    ]
                )
            ]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_candidates(_PAGE_HTML, api_key="k")
        assert isinstance(result, DiscoverySchemaResult)
        assert result.validated is True
        assert result.field_schema == {"employer": str, "count": int}
        assert result.strategy == {"employer": "td.employer", "count": "td.count"}
        assert result.sample_records == [{"employer": "Acme Corp", "count": 42}]
        assert {f.name for f in result.fields} == {"employer", "count"}
        assert ainvoke_mock.await_count == 1

    async def test_a_proposed_field_that_does_not_validate_is_dropped_not_included(self):
        parsed = _DiscoveredCandidateList(
            candidates=[
                _DiscoveredCandidate(
                    fields=[
                        _DiscoveredFieldProposal(
                            name="employer", type_name="str", selector="td.employer", sample_value_hint="Acme Corp"
                        ),
                        _DiscoveredFieldProposal(
                            name="ghost", type_name="str", selector=".does-not-exist", sample_value_hint="?"
                        ),
                    ]
                )
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_candidates(_PAGE_HTML, api_key="k")
        assert result.validated is True
        assert "ghost" not in result.field_schema
        assert set(result.field_schema) == {"employer"}

    async def test_most_fields_validated_wins_no_judge(self):
        worse = _DiscoveredCandidate(
            fields=[_DiscoveredFieldProposal(name="employer", type_name="str", selector="td.employer", sample_value_hint="x")]
        )
        better = _DiscoveredCandidate(
            fields=[
                _DiscoveredFieldProposal(name="employer", type_name="str", selector="td.employer", sample_value_hint="x"),
                _DiscoveredFieldProposal(name="count", type_name="int", selector="td.count", sample_value_hint="42"),
            ]
        )
        parsed = _DiscoveredCandidateList(candidates=[worse, better])
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_candidates(_PAGE_HTML, api_key="k")
        assert set(result.field_schema) == {"employer", "count"}

    async def test_zero_fields_validate_returns_honest_empty_result(self):
        parsed = _DiscoveredCandidateList(
            candidates=[
                _DiscoveredCandidate(
                    fields=[_DiscoveredFieldProposal(name="ghost", type_name="str", selector=".nope", sample_value_hint="?")]
                )
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_candidates(_PAGE_HTML, api_key="k")
        assert result.validated is False
        assert result.fields == []
        assert result.field_schema == {}

    async def test_total_llm_failure_returns_honest_empty_result_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            result = await discover_candidates(_PAGE_HTML, api_key="k")
        assert result.validated is False
        assert result.fields == []


# ===========================================================================
# discover_row_candidates
# ===========================================================================


class TestDiscoverRowCandidates:
    async def test_discovers_and_validates_real_row_fields(self):
        parsed = _DiscoveredRowCandidateList(
            candidates=[
                _DiscoveredRowCandidate(
                    row_selector="tbody tr",
                    fields=[
                        _DiscoveredFieldProposal(
                            name="employer", type_name="str", selector="td.employer", sample_value_hint="Dejana Truck"
                        ),
                        _DiscoveredFieldProposal(
                            name="county", type_name="str", selector="td.county", sample_value_hint="Baltimore"
                        ),
                    ],
                )
            ]
        )
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_row_candidates(_ROWS_PAGE_HTML, api_key="k")
        assert result.validated is True
        assert result.field_schema == {"employer": str, "county": str}
        assert result.strategy == {
            "row_selector": "tbody tr",
            "field_selectors": {"employer": "td.employer", "county": "td.county"},
        }
        # third row's employer is empty -- only 2 of 3 rows fully validate
        assert len(result.sample_records) == 2
        assert ainvoke_mock.await_count == 1

    async def test_one_candidates_invalid_jquery_style_selector_does_not_crash_the_whole_call(self):
        # Real, live-discovered failure mode (real Maryland WARN page, 2026-07-15): an LLM
        # genuinely proposed a jQuery-ism like "td:eq(0)" for one candidate. That candidate must
        # be skipped (or its field dropped), never crash the whole discovery call -- a good
        # sibling candidate's real fields must still be found and returned.
        parsed = _DiscoveredRowCandidateList(
            candidates=[
                _DiscoveredRowCandidate(
                    row_selector="tbody tr",
                    fields=[_DiscoveredFieldProposal(name="bad", type_name="str", selector="td:eq(0)", sample_value_hint="x")],
                ),
                _DiscoveredRowCandidate(
                    row_selector="tbody tr",
                    fields=[
                        _DiscoveredFieldProposal(
                            name="employer", type_name="str", selector="td.employer", sample_value_hint="Dejana Truck"
                        )
                    ],
                ),
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_row_candidates(_ROWS_PAGE_HTML, api_key="k")
        assert result.validated is True
        assert set(result.field_schema) == {"employer"}

    async def test_disjoint_fields_that_never_co_occur_in_one_row_are_rejected(self):
        # Critic-caught, chunk review: _fields_matching_any_row only confirms each field
        # matches SOME row independently -- two fields whose non-empty rows are DISJOINT can
        # each survive that pre-filter yet never co-occur in a single row together, so
        # validate_row_candidate's all-or-nothing-per-record check returns zero real records.
        # This candidate must be rejected, not accepted with validated=True and no records.
        disjoint_html = (
            "<html><body><table><tbody>"
            '<tr><td class="employer">Acme Corp</td><td class="county"></td></tr>'
            '<tr><td class="employer"></td><td class="county">Baltimore</td></tr>'
            "</tbody></table></body></html>"
        )
        parsed = _DiscoveredRowCandidateList(
            candidates=[
                _DiscoveredRowCandidate(
                    row_selector="tbody tr",
                    fields=[
                        _DiscoveredFieldProposal(name="employer", type_name="str", selector="td.employer", sample_value_hint="x"),
                        _DiscoveredFieldProposal(name="county", type_name="str", selector="td.county", sample_value_hint="x"),
                    ],
                )
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_row_candidates(disjoint_html, api_key="k")
        assert result.validated is False
        assert result.sample_records == []

    async def test_a_field_that_never_validates_across_any_row_is_dropped(self):
        parsed = _DiscoveredRowCandidateList(
            candidates=[
                _DiscoveredRowCandidate(
                    row_selector="tbody tr",
                    fields=[
                        _DiscoveredFieldProposal(
                            name="employer", type_name="str", selector="td.employer", sample_value_hint="x"
                        ),
                        _DiscoveredFieldProposal(name="ghost", type_name="str", selector=".nope", sample_value_hint="?"),
                    ],
                )
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_row_candidates(_ROWS_PAGE_HTML, api_key="k")
        assert result.validated is True
        assert set(result.field_schema) == {"employer"}

    async def test_bad_row_selector_returns_honest_empty_result(self):
        parsed = _DiscoveredRowCandidateList(
            candidates=[
                _DiscoveredRowCandidate(
                    row_selector=".does-not-exist",
                    fields=[_DiscoveredFieldProposal(name="employer", type_name="str", selector="td.employer", sample_value_hint="x")],
                )
            ]
        )
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            result = await discover_row_candidates(_ROWS_PAGE_HTML, api_key="k")
        assert result.validated is False
        assert result.sample_records == []

    async def test_total_llm_failure_returns_honest_empty_result_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            result = await discover_row_candidates(_ROWS_PAGE_HTML, api_key="k")
        assert result.validated is False
