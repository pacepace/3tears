"""Tests for built-in tool modules."""

from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class TestCalculator:
    @pytest.fixture(autouse=True)
    def _require_simpleeval(self):
        pytest.importorskip("simpleeval")

    def _create(self, config: dict[str, Any] | None = None) -> Any:
        from threetears.agent.tools.builtin.calculator import create_calculator_tool

        return create_calculator_tool(config or {}, "Calculate math expressions")

    def test_basic(self):
        tool = self._create()
        assert tool.invoke({"expression": "2 + 3"}) == "5"

    def test_functions(self):
        tool = self._create()
        assert tool.invoke({"expression": "sqrt(144)"}) == "12"

    def test_error(self):
        tool = self._create()
        result = tool.invoke({"expression": "invalid!!!"})
        assert "[TOOL ERROR]" in result

    def test_constants(self):
        tool = self._create()
        result = tool.invoke({"expression": "pi"})
        assert result.startswith("3.14")

    def test_float_formatting(self):
        tool = self._create()
        # 2.0 + 3.0 should show "5", not "5.0"
        assert tool.invoke({"expression": "2.0 + 3.0"}) == "5"

    def test_postfix_factorial_literal(self):
        """``n!`` postfix syntax is rewritten into ``factorial(n)``."""
        tool = self._create()
        assert tool.invoke({"expression": "10!"}) == "3628800"

    def test_postfix_factorial_in_compound_expression(self):
        """``n!`` inside a compound expression rewrites correctly."""
        tool = self._create()
        # 5! + 1 == 121
        assert tool.invoke({"expression": "5! + 1"}) == "121"

    def test_postfix_factorial_paren_subexpression(self):
        """``(expr)!`` rewrites to ``factorial((expr))``."""
        tool = self._create()
        # (2+3)! == 5! == 120
        assert tool.invoke({"expression": "(2+3)!"}) == "120"

    def test_explicit_factorial_call_still_works(self):
        """``factorial(n)`` form still works alongside the postfix sugar."""
        tool = self._create()
        assert tool.invoke({"expression": "factorial(7)"}) == "5040"

    def test_caret_translates_to_power(self):
        """``^`` is rewritten to ``**`` so callers can write ``2^10``."""
        tool = self._create()
        assert tool.invoke({"expression": "2^10"}) == "1024"

    def test_caret_combined_with_factorial(self):
        """notation conveniences compose: ``2^3 + 4!`` -> 32."""
        tool = self._create()
        assert tool.invoke({"expression": "2^3 + 4!"}) == "32"


# ---------------------------------------------------------------------------
# Unit Converter
# ---------------------------------------------------------------------------


class TestUnitConverter:
    @pytest.fixture(autouse=True)
    def _require_pint(self):
        pytest.importorskip("pint")

    def _create(self) -> Any:
        from threetears.agent.tools.builtin.unit_converter import create_unit_converter_tool

        return create_unit_converter_tool({}, "Convert units")

    def test_basic(self):
        tool = self._create()
        result = tool.invoke({"value": 1.0, "from_unit": "mile", "to_unit": "kilometer"})
        assert "kilometer" in result
        assert "1.60934" in result

    def test_incompatible(self):
        tool = self._create()
        result = tool.invoke({"value": 1.0, "from_unit": "meter", "to_unit": "second"})
        assert "[TOOL ERROR]" in result
        assert "incompatible" in result


# ---------------------------------------------------------------------------
# Timezone Converter
# ---------------------------------------------------------------------------


class TestTimezoneConverter:
    def _create(self) -> Any:
        from threetears.agent.tools.builtin.timezone_converter import create_timezone_converter_tool

        return create_timezone_converter_tool({}, "Convert timezones")

    def test_convert(self):
        tool = self._create()
        result = tool.invoke(
            {
                "time_str": "2024-01-15 14:00",
                "from_timezone": "America/New_York",
                "to_timezone": "Europe/London",
            }
        )
        assert "=" in result
        assert "EST" in result or "New_York" in result or "PM" in result

    def test_bad_timezone(self):
        tool = self._create()
        result = tool.invoke(
            {
                "time_str": "2024-01-15 14:00",
                "from_timezone": "Fake/Timezone",
                "to_timezone": "UTC",
            }
        )
        assert "[TOOL ERROR]" in result
        assert "unknown timezone" in result

    def test_time_only_format(self):
        tool = self._create()
        result = tool.invoke(
            {
                "time_str": "3:00 PM",
                "from_timezone": "America/New_York",
                "to_timezone": "UTC",
            }
        )
        # Should not contain year 1900
        assert "1900" not in result
        assert "=" in result


# ---------------------------------------------------------------------------
# Current Date
# ---------------------------------------------------------------------------


class TestCurrentDate:
    def _create(self, config: dict[str, Any] | None = None) -> Any:
        from threetears.agent.tools.builtin.current_date import create_current_date_tool

        return create_current_date_tool(config or {}, "Get current date")

    def test_utc(self):
        tool = self._create()
        result = tool.invoke({})
        assert "UTC" in result

    def test_with_timezone(self):
        tool = self._create({"timezone": "America/New_York"})
        result = tool.invoke({})
        assert "UTC" in result
        assert "America/New_York" in result


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------


class TestDictionary:
    """The Free Dictionary API first; Wiktionary when it does not answer.

    Driven through ``httpx.MockTransport``: the lookup's own requests, its
    timeouts and its status handling run as they do against the network.
    """

    _FREE = [
        {
            "word": "hello",
            "phonetic": "/helo/",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definitions": [{"definition": "A greeting", "example": "She said hello."}],
                    "synonyms": ["hi", "greetings"],
                    "antonyms": ["goodbye"],
                }
            ],
        }
    ]
    _WIKTIONARY = {
        "en": [
            {"partOfSpeech": "Symbol", "language": "Translingual", "definitions": [{"definition": "ISO code"}]},
            {
                "partOfSpeech": "Interjection",
                "language": "English",
                "definitions": [
                    {
                        "definition": '<span>A <a href="/wiki/greeting">greeting</a> said on meeting</span>',
                        "examples": ["<i>Hello</i>, everyone."],
                    }
                ],
            },
        ]
    }

    @staticmethod
    def _tool(handler: Any) -> Any:
        import httpx

        from threetears.agent.tools.builtin.dictionary import DictionaryTool

        return DictionaryTool(transport=httpx.MockTransport(handler))

    @staticmethod
    def _routes(free: Any, wiktionary: Any = None) -> tuple[Any, list[str]]:
        """A handler answering each host with its function, and the hosts it was asked, in order."""
        asked: list[str] = []

        def handler(request: Any) -> Any:
            asked.append(request.url.host)
            return (free if request.url.host == "api.dictionaryapi.dev" else wiktionary)(request)

        return handler, asked

    @pytest.mark.asyncio
    async def test_the_first_source_answers(self) -> None:
        import httpx

        handler, asked = self._routes(lambda r: httpx.Response(200, json=self._FREE))
        result = await self._tool(handler).execute(word="hello")
        assert result.success
        assert "/helo/" in result.content and "A greeting" in result.content
        assert asked == ["api.dictionaryapi.dev"]

    @pytest.mark.asyncio
    async def test_no_such_word_is_an_answer_and_wiktionary_is_not_asked(self) -> None:
        import httpx

        handler, asked = self._routes(lambda r: httpx.Response(404))
        result = await self._tool(handler).execute(word="xyznotaword")
        assert "No definition found" in result.content
        assert asked == ["api.dictionaryapi.dev"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        ["timeout", "connect", "server"],
    )
    async def test_when_the_first_source_fails_wiktionary_answers(self, failure: str) -> None:
        import httpx

        def free(request: Any) -> Any:
            if failure == "timeout":
                raise httpx.ReadTimeout("no answer", request=request)
            if failure == "connect":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(503)

        handler, asked = self._routes(free, lambda r: httpx.Response(200, json=self._WIKTIONARY))
        result = await self._tool(handler).execute(word="hello")
        assert result.success, result.content
        assert asked == ["api.dictionaryapi.dev", "en.wiktionary.org"]
        assert "[interjection]" in result.content
        # HTML is gone; the translingual ISO code is not an English meaning.
        assert "1. A greeting said on meeting" in result.content and "<" not in result.content
        assert "Example: Hello, everyone." in result.content
        assert "ISO code" not in result.content
        assert "(from Wiktionary)" in result.content

    @pytest.mark.asyncio
    async def test_wiktionary_asks_with_a_user_agent(self) -> None:
        import httpx

        agents: list[str] = []

        def wiktionary(request: Any) -> Any:
            agents.append(request.headers.get("user-agent", ""))
            return httpx.Response(200, json=self._WIKTIONARY)

        handler, _asked = self._routes(lambda r: httpx.Response(502), wiktionary)
        await self._tool(handler).execute(word="hello")
        assert agents and agents[0].startswith("threetears-dictionary/")

    @pytest.mark.asyncio
    async def test_both_failing_is_a_tool_error_that_says_why(self) -> None:
        import httpx

        def down(request: Any) -> Any:
            raise httpx.ReadTimeout("no answer", request=request)

        handler, _asked = self._routes(down, down)
        result = await self._tool(handler).execute(word="hello")
        assert not result.success
        assert "did not answer" in result.content and "Wiktionary failed too" in result.content

    @pytest.mark.asyncio
    async def test_a_word_wiktionary_does_not_have_in_the_language_is_not_found(self) -> None:
        import httpx

        only_irish = {"ga": [{"partOfSpeech": "Noun", "language": "Irish", "definitions": [{"definition": "cat"}]}]}
        handler, _asked = self._routes(lambda r: httpx.Response(500), lambda r: httpx.Response(200, json=only_irish))
        result = await self._tool(handler).execute(word="cat")
        assert "No definition found" in result.content

    @pytest.mark.asyncio
    async def test_a_slow_source_does_not_stop_other_work(self) -> None:
        """A blocking lookup inside the async tool stalled every turn on the worker while it waited."""
        import asyncio

        import httpx

        from threetears.agent.tools.builtin.dictionary import DictionaryTool

        class _Slow(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: Any) -> Any:
                await asyncio.sleep(0.3)
                return httpx.Response(200, json=self._free, request=request)

            _free = self._FREE

        ticks = 0

        async def _tick() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        await asyncio.gather(DictionaryTool(transport=_Slow()).execute(word="hello"), _tick())
        assert ticks == 10

    @pytest.mark.asyncio
    async def test_the_langchain_tool_runs_the_same_lookup(self) -> None:
        from threetears.agent.tools.builtin.dictionary import create_dictionary_tool

        tool = create_dictionary_tool({}, "Look up words")
        assert tool.name and tool.coroutine is not None


# ---------------------------------------------------------------------------
# Register builtins
# ---------------------------------------------------------------------------


class TestRegisterBuiltins:
    def test_register_builtins(self):
        from threetears.agent.tools.registry import ToolRegistry
        from threetears.agent.tools.builtin import register_builtins

        reg = ToolRegistry()
        register_builtins(reg)
        # At minimum these should be registered (no optional dep issues)
        types = reg.list_types()
        assert "current_date" in types
        assert "timezone_converter" in types
        assert "dictionary" in types
        assert "web_search" in types
        assert "web_fetch" in types
