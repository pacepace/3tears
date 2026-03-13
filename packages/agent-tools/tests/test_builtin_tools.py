"""Tests for built-in tool modules."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch, MagicMock

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
    def _create(self) -> Any:
        from threetears.agent.tools.builtin.dictionary import create_dictionary_tool

        return create_dictionary_tool({}, "Look up words")

    def test_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
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

        tool = self._create()
        with patch("threetears.agent.tools.builtin.dictionary.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = tool.invoke({"word": "hello"})
            assert "hello" in result
            assert "/helo/" in result
            assert "greeting" in result.lower()

    def test_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        tool = self._create()
        with patch("threetears.agent.tools.builtin.dictionary.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = tool.invoke({"word": "xyznotaword"})
            assert "No definition found" in result


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
