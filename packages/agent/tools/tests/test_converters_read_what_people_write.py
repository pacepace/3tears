"""The small tools read what people write and say how they read it.

A time with no date takes the source zone's day, read from a frozen clock at
03:21 UTC: at that instant it is still Friday in Chicago and already Saturday in
UTC, so a test run at 15:00 UTC would pass with the server's day in place.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from threetears.agent.tools.builtin.calculator import CalculatorTool
from threetears.agent.tools.builtin.date_math import DateMathTool
from threetears.agent.tools.builtin.timezone_converter import TimezoneConverterTool
from threetears.agent.tools.builtin.unit_converter import UnitConverterTool

#: Saturday 10 October 2026, 03:21 UTC: Friday 9 October, 22:21 in Chicago.
FROZEN = datetime(2026, 10, 10, 3, 21, tzinfo=UTC)


def _frozen() -> datetime:
    return FROZEN


def _tz(person_zone: str | None = None, **kwargs: object) -> str:
    tool = TimezoneConverterTool(timezone=person_zone, clock=_frozen)
    return asyncio.run(tool.run(**kwargs)).content


def _units(**kwargs: object) -> str:
    return asyncio.run(UnitConverterTool().run(**kwargs)).content


def _calc(expression: str) -> str:
    return asyncio.run(CalculatorTool().run(expression=expression)).content


def _dates(person_zone: str | None = "America/Chicago", **kwargs: object) -> str:
    return asyncio.run(DateMathTool(timezone=person_zone, clock=_frozen).run(**kwargs)).content


class TestATimeWithNoDateTakesTheSourceZonesDay:
    def test_3pm_chicago_at_0321_utc_is_friday_in_chicago(self) -> None:
        result = _tz(time_str="3:00 PM", from_timezone="America/Chicago", to_timezone="Asia/Tokyo")

        assert result.startswith("3:00 PM Friday, October 9, 2026 in America/Chicago (CDT, UTC-05:00) = ")
        assert "5:00 AM Saturday, October 10, 2026 in Asia/Tokyo (JST, UTC+09:00), the next day" in result
        assert "no date given, so today in America/Chicago, Friday, October 9" in result

    def test_the_same_time_in_tokyo_is_already_saturday(self) -> None:
        """good traffic: a zone already on the server's day keeps that day."""
        result = _tz(time_str="9am", from_timezone="Asia/Tokyo", to_timezone="UTC")

        assert result.startswith("9:00 AM Saturday, October 10, 2026 in Asia/Tokyo")


class TestWhatPeopleWriteIsRead:
    def test_3pm_and_city_names(self) -> None:
        result = _tz(time_str="3pm", from_timezone="chicago", to_timezone="tokyo")

        assert result.startswith("3:00 PM Friday, October 9, 2026 in America/Chicago")
        assert "chicago read as America/Chicago" in result

    def test_pst_in_july_is_los_angeles_on_pdt_and_says_so(self) -> None:
        result = _tz(time_str="2026-07-04 12:00", from_timezone="PST", to_timezone="UTC")

        assert "in America/Los_Angeles (PDT, UTC-07:00)" in result
        assert "7:00 PM Saturday, July 4, 2026 in UTC" in result
        assert "PST read as America/Los_Angeles (US Pacific)" in result

    def test_est_for_a_july_date_is_new_york_on_edt(self) -> None:
        result = _tz(time_str="2026-07-04 9am", from_timezone="EST", to_timezone="GMT")

        assert "in America/New_York (EDT, UTC-04:00)" in result
        assert "1:00 PM Saturday, July 4, 2026 in Etc/GMT" in result

    def test_ist_is_india_and_the_mapping_is_stated(self) -> None:
        result = _tz(time_str="11:30pm", from_timezone="PST", to_timezone="IST")

        assert "in Asia/Kolkata (IST, UTC+05:30), the next day" in result
        assert "IST read as Asia/Kolkata (India; for Israel or Ireland name the city)" in result

    def test_an_iana_name_in_lower_case(self) -> None:
        result = _tz(time_str="noon", from_timezone="america/chicago", to_timezone="europe/london")

        assert "12:00 PM Friday, October 9, 2026 in America/Chicago" in result

    def test_a_left_out_zone_is_the_persons_and_the_result_says_so(self) -> None:
        result = _tz(person_zone="America/Chicago", time_str="3pm", to_timezone="UTC")

        assert "in America/Chicago (CDT, UTC-05:00)" in result
        assert "no from zone given, so the person's zone, America/Chicago, was used" in result

    def test_tomorrow_counts_from_the_source_zones_day(self) -> None:
        result = _tz(time_str="tomorrow 9:00", from_timezone="America/Chicago", to_timezone="UTC")

        assert result.startswith("9:00 AM Saturday, October 10, 2026 in America/Chicago")

    def test_a_time_with_its_own_offset(self) -> None:
        result = _tz(time_str="2026-10-10T14:00:00Z", from_timezone="UTC", to_timezone="Asia/Kolkata")

        assert "7:30 PM Saturday, October 10, 2026 in Asia/Kolkata" in result

    def test_a_skipped_clock_time_is_refused(self) -> None:
        result = _tz(time_str="2026-03-08 02:30", from_timezone="America/New_York", to_timezone="UTC")

        assert result.startswith("[TOOL ERROR]")
        assert "does not happen in America/New_York" in result

    def test_no_zone_at_all_is_refused_with_what_to_write(self) -> None:
        result = _tz(time_str="3pm")

        assert result.startswith("[TOOL ERROR]")


class TestUnits:
    def test_five_feet_ten_inches(self) -> None:
        result = _units(quantity="5 feet 10 inches", to_unit="cm")

        assert result.startswith("5 feet + 10 inches = 177.8 cm")

    def test_five_feet_ten_inches_written_in_from_unit(self) -> None:
        assert _units(from_unit="5 feet 10 inches", to_unit="cm").startswith("5 feet + 10 inches = 177.8 cm")

    def test_f_to_c_is_temperature(self) -> None:
        result = _units(value=350, from_unit="F", to_unit="C")

        assert result.startswith("350 F = 176.667 C")
        assert "F read as degrees Fahrenheit" in result

    def test_f_to_uf_stays_farads(self) -> None:
        result = _units(value=1, from_unit="F", to_unit="uF")

        assert result.startswith("1 F = 1e+06 uF")
        assert "F read as farads" in result

    def test_capitalised_words(self) -> None:
        assert _units(value=100, from_unit="Celsius", to_unit="Fahrenheit").startswith("100 Celsius = 212 Fahrenheit")
        assert _units(value=26.2, from_unit="Miles", to_unit="km").startswith("26.2 Miles = 42.1648 km")

    def test_the_result_says_it_is_rounded_and_which_gallon(self) -> None:
        result = _units(value=3, from_unit="gallons", to_unit="liters")

        assert "rounded to 6 significant figures" in result
        assert "gallons is the US measure; write imperial_gallon for the UK one" in result

    def test_currency_changes_nothing_and_says_so(self) -> None:
        result = _units(value=10, from_unit="USD", to_unit="EUR")

        assert result.startswith("[NO CHANGE] unit_converter: USD is money.")

    def test_pound_the_weight_is_not_money(self) -> None:
        """the money guard's good traffic: a pound of mass converts."""
        result = _units(value=2, from_unit="pound", to_unit="kg")

        assert result.startswith("2 pound = 0.907185 kg")


class TestCalculator:
    def test_the_result_shows_the_expression(self) -> None:
        assert _calc("84.50*0.15") == "84.50*0.15 = 12.675"

    def test_15_percent_of(self) -> None:
        assert _calc("15% of 84.50") == "15% of 84.50 = 12.675"

    def test_money_and_thousands(self) -> None:
        assert _calc("$22.50*3.5") == "$22.50*3.5 = 78.75"
        assert _calc("1,000*2") == "1,000*2 = 2000"
        assert _calc("2 * 1,000") == "2 * 1,000 = 2000"

    def test_a_comma_between_arguments_is_not_a_thousands_comma(self) -> None:
        """the thousands rule's good traffic: function arguments keep their commas."""
        assert _calc("max(1, 500)") == "max(1, 500) = 500"
        assert _calc("gcd(12,18)") == "gcd(12,18) = 6"
        assert _calc("max(5,250)") == "max(5,250) = 250"

    def test_a_percent_between_numbers_is_still_the_remainder(self) -> None:
        """the percent rule's good traffic: 10 % 3 is the remainder."""
        assert _calc("10 % 3") == "10 % 3 = 1"

    def test_float_noise_is_not_shown(self) -> None:
        assert _calc("0.1+0.2") == "0.1+0.2 = 0.3"

    def test_trig_says_radians_and_has_degrees(self) -> None:
        assert _calc("sin(30)").endswith(
            "(sin, cos, tan and their inverses work in radians; sind, cosd, tand, asind, acosd, atand use degrees)"
        )
        assert _calc("sind(30)") == "sind(30) = 0.5"

    def test_log_says_it_is_natural(self) -> None:
        result = _calc("log(100)")

        assert result.startswith("log(100) = 4.60517018599")
        assert "log is the natural log (base e); log10 is base 10" in result
        assert _calc("log10(100)") == "log10(100) = 2"

    def test_text_is_not_a_calculation(self) -> None:
        assert _calc("'a'*3").startswith("[TOOL ERROR] calculator: evaluate failed — 'a'*3: that is not a calculation")

    def test_an_error_says_what_to_write(self) -> None:
        result = _calc("(1,2)")

        assert result.startswith("[TOOL ERROR]")
        assert "For example 84.50*0.15, 15% of 84.50" in result


class TestDateMath:
    def test_today_is_the_persons_day(self) -> None:
        result = _dates()

        assert result.startswith("Friday, October 9, 2026: day 282 of 2026")
        assert "today is Friday, October 9, 2026 in America/Chicago" in result

    def test_ninety_days_from_today(self) -> None:
        assert _dates(add="90 days").startswith("Friday, October 9, 2026 + 90 days = Thursday, January 7, 2027.")

    def test_days_until(self) -> None:
        assert _dates(until="Dec 25").startswith(
            "Friday, December 25, 2026 is 77 days (11 weeks) after Friday, October 9, 2026."
        )

    def test_the_weekday_of_a_date(self) -> None:
        result = _dates(date="2027-07-04")

        assert result.startswith("Sunday, July 4, 2027")

    def test_a_month_on_from_the_31st_lands_on_the_months_last_day_and_says_so(self) -> None:
        result = _dates(date="2026-01-31", add="1 month")

        assert result.startswith("Saturday, January 31, 2026 + 1 month = Saturday, February 28, 2026.")
        assert "its last day was used" in result

    def test_subtracting(self) -> None:
        assert _dates(add="-2 weeks").startswith("Friday, October 9, 2026 - 2 weeks = Friday, September 25, 2026.")

    def test_a_unit_it_does_not_count_in_is_refused(self) -> None:
        assert _dates(add="3 fish").startswith("[TOOL ERROR] date_math: read failed")

    @pytest.mark.parametrize("zone", ["Asia/Tokyo", "UTC"])
    def test_a_zone_already_on_saturday_says_saturday(self, zone: str) -> None:
        assert _dates(person_zone=zone).startswith("Saturday, October 10, 2026")
