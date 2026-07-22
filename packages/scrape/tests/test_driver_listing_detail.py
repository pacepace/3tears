"""Unit tests for ListingDetailDriver -- listing table + per-row detail-page
fetch, merged into one synthetic table.

All tests are fully mocked -- no real network calls. The real, live proof
against Arizona/Delaware/Kansas/Vermont/Maine's actual pages lives in
tests/e2e/test_warn_act_eval_loop_live.py (faidh repo).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from threetears.scrape.drivers.listing_detail import (
    ListingDetailDriver,
    ListingDetailDriverError,
    _extract_listing_row_fields,
    _parse_definition_list,
)

_LISTING_FIELD_COLUMNS = {0: "employer", 1: "city", 2: "notice_date"}
_DETAIL_LINK_COLUMN = 0
_DETAIL_FIELD_LABELS = {"notice_date": "Notice Date", "affected_count": "Number of Employees Affected"}


def _listing_html(rows: list[tuple[str, str, str, str]]) -> str:
    """*rows*: (employer, href, city, notice_date) tuples."""
    body_rows = "".join(
        f"<tr><td><a href='{href}'>{employer}</a></td><td>{city}</td><td>{notice_date}</td></tr>"
        for employer, href, city, notice_date in rows
    )
    return f"<html><body><table><tbody>{body_rows}</tbody></table></body></html>"


def _detail_html(*, notice_date: str = "", affected_count: str = "") -> str:
    return (
        "<html><body><div class='definition-list'>"
        "<h3 class='definition-list__title'>Company Name</h3><p class='definition-list__definition'>Acme Corp</p>"
        f"<h3 class='definition-list__title'>Notice Date</h3><p class='definition-list__definition'>{notice_date}</p>"
        "<h3 class='definition-list__title'>Number of Employees Affected</h3>"
        f"<p class='definition-list__definition'>{affected_count}</p>"
        "</div></body></html>"
    )


def _driver(handler, **kwargs) -> ListingDetailDriver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ListingDetailDriver(
        row_selector="table tr",
        listing_field_columns=_LISTING_FIELD_COLUMNS,
        detail_link_column=_DETAIL_LINK_COLUMN,
        detail_field_labels=_DETAIL_FIELD_LABELS,
        client=client,
        pace_delay_seconds=0.0,
        **kwargs,
    )


# ===========================================================================
# _extract_listing_row_fields
# ===========================================================================


class TestExtractListingRowFields:
    def test_reads_requested_column_indices(self):
        from bs4 import BeautifulSoup

        row = BeautifulSoup("<tr><td>Acme Corp</td><td>Springfield</td><td>Jun 1, 2026</td></tr>", "html.parser").tr
        result = _extract_listing_row_fields(row, {0: "employer", 1: "city", 2: "notice_date"})
        assert result == {"employer": "Acme Corp", "city": "Springfield", "notice_date": "Jun 1, 2026"}

    def test_an_empty_cell_is_absent_not_an_empty_string(self):
        from bs4 import BeautifulSoup

        row = BeautifulSoup("<tr><td>Acme Corp</td><td></td></tr>", "html.parser").tr
        result = _extract_listing_row_fields(row, {0: "employer", 1: "city"})
        assert result == {"employer": "Acme Corp"}

    def test_an_out_of_range_column_index_is_skipped_not_a_crash(self):
        from bs4 import BeautifulSoup

        row = BeautifulSoup("<tr><td>Acme Corp</td></tr>", "html.parser").tr
        result = _extract_listing_row_fields(row, {0: "employer", 5: "county"})
        assert result == {"employer": "Acme Corp"}


# ===========================================================================
# _parse_definition_list
# ===========================================================================


class TestParseDefinitionList:
    def test_reads_labeled_fields_by_exact_label_text(self):
        html = _detail_html(notice_date="Jun 17, 2026", affected_count="81")
        result = _parse_definition_list(html, _DETAIL_FIELD_LABELS)
        assert result == {"notice_date": "Jun 17, 2026", "affected_count": "81"}

    def test_a_blank_value_is_absent_not_an_empty_string(self):
        html = _detail_html(notice_date="", affected_count="81")
        result = _parse_definition_list(html, _DETAIL_FIELD_LABELS)
        assert result == {"affected_count": "81"}

    def test_a_label_not_present_at_all_is_absent(self):
        html = "<html><body><div class='definition-list'></div></body></html>"
        result = _parse_definition_list(html, _DETAIL_FIELD_LABELS)
        assert result == {}

    def test_a_requested_label_with_no_matching_field_name_is_ignored(self):
        html = _detail_html(notice_date="Jun 17, 2026")
        result = _parse_definition_list(html, {"notice_date": "Notice Date", "reason": "Reason -- Comments"})
        assert result == {"notice_date": "Jun 17, 2026"}


# ===========================================================================
# ListingDetailDriver
# ===========================================================================


class TestListingDetailDriver:
    def test_name(self):
        driver = ListingDetailDriver(
            row_selector="tr", listing_field_columns={}, detail_link_column=0, detail_field_labels={}
        )
        assert driver.name == "listing_detail"

    async def test_merges_listing_and_detail_fields_preferring_detail(self):
        listing = _listing_html([("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026")])

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                return httpx.Response(200, content=_detail_html(notice_date="Jun 2, 2026", affected_count="42").encode())
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html
        assert "Springfield" in page.html
        assert "42" in page.html
        # detail's own notice_date (Jun 2) wins over the listing's (Jun 1) -- prefer-detail rule
        assert "Jun 2, 2026" in page.html
        assert "Jun 1, 2026" not in page.html

    async def test_falls_back_to_listing_value_when_detail_field_is_blank(self):
        """The real Kansas gap this driver exists to handle: a detail page's own
        Notice Date can be blank even though the listing row had a real one."""
        listing = _listing_html([("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026")])

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                return httpx.Response(200, content=_detail_html(notice_date="", affected_count="42").encode())
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Jun 1, 2026" in page.html  # fell back to the listing's own value
        assert "42" in page.html

    async def test_a_field_absent_on_both_layers_is_absent_not_fabricated(self):
        """One record genuinely has no city (neither layer provides it); a sibling
        record does -- the city column exists (union across records), but the
        first record's own city cell must be empty, never a fabricated value."""
        listing = (
            "<html><body><table><tbody>"
            "<tr><td><a href='/notices/1'>Acme Corp</a></td></tr>"
            "<tr><td><a href='/notices/2'>Beta LLC</a></td><td>Springfield</td></tr>"
            "</tbody></table></body></html>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1") or str(request.url).endswith("/notices/2"):
                return httpx.Response(200, content=_detail_html(affected_count="42").encode())
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        import re

        rows = re.findall(r"<tr>(.*?)</tr>", page.html)
        assert len(rows) == 3  # header + two data rows
        acme_row = next(r for r in rows if "Acme Corp" in r)
        assert "Springfield" not in acme_row  # never fabricated for the record that lacks it

    async def test_a_failed_detail_fetch_still_yields_the_record_from_listing_fields_alone(self):
        listing = _listing_html([("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026")])

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html
        assert "Springfield" in page.html
        assert "Jun 1, 2026" in page.html

    async def test_a_detail_fetch_returning_an_error_status_still_yields_the_record_from_listing_fields_alone(self):
        listing = _listing_html([("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026")])

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                return httpx.Response(500, content=b"server error")
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html
        assert "Springfield" in page.html

    async def test_a_row_with_no_detail_link_still_yields_listing_fields_alone(self):
        listing = "<html><body><table><tbody><tr><td>Acme Corp</td><td>Springfield</td><td>Jun 1, 2026</td></tr></tbody></table></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html
        assert "Springfield" in page.html

    async def test_a_detail_link_column_index_out_of_range_for_a_row_still_yields_listing_fields_alone(self):
        """Distinct from "cell exists but has no <a>" (the test above) -- here
        the configured detail_link_column index doesn't even exist on this
        row at all (fewer <td> cells than expected)."""
        listing = "<html><body><table><tbody><tr><td>Acme Corp</td></tr></tbody></table></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=listing.encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        detail_fetch_count = 0
        original_get = client.get

        async def counting_get(*args, **kwargs):
            nonlocal detail_fetch_count
            detail_fetch_count += 1
            return await original_get(*args, **kwargs)

        client.get = counting_get  # type: ignore[method-assign]
        driver = ListingDetailDriver(
            row_selector="table tr",
            listing_field_columns={0: "employer"},
            detail_link_column=5,  # this row only has 1 <td> (index 0)
            detail_field_labels=_DETAIL_FIELD_LABELS,
            client=client,
            pace_delay_seconds=0.0,
        )
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html
        assert detail_fetch_count == 1  # only the listing fetch -- no detail fetch attempted at all

    async def test_multiple_rows_all_resolved(self):
        listing = _listing_html(
            [
                ("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026"),
                ("Beta LLC", "/notices/2", "Shelbyville", "Jun 2, 2026"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                return httpx.Response(200, content=_detail_html(affected_count="10").encode())
            if str(request.url).endswith("/notices/2"):
                return httpx.Response(200, content=_detail_html(affected_count="20").encode())
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler)
        page = await driver.render("https://example.gov/warn")

        assert "Acme Corp" in page.html and "10" in page.html
        assert "Beta LLC" in page.html and "20" in page.html

    async def test_max_rows_caps_how_many_are_resolved(self):
        rows = [(f"Corp {i}", f"/notices/{i}", "City", "Jun 1, 2026") for i in range(5)]
        listing = _listing_html(rows)
        detail_fetch_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal detail_fetch_count
            if "/notices/" in str(request.url):
                detail_fetch_count += 1
                return httpx.Response(200, content=_detail_html(affected_count="1").encode())
            return httpx.Response(200, content=listing.encode())

        driver = _driver(handler, max_rows=2)
        await driver.render("https://example.gov/warn")

        assert detail_fetch_count == 2

    async def test_pace_delay_is_awaited_between_detail_fetches(self):
        listing = _listing_html(
            [
                ("Acme Corp", "/notices/1", "City", "Jun 1, 2026"),
                ("Beta LLC", "/notices/2", "City", "Jun 2, 2026"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if "/notices/" in str(request.url):
                return httpx.Response(200, content=_detail_html(affected_count="1").encode())
            return httpx.Response(200, content=listing.encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = ListingDetailDriver(
            row_selector="table tr",
            listing_field_columns=_LISTING_FIELD_COLUMNS,
            detail_link_column=_DETAIL_LINK_COLUMN,
            detail_field_labels=_DETAIL_FIELD_LABELS,
            client=client,
            pace_delay_seconds=1.0,
        )
        sleep_mock = AsyncMock()
        import threetears.scrape.drivers.listing_detail as listing_detail_module

        original_sleep = listing_detail_module.asyncio.sleep
        listing_detail_module.asyncio.sleep = sleep_mock
        try:
            await driver.render("https://example.gov/warn")
        finally:
            listing_detail_module.asyncio.sleep = original_sleep

        # 2 rows -> only 1 delay (no trailing pace-wait after the LAST row -- nothing
        # left to protect a server from once every detail fetch has already happened)
        assert sleep_mock.await_count == 1
        assert sleep_mock.await_args.args == (1.0,)

    async def test_a_429_detail_response_backs_off_with_extra_delay_and_falls_back_to_listing_fields(self):
        """Live-found (scrape-task-07 follow-up): the real Arizona/Delaware/
        Kansas/Vermont/Maine servers returned intermittent 429s at a 0.1s
        pace -- a 429 is the server's own explicit "slow down" signal, not
        a generic error, and must extend the NEXT pause rather than just
        moving on at the normal rate that triggered it. Two rows so the
        429'd first row's own delay is actually observable (no trailing
        delay after the last row)."""
        listing = _listing_html(
            [
                ("Acme Corp", "/notices/1", "Springfield", "Jun 1, 2026"),
                ("Beta LLC", "/notices/2", "Shelbyville", "Jun 2, 2026"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/notices/1"):
                return httpx.Response(429, content=b"too many requests")
            if str(request.url).endswith("/notices/2"):
                return httpx.Response(200, content=_detail_html(affected_count="1").encode())
            return httpx.Response(200, content=listing.encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = ListingDetailDriver(
            row_selector="table tr",
            listing_field_columns=_LISTING_FIELD_COLUMNS,
            detail_link_column=_DETAIL_LINK_COLUMN,
            detail_field_labels=_DETAIL_FIELD_LABELS,
            client=client,
            pace_delay_seconds=0.5,
        )
        sleep_mock = AsyncMock()
        import threetears.scrape.drivers.listing_detail as listing_detail_module

        original_sleep = listing_detail_module.asyncio.sleep
        listing_detail_module.asyncio.sleep = sleep_mock
        try:
            page = await driver.render("https://example.gov/warn")
        finally:
            listing_detail_module.asyncio.sleep = original_sleep

        # normal pace (0.5) plus the exact extra rate-limited backoff (2.0), pinned precisely
        assert sleep_mock.await_args_list[0].args == (2.5,)
        # never fabricated affected_count from a 429'd detail page -- listing fields alone still yield the record
        assert "Acme Corp" in page.html
        assert "Springfield" in page.html

    async def test_listing_fetch_failure_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b"server error")

        driver = _driver(handler)
        with pytest.raises(ListingDetailDriverError):
            await driver.render("https://example.gov/warn")

    async def test_listing_fetch_transport_failure_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        driver = _driver(handler)
        with pytest.raises(ListingDetailDriverError):
            await driver.render("https://example.gov/warn")

    async def test_default_pace_delay_is_nonzero(self):
        """This module's own docstring, 'Politeness, on by default' -- unlike
        MultiDocumentDriver, this brand-new driver defaults to NOT hammering an
        unprotected government server."""
        driver = ListingDetailDriver(
            row_selector="tr", listing_field_columns={}, detail_link_column=0, detail_field_labels={}
        )
        assert driver._pace_delay_seconds > 0  # noqa: SLF001 -- asserting the documented default itself
