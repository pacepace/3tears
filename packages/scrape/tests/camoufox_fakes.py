"""In-memory stand-ins for the Playwright surface ``CamoufoxDriver`` drives.

Shared by ``test_driver_camoufox.py`` and the backend-agnostic ``test_driver_contract.py``, so
they live in a support module under public names rather than as one test module's privates a
sibling test reaches into. No real browser is launched and no Camoufox binary is needed.
"""

from __future__ import annotations

__all__ = [
    "FakeCamoufoxBrowser",
    "FakeCamoufoxLocator",
    "FakeCamoufoxMouse",
    "FakeCamoufoxNetworkResponse",
    "FakeCamoufoxPage",
    "FakeCamoufoxRequest",
    "FakeCamoufoxResponse",
]


# parity-exempt: hand-rolled subset stub of Playwright's third-party Locator (only scroll_into_view_if_needed, the only surface CamoufoxDriver calls)
class FakeCamoufoxLocator:
    def __init__(self, selector: str, *, scroll_into_view_calls: list[dict], scroll_into_view_exc=None) -> None:
        self._selector = selector
        self._scroll_into_view_calls = scroll_into_view_calls
        self._scroll_into_view_exc = scroll_into_view_exc

    async def scroll_into_view_if_needed(self, *, timeout=None):
        self._scroll_into_view_calls.append({"selector": self._selector, "timeout": timeout})
        if self._scroll_into_view_exc is not None:
            raise self._scroll_into_view_exc


# parity-exempt: hand-rolled subset stub of Playwright's third-party Mouse (only wheel, the only surface CamoufoxDriver calls)
class FakeCamoufoxMouse:
    def __init__(self, *, wheel_calls: list[dict]) -> None:
        self._wheel_calls = wheel_calls

    async def wheel(self, delta_x, delta_y):
        self._wheel_calls.append({"delta_x": delta_x, "delta_y": delta_y})


# parity-exempt: hand-rolled subset stub of Playwright's third-party Page (only goto/wait_for_selector/click/fill/wait_for_timeout/locator/mouse/viewport_size/content/url/close/on, the only surface CamoufoxDriver calls)
class FakeCamoufoxPage:
    def __init__(
        self,
        *,
        goto_result=None,
        goto_exc=None,
        wait_for_exc=None,
        click_exc=None,
        fill_exc=None,
        scroll_into_view_exc=None,
        evaluate_returns=None,
        evaluate_exc=None,
        html="<html>ok</html>",
        url=None,
        network_responses=None,
        viewport_size=None,
    ):
        self._goto_result = goto_result
        self._goto_exc = goto_exc
        self._wait_for_exc = wait_for_exc
        self._click_exc = click_exc
        self._fill_exc = fill_exc
        self._scroll_into_view_exc = scroll_into_view_exc
        self._evaluate_returns = list(evaluate_returns) if evaluate_returns is not None else []
        self._evaluate_exc = evaluate_exc
        self._html = html
        self.url = url or "https://example.gov/final"
        self.viewport_size = viewport_size or {"width": 1920, "height": 1080}
        self.goto_calls: list[dict] = []
        self.wait_for_calls: list[dict] = []
        self.click_calls: list[dict] = []
        self.fill_calls: list[dict] = []
        self.wait_for_timeout_calls: list[int] = []
        self.scroll_into_view_calls: list[dict] = []
        self.wheel_calls: list[dict] = []
        self.evaluate_calls: list[str] = []
        self.mouse = FakeCamoufoxMouse(wheel_calls=self.wheel_calls)
        self.closed = False
        # Simulates the responses Playwright would have fired via page.on("response", ...)
        # during navigation -- goto() replays these into the registered handler.
        self._network_responses = network_responses or []
        self._response_handler = None

    async def goto(self, url, *, timeout=None, wait_until=None):
        self.goto_calls.append({"url": url, "timeout": timeout, "wait_until": wait_until})
        if self._goto_exc is not None:
            raise self._goto_exc
        if self._response_handler is not None:
            for resp in self._network_responses:
                self._response_handler(resp)
        return self._goto_result

    async def wait_for_selector(self, selector, *, timeout=None):
        self.wait_for_calls.append({"selector": selector, "timeout": timeout})
        if self._wait_for_exc is not None:
            raise self._wait_for_exc

    async def click(self, selector, *, timeout=None):
        self.click_calls.append({"selector": selector, "timeout": timeout})
        if self._click_exc is not None:
            raise self._click_exc

    async def fill(self, selector, value, *, timeout=None):
        self.fill_calls.append({"selector": selector, "value": value, "timeout": timeout})
        if self._fill_exc is not None:
            raise self._fill_exc

    async def wait_for_timeout(self, ms):
        self.wait_for_timeout_calls.append(ms)

    def locator(self, selector):
        return FakeCamoufoxLocator(
            selector,
            scroll_into_view_calls=self.scroll_into_view_calls,
            scroll_into_view_exc=self._scroll_into_view_exc,
        )

    async def evaluate(self, expression):
        self.evaluate_calls.append(expression)
        if self._evaluate_exc is not None:
            raise self._evaluate_exc
        return self._evaluate_returns.pop(0) if self._evaluate_returns else None

    async def content(self):
        return self._html

    async def close(self):
        self.closed = True

    def on(self, event, handler):
        if event == "response":
            self._response_handler = handler


# parity-exempt: hand-rolled subset stub of Playwright's third-party Request (only .resource_type/.method/.post_data, the only attributes CamoufoxDriver reads)
class FakeCamoufoxRequest:
    def __init__(self, resource_type: str, method: str = "GET", post_data: str | None = None) -> None:
        self.resource_type = resource_type
        self.method = method
        # Playwright's own name and its own "no body" value -- None for every GET.
        self.post_data = post_data


# parity-exempt: hand-rolled subset stub of Playwright's third-party Response used for network-capture (only .request/.status/.url/.text()/.body()/.all_headers(), the only surface CamoufoxDriver's capture_network path reads)
class FakeCamoufoxNetworkResponse:
    def __init__(
        self,
        *,
        url: str,
        status: int = 200,
        resource_type: str = "xhr",
        body: str = "{}",
        content_type: str = "application/json",
        text_exc: Exception | None = None,
        headers_exc: Exception | None = None,
        method: str = "GET",
        post_data: str | None = None,
    ):
        self.url = url
        self.status = status
        self.request = FakeCamoufoxRequest(resource_type, method=method, post_data=post_data)
        self._body = body
        self._content_type = content_type
        self._text_exc = text_exc
        self._headers_exc = headers_exc

    async def text(self):
        if self._text_exc is not None:
            raise self._text_exc
        if isinstance(self._body, bytes):
            return self._body.decode()  # mirrors Playwright: UTF-8 only, raises UnicodeDecodeError
        return self._body

    async def body(self):
        if isinstance(self._body, bytes):
            return self._body
        return self._body.encode()

    async def all_headers(self):
        if self._headers_exc is not None:
            raise self._headers_exc
        return {"content-type": self._content_type}


# parity-exempt: hand-rolled subset stub of Playwright's third-party Response (only .status, the only attribute CamoufoxDriver reads)
class FakeCamoufoxResponse:
    def __init__(self, status: int) -> None:
        self.status = status


# parity-exempt: hand-rolled subset stub of Playwright's third-party Browser (only new_page(), the only method CamoufoxDriver calls)
class FakeCamoufoxBrowser:
    def __init__(self, page: FakeCamoufoxPage | list[FakeCamoufoxPage]) -> None:
        self._pages = page if isinstance(page, list) else [page]
        self.new_page_calls = 0

    async def new_page(self):
        # Repeats the last page if new_page() is called more times than pages
        # were supplied -- single-page callers never need to think about this.
        result = self._pages[min(self.new_page_calls, len(self._pages) - 1)]
        self.new_page_calls += 1
        return result
