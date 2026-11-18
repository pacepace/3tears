"""Daily traffic figures from the web server's access logs: hits and bytes for each page.

The server writes Apache's combined log format, one request a line::

    203.0.113.9 - - [02/Nov/2026:08:14:07 +0000] "GET /menu?lang=en HTTP/1.1" 200 5120 "-" "Mozilla/5.0 ..."

The fields are the client, two unused fields, the time, the request line (method, path, protocol), the status, the
size of the response body in bytes, the referring page and the browser. A field the server has no value for is
logged as ``-``: a request typed into the address bar has ``-`` for its referring page, and a response with no body
(a ``304 Not Modified``, a ``204 No Content``, the answer to a ``HEAD``) has ``-`` for its size. Such a response is a
hit like any other, of 0 bytes.

:func:`tally` counts a day's lines by page. A page is the request's path without its query string, so ``/menu`` and
``/menu?lang=en`` are one page. Each page gets its hits (every request for it, whatever its status), the bytes sent
for it, and how many of its requests were answered ``404`` or with a ``5xx`` error. A line that is not in the
combined format at all (a line cut short when the disk filled, a stray trace line) is counted in ``unparsed`` and
otherwise left out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

LOGS = Path(__file__).with_name("data") / "logs"

_LINE = re.compile(
    r"^(?P<client>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "
    r'"(?P<method>[A-Z]+) (?P<target>\S+) [^"]*" '
    r"(?P<status>\d{3}) (?P<size>\d+) "
    r'"(?P<referer>[^"]*)" "(?P<agent>[^"]*)"\s*$'
)


@dataclass(frozen=True)
class Request:
    method: str
    page: str
    status: int
    size: int
    referer: str | None


@dataclass
class PageStats:
    hits: int = 0
    bytes: int = 0
    not_found: int = 0
    errors: int = 0


@dataclass
class Tally:
    pages: dict[str, PageStats] = field(default_factory=dict)
    unparsed: int = 0

    def busiest(self, n: int = 5) -> list[tuple[str, int]]:
        """The ``n`` pages with the most hits, most first; pages with as many hits go in alphabetical order."""
        ranked = sorted(self.pages.items(), key=lambda item: (-item[1].hits, item[0]))
        return [(page, stats.hits) for page, stats in ranked[:n]]


def _value(field: str) -> str | None:
    """A log field's value, or ``None`` for ``-``, the log's mark for a field with no value."""
    return None if field == "-" else field


def parse(line: str) -> Request | None:
    """One log line as a :class:`Request`, or ``None`` when it is not in the combined format."""
    match = _LINE.match(line)
    if match is None:
        return None
    page = match["target"].split("?", 1)[0]
    return Request(
        method=match["method"],
        page=page,
        status=int(match["status"]),
        size=int(_value(match["size"]) or 0),
        referer=_value(match["referer"]),
    )


def tally(lines: Iterable[str]) -> Tally:
    """Hits, bytes, 404s and errors by page over ``lines``; blank lines are skipped."""
    result = Tally()
    for line in lines:
        if not line.strip():
            continue
        request = parse(line)
        if request is None:
            result.unparsed += 1
            continue
        stats = result.pages.setdefault(request.page, PageStats())
        stats.hits += 1
        stats.bytes += request.size
        stats.not_found += request.status == 404
        stats.errors += request.status >= 500
    return result


def tally_file(path: Path) -> Tally:
    """:func:`tally` over a log file."""
    with path.open(encoding="utf-8") as lines:
        return tally(lines)


def day_files(directory: Path = LOGS) -> list[Path]:
    """The day's log files in ``directory``, ``access-YYYY-MM-DD.log``, oldest first."""
    return sorted(directory.glob("access-*.log"))
