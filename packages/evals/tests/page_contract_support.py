"""What a report's HTML page may load or run, read by parsing it: the contract every page test holds the page to.

The principle: nothing an author or a model wrote can make the page run or load anything. The package itself loads
exactly one thing, the kit's fonts, through three links it writes into the page's head: a preconnect to each Google
Fonts host and the stylesheet naming the three IBM Plex faces, the same three the kit's reference memos carry
(``docs/eval-report-kit/references/*.src.html``). A ``<link>`` is allowed only when it is one of those three, attribute
for attribute; every other element that runs, embeds or loads, every ``on*`` attribute and every ``javascript:`` URL
is refused, and the page holds no ``<script>`` at all.

What "runs" means here: a ``script`` element; any element that embeds or loads another document or resource
(``iframe``, ``object``, ``embed``, ``link`` other than the font links, ``base``, ``img``, ``meta`` with
``http-equiv``, and inside the inline SVG ``image``, ``use`` and ``foreignObject``); an event-handler attribute
(``on*``); and any attribute value that is a ``javascript:`` URL. The page links otherwise only to places on itself
(``#F1``, ``#T3``), so the last is checked on every attribute rather than on a list of URL attributes — except
``data-*``, which a browser never navigates to or runs.
"""

from __future__ import annotations

from html.parser import HTMLParser

__all__ = [
    "ACTIVE_ELEMENTS",
    "FONT_HOSTS",
    "FONT_LINKS",
    "FONT_LINK_MARKUP",
    "Page",
    "font_links",
    "parsed",
    "what_runs",
    "without_font_links",
]

#: The hosts the package's font links name, and no other.
FONT_HOSTS = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")

#: The package's three font links, attribute for attribute, in the order the head carries them.
FONT_LINKS: tuple[tuple[tuple[str, str | None], ...], ...] = (
    (("rel", "preconnect"), ("href", "https://fonts.googleapis.com")),
    (("rel", "preconnect"), ("href", "https://fonts.gstatic.com"), ("crossorigin", None)),
    (
        ("rel", "stylesheet"),
        (
            "href",
            "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
            "&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400"
            "&display=swap",
        ),
    ),
)

#: The same three links as the reference memos write them, verbatim (their source's lines 3 and 4).
FONT_LINK_MARKUP = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">'
)

#: Elements that execute, embed or load something. A ``link`` is among them; only the package's font links pass.
ACTIVE_ELEMENTS = frozenset(
    {"script", "iframe", "object", "embed", "link", "base", "img", "frame", "frameset", "image", "use", "foreignobject"}
)


class Page(HTMLParser):
    """Every start tag and its attributes, and every text node, as the parser read them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, list[tuple[str, str | None]]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, attrs))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def parsed(page: str) -> Page:
    """``page`` read by the parser."""
    parser = Page()
    parser.feed(page)
    parser.close()
    return parser


def font_links(page: str) -> list[tuple[tuple[str, str | None], ...]]:
    """Every ``<link>`` on the page, as its attributes in order."""
    return [tuple(attrs) for tag, attrs in parsed(page).tags if tag == "link"]


def what_runs(page: str) -> list[str]:
    """Everything on the page that would execute or load, the package's font links aside — empty for a page that
    keeps the contract."""
    found: list[str] = []
    for tag, attrs in parsed(page).tags:
        if tag in ACTIVE_ELEMENTS and not (tag == "link" and tuple(attrs) in FONT_LINKS):
            found.append(f"<{tag}>")
        if tag == "meta" and any(name == "http-equiv" for name, _ in attrs):
            found.append("<meta http-equiv>")
        for name, value in attrs:
            if name.startswith("on"):
                found.append(f"{tag}[{name}]")
            if not name.startswith("data-") and value is not None and "javascript:" in value.replace(" ", "").lower():
                found.append(f"{tag}[{name}=javascript:]")
    return found


def without_font_links(page: str) -> str:
    """``page`` with the package's font links taken out of its head, once; the page unchanged where its head does
    not carry them exactly, so a check of the rest still sees whatever is there."""
    head, marker, body = page.partition("</head>")
    return head.replace(FONT_LINK_MARKUP, "", 1) + marker + body
