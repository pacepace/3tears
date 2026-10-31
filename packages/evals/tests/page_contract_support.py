"""What a report's HTML page may load or run, read by parsing it: the contract every page test holds the page to.

The principle: nothing an author or a model wrote can make the page run, embed or load anything. The package itself
writes exactly two things that do. Its fonts: three links in the head, a preconnect to each Google Fonts host and the
stylesheet naming the three IBM Plex faces, the same three the kit's reference memos carry
(``docs/eval-report-kit/references/*.src.html``); those pass only in the head and only attribute for attribute. And
the kit's script: one ``<script>``, with no attribute, as the last element of the body, whose text is byte for byte
the ``kit/kit.js`` the package ships (:data:`KIT_JS`), which draws each chart from the JSON in its ``data-chart``.
That one passes only alone: a second ``<script>``, a copy of the kit beside it, a script with a ``src`` or any other
attribute, one in the head, one followed by another element, or one whose text differs by a byte is refused, and so
is the kit's own then.

What :func:`what_runs` refuses, on any element wherever it sits:

- an element that runs, embeds, loads or submits (:data:`ACTIVE_ELEMENTS`): ``script`` other than the kit's,
  ``iframe``, ``frame``, ``frameset``, ``object``, ``embed``, ``portal``, ``base``, ``img``, ``picture``, ``video``,
  ``audio``, ``source``, ``track``, ``form``, a ``link`` other than the font links in the head, and inside an inline
  SVG ``image``, ``use`` and ``foreignObject``; an ``input`` of ``type=image``; a ``meta`` with ``http-equiv``;
- an attribute that loads or navigates (:data:`LOADING_ATTRIBUTES`): ``src``, ``srcset``, ``poster``, ``data``,
  ``action``, ``formaction``, ``xlink:href``, ``background``, ``ping``, and an ``href`` that is not a link to a place
  on the page itself (``#F1``, ``#T3``) and not one of the font links in the head;
- a ``style`` attribute whose value holds ``url(``, and a ``<style>`` element whose text holds ``url(`` or
  ``@import``;
- an event-handler attribute (``on*``);
- any attribute value that is a ``javascript:`` URL, checked on every attribute rather than on a list of URL
  attributes, except ``data-*``, which a browser never navigates to or runs: a chart's JSON is data the kit reads and
  sets as text.

The head is what the HTML parser would put there: everything before ``</head>``, ``<body>`` or the first element that
belongs only in a body.
"""

from __future__ import annotations

from html.parser import HTMLParser
from importlib.resources import files

__all__ = [
    "ACTIVE_ELEMENTS",
    "FONT_LINKS",
    "FONT_LINK_MARKUP",
    "KIT_JS",
    "LOADING_ATTRIBUTES",
    "Page",
    "font_links",
    "kit_script",
    "parsed",
    "what_runs",
    "without_font_links",
    "without_kit_script",
]

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

#: The kit's script as the package ships it: the one script a page may carry, verbatim.
KIT_JS = files("threetears.evals.analysis.report").joinpath("kit", "kit.js").read_text(encoding="utf-8")

#: Elements that execute, embed, load or submit something, the SVG ones by the parser's lower-cased names. A ``link``
#: is among them; only the package's font links, in the head, pass.
ACTIVE_ELEMENTS = frozenset(
    {
        "script",
        "iframe",
        "frame",
        "frameset",
        "object",
        "embed",
        "portal",
        "link",
        "base",
        "img",
        "picture",
        "video",
        "audio",
        "source",
        "track",
        "form",
        "image",
        "use",
        "foreignobject",
    }
)

#: Attributes that load a resource or navigate somewhere, on whatever element carries them. ``href`` is checked
#: apart: a link to a place on the page is the page's own, and the font links in the head are the package's.
LOADING_ATTRIBUTES = frozenset(
    {"src", "srcset", "poster", "data", "action", "formaction", "xlink:href", "background", "ping"}
)

#: Elements the HTML parser keeps in the head; any other start tag opens the body.
_HEAD_ELEMENTS = frozenset({"html", "head", "title", "meta", "link", "style", "base", "script", "noscript", "template"})


class Page(HTMLParser):
    """Every start tag and its attributes, and every text node, as the parser read them; which start tags sit in the
    head, the text of every ``<style>``, and the text of every ``<script>`` by its start tag's place in :attr:`tags`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, list[tuple[str, str | None]]]] = []
        self.in_head: list[bool] = []
        self.text: list[str] = []
        self.styles: list[str] = []
        self.scripts: dict[int, str] = {}
        self._head_open = True
        self._in_style = False
        self._in_script: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _HEAD_ELEMENTS:
            self._head_open = False
        self.tags.append((tag, attrs))
        self.in_head.append(self._head_open)
        self._in_style = tag == "style"
        if tag == "script":
            self._in_script = len(self.tags) - 1
            self.scripts[self._in_script] = ""

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _HEAD_ELEMENTS:
            self._head_open = False
        self.tags.append((tag, attrs))
        self.in_head.append(self._head_open)

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._head_open = False
        if tag == "style":
            self._in_style = False
        if tag == "script":
            self._in_script = None

    def handle_data(self, data: str) -> None:
        if self._in_script is not None:
            self.scripts[self._in_script] += data
        elif self._in_style:
            self.styles.append(data)
        else:
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


def kit_script(read: Page) -> int | None:
    """Where the page's one kit script is among its start tags: the page's only ``<script>``, with no attribute, in the
    body, the last element on the page, its text :data:`KIT_JS` exactly; ``None`` where the page has no such script."""
    if len(read.scripts) != 1:
        return None
    ((at, text),) = read.scripts.items()
    _, attrs = read.tags[at]
    if attrs or read.in_head[at] or at != len(read.tags) - 1 or text != KIT_JS:
        return None
    return at


def what_runs(page: str) -> list[str]:
    """Everything on the page that would execute, embed, load or navigate away, the package's font links in the head
    and its one kit script aside — empty for a page that keeps the contract."""
    found: list[str] = []
    read = parsed(page)
    kit = kit_script(read)
    for at, ((tag, attrs), in_head) in enumerate(zip(read.tags, read.in_head, strict=True)):
        font_link = tag == "link" and in_head and tuple(attrs) in FONT_LINKS
        if tag in ACTIVE_ELEMENTS and not font_link and at != kit:
            found.append(f"<{tag}>")
        if tag == "input" and any(name == "type" and (value or "").strip().lower() == "image" for name, value in attrs):
            found.append("<input type=image>")
        if tag == "meta" and any(name == "http-equiv" for name, _ in attrs):
            found.append("<meta http-equiv>")
        for name, value in attrs:
            if name.startswith("on"):
                found.append(f"{tag}[{name}]")
            if name in LOADING_ATTRIBUTES:
                found.append(f"{tag}[{name}]")
            if name == "href" and not font_link and not (value or "").startswith("#"):
                found.append(f"{tag}[href]")
            if name == "style" and value is not None and "url(" in _compact(value):
                found.append(f"{tag}[style=url()]")
            if not name.startswith("data-") and value is not None and "javascript:" in _compact(value):
                found.append(f"{tag}[{name}=javascript:]")
    for style in read.styles:
        compact = _compact(style)
        if "url(" in compact or "@import" in compact:
            found.append("<style> url()/@import")
    return found


def _compact(value: str) -> str:
    """``value`` in lower case with every whitespace character taken out, as a browser reads a URL's scheme."""
    return "".join(value.split()).lower()


def without_font_links(page: str) -> str:
    """``page`` with the package's font links taken out of its head, once; the page unchanged where its head does
    not carry them exactly, so a check of the rest still sees whatever is there."""
    head, marker, body = page.partition("</head>")
    return head.replace(FONT_LINK_MARKUP, "", 1) + marker + body


def without_kit_script(page: str) -> str:
    """``page`` with the kit's script taken out, once: the ``<script>`` element whose text is :data:`KIT_JS`; the page
    unchanged where it does not carry it exactly, so a check of the rest still sees whatever is there."""
    return page.replace(f"<script>{KIT_JS}</script>", "", 1)
