"""Names for the document store: every upload is filed under a name built from its upload date and its title.

``stored_name(title, uploaded, extension)`` gives the name, for example ``2026-03-04-spring-menu-costings.pdf``:

- the upload date, ``YYYY-MM-DD``, then the title's slug, then the extension in lower case;
- the slug is the title in lower case ASCII: accented letters keep their base letter (``é`` is ``e``, ``ñ`` is ``n``),
  and any other character that is not a letter or a digit separates words;
- words are joined by single hyphens, with none at either end;
- a slug longer than :data:`MAX_SLUG` characters is cut back to the last whole word that fits;
- a title with nothing left after all that is ``untitled``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

#: The longest slug a name carries; longer titles are cut back to a whole word.
MAX_SLUG = 60

_SEPARATORS = re.compile(r"[^a-z0-9]+")


def slug(title: str) -> str:
    """A title as the slug part of a stored name."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    words = [word for word in _SEPARATORS.split(folded.lower()) if word]
    text = "-".join(words)
    if len(text) > MAX_SLUG:
        cut = text.rfind("-", 0, MAX_SLUG + 1)
        text = text[:cut] if cut > 0 else text[:MAX_SLUG]
    return text or "untitled"


def stored_name(title: str, uploaded: str | date, extension: str) -> str:
    """The name an upload is filed under."""
    day = uploaded if isinstance(uploaded, date) else date.fromisoformat(uploaded)
    ext = extension.lower().lstrip(".")
    if not ext.isalnum():
        raise ValueError(f"extension {extension!r} is not letters and digits")
    return f"{day.isoformat()}-{slug(title)}.{ext}"
