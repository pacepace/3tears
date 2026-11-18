"""Postal addresses for the label printer: an address as a customer typed it, as the lines a parcel label carries.

:func:`label_lines` reads one address written on a single line with its parts separated by commas, as our checkout
joins its address fields (``"flat 2, 14 mill ln, leeds, ls6 2ab"``), and returns the label's lines:

- Everything is in capitals, full stops are dropped, runs of spaces become one space, and empty parts are dropped.
- A flat or unit at the start of the address (``Flat 2``, ``Apt 2``, ``Apartment 2``, ``Unit 4``, any case, with or
  without a full stop) is the label's first line on its own: ``FLAT 2`` (an apartment is a flat), or ``UNIT 4``. The
  flat's number or name runs up to the first comma.
- Each remaining part is one line, in the order typed. A street type abbreviated at the end of a part is written in
  full (``Mill Ln`` is ``MILL LANE``; :data:`STREET_TYPES` lists them).
- The postcode, the last thing in the address, is its own last line, with one space before its last three
  characters (``ls62ab`` is ``LS6 2AB``). An address with no postcode is refused with :class:`AddressError`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ADDRESSES = Path(__file__).with_name("data") / "order_addresses.jsonl"

#: Abbreviated street types, and how a label writes them.
STREET_TYPES = {
    "RD": "ROAD",
    "ST": "STREET",
    "AVE": "AVENUE",
    "AV": "AVENUE",
    "LN": "LANE",
    "DR": "DRIVE",
    "CL": "CLOSE",
    "CRES": "CRESCENT",
    "CT": "COURT",
    "GDNS": "GARDENS",
    "GRN": "GREEN",
    "GRO": "GROVE",
    "PL": "PLACE",
    "SQ": "SQUARE",
    "TER": "TERRACE",
    "TERR": "TERRACE",
}
_FLAT_KINDS = {"FLAT": "FLAT", "APT": "FLAT", "APARTMENT": "FLAT", "UNIT": "UNIT"}

_POSTCODE = re.compile(r"[\s,]*([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\s*$", re.IGNORECASE)
_FLAT = re.compile(r"^\s*(flat|apartment|apt|unit)\.?\s+(.+),\s*", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


class AddressError(ValueError):
    """An address the label printer cannot take."""


def _tidy(part: str) -> str:
    return _SPACES.sub(" ", part.replace(".", " ")).strip().upper()


def _street(line: str) -> str:
    words = line.split(" ")
    if len(words) > 1 and words[-1] in STREET_TYPES:
        words[-1] = STREET_TYPES[words[-1]]
    return " ".join(words)


def label_lines(raw: str) -> list[str]:
    """The lines a label carries for ``raw``, a comma-separated address with a postcode last."""
    postcode = _POSTCODE.search(raw)
    if postcode is None:
        raise AddressError(f"no postcode at the end of {raw!r}")
    rest = raw[: postcode.start()]
    lines = []
    flat = _FLAT.match(rest)
    if flat is not None:
        lines.append(f"{_FLAT_KINDS[flat.group(1).upper()]} {_tidy(flat.group(2))}")
        rest = rest[flat.end() :]
    lines.extend(_street(line) for line in (_tidy(part) for part in rest.split(",")) if line)
    lines.append(f"{postcode.group(1).upper()} {postcode.group(2).upper()}")
    return lines


def label(raw: str) -> str:
    """The label for ``raw``, one line under another."""
    return "\n".join(label_lines(raw))


def load_addresses(path: Path = ADDRESSES) -> list[str]:
    """The ``address`` of every order in a file of one JSON object a line."""
    rows = [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines() if text.strip()]
    return [row["address"] for row in rows]
