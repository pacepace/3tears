"""Bank statement import for the household budget: reads the CSV file a bank exports into :class:`Transaction` rows.

``read_statement(text)`` (or ``read_statement_file(path)``) handles the layouts our banks use:

- a header row naming the columns, in any order: a date column (``Date`` or ``Transaction Date``), a description
  (``Description``, ``Details`` or ``Narrative``), and either one signed ``Amount`` column or a ``Paid out`` and a
  ``Paid in`` column;
- dates as ``YYYY-MM-DD`` or with slashes, ``DD/MM/YYYY`` (UK banks write the day first); a slashed date that cannot be
  read that way (a 13th month) is read the other way round;
- amounts in pounds, with or without ``£`` and thousands commas, negative as ``-12.50`` or ``(12.50)``.

Amounts are kept as whole pence: money out of the account is negative. Blank rows are skipped.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATE_COLUMNS = ("date", "transaction date")
DESCRIPTION_COLUMNS = ("description", "details", "narrative")
#: Whether a slashed date is read day first (``04/03/2026`` is 4 March) when the caller does not say.
DAY_FIRST = True


class StatementError(ValueError):
    """A statement this importer cannot read: an unknown layout, or a row whose date or amount does not parse."""


@dataclass(frozen=True)
class Transaction:
    date: date
    description: str
    pence: int


def parse_date(text: str, *, day_first: bool = DAY_FIRST) -> date:
    """A statement date: ``2026-03-04``, or ``04/03/2026`` read day first unless ``day_first`` is false."""
    text = text.strip()
    slashed = ("%d/%m/%Y", "%m/%d/%Y") if day_first else ("%m/%d/%Y", "%d/%m/%Y")
    for layout in ("%Y-%m-%d", *slashed):
        try:
            return datetime.strptime(text, layout).date()
        except ValueError:
            continue
    raise StatementError(f"date {text!r} is not YYYY-MM-DD or DD/MM/YYYY")


def parse_pence(text: str) -> int:
    """An amount in pounds as whole pence: ``"£1,234.50"`` is 123450 and ``"(12.00)"`` is -1200."""
    cleaned = text.strip().replace("£", "").replace(",", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        raise StatementError(f"amount {text!r} is not a number of pounds") from None
    pence = int((amount * 100).to_integral_value())
    return -pence if negative else pence


def _column(header: list[str], names: tuple[str, ...]) -> int | None:
    lowered = [name.strip().lower() for name in header]
    return next((lowered.index(name) for name in names if name in lowered), None)


def read_statement(text: str, *, day_first: bool = DAY_FIRST) -> list[Transaction]:
    """The transactions in one exported statement, in the file's order."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    when, what = _column(header, DATE_COLUMNS), _column(header, DESCRIPTION_COLUMNS)
    amount, paid_out, paid_in = (_column(header, (name,)) for name in ("amount", "paid out", "paid in"))
    if when is None or what is None or (amount is None and (paid_out is None or paid_in is None)):
        raise StatementError(f"unknown statement layout: {', '.join(header)}")
    found = []
    for row in body:
        if amount is not None:
            pence = parse_pence(row[amount])
        else:
            out, into = row[paid_out].strip(), row[paid_in].strip()
            pence = parse_pence(into) if into else -parse_pence(out)
        found.append(Transaction(parse_date(row[when], day_first=day_first), row[what].strip(), pence))
    return found


def read_statement_file(path: str | Path, *, day_first: bool = DAY_FIRST) -> list[Transaction]:
    """The transactions in a statement file."""
    return read_statement(Path(path).read_text(encoding="utf-8-sig"), day_first=day_first)
