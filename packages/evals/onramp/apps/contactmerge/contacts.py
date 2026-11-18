"""One address book from our contact exports: the CRM, the newsletter list and the event sign-up sheets.

Each export is a CSV file with a header row. The columns read, in any order and any case, are ``name`` (or
``full name``), ``email`` (or ``e-mail``), ``phone`` (or ``mobile``) and ``company``; other columns are ignored.

Two rows are the same person when they share an email address or a phone number:

- an email address is the same address whatever its case or the spaces around it (``Jo.Bloggs@Example.com`` and
  ``jo.bloggs@example.com `` reach one inbox);
- a phone number is compared by its digits, with a UK number written nationally (``07700 900123``) the same as written
  internationally (``+44 7700 900123``, ``0044 7700 900123``).

Names are not compared: two different people can share a name, and one person writes theirs several ways. ``merge``
joins the rows into people transitively (A shares an email with B and B a phone with C, so all three are one person)
and keeps every source row on the person it joined.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

NAME_COLUMNS = ("name", "full name")
EMAIL_COLUMNS = ("email", "e-mail")
PHONE_COLUMNS = ("phone", "mobile")
COMPANY_COLUMNS = ("company",)


@dataclass(frozen=True)
class Contact:
    name: str
    email: str = ""
    phone: str = ""
    company: str = ""
    source: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str], source: str = "") -> Contact:
        """A contact from one CSV row (or any mapping of column name to value)."""
        lowered = {key.strip().lower(): (value or "").strip() for key, value in row.items() if key}

        def first(names: tuple[str, ...]) -> str:
            return next((lowered[name] for name in names if lowered.get(name)), "")

        return cls(first(NAME_COLUMNS), first(EMAIL_COLUMNS), first(PHONE_COLUMNS), first(COMPANY_COLUMNS), source)


@dataclass
class Person:
    """One person in the merged address book, with every export row that is them."""

    rows: list[Contact] = field(default_factory=list)

    @property
    def name(self) -> str:
        return max((row.name for row in self.rows), key=len, default="")

    @property
    def emails(self) -> list[str]:
        return sorted({row.email for row in self.rows if row.email})

    @property
    def phones(self) -> list[str]:
        return sorted({phone_key(row.phone) for row in self.rows if row.phone})


def email_key(email: str) -> str:
    """The form of an email address two rows are compared by."""
    return email.strip()


def phone_key(phone: str) -> str:
    """A phone number's digits in international form: ``07700 900123`` and ``+44 7700 900123`` are ``447700900123``."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = "44" + digits[1:]
    return digits


def keys(contact: Contact) -> set[str]:
    """What a contact can be matched on: its email address and its phone number, each where it has one."""
    found = set()
    if contact.email.strip():
        found.add("email:" + email_key(contact.email))
    if phone_key(contact.phone):
        found.add("phone:" + phone_key(contact.phone))
    return found


def same_person(a: Contact, b: Contact) -> bool:
    """Whether two rows are one person: they share an email address or a phone number."""
    return bool(keys(a) & keys(b))


def merge(contacts: list[Contact]) -> list[Person]:
    """The people in ``contacts``, joined transitively, in the order each person's first row appears."""
    parent = list(range(len(contacts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, contact in enumerate(contacts):
        for key in keys(contact):
            if key in owner:
                parent[root(i)] = root(owner[key])
            else:
                owner[key] = i
    people: dict[int, Person] = {}
    for i, contact in enumerate(contacts):
        people.setdefault(root(i), Person()).rows.append(contact)
    return list(people.values())


def read_export(path: str | Path) -> list[Contact]:
    """The contacts in one export file, each tagged with the file's name as its source."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            Contact.from_row(row, path.stem)
            for row in csv.DictReader(handle)
            if any((v or "").strip() for v in row.values())
        ]


def merge_exports(paths: list[str | Path]) -> list[Person]:
    """The people across several export files."""
    return merge([contact for path in paths for contact in read_export(path)])
