"""glossator: turn raw glossary terms into entries a printed glossary can sort and link.

    python glossator.py TERMS.jsonl OUT.jsonl
    python glossator.py - -            # read stdin, write stdout

Each input line is ``{"id": ..., "lang": "en" | "fr" | "de" | "es", "term": ...}``. Each output line keeps the ``id``
and ``lang`` and adds:

- ``headword``: the term as the glossary prints it — trimmed, inner spaces collapsed, first letter upper case, every
  other letter as written (accents included);
- ``slug``: the anchor it is linked by — lower case ASCII, accents folded to their base letter, words joined by ``-``.

A line that is not a JSON object with all three keys is reported on stderr and skipped; the exit code is then 1.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata

LANGS = ("en", "fr", "de", "es")


def _fold(text: str) -> str:
    """Accents folded to their base letters: ``é`` to ``e``, ``ß`` left as it is."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def headword(term: str, lang: str) -> str:
    words = " ".join(term.split())
    return words[:1].upper() + words[1:]


def slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _fold(term).lower()).strip("-")


def gloss(entry: dict) -> dict:
    if entry["lang"] not in LANGS:
        raise ValueError(f"unknown lang {entry['lang']!r}")
    return {
        "id": entry["id"],
        "lang": entry["lang"],
        "headword": headword(entry["term"], entry["lang"]),
        "slug": slug(entry["term"]),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: glossator.py TERMS.jsonl OUT.jsonl  (- for stdin or stdout)\n")
        return 2
    source = sys.stdin if argv[0] == "-" else open(argv[0], encoding="utf-8")
    sink = sys.stdout if argv[1] == "-" else open(argv[1], "w", encoding="utf-8")
    code = 0
    for number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            sink.write(json.dumps(gloss(json.loads(line)), ensure_ascii=False) + "\n")
        except (ValueError, KeyError, TypeError) as bad:
            sys.stderr.write(f"line {number}: {bad}\n")
            code = 1
    sink.flush()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
