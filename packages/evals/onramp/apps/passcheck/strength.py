"""Password strength for the sign-up form: a score from 0 to 100 and the band the form shows beside the box.

:func:`rate` scores a password, given the username it is for, by these rules:

- **Length.** 4 points a character, up to 60 points (15 characters).
- **Variety.** Four kinds of character count: lower-case letters, capitals, digits, and anything else. 10 points for
  every kind present beyond the first, up to 30.
- **Common words.** 25 points off when the password contains any word on the common list (``data/common.txt``:
  passwords and words that lead every breach list), ignoring case: ``Summer``, ``SUMMER`` and ``summer`` are all the
  common word ``summer``.
- **The username.** 25 points off when the password contains the username (of 3 characters or more), ignoring case.
- **Runs.** 10 points off for every run of three or more letters or digits that go up or down by one (``abc``,
  ``321``, ``XYZ``).
- **Repeats.** 10 points off for every run of three or more of the same character (``aaa``, ``!!!``).

The score is then kept between 0 and 100. Its band is ``weak`` below 40, ``fair`` from 40, ``good`` from 60 and
``strong`` from 80. The sign-up form refuses a weak password.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

COMMON = Path(__file__).with_name("data") / "common.txt"

#: The least score of each band, from the top.
BANDS = (("strong", 80), ("good", 60), ("fair", 40), ("weak", 0))
_REPEATS = re.compile(r"(.)\1{2,}")


def _common_words(path: Path = COMMON) -> frozenset[str]:
    words = (line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(word for word in words if word and not word.startswith("#"))


_COMMON_WORDS = _common_words()


@dataclass(frozen=True)
class Rating:
    score: int
    band: str
    reasons: tuple[str, ...]


def _kinds(password: str) -> int:
    kinds = [
        any(c in string.ascii_lowercase for c in password),
        any(c in string.ascii_uppercase for c in password),
        any(c in string.digits for c in password),
        any(not c.isascii() or not c.isalnum() for c in password),
    ]
    return sum(kinds)


def _runs(password: str) -> int:
    """Runs of three or more letters or digits going up or down by one, each counted once."""
    runs, length, step = 0, 1, 0
    lowered = password.lower()
    for before, here in zip(lowered, lowered[1:]):
        gap = ord(here) - ord(before)
        pair = before + here
        if gap in (1, -1) and pair.isascii() and pair.isalnum() and before.isdigit() == here.isdigit():
            length = length + 1 if gap == step else 2
            step = gap
            if length == 3:
                runs += 1
        else:
            length, step = 1, 0
    return runs


def band_of(score: int) -> str:
    """The band a score falls in."""
    return next(name for name, least in BANDS if score >= least)


def rate(password: str, username: str = "") -> Rating:
    """``password``'s score, band and the reasons points were taken off."""
    reasons = []
    score = min(4 * len(password), 60)
    score += min(10 * max(_kinds(password) - 1, 0), 30)
    found = sorted(word for word in _COMMON_WORDS if word in password)
    if found:
        score -= 25
        reasons.append(f"contains the common word {found[0]!r}")
    if len(username) >= 3 and username.lower() in password.lower():
        score -= 25
        reasons.append("contains the username")
    runs = _runs(password)
    if runs:
        score -= 10 * runs
        reasons.append(f"{runs} run(s) like abc or 123")
    repeats = len(_REPEATS.findall(password))
    if repeats:
        score -= 10 * repeats
        reasons.append(f"{repeats} repeated character(s)")
    score = max(0, min(score, 100))
    return Rating(score=score, band=band_of(score), reasons=tuple(reasons))
