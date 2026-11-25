"""A spam score for the shared support inbox, from 0 (plainly fine) to 10 (plainly spam).

Messages scoring 5 or more go to the review folder instead of the queue. The score is the sum of these signals,
kept between 0 and 10:

- a link from a stranger, +3: the body has a link, and either the sender is not in our contacts or the reply-to
  address is on another domain than the sender's;
- an attachment a support agent should never open (``.exe``, ``.scr``, ``.js``, ``.bat``, ``.iso``), +4;
- money words (``free``, ``winner``, ``prize``, ``cash``, ``crypto``, ``urgent``, ``guaranteed``, ``bitcoin``), +1
  for each different one in the subject or body, at most +3;
- a shouted subject, +1.5: at least eight letters, and more than 60% of them capitals;
- three or more exclamation marks in the subject and body together, +1;
- a sender in our contacts, -2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
WORD = re.compile(r"[a-z]+")
MONEY_WORDS = frozenset({"free", "winner", "prize", "cash", "crypto", "urgent", "guaranteed", "bitcoin"})
RISKY_EXTENSIONS = (".exe", ".scr", ".js", ".bat", ".iso")

STRANGER_LINK = 3.0
RISKY_ATTACHMENT = 4.0
MONEY_WORD = 1.0
MONEY_WORDS_AT_MOST = 3.0
SHOUTING = 1.5
EXCLAIMING = 1.0
KNOWN_SENDER = -2.0
REVIEW_AT = 5.0


@dataclass(frozen=True)
class Message:
    sender: str
    subject: str
    body: str
    reply_to: str | None = None
    attachments: tuple[str, ...] = field(default=())

    @classmethod
    def from_row(cls, row: dict) -> Message:
        return cls(row["sender"], row["subject"], row["body"], row.get("reply_to"), tuple(row.get("attachments", ())))


def domain(address: str) -> str:
    return address.rsplit("@", 1)[-1].lower()


def load_contacts(path: str | Path) -> frozenset[str]:
    """The contacts file: one address per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip().lower() for line in lines if line.strip())


def shouted(subject: str) -> bool:
    letters = [c for c in subject if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) > 0.6 * len(letters)


def score(message: Message, contacts: frozenset[str]) -> float:
    """The message's spam score, from 0 to 10."""
    known = message.sender.lower() in contacts
    elsewhere = message.reply_to is not None and domain(message.reply_to) != domain(message.sender)
    text = f"{message.subject}\n{message.body}"
    total = 0.0
    if LINK.search(message.body) and (not known or elsewhere):
        total += STRANGER_LINK
    if any(name.lower().endswith(RISKY_EXTENSIONS) for name in message.attachments):
        total += RISKY_ATTACHMENT
    total += min(MONEY_WORD * len(MONEY_WORDS & set(WORD.findall(text.lower()))), MONEY_WORDS_AT_MOST)
    if shouted(message.subject):
        total += SHOUTING
    if text.count("!") >= 3:
        total += EXCLAIMING
    if known:
        total += KNOWN_SENDER
    return min(max(total, 0.0), 10.0)


def to_review(message: Message, contacts: frozenset[str]) -> bool:
    """Whether the message goes to the review folder."""
    return score(message, contacts) >= REVIEW_AT


def score_file(path: str | Path, contacts: frozenset[str]) -> dict[str, float]:
    """Every message's score in a file of messages (one JSON message per line, with an ``id``), by id."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: score(Message.from_row(row), contacts) for row in rows}
