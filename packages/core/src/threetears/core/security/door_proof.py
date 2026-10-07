"""the per-door proof switch: a dated window in code, ``optional`` before its date, ``required`` from it.

a door that gains a proof (a token, a signed envelope) cannot require it from the release that adds
it: every pod that has not upgraded yet would be refused. under expand / migrate / contract (owner
ruling 2026-10-07) the door declares a :class:`DoorWindow` in code, beside the door: it accepts a
sender without the proof until a fixed date, and refuses one from that date.

**the date, not the traffic, ends the window.** every sender without the proof is counted per door
and claimed sender, so an operator can see who still sends the old form, but the count never moves
the date: a forger sending one unsigned message a day would otherwise hold the door open forever.
the date is chosen against the inventory of deployed sender versions, and no window is longer than
:data:`MAX_PROOF_WINDOW`.

**config can only tighten.** an operator who has confirmed every sender upgraded may require a door
before its date; no config spelling makes a door optional past its date, so a config edit or rollback
cannot reopen a door. moving a date later is a code change, reviewed like any other.

**a window past its date is dead code.** :func:`expired_windows` lets a repo's gate fail until the
window and the door's optional path are deleted.

the switch governs ABSENCE only. a proof that is presented is the door's own to verify, in either
mode, and a presented proof that fails verification is refused whatever the mode: ``optional`` never
means "accept a bad proof".

every absent proof is logged as a decision naming the door and the claimed sender, and counted on the
``door_proof_absent`` prometheus counter (labels ``door``, ``outcome``). the claimed sender is what the
message says it is, unverified by definition: a lead for the operator, not an identity.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger
from threetears.observe.metrics import counter

__all__ = [
    "MAX_PROOF_WINDOW",
    "AbsentSender",
    "DoorProofModes",
    "DoorProofSwitch",
    "DoorWindow",
    "ProofMode",
    "ProofOutcome",
    "expired_windows",
]

log = get_logger(__name__)

#: the longest a door may accept a sender without its proof. long enough for every separately
#: deployed sender to take a release on its own schedule; short enough that a door adopting a proof
#: is not an open door for a season.
MAX_PROOF_WINDOW: Final[timedelta] = timedelta(days=90)

#: the name an absent sender is counted under when its message names nobody.
_UNCLAIMED: Final[str] = "unclaimed"

_ABSENT_COUNTER: Final[str] = "door_proof_absent"


class ProofMode(StrEnum):
    """whether a door accepts a sender that presents no proof."""

    OPTIONAL = "optional"
    REQUIRED = "required"


class ProofOutcome(StrEnum):
    """what the switch decided about one message."""

    PRESENTED = "presented"
    ACCEPTED_WITHOUT_PROOF = "accepted_without_proof"
    REFUSED_WITHOUT_PROOF = "refused_without_proof"


@dataclass(frozen=True)
class DoorWindow:
    """one door's dated window for adopting a proof, declared in code beside the door.

    :ivar door: the door's name, used in logs, metrics and config
    :ivar opened: the day the release carrying the proof shipped
    :ivar required_from: the first day a sender without the proof is refused
    """

    door: str
    opened: date
    required_from: date

    def __post_init__(self) -> None:
        """
        refuse window that runs backwards or longer than the norm allows.

        :raises ValueError: when required_from precedes opened or the window exceeds the maximum
        """
        if self.required_from < self.opened:
            raise ValueError(f"door {self.door!r}: window ends {self.required_from} before it opens {self.opened}")
        if self.required_from - self.opened > MAX_PROOF_WINDOW:
            raise ValueError(
                f"door {self.door!r}: window {self.opened}..{self.required_from} is longer than "
                f"{MAX_PROOF_WINDOW.days} days"
            )


@dataclass(frozen=True)
class AbsentSender:
    """one claimed sender that reached a door without its proof, as this process counted it.

    :ivar door: the door's name
    :ivar claimed_sender: who the message said it was from, unverified
    :ivar count: how many of its messages arrived without the proof
    :ivar date_last_seen: when the most recent one arrived, utc
    """

    door: str
    claimed_sender: str
    count: int
    date_last_seen: datetime


def _utc_today() -> date:
    """
    answer today's utc date.

    :return: today, utc
    :rtype: date
    """
    return datetime.now(UTC).date()


def expired_windows(windows: Iterable[DoorWindow], *, today: date) -> tuple[DoorWindow, ...]:
    """
    answer windows whose required-from date has arrived, whose optional path is now dead code.

    :param windows: every window a repo declares
    :ptype windows: Iterable[DoorWindow]
    :param today: the day to judge on
    :ptype today: date
    :return: windows past their date
    :rtype: tuple[DoorWindow, ...]
    """
    return tuple(window for window in windows if today >= window.required_from)


class DoorProofSwitch:
    """one door's window and its count of senders without the proof."""

    def __init__(
        self,
        window: DoorWindow,
        *,
        required_early: bool = False,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        """
        build switch for one door.

        :param window: the door's dated window
        :ptype window: DoorWindow
        :param required_early: whether config requires the door before its date
        :ptype required_early: bool
        :param today: clock answering today's utc date
        :ptype today: Callable[[], date]
        """
        self._window = window
        self._required_early = required_early
        self._today = today
        self._lock = threading.Lock()
        self._absent: dict[str, tuple[int, datetime]] = {}

    @property
    def door(self) -> str:
        """
        answer door's name.

        :return: door name
        :rtype: str
        """
        return self._window.door

    @property
    def window(self) -> DoorWindow:
        """
        answer door's window.

        :return: window
        :rtype: DoorWindow
        """
        return self._window

    @property
    def mode(self) -> ProofMode:
        """
        answer door's mode today.

        :return: required from the window's date or when config required it early, else optional
        :rtype: ProofMode
        """
        required = self._required_early or self._today() >= self._window.required_from
        return ProofMode.REQUIRED if required else ProofMode.OPTIONAL

    def judge(self, *, presented: bool, claimed_sender: str | None) -> ProofOutcome:
        """
        decide one message by whether it carried the proof, counting and logging an absent one.

        :param presented: whether message carried the proof at all; verifying it is the door's job
        :ptype presented: bool
        :param claimed_sender: who message says it is from, unverified, or none
        :ptype claimed_sender: str | None
        :return: presented, accepted without proof, or refused without proof
        :rtype: ProofOutcome
        """
        outcome = ProofOutcome.PRESENTED
        if not presented:
            mode = self.mode
            outcome = (
                ProofOutcome.ACCEPTED_WITHOUT_PROOF
                if mode is ProofMode.OPTIONAL
                else ProofOutcome.REFUSED_WITHOUT_PROOF
            )
            sender = claimed_sender or _UNCLAIMED
            now = datetime.now(UTC)
            with self._lock:
                count = self._absent.get(sender, (0, now))[0] + 1
                self._absent[sender] = (count, now)
            counter(
                _ABSENT_COUNTER,
                description="messages that reached a door without the proof it is adopting",
                label_names=("door", "outcome"),
            ).labels(door=self.door, outcome=outcome.value).inc()
            log.info(
                "door proof absent",
                extra={
                    "extra_data": {
                        "door": self.door,
                        "mode": mode.value,
                        "outcome": outcome.value,
                        "required_from": self._window.required_from.isoformat(),
                        "claimed_sender": sender,
                        "count": count,
                    }
                },
            )
        return outcome

    def absent_senders(self) -> tuple[AbsentSender, ...]:
        """
        answer every sender this process saw arrive without the proof.

        :return: absent senders, ordered by claimed sender
        :rtype: tuple[AbsentSender, ...]
        """
        with self._lock:
            entries = sorted(self._absent.items())
        return tuple(
            AbsentSender(door=self.door, claimed_sender=sender, count=count, date_last_seen=seen)
            for sender, (count, seen) in entries
        )


class DoorProofModes:
    """every door's switch in one process, with the doors config requires early."""

    def __init__(self, required_early: frozenset[str], *, today: Callable[[], date] = _utc_today) -> None:
        """
        build registry of door switches.

        :param required_early: doors config requires before their window's date
        :ptype required_early: frozenset[str]
        :param today: clock answering today's utc date
        :ptype today: Callable[[], date]
        """
        self._required_early = required_early
        self._today = today
        self._lock = threading.Lock()
        self._switches: dict[str, DoorProofSwitch] = {}

    @classmethod
    def parse(cls, spec: str, *, today: Callable[[], date] = _utc_today) -> DoorProofModes:
        """
        build registry from config: a comma-separated list of doors to require early.

        :param spec: comma-separated door names; empty for none
        :ptype spec: str
        :param today: clock answering today's utc date
        :ptype today: Callable[[], date]
        :return: registry
        :rtype: DoorProofModes
        :raises ValueError: when an entry is not a bare door name
        """
        doors: set[str] = set()
        for raw in spec.split(","):
            door = raw.strip()
            if not door:
                continue
            if any(character in door for character in "=@: "):
                raise ValueError(
                    f"door proof entry {door!r} is not a door name; config only lists doors to require early"
                )
            doors.add(door)
        return cls(frozenset(doors), today=today)

    def switch(self, window: DoorWindow) -> DoorProofSwitch:
        """
        answer the one switch for a door, building it on first use.

        :param window: the door's window, as declared beside the door
        :ptype window: DoorWindow
        :return: that door's switch
        :rtype: DoorProofSwitch
        :raises ValueError: when the door already has a different window
        """
        with self._lock:
            switch = self._switches.get(window.door)
            if switch is None:
                switch = DoorProofSwitch(window, required_early=window.door in self._required_early, today=self._today)
                self._switches[window.door] = switch
            elif switch.window != window:
                raise ValueError(f"door {window.door!r} already has a window {switch.window}; got {window}")
        return switch

    def absent_senders(self) -> tuple[AbsentSender, ...]:
        """
        answer every absent sender across every door this registry built.

        :return: absent senders, ordered by door then sender
        :rtype: tuple[AbsentSender, ...]
        """
        with self._lock:
            switches = sorted(self._switches.items())
        return tuple(sender for _, switch in switches for sender in switch.absent_senders())
