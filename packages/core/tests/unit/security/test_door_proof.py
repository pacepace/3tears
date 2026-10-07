"""The per-door proof switch: a dated window in code, optional before its date, required from it.

A door that gains a proof (a token, a signed envelope) cannot require it from the release that adds
it without refusing every pod that has not upgraded yet. So the door declares a window in code: it
accepts a sender without the proof, counting who still sends the old form, until a fixed date, and
refuses from that date. The count is information for the operator; it never moves the date, because
a forger sending one unsigned message a day would otherwise hold the door open forever. Config can
only require a door EARLIER, never later, and a window past its date is a build failure until its
optional path is deleted.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from threetears.core.security.door_proof import (
    MAX_PROOF_WINDOW,
    DoorProofModes,
    DoorProofSwitch,
    DoorWindow,
    ProofMode,
    ProofOutcome,
    expired_windows,
)

_OPENED = date(2026, 10, 7)
_WINDOW = DoorWindow(door="hub.usage.track", opened=_OPENED, required_from=date(2026, 11, 6))


def _switch(today: date, *, required_early: bool = False) -> DoorProofSwitch:
    """
    build the usage door's switch as of a given day.

    :param today: the day the switch judges on
    :ptype today: date
    :param required_early: whether config required the door before its date
    :ptype required_early: bool
    :return: switch
    :rtype: DoorProofSwitch
    """
    return DoorProofSwitch(_WINDOW, required_early=required_early, today=lambda: today)


def test_a_presented_proof_passes_in_either_mode() -> None:
    """the switch governs absence only; a presented proof is the door's own to verify."""
    for day in (_OPENED, _WINDOW.required_from):
        switch = _switch(day)
        assert switch.judge(presented=True, claimed_sender="agent-1") is ProofOutcome.PRESENTED
        assert switch.absent_senders() == ()


def test_before_its_date_a_door_accepts_an_old_sender_and_counts_it() -> None:
    """an older-shape sender is accepted while the window is open, and the counter names it."""
    switch = _switch(_OPENED)
    assert switch.mode is ProofMode.OPTIONAL
    assert switch.judge(presented=False, claimed_sender="agent-1") is ProofOutcome.ACCEPTED_WITHOUT_PROOF
    assert switch.judge(presented=False, claimed_sender="agent-1") is ProofOutcome.ACCEPTED_WITHOUT_PROOF
    assert switch.judge(presented=False, claimed_sender="agent-2") is ProofOutcome.ACCEPTED_WITHOUT_PROOF
    assert switch.judge(presented=True, claimed_sender="agent-3") is ProofOutcome.PRESENTED
    counted = {sender.claimed_sender: sender.count for sender in switch.absent_senders()}
    assert counted == {"agent-1": 2, "agent-2": 1}
    assert all(sender.door == "hub.usage.track" for sender in switch.absent_senders())


def test_from_its_date_a_door_refuses_an_old_sender_and_still_counts_it() -> None:
    """on the required-from date the old shape is refused, whatever the count says."""
    switch = _switch(_WINDOW.required_from)
    assert switch.mode is ProofMode.REQUIRED
    assert switch.judge(presented=False, claimed_sender="agent-1") is ProofOutcome.REFUSED_WITHOUT_PROOF
    (sender,) = switch.absent_senders()
    assert (sender.claimed_sender, sender.count) == ("agent-1", 1)


def test_old_senders_never_move_the_date() -> None:
    """a stream of unsigned messages the day before does not keep the door optional the day after."""
    days = [_WINDOW.required_from - timedelta(days=1), _WINDOW.required_from]
    switch = DoorProofSwitch(_WINDOW, today=lambda: days[0])
    for _ in range(100):
        switch.judge(presented=False, claimed_sender="forger")
    days.pop(0)
    assert switch.judge(presented=False, claimed_sender="forger") is ProofOutcome.REFUSED_WITHOUT_PROOF


def test_config_can_require_a_door_early() -> None:
    """an operator who has confirmed every sender upgraded flips the door before its date."""
    assert _switch(_OPENED, required_early=True).mode is ProofMode.REQUIRED


def test_config_names_only_doors_to_require_early_so_it_cannot_loosen_one() -> None:
    """there is no spelling that makes a door optional past its date."""
    modes = DoorProofModes.parse("hub.usage.track", today=lambda: _OPENED)
    assert modes.switch(_WINDOW).mode is ProofMode.REQUIRED
    other = DoorWindow(door="knowledge.draft", opened=_OPENED, required_from=date(2026, 11, 6))
    assert modes.switch(other).mode is ProofMode.OPTIONAL
    late = DoorProofModes.parse("", today=lambda: date(2026, 12, 1))
    assert late.switch(other).mode is ProofMode.REQUIRED
    with pytest.raises(ValueError, match="door name"):
        DoorProofModes.parse("hub.usage.track=optional", today=lambda: _OPENED)


def test_a_window_is_bounded() -> None:
    """no door may stay optional longer than the norm's maximum window."""
    with pytest.raises(ValueError, match="longer than"):
        DoorWindow(door="x", opened=_OPENED, required_from=_OPENED + MAX_PROOF_WINDOW + timedelta(days=1))
    with pytest.raises(ValueError, match="before it opens"):
        DoorWindow(door="x", opened=_OPENED, required_from=_OPENED - timedelta(days=1))


def test_a_window_past_its_date_is_reported_for_deletion() -> None:
    """once a door is required by date, its window and optional path are dead code to delete."""
    assert expired_windows((_WINDOW,), today=_OPENED) == ()
    assert expired_windows((_WINDOW,), today=_WINDOW.required_from) == (_WINDOW,)


def test_an_unclaimed_sender_is_counted_under_a_named_placeholder() -> None:
    """a sender that names nobody is still counted, under one placeholder rather than dropped."""
    switch = _switch(_OPENED)
    switch.judge(presented=False, claimed_sender=None)
    (sender,) = switch.absent_senders()
    assert sender.claimed_sender == "unclaimed"
    assert sender.date_last_seen.tzinfo is not None


def test_one_door_has_one_switch_and_one_counter() -> None:
    """asking for a door twice answers the same switch, so its count is the door's whole count."""
    modes = DoorProofModes.parse("", today=lambda: _OPENED)
    modes.switch(_WINDOW).judge(presented=False, claimed_sender="agent-1")
    modes.switch(_WINDOW).judge(presented=False, claimed_sender="agent-1")
    (sender,) = modes.absent_senders()
    assert (sender.door, sender.count) == ("hub.usage.track", 2)


def test_two_windows_for_one_door_are_refused() -> None:
    """a door has one window; a second spelling of it is a bug, not an extension."""
    modes = DoorProofModes.parse("", today=lambda: _OPENED)
    modes.switch(_WINDOW)
    with pytest.raises(ValueError, match="already has a window"):
        modes.switch(DoorWindow(door="hub.usage.track", opened=_OPENED, required_from=date(2026, 12, 1)))
