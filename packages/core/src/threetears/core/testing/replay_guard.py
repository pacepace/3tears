"""the one shipped double of :class:`~threetears.core.coordination.replay_guard.ReplayGuard`.

A verifier -- ``ToolServer``, ``CallProxy``, ``validate_dpop_proof`` -- calls three things on its
guard: ``require_covers`` where it is configured, ``bind`` when it starts, ``record_unique`` per
artifact. Every repo that tested a verifier used to write its own stand-in for that surface, so a
method added to the guard became an edit to each of them, and the only thing naming a missed one
was an ``AttributeError`` at startup. This double is declared against the real class, and the
fake-parity gate compares the two, so a new public method on the guard fails that gate here
instead.

Use a real ``ReplayGuard`` over :class:`~threetears.core.testing.kv.FakeNatsClient` when the test
is about the guard's own semantics (the watermark, the anchor, the bucket). Use this when the test
is about the verifier: it records what the verifier asked, answers what the test chose, and can
refuse to bind.
"""

from __future__ import annotations

from datetime import datetime, timedelta

__all__ = ["FakeReplayGuard"]


# parity-with: threetears.core.coordination.replay_guard.ReplayGuard
class FakeReplayGuard:
    """a replay guard whose verdicts, bind outcome and recorded calls the test controls.

    :param fresh: the verdict for every record. ``None`` (the default) remembers each nonce and
        answers ``True`` for its first sighting and ``False`` after -- a real guard with no
        watermark. ``True`` or ``False`` answers that for every record, seen or not
    :ptype fresh: bool | None
    :param bind_error: raised by :meth:`bind` after it is counted, to drive a verifier's startup
        failure path. ``None`` binds
    :ptype bind_error: BaseException | None
    :param events: a log shared with the test's other doubles; ``"bind"`` and ``"record"`` are
        appended as they happen, so a test can assert the order a verifier calls them in
    :ptype events: list[str] | None
    :param bucket_name: reported by :attr:`bucket_name`
    :ptype bucket_name: str
    :param verifier_future_tolerance: the tolerance :meth:`require_covers` checks against, as the
        real guard does. The default covers any verifier
    :ptype verifier_future_tolerance: timedelta
    """

    def __init__(
        self,
        *,
        fresh: bool | None = None,
        bind_error: BaseException | None = None,
        events: list[str] | None = None,
        bucket_name: str = "fake_nonces",
        verifier_future_tolerance: timedelta = timedelta.max,
    ) -> None:
        """hold the chosen behaviour and start with nothing recorded.

        :param fresh: the verdict for every record, or ``None`` to remember nonces
        :ptype fresh: bool | None
        :param bind_error: what :meth:`bind` raises, or ``None``
        :ptype bind_error: BaseException | None
        :param events: a shared ordered log, or ``None``
        :ptype events: list[str] | None
        :param bucket_name: the reported bucket name
        :ptype bucket_name: str
        :param verifier_future_tolerance: the tolerance the guard was sized for
        :ptype verifier_future_tolerance: timedelta
        :return: None
        :rtype: None
        """
        self._fresh = fresh
        self._bind_error = bind_error
        self._bucket_name = bucket_name
        self._verifier_future_tolerance = verifier_future_tolerance
        self.events: list[str] = events if events is not None else []
        self.binds = 0
        self.seen: list[str] = []
        self.issued_at: list[datetime] = []

    @property
    def bucket_name(self) -> str:
        """the reported bucket name.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    @property
    def verifier_future_tolerance(self) -> timedelta:
        """the tolerance this double was sized for.

        :return: the tolerance
        :rtype: timedelta
        """
        return self._verifier_future_tolerance

    def require_covers(self, future_tolerance: timedelta) -> None:
        """refuse a verifier whose tolerance exceeds the sized one, as the real guard does.

        :param future_tolerance: the calling verifier's tolerance
        :ptype future_tolerance: timedelta
        :return: None
        :rtype: None
        :raises ValueError: when ``future_tolerance`` exceeds :attr:`verifier_future_tolerance`
        """
        if future_tolerance > self._verifier_future_tolerance:
            raise ValueError(
                f"FakeReplayGuard was sized for a verifier_future_tolerance of "
                f"{self._verifier_future_tolerance}, but its verifier accepts {future_tolerance}"
            )

    async def bind(self) -> None:
        """count the bind and log it, then raise the chosen error if there is one.

        :return: None
        :rtype: None
        :raises BaseException: the ``bind_error`` given at construction
        """
        self.binds += 1
        self.events.append("bind")
        if self._bind_error is not None:
            raise self._bind_error

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        """record the nonce and its issue time, and answer the chosen verdict.

        :param nonce: the nonce the verifier consumed
        :ptype nonce: str
        :param issued_at: the issue time the verifier passed; must be timezone-aware
        :ptype issued_at: datetime
        :return: the verdict
        :rtype: bool
        :raises ValueError: when ``issued_at`` is timezone-naive, as the real guard refuses it
        """
        if issued_at.tzinfo is None:
            raise ValueError("FakeReplayGuard.record_unique requires a timezone-aware issued_at")
        first_sighting = nonce not in self.seen
        self.events.append("record")
        self.seen.append(nonce)
        self.issued_at.append(issued_at)
        return first_sighting if self._fresh is None else self._fresh
