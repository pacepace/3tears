"""A key pool's credentials: a redacted :class:`Secret`, and the :class:`KeySpec` an operator configures.

A secret is opaque to the pool (KP-A1). The only way to its value is :meth:`Secret.reveal`, which the
code that sends a call uses and nothing else does: ``str``, ``repr`` and every ``format`` show a
redaction, so a secret logged, formatted into an exception or printed in a dataclass's ``repr`` shows
nothing (KP-B1). Keys are named by their ``label`` everywhere else.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, NoReturn, final

from threetears.observe import BuildOnce

__all__ = ["KeySpec", "Secret"]

#: What every rendering of a secret shows in its place.
_REDACTED: Final = "<redacted>"

#: The per-process key a secret's hash is drawn with, so ``hash()`` says nothing about the value that
#: another process, or a dump of hashes, could compare. Drawn on first use, not at import: importing
#: ``create_chat_model`` must draw no randomness.
_HASH_KEY: Final[BuildOnce[None, bytes]] = BuildOnce()

#: A label is short and drawn from a charset safe as a metric label value and inside the pool's store
#: account names, where ``/`` separates the parts.
_LABEL: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")

#: The one currency every ``ModelCapabilities`` price is in (KP-A3).
_PRICE_CURRENCY: Final = "USD"


@final
class Secret:
    """A credential the pool holds without showing it.

    ``str()``, ``repr()`` and any ``format`` spec show ``<redacted>``. The value is reached only through
    :meth:`reveal`. A secret equals another secret with the same value, compared in constant time, and
    never its plain string; its hash is drawn with a key private to the process. It refuses pickling,
    since a pickle would carry the value to wherever the bytes go.

    :param value: the credential, as the provider issued it
    :ptype value: str
    :raises TypeError: when ``value`` is not a ``str``
    :raises ValueError: when ``value`` is empty or only whitespace
    """

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"a Secret wraps a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError("a Secret must not be empty")
        self.__value = value

    def reveal(self) -> str:
        """The credential itself, for the code that sends a call with it and for nothing else.

        :return: the value this secret wraps
        :rtype: str
        """
        return self.__value

    def __repr__(self) -> str:
        """``Secret('<redacted>')``, never the value."""
        return f"Secret({_REDACTED!r})"

    def __str__(self) -> str:
        """``<redacted>``, never the value."""
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        """``<redacted>`` under every format spec, never the value."""
        return _REDACTED

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is a :class:`Secret` with the same value; a plain string never is."""
        if not isinstance(other, Secret):
            return NotImplemented
        return hmac.compare_digest(self.__value.encode(), other.reveal().encode())

    def __hash__(self) -> int:
        """A hash drawn with a key private to this process, so equal secrets hash equal and no hash
        can be compared across processes."""
        digest = hashlib.blake2b(
            self.__value.encode(), key=_HASH_KEY.get(None, lambda: secrets.token_bytes(32)), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big")

    def __copy__(self) -> Secret:
        """The secret itself: it is immutable, so a copy needs no new object (and no pickling)."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Secret:
        """The secret itself, as for :meth:`__copy__`."""
        return self

    def __reduce__(self) -> NoReturn:
        """Refuse pickling: the bytes would carry the value.

        :raises TypeError: always
        """
        raise TypeError("a Secret is not pickled; pass its holder the KeySpec in-process instead")


@dataclass(frozen=True, slots=True)
class KeySpec:
    """One credential in a pool, with the operator's configuration for it (KP-A1).

    :param secret: the credential
    :ptype secret: Secret
    :param label: the key's name in logs, metrics, ``status()`` and errors: unique in its pool, 1 to 32
        characters of letters, digits, ``_``, ``.`` and ``-``, starting with a letter or digit
    :ptype label: str
    :param budget: what the key may spend in one billing period, more than zero
    :ptype budget: Decimal
    :param currency: the budget's ISO currency code; only ``"USD"``, the currency of every price
        (KP-A3), is accepted: a budget in another currency is refused, not converted
    :ptype currency: str
    :param priority: lower is preferred; ``None`` (the default) ranks after every set priority. With
        distinct priorities the pool returns to a more-preferred key as soon as it can serve again
        (KP-D5); with equal ones it stays where it is
    :ptype priority: int | None
    :param account: the provider-side account the key's credit and rate limits belong to, shared by
        the keys that share it; ``None`` (the default) makes the key its own account (KP-C5)
    :ptype account: str | None
    :raises TypeError: when ``secret`` is not a :class:`Secret`
    :raises ValueError: when the label, budget, currency, priority or account is invalid
    """

    secret: Secret
    label: str
    budget: Decimal
    currency: str = _PRICE_CURRENCY
    priority: int | None = None
    account: str | None = None

    def __post_init__(self) -> None:
        """Refuse a spec the pool can't budget or name safely.

        :raises TypeError: when ``secret`` is not a :class:`Secret`
        :raises ValueError: when the label, budget, currency, priority or account is invalid
        """
        if not isinstance(self.secret, Secret):
            # never echo the value: a plain string here is very likely the credential itself
            raise TypeError(f"KeySpec.secret must be a Secret, got {type(self.secret).__name__}")
        if not isinstance(self.label, str) or _LABEL.fullmatch(self.label) is None:
            raise ValueError(
                f"key label {self.label!r} must be 1 to 32 letters, digits, '_', '.' or '-', "
                "starting with a letter or digit"
            )
        if not isinstance(self.budget, Decimal) or not self.budget.is_finite() or self.budget <= 0:
            raise ValueError(f"key {self.label!r}: budget must be a finite Decimal above zero, got {self.budget!r}")
        if self.currency != _PRICE_CURRENCY:
            raise ValueError(
                f"key {self.label!r}: budget currency must be {_PRICE_CURRENCY}, the currency every model "
                f"price is in; got {self.currency!r}, and a budget is refused rather than converted"
            )
        if self.priority is not None and (isinstance(self.priority, bool) or not isinstance(self.priority, int)):
            raise ValueError(f"key {self.label!r}: priority must be an int or None, got {self.priority!r}")
        if self.account is not None and (not isinstance(self.account, str) or not self.account.strip()):
            raise ValueError(f"key {self.label!r}: account must be a non-empty str or None, got {self.account!r}")

    @property
    def account_tag(self) -> str:
        """The account this key's account-scoped verdicts reach: its ``account``, or itself alone.

        :return: the account, or a tag only this key carries when it has none
        :rtype: str
        """
        return self.account if self.account is not None else f"key:{self.label}"
