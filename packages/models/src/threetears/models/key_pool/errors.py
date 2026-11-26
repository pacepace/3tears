"""The errors a key pool raises for a call it can't serve or can't budget.

A store failure is :class:`~threetears.models.key_pool.store.KeyPoolStoreError` (KP-G5), defined with
the store. None of these carries a secret or a fingerprint (KP-I3): keys are named by label only.
"""

from __future__ import annotations

from datetime import datetime

from threetears.models.errors import NO_KEY_AVAILABLE_REASON, ModelProviderError
from threetears.models.key_pool.status import KeyStatus

__all__ = ["NoKeyAvailableError", "NoOutputLimitError"]


class NoKeyAvailableError(ModelProviderError):
    """No key in the pool can serve the call (KP-I1).

    Raised before any network call when no key is eligible, and once every eligible key has been
    tried. It is a :class:`~threetears.models.errors.ModelProviderError`, so ``is_provider_error()`` is
    true for it and ``friendly_api_error()`` gives a fixed message (KP-I2).

    :param pool: the pool's name
    :ptype pool: str
    :param provider: the provider the pool's keys are for
    :ptype provider: str
    :param model_id: the model the call was for, ``None`` for a plain reservation
    :ptype model_id: str | None
    :param keys: every key, in preference order, each with the state that kept it from the call
    :ptype keys: tuple[KeyStatus, ...]
    :param earliest: the earliest time any key could serve again, ``None`` when none is known to (every
        key revoked, forbidden or already tried)
    :ptype earliest: datetime | None
    """

    def __init__(
        self,
        *,
        pool: str,
        provider: str,
        model_id: str | None,
        keys: tuple[KeyStatus, ...],
        earliest: datetime | None,
    ) -> None:
        self.pool = pool
        self.model_id = model_id
        self.keys = keys
        self.earliest = earliest
        states = "; ".join(f"{key.label}: {key.describe()}" for key in keys)
        target = f"model {model_id!r}" if model_id is not None else "a reservation"
        when = earliest.isoformat() if earliest is not None else "unknown"
        detail = f"no key in pool {pool!r} can serve {target} ({states}); the earliest a key could serve again: {when}"
        super().__init__(detail, provider=provider, reason=NO_KEY_AVAILABLE_REASON)


class NoOutputLimitError(ValueError):
    """A call can't be reserved for: it sets no ``max_tokens`` and the model declares no
    ``max_output_tokens``, so its output cost has no bound (KP-E1).

    Reserving it at zero would let it spend past the budget unseen.

    :param model_id: the model the call was for
    :ptype model_id: str
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f"model {model_id!r} has no max_output_tokens and the call sets no max_tokens, "
            "so its cost can't be reserved; set max_tokens on the call"
        )
