"""the one owner of every ``langchain-anthropic`` private member ``threetears.models`` touches.

``ChatAnthropic`` builds its ``anthropic`` SDK clients itself, from a private ``_client_params``
dict, and takes no middleware, transport or HTTP client from its caller. The SDK's client does
take ``middleware=`` (public: ``anthropic.Middleware``), which is the hook a model needs to see each
HTTP attempt's response inside the SDK's retry loop -- the key pool's 429-unretried mode
(``providers/_key_errors_unretried.py``). This module is the only place in ``threetears.models``
that names a ``langchain-anthropic`` member with a leading underscore: it overrides
``_client_params`` to add the model's ``sdk_middleware``, and nothing else.

**Verified against langchain-anthropic 1.7.5 and anthropic 1.12.1** (``chat_models.py``:
``_client_params``, ``_client`` and ``_async_client``, which spread ``_client_params`` into
``anthropic.Client`` / ``anthropic.AsyncClient``; ``anthropic/_client.py`` and ``_middleware.py``:
the ``middleware`` argument, and a ``Middleware`` subclass passing both the sync and the async
client's validation when it implements ``handle`` and ``handle_async``). A release that renames
``_client_params`` fails ``tests/unit/models/test_key_errors_unretried.py``, which drives a 429
through a model built here and counts the requests.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import Field

__all__ = ["SdkMiddlewareChatAnthropic"]


class SdkMiddlewareChatAnthropic(ChatAnthropic):
    """``ChatAnthropic`` whose SDK clients carry middleware.

    :ivar sdk_middleware: ``anthropic.Middleware`` instances the model's sync and async SDK clients
        run on every HTTP attempt, in order; each must implement both ``handle`` and ``handle_async``,
        since one list reaches both clients. Empty, the clients are built exactly as
        ``ChatAnthropic`` builds them.
    """

    sdk_middleware: list[Any] = Field(default_factory=list, exclude=True)

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        """``ChatAnthropic``'s client parameters, with :attr:`sdk_middleware` added when there is any.

        :return: the keyword arguments both SDK clients are built from
        :rtype: dict[str, Any]
        """
        params = dict(super()._client_params)
        if self.sdk_middleware:
            params["middleware"] = list(self.sdk_middleware)
        return params
