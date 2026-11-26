"""A chat model that runs each call on a key pool's key: reserved, failed over, and settled (KP-E, H, J).

:func:`~threetears.models.factory.create_chat_model` returns a :class:`PooledChatModel` for
``key_pool=``. It holds one model per key, each built as an ``api_key=`` model is, with that key's
breakers and its label on every usage record. A call:

1. estimates its cost -- its messages' tokens at the input rate, its ``max_tokens`` (or the model's
   ``max_output_tokens``) at the output rate -- and reserves it on the key the pool hands it (KP-E1);
2. runs on that key's model, so that key's tracker and breaker callbacks fire, and every callback the
   caller passes sees the attempt start and end or fail (KP-J3);
3. settles the reservation to the cost of the usage the provider reported, through the one cost
   function the tracker prices with; a call whose usage the provider didn't report keeps its
   reservation as spend (KP-E3, J5).

**Failover** (KP-H). A failure is read by the pool's classifier. One that is the key's -- exhausted,
revoked, forbidden, rate-limited -- is recorded on the key, which moves the pool off it, and the call
runs again on the next key; any other failure closes the reservation as the classifier says and
propagates unchanged (KP-E4, H3). A key is tried once per call, with one exception: a key that refused
the call for a rate limit may serve it once more, after its hold, when the pool waits for holds (its
``max_rate_limit_wait``). That is the wait a single key's SDK made before the pool turned its retries
off (Q4), and with the default cap of 0 the pool raises at once instead. A key whose breaker refused
the call (it opened after the pool chose the key) is released, since nothing was sent, and counts as
tried. Once a stream has handed the caller a chunk, nothing is retried (KP-H5): the failure is recorded,
the reservation settled as the classifier says, and the provider's error raised.

Each failover is counted by the pool (``KeyPool.record_failover``, KP-K1) when the call's next attempt
runs on another key, by why the key refused it: the verdict's kind, or ``tripped`` for its breaker. A
retry on the same key after its rate-limit hold is not a failover, and nor is a refusal that ends the
call in ``NoKeyAvailableError``.

**Re-binding** (KP-J2). ``bind_tools``, ``with_structured_output``, ``bind``, ``with_config``,
``with_retry``, ``with_types`` -- any method of the model that returns a runnable -- is applied to every
key's model, and returns a pooled model over the results, so the call runs the same binding whichever
key serves it. Every other attribute is the first key's model's: all of them are built alike.

**The synchronous paths** (``invoke``, ``stream``, ``batch``) drive the pool, which is async, on one
private event loop in a daemon thread, waiting for each pool call there. That works whether or not the
calling thread runs an event loop of its own, and never waits on the caller's loop from the caller's
loop. The pool's in-memory store guards itself with a thread lock, not an asyncio primitive, so driving
it from that loop and from the caller's loop at once is safe. A store whose connections belong to one
event loop -- a database pool, a NATS client -- can only be driven from that loop: use the async paths
(``ainvoke``, ``astream``, ``abatch``) with such a store.

**Reservations carry no deadline.** A LangChain call has none, and a provider's request timeout bounds
silence, not a stream's length, so the reservation expires the pool's ``reservation_margin`` after the
call started. A call that outlives it still settles: the store accepts a settle after the sweep closed
the reservation.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import ensure_config, merge_configs
from threetears.observe import BuildOnce, get_logger

from threetears.models.errors import CircuitOpenError
from threetears.models.key_pool import (
    KeyPool,
    KeyPoolLease,
    KeyPoolStoreError,
    MoveReason,
    NoKeyAvailableError,
    VerdictKind,
    reservation_estimate,
)
from threetears.models.pricing import cost_of_usage
from threetears.models.tracking import estimate_input_tokens, extract_usage

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

    from threetears.models.capabilities import ModelCapabilities

__all__ = ["PooledChatModel"]

logger = get_logger(__name__)

#: The per-call keyword that bounds a call's output, which its reservation prices (KP-E1).
_MAX_TOKENS: Final = "max_tokens"

#: The failover reason each key-moving verdict gives (KP-K1).
_FAILOVER_REASON: Final = {
    VerdictKind.EXHAUSTED: MoveReason.EXHAUSTED,
    VerdictKind.REVOKED: MoveReason.REVOKED,
    VerdictKind.FORBIDDEN: MoveReason.FORBIDDEN,
    VerdictKind.RATE_LIMITED: MoveReason.RATE_LIMITED,
}


class PooledChatModel(Runnable[Any, Any]):
    """A chat model whose calls run on a :class:`~threetears.models.key_pool.KeyPool`'s keys.

    Built by :func:`~threetears.models.factory.create_chat_model` with ``key_pool=``; see the module
    docstring for what a call does.

    :param pool: the pool whose keys serve the calls
    :ptype pool: KeyPool
    :param model_id: the model every call is for
    :ptype model_id: str
    :param capabilities: the model's capabilities, carrying its prices and ``max_output_tokens``
    :ptype capabilities: ModelCapabilities
    :param models: each key's model, by label, all bound alike
    :ptype models: Mapping[str, Runnable[Any, Any]]
    :param max_tokens: the ``max_tokens`` the models were built or bound with, ``None`` when unset
    :ptype max_tokens: int | None
    :raises ValueError: when ``models`` doesn't hold exactly the pool's labels
    """

    def __init__(
        self,
        pool: KeyPool,
        model_id: str,
        capabilities: ModelCapabilities,
        models: Mapping[str, Runnable[Any, Any]],
        *,
        max_tokens: int | None = None,
    ) -> None:
        if set(models) != set(pool.labels):
            raise ValueError(f"a pooled model needs one model per key of pool {pool.name!r}: {pool.labels}")
        self._pool = pool
        self._model_id = model_id
        self._capabilities = capabilities
        # in the pool's preference order, so the first is the model a key serves most calls on
        self._models: dict[str, Runnable[Any, Any]] = {label: models[label] for label in pool.labels}
        self._first = self._models[pool.labels[0]]
        self._max_tokens = max_tokens

    def __repr__(self) -> str:
        """The model by its id and pool, never a secret."""
        return f"PooledChatModel(model_id={self._model_id!r}, pool={self._pool.name!r}, labels={self._pool.labels!r})"

    # -- re-binding (KP-J2) ----------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- whatever the chat model's attribute is
        """The chat model's attribute; a method that returns a runnable is applied to every key's model.

        :param name: the attribute
        :ptype name: str
        :return: the first key's model's attribute, or, for a method, one that re-binds every key's
        :rtype: Any
        :raises AttributeError: when the chat model has no such attribute
        """
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._first, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def rebound(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- the chat model's method's result
            result = attr(*args, **kwargs)
            if not isinstance(result, Runnable):
                return result
            return self._rebound(name, args, kwargs, first=result)

        return rebound

    def bind(self, **kwargs: Any) -> Runnable[Any, Any]:
        """Bind call arguments on every key's model.

        :param kwargs: the arguments every call is made with
        :ptype kwargs: Any
        :return: the pooled model over the bound models
        :rtype: Runnable[Any, Any]
        """
        return self._rebound("bind", (), kwargs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Runnable[Any, Any]:
        """Bind a config on every key's model.

        :param config: the config every call runs with
        :ptype config: RunnableConfig | None
        :param kwargs: further config keys
        :ptype kwargs: Any
        :return: the pooled model over the configured models
        :rtype: Runnable[Any, Any]
        """
        return self._rebound("with_config", (config,), kwargs)

    def with_retry(self, **kwargs: Any) -> Runnable[Any, Any]:
        """Retry each key's model as ``Runnable.with_retry`` does, inside the attempt on that key.

        :param kwargs: ``Runnable.with_retry``'s arguments
        :ptype kwargs: Any
        :return: the pooled model over the retrying models
        :rtype: Runnable[Any, Any]
        """
        return self._rebound("with_retry", (), kwargs)

    def with_types(self, **kwargs: Any) -> Runnable[Any, Any]:
        """Bind input and output types on every key's model.

        :param kwargs: ``Runnable.with_types``'s arguments
        :ptype kwargs: Any
        :return: the pooled model over the typed models
        :rtype: Runnable[Any, Any]
        """
        return self._rebound("with_types", (), kwargs)

    @property
    def InputType(self) -> Any:  # noqa: N802, ANN401 -- LangChain's name; the chat model's input type
        """The chat model's input type.

        :return: the type
        :rtype: Any
        """
        return self._first.InputType

    @property
    def OutputType(self) -> Any:  # noqa: N802, ANN401 -- LangChain's name; the bound model's output type
        """The bound model's output type: a message, or what structured output parses to.

        :return: the type
        :rtype: Any
        """
        return self._first.OutputType

    def _call(self, input: Any, kwargs: Mapping[str, Any]) -> _Call:  # noqa: A002, ANN401 -- the chat model's input
        """One call's attempts, reserving its estimate: the prompt's tokens and its output bound (KP-E1)."""
        amount = reservation_estimate(
            self._capabilities,
            input_tokens=estimate_input_tokens(_prompt_messages(input)),
            max_tokens=kwargs.get(_MAX_TOKENS, self._max_tokens),
        )
        return _Call(self._pool, self._model_id, self._capabilities, amount)

    def _rebound(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], *, first: Any = None
    ) -> PooledChatModel:  # noqa: ANN401 -- a runnable the first key's model already returned
        """The pooled model over every key's model with ``name(*args, **kwargs)`` applied."""
        models: dict[str, Runnable[Any, Any]] = {}
        for label, model in self._models.items():
            if first is not None and model is self._first:
                models[label] = first
            else:
                models[label] = getattr(model, name)(*args, **kwargs)
        max_tokens = self._max_tokens
        if name == "bind" and _MAX_TOKENS in kwargs:
            max_tokens = kwargs[_MAX_TOKENS]
        return PooledChatModel(self._pool, self._model_id, self._capabilities, models, max_tokens=max_tokens)

    # -- calls -----------------------------------------------------------------------------------

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:  # noqa: A002, ANN401
        """Run the call on a pool key, failing over between keys; synchronous.

        :param input: the prompt, as the chat model takes it
        :ptype input: Any
        :param config: the call's config
        :ptype config: RunnableConfig | None
        :param kwargs: call arguments for the model
        :ptype kwargs: Any
        :return: what the bound model returns
        :rtype: Any
        :raises NoKeyAvailableError: when no key can serve the call, or every key was tried
        :raises KeyPoolStoreError: when the pool's store failed before the call ran
        """
        call = self._call(input, kwargs)
        while True:
            lease = _run_sync(call.acquire())
            meter = call.meter()
            try:
                result = self._models[lease.label].invoke(input, call.config(config, meter), **kwargs)
            except Exception as exc:
                if _run_sync(call.failed(lease, exc, meter, stream_started=False)):
                    continue
                raise
            else:
                _run_sync(call.succeeded(lease, meter))
                return result
            finally:
                if call.open(lease):
                    _run_sync(call.abandoned(lease, meter))

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:  # noqa: A002, ANN401
        """Run the call on a pool key, failing over between keys.

        :param input: the prompt, as the chat model takes it
        :ptype input: Any
        :param config: the call's config
        :ptype config: RunnableConfig | None
        :param kwargs: call arguments for the model
        :ptype kwargs: Any
        :return: what the bound model returns
        :rtype: Any
        :raises NoKeyAvailableError: when no key can serve the call, or every key was tried
        :raises KeyPoolStoreError: when the pool's store failed before the call ran
        """
        call = self._call(input, kwargs)
        while True:
            lease = await call.acquire()
            meter = call.meter()
            try:
                result = await self._models[lease.label].ainvoke(input, call.config(config, meter), **kwargs)
            except Exception as exc:
                if await call.failed(lease, exc, meter, stream_started=False):
                    continue
                raise
            else:
                await call.succeeded(lease, meter)
                return result
            finally:
                if call.open(lease):
                    await call.abandoned(lease, meter)

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any | None) -> Iterator[Any]:  # noqa: A002
        """Stream the call from a pool key; a key's refusal before the first chunk fails over.

        :param input: the prompt, as the chat model takes it
        :ptype input: Any
        :param config: the call's config
        :ptype config: RunnableConfig | None
        :param kwargs: call arguments for the model
        :ptype kwargs: Any | None
        :return: the bound model's chunks
        :rtype: Iterator[Any]
        :raises NoKeyAvailableError: when no key can serve the call, or every key was tried
        :raises KeyPoolStoreError: when the pool's store failed before the call ran
        """
        call = self._call(input, kwargs)
        while True:
            lease = _run_sync(call.acquire())
            meter = call.meter()
            chunks = self._models[lease.label].stream(input, call.config(config, meter), **kwargs)
            started = False
            try:
                for chunk in chunks:
                    started = True
                    yield chunk
            except Exception as exc:
                if _run_sync(call.failed(lease, exc, meter, stream_started=started)):
                    continue
                raise
            else:
                _run_sync(call.succeeded(lease, meter))
                return
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
                if call.open(lease):
                    _run_sync(call.abandoned(lease, meter))

    async def astream(
        self,
        input: Any,  # noqa: A002, ANN401 -- the chat model's input
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        """Stream the call from a pool key; a key's refusal before the first chunk fails over.

        :param input: the prompt, as the chat model takes it
        :ptype input: Any
        :param config: the call's config
        :ptype config: RunnableConfig | None
        :param kwargs: call arguments for the model
        :ptype kwargs: Any | None
        :return: the bound model's chunks
        :rtype: AsyncIterator[Any]
        :raises NoKeyAvailableError: when no key can serve the call, or every key was tried
        :raises KeyPoolStoreError: when the pool's store failed before the call ran
        """
        call = self._call(input, kwargs)
        while True:
            lease = await call.acquire()
            meter = call.meter()
            chunks = self._models[lease.label].astream(input, call.config(config, meter), **kwargs)
            started = False
            try:
                async for chunk in chunks:
                    started = True
                    yield chunk
            except Exception as exc:
                if await call.failed(lease, exc, meter, stream_started=started):
                    continue
                raise
            else:
                await call.succeeded(lease, meter)
                return
            finally:
                close = getattr(chunks, "aclose", None)
                if close is not None:
                    await close()
                if call.open(lease):
                    await call.abandoned(lease, meter)


class _Meter(BaseCallbackHandler):
    """The usage one attempt's model runs reported, priced for the reservation's settlement.

    Runs inline, so it has counted a run's usage before the call returns to the pooled model.
    """

    run_inline = True

    def __init__(self, capabilities: ModelCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._costs: list[Decimal] = []
        self._unreported = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Count one model run's usage: its price when the provider reported it, else unreported."""
        del run_id, parent_run_id, kwargs
        usage = extract_usage(response)
        if usage.source != "reported":
            self._unreported += 1
            return
        self._costs.append(
            cost_of_usage(
                self._capabilities,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_creation_tokens,
            )
        )

    @property
    def reported(self) -> bool:
        """Whether every model run the attempt made ended, with the provider's usage, and one did."""
        return bool(self._costs) and not self._unreported

    def cost(self, reserved: Decimal) -> Decimal:
        """What to settle the attempt at: its reported cost, or, where any usage went unreported,
        never less than the reservation (KP-E3)."""
        total = sum(self._costs, Decimal(0))
        return total if self.reported else max(total, reserved)


class _Call:
    """One call's attempts: its reservation estimate, the keys it has tried, and how each attempt closes."""

    def __init__(self, pool: KeyPool, model_id: str, capabilities: ModelCapabilities, amount: Decimal) -> None:
        self._pool = pool
        self._model_id = model_id
        self._capabilities = capabilities
        self._amount = amount
        self._tried: set[str] = set()
        self._rate_limited: set[str] = set()
        self._last: Exception | None = None
        self._attempts = 0
        # the key the last attempt's refusal moved the call off, and why, until the next attempt's key
        self._refused: tuple[str, MoveReason] | None = None
        # reservations of served attempts whose settle failed: left open on purpose, for the sweep
        self._left_open: set[str] = set()

    def meter(self) -> _Meter:
        return _Meter(self._capabilities)

    def config(self, config: RunnableConfig | None, meter: _Meter) -> RunnableConfig:
        """The attempt's config: the caller's, with the meter added; a retry gets its own run id."""
        merged = merge_configs(ensure_config(config), {"callbacks": [meter]})
        if self._attempts > 1:
            merged.pop("run_id", None)
        return merged

    async def acquire(self) -> KeyPoolLease:
        """Reserve the estimate on the key the attempt runs on (KP-E1, H2)."""
        try:
            lease = await self._pool.acquire(self._model_id, self._amount, tried=frozenset(self._tried))
        except NoKeyAvailableError as exc:
            if self._last is not None:
                raise exc from self._last
            raise
        self._attempts += 1
        if self._refused is not None:
            refused_label, reason = self._refused
            self._refused = None
            if lease.label != refused_label:
                self._pool.record_failover(reason)
        return lease

    async def succeeded(self, lease: KeyPoolLease, meter: _Meter) -> None:
        """Settle a served attempt at its reported cost (KP-E3).

        The answer is the caller's whether or not the settle lands: a store that fails here was logged
        and counted by the pool, and the reservation it still holds settles at its full amount when it
        expires, so the spend is kept either way.
        """
        try:
            await lease.settle(meter.cost(lease.amount))
        except KeyPoolStoreError:
            self._left_open.add(lease.reservation_id)
            logger.warning(
                "key pool %r: the call on key %r was served but its settlement failed; its reservation of %s "
                "USD stays open and settles at its full amount when it expires",
                self._pool.name,
                lease.label,
                lease.amount,
                extra={"extra_data": {"pool": self._pool.name, "label": lease.label, "model": self._model_id}},
            )

    async def failed(self, lease: KeyPoolLease, exc: Exception, meter: _Meter, *, stream_started: bool) -> bool:
        """Close a failed attempt's lease by what its failure says; whether to try another key.

        :param lease: the attempt's lease
        :ptype lease: KeyPoolLease
        :param exc: what the attempt raised
        :ptype exc: Exception
        :param meter: the attempt's usage
        :ptype meter: _Meter
        :param stream_started: whether the caller has had a chunk of the attempt's answer
        :ptype stream_started: bool
        :return: ``True`` to run the call again, ``False`` to raise ``exc``
        :rtype: bool
        """
        self._last = exc
        if isinstance(exc, CircuitOpenError):
            # the key's breaker refused the call before it was sent: nothing to bill, and no verdict
            await lease.release()
            self._tried.add(lease.label)
            self._refused = (lease.label, MoveReason.TRIPPED)
            return True
        verdict = self._pool.classify(exc, stream_started=stream_started)
        if not verdict.moves_key:
            if meter.reported:
                # the provider answered and reported what it billed; what failed came after
                await lease.settle(meter.cost(lease.amount))
            else:
                await lease.fail(verdict)
            return False
        await lease.fail(verdict)
        if stream_started:
            return False
        if verdict.kind is VerdictKind.RATE_LIMITED and lease.label not in self._rate_limited:
            # it may serve once more, after its hold, if the pool waits for holds (Q4)
            self._rate_limited.add(lease.label)
        else:
            self._tried.add(lease.label)
        self._refused = (lease.label, _FAILOVER_REASON[verdict.kind])
        return True

    def open(self, lease: KeyPoolLease) -> bool:
        """Whether an attempt's lease still needs closing when the attempt ends: it isn't closed, and
        isn't a served attempt's whose settle failed, which stays open for the sweep to settle."""
        return not lease.closed and lease.reservation_id not in self._left_open

    async def abandoned(self, lease: KeyPoolLease, meter: _Meter) -> None:
        """Close the lease of an attempt that ended neither served nor failed -- cancelled, or a stream
        its caller stopped reading: it may have been billed, so at no less than its reservation."""
        await lease.settle(meter.cost(lease.amount))


def _prompt_messages(input: Any) -> list[BaseMessage]:  # noqa: A002, ANN401 -- the chat model's input
    """The messages a chat model's input stands for, for the reservation's token estimate."""
    if isinstance(input, PromptValue):
        return input.to_messages()
    if isinstance(input, str):
        return [HumanMessage(content=input)]
    return convert_to_messages(input)


# -- the synchronous paths' event loop ------------------------------------------------------------


@dataclass(frozen=True)
class _Bridge:
    """The private event loop the synchronous paths run pool calls on, and the thread running it."""

    loop: asyncio.AbstractEventLoop
    thread: threading.Thread


def _start_bridge() -> _Bridge:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="threetears-key-pool", daemon=True)
    thread.start()
    return _Bridge(loop, thread)


#: One bridge per process, started on first use and started again if its thread has died.
_BRIDGE: BuildOnce[None, _Bridge] = BuildOnce(is_current=lambda bridge: bridge.thread.is_alive())


def _run_sync[T](call: Awaitable[T]) -> T:
    """Run one pool call on the private loop and wait for it, from any thread but that loop's.

    The caller's context variables go with it. Interrupted while waiting, the call is cancelled.
    """
    bridge = _BRIDGE.get(None, _start_bridge)
    loop = bridge.loop
    if threading.current_thread() is bridge.thread:
        raise RuntimeError("a pooled model's synchronous path was called from its own pool loop")

    async def run() -> T:
        return await call

    future: Future[T] = asyncio.run_coroutine_threadsafe(run(), loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise
