"""Per-run usage totals, and adding callbacks to a factory-built model.

:class:`~threetears.models.UsageTracker` records one row per LLM call. Metering a TURN -- a chat turn
that calls the model several times, a capture pass, a continuity check -- needs the total of the calls
it made, and both multi-tenant consumers hand-rolled that accumulator. :class:`UsageAccumulator` is it.

``create_chat_model`` returns a ``RunnableBinding`` (the model with its tracker and breaker callbacks
bound), where ``model_copy(update={"callbacks": ...})`` silently does nothing. :func:`attach_callbacks`
adds handlers to it (or to a bare model) and keeps the ones already bound.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableBinding

from threetears.models.tracking import TokenSource, extract_usage

__all__ = ["UsageAccumulator", "attach_callbacks"]


def _combine(current: TokenSource | None, new: TokenSource) -> TokenSource:
    """the source of a total.

    ``"reported"`` only while every call was reported, and ``"unavailable"`` only while no call
    counted anything. any mix -- a reported call beside an estimated one, or beside one that counted
    nothing -- is ``"estimated"``: the total is not the provider's exact figure.

    :param current: the source so far (``None`` before the first call)
    :ptype current: TokenSource | None
    :param new: the next call's source
    :ptype new: TokenSource
    :return: the combined source
    :rtype: TokenSource
    """
    return new if current is None or current == new else "estimated"


class UsageAccumulator(BaseCallbackHandler):
    """totals the usage of every LLM call it hears, for metering one run.

    attach it for the run (``attach_callbacks(model, accumulator)``, or ``extra_callbacks=`` at
    ``create_chat_model``), then read the totals. runs inline, so the totals are complete the
    moment a call returns.

    :param cost_per_input_token: USD per input token, to total ``cost_usd``
    :ptype cost_per_input_token: Decimal | None
    :param cost_per_output_token: USD per output token
    :ptype cost_per_output_token: Decimal | None
    :ivar input_tokens: input tokens so far
    :ivar output_tokens: output tokens so far
    :ivar cache_read_tokens: prompt-cache reads so far
    :ivar cache_creation_tokens: prompt-cache writes so far
    :ivar calls: LLM calls heard
    :ivar token_source: ``"reported"`` while every call's counts came from the provider,
        ``"unavailable"`` while none counted anything, ``"estimated"`` for any mix, ``None`` before
        the first call
    """

    run_inline = True

    def __init__(
        self,
        *,
        cost_per_input_token: Decimal | None = None,
        cost_per_output_token: Decimal | None = None,
    ) -> None:
        super().__init__()
        self._cost_in = cost_per_input_token
        self._cost_out = cost_per_output_token
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.calls = 0
        self.token_source: TokenSource | None = None
        self._prompts: dict[Any, list[Any]] = {}

    @property
    def total_tokens(self) -> int:
        """input plus output tokens so far.

        :return: the total
        :rtype: int
        """
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Decimal | None:
        """the cost so far at the configured per-token prices; ``None`` without prices.

        :return: the cost
        :rtype: Decimal | None
        """
        if self._cost_in is None or self._cost_out is None:
            return None
        return self._cost_in * Decimal(self.input_tokens) + self._cost_out * Decimal(self.output_tokens)

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """remember the prompt, for an input estimate if the provider reports no usage.

        :param serialized: the model (unused)
        :ptype serialized: dict[str, Any]
        :param messages: the prompt
        :ptype messages: list[list[Any]]
        :param run_id: the call's run id
        :ptype run_id: UUID
        :param kwargs: other callback context (unused)
        :ptype kwargs: Any
        :return: None
        :rtype: None
        """
        _ = serialized, kwargs
        self._prompts[run_id] = [m for batch in messages for m in batch]

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        """remember a completion-style call's prompt, for an input estimate.

        :param serialized: the model (unused)
        :ptype serialized: dict[str, Any]
        :param prompts: the prompt strings
        :ptype prompts: list[str]
        :param run_id: the call's run id
        :ptype run_id: UUID
        :param kwargs: other callback context (unused)
        :ptype kwargs: Any
        :return: None
        :rtype: None
        """
        _ = serialized, kwargs
        self._prompts[run_id] = [SimpleNamespace(content=p) for p in prompts]

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """add one call's usage to the totals.

        :param response: the call's result
        :ptype response: LLMResult
        :param run_id: the call's run id
        :ptype run_id: UUID
        :param kwargs: other callback context (unused)
        :ptype kwargs: Any
        :return: None
        :rtype: None
        """
        _ = kwargs
        extracted = extract_usage(response, prompt_messages=self._prompts.pop(run_id, None))
        self.input_tokens += extracted.input_tokens
        self.output_tokens += extracted.output_tokens
        self.cache_read_tokens += extracted.cache_read_tokens
        self.cache_creation_tokens += extracted.cache_creation_tokens
        self.calls += 1
        self.token_source = _combine(self.token_source, extracted.source)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """forget a failed call's prompt.

        :param error: the failure (unused)
        :ptype error: BaseException
        :param run_id: the call's run id
        :ptype run_id: UUID
        :param kwargs: other callback context (unused)
        :ptype kwargs: Any
        :return: None
        :rtype: None
        """
        _ = error, kwargs
        self._prompts.pop(run_id, None)


def attach_callbacks(model: BaseChatModel, *handlers: BaseCallbackHandler) -> BaseChatModel:
    """``model`` with ``handlers`` added to the callbacks it already carries.

    works on the ``RunnableBinding`` ``create_chat_model`` returns -- whose tracker and breaker
    callbacks are kept, whether bound as a list or inside a callback manager -- and on a bare chat
    model. (``with_config(callbacks=...)`` REPLACES a binding's callbacks, and ``model_copy`` drops
    them, which is why this exists.)

    :param model: a chat model, bound or bare
    :ptype model: BaseChatModel
    :param handlers: the callbacks to add
    :ptype handlers: BaseCallbackHandler
    :return: the model with every callback attached
    :rtype: BaseChatModel
    """
    existing: list[Any] = []
    if isinstance(model, RunnableBinding) and model.config:
        bound = model.config.get("callbacks")
        if isinstance(bound, list):
            existing = list(bound)
        elif isinstance(bound, BaseCallbackManager):
            existing = list(bound.handlers)
    return cast("BaseChatModel", model.with_config(callbacks=[*existing, *handlers]))
