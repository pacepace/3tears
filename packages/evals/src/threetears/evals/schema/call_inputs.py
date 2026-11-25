"""What a model call is asked about, named by its inputs: the binding a recorded replay keys a call by.

A recorded replay (:class:`~threetears.evals.testing.RecordedCompletions`) matches a call on its prompt. That is right
where the prompt is the input: a candidate's reply, or a judge reading a candidate's own output. It is wrong where the
prompt holds text the package rendered from the inputs, such as a report page a judge reads: a word changed on the page
changes the prompt, the call is no longer found, and only a paid re-record clears it, though nothing that was asked
changed.

So the caller that knows what a call is about binds it here, as structured values (which page by its stored document's
digest, which criterion by its prompt's digest, which case), for the extent of the call. A recorder that finds inputs
bound keys the call by them instead of by its prompt. Nothing else reads them, and a call made with none bound is keyed
by its prompt exactly as before.

The binding is a context variable, as the cell identity is (:func:`~threetears.evals.schema.traces.bind_cell_identity`),
so concurrent calls each see their own and a task started inside the block inherits it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

__all__ = ["bind_call_inputs", "current_call_inputs"]

#: The inputs the running call is asked about, or ``None`` where its caller bound none.
_CURRENT_INPUTS: ContextVar[Mapping[str, Any] | None] = ContextVar("threetears_evals_call_inputs", default=None)


def current_call_inputs() -> Mapping[str, Any] | None:
    """The inputs bound around the calling code's model call, or ``None`` where none are.

    :return: The inputs, read-only: a JSON object of structured values.
    """
    return _CURRENT_INPUTS.get()


@contextmanager
def bind_call_inputs(inputs: Mapping[str, Any]) -> Iterator[None]:
    """Make ``inputs`` the answer of :func:`current_call_inputs` for the extent of the block.

    Bind them around one call (and the retries that re-ask it), never around calls about different things: every
    call made inside the block is keyed by them. The previous value is restored on exit, including when the block is
    cancelled or raises.

    :param inputs: What the call is about, as a JSON object: values that move only when what is asked moves, never
        text a presentation change can reword.
    :yield: Nothing.
    """
    token = _CURRENT_INPUTS.set(MappingProxyType(dict(inputs)))
    try:
        yield
    finally:
        _CURRENT_INPUTS.reset(token)
