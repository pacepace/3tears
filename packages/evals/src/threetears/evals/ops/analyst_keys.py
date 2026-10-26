"""A key check as an operation: it reads one snapshot's frozen report input and reads each key on it.

The operation reads the input by the snapshot's id (:func:`~threetears.evals.ops.analyst.snapshot_input`), so the
analyst and the editor check keys against the figures the report was written from, and runs the check
(:func:`~threetears.evals.analysis.analyst.keys.figure_keys_check`) on the host's blocking executor. A check that
refuses its arguments (no key, too many) is refused as the call, with the remedy it names; a key that reads nothing is
a row saying why, not a refusal of the call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from threetears.evals.analysis.analyst.keys import KEYS_LIMIT, KEYS_MOST, KeysChecked
from threetears.evals.analysis.analyst.keys import figure_keys_check as checked
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analyst import sized_page, snapshot_input
from threetears.evals.ops.host import OpsHost


async def figure_keys_check(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    keys: Sequence[str],
    limit: int = KEYS_LIMIT,
    offset: int = 0,
    fits: Callable[[KeysChecked], bool] | None = None,
) -> KeysChecked:
    """What each key reads on a snapshot, or why it reads nothing and what to write instead.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param keys: The keys, as a report writes them.
    :ptype keys: Sequence[str]
    :param limit: The most keys one page holds.
    :ptype limit: int
    :param offset: The first key the page holds.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``None`` takes ``limit`` as it is.
    :ptype fits: Callable[[KeysChecked], bool] | None
    :return: Each key on the page, read or refused.
    :rtype: KeysChecked
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; or no key is given, or too many.
    """
    held = await snapshot_input(host, scope_id, snapshot_id)
    try:
        return await run_blocking(
            host.eval_host.blocking_executor,
            sized_page,
            lambda rows: checked(held, keys, limit=rows, offset=offset),
            limit,
            fits,
        )
    except RefusedInput as refused:
        raise ValidationFailedError.refusing(refused) from refused


__all__ = ["KEYS_LIMIT", "KEYS_MOST", "KeysChecked", "figure_keys_check"]
