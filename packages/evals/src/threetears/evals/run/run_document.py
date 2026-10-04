"""The safe-edit protocol for an eval run document.

A run document is written by more than one writer as a run finishes — its
completeness record, its terminal status, and the no-live-job repair path — and
the store reports every write failure the same way, by returning ``False``. A
caller that reads that value once and gives up loses the edit silently. This
module holds the read-modify-write policy that answer needs, plus the two-method
port it composes: :func:`update_eval_run` over
:class:`EvalRunDocumentStore`.

**Why it is not in** :mod:`threetears.evals.contracts.storage`. The policy is about the
conditional-write protocol, and it names no backend, no tier and no host —
:class:`~threetears.evals.contracts.storage.EvalStorage` is one implementation of the two
methods it drives, and a test double is another; the policy depends on neither.

The scope argument is named ``scope_id`` here, as it is everywhere in the
engine's own vocabulary: what partitions a run document is the host's business
and the engine passes it back untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from threetears.evals.contracts.errors import ConflictError
from threetears.evals.contracts.models import EvalRun
from threetears.observe import get_logger

log = get_logger(__name__)

#: What :func:`update_eval_run` reports back: the write landed, there was no such
#: run to write to, or no attempt completed (a rejected write or a faulting read).
RunUpdateOutcome = Literal["saved", "missing", "refused", "declined"]

#: Read-modify-write attempts before :func:`update_eval_run` gives up.
#: Deliberately a constant rather than a configured limit: it governs how hard a
#: write *about* an eval run tries to land, never whether the run's own work is
#: allowed to finish — the cells are executed and persisted before either caller
#: reaches this, and no value here can rescue or kill one. Small on purpose. Each
#: attempt costs a fresh read, so this trades against nothing but latency on a
#: path that is already contended; what it buys is that a single concurrent
#: writer — the ordinary case, since the terminal-status write follows the
#: completeness write on the same document — no longer costs the edit. A run
#: document that refuses three consecutive conditional writes is a sustained
#: condition and wants an operator, not a fourth try.
RUN_DOCUMENT_WRITE_ATTEMPTS = 3


class EvalRunDocumentStore(Protocol):
    """The two primitives :func:`update_eval_run` composes into a safe edit.

    Narrower than :class:`~threetears.evals.contracts.storage.EvalStorage` on purpose: the
    retry policy is about the conditional-write protocol, not about any one database, so
    it is stated over the protocol and every caller — production or test — drives
    the same policy through whatever implements these two.

    It is also the whole of what :class:`~threetears.evals.run.jobs.EvalJobManager`
    asks of storage, so the manager takes this rather than declaring a second
    port with the same two members.

    The positional parameters are positional-only, so an implementation is free
    to name the scope after whatever it partitions by. Every caller here passes
    them positionally, and a port that pinned the names would make the engine's
    vocabulary a constraint on the host's.
    """

    def load_eval_run_with_etag(self, run_id: str, scope_id: str, /) -> tuple[EvalRun | None, str | None]:
        """Load a run plus the ETag a conditional write must present."""
        ...

    def save_eval_run(self, run: EvalRun, /, *, if_match: str | None = None) -> None:
        """Write a run; raises ``StorageError`` (``ConflictError`` on a lost ``if_match``) rather than returning a flag."""
        ...


def update_eval_run(
    store: EvalRunDocumentStore,
    run_id: str,
    scope_id: str,
    mutate: Callable[[EvalRun], dict[str, Any] | None],
    *,
    attempts: int = RUN_DOCUMENT_WRITE_ATTEMPTS,
) -> RunUpdateOutcome:
    """Read a run, apply ``mutate`` to it, and write it back under its own ETag.

    The write is conditional, so a concurrent writer refuses it — and
    :meth:`~threetears.evals.contracts.storage.EvalStorage.save_eval_run` reports *every*
    failure the same way, an ETag conflict and a dropped connection alike, by
    returning ``False``. A caller that read that value once and gave up lost the
    edit silently, which is how a finished run came to carry no completeness
    record at all.

    So a refusal re-reads and re-applies. Re-sending the document from the first
    read cannot be right in either direction: with the stale ETag the write is
    refused again forever, and without one it overwrites whatever the other
    writer just committed. ``mutate`` therefore runs again on each attempt,
    against the winner's document, and must be written to suit that — it receives
    the current run and returns the document to save, rather than editing a dict
    captured outside the loop.

    Failure is reported, never raised. Its callers write *about* a run whose real
    work is already finished and persisted; raising here would reach the job's
    top-level boundary and stamp ``failed`` on a run that succeeded, trading a
    line of disclosure for a false verdict about the run it describes.

    Args:
        store: Where the run lives.
        run_id: The run to update.
        scope_id: Partition key — the scope the run lives in, opaque to the
            engine, which passes it back to the store and never interprets it.
            The log lines below key it ``scope=``, as every ``eval.*`` line in
            the package does, so an operator greps one vocabulary across a
            whole run.
        mutate: Given the run as currently stored, returns the document to save,
            or ``None`` to decline the write. Called once per attempt — and the
            decision to decline belongs here rather than at the call site
            precisely because this is the only place that has read the WINNER's
            document. A caller that checked the run before calling has checked a
            document another writer may already have replaced.
        attempts: How many read-modify-writes before giving up.

    Returns:
        ``"saved"`` when the write landed; ``"declined"`` when ``mutate`` returned
        ``None``, which is a decision and not a failure — the stored document is
        the one that should stand, and a caller must not report or broadcast the
        state it was going to write; ``"missing"`` when there is no such run
        to update (a delete racing the write, or a job whose run document was
        never created); ``"refused"`` when no attempt completed — which covers a
        conditional write rejected every time AND a read that faulted every time,
        deliberately as one outcome, because both leave the caller with the same
        fact (the edit is not stored) and the same remedy (none, in-process). The
        two are distinguishable in the log, where the read fault is logged with
        its exception; a caller's own message must therefore not assert *which*
        one happened.
    """
    for attempt in range(1, attempts + 1):
        try:
            current, etag = store.load_eval_run_with_etag(run_id, scope_id)
        except (
            Exception
        ):  # prawduct:ok-broad-except — a read fault must not raise into a caller whose run already finished
            log.exception(
                "eval.update_run read failed run=%s scope=%s (attempt %d/%d)", run_id, scope_id, attempt, attempts
            )
            continue
        if current is None:
            return "missing"
        document = mutate(current)
        if document is None:
            return "declined"
        try:
            store.save_eval_run(EvalRun.from_dict(document), if_match=etag)
        except ConflictError:
            # A lost race: another writer's document now stands. Says what happens NEXT, so the last line does not
            # promise a re-apply that is not coming — the caller's own error log is what follows it.
            log.warning(
                "eval.update_run write refused run=%s scope=%s (attempt %d/%d) — %s",
                run_id,
                scope_id,
                attempt,
                attempts,
                "re-reading and re-applying" if attempt < attempts else "giving up",
            )
            continue
        except Exception:  # prawduct:ok-broad-except — same boundary as the read above
            log.exception(
                "eval.update_run write failed run=%s scope=%s (attempt %d/%d)", run_id, scope_id, attempt, attempts
            )
            continue
        return "saved"
    return "refused"


__all__ = [
    "RUN_DOCUMENT_WRITE_ATTEMPTS",
    "EvalRunDocumentStore",
    "RunUpdateOutcome",
    "update_eval_run",
]
