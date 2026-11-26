"""a pod's write generations, advanced by the L3 broker that committed its writes.

A pod cannot write the epoch bucket: a pod that could would be able to fake an advance the whole
fleet acts on. So the hub's L3 broker, which ends every transaction a pod's write runs in, advances
the generation of each table the commit wrote that is switched on, after the commit, and names the
token it wrote for each one in its reply. This module is the pod's half of that:

- :class:`NatsProxyL3Backend <threetears.core.backends.nats_proxy.NatsProxyL3Backend>` hands every
  reply that ends a write to :func:`record_reply_generations`, which keeps the tokens it names for
  the code that made the request;
- :class:`BrokerGenerationSource` is the :class:`~threetears.core.collections.generation.GenerationSource`
  a pod's registry is wired with. A collection calls ``advance`` after its commit, exactly as it does
  over the epoch bucket directly, and is handed the token the broker's reply carried for its table.

**"The commit just made" is the calling code's own.** Tokens are kept in a context variable, so a
collection is answered only from replies to requests made in its own asyncio task, never another
task's: two tasks writing one table concurrently are each handed their own commit's token. A task
started while tokens are held (``asyncio.shield`` around a transaction's settling is one) reads the
tokens of the task that started it; one that makes a write request of its own starts its own
record, and leaves the record it inherited as it was.

**One commit, one advance per table.** The broker advanced each table once for the commit, and a
follower counts that advance complete once it has heard as many row broadcasts as the broadcasts
say it covered. So a commit's token is handed out once per table, to the one advance that stamps
every broadcast of that commit with their total: a collection that updates many rows in one
statement evicts them in one call (``invalidate_cache_many``), and a
:class:`~threetears.core.collections.caller_transaction.CallerTransaction` settling two collection
instances of one table advances it once for the rows of both. A second advance of the table for
the same commit raises :class:`~threetears.core.exceptions.GenerationUnavailableError` rather than
publish rows under a count the first already gave.

**A token belongs to the commit that produced it.** Every reply that ends a write -- an ``l3.query``
whose statement writes, an ``l3.batch``, an ``l3.tx.commit`` -- replaces the task's record with its
own, naming generations or not, so an advance after a commit whose reply named nothing for the
table is never handed an earlier commit's token. A read's reply, which ends no write, leaves the
record as it was. A transaction that rolls back, a commit the broker refuses, and a commit whose
reply never came back leave a record that says so
(:class:`~threetears.core.exceptions.GenerationNotCommittedError`).

**What a reply says.** Every reply that ends a commit is read -- ``l3.query``, ``l3.batch`` (a batch
run statement by statement that failed partway included, since the statements before the failure
committed) and ``l3.tx.commit``. :data:`GENERATIONS_REPLY_FIELD` maps each switched-on table the commit wrote
to the token the broker's advance wrote. :data:`GENERATIONS_FAILED_REPLY_FIELD` lists each one the
broker could not advance, beside :data:`GENERATION_UNAVAILABLE_ERROR_CODE`, on a reply that is
still a success: the rows were committed, so the write must not be retried. A collection advancing
such a table raises :class:`~threetears.core.exceptions.GenerationUnavailableError`. Both fields are
optional: a pod built before them reads the reply as before.

**A reply that carries neither field says the broker advanced nothing** (owner, 2026-10-08). A broker
built before generations, or one whose hub does not yet hold the table's class switched on, sends
neither, so a pod switched on ahead of the hub must still write: its advance returns ``None``, its
rows go out naming no generation, exactly as an undeclared table's do, and a warning is logged once
per table. No follower is misled by it: none trusts an unmoved generation while any writer of the
table is on an older release. The alternative, raising and so requiring every hub to be released
before any pod, is lockstep and was rejected. A reply that does carry the fields is a broker that
advances: one listing the table failed, or naming generations and not this table, still raises.

**How each side knows a table's writes advance.** The pod: the collection's own class, which calls
``advance`` when it declares ``write_generation = WRITE_GENERATION`` or caches absences. The broker:
the same class,
imported in the hub, through
:func:`~threetears.core.collections.base.tables_with_write_generation`. One class per table is what
keeps the two answers the same.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from threetears.core.ddl.row_epoch import UntouchedScopesError
from threetears.core.exceptions import GenerationNotCommittedError, GenerationUnavailableError
from threetears.core.frozen_state import EMPTY, frozen
from threetears.observe import BuildOnce, get_logger

__all__ = [
    "CHANGED_ROWS_FIELD",
    "CLEANUP_DELETE_FIELD",
    "CLEANUP_FLOOR_FIELD",
    "COMMIT_SCOPES_FIELD",
    "COMMIT_S_MIN_FIELD",
    "DELTA_FILES_FIELD",
    "GENERATIONS_FAILED_REPLY_FIELD",
    "GENERATIONS_REPLY_FIELD",
    "GENERATION_BY_EPOCH",
    "GENERATION_UNAVAILABLE_ERROR_CODE",
    "HUB_CAPABILITIES_FIELD",
    "STAMPS_EPOCH_FIELD",
    "STAMP_S_MIN_FIELD",
    "BrokerGenerationSource",
    "GenerationReader",
    "collecting_advanced_generations",
    "collecting_load_answers",
    "committed_generations_record",
    "forget_reply_generations",
    "named_changed_rows",
    "named_commit_generation",
    "named_commit_s_min",
    "named_commit_scopes",
    "named_delta_files",
    "named_unconfirmed_delta_files",
    "naming_changed_rows",
    "naming_commit_generations",
    "naming_commit_s_min",
    "naming_commit_scopes",
    "naming_delta_files",
    "naming_unconfirmed_delta_files",
    "note_load_answer",
    "note_advanced_generation",
    "CleanupMixedWriteError",
    "cleanup_delete_floor",
    "cleanup_deleting",
    "cleanup_mix_refusal",
    "cleanup_write_refusal",
    "refuse_cleanup_non_delete",
    "holding_sent_requests",
    "note_sent_request",
    "releasing_sent_requests",
    "note_statement_stamp",
    "record_reply_generations",
    "reply_generations_since",
    "StampedScopes",
    "holding_touched_scopes",
    "releasing_touched_scopes",
    "stamp_fields",
    "stamped_send",
    "stamping_write",
    "taken_statement_stamp",
]

_logger = get_logger(__name__)

#: the reply field in which the hub's broker names what it can do, on every ``l3.query`` reply: a list of
#: capability names. A broker built before it sends none, which a pod reads as no capability at all. The hub
#: imports this name.
HUB_CAPABILITIES_FIELD: Final = "hub_capabilities"

#: the broker advances a table switched on by the epoch it carries alone (a tool pod's own table, which no class the
#: hub imports declares), keyed by the schema it lives in
#: (:func:`~threetears.core.collections.lost_writes.tables_switched_by_epoch`; the change-note trigger until the ring
#: retired, append-only L3 chunk 7, and the write-mark trigger before it). A pod whose copy follows such tables
#: (:class:`~threetears.core.collections.copy_follower.CopyFollower`) refuses to serve without it: a hub without it
#: advances nothing for them, and the copy would be silently stale. A new name on the wire: a hub that switched tables
#: by their trigger announces the old one, which a pod on this release refuses (the outage-window cutover)
GENERATION_BY_EPOCH: Final = "generation_by_epoch"

#: the request field naming, for each table a collection's write changes, the rows it changes:
#: ``{table: [ChangedRow, ...]}``, each row as ``ChangedRow`` dumps it (``ids``, and the optional
#: ``l2_current_scope`` and ``columns``). Sent on ``l3.query``, ``l3.batch`` and ``l3.tx.commit`` for a
#: write made inside :func:`naming_changed_rows`, so the broker that advances the table's generation
#: stores exactly those rows as the epoch's change record. A write that names none sends no field;
#: the broker then records the epoch's keys as unknown. The hub imports this name.
CHANGED_ROWS_FIELD: Final = "changed_rows"

#: the request field naming, for each table a commit wrote that opts in to delta files, the delta file saved for it
#: before the commit: ``{table: "<table>/<write id>.arrow"}``, the name under the shared volume's delta files'
#: directory (:mod:`~threetears.core.collections.delta_files`). Sent on ``l3.tx.commit``, and on ``l3.load_files``
#: for a load of a table that opts in (each request names its own file, holding that request's rows), so the broker
#: that advances each table's generation names the file in every epoch record it writes for the commit or the request
#: (``EpochChange.delta_file``; ``design-pod-delta-files.md``, decisions 4 and 17). A load request that fails after
#: committing some of its rows names its file unconfirmed (``unconfirmed_delta_file``, ruling C of 2026-11-23): the file
#: may hold rows L3 did not take, so no copy applies it until the writer has written it again from L3. A broker built
#: before it ignores it, and its records name no file. The hub imports this name.
DELTA_FILES_FIELD: Final = "delta_files"

#: the delta files of the commit being made in this context, by table (:func:`naming_delta_files`)
_DELTA_FILES: ContextVar[Mapping[str, str]] = ContextVar("threetears_commit_delta_files", default={})


@contextmanager
def naming_delta_files(files: Mapping[str, str]) -> Iterator[None]:
    """name the delta files of the commit made inside this block, by table: its ``l3.tx.commit`` carries them
    (:data:`DELTA_FILES_FIELD`), and an advance a process writes its own record for names them
    (:func:`~threetears.core.collections.generation.advance_for_write`). Replaces, never adds to, what an enclosing
    block named: a commit is named by its own files alone.

    :param files: table -> its delta file's name; empty names none
    :ptype files: Mapping[str, str]
    :return: a context in which the commit names them
    :rtype: Iterator[None]
    """
    token = _DELTA_FILES.set(dict(files))
    try:
        yield
    finally:
        _DELTA_FILES.reset(token)


def named_delta_files() -> dict[str, str]:
    """the delta files of the commit being made in this context, by table, as :data:`DELTA_FILES_FIELD` carries them.

    :return: table -> its delta file's name; empty when the commit names none
    :rtype: dict[str, str]
    """
    return dict(_DELTA_FILES.get())


#: the tokens the broker's reply to the commit being settled in this context named, by table
#: (:func:`naming_commit_generations`): read-only, replaced whole by each block
_COMMIT_GENERATIONS: ContextVar[Mapping[str, str]] = ContextVar("threetears_commit_generations", default=EMPTY)


@contextmanager
def naming_commit_generations(tokens: Mapping[str, str]) -> Iterator[None]:
    """name, while a commit's writes are settled, the token the broker's reply to that commit named for each table it
    wrote (:func:`reply_generations_since`): a collection that advances no generation itself (a table the broker
    advances by its trigger alone) stamps the keys it tells a copy, and its row messages, with its table's
    (:func:`named_commit_generation`). Replaces, never adds to, what an enclosing block named.

    Why: such a table's local write told the copy its keys at no epoch, so the copy read every one of them from L3 by
    key instead of merging it from the commit's delta file (``copy_follower_delta_file_fallback_keys_total``,
    ``why="no_epoch"``): 62,800 to 98,200 keys a pass after a reset rewrote every state (the local burst replay,
    2026-11-25), while the commit's epoch record named the file all along.

    :param tokens: table -> the token the reply named; empty names none (a direct pool's commit, one that did not land)
    :ptype tokens: Mapping[str, str]
    :return: a context in which they are named
    :rtype: Iterator[None]
    """
    token = _COMMIT_GENERATIONS.set(frozen(tokens))
    try:
        yield
    finally:
        _COMMIT_GENERATIONS.reset(token)


def named_commit_generation(table_name: str) -> str | None:
    """the token the broker's reply to the commit being settled in this context named for ``table_name``
    (:func:`naming_commit_generations`): the epoch whose record names the commit's delta file of the table.

    :param table_name: the table
    :ptype table_name: str
    :return: the token; None outside a settle, or when the reply named none for the table
    :rtype: str | None
    """
    return _COMMIT_GENERATIONS.get().get(table_name)


#: the request field naming the scopes a commit's transaction touched (the states, ``state:CA``, as the writer's
#: seqlock names them: ``ScopeEpochs.touch``), sent on ``l3.tx.commit`` beside :data:`DELTA_FILES_FIELD` when the
#: commit saved a delta file: a list of at most ``_EPOCH_SCOPES_MAX`` strings (``recordable_scopes``), none past it.
#: A broker that records the commit keys unknown (its rows too many to name) names them on each such table's record
#: (``EpochChange.scopes``), so the writer can recover the commit's rows by those states should its delta file be lost
#: past the change notes' window (``design-pod-delta-files.md``, decision 31). A broker built before it ignores it,
#: and its records name none. The hub imports this name.
COMMIT_SCOPES_FIELD: Final = "scopes"

#: the unconfirmed delta files of the advance being made in this context, by table (:func:`naming_unconfirmed_delta_files`)
_UNCONFIRMED_DELTA_FILES: ContextVar[Mapping[str, str]] = ContextVar("threetears_unconfirmed_delta_files", default={})


@contextmanager
def naming_unconfirmed_delta_files(files: Mapping[str, str]) -> Iterator[None]:
    """name, unconfirmed, the delta files of a load request that failed after it may have committed some rows, by
    table: an advance a process writes its own record for inside this block names each so
    (:func:`~threetears.core.collections.generation.advance_for_write`, ``EpochChange.delta_file_unconfirmed``). For a
    process that owns its load's records (a direct pool); a pod's broker names them itself. Replaces what an enclosing
    block named.

    :param files: table -> the request's delta file's name; empty names none
    :ptype files: Mapping[str, str]
    :return: a context in which the advance names them unconfirmed
    :rtype: Iterator[None]
    """
    token = _UNCONFIRMED_DELTA_FILES.set(dict(files))
    try:
        yield
    finally:
        _UNCONFIRMED_DELTA_FILES.reset(token)


def named_unconfirmed_delta_files() -> dict[str, str]:
    """the unconfirmed delta files of the advance being made in this context, by table.

    :return: table -> its unconfirmed delta file's name; empty when none is named
    :rtype: dict[str, str]
    """
    return dict(_UNCONFIRMED_DELTA_FILES.get())


#: where an ``l3.load_files`` request made in this context notes the broker's answer: the rows the broker says it
#: committed, success or failure (:func:`collecting_load_answers`); None when nothing collects them
_LOAD_ANSWERS: ContextVar[list[int] | None] = ContextVar("threetears_load_answers", default=None)


@contextmanager
def collecting_load_answers(into: list[int]) -> Iterator[None]:
    """collect into ``into`` the rows the broker answered it committed, for each ``l3.load_files`` request made inside
    this block, success or failure (:func:`note_load_answer`). How a pod tells a request the broker answered (it owns
    the request's epoch record: one naming the file unconfirmed when rows committed, none when none did) from one that
    had no answer (a cancel, a lost reply). ``into`` is the caller's own, local to the one request.

    :param into: the rows each answer named, appended as answers arrive
    :ptype into: list[int]
    :return: a context in which answers are collected
    :rtype: Iterator[None]
    """
    token = _LOAD_ANSWERS.set(into)
    try:
        yield
    finally:
        _LOAD_ANSWERS.reset(token)


def note_load_answer(rows: int) -> None:
    """note the broker's answer to an ``l3.load_files`` request, where this context collects them; nothing otherwise.

    :param rows: the rows the answer says the broker committed
    :ptype rows: int
    :return: nothing
    :rtype: None
    """
    into = _LOAD_ANSWERS.get()
    if into is not None:
        into.append(rows)


#: the scopes the commit being made in this context touched (:func:`naming_commit_scopes`)
_COMMIT_SCOPES: ContextVar[tuple[str, ...]] = ContextVar("threetears_commit_scopes", default=())


@contextmanager
def naming_commit_scopes(scopes: Sequence[str]) -> Iterator[None]:
    """name the scopes the commit made inside this block touched: its ``l3.tx.commit`` carries them
    (:data:`COMMIT_SCOPES_FIELD`), and a keys-unknown advance a process writes its own record for names them
    (:func:`~threetears.core.collections.generation.advance_for_write`). Replaces what an enclosing block named.

    :param scopes: the scopes, already bounded (``recordable_scopes``); empty names none
    :ptype scopes: Sequence[str]
    :return: a context in which the commit names them
    :rtype: Iterator[None]
    """
    token = _COMMIT_SCOPES.set(tuple(scopes))
    try:
        yield
    finally:
        _COMMIT_SCOPES.reset(token)


def named_commit_scopes() -> tuple[str, ...]:
    """the scopes the commit being made in this context touched, as :data:`COMMIT_SCOPES_FIELD` carries them.

    :return: the scopes; empty when the commit names none
    :rtype: tuple[str, ...]
    """
    return _COMMIT_SCOPES.get()


#: the request field naming the lowest row epoch a commit's transaction stamped (``s_min``; the row epoch's hybrid
#: logical clock, :mod:`~threetears.core.ddl.row_epoch`), sent on ``l3.tx.commit`` when the transaction stamped any:
#: one BIGINT. The broker names it on every epoch record it writes for the commit (``EpochChange.s_min``), so a scrub of
#: a restore proves its margin held: a record after the keeper's mark P whose ``s_min`` is below the restore's floor X
#: is a violation (``design-append-only-l3.md``, decision 3). A broker built before it ignores it, and its records name
#: none. The hub imports this name.
COMMIT_S_MIN_FIELD: Final = "s_min"

#: the lowest row epoch the commit being made in this context stamped (:func:`naming_commit_s_min`)
_COMMIT_S_MIN: ContextVar[int | None] = ContextVar("threetears_commit_s_min", default=None)


@contextmanager
def naming_commit_s_min(s_min: int | None) -> Iterator[None]:
    """name the lowest row epoch the commit made inside this block stamped: its ``l3.tx.commit`` carries it
    (:data:`COMMIT_S_MIN_FIELD`), and an advance a process writes its own record for names it
    (:func:`~threetears.core.collections.generation.advance_for_write`). Replaces what an enclosing block named.

    :param s_min: the stamp; None names none (a commit that stamped no row, or one that did not land)
    :ptype s_min: int | None
    :return: a context in which the commit names it
    :rtype: Iterator[None]
    """
    token = _COMMIT_S_MIN.set(s_min)
    try:
        yield
    finally:
        _COMMIT_S_MIN.reset(token)


def named_commit_s_min() -> int | None:
    """the lowest row epoch the commit being made in this context stamped, as :data:`COMMIT_S_MIN_FIELD` carries it.

    :return: the stamp; None when the commit names none
    :rtype: int | None
    """
    return _COMMIT_S_MIN.get()


#: the request field saying whether the write a request carries stamps row epochs: ``true`` for rows of a table that
#: declares a row epoch (``TableSchema.stamps_epoch``: ``row_versions`` or ``ROW_STAMPED``), each carrying its write's
#: stamp, ``false`` for every other request. Sent on every L3 request this backend makes (``l3.query``, ``l3.batch``,
#: ``l3.load_files`` and every ``l3.tx.*`` statement, the copy and Arrow door and ``delete_prefixes`` among them):
#: only 3tears knows which tables stamp, so it says so, and the broker reads it rather than guessing from the
#: catalog (a column named ``epoch`` is no row epoch: ``scope_epochs`` holds one). The broker, for a request that
#: says ``true``: rises a stamped table's epoch on its upsert (``rising_epoch_assignment``), marks its transaction
#: stamped, so its commit must name :data:`COMMIT_S_MIN_FIELD`, and names :data:`STAMP_S_MIN_FIELD` on every epoch
#: record of a write it commits itself (``l3.query``, ``l3.batch``, ``l3.load_files``). A request that carries an
#: epoch and says neither is refused (the release is an outage-window cutover: no pod sends the older form). The hub
#: imports this name.
STAMPS_EPOCH_FIELD: Final = "stamps_epoch"

#: beside :data:`STAMPS_EPOCH_FIELD` ``true``, the lowest row epoch the request's rows carry: one BIGINT. The same
#: name as :data:`COMMIT_S_MIN_FIELD`, on purpose: on a request the broker commits itself it is that commit's
#: ``s_min``; inside a transaction the commit's own (the transaction's lowest) is never above it. The hub imports
#: this name.
STAMP_S_MIN_FIELD: Final = "s_min"

#: the lowest row epoch the write being sent in this context stamps (:func:`stamping_write`); None for a write that
#: stamps none
_STAMPED_WRITE: ContextVar[int | None] = ContextVar("threetears_stamped_write", default=None)


@contextmanager
def stamping_write(s_min: int | None, *, table: str | None = None) -> Iterator[None]:
    """send every L3 request made inside this block as a stamped write whose rows' lowest row epoch is ``s_min``
    (:data:`STAMPS_EPOCH_FIELD`, :data:`STAMP_S_MIN_FIELD`); None sends them unstamped. Replaces what an enclosing
    block named. The one place a write says it stamps: the store's writes of a table that declares a row epoch enter
    it around their requests (``SqlL3Backend``), and every request's payload reads it (:func:`stamp_fields`).

    Given ``table``, a block that sent its requests without raising notes ``s_min`` as that table's statement stamp
    (:func:`note_statement_stamp`), so the epoch record this process writes for the write names it, in a transaction
    or not (:func:`taken_statement_stamp`). A block that only measures requests names no table.

    :param s_min: the lowest stamp the block's rows carry, a positive BIGINT; None when they carry none
    :ptype s_min: int | None
    :param table: the table the block's requests write, or None for a block that sends nothing
    :ptype table: str | None
    :return: a context in which the requests say so
    :rtype: Iterator[None]
    :raises ValueError: when ``s_min`` is no row epoch
    """
    if s_min is not None and not 0 < s_min < 1 << 63:
        raise ValueError(f"a stamped write's s_min is a row epoch, a positive BIGINT: got {s_min}")
    token = _STAMPED_WRITE.set(s_min)
    try:
        yield
    finally:
        _STAMPED_WRITE.reset(token)
    # reached only when the block raised nothing: a write that failed names its stamp on no record
    if table is not None and s_min is not None:
        note_statement_stamp(table, s_min)


#: the stamps the stamped writes made in this context carried since their table's last advance, as ``(table, lowest)``
#: pairs, one a table: what the epoch record of a write that joined no transaction names as its ``s_min``
#: (:func:`taken_statement_stamp`). An immutable tuple, replaced whole: never shared mutable state. WHY a context, not
#: an argument: the stamp is drawn (``CallerTransaction.stamp_for``) and sent (``SqlL3Backend``, :func:`stamping_write`)
#: below the collection whose advance writes the record (``advance_for_write``), in the same task, and every write path
#: between them would otherwise carry it. A one-statement write is its own commit, so its ``s_min`` is its own stamp; a
#: transaction's commit names its own (:func:`naming_commit_s_min`), never above a stamp noted here
_STATEMENT_STAMPS: ContextVar[tuple[tuple[str, int], ...]] = ContextVar("threetears_statement_stamps", default=())


def note_statement_stamp(table: str, s_min: int) -> None:
    """note that a stamped write of ``table`` made in this context carried ``s_min``: the table's next advance in this
    context names the lowest noted since its last one as its record's ``s_min`` (:func:`taken_statement_stamp`).

    :param table: the table written
    :ptype table: str
    :param s_min: the lowest stamp the write's rows carry, a positive BIGINT
    :ptype s_min: int
    :raises ValueError: when ``s_min`` is no row epoch
    """
    if not 0 < s_min < 1 << 63:
        raise ValueError(f"a stamped write's s_min is a row epoch, a positive BIGINT: got {s_min}")
    held = _STATEMENT_STAMPS.get()
    lowest = min([s_min, *(stamp for name, stamp in held if name == table)])
    _STATEMENT_STAMPS.set((*(pair for pair in held if pair[0] != table), (table, lowest)))


def taken_statement_stamp(table: str) -> int | None:
    """the lowest stamp the stamped writes of ``table`` made in this context carried since its last advance
    (:func:`note_statement_stamp`), taken: the advance writing the table's record names it, and the next starts afresh.

    :param table: the table being advanced
    :ptype table: str
    :return: the stamp; None when no stamped write of the table was noted
    :rtype: int | None
    """
    held = _STATEMENT_STAMPS.get()
    found = [stamp for name, stamp in held if name == table]
    if not found:
        return None
    _STATEMENT_STAMPS.set(tuple(pair for pair in held if pair[0] != table))
    return found[0]


#: the request field marking a delete as quiet-time cleanup's (:mod:`~threetears.core.collections.quiet_cleanup`):
#: ``true`` on every L3 request made inside :func:`cleanup_deleting`, beside :data:`CLEANUP_FLOOR_FIELD`. A versioned
#: table (``row_versions``) is never deleted from but by cleanup: the broker refuses any other delete of one, and takes
#: a marked one only for stored versions below the named floor, a floor no higher than its own reading of the change
#: log's window (``cleanup_floor``). The hub imports this name.
CLEANUP_DELETE_FIELD: Final = "cleanup_delete"

#: beside :data:`CLEANUP_DELETE_FIELD`, the floor F the delete was worked out against
#: (:func:`~threetears.core.ddl.row_epoch.cleanup_floor`): every row it removes is stamped below it. One BIGINT. The
#: hub imports this name.
CLEANUP_FLOOR_FIELD: Final = "cleanup_floor"

#: the floor of the cleanup delete being sent in this context (:func:`cleanup_deleting`); None for any other request
_CLEANUP_FLOOR: ContextVar[int | None] = ContextVar("threetears_cleanup_floor", default=None)


@contextmanager
def cleanup_deleting(floor: int) -> Iterator[None]:
    """send every L3 request made inside this block as quiet-time cleanup's delete, worked out against ``floor``
    (:data:`CLEANUP_DELETE_FIELD`, :data:`CLEANUP_FLOOR_FIELD`). The one place a request says it is cleanup's: every
    request's payload reads it (:func:`stamp_fields`), so no door can send a cleanup delete unmarked.

    :param floor: the floor F the block's deletes stay below, a positive BIGINT
    :ptype floor: int
    :return: a context in which the requests say so
    :rtype: Iterator[None]
    :raises ValueError: when ``floor`` is no row epoch
    """
    if not 0 < floor < 1 << 63:
        raise ValueError(f"a cleanup delete's floor is a row epoch, a positive BIGINT: got {floor}")
    token = _CLEANUP_FLOOR.set(floor)
    try:
        yield
    finally:
        _CLEANUP_FLOOR.reset(token)


def cleanup_delete_floor() -> int | None:
    """the floor of the quiet-time cleanup delete being sent in this context (:func:`cleanup_deleting`); None outside
    one. The keyed deletes ask it before refusing a delete of an append-only table
    (``threetears.core.keyed._filters.refuse_in_place``): cleanup's is the one delete such a table takes.

    :return: the floor, or None
    :rtype: int | None
    """
    return _CLEANUP_FLOOR.get()


class CleanupMixedWriteError(ValueError):
    """a transaction mixed quiet-time cleanup's delete with another write (:func:`cleanup_mix_refusal`)."""


def cleanup_mix_refusal(*, cleanup: bool, writes: bool, cleaned: bool, wrote: bool) -> str | None:
    """the one rule by which a commit that carries quiet-time cleanup's delete is ONLY cleanup: a transaction that sent
    a request marked cleanup's (:func:`cleanup_deleting`) makes no write not so marked, and one that made such a write
    sends no marked request. ``CallerTransaction`` holds a pod's transactions to it, the hub's broker its sessions.

    WHY: every record of a commit that carried a cleanup delete names its floor F (``EpochChange.cleanup_floor``), and
    a copy applies such a record by marking it applied, reading nothing (``CollectionRegistry.catch_up_generation``):
    every row cleanup removes is stamped below F, so no reader loses anything. A write beside it would be skipped too,
    and every copy would keep serving what L3 no longer holds.

    :param cleanup: the request now sent is marked cleanup's
    :ptype cleanup: bool
    :param writes: the request now sent writes a table
    :ptype writes: bool
    :param cleaned: the transaction already sent a marked request
    :ptype cleaned: bool
    :param wrote: the transaction already made a write not so marked
    :ptype wrote: bool
    :return: the refusal's message, or None when the request may be sent
    :rtype: str | None
    """
    if (cleanup and wrote) or (writes and not cleanup and cleaned):
        return (
            "a transaction that carries quiet-time cleanup's delete (cleanup_deleting) makes no other write: its "
            "commit is recorded as cleanup's, which every copy skips, reading nothing. Send cleanup's deletes in a "
            "transaction of their own, and every other write in another"
        )
    return None


class _CleanupWriteNotDeleteError(ValueError):
    """a request marked quiet-time cleanup's writes anything but a delete (:func:`cleanup_write_refusal`)."""


#: the one row-changing verb a request marked cleanup's may carry
_CLEANUP_VERBS: Final = frozenset({"delete"})


def cleanup_write_refusal(*, cleanup: bool, verbs: Iterable[str]) -> str | None:
    """the rule by which a request marked quiet-time cleanup's (:func:`cleanup_deleting`) only deletes: any other
    row-changing verb it carries (an insert, an update, an upsert, a copy's rows, a versioned table's run or reset
    markers) is refused before it is sent, by the pod (the L3 rail, :func:`refuse_cleanup_non_delete`) and by the hub's
    broker alike, on every door. A marked read changes nothing and passes.

    WHY: the records of a marked request's commit name cleanup's floor F, and every copy marks such a record applied,
    reading nothing (``CollectionRegistry.catch_up_generation``): right for a delete of rows below F, which no reader
    holds; a marked insert or update would be skipped by every copy, which would keep serving what L3 no longer holds.

    :param cleanup: the request is marked cleanup's
    :ptype cleanup: bool
    :param verbs: the row-changing verbs the request carries (``insert``, ``update``, ``delete``, ``upsert``); none for
        a read
    :ptype verbs: Iterable[str]
    :return: the refusal's message, or None when the request may be sent
    :rtype: str | None
    """
    others = sorted(set(verbs) - _CLEANUP_VERBS) if cleanup else []
    if not others:
        return None
    return (
        f"a request marked quiet-time cleanup's (cleanup_deleting) only deletes; this one also writes by "
        f"{', '.join(others)}: its commit would be recorded as cleanup's, which every copy skips, reading nothing"
    )


def refuse_cleanup_non_delete(verbs: Iterable[str]) -> None:
    """refuse, before it is sent, a request made in this context that is marked cleanup's (:func:`cleanup_deleting`)
    and writes by anything but a delete (:func:`cleanup_write_refusal`); nothing outside such a block.

    :param verbs: the row-changing verbs the request carries; none for a read
    :ptype verbs: Iterable[str]
    :return: nothing
    :rtype: None
    :raises _CleanupWriteNotDeleteError: when the marked request writes otherwise
    """
    refusal = cleanup_write_refusal(cleanup=_CLEANUP_FLOOR.get() is not None, verbs=verbs)
    if refusal is not None:
        raise _CleanupWriteNotDeleteError(refusal)


#: each connection a transaction is open on in this context that is told of every request sent on it, before it is
#: sent (:func:`holding_sent_requests`), outermost first. A context variable, as ``_TOUCH_SINKS`` is
_REQUEST_SINKS: ContextVar[tuple[tuple[Any, Callable[[bool, bool], None]], ...]] = ContextVar(
    "threetears_sent_request_sinks", default=()
)


def holding_sent_requests(conn: Any, sink: Callable[[bool, bool], None]) -> Token[Any]:
    """name ``sink`` as what is told of every statement or write request sent on ``conn`` in this context, before it is
    sent, with whether it is marked cleanup's and whether it writes, until :func:`releasing_sent_requests`. The transaction that will COMMIT them calls it
    as it opens (``CallerTransaction``), and refuses a request that breaks :func:`cleanup_mix_refusal` by raising.

    :param conn: the connection the transaction is open on
    :ptype conn: Any
    :param sink: told ``(cleanup, writes)`` for each request, on the event loop, before it is sent; raises to refuse it
    :ptype sink: Callable[[bool, bool], None]
    :return: the token to release it with
    :rtype: Token
    """
    return _REQUEST_SINKS.set((*_REQUEST_SINKS.get(), (conn, sink)))


def releasing_sent_requests(token: Token[Any]) -> None:
    """stop telling the sink :func:`holding_sent_requests` named, as its transaction ends.

    :param token: what :func:`holding_sent_requests` answered
    :ptype token: Token
    :return: nothing
    :rtype: None
    """
    _REQUEST_SINKS.reset(token)


def note_sent_request(conn: Any, *, cleanup: bool, writes: bool) -> None:
    """tell the transaction open on ``conn`` in this context (:func:`holding_sent_requests`) of a request about to be
    sent on it; nothing when none is. The L3 rail's transaction connection calls it for every statement, copy and
    key-prefix delete it sends.

    :param conn: the connection the request is sent on
    :ptype conn: Any
    :param cleanup: whether the request is marked quiet-time cleanup's (:data:`CLEANUP_DELETE_FIELD`)
    :ptype cleanup: bool
    :param writes: whether the request writes a table
    :ptype writes: bool
    :return: nothing
    :rtype: None
    :raises CleanupMixedWriteError: as the transaction's sink refuses (:func:`cleanup_mix_refusal`)
    """
    sink = next((taker for held, taker in _REQUEST_SINKS.get() if held is conn), None)
    if sink is not None:
        sink(cleanup, writes)


def stamp_fields() -> dict[str, Any]:
    """the fields every L3 request carries to say whether its write stamps row epochs, as the enclosing
    :func:`stamping_write` named it: ``{stamps_epoch: false}``, or ``{stamps_epoch: true, s_min: <lowest>}``; and,
    inside :func:`cleanup_deleting`, that it is quiet-time cleanup's delete with its floor (``{cleanup_delete: true,
    cleanup_floor: F}``). A request measured before it is sent (a bulk write's batches) is measured inside the same
    block, so it is sized exactly.

    :return: the fields, a fresh dict the caller merges into its payload
    :rtype: dict[str, Any]
    """
    s_min = _STAMPED_WRITE.get()
    fields: dict[str, Any] = (
        {STAMPS_EPOCH_FIELD: False} if s_min is None else {STAMPS_EPOCH_FIELD: True, STAMP_S_MIN_FIELD: s_min}
    )
    floor = _CLEANUP_FLOOR.get()
    if floor is not None:
        fields[CLEANUP_DELETE_FIELD] = True
        fields[CLEANUP_FLOOR_FIELD] = floor
    return fields


@dataclass(frozen=True, slots=True)
class StampedScopes:
    """what one stamped write of a versioned table's rows touch: the scope epochs scopes they name and their stamps.

    Read off the rows by the store, never named by the writer (``ddl.row_epoch``'s "Every scope a stamped write's rows
    name is touched"), and handed to the transaction the write joins (:func:`stamped_send`), which raises each scope's
    ``last_epoch`` to its highest stamp just before its COMMIT.

    :ivar epochs: the scope epochs table the scopes are rows of (``RowEpoch.epochs``), as the connection names it
    :ivar scopes: the scopes, ``scope_prefix + value``, distinct and sorted
    :ivar s_min: the rows' lowest stamp
    :ivar s_max: the rows' highest stamp
    """

    epochs: str
    scopes: tuple[str, ...]
    s_min: int
    s_max: int


#: each connection a transaction is open on in this context that a stamped write's scopes are handed to, with what takes
#: them (:func:`holding_touched_scopes`), outermost first. A context variable, as ``CallerTransaction``'s own register of
#: open transactions is: a task started inside the transaction still finds it, one started outside never does
_TOUCH_SINKS: ContextVar[tuple[tuple[Any, Callable[[StampedScopes], None]], ...]] = ContextVar(
    "threetears_touched_scope_sinks", default=()
)


def holding_touched_scopes(conn: Any, sink: Callable[[StampedScopes], None]) -> Token[Any]:
    """name ``sink`` as what takes the scopes of every stamped write made on ``conn`` in this context, until
    :func:`releasing_touched_scopes`. The transaction that will COMMIT them calls it as it opens (``CallerTransaction``).

    :param conn: the connection the transaction is open on
    :ptype conn: Any
    :param sink: what takes each write's scopes, on the event loop, before the write is sent
    :ptype sink: Callable[[StampedScopes], None]
    :return: the token to release it with
    :rtype: Token
    """
    return _TOUCH_SINKS.set((*_TOUCH_SINKS.get(), (conn, sink)))


def releasing_touched_scopes(token: Token[Any]) -> None:
    """stop handing scopes to the sink :func:`holding_touched_scopes` named, as its transaction ends.

    :param token: what :func:`holding_touched_scopes` answered
    :ptype token: Token
    :return: nothing
    :rtype: None
    """
    _TOUCH_SINKS.reset(token)


@contextmanager
def stamped_send(
    conn: Any, s_min: int | None, touched: StampedScopes | None = None, *, table: str | None = None
) -> Iterator[None]:
    """send a stamped write's requests (:func:`stamping_write`), its scopes first handed to the transaction open on
    ``conn`` that commits it: the one path every write the store makes of a table that declares a row epoch goes through,
    so no writer can leave a scope its rows name untouched (``ddl.row_epoch``).

    :param conn: the connection the write is sent on; None for the store's own pool
    :ptype conn: Any
    :param s_min: the rows' lowest stamp; None for a write that stamps none
    :ptype s_min: int | None
    :param touched: a versioned table's write's scopes and stamps, read off its rows; None for any other write
    :ptype touched: StampedScopes | None
    :param table: the table the requests write, its statement stamp noted when the block raised nothing
        (:func:`stamping_write`'s ``table``); None for a block that sends nothing
    :ptype table: str | None
    :return: a context in which the requests say they stamp
    :rtype: Iterator[None]
    :raises UntouchedScopesError: when the write names scopes and no transaction open on ``conn`` takes them: nothing
        would touch them before its COMMIT, and a restore would never hear its rows
    """
    if touched is not None and touched.scopes:
        sink = next((taker for held, taker in _TOUCH_SINKS.get() if held is conn), None) if conn is not None else None
        if sink is None:
            raise UntouchedScopesError(
                f"a stamped write of {len(touched.scopes)} scopes of {touched.epochs} joins no transaction to touch "
                f"them in before its COMMIT (a restore finds changed scopes by last_epoch alone): write it on a "
                f"CallerTransaction's connection"
            )
        sink(touched)
    with stamping_write(s_min, table=table):
        yield


#: where the advances made while a commit settles in this context note the token each wrote, by table
#: (:func:`collecting_advanced_generations`); None when nothing collects them
_ADVANCED: ContextVar[dict[str, str] | None] = ContextVar("threetears_commit_advanced_generations", default=None)


@contextmanager
def collecting_advanced_generations(into: dict[str, str]) -> Iterator[None]:
    """collect into ``into`` the token each advance made inside this block wrote, by table
    (:func:`note_advanced_generation`): how a process that advances its own generations (a direct pool) learns what
    its commit's settling wrote, as a pod learns it from the broker's reply (:func:`reply_generations_since`). A task
    started inside the block (``asyncio.shield`` around a settle) shares ``into``: the context copies the reference.

    :param into: table -> the token its advance wrote, filled as advances are made
    :ptype into: dict[str, str]
    :return: a context in which advances are collected
    :rtype: Iterator[None]
    """
    token = _ADVANCED.set(into)
    try:
        yield
    finally:
        _ADVANCED.reset(token)


def note_advanced_generation(table_name: str, token: str) -> None:
    """note the token an advance of ``table_name`` wrote, where this context collects them; nothing otherwise.

    :param table_name: the table advanced
    :ptype table_name: str
    :param token: the token the advance wrote
    :ptype token: str
    :return: nothing
    :rtype: None
    """
    into = _ADVANCED.get()
    if into is not None:
        into[table_name] = token


#: the rows the writes in flight in this context change, as ``(table, row)`` pairs, innermost last
_NAMED_ROWS: ContextVar[tuple[tuple[str, Mapping[str, Any]], ...]] = ContextVar(
    "threetears_named_changed_rows", default=()
)


@contextmanager
def naming_changed_rows(table_name: str, rows: Sequence[Mapping[str, Any]]) -> Iterator[None]:
    """name the rows the L3 writes made inside this block change, for the broker's epoch record.

    :param table_name: the table written
    :ptype table_name: str
    :param rows: each changed row as ``ChangedRow.model_dump`` gives it
    :ptype rows: Sequence[Mapping[str, Any]]
    :return: a context in which every write request names the rows
    :rtype: Iterator[None]
    """
    token = _NAMED_ROWS.set(_NAMED_ROWS.get() + tuple((table_name, row) for row in rows))
    try:
        yield
    finally:
        _NAMED_ROWS.reset(token)


def named_changed_rows() -> dict[str, list[Mapping[str, Any]]]:
    """the rows the writes in flight in this context change, by table, as :data:`CHANGED_ROWS_FIELD` carries them.

    :return: each table's rows, each once; empty when no write in flight named any
    :rtype: dict[str, list[Mapping[str, Any]]]
    """
    by_table: dict[str, dict[tuple[str, ...], Mapping[str, Any]]] = {}
    for table, row in _NAMED_ROWS.get():
        by_table.setdefault(table, {})[tuple(str(each) for each in row["ids"])] = row
    return {table: list(rows.values()) for table, rows in by_table.items()}


#: the reply field naming, for each switched-on table a commit wrote, the generation token the
#: broker's advance after that commit wrote. The hub imports this name.
GENERATIONS_REPLY_FIELD: Final = "generations"

#: the reply field listing each switched-on table the commit wrote whose generation the broker could
#: not advance. The hub imports this name.
GENERATIONS_FAILED_REPLY_FIELD: Final = "generations_failed"

#: the broker's ``error_code`` on a reply whose statements COMMITTED and whose generation advance
#: failed for at least one table. The reply is still ``success``: the write landed and must not be
#: sent again. The hub imports this name.
GENERATION_UNAVAILABLE_ERROR_CODE: Final = "GENERATION_UNAVAILABLE"


@dataclass(slots=True)
class _Committed:
    """the generations the broker named for one task's commits, until a collection takes them.

    :ivar tokens: table to the token the broker wrote for it
    :ivar failed: table to why the broker could not advance it
    :ivar not_committed: why the task's last commit landed nothing (rolled back, refused, or its
        outcome unknown); ``None`` otherwise
    :ivar taken: the tables whose token an advance was already handed
    :ivar names_generations: whether the reply carried either generations field, which only a broker
        that advances sends; without them the broker advanced nothing for the commit
    """

    tokens: Mapping[str, str] = EMPTY
    failed: Mapping[str, str] = EMPTY
    not_committed: str | None = None
    taken: frozenset[str] = frozenset()
    names_generations: bool = False


#: tables a broker that advances nothing has been reported for, so the warning is logged once each (BuildOnce: two
#: threads that both see the table unreported never log it twice)
_WARNED_UNADVANCED: BuildOnce[str, bool] = BuildOnce()


def _warn_unadvanced(table_name: str) -> bool:
    """log, once per table, that the broker advanced no write generation for it.

    :param table_name: the table
    :ptype table_name: str
    :return: True, the mark that it was logged
    :rtype: bool
    """
    _logger.warning(
        "the L3 broker advanced no write generation for a switched-on table: its hub is older "
        "than write generations or does not hold the table's class switched on yet; this pod's "
        "rows of it name no generation until the hub does",
        extra={"extra_data": {"table": table_name}},
    )
    return True


_COMMITTED: ContextVar[_Committed | None] = ContextVar("threetears_broker_committed_generations", default=None)


def record_reply_generations(response: Mapping[str, Any], *, ends_write: bool) -> None:
    """make the generations a reply names this task's record of its last commit.

    Called by the L3 proxy on every successful ``l3.query`` and ``l3.tx.commit`` reply and every
    ``l3.batch`` reply, a statement-by-statement batch that failed partway included. A reply that
    ends a write, or names a generation, replaces the task's record: what an earlier commit named is
    not this one's. A read's reply that names none leaves it. A table both named and listed failed
    is failed.

    :param response: the parsed reply
    :ptype response: Mapping[str, Any]
    :param ends_write: whether the request was a write's (a statement that writes, a batch, a
        transaction's commit); its reply ends the scope of every earlier commit's tokens
    :ptype ends_write: bool
    :return: nothing
    :rtype: None
    """
    tokens = response.get(GENERATIONS_REPLY_FIELD)
    failed = response.get(GENERATIONS_FAILED_REPLY_FIELD)
    named = {
        table: token
        for table, token in (tokens.items() if isinstance(tokens, Mapping) else ())
        if isinstance(table, str) and isinstance(token, str) and token
    }
    unadvanced = [table for table in (failed if isinstance(failed, list) else ()) if isinstance(table, str)]
    if not named and not unadvanced and not ends_write:
        return
    # built whole before the record is set: a task sharing this context never sees it part-filled
    reasons: dict[str, str] = {}
    if unadvanced:
        reason = str(response.get("error_message") or "the broker could not advance the generation")
        _logger.error(
            "the L3 broker committed this write and could not advance the write generation of a table it "
            "wrote; a pod following the table cannot tell from it that this write happened",
            extra={"extra_data": {"tables": sorted(unadvanced), "error_code": response.get("error_code")}},
        )
        reasons = {
            table: (
                f"the L3 broker committed the write to {table!r} and could not advance its write generation: {reason}"
            )
            for table in unadvanced
        }
    _COMMITTED.set(
        _Committed(
            tokens=frozen((table, token) for table, token in named.items() if table not in reasons),
            failed=frozen(reasons),
            names_generations=GENERATIONS_REPLY_FIELD in response or GENERATIONS_FAILED_REPLY_FIELD in response,
        )
    )


def forget_reply_generations(reason: str) -> None:
    """drop every generation this task holds, because its last commit landed nothing.

    Called when a transaction is rolled back, its commit is refused, or the commit's reply never
    arrives: a collection settling it must not be handed a token from an earlier commit, and is told
    why there is none (:class:`~threetears.core.exceptions.GenerationNotCommittedError`).

    :param reason: what happened to the commit, for the error a later advance raises
    :ptype reason: str
    :return: nothing
    :rtype: None
    """
    _COMMITTED.set(_Committed(not_committed=reason))


def committed_generations_record() -> object | None:
    """this task's record of its last commit's generations, as it stands: taken just before a commit, it tells
    :func:`reply_generations_since` whether the commit's reply replaced it.

    :return: the record, opaque; None when this task holds none
    :rtype: object | None
    """
    return _COMMITTED.get()


def reply_generations_since(before: object | None) -> dict[str, str]:
    """the tokens the broker's reply to this task's commit named, by table, when a reply replaced the record since
    ``before`` (:func:`committed_generations_record`, taken just before the commit); empty when none did (the commit
    was not the broker's: a direct pool's) or it landed nothing. Every token, whether an advance took it yet or not.

    :param before: the record as it stood before the commit
    :ptype before: object | None
    :return: table -> the token the broker's advance after the commit wrote
    :rtype: dict[str, str]
    """
    held = _COMMITTED.get()
    if held is None or held is before or held.not_committed is not None:
        return {}
    return dict(held.tokens)


def _take_committed_generation(table_name: str) -> str | None:
    """the token the broker wrote for ``table_name`` after the commit just made, handed out once.

    A second advance of the table for the same commit would publish its rows under a count the
    first advance's broadcasts already gave, and a follower would count the advance complete before
    hearing them; it raises instead.

    :param table_name: the table a collection is advancing
    :ptype table_name: str
    :return: the generation token, or ``None`` when the commit's reply carried no generations field
        at all: the broker advanced nothing (it is older than generations, or its hub does not hold
        the table's class switched on yet)
    :rtype: str | None
    :raises GenerationNotCommittedError: when the task's last commit landed nothing
    :raises GenerationUnavailableError: when the broker reported it could not advance the table, or
        named generations and none for it, or nothing this task committed wrote anything; or when
        the commit's token was already handed to an advance
    """
    held = _COMMITTED.get()
    if held is not None:
        token = held.tokens.get(table_name)
        if token is not None:
            if table_name in held.taken:
                raise GenerationUnavailableError(
                    f"the write generation of {table_name!r} for the commit just made was already advanced; "
                    f"settle every row of one commit of a table in one advance (invalidate_cache_many), so "
                    f"its broadcasts carry one count"
                )
            held.taken = held.taken | {table_name}
            return token
        failure = held.failed.get(table_name)
        if failure is not None:
            raise GenerationUnavailableError(failure)
        if held.not_committed is not None:
            raise GenerationNotCommittedError(
                f"no write generation for {table_name!r}: {held.not_committed}, so nothing it wrote landed"
            )
        if not held.names_generations:
            _WARNED_UNADVANCED.get(table_name, lambda: _warn_unadvanced(table_name))
            return None
    raise GenerationUnavailableError(
        f"the L3 broker named no write generation for {table_name!r} for the commit just made: it named "
        f"generations for other tables, or nothing committed here wrote anything"
    )


@runtime_checkable
class GenerationReader(Protocol):
    """reads a table's generation without writing it: what a pod's grant on the epoch bucket allows.

    :class:`threetears.epoch.EpochGenerationReader` is the platform's.
    """

    async def read(self, table_name: str) -> str | None:
        """the table's current generation, or ``None`` when the store holds none for it.

        :param table_name: the table
        :ptype table_name: str
        :return: the token, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation cannot be read
        """
        ...


class BrokerGenerationSource:
    """a pod's generation source: the broker advances, the pod is told what it wrote.

    It reads a generation only with a reader (:attr:`reads_generations`); without one, absence
    caching stays off on its registry, as it was with no source at all.

    :meth:`advance` makes no request. The broker advanced the table after committing the pod's
    write and named the token in its reply; this hands that token to the collection advancing, once
    per commit.
    :meth:`current` reads the epoch bucket through ``reader``, which a pod is granted; with no
    reader, or for a table that has no generation yet, it raises, so a collection that caches
    absences trusts none rather than one no advance can invalidate.
    """

    def __init__(self, reader: GenerationReader | None = None) -> None:
        """capture the reader; no I/O.

        :param reader: reads a table's generation from the epoch bucket, for :meth:`current`;
            ``None`` when the pod reads none, so :meth:`current` always raises
        :ptype reader: GenerationReader | None
        :return: nothing
        :rtype: None
        """
        self._reader = reader

    @property
    def reads_generations(self) -> bool:
        """whether :meth:`current` can answer: only with a reader.

        :return: ``True`` when a reader was given
        :rtype: bool
        """
        return self._reader is not None

    async def current(self, table_name: str) -> str:
        """the table's current generation, as the epoch bucket holds it.

        A pod cannot mint a table's first generation, so a table with none raises rather than
        answering a token an absence could be stamped with and a later wipe could bring back.

        :param table_name: the table
        :ptype table_name: str
        :return: the token
        :rtype: str
        :raises GenerationUnavailableError: when there is no reader, the bucket cannot be read, or
            it holds no generation for the table yet
        """
        if self._reader is None:
            raise GenerationUnavailableError(
                f"this pod reads no write generation for {table_name!r}: its generation source has no reader"
            )
        token = await self._reader.read(table_name)
        if token is None:
            raise GenerationUnavailableError(
                f"no write generation exists yet for {table_name!r}, and a pod cannot mint one"
            )
        return token

    async def advance(self, table_name: str) -> str | None:
        """the token the broker wrote for ``table_name`` after the commit this task just made.

        :param table_name: the table
        :ptype table_name: str
        :return: the generation token the broker's advance wrote, or ``None`` when the commit's reply
            carried no generations field at all, because the broker advanced nothing
        :rtype: str | None
        :raises GenerationNotCommittedError: when the task's last commit landed nothing
        :raises GenerationUnavailableError: when the broker reported it could not advance the
            table, or named generations and none for it
        """
        return _take_committed_generation(table_name)
