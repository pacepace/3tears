"""query a hub-served datasource over NATS, on the caller's own identity.

**Why this exists.** A process that holds no warehouse credential -- a tool pod
above all -- reaches a datasource the way it reaches L3: it publishes a request
on a subject the hub answers, carrying the hub-minted identity token it obtained
at handshake, and the hub decides. The hub verifies the token at the door, pins
the principal from the SIGNED ``sub``, evaluates that principal's grant on the
datasource's namespace, runs the query on the driver it already holds, and
replies with rows or a typed refusal. Nothing on this wire names a principal, so
a caller can be authorized only as whoever the hub minted the token for.

**Identity is a PROVIDER, never a value.** The token is short-lived and re-minted
in place by the holder's refresh loop, so a string captured at construction is
expired within the hour and every query after that is refused by a client that
looks correctly wired. The provider is read on EVERY call, exactly as
:class:`threetears.core.backends.nats_proxy.NatsProxyL3Backend` reads its own.
An empty or missing token is refused HERE, before the bus sees the request,
because the hub refuses it identically and a refusal that names the wiring gap
beats one that reads as a permission problem.

**One wire model, both ends.** The hub's responder imports these same classes,
so the two sides cannot drift apart the way a hand-copied mirror does. The
request forbids unknown fields for the reason the L3 request models do: a
stale client sending ``agent_id`` beside a valid token must be refused at the
border, not silently authorized as the token's principal.

**Errors carry the hub's code.** A caller branches on
:attr:`DatasourceQueryError.error_code` and never on the message, which is for
a human and may be reworded at any time. ``ACCESS_DENIED`` is the one a consumer
most needs to tell apart from a transport fault: the first is a grant to fix,
the second is a bus to look at.

**No markdown, no honesty imperatives.** The datasource TOOL renders rows for a
model and annotates what is missing so the model cannot invent it. This client
serves a program, which gets the rows and the one fact about them it cannot
recover on its own: whether the hub CUT the result at its row cap. A program
that wants the model-facing rendering calls the tool.

**Truncation is returned, never swallowed.** The hub caps a result and says so
on the wire; a client that handed back the rows alone would turn a cap into a
silent prefix, and a consumer that derives state from a full read -- prune what
the upstream no longer has, say -- would rewrite its state from a lie. So
:meth:`DatasourceQueryClient.query` returns a :class:`DatasourceQueryResult`
carrying ``truncated`` beside ``rows``, and the caller decides: a preview takes
the prefix, a derivation refuses it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer
from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = [
    "DEFAULT_QUERY_TIMEOUT_SECONDS",
    "QUERY_STATEMENT_TIMEOUT_SECONDS",
    "DatasourceQueryClient",
    "DatasourceQueryError",
    "DatasourceQueryRequest",
    "DatasourceQueryResponse",
    "DatasourceQueryResult",
    "IncompleteReadError",
    "read_all",
]

log = get_logger(__name__)


#: how long one query may take, end to end, before the client gives up.
#:
#: a plain request/reply on the caller's own connection, and that connection is
#: rebuilt on every NATS re-authentication: a reply that lands after the
#: connection that asked is gone lands nowhere. the platform's re-auth cadence
#: is tuned so a request of this length completes on one connection; a longer
#: budget would need the durable result rail the tool path uses, which this
#: wire deliberately does not carry.
DEFAULT_QUERY_TIMEOUT_SECONDS: float = 120.0

#: how long the hub lets one statement run on the warehouse before it cancels.
#:
#: BELOW the client deadline by a margin, and that ordering is the whole point:
#: a stuck warehouse comes back as the hub's ``QUERY_TIMEOUT`` refusal, which
#: names the cause, rather than as the client's ``REQUEST_FAILED``, which cannot
#: tell a slow warehouse from a dead bus and steers a retry that stacks a second
#: statement on the first. the margin is what the cancel, the refusal envelope
#: and the reply need to travel while the caller is still listening. lives here
#: beside the client deadline so the two cannot drift apart; the hub's responder
#: imports it rather than choosing its own.
QUERY_STATEMENT_TIMEOUT_SECONDS: int = 100


class DatasourceQueryRequest(BaseModel):
    """one datasource query, carrying the caller's identity as a forwarded token.

    extras are FORBIDDEN, and that is a security property rather than a style
    choice. ``agent_id`` and ``user_id`` are deliberately absent: a request that
    carried them would be one whose ACL check a caller could steer. Pydantic's
    default of ignoring an unknown field would let a stale client send one,
    have it dropped, and be authorized as the token's principal with no error
    anywhere -- broader authority than it asked for, silently. Refusing turns
    that into a border failure the caller sees.

    :param correlation_id: trace id echoed back on the reply and bound into the
        hub's logs, so one query can be followed from the caller to the driver
    :ptype correlation_id: UUID
    identity is carried as :class:`~pydantic.SecretStr`, not plain ``str``. the
    two fields are bearer tokens: a plain-``str`` field would surface the token
    in a ``repr``, a log line, or a pydantic ``ValidationError`` (which echoes
    the offending input), which is the leak the platform's "log the expiry,
    never the token" rule exists to prevent. ``SecretStr`` redacts all of those
    to ``'**********'``. The wire is unaffected: a JSON field serializer emits
    the real token, because :meth:`threetears.nats.NatsClient.request` sends the
    request via ``model_dump_json()`` and the hub reads the token off the
    signed claims, so the value MUST cross verbatim.

    :param correlation_id: trace id echoed back on the reply and bound into the
        hub's logs, so one query can be followed from the caller to the driver
    :ptype correlation_id: UUID
    :param identity_token: the caller's hub-minted identity token, forwarded
        verbatim; the hub verifies it and takes the principal off the signed
        claims
    :ptype identity_token: SecretStr
    :param user_identity_token: the per-turn hub-minted user assertion when a
        human is in the loop; ``None`` for a call with nobody's behalf to act
        on, in which case the hub evaluates the principal's own grants alone
    :ptype user_identity_token: SecretStr | None
    :param query: the sql to run; the hub admits SELECT and nothing else
    :ptype query: str
    :param params: positional query parameters, JSON-native values only
    :ptype params: list[Any]
    """

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    identity_token: SecretStr
    user_identity_token: SecretStr | None = None
    query: str
    params: list[Any] = Field(default_factory=list)

    @field_serializer("identity_token", "user_identity_token", when_used="json")
    def _emit_token_on_the_wire(self, value: SecretStr | None) -> str | None:
        """emit the real token for JSON, so the hub receives it verbatim.

        ``SecretStr`` would otherwise serialize to ``'**********'``, which would
        travel the bus and fail every authorization as if the grant were wrong.
        this runs ONLY for JSON (``when_used="json"``), so ``repr`` and a python
        ``model_dump`` keep redacting; only the wire sees the secret.

        :param value: the held token, or ``None`` for an absent user assertion
        :ptype value: SecretStr | None
        :return: the plaintext token, or ``None``
        :rtype: str | None
        """
        return value.get_secret_value() if value is not None else None


class DatasourceQueryResponse(BaseModel):
    """the hub's answer: rows, or a refusal with a code a caller branches on.

    :param success: whether the query ran
    :ptype success: bool
    :param rows: result rows, one dict per row, empty on refusal. a ``bytes``
        column arrives hex-encoded, the hub's convention for every JSON wire
        it serves rows on; decode with ``bytes.fromhex``
    :ptype rows: list[dict[str, Any]]
    :param row_count: how many rows ``rows`` carries
    :ptype row_count: int
    :param truncated: whether the hub cut the result at its row cap, so
        ``rows`` is a prefix of what the query produced
    :ptype truncated: bool
    :param error_code: the machine-readable refusal code, ``None`` on success
    :ptype error_code: str | None
    :param error_message: the human-readable refusal, never parsed
    :ptype error_message: str | None
    :param correlation_id: the request's id, echoed; ``None`` when the hub
        could not parse the request and so never saw one
    :ptype correlation_id: UUID | None
    """

    success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None
    correlation_id: UUID | None = None


class DatasourceQueryResult(BaseModel):
    """what a successful query hands the caller: the rows, and whether they are all of them.

    Distinct from :class:`DatasourceQueryResponse`, which is the WIRE envelope
    and carries the refusal fields too; by the time a caller holds this, a
    refusal has already become :class:`DatasourceQueryError`, so the fields
    left are the ones a program acts on.

    :param rows: result rows, one dict per row, in the hub's order. a ``bytes``
        column arrives hex-encoded; decode with ``bytes.fromhex``
    :ptype rows: list[dict[str, Any]]
    :param row_count: how many rows ``rows`` carries
    :ptype row_count: int
    :param truncated: whether the hub cut the result at its row cap, so
        ``rows`` is a prefix of what the query produced. a caller deriving
        state from a full read must refuse a truncated one rather than treat
        the missing rows as absent upstream
    :ptype truncated: bool
    :param correlation_id: the request's id, echoed by the hub
    :ptype correlation_id: UUID
    """

    model_config = ConfigDict(frozen=True)

    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    correlation_id: UUID


class DatasourceQueryError(RuntimeError):
    """a query did not return rows, and this is why.

    one exception type for every non-row outcome -- the hub's refusal, a
    missing identity, a transport fault -- so a caller writes one ``except`` and
    branches on :attr:`error_code`. the hub's own codes ride through unchanged;
    the two the client mints itself are ``NO_IDENTITY_TOKEN`` (refused before
    any publish) and ``REQUEST_FAILED`` (the bus, not the hub, said no).

    :param error_code: the machine-readable code
    :ptype error_code: str
    :param message: the human-readable explanation
    :ptype message: str
    """

    def __init__(self, error_code: str, message: str) -> None:
        """bind the code beside the message.

        :param error_code: the machine-readable code
        :ptype error_code: str
        :param message: the human-readable explanation
        :ptype message: str
        """
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code


class DatasourceQueryClient:
    """publishes datasource queries on the hub's subject and returns the rows.

    built once per process over the connected canonical NATS client and the
    process's identity token PROVIDER, then called per query. the subject is
    built through :meth:`threetears.nats.Subjects.datasource_query`, which reads
    the namespace the NATS client bound at connect, so the client and the hub's
    responder derive the identical subject from the identical source.

    :param nats_client: the connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param identity_token: zero-arg provider returning this process's CURRENT
        hub-minted identity token, read on every call; the holder's bound
        ``get``, never a captured string
    :ptype identity_token: Callable[[], str | None]
    :param timeout: how long one query may take before the client gives up
    :ptype timeout: timedelta | float
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        identity_token: Callable[[], str | None],
        timeout: timedelta | float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        """bind the transport, the identity provider and the deadline.

        :param nats_client: the connected canonical NATS wrapper client
        :ptype nats_client: NatsClient
        :param identity_token: zero-arg provider returning the current token
        :ptype identity_token: Callable[[], str | None]
        :param timeout: per-query deadline; a number is seconds
        :ptype timeout: timedelta | float
        """
        self._nats_client = nats_client
        self._identity_token = identity_token
        self._timeout = timeout if isinstance(timeout, timedelta) else timedelta(seconds=float(timeout))

    def forwarded_identity_token(self) -> str:
        """the CURRENT hub-minted identity token to forward on a query.

        read through the provider on every call, never cached: the token is
        short-lived and re-minted by the refresh loop, so a value captured once
        is expired within the hour.

        :return: the caller's current identity token
        :rtype: str
        :raises DatasourceQueryError: with code ``NO_IDENTITY_TOKEN`` when the
            provider returns nothing; the hub refuses an empty token exactly as
            a missing one, so failing here names the wiring gap instead of
            sending a request that cannot be authorized
        """
        token = self._identity_token()
        if not token:
            raise DatasourceQueryError(
                "NO_IDENTITY_TOKEN",
                "the identity_token provider returned no token. The handshake has not "
                "completed or its result was not threaded through; the hub refuses an "
                "empty token exactly as a missing one, so the request is not sent.",
            )
        return token

    @traced
    async def query(
        self,
        datasource_name: str,
        query: str,
        params: list[Any] | None = None,
        *,
        user_identity_token: str | None = None,
        correlation_id: UUID | None = None,
    ) -> DatasourceQueryResult:
        """run one read query against a datasource and return its rows, flagged if cut.

        :param datasource_name: the datasource's name as the hub's ``datasources``
            table holds it
        :ptype datasource_name: str
        :param query: the sql to run; the hub admits SELECT and nothing else
        :ptype query: str
        :param params: positional query parameters, JSON-native values only
        :ptype params: list[Any] | None
        :param user_identity_token: the per-turn user assertion when a human is
            in the loop; ``None`` evaluates the principal's own grants alone
        :ptype user_identity_token: str | None
        :param correlation_id: trace id for this query; minted when omitted
        :ptype correlation_id: UUID | None
        :return: the rows, with ``truncated`` set when the hub cut them at its
            row cap; a caller that needs every row must check it
        :rtype: DatasourceQueryResult
        :raises DatasourceQueryError: when the datasource name is empty, when
            no identity token is available, when the hub refuses (its code
            rides on the exception), or when the bus fails to deliver a
            decodable reply
        """
        if not datasource_name:
            raise DatasourceQueryError(
                "INVALID_DATASOURCE_NAME",
                "a datasource query needs the datasource's name; an empty name composes no subject",
            )
        request = DatasourceQueryRequest(
            correlation_id=correlation_id if correlation_id is not None else uuid7(),
            # wrap at the border: the field is SecretStr so the token cannot leak through a repr or
            # a validation error; the JSON serializer re-emits the real value for the wire.
            identity_token=SecretStr(self.forwarded_identity_token()),
            user_identity_token=SecretStr(user_identity_token) if user_identity_token is not None else None,
            query=query,
            params=list(params) if params is not None else [],
        )
        subject = Subjects.datasource_query(datasource_name)
        log.info(
            "datasource query sent",
            extra={
                "extra_data": {
                    "datasource": datasource_name,
                    "correlation_id": f"{request.correlation_id}",
                    "user_in_loop": user_identity_token is not None,
                }
            },
        )
        try:
            response: DatasourceQueryResponse = await self._nats_client.request(
                subject=subject,
                message=request,
                response_type=DatasourceQueryResponse,
                timeout=self._timeout,
            )
        except RequestError as exc:
            log.warning(
                "datasource query did not complete",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "correlation_id": f"{request.correlation_id}",
                        "error": str(exc),
                    }
                },
            )
            raise DatasourceQueryError("REQUEST_FAILED", f"datasource query on {datasource_name!r}: {exc}") from exc

        result: DatasourceQueryResult
        if response.success:
            if response.truncated:
                # said here as well as carried, because a program that ignores
                # the flag is the failure this warns about, and the hub's own
                # line is on the other side of the bus.
                log.warning(
                    "datasource query result was cut at the hub's row cap; rows is a prefix",
                    extra={
                        "extra_data": {
                            "datasource": datasource_name,
                            "correlation_id": f"{request.correlation_id}",
                            "row_count": response.row_count,
                        }
                    },
                )
            result = DatasourceQueryResult(
                rows=response.rows,
                row_count=response.row_count,
                truncated=response.truncated,
                correlation_id=response.correlation_id
                if response.correlation_id is not None
                else request.correlation_id,
            )
        else:
            error_code = response.error_code or "UNKNOWN"
            log.info(
                "datasource query refused",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "correlation_id": f"{request.correlation_id}",
                        "error_code": error_code,
                    }
                },
            )
            raise DatasourceQueryError(
                error_code,
                response.error_message or f"datasource query on {datasource_name!r} refused",
            )
        return result


#: The hub's per-call row cap, MIRRORED rather than imported.
#:
#: It lives in `aibots.hub.datasources.sql_safety.MAX_RESULT_ROWS`, in the hub
#: repo, which this package cannot import -- the dependency runs the other way.
#: So this is a copy, and a copy of someone else's constant is a thing that goes
#: stale.
#:
#: What makes the staleness safe rather than silent: being WRONG-LOW only costs a
#: smaller page, while being wrong-high is refused at the door by `read_all`. The
#: failure mode of a hub that LOWERS its cap is therefore a caller passing a size
#: between the new cap and this value and losing the guard -- so if the hub's
#: value ever changes, this one moves in the same release.
_HUB_ROW_CAP: Final[int] = 1000

#: Comfortably under the cap, because ON the cap disarms the guard.
#:
#: The first shipped default was 1000, exactly the cap, which made
#: `previous_truncated` permanently False and the empty-page-after-truncated
#: check unreachable. The docstring said "must stay under the hub's row cap"
#: while the default sat on it, and the default is what a caller gets by not
#: thinking about it -- which is precisely the caller the guard protects. Found
#: by a consumer reading both sides rather than trusting either.
_DEFAULT_PAGE_SIZE: Final[int] = 500


class IncompleteReadError(RuntimeError):
    """a paged read could not be shown to have returned every row.

    Raised rather than returned, because the failure this guards is a caller
    deriving state from a PREFIX it believes is the whole relation. A short list
    of published files, treated as complete, prunes the entries it never saw.
    """


async def read_all(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    columns: Sequence[str],
    relation: str,
    key: Sequence[str],
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_pages: int = 10_000,
) -> list[dict[str, Any]]:
    """read an entire relation, or raise. never return a prefix.

    ``DatasourceQueryResult.truncated`` tells a caller the hub cut THIS page. It
    is necessary and not sufficient: a caller that checks it faithfully on every
    page can still lose rows, and this exists because that happened.

    **The trap.** Keyset paging steps past the last key it saw. When the key is
    unique only BY PROMISE -- and a warehouse enforces nothing, including a
    declared primary key -- duplicate keys make the cursor step OVER the
    duplicates. The next page comes back empty with ``truncated`` false, which is
    byte-identical to a clean finish. A hand-written helper returned 3 of 8 rows
    and reported success.

    **The signal that catches it.** The PREVIOUS page said truncated. An empty
    page after a truncated one cannot mean "reached the end" -- there were more
    rows a moment ago -- so it means the cursor skipped them, or a concurrent
    writer deleted them. Both are an incomplete read, so both raise.

    ``OFFSET`` is not offered. It looks correct in testing and silently drops a
    row when anything is deleted between pages, which is the failure mode this
    function exists to make impossible rather than merely discouraged.

    The predicate is nested-OR rather than the row-constructor form
    ``(a,b) > (?,?)``. This runs against every datasource type the platform
    admits -- redshift, snowflake, bigquery, postgres, yugabyte -- and
    row-constructor comparison is not portable across them.

    :param client: the connected query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource to read, as the hub names it
    :ptype datasource_name: str
    :param columns: columns to select, TRUSTED identifiers, never caller input
    :ptype columns: Sequence[str]
    :param relation: the table or view to read, TRUSTED, never caller input
    :ptype relation: str
    :param key: the ordering key. Must be unique for the read to be complete;
        non-uniqueness is detected rather than assumed
    :ptype key: Sequence[str]
    :param page_size: rows per page. Must be UNDER the hub's row cap and is
        REFUSED at or above it, because the hub computes ``truncated`` as
        ``total > cap`` and a ``LIMIT`` at the cap makes that unsatisfiable --
        which silently disarms the guard above. The cap bounds what crosses the
        bus and NOT what the warehouse returns: the hub materializes the whole
        result and slices, so the ``LIMIT`` here is also what keeps a page cheap
    :ptype page_size: int
    :param max_pages: refuse rather than loop forever
    :ptype max_pages: int
    :return: every row of the relation
    :rtype: list[dict[str, Any]]
    :raises IncompleteReadError: when completeness cannot be demonstrated
    :raises ValueError: when the arguments cannot describe a complete read
    """
    if not columns:
        raise ValueError("columns must not be empty: a read of no columns cannot be checked for completeness")
    if not key:
        raise ValueError("key must not be empty: keyset paging has no cursor without one")
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if page_size >= _HUB_ROW_CAP:
        # The query asks for page_size + 1, so page_size must leave room for the
        # sentinel under the hub's cap. At page_size == cap the hub would cut the
        # result at the cap and the sentinel would be the row it dropped, turning
        # "there are more rows" into "this was the last page" -- the exact silent
        # short read this function exists to prevent.
        raise ValueError(
            f"page_size must be UNDER the hub's row cap of {_HUB_ROW_CAP}, got {page_size}. "
            f"this function reads page_size + 1 rows and uses the extra one as a has-more "
            f"sentinel, so the sentinel must fit under the cap; at or above it the cap would "
            f"eat the sentinel and a short read would be returned as a complete one. "
            f"the default is {_DEFAULT_PAGE_SIZE}."
        )

    selected = ", ".join(columns)
    ordering = ", ".join(key)
    rows: list[dict[str, Any]] = []
    cursor: tuple[Any, ...] | None = None
    previous_had_more = False

    for _ in range(max_pages):
        predicate, params = _keyset_predicate(key, cursor)
        # LIMIT page_size + 1: the extra row is a SENTINEL, not data. Getting it
        # back proves more rows exist; not getting it proves they do not. That is
        # the has-more signal, and it is computed HERE from a row count we asked
        # for, independent of anything the hub decides.
        #
        # It replaces `DatasourceQueryResult.truncated`, which this function used
        # until 0.41.2 and which CANNOT serve paging. The hub computes
        # `truncated = total > MAX_RESULT_ROWS` over what the query returned, so
        # under any `LIMIT n <= cap` the comparison is unsatisfiable and truncated
        # is permanently false. `truncated` answers "did the hub cut an UNBOUNDED
        # result"; it was read here as "are there more rows", which is a different
        # question the hub is not being asked.
        sql = f"SELECT {selected} FROM {relation}{predicate} ORDER BY {ordering} LIMIT {int(page_size) + 1}"
        page = await client.query(datasource_name, sql, params=params)

        if not page.rows:
            if previous_had_more:
                raise IncompleteReadError(
                    f"{datasource_name}: empty page after a page that had more. the previous page returned "
                    f"its sentinel row, so more rows existed a moment ago and this cannot be the end of the "
                    f"relation. either the ordering key {tuple(key)} is not unique and the cursor stepped "
                    f"over duplicates, or rows were deleted mid-read. {len(rows)} rows were read and they "
                    f"are NOT the whole relation."
                )
            return rows

        had_more = len(page.rows) > page_size
        kept = page.rows[:page_size] if had_more else page.rows

        advanced = tuple(kept[-1][column] for column in key)
        if cursor is not None and advanced == cursor:
            raise IncompleteReadError(
                f"{datasource_name}: the cursor did not advance past {advanced!r}. every row in this page "
                f"carries the same key, so paging cannot make progress and the remaining rows are "
                f"unreachable by this key. {tuple(key)} is not unique in {relation}."
            )

        rows.extend(kept)
        if not had_more:
            # The sentinel was absent, so the warehouse had nothing past this
            # page and there is no need to ask again. Returning here rather than
            # looping to an empty page saves a round trip per read, and makes the
            # terminating condition the SIGNAL rather than an empty result --
            # which is the distinction this function got wrong twice.
            return rows

        cursor = advanced
        previous_had_more = had_more

    raise IncompleteReadError(
        f"{datasource_name}: still reading after {max_pages} pages ({len(rows)} rows). raising rather than "
        f"continuing, because an unbounded read against a growing relation never terminates."
    )


def _keyset_predicate(key: Sequence[str], cursor: tuple[Any, ...] | None) -> tuple[str, list[Any]]:
    """build the ``WHERE`` fragment selecting rows strictly after ``cursor``.

    Nested OR rather than a row constructor, for portability across every
    datasource type the platform admits. For key ``(a, b)`` the shape is::

        WHERE (a > ?) OR (a = ? AND b > ?)

    Values bind as parameters, so a cursor value never reaches the SQL as text.

    :param key: the ordering columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param cursor: the last key read, or ``None`` for the first page
    :ptype cursor: tuple[Any, ...] | None
    :return: the ``WHERE`` fragment (empty for page one) and its parameters
    :rtype: tuple[str, list[Any]]
    """
    if cursor is None:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []
    for index, column in enumerate(key):
        equalities = " AND ".join(f"{earlier} = ?" for earlier in key[:index])
        comparison = f"{column} > ?"
        clauses.append(f"({equalities} AND {comparison})" if equalities else f"({comparison})")
        params.extend(cursor[:index])
        params.append(cursor[index])

    return f" WHERE {' OR '.join(clauses)}", params
