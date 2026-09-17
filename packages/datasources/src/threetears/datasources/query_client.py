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

import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer, field_validator, model_validator
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
    "RelationFingerprintRequest",
    "RelationFingerprintResult",
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

#: one unquoted SQL identifier, the shape every admitted engine accepts unquoted.
#:
#: Deliberately NOT permitting a quoted identifier. A quoted name may contain the quote
#: character doubled, so admitting them means implementing the escaping too -- and getting
#: that wrong reopens exactly the hole this closes, in a form that reads as handled. A
#: relation needing quoting is a relation this ask does not serve.
_IDENTIFIER_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

#: a relation: one identifier, or two joined by a single dot (``schema.table``).
#:
#: Two parts at most. A three-part ``catalog.schema.table`` is admitted by some engines and
#: not others, and the value is interpolated verbatim, so widening this is a per-engine
#: decision rather than a regex change.
_RELATION_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")


class RelationFingerprintRequest(BaseModel):
    """ask the hub to count and fingerprint a relation, instead of running SQL.

    Declarative rather than a statement, because the statement is dialect-specific
    and the caller does not know which engine answers. Turning a hash into a
    summable number is spelled three different ways across the admitted engines, so
    the caller names WHAT it wants fingerprinted and the driver writes the SQL.

    :param relation: schema-qualified relation name
    :ptype relation: str
    :param key_columns: the ordering columns the digest is computed over -- the same
        key the caller pages by, so the digest describes exactly the rows it is reading.
        Spelled ``key_columns`` rather than ``key`` because the secrets gate cannot tell a
        database key from a credential by name, and states the database meaning in the name
        rather than asking a reader to trust a pattern
    :ptype key_columns: list[str]
    :raises ValueError: if ``relation`` or any ``key_columns`` entry is not a plain SQL
        identifier, or if ``key_columns`` is empty
    """

    model_config = ConfigDict(extra="forbid")

    relation: str
    key_columns: list[str]

    @field_validator("relation")
    @classmethod
    def _relation_is_an_identifier(cls, value: str) -> str:
        """refuse anything that is not one or two plain identifiers joined by a dot.

        **THIS IS THE INJECTION BOUNDARY, and it is the only one.** Every driver
        implementing ``relation_fingerprint`` interpolates this value straight into a
        statement -- it has to, because a relation name cannot be a bind parameter in any
        admitted engine -- and each documents it as a TRUSTED identifier. Nothing made it
        trusted: these fields arrive off the wire, and the broker's fingerprint branch
        deliberately skips ``validate_read_sql`` because the ask is declarative rather than
        a statement. So the trust the drivers assume is established HERE or nowhere, and
        ``fingerprint.relation = 'x; DROP TABLE y --'`` reaches the warehouse.

        Validated at the MODEL rather than at each broker, because there is one model and
        several brokers, and a check per caller is a check somebody adds a caller without.

        :param value: the proposed relation name
        :ptype value: str
        :return: the value unchanged
        :rtype: str
        :raises ValueError: if it is not ``name`` or ``schema.name``, each a plain
            identifier
        """
        if not _RELATION_GRAMMAR.match(value):
            raise ValueError(
                f"relation {value!r} is not a plain SQL identifier or schema-qualified pair; "
                f"it is interpolated into a statement, so it must match "
                f"{_RELATION_GRAMMAR.pattern}"
            )
        return value

    @field_validator("key_columns")
    @classmethod
    def _key_columns_are_identifiers(cls, value: list[str]) -> list[str]:
        """refuse an empty key, and any entry that is not a plain identifier.

        Empty is refused here rather than at the driver because every driver already
        raises on it separately -- surfacing as a warehouse error the caller cannot tell
        from a connection fault, rather than as the malformed request it is.

        :param value: the proposed ordering columns
        :ptype value: list[str]
        :return: the value unchanged
        :rtype: list[str]
        :raises ValueError: if empty, or if any entry is not a plain identifier
        """
        if not value:
            raise ValueError("key_columns must name at least one column; the digest has no order without it")
        bad = [c for c in value if not _IDENTIFIER_GRAMMAR.match(c)]
        if bad:
            raise ValueError(
                f"key_columns {bad!r} are not plain SQL identifiers; they are interpolated "
                f"into a statement, so each must match {_IDENTIFIER_GRAMMAR.pattern}"
            )
        return value


class RelationFingerprintResult(BaseModel):
    """a relation's row count and key digest at one instant.

    Compared for equality against a later reading of the same relation. The digest
    is OPAQUE: never parsed, never compared across engines, never carried across a
    driver upgrade.

    :param row_count: rows in the relation at the moment of the read
    :ptype row_count: int
    :param digest: opaque value identifying the ordering-key values present
    :ptype digest: str
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int
    digest: str


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
    query: str | None = None
    params: list[Any] = Field(default_factory=list)
    fingerprint: RelationFingerprintRequest | None = None

    @model_validator(mode="after")
    def _exactly_one_operation(self) -> "DatasourceQueryRequest":
        """require exactly one of ``query`` and ``fingerprint``.

        The two are alternative asks on one subject, and the alternative to this
        check is a model that can represent a request meaning nothing (neither
        set) or two things at once (both set). Refused at the border, so no
        handler below has to decide which one wins.

        Reusing the subject rather than minting a second one is deliberate: a new
        subject is a new NATS grant on a security surface, and it would buy
        nothing -- the hub verifies the forwarded identity and evaluates
        ``datasource.read`` on the same namespace for either ask.

        :return: the validated request
        :rtype: DatasourceQueryRequest
        :raises ValueError: when neither or both are set
        """
        asked = [
            name for name, value in (("query", self.query), ("fingerprint", self.fingerprint)) if value is not None
        ]
        if len(asked) != 1:
            raise ValueError(
                f"a datasource request carries exactly one of query or fingerprint, got {asked or 'neither'}"
            )
        return self

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
    #: set only in answer to a ``fingerprint`` request; ``None`` for a query, whose
    #: answer is ``rows``. A separate field rather than a row, because a caller
    #: comparing two readings must not have to know which column the digest landed in
    #: or what the driver called it.
    fingerprint: RelationFingerprintResult | None = None


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
    async def relation_fingerprint(
        self,
        datasource_name: str,
        *,
        relation: str,
        key: Sequence[str],
        user_identity_token: str | None = None,
        correlation_id: UUID | None = None,
    ) -> RelationFingerprintResult:
        """count and fingerprint a relation, for a caller proving a read was complete.

        The caller names WHAT to fingerprint rather than sending SQL, because the
        statement is dialect-specific: the hash-to-number step is spelled three
        different ways across the admitted engines, and a client reaching a
        datasource through the hub does not know which one answers.

        Compare two readings for equality. An unchanged pair means the relation
        held still between them; ``row_count`` separately answers whether a paged
        read returned all of it.

        :param datasource_name: the datasource's name as the hub's ``datasources``
            table holds it
        :ptype datasource_name: str
        :param relation: schema-qualified relation name
        :ptype relation: str
        :param key: the ordering columns the digest is computed over
        :ptype key: Sequence[str]
        :param user_identity_token: the per-turn user assertion when a human is
            in the loop; ``None`` evaluates the principal's own grants alone
        :ptype user_identity_token: str | None
        :param correlation_id: trace id to carry; generated when omitted
        :ptype correlation_id: UUID | None
        :return: the relation's current row count and key digest
        :rtype: RelationFingerprintResult
        :raises DatasourceQueryError: on a refusal, a transport failure, or a
            success carrying no fingerprint -- which would otherwise read as a
            relation that had not changed
        """
        request = DatasourceQueryRequest(
            correlation_id=correlation_id if correlation_id is not None else uuid7(),
            identity_token=SecretStr(self.forwarded_identity_token()),
            user_identity_token=SecretStr(user_identity_token) if user_identity_token is not None else None,
            fingerprint=RelationFingerprintRequest(relation=relation, key_columns=list(key)),
        )
        subject = Subjects.datasource_query(datasource_name)
        try:
            response: DatasourceQueryResponse = await self._nats_client.request(
                subject=subject,
                message=request,
                response_type=DatasourceQueryResponse,
                timeout=self._timeout,
            )
        except RequestError as exc:
            raise DatasourceQueryError("REQUEST_FAILED", f"relation fingerprint on {datasource_name!r}: {exc}") from exc

        if not response.success:
            raise DatasourceQueryError(
                response.error_code or "UNKNOWN",
                response.error_message or f"relation fingerprint on {datasource_name!r} was refused",
            )
        if response.fingerprint is None:
            # A success with no fingerprint is a hub that did not understand the ask.
            # Refusing beats returning a sentinel: a caller comparing two of those
            # would find them equal and conclude the relation had not changed.
            raise DatasourceQueryError(
                "MALFORMED_RESPONSE",
                f"relation fingerprint on {datasource_name!r} returned success with no fingerprint",
            )
        return response.fingerprint

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

    **The trap.** Keyset paging steps past the last key it saw. When the key is
    unique only BY PROMISE -- and a warehouse enforces nothing, including a
    declared primary key -- a run of duplicate keys straddling a page boundary
    is stepped OVER: the next page asks for rows strictly greater than the
    cursor, and the rest of that run is not greater. A hand-written helper
    returned 3 of 8 rows and reported success.

    **What makes the read complete.** The relation is COUNTED AND FINGERPRINTED
    before paging begins and again after the last page. A read that does not
    return as many rows as the count raises, and so does one where the two
    readings differ. That is the contract, and it is deliberately not a statement
    about any mechanism: it holds for the ways to lose a row described below, and
    for the ones nobody has thought of. Four releases of this function each
    shipped a guard that was correct about the failure it named and blind to the
    next one, which is the evidence that enumerating mechanisms does not converge.

    **Why a count alone was not enough.** A count proves only THAT the right
    number of rows arrived. A delete and an insert landing during the read leave
    it unchanged, so a list half from before the change and half from after --
    a state of the relation that never existed at any instant -- passed as
    complete. The digest over the ordering key is what sees that.

    The readings and the pages are not one transaction, and no portable way to
    ask for that exists across the admitted engines. So a relation written during
    the read is REPORTED rather than serialised around: on a changing relation
    there is no whole relation to return, and this function promises the whole
    relation.

    Everything below is still here because the readings say only THAT rows are
    missing or changed. These say WHICH mechanism lost them, which is what an
    operator needs at three in the morning.

    **The sentinel.** Each page asks for ``page_size + 1`` rows.
    The extra row is a SENTINEL, not data: getting it back proves more rows
    exist. When it comes back, every row sharing ITS key is dropped from this
    page and re-read at the head of the next one, so a run of equal keys is
    never split by a boundary and the cursor never steps over an unread row.
    Re-reading at most one key group per page is the cost of that guarantee.

    **What cannot be made complete.** A single key value filling an entire page
    leaves paging nowhere to step, so it raises rather than advancing. So does
    an empty page arriving after one that returned its sentinel, which means
    rows that existed a moment ago are gone -- a concurrent delete, and an
    incomplete read either way.

    ``DatasourceQueryResult.truncated`` is NOT part of any of this. The hub
    computes it as ``total > cap`` over what a query returned, so under any
    ``LIMIT n <= cap`` it is permanently false. It answers "did the hub cut an
    UNBOUNDED result", which is a different question from "are there more rows";
    reading it as the latter is what shipped three releases with a dead guard.

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
        REFUSED at or above it, because the query asks for ``page_size + 1`` and
        the sentinel must fit under the cap; at the cap the hub would cut the
        result and the sentinel would be the row it dropped, reporting a short
        read as a complete one. Must also exceed the largest run of equal keys,
        or the read raises rather than losing rows. The cap bounds what crosses
        the bus and NOT what the warehouse returns: the hub materializes the
        whole result and slices, so the ``LIMIT`` here is what keeps a page cheap
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

    # THE TOTAL IS THE PROOF. Everything below this line guards a specific way to
    # lose a row, and four releases of this function shipped a guard that was
    # correct about the mechanism it named and blind to the next one. Counting
    # first and comparing at the end asks a different question -- not "did I think
    # of every way to skip a row" but "are they all here" -- and that question has
    # one answer that does not depend on enumerating anything.
    #
    # It is what catches a NULL in the key, which no keyset predicate can reach:
    # `column > value` is NULL rather than true for such a row, so it is skipped
    # and so is everything ordered after it, and the page that excluded it comes
    # back short, which is indistinguishable from reaching the end.
    #
    # The count and the pages are not one transaction, so a relation being written
    # during the read will not match. That is reported rather than hidden: on a
    # relation that is changing there is no "whole relation" to return, and this
    # function's promise is the whole relation or a raise.
    before = await client.relation_fingerprint(datasource_name, relation=relation, key=key)

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
            return await _proven(client, datasource_name, relation, key, rows, before)

        had_more = len(page.rows) > page_size
        if not had_more:
            # No sentinel: the warehouse had nothing past this page, so every row
            # is safe to keep and there is no boundary to worry about.
            rows.extend(page.rows)
            return await _proven(client, datasource_name, relation, key, rows, before)

        # A KEY GROUP MUST NOT STRADDLE THE BOUNDARY. The next page asks for rows
        # strictly greater than the cursor, so any row sharing the cursor's key is
        # unreachable once we move past it. Trimming blindly at page_size splits a
        # run of equal keys and silently drops its tail -- a 7-row relation with a
        # 3-run in the middle returned 6 rows and reported success.
        #
        # So the trailing group is dropped from this page and re-read at the head
        # of the next one. It costs re-reading at most one group per page and it
        # is what makes the promise hold for a key that is unique only by promise.
        kept = page.rows[:page_size]
        sentinel_key = tuple(page.rows[page_size][column] for column in key)
        while kept and tuple(kept[-1][column] for column in key) == sentinel_key:
            kept.pop()

        if not kept:
            # Every row in the page shares the sentinel's key, so the group is
            # larger than the page and no page size below it can advance. Raising
            # names the cause; a bigger page_size is the fix when the group is
            # genuinely smaller than the cap.
            raise IncompleteReadError(
                f"{datasource_name}: a single value of {tuple(key)} fills an entire page of "
                f"{page_size} rows in {relation}, so paging cannot step past it without dropping "
                f"rows. {tuple(key)} is not unique. read with a larger page_size, or use a key "
                f"that is."
            )

        advanced = tuple(kept[-1][column] for column in key)
        if cursor is not None and advanced == cursor:
            # Defensive: every row here should already satisfy `> cursor`, so this
            # can only fire if the warehouse returned rows the predicate excluded.
            raise IncompleteReadError(
                f"{datasource_name}: the cursor did not advance past {advanced!r}, though the "
                f"predicate asked for rows strictly greater than it. the result did not honour "
                f"the keyset predicate, so completeness cannot be established."
            )

        rows.extend(kept)
        cursor = advanced
        previous_had_more = had_more

    raise IncompleteReadError(
        f"{datasource_name}: still reading after {max_pages} pages ({len(rows)} rows). raising rather than "
        f"continuing, because an unbounded read against a growing relation never terminates."
    )


async def _proven(
    client: DatasourceQueryClient,
    datasource_name: str,
    relation: str,
    key: Sequence[str],
    rows: list[dict[str, Any]],
    before: RelationFingerprintResult,
) -> list[dict[str, Any]]:
    """return ``rows`` only if it provably holds the whole relation, unchanged.

    The completeness check of last resort, and the only one that does not depend
    on naming the mechanism that lost a row. Guards elsewhere in this module each
    catch one skip and say something useful about it; this catches any skip at
    all, including the ones nobody has thought of yet.

    **Two questions, and a count answers only the first.** The row count says
    whether the read returned as many rows as the relation held. The digest says
    whether they were the same rows: a delete and an insert during the read leave
    the count identical, so a list half old and half new passes a count check
    while being a state of the relation that never existed at any instant.

    The second fingerprint is taken AFTER the last page, so the pair brackets the
    whole read. Neither reading shares a transaction with the pages -- there is no
    cross-engine way to ask for that through this wire -- so a relation written
    during the read is REPORTED rather than serialised around. On a changing
    relation there is no whole relation to return, and this function's promise is
    the whole relation or a raise.

    :param client: the client the second fingerprint is taken through
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource read, as the hub names it
    :ptype datasource_name: str
    :param relation: the table or view read
    :ptype relation: str
    :param key: the ordering key, named in the message because it is the usual cause
    :ptype key: Sequence[str]
    :param rows: every row collected by the paging loop
    :ptype rows: list[dict[str, Any]]
    :param before: the fingerprint taken before paging began
    :ptype before: RelationFingerprintResult
    :return: ``rows`` unchanged, when it is provably complete
    :rtype: list[dict[str, Any]]
    :raises IncompleteReadError: when the row count disagrees, or the relation
        changed under the read
    """
    if len(rows) != before.row_count:
        short = before.row_count - len(rows)
        cause = (
            f"{short} row(s) were never returned. the usual cause is a value the ordering key cannot "
            f"page past: a NULL in {tuple(key)} is skipped by every keyset predicate, because "
            f"`column > $1` is NULL rather than true for such a row, and so is every row ordered after "
            f"it. read with a key whose columns are NOT NULL."
            if short > 0
            else f"{-short} row(s) MORE than the count arrived, so {relation} was written during the read."
        )
        raise IncompleteReadError(
            f"{datasource_name}: {relation} counted {before.row_count} rows and the read returned {len(rows)}. {cause}"
        )

    after = await client.relation_fingerprint(datasource_name, relation=relation, key=key)
    if after != before:
        raise IncompleteReadError(
            f"{datasource_name}: {relation} changed while it was being read. the row count and the "
            f"ordering-key digest taken before the first page do not match the pair taken after the "
            f"last, so these {len(rows)} rows are not any one state of the relation -- they may hold "
            f"rows that no longer exist beside rows that did not exist when the read began. re-read "
            f"once the relation is settled."
        )

    return rows


def _keyset_predicate(key: Sequence[str], cursor: tuple[Any, ...] | None) -> tuple[str, list[Any]]:
    """build the ``WHERE`` fragment selecting rows strictly after ``cursor``.

    Nested OR rather than a row constructor, for portability across every
    datasource type the platform admits. For key ``(a, b)`` the shape is::

        WHERE (a > $1) OR (a = $2 AND b > $3)

    Values bind as parameters, so a cursor value never reaches the SQL as text.

    **``$N``, not ``?``, and the distinction is the whole reason page two
    executes.** Every driver normalises placeholders through
    :func:`threetears.datasources.drivers._util._translate_placeholders`, which
    recognises ``$N`` alone -- rewriting it to ``%s``, ``:N`` or ``@pN`` for the
    engine in front of it. A ``?`` is not a placeholder to any of them, so it
    travels to the engine verbatim and the bound values arrive with nothing to
    bind to.

    Page one hid that for as long as it existed: with no cursor this returns no
    fragment and no parameters, so single-page reads and every relation smaller
    than one page succeed. The failure arms on the day a relation outgrows a
    page.

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
        # numbered in emission order, so the Nth placeholder names the Nth
        # parameter appended just below and the two cannot drift apart.
        equalities = " AND ".join(
            f"{earlier} = ${len(params) + offset + 1}" for offset, earlier in enumerate(key[:index])
        )
        comparison = f"{column} > ${len(params) + index + 1}"
        clauses.append(f"({equalities} AND {comparison})" if equalities else f"({comparison})")
        params.extend(cursor[:index])
        params.append(cursor[index])

    return f" WHERE {' OR '.join(clauses)}", params
