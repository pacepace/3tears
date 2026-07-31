"""Describing an authentication method well enough for a UI to configure it.

An authenticating service supports some set of auth methods -- password, OIDC,
SAML, TOTP, a passkey, a signed channel handshake -- and each one needs
different things configured before it works. A descriptor is that statement,
made in data: *this method is called "OpenID Connect", it can be configured for
the whole platform or for one tenant, and configuring it requires an issuer
URL, a client id, and a client secret that is written and never read back.* An
admin surface reads descriptors and renders a form per method, so a newly
supported method reaches operators by adding a descriptor rather than by
editing a UI.

**Why this vocabulary lives in a shared package rather than in each service.**
The producing service and the relaying service used to declare these shapes
independently, on the usual reasoning for cross-repo RPC payloads: a field-name
mismatch fails closed, because a subject nobody answers has no responders. That
reasoning does not hold here. Something *does* answer, so a divergence in the
payload's shape is not a refusal -- it is a field silently dropped on the way
to the operator, or a required field the form never asks for, discovered when
someone's IdP will not accept the connection that was saved. Two independent
declarations agreeing was luck; agreeing at install time, because both sides
import the same class from the same pinned distribution, is a property.

It lives in this package specifically because the methods being described are
the ones this package implements -- :mod:`threetears.iam.oidc`,
:mod:`threetears.iam.saml`, :mod:`threetears.iam.passwords`,
:mod:`threetears.iam.totp`, :mod:`threetears.iam.webauthn`. A descriptor saying
what OIDC needs configured is vocabulary about the same subject as the code that
performs OIDC, and every service that needs these descriptors already depends on
this distribution to verify a session token.

**The invariants travel with the types, and that is the point.** Two of them are
security rules, not conveniences (a secret is write-only; domain routing needs
platform scope). Enforced as model validators, a violating descriptor cannot be
constructed at all -- not by this package, not by a service adding a method
years from now, and not by a consumer parsing a payload off a wire, since
parsing runs the same validators. A rule that lived in one service's review
habits would be a rule the next service does not have.

**What is deliberately NOT here: the transport.** The request/reply envelopes
that carry descriptors between services, and the subject or route they travel
over, belong to the services. Those genuinely do fail closed on a mismatch, and
they are one service's API rather than platform vocabulary.

**``type`` and ``kind`` are plain strings on the wire, not enums.** This looks
like an oversight in a module otherwise built out of closed vocabularies, so
here is the rule it follows: *refuse what is dangerous, tolerate what is merely
unrecognised.* Both of these are unrecognised-not-dangerous.

The set of method names is the serving service's to define. It is the service
that grows a ninth method, and a ninth method has to reach an admin UI the moment
it is served -- not one release of this package later, and without a relay in the
middle rejecting an entire registry over one member it has never heard of. A
service wanting a closed set declares its own enum and passes its values in.

A field ``kind`` is the same case, for a reason worth stating because the
alternative is tempting. Validation happens on parse, so a closed ``kind`` would
mean a consumer pinned to an older release of this package rejects the ENTIRE
payload over one value it cannot place -- an operator gets a blank page instead
of seven usable methods and one field rendered plainly, and the failure lands on
whichever of two deploys happens to be second, which is not a property anyone can
reason about at review time. What a tolerated unknown kind actually costs is a
text input where a checkbox belonged. :class:`ConnectionFieldKind` therefore
ships as the KNOWN vocabulary for producers to build descriptors from, not as a
parse-time gate: the vocabulary is real, it is simply not worth an outage.

:class:`ConnectionScope` stays closed, and the asymmetry is the point. Its
vocabulary is exhaustive by construction -- there is no third answer to "whose
connection is this" -- and a scope value this package cannot recognise IS
dangerous, because scope is what decides whether ``routes_by_domain`` is
permitted at all. An unreadable scope cannot be allowed to slip past the
cross-tenant check as an unknown string.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "ConnectionFieldDescriptor",
    "ConnectionFieldKind",
    "ConnectionScope",
    "ConnectionTypeDescriptor",
]


class ConnectionScope(StrEnum):
    """Whether a connection of some type belongs to the platform or to one tenant.

    ``platform`` is a connection with no owning tenant -- one configuration
    serving every caller, the shape a platform-wide password or staff SSO
    connection takes. ``customer`` is a connection owned by exactly one tenant,
    configured by that tenant, and usable only for that tenant's users.

    The distinction is load-bearing rather than descriptive: it decides who may
    configure the connection, and therefore whose word the platform is taking
    for what the configuration says. See
    :class:`ConnectionTypeDescriptor.routes_by_domain` for the case where that
    matters most.
    """

    PLATFORM = "platform"
    CUSTOMER = "customer"


class ConnectionFieldKind(StrEnum):
    """How one operator-facing configuration field is collected and rendered.

    **The known vocabulary, deliberately NOT a closed set on the wire.** A
    producer builds descriptors from these members, and should -- naming a kind
    from an enum is how a typo becomes an error at the point it is written. But
    :attr:`ConnectionFieldDescriptor.kind` is annotated ``str``, so a value not
    listed here parses and is carried through rather than rejected.

    That is a decision, not a gap. Validation runs on parse, so making this a
    parse-time gate would mean a consumer pinned to an older release of this
    package refusing a whole registry over one kind it cannot place -- a blank
    admin page instead of every other method still being configurable, arriving
    at whichever of two deploys is second. A tolerated unknown kind costs a text
    input where a checkbox belonged, which is what a consumer should degrade to.

    A surface still renders a field from its kind alone, with no per-method
    special-casing anywhere, so adding a member remains a real decision: every
    consumer has to learn to render it, and one that has not will render it
    plainly.
    """

    STRING = "string"
    SECRET = "secret"
    URL = "url"
    LIST = "list"
    BOOL = "bool"
    INT = "int"
    ENUM = "enum"


class ConnectionFieldDescriptor(BaseModel):
    """One operator-configurable key of a connection's configuration.

    ``name`` is the literal configuration key the connection's own code reads --
    with one deliberate exception, described on :class:`ConnectionTypeDescriptor`:
    a secret is described by the operator-facing field, not by the reference
    plumbing that stores it.

    ``write_only`` means the value may be submitted but is never returned on a
    read. A ``secret`` kind implies it, and that implication is enforced below
    rather than left to whoever writes the next descriptor: a secret that can be
    read back is not a secret.

    ``help`` is required, not optional. A field an operator must fill in without
    being told what it is for is a field they will fill in wrongly, and an
    optional explanation is one the next descriptor omits.

    ``kind`` is annotated ``str`` rather than :class:`ConnectionFieldKind`, so a
    kind this release has never heard of parses and is carried through instead of
    taking the whole payload down with it; build descriptors FROM the enum
    anyway. :class:`ConnectionFieldKind` documents why. Note the consequence for
    the rule below: the write-only check compares the string value, so it holds
    for a descriptor parsed off a wire exactly as it does for one built from the
    enum -- naming the kind ``"secret"`` is enough to be bound by it, and there is
    no spelling of a secret field that escapes the check by declining to use the
    enum.

    A descriptor names a field; it never carries that field's stored value, so a
    descriptor is not itself sensitive and is safe to serve to anyone permitted
    to configure the connection.

    **The constraint fields describe the value; they do not police it.**
    ``options``, ``minimum``, ``maximum`` and ``pattern`` exist so a configuration
    surface can offer the right control and reject an obviously-wrong value before
    an operator submits it. They are NOT the authority on what the serving service
    accepts: that service validates on write regardless, because a descriptor
    travels to consumers that may be older, newer, or simply not bothering. Read
    them as "what this field means", never as "what has already been checked".

    :param name: the configuration key this field's value is stored under
    :ptype name: str
    :param label: operator-facing field name
    :ptype label: str
    :param kind: how to collect and render the value -- a
        :class:`ConnectionFieldKind` value for anything this release knows about
    :ptype kind: str
    :param required: whether a connection of this type is unusable without it
    :ptype required: bool
    :param write_only: whether the value is accepted but never returned
    :ptype write_only: bool
    :param help: short operator-facing explanation, including any default
    :ptype help: str
    :param options: the permitted values, for an ``enum`` kind. Empty for every
        other kind.
    :ptype options: tuple[str, ...]
    :param minimum: smallest accepted value, for an ``int`` kind
    :ptype minimum: int | None
    :param maximum: largest accepted value, for an ``int`` kind
    :ptype maximum: int | None
    :param pattern: regular expression the value must match, for a ``string`` or
        ``url`` kind
    :ptype pattern: str | None
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    kind: str
    required: bool
    write_only: bool = False
    help: str
    options: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def _secret_fields_are_write_only(self) -> ConnectionFieldDescriptor:
        """A ``secret`` kind that could be read back would hand every reader of the
        connection config the credential itself. Enforced on the type so no descriptor
        anywhere -- including one added years from now, in another repository -- can
        declare one.

        Compared by value rather than by enum identity, because ``kind`` is a ``str``:
        the rule must bind a descriptor that arrived as JSON, which is the case where
        nobody local chose the kind."""
        if self.kind == ConnectionFieldKind.SECRET and not self.write_only:
            raise ValueError(
                f"field {self.name!r} is kind=secret but not write_only: a secret that can be read back is not a secret"
            )
        return self

    @model_validator(mode="after")
    def _an_enum_offers_something_to_choose(self) -> ConnectionFieldDescriptor:
        """An ``enum`` with no options is a dropdown nobody can fill in.

        Unlike the checks below, this one is about the field being usable at all
        rather than about a producer's tidiness, so it binds regardless of where the
        descriptor came from."""
        if self.kind == ConnectionFieldKind.ENUM and not self.options:
            raise ValueError(f"field {self.name!r} is kind=enum but declares no options: nothing could be selected")
        return self

    @model_validator(mode="after")
    def _constraints_match_the_kind_they_constrain(self) -> ConnectionFieldDescriptor:
        """A constraint on a kind it cannot apply to means the producer meant something
        else -- ``minimum`` on a string, ``options`` on a URL -- and a consumer would
        silently ignore it, so the field would quietly not be constrained at all.

        **Checked only against KINDS THIS RELEASE KNOWS.** An unrecognised kind carries
        its constraints through untouched, on the same reasoning `ConnectionFieldKind`
        gives for tolerating the kind itself: a future kind that legitimately uses
        ``options`` must not be rejected by every consumer released before it, which
        would be exactly the flag-day this module is built to avoid. Refuse what is
        wrong; tolerate what is merely unfamiliar.
        """
        known = {member.value for member in ConnectionFieldKind}
        if self.kind not in known:
            return self
        if self.options and self.kind != ConnectionFieldKind.ENUM:
            raise ValueError(f"field {self.name!r} is kind={self.kind} but declares options: only an enum has options")
        if (self.minimum is not None or self.maximum is not None) and self.kind != ConnectionFieldKind.INT:
            raise ValueError(
                f"field {self.name!r} is kind={self.kind} but declares minimum/maximum: only an int has a range"
            )
        if self.pattern is not None and self.kind not in {ConnectionFieldKind.STRING, ConnectionFieldKind.URL}:
            raise ValueError(
                f"field {self.name!r} is kind={self.kind} but declares a pattern: only a string or url has one"
            )
        return self

    @model_validator(mode="after")
    def _the_range_contains_something(self) -> ConnectionFieldDescriptor:
        """A minimum above its maximum accepts no value at all, so a field carrying one
        is unfillable -- and would present to an operator as a box that rejects
        everything they type with no way to discover why."""
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(
                f"field {self.name!r} has minimum {self.minimum} above maximum {self.maximum}: no value satisfies it"
            )
        return self

    @model_validator(mode="after")
    def _the_pattern_compiles(self) -> ConnectionFieldDescriptor:
        """A pattern that does not compile is not a constraint, it is an exception in
        whichever consumer tries to apply it first -- and that consumer is a UI, so the
        failure lands on an operator rather than on whoever wrote the descriptor."""
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"field {self.name!r} declares a pattern that does not compile: {exc}") from exc
        return self


class ConnectionTypeDescriptor(BaseModel):
    """Everything an admin surface needs to configure one authentication method.

    The point of this shape is that a surface serving auth-method configuration
    reads descriptors and renders them, rather than carrying a bespoke form per
    method -- so a new method becomes reachable by adding a descriptor, not by
    editing the UI.

    **Secrets are described, not plumbed.** Where a connection's code resolves a
    secret through a ``scheme://locator`` reference
    (:mod:`threetears.core.security.secret_refs`), the descriptor names the
    operator-facing field -- ``client_secret``, kind ``secret`` -- and not the
    ``*_ref`` key the stored configuration actually holds. How a secret gets from
    an operator to the resolver is the backend's business, and an operator who
    has to know about the reference indirection has been handed an implementation
    detail to get wrong.

    **``routes_by_domain`` may only be true when ``platform`` is an allowed
    scope.** This is a security constraint, not a stylistic one. Domain-to-tenant
    allocation decides which tenant an otherwise unrecognised sign-in belongs to.
    A customer-scoped connection is configured by that customer, who could assert
    a domain they do not own -- so letting one consult the domain table is a
    cross-tenant path: assert ``example.com``, receive sign-ins belonging to
    whoever actually owns it. Enforced below.

    ``type`` is a plain string on purpose; the module docstring explains why.

    :param type: the method described, as the serving service names it
    :ptype type: str
    :param label: operator-facing name, e.g. "OpenID Connect"
    :ptype label: str
    :param scopes: the scopes a connection of this type may be created in
    :ptype scopes: tuple[ConnectionScope, ...]
    :param routes_by_domain: whether it participates in domain-to-tenant allocation
    :ptype routes_by_domain: bool
    :param fields: the operator-configurable fields, empty for methods with none
    :ptype fields: tuple[ConnectionFieldDescriptor, ...]
    """

    model_config = ConfigDict(frozen=True)

    type: str
    label: str
    scopes: tuple[ConnectionScope, ...]
    routes_by_domain: bool
    fields: tuple[ConnectionFieldDescriptor, ...] = ()

    @model_validator(mode="after")
    def _scopes_are_a_non_empty_set(self) -> ConnectionTypeDescriptor:
        """A type creatable in no scope at all could never be configured, and a repeated
        scope means whoever wrote the descriptor was not sure what they meant."""
        if not self.scopes:
            raise ValueError(f"connection type {self.type!r} declares no scopes")
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError(f"connection type {self.type!r} repeats a scope: {list(self.scopes)}")
        return self

    @model_validator(mode="after")
    def _domain_routing_requires_platform_scope(self) -> ConnectionTypeDescriptor:
        """See the class docstring: a customer-scoped connection consulting the domain
        table is a cross-tenant path, because the customer controls that identity
        provider and could assert a domain it does not own."""
        if self.routes_by_domain and ConnectionScope.PLATFORM not in self.scopes:
            raise ValueError(
                f"connection type {self.type!r} sets routes_by_domain without the "
                "'platform' scope: domain-to-tenant allocation from a customer-controlled "
                "connection is a cross-tenant path"
            )
        return self

    @model_validator(mode="after")
    def _field_names_are_unique(self) -> ConnectionTypeDescriptor:
        """Two descriptors for one configuration key means one of them is silently
        ignored, and which one depends on iteration order at the consumer."""
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"connection type {self.type!r} declares a duplicate field name: {names}")
        return self
