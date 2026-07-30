"""Auth-method configuration descriptors, and the rules they refuse to break."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.iam.connection_types import (
    ConnectionFieldDescriptor,
    ConnectionFieldKind,
    ConnectionScope,
    ConnectionTypeDescriptor,
)


def _field(
    *,
    name: str = "issuer",
    kind: ConnectionFieldKind | str = ConnectionFieldKind.URL,
    required: bool = True,
    write_only: bool = False,
) -> ConnectionFieldDescriptor:
    return ConnectionFieldDescriptor(
        name=name,
        label=name.replace("_", " ").title(),
        kind=kind,
        required=required,
        write_only=write_only,
        help=f"what {name} is for",
    )


def _descriptor(
    *,
    type: str = "oidc",
    scopes: tuple[ConnectionScope, ...] = (ConnectionScope.PLATFORM, ConnectionScope.CUSTOMER),
    routes_by_domain: bool = False,
    fields: tuple[ConnectionFieldDescriptor, ...] = (),
) -> ConnectionTypeDescriptor:
    return ConnectionTypeDescriptor(
        type=type,
        label="OpenID Connect",
        scopes=scopes,
        routes_by_domain=routes_by_domain,
        fields=fields,
    )


# --- the vocabularies -------------------------------------------------------


def test_the_two_scopes_are_the_wire_strings_consumers_render() -> None:
    assert ConnectionScope.PLATFORM == "platform"
    assert ConnectionScope.CUSTOMER == "customer"


def test_the_known_field_kinds_are_the_wire_strings_consumers_render() -> None:
    # The vocabulary a producer builds from and every consumer is expected to render.
    # Not a parse-time gate -- see the tolerance tests below.
    assert {kind.value for kind in ConnectionFieldKind} == {"string", "secret", "url", "list", "bool", "int"}


# --- secret implies write_only ----------------------------------------------


def test_a_secret_field_that_can_be_read_back_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="not a secret"):
        _field(name="client_secret", kind=ConnectionFieldKind.SECRET, write_only=False)


def test_the_secret_rule_binds_a_kind_spelled_as_a_plain_string() -> None:
    # `kind` is a `str`, so declining to use the enum must not be a way out of the rule.
    # This is the shape a descriptor arrives in off a wire, where nobody local chose it.
    with pytest.raises(ValidationError, match="not a secret"):
        _field(name="client_secret", kind="secret", write_only=False)


def test_a_secret_field_omitting_write_only_cannot_be_constructed() -> None:
    # write_only defaults to False, so silence is a violation rather than an opt-in.
    with pytest.raises(ValidationError, match="not a secret"):
        ConnectionFieldDescriptor(
            name="client_secret",
            label="Client Secret",
            kind=ConnectionFieldKind.SECRET,
            required=True,
            help="issued by the provider",
        )


def test_a_secret_field_that_is_write_only_is_accepted() -> None:
    field = _field(name="client_secret", kind=ConnectionFieldKind.SECRET, write_only=True)
    assert field.write_only is True


def test_a_non_secret_field_may_be_readable() -> None:
    assert _field(name="client_id", kind=ConnectionFieldKind.STRING).write_only is False


def test_a_non_secret_field_may_still_be_write_only() -> None:
    # The implication runs one way only: a service may withhold a non-credential value.
    assert _field(name="private_notes", kind=ConnectionFieldKind.STRING, write_only=True).write_only is True


def test_a_secret_field_parsed_off_a_wire_is_refused_too() -> None:
    # Parsing runs the same validators, so a peer cannot deliver a descriptor this
    # process would not have been allowed to build.
    payload = (
        '{"name": "client_secret", "label": "Client Secret", "kind": "secret", '
        '"required": true, "write_only": false, "help": "issued by the provider"}'
    )
    with pytest.raises(ValidationError, match="not a secret"):
        ConnectionFieldDescriptor.model_validate_json(payload)


def test_a_field_must_carry_an_explanation() -> None:
    with pytest.raises(ValidationError):
        ConnectionFieldDescriptor(
            name="issuer",
            label="Issuer",
            kind=ConnectionFieldKind.URL,
            required=True,
        )  # type: ignore[call-arg]


# --- domain routing requires platform scope ---------------------------------


def test_domain_routing_from_a_customer_only_type_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="cross-tenant path"):
        _descriptor(scopes=(ConnectionScope.CUSTOMER,), routes_by_domain=True)


def test_domain_routing_is_accepted_when_platform_is_an_allowed_scope() -> None:
    assert _descriptor(scopes=(ConnectionScope.PLATFORM,), routes_by_domain=True).routes_by_domain is True


def test_domain_routing_is_accepted_when_platform_is_one_of_several_scopes() -> None:
    descriptor = _descriptor(
        scopes=(ConnectionScope.CUSTOMER, ConnectionScope.PLATFORM),
        routes_by_domain=True,
    )
    assert descriptor.routes_by_domain is True


def test_a_customer_only_type_that_does_not_route_by_domain_is_fine() -> None:
    assert _descriptor(scopes=(ConnectionScope.CUSTOMER,), routes_by_domain=False).routes_by_domain is False


def test_domain_routing_parsed_off_a_wire_is_refused_too() -> None:
    payload = (
        '{"type": "oidc", "label": "OpenID Connect", "scopes": ["customer"], "routes_by_domain": true, "fields": []}'
    )
    with pytest.raises(ValidationError, match="cross-tenant path"):
        ConnectionTypeDescriptor.model_validate_json(payload)


# --- scopes are a non-empty set ---------------------------------------------


def test_a_type_creatable_in_no_scope_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="declares no scopes"):
        _descriptor(scopes=())


def test_a_repeated_scope_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="repeats a scope"):
        _descriptor(scopes=(ConnectionScope.CUSTOMER, ConnectionScope.CUSTOMER))


def test_an_unknown_scope_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError):
        ConnectionTypeDescriptor(
            type="oidc",
            label="OpenID Connect",
            scopes=("everyone",),  # type: ignore[arg-type]
            routes_by_domain=False,
        )


# --- field names are unique -------------------------------------------------


def test_two_fields_sharing_a_config_key_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="duplicate field name"):
        _descriptor(fields=(_field(name="issuer"), _field(name="issuer")))


def test_distinct_field_names_are_accepted_in_the_order_given() -> None:
    descriptor = _descriptor(
        fields=(
            _field(name="issuer"),
            _field(name="client_id", kind=ConnectionFieldKind.STRING),
            _field(name="client_secret", kind=ConnectionFieldKind.SECRET, write_only=True),
        )
    )
    assert [field.name for field in descriptor.fields] == ["issuer", "client_id", "client_secret"]


def test_a_type_with_no_configurable_fields_is_legitimate() -> None:
    # A passkey connection has nothing for an operator to fill in; that is a descriptor
    # with no fields, not a missing descriptor.
    assert _descriptor(type="passkey").fields == ()


# --- the method name and the field kind stay open ---------------------------


def test_a_method_name_this_release_has_never_heard_of_is_accepted() -> None:
    # The serving service owns the vocabulary: a ninth method must reach an admin UI
    # without a release of this package, and a relay must not reject a whole registry
    # over one member it does not recognise.
    descriptor = _descriptor(type="a-method-invented-after-this-release")
    assert descriptor.type == "a-method-invented-after-this-release"


def test_a_field_kind_this_release_has_never_heard_of_is_accepted() -> None:
    # Refuse what is dangerous, tolerate what is merely unrecognised. An unknown kind
    # costs a consumer a plain text input where a richer control belonged; refusing it
    # would cost the operator every other field on the form as well.
    field = _field(name="retry_budget", kind="duration")
    assert field.kind == "duration"


def test_an_unknown_field_kind_parsed_off_a_wire_is_carried_through_verbatim() -> None:
    # The case the tolerance exists for: a producer on a newer release of this package
    # serving a kind this consumer predates. The whole descriptor must survive.
    payload = (
        '{"type": "oidc", "label": "OpenID Connect", "scopes": ["platform"], '
        '"routes_by_domain": false, "fields": ['
        '{"name": "issuer", "label": "Issuer", "kind": "url", "required": true, '
        '"write_only": false, "help": "the discovery origin"}, '
        '{"name": "retry_budget", "label": "Retry Budget", "kind": "duration", '
        '"required": false, "write_only": false, "help": "how long to keep trying"}]}'
    )
    descriptor = ConnectionTypeDescriptor.model_validate_json(payload)
    assert [(field.name, field.kind) for field in descriptor.fields] == [
        ("issuer", "url"),
        ("retry_budget", "duration"),
    ]


def test_a_known_kind_given_as_an_enum_member_serializes_to_its_wire_string() -> None:
    # Producers should build from the enum; that must not change the JSON a consumer sees.
    dumped = _field(name="enabled", kind=ConnectionFieldKind.BOOL).model_dump(mode="json")
    assert dumped["kind"] == "bool"


# --- descriptors are values -------------------------------------------------


def test_a_descriptor_cannot_be_mutated_after_construction() -> None:
    descriptor = _descriptor(scopes=(ConnectionScope.CUSTOMER,), routes_by_domain=False)
    with pytest.raises(ValidationError):
        descriptor.routes_by_domain = True  # type: ignore[misc]


def test_a_field_cannot_be_mutated_after_construction() -> None:
    # Mutation would be the way around the secret rule: build it clean, then flip the
    # flag. Frozen models close that path as well as making descriptors shareable.
    field = _field(name="client_secret", kind=ConnectionFieldKind.SECRET, write_only=True)
    with pytest.raises(ValidationError):
        field.write_only = False  # type: ignore[misc]


def test_a_descriptor_round_trips_through_json_unchanged() -> None:
    descriptor = _descriptor(
        type="saml",
        scopes=(ConnectionScope.PLATFORM, ConnectionScope.CUSTOMER),
        routes_by_domain=True,
        fields=(
            _field(name="metadata_url"),
            _field(name="sp_private_key", kind=ConnectionFieldKind.SECRET, write_only=True),
            _field(name="allowed_domains", kind=ConnectionFieldKind.LIST, required=False),
        ),
    )
    assert ConnectionTypeDescriptor.model_validate_json(descriptor.model_dump_json()) == descriptor


def test_scopes_and_fields_serialize_as_json_arrays() -> None:
    # A consumer may model these as lists; tuples here must not change the wire form.
    dumped = _descriptor(fields=(_field(name="issuer"),)).model_dump(mode="json")
    assert dumped["scopes"] == ["platform", "customer"]
    assert dumped["fields"][0]["kind"] == "url"
