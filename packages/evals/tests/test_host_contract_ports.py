"""A host-contract port answers only with declared types: every type its methods take or return, and every type its
data members hold, is ``host-contract`` or ``stable``.

A port is a declared protocol a host's code implements and hands the engine (``port-host-implements`` in
``tests/fixtures/declared_holdbacks.txt``). A host implementing one writes each type its members name, as a method's
parameter or return or as an attribute it holds, so a ``provisional`` type there makes a host binding to the contract bind to a name that may move,
and carry an exemption for it. Each type is read from the signatures as written, the walk ``scripts/surface_kinds.py``
reads for reach, and a type counts as declared when any public root exports it under a declared name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")
kinds_of = load_script("surface_kinds.py")


def host_contract_ports(installed: Mapping[str, Any]) -> list[str]:
    """Each ``host-contract`` protocol a host implements, by ``root.name``."""
    sides = kinds_of.protocol_sides()
    return [
        name
        for name, entry in installed.items()
        if entry.tier == "host-contract" and entry.kind == "protocol" and sides.get(name) == kinds_of.HOST_IMPLEMENTS
    ]


def undeclared_port_types(installed: Mapping[str, Any]) -> dict[str, set[str]]:
    """Each host-contract port naming a type no declared name exports, as a method's parameter or return or as a data
    member's type, with those types by every name they go by.

    :param installed: ``root.name`` → its surface entry, with its tier (``public_surface.installed_surface()``).
    :return: Port → the undeclared types its members name; empty when every port is closed.
    """
    names_of: dict[int, set[str]] = {}
    objects: dict[str, object] = {}
    for root, name, obj in surface.public_names():
        names_of.setdefault(id(obj), set()).add(f"{root}.{name}")
        objects[f"{root}.{name}"] = obj
    found: dict[str, set[str]] = {}
    for port in host_contract_ports(installed):
        for position, named in kinds_of.edges(objects[port]):
            if position not in ("parameter", "return", "field"):
                continue
            names = names_of.get(id(named), set())
            if any(surface.declared(installed[each].tier) for each in names if each in installed):
                continue
            found.setdefault(port, set()).add(" = ".join(sorted(names)) or f"{named!r} (exported from no root)")
    return found


def test_every_type_a_host_contract_port_names_is_declared() -> None:
    open_ports = undeclared_port_types(surface.installed_surface())
    assert not open_ports, (
        "a host-contract port takes or returns a type that is not host-contract or stable, so a host implementing it "
        "binds to a name that may move. Promote each type to host-contract in scripts/public_surface.py and run "
        f"`{surface.REGENERATE_SNAPSHOT}`:\n  "
        + "\n  ".join(f"{port}: {', '.join(sorted(types))}" for port, types in sorted(open_ports.items()))
    )


def test_the_walk_reads_each_port_and_sees_a_provisional_type() -> None:
    """The rule above is not vacuous: the client factory's ports are walked, and the model listing the optional one
    returns would be named the moment it stopped being declared."""
    installed = surface.installed_surface()
    ports = host_contract_ports(installed)
    assert {"threetears.evals.kernel.host.CompletionClients", "threetears.evals.kernel.host.ModelListingReader"} <= set(
        ports
    )
    listing = "threetears.evals.schema.ModelListing"
    demoted = {**installed, listing: dataclasses.replace(installed[listing], tier="provisional")}
    assert undeclared_port_types(demoted) == {"threetears.evals.kernel.host.ModelListingReader": {listing}}


@pytest.mark.parametrize(
    ("port", "member_type"),
    [
        # An annotated data member: the artifact a candidate kind declares it hands the judge.
        ("threetears.evals.kernel.CandidateKind", "threetears.evals.schema.JudgedArtifact"),
        # A property: the normalized stop reason a completion carries.
        ("threetears.evals.schema.CompletionResult", "threetears.evals.schema.StopReason"),
    ],
)
def test_a_provisional_type_a_port_holds_as_a_member_is_named(port: str, member_type: str) -> None:
    """A port's data members are part of what a host writes, so each type one holds is named the moment it stopped
    being declared."""
    installed = surface.installed_surface()
    demoted = {**installed, member_type: dataclasses.replace(installed[member_type], tier="provisional")}
    assert member_type in undeclared_port_types(demoted).get(port, set())
