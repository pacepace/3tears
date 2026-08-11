"""Canonical serialization -- a public contract feature, not a replay internal.

One canonical form, two consumers that must agree on it (D26, SR-F1): the
replay key hashes it, and eval run identity digests it (search parameters
already participate in discodon's ``canonical_digest``). The rules:

- **explicitly-set parameters only** -- resolved defaults never appear, so
  adding a parameter with a default shifts no existing key;
- **absent and defaulted are canonically identical** -- a caller that
  explicitly passes a field's default value produces the same form as one
  that omitted the field;
- **stable form** -- sorted keys, compact separators, and (per type, via
  :attr:`ContractModel.CANONICAL_ORDER_INSENSITIVE`) order-insensitive
  fields sorted element-wise, so equal requests can never serialize
  unequally.

The digest is an opaque equality token: nobody ever parses it. Replay's
key-derivation versioning (the D26 envelope field) lives with the replay
record, not here -- this module versions the *form* so a derivation change
is nameable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from pydantic import BaseModel
from pydantic_core import PydanticUndefined, to_jsonable_python

from threetears.search.contracts._base import ContractModel

__all__ = ["CANONICAL_FORM_VERSION", "canonical_digest", "canonicalize"]

#: version of the canonical-form rules above. Bumped only by a genuinely
#: incompatible change to how the form is derived; consumers that persist
#: digests (replay envelopes) record it so a mismatch names both versions
#: instead of being a mysterious miss (D26).
CANONICAL_FORM_VERSION: Final[int] = 1


def _sort_key(value: object) -> str:
    """Stable sort key for elements of an order-insensitive field.

    :param value: one already-JSON-safe element
    :ptype value: object
    :return: a deterministic string the element sorts by
    :rtype: str
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize(model: BaseModel) -> str:
    """Render ``model``'s explicitly-set fields in canonical form.

    :param model: a contract request/parameter model. Nested payload models
        in canonicalized types keep their fields required (the contracts'
        criteria do), so the explicitly-set rule is exact at every depth
    :ptype model: BaseModel
    :return: the canonical JSON text -- sorted keys, compact separators,
        explicitly-set-and-non-default fields only
    :rtype: str
    """
    dumped = model.model_dump(mode="json")
    fields = type(model).model_fields
    order_insensitive: frozenset[str] = frozenset()
    if isinstance(model, ContractModel):
        order_insensitive = type(model).CANONICAL_ORDER_INSENSITIVE

    payload: dict[str, object] = {}
    for name in fields:
        if name not in model.model_fields_set:
            continue
        value = dumped[name]
        default = fields[name].get_default(call_default_factory=True)
        if default is not PydanticUndefined and to_jsonable_python(default) == value:
            continue
        if name in order_insensitive and isinstance(value, list):
            value = sorted(value, key=_sort_key)
        payload[name] = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(model: BaseModel) -> str:
    """Digest ``model``'s canonical form for equality keying.

    :param model: a contract request/parameter model
    :ptype model: BaseModel
    :return: hex SHA-256 of :func:`canonicalize`'s output
    :rtype: str
    """
    return hashlib.sha256(canonicalize(model).encode("utf-8")).hexdigest()
