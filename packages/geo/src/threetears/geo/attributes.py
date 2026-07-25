"""coercion from SQL column values to MVT attribute values.

MVT carries only strings, numbers and booleans. everything a source table can
hold has to be mapped onto those three, and the mapping is fixed here rather
than left to each implementation, because two of the cases are quietly
lossy if chosen badly:

**NULL becomes an absent key, not a zero.** the format has no null, so the
only faithful encoding is to omit the attribute. that puts a real obligation
on the client: a style expression reading an omitted attribute must supply
its own fallback. it also means "no data" and "a genuine zero" remain
distinguishable in the tile -- which matters most on a choropleth, where
collapsing them shades empty regions as though they were measured.

**Datetimes become ISO-8601 UTC strings.** MVT numbers are doubles, so an
epoch timestamp loses sub-second precision at present-day magnitudes and
reads as an opaque number in every debugging tool.

**Structured values are rejected at registration, not silently stringified.**
a JSON blob rendered into a tile attribute is both large and unusable by
style expressions; the layer should project the scalar it actually needs into
its own column instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

from threetears.observe import get_logger

__all__ = [
    "MVT_SCALAR_TYPES",
    "UnsupportedAttributeError",
    "coerce_attribute",
    "coerce_attributes",
    "validate_attribute_value",
]

log = get_logger(__name__)

#: the complete MVT attribute value domain.
MVT_SCALAR_TYPES: Final = (str, bool, int, float)

#: types that must not reach a tile. rejected loudly at layer registration so
#: the failure lands on whoever declared the layer rather than on whoever
#: later wonders why a style expression never matches.
_STRUCTURED_TYPES: Final = (dict, list, tuple, set, bytes, bytearray, Mapping)


class UnsupportedAttributeError(TypeError):
    """raised when a column type has no faithful MVT representation."""


def validate_attribute_value(column: str, value: Any) -> None:
    """raise when ``value`` cannot be represented as an MVT attribute.

    called at layer registration against a sampled row, so a bad column
    declaration fails at configuration time rather than mid-build.

    :param column: column name, for the error message
    :ptype column: str
    :param value: sampled value
    :ptype value: Any
    :return: nothing
    :rtype: None
    :raises UnsupportedAttributeError: for structured values
    """
    if value is None:
        return
    if isinstance(value, _STRUCTURED_TYPES):
        raise UnsupportedAttributeError(
            f"column {column!r} holds {type(value).__name__}, which has no MVT representation; "
            "project the scalar the layer actually needs into its own column"
        )


def coerce_attribute(value: Any) -> str | bool | int | float | None:
    """map one SQL value onto the MVT scalar domain.

    ``None`` propagates as ``None`` and is dropped by :func:`coerce_attributes`
    rather than encoded, per the module docstring.

    :param value: source value
    :ptype value: Any
    :return: an MVT-representable scalar, or ``None`` to omit the key
    :rtype: str | bool | int | float | None
    :raises UnsupportedAttributeError: for structured values
    """
    if value is None:
        return None
    # bool first: it is an int subclass, and encoding True as 1 loses the
    # type that style expressions match on.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, datetime):
        # naive timestamps are assumed UTC rather than local: a tile is served
        # to every timezone, so local time is never the right reading.
        stamped = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return stamped.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        # MVT numbers are doubles; a Decimal that does not fit is a data
        # modelling problem, not something to paper over silently.
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, _STRUCTURED_TYPES):
        raise UnsupportedAttributeError(
            f"{type(value).__name__} has no MVT representation; project a scalar column instead"
        )
    return str(value)


def coerce_attributes(row: Mapping[str, Any], columns: tuple[str, ...]) -> dict[str, str | bool | int | float]:
    """project ``columns`` out of ``row`` into MVT-safe attributes.

    keys whose value coerces to ``None`` are omitted, which is the format's
    only representation of absence.

    :param row: source row keyed by column name
    :ptype row: Mapping[str, Any]
    :param columns: declared attribute columns for the layer
    :ptype columns: tuple[str, ...]
    :return: attributes ready for MVT encoding
    :rtype: dict[str, str | bool | int | float]
    :raises UnsupportedAttributeError: for structured values
    """
    attributes: dict[str, str | bool | int | float] = {}
    for column in columns:
        coerced = coerce_attribute(row.get(column))
        if coerced is not None:
            attributes[column] = coerced
    return attributes
