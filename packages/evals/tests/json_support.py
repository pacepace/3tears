"""Reading a value out of stored or computed JSON in a test, its shape checked at each step."""

from __future__ import annotations

from pydantic import JsonValue


def json_at(value: JsonValue, *path: str | int) -> JsonValue:
    """The JSON value at ``path`` below ``value``: a key steps into an object, an index into an array.

    A step whose container is the wrong shape fails here, naming it, rather than reading as a missing value.

    :param value: The JSON to read.
    :ptype value: JsonValue
    :param path: Object keys and array indexes, outermost first.
    :ptype path: str | int
    :return: The value the path reaches.
    :rtype: JsonValue
    """
    for step in path:
        if isinstance(step, int):
            assert isinstance(value, list), f"{step!r} indexes {value!r}, not an array"
            value = value[step]
        else:
            assert isinstance(value, dict), f"{step!r} keys {value!r}, not an object"
            value = value[step]
    return value


def json_list(value: JsonValue, *path: str | int) -> list[JsonValue]:
    """The JSON array at ``path`` below ``value``; anything else there fails, naming it.

    :param value: The JSON to read.
    :ptype value: JsonValue
    :param path: Object keys and array indexes, outermost first.
    :ptype path: str | int
    :return: The array the path reaches.
    :rtype: list[JsonValue]
    """
    found = json_at(value, *path)
    assert isinstance(found, list), f"{path!r} reaches {found!r}, not an array"
    return found


def json_object(value: JsonValue, *path: str | int) -> dict[str, JsonValue]:
    """The JSON object at ``path`` below ``value``; anything else there fails, naming it.

    :param value: The JSON to read.
    :ptype value: JsonValue
    :param path: Object keys and array indexes, outermost first.
    :ptype path: str | int
    :return: The object the path reaches.
    :rtype: dict[str, JsonValue]
    """
    found = json_at(value, *path)
    assert isinstance(found, dict), f"{path!r} reaches {found!r}, not an object"
    return found


def json_number(value: JsonValue, *path: str | int) -> float:
    """The JSON number at ``path`` below ``value``; anything else there fails, naming it.

    :param value: The JSON to read.
    :ptype value: JsonValue
    :param path: Object keys and array indexes, outermost first.
    :ptype path: str | int
    :return: The number the path reaches.
    :rtype: float
    """
    found = json_at(value, *path)
    assert isinstance(found, (int, float)) and not isinstance(found, bool), f"{path!r} reaches {found!r}, not a number"
    return found
