"""A write JetStream refuses for want of room is an ObjectStoreFullError; every other refusal is not.

The scoped snapshot sweeps its store and tries once more only on this error, so the mapping from
JetStream's codes is what its recovery rests on: a code mapped wrongly either never sweeps a full
store or sweeps one that is failing for another reason.
"""

from __future__ import annotations

from typing import Any

import pytest
from nats.js.errors import APIError

from threetears.nats import ObjectStoreError, ObjectStoreFullError
from threetears.nats.object_store import NatsObjectStore


class _JetStream:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def publish(self, subject: str, payload: bytes, *, headers: Any = None) -> None:
        raise self._error


def _store() -> NatsObjectStore:
    return NatsObjectStore(client=object(), full_name="OBJ_pod-objects")  # type: ignore[arg-type]


async def _publish(error: BaseException) -> None:
    await _store()._publish(_JetStream(error), "$O.pod-objects.C.x", b"data", headers=None, name="enr/TX/1/r")  # noqa: SLF001


async def test_a_write_past_max_bytes_is_a_full_store_naming_the_bucket_and_object() -> None:
    error = APIError(code=503, err_code=10077, description="maximum bytes exceeded")
    with pytest.raises(ObjectStoreFullError) as raised:
        await _publish(error)
    assert (raised.value.bucket, raised.value.name) == ("OBJ_pod-objects", "enr/TX/1/r")
    assert "maximum bytes exceeded" in str(raised.value)


async def test_insufficient_server_resources_is_a_full_store_said_so() -> None:
    with pytest.raises(ObjectStoreFullError, match="out of the storage"):
        await _publish(APIError(code=503, err_code=10047, description="insufficient resources"))


@pytest.mark.parametrize(
    ("err_code", "description"),
    [(10077, "maximum messages exceeded"), (10077, "store closed"), (10059, "stream not found")],
)
async def test_any_other_store_failure_is_a_plain_store_error(err_code: int, description: str) -> None:
    with pytest.raises(ObjectStoreError) as raised:
        await _publish(APIError(code=503, err_code=err_code, description=description))
    assert not isinstance(raised.value, ObjectStoreFullError)
    assert description in str(raised.value)
