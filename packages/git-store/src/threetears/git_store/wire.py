"""the bytes a change set travels as, from a process that is not the writer to the one that is.

A row's values keep their types across the wire: a datetime arrives a datetime, a UUID a UUID, a
Decimal a Decimal, so the writer encodes the row to exactly the bytes the sender's own save would
have written, and compares a compare-and-swap version as an instant. Each such value is sent as a
one-key object (``{"$datetime": "..."}``); a row's own one-key objects whose key starts with ``$``
are wrapped (``{"$literal": {...}}``) so they can never be read as a tag.

A refusal comes back as data, not as a raised error, so the sender raises the same typed error the
writer did: :class:`~threetears.git_store.errors.ChangeConflictError` with every conflicted row,
:class:`~threetears.git_store.errors.HandEditPendingError` with every path,
:class:`~threetears.git_store.errors.NotTheWriterError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, cast
from uuid import UUID

from threetears.git_store.change import Change, CommitResult, OnConflict
from threetears.git_store.errors import (
    ChangeConflictError,
    ConflictedRow,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
)
from threetears.git_store.git import Author

__all__ = [
    "ForwardedRequest",
    "decode_read",
    "decode_reply",
    "decode_request",
    "decode_rows",
    "encode_failure",
    "encode_read",
    "encode_request",
    "encode_result",
    "encode_rows",
]

#: the operations a writer runs for another process.
Operation = Literal["commit", "commit_paths", "push"]


def _tag(value: Any) -> Any:
    """a value in its wire form, types kept.

    :param value: the value
    :ptype value: Any
    :return: its JSON-ready form
    :rtype: Any
    :raises TypeError: for a value with no wire form
    """
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, Mapping):
        tagged = {str(k): _tag(v) for k, v in value.items()}
        if len(tagged) == 1 and next(iter(tagged)).startswith("$"):
            return {"$literal": tagged}
        return tagged
    if isinstance(value, list | tuple):
        return [_tag(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"{type(value).__name__} cannot be forwarded in a git-store change")


def _untag(value: Any) -> Any:
    """the value a wire form stands for.

    :param value: the JSON-decoded form
    :ptype value: Any
    :return: the value
    :rtype: Any
    """
    if isinstance(value, dict):
        if len(value) == 1:
            ((key, inner),) = value.items()
            if key == "$datetime":
                return datetime.fromisoformat(inner)
            if key == "$date":
                return date.fromisoformat(inner)
            if key == "$uuid":
                return UUID(inner)
            if key == "$decimal":
                return Decimal(inner)
            if key == "$literal":
                return {k: _untag(v) for k, v in inner.items()}
        return {k: _untag(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_untag(item) for item in value]
    return value


def _author(value: Any) -> Author | None:
    """an author from its wire form.

    :param value: ``{"name", "email"}`` or ``None``
    :ptype value: Any
    :return: the author, or ``None``
    :rtype: Author | None
    """
    return None if value is None else Author(str(value["name"]), str(value["email"]))


@dataclass(frozen=True, slots=True)
class ForwardedRequest:
    """an operation a writer runs for another process.

    :ivar op: which one
    :ivar changes: the change set's writes (``commit``)
    :ivar paths: the files to adopt (``commit_paths``); ``None`` for every changed file
    :ivar message: the commit message
    :ivar author: the commit's author
    :ivar trailers: trailer lines
    :ivar date: the author date
    """

    op: Operation
    changes: tuple[Change, ...] = ()
    paths: tuple[str, ...] | None = None
    message: str = ""
    author: Author | None = None
    trailers: Mapping[str, str] | None = None
    date: datetime | None = None


def encode_request(request: ForwardedRequest) -> bytes:
    """the wire bytes of a request.

    :param request: the request
    :ptype request: ForwardedRequest
    :return: UTF-8 JSON
    :rtype: bytes
    """
    body = {
        "op": request.op,
        "changes": [
            {
                "table": change.table,
                "pk": _tag(dict(change.pk)),
                "row": None if change.row is None else _tag(dict(change.row)),
                "cas": _tag(change.cas),
                "on_conflict": change.on_conflict,
            }
            for change in request.changes
        ],
        "paths": None if request.paths is None else list(request.paths),
        "message": request.message,
        "author": None if request.author is None else {"name": request.author.name, "email": request.author.email},
        "trailers": dict(request.trailers or {}),
        "date": _tag(request.date),
    }
    return json.dumps(body).encode("utf-8")


def decode_request(data: bytes) -> ForwardedRequest:
    """a request from its wire bytes.

    :param data: UTF-8 JSON
    :ptype data: bytes
    :return: the request
    :rtype: ForwardedRequest
    :raises ValueError: for an operation this writer does not run
    """
    body = json.loads(data.decode("utf-8"))
    op = body["op"]
    if op not in ("commit", "commit_paths", "push"):
        raise ValueError(f"a git-store writer does not run {op!r}")
    changes = tuple(
        Change(
            table=str(item["table"]),
            pk=_untag(item["pk"]),
            row=None if item["row"] is None else _untag(item["row"]),
            cas=_untag(item["cas"]),
            on_conflict=cast("OnConflict", item["on_conflict"]),
        )
        for item in body["changes"]
    )
    paths = body.get("paths")
    return ForwardedRequest(
        op=cast("Operation", op),
        changes=changes,
        paths=None if paths is None else tuple(str(path) for path in paths),
        message=str(body.get("message", "")),
        author=_author(body.get("author")),
        trailers={str(k): str(v) for k, v in (body.get("trailers") or {}).items()},
        date=_untag(body.get("date")),
    )


def encode_result(result: CommitResult) -> bytes:
    """the wire bytes of a commit's result.

    :param result: the result
    :ptype result: CommitResult
    :return: UTF-8 JSON
    :rtype: bytes
    """
    return json.dumps({"ok": True, "sha": result.sha, "paths": list(result.paths), "pushed": result.pushed}).encode(
        "utf-8"
    )


def encode_failure(error: GitStoreError | ChangeConflictError) -> bytes:
    """the wire bytes of a refusal the sender raises as the same type.

    :param error: the refusal
    :ptype error: GitStoreError | ChangeConflictError
    :return: UTF-8 JSON
    :rtype: bytes
    """
    detail: dict[str, Any]
    if isinstance(error, ChangeConflictError):
        detail = {
            "type": "conflict",
            "rows": [
                {"table": row.table, "pk": _tag(dict(row.pk)), "reason": row.reason, "expected": _tag(row.expected)}
                for row in error.rows
            ],
        }
    elif isinstance(error, HandEditPendingError):
        detail = {"type": "hand_edit", "paths": list(error.paths)}
    elif isinstance(error, NotTheWriterError):
        detail = {"type": "not_writer", "message": str(error)}
    else:
        detail = {"type": "refused", "message": str(error)}
    return json.dumps({"ok": False, "error": detail}).encode("utf-8")


def decode_reply(data: bytes) -> CommitResult:
    """the result a writer replied with, or its refusal raised as the writer raised it.

    :param data: UTF-8 JSON
    :ptype data: bytes
    :return: the result
    :rtype: CommitResult
    :raises ChangeConflictError: when the writer found rows changed since they were read
    :raises HandEditPendingError: when uncommitted hand edits were in the way
    :raises NotTheWriterError: when the process that answered was no longer the writer
    :raises GitStoreError: for any other refusal
    """
    body = json.loads(data.decode("utf-8"))
    if body.get("ok"):
        return CommitResult(sha=body["sha"], paths=tuple(body["paths"]), pushed=bool(body["pushed"]))
    error = body["error"]
    kind = error["type"]
    if kind == "conflict":
        rows: Sequence[ConflictedRow] = [
            ConflictedRow(
                table=str(row["table"]),
                pk=_untag(row["pk"]),
                reason=row["reason"],
                expected=_untag(row["expected"]),
            )
            for row in error["rows"]
        ]
        raise ChangeConflictError(rows)
    if kind == "hand_edit":
        raise HandEditPendingError([str(path) for path in error["paths"]])
    if kind == "not_writer":
        raise NotTheWriterError(str(error["message"]))
    raise GitStoreError(str(error.get("message", "the writer refused the change")))


def encode_read(kind: str, table: str, key: Mapping[str, Any] | None) -> bytes:
    """a read a pod that isn't the writer asks the writer for (``kind``: ``"fetch"`` by key, ``"scan"`` by filters).

    :param kind: ``"fetch"`` or ``"scan"``
    :ptype kind: str
    :param table: the table
    :ptype table: str
    :param key: the key (fetch) or the equality filters (scan)
    :ptype key: Mapping[str, Any] | None
    :return: the request
    :rtype: bytes
    """
    return json.dumps({"kind": kind, "table": table, "key": _tag(dict(key or {}))}).encode()


def decode_read(data: bytes) -> tuple[str, str, dict[str, Any]]:
    """the read a request asks for.

    :param data: the request
    :ptype data: bytes
    :return: ``(kind, table, key or filters)``
    :rtype: tuple[str, str, dict[str, Any]]
    :raises ValueError: for a request that isn't a read
    """
    body = json.loads(data)
    if body.get("kind") not in ("fetch", "scan") or not isinstance(body.get("table"), str):
        raise ValueError("not a git-store read request")
    return body["kind"], body["table"], _untag(body.get("key") or {})


def encode_rows(rows: list[dict[str, Any]]) -> bytes:
    """rows, as the writer answers a read.

    :param rows: the rows
    :ptype rows: list[dict[str, Any]]
    :return: the reply
    :rtype: bytes
    """
    return json.dumps([_tag(row) for row in rows]).encode()


def decode_rows(data: bytes) -> list[dict[str, Any]]:
    """the rows a read's reply carries.

    :param data: the reply
    :ptype data: bytes
    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    return [_untag(row) for row in json.loads(data)]
