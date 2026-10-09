"""a tool result's content, compressed for a caller that reads it compressed.

A large answer crosses the bus as a NATS message, which the broker caps (``max_payload``, 16 MiB on
the platform). JSON answers compress tenfold or more, so a caller that serves the bytes on as they are
-- the hub's REST face, answering ``Content-Encoding: gzip`` to a browser or an edge -- asks for them
compressed (:attr:`CallContext.accept_encoding`), and the tool server compresses the success it
answers (:func:`encode_for_caller`). Nothing reads the compressed form but that caller: an agent never
asks, so its model reads the text, and a result a tool compressed itself is decompressed for a caller
that did not ask (:func:`plain_content`).

On the wire the content is base64 text (the envelope is JSON) and ``metadata["content_encoding"]`` is
``"gzip"``; :func:`gzip_bytes` gives the reader back the compressed bytes.
"""

from __future__ import annotations

import base64
import gzip
from typing import Any, Final

from threetears.agent.tools.context_envelope import CallContext

__all__ = [
    "CONTENT_ENCODING_METADATA_KEY",
    "GZIP",
    "GZIP_MIN_BYTES",
    "accepts_gzip",
    "encode_for_caller",
    "gzip_bytes",
    "gzipped_content",
    "plain_content",
]

#: the metadata key naming a result's content encoding
CONTENT_ENCODING_METADATA_KEY: Final = "content_encoding"

#: the one encoding there is
GZIP: Final = "gzip"

#: the smallest content worth compressing: below it the base64 and the gzip header cost more than
#: they save, and nothing is near a limit
GZIP_MIN_BYTES: Final = 1024


def accepts_gzip(context: CallContext | None) -> bool:
    """whether the call's caller reads a result in gzip.

    :param context: the call's context
    :ptype context: CallContext | None
    :return: ``True`` when it asked for gzip
    :rtype: bool
    """
    return context is not None and context.accept_encoding == GZIP


def gzipped_content(compressed: bytes) -> str:
    """the wire form of gzip bytes: base64 text, as a result's content carries it.

    :param compressed: gzip bytes
    :ptype compressed: bytes
    :return: base64 text
    :rtype: str
    """
    return base64.b64encode(compressed).decode("ascii")


def gzip_bytes(content: str, metadata: dict[str, Any] | None) -> bytes | None:
    """the gzip bytes a result carries, or ``None`` when its content is plain text.

    :param content: the result's content
    :ptype content: str
    :param metadata: the result's metadata
    :ptype metadata: dict[str, Any] | None
    :return: the compressed bytes, or ``None``
    :rtype: bytes | None
    """
    if metadata is None or metadata.get(CONTENT_ENCODING_METADATA_KEY) != GZIP:
        return None
    return base64.b64decode(content)


def plain_content(content: str, metadata: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """a result's content and metadata as plain text, decompressing a gzip one.

    :param content: the result's content
    :ptype content: str
    :param metadata: the result's metadata
    :ptype metadata: dict[str, Any] | None
    :return: the text, and the metadata without the encoding
    :rtype: tuple[str, dict[str, Any] | None]
    """
    compressed = gzip_bytes(content, metadata)
    if compressed is None or metadata is None:
        return content, metadata
    rest = {key: value for key, value in metadata.items() if key != CONTENT_ENCODING_METADATA_KEY}
    return gzip.decompress(compressed).decode("utf-8"), rest or None


def encode_for_caller(
    content: str,
    metadata: dict[str, Any] | None,
    context: CallContext | None,
    *,
    min_bytes: int = GZIP_MIN_BYTES,
) -> tuple[str, dict[str, Any] | None]:
    """a success's content and metadata as its caller reads them.

    A caller that asked for gzip gets content of ``min_bytes`` or more compressed (or a result its tool
    compressed itself, as it is). Any other caller gets plain text, a tool's own compression undone. An
    imported API's passthrough result (``metadata["http"]``) replays its upstream body verbatim and is
    left alone.

    :param content: the result's content
    :ptype content: str
    :param metadata: the result's metadata
    :ptype metadata: dict[str, Any] | None
    :param context: the call's context
    :ptype context: CallContext | None
    :param min_bytes: the smallest content compressed
    :ptype min_bytes: int
    :return: the content and metadata to answer with
    :rtype: tuple[str, dict[str, Any] | None]
    """
    if metadata is not None and "http" in metadata:
        return content, metadata
    if not accepts_gzip(context):
        return plain_content(content, metadata)
    if gzip_bytes(content, metadata) is not None:
        return content, metadata
    raw = content.encode("utf-8")
    if len(raw) < min_bytes:
        return content, metadata
    encoded = gzipped_content(gzip.compress(raw, mtime=0))
    return encoded, {**(metadata or {}), CONTENT_ENCODING_METADATA_KEY: GZIP}
