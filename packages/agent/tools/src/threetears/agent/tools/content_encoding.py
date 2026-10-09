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

**Every decode is guarded and bounded.** A declared encoding the bytes do not match (bad base64, a
corrupt or truncated gzip, text that is not UTF-8) raises :class:`ContentEncodingError`, and an answer
that would decode past ``max_bytes`` raises :class:`DecodedTooLargeError` before it is held: JSON
compresses tenfold to a hundredfold, so the bus's limit on the compressed size bounds nothing about the
text. :func:`body_for_client` is the one place a server decides what a client gets: the gzip bytes for
one that reads gzip, the text, decoded within the bound, for one that does not.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import zlib
from typing import Any, Final

from threetears.agent.tools.context_envelope import CallContext

__all__ = [
    "CONTENT_ENCODING_METADATA_KEY",
    "DECODED_MAX_BYTES",
    "GZIP",
    "GZIP_MIN_BYTES",
    "ContentEncodingError",
    "DecodedTooLargeError",
    "accepts_gzip",
    "body_for_client",
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

#: the most a compressed result may decode to: four times the bus's 16 MiB, which no answer a page
#: can draw comes near, and small enough that a server decoding for a client cannot be made to hold
#: hundreds of megabytes per request
DECODED_MAX_BYTES: Final = 64 * 1024 * 1024


class ContentEncodingError(ValueError):
    """a result's content does not decode as its metadata declares."""


class DecodedTooLargeError(ContentEncodingError):
    """a result's content would decode past the bound it is read under."""


def _gunzip(compressed: bytes, max_bytes: int) -> bytes:
    """gzip bytes decoded, never past ``max_bytes``.

    :param compressed: gzip bytes
    :ptype compressed: bytes
    :param max_bytes: the most the decoded bytes may be
    :ptype max_bytes: int
    :return: the decoded bytes
    :rtype: bytes
    :raises DecodedTooLargeError: when they would be more than ``max_bytes``
    :raises ContentEncodingError: when the bytes are not one whole gzip member
    """
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decoded = decoder.decompress(compressed, max_bytes + 1)
    except zlib.error as exc:
        raise ContentEncodingError(f"the content is declared gzip and is not: {exc}") from exc
    if len(decoded) > max_bytes or decoder.unconsumed_tail:
        raise DecodedTooLargeError(f"the content decodes to more than {max_bytes} bytes")
    if not decoder.eof:
        raise ContentEncodingError("the content is declared gzip and ends before its gzip member does")
    return decoded


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
    :raises ContentEncodingError: when the content is declared gzip and is not base64
    """
    if metadata is None or metadata.get(CONTENT_ENCODING_METADATA_KEY) != GZIP:
        return None
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContentEncodingError(f"the content is declared gzip and is not base64: {exc}") from exc


def plain_content(
    content: str, metadata: dict[str, Any] | None, *, max_bytes: int = DECODED_MAX_BYTES
) -> tuple[str, dict[str, Any] | None]:
    """a result's content and metadata as plain text, decompressing a gzip one within ``max_bytes``.

    :param content: the result's content
    :ptype content: str
    :param metadata: the result's metadata
    :ptype metadata: dict[str, Any] | None
    :param max_bytes: the most the text may be
    :ptype max_bytes: int
    :return: the text, and the metadata without the encoding
    :rtype: tuple[str, dict[str, Any] | None]
    :raises DecodedTooLargeError: when the text would be more than ``max_bytes``
    :raises ContentEncodingError: when the content does not decode as declared
    """
    compressed = gzip_bytes(content, metadata)
    if compressed is None or metadata is None:
        return content, metadata
    rest = {key: value for key, value in metadata.items() if key != CONTENT_ENCODING_METADATA_KEY}
    try:
        text = _gunzip(compressed, max_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContentEncodingError(f"the content decodes to bytes that are not UTF-8: {exc}") from exc
    return text, rest or None


def body_for_client(
    content: str, metadata: dict[str, Any] | None, *, client_accepts_gzip: bool, max_bytes: int = DECODED_MAX_BYTES
) -> tuple[bytes, bool]:
    """what a server sends a client for a result: the gzip bytes as they came, or the text.

    The one place the choice is made: a compressed result goes to a client that reads gzip as it is,
    and to any other as its text, decoded within ``max_bytes``; a plain result goes as its text.

    :param content: the result's content
    :ptype content: str
    :param metadata: the result's metadata
    :ptype metadata: dict[str, Any] | None
    :param client_accepts_gzip: whether the client reads a gzip body
    :ptype client_accepts_gzip: bool
    :param max_bytes: the most a decoded body may be
    :ptype max_bytes: int
    :return: the body, and whether it is gzip
    :rtype: tuple[bytes, bool]
    :raises DecodedTooLargeError: when a body decoded for the client would be more than ``max_bytes``
    :raises ContentEncodingError: when the content does not decode as declared
    """
    compressed = gzip_bytes(content, metadata)
    if compressed is not None and client_accepts_gzip:
        return compressed, True
    text, _ = plain_content(content, metadata, max_bytes=max_bytes)
    return text.encode("utf-8"), False


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
