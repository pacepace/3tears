"""Protocols for media-related tool capabilities.

These define the contracts that host applications implement. Tool
implementations can depend on these protocols without coupling to
any specific infrastructure (S3, specific vision APIs, etc.).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "EXTRACTION_STATUS_COMPLETE",
    "EXTRACTION_STATUS_FAILED",
    "EXTRACTION_STATUS_NONE",
    "EXTRACTION_STATUS_PENDING",
    "EXTRACTION_STATUS_REFUSED",
    "EXTRACTION_STATUS_UNCHANGED",
    "OBJECT_HANDLE_METADATA_KEY",
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaInfo",
    "MediaStorage",
    "ObjectHandle",
    "ObjectListing",
    "ObjectStore",
    "TextProvider",
    "TranscriptionProvider",
    "VisionProvider",
]

# ``MediaInfo.extraction_status``: the spellings below are the ones already
# in the database, and that is deliberately where this vocabulary is
# defined from. ``agent-memory``'s v021 migration declares the column
# ``TEXT NOT NULL DEFAULT 'none'`` and v022 builds a partial index on
# ``WHERE extraction_status = 'pending'`` -- a spelling sitting in a column
# default and an index predicate is changed by a migration and an index
# rebuild, not by editing a constant. So these constants name what is
# stored; they do not propose it (docs/search-spec.md §3.5).
#
# The field stays ``str | None``. It is not narrowed to a ``Literal`` or a
# ``StrEnum``: consumers assign and compare bare ``str`` today, and the
# column carries no CHECK constraint, so a producer may legitimately store
# a value this list has not caught up with. Read an unrecognised status the
# way :mod:`threetears.media.contracts.facets` reads an unrecognised facet
# -- ignore it, do not reject it.

#: No extraction has been attempted. The column's server default, and the
#: value a row carries until something asks for extraction.
EXTRACTION_STATUS_NONE = "none"

#: Extraction has been requested and has not finished. The value v022's
#: partial index is built on, which is what makes it a work queue.
EXTRACTION_STATUS_PENDING = "pending"

#: Extraction finished and the content is available.
EXTRACTION_STATUS_COMPLETE = "complete"

#: Extraction was attempted and did not produce content -- the fetch died,
#: the carrier was unreadable, the extractor gave up. Distinct from
#: :data:`EXTRACTION_STATUS_REFUSED`: something tried and failed.
EXTRACTION_STATUS_FAILED = "failed"

#: Extraction was declined before it was attempted -- ``robots.txt``
#: disallowed the fetch, a required extra was absent, a cap refused the
#: read. Distinct from :data:`EXTRACTION_STATUS_FAILED`: nothing tried, and
#: retrying under the same rules will decline again.
EXTRACTION_STATUS_REFUSED = "refused"

#: Upstream confirmed the caller's existing copy is still current, so no
#: content was produced and none was needed -- an HTTP ``304`` answering a
#: conditional request (D30 / SR-M4). Distinct from every status above it:
#: it is a **success**, unlike :data:`EXTRACTION_STATUS_FAILED`; something
#: did try, unlike :data:`EXTRACTION_STATUS_REFUSED`; and it produced no new
#: content, unlike :data:`EXTRACTION_STATUS_COMPLETE`.
#:
#: A reader that treats "no content" as failure will misread this. The
#: correct reading is that the caller's own copy is the content, and it has
#: just been re-confirmed.
EXTRACTION_STATUS_UNCHANGED = "unchanged"


@dataclass
class GeneratedImage:
    """Result from an image generation backend."""

    data: bytes
    mime_type: str
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MediaInfo:
    """Metadata about a media item, returned by :meth:`MediaStorage.get_media`.

    ``extraction_status`` carries one of the ``EXTRACTION_STATUS_*``
    constants in this module, or ``None``. ``None`` and
    :data:`EXTRACTION_STATUS_NONE` both mean *no extraction attempted*:
    the dataclass defaults to ``None`` while the database column is
    ``NOT NULL DEFAULT 'none'``, so which one a reader sees depends on
    whether the value came from a constructor or a row. That split is
    recorded rather than reconciled -- collapsing it is a migration, not a
    contract edit -- so a consumer testing for "nothing has happened yet"
    must accept both.
    """

    media_id: UUID
    media_category: str  # "image", "audio", "video", "document"
    mime_type: str
    extraction_status: str | None = None
    has_downloadable_data: bool = True


@runtime_checkable
class ImageGenerationBackend(Protocol):
    """Protocol for image generation backends."""

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        :param prompt: text description of the image to generate
        :ptype prompt: str
        :param style: optional style modifier
        :ptype style: str | None
        :param source_image: optional source image for img2img
        :ptype source_image: bytes | None
        :param source_mime_type: MIME type of source image
        :ptype source_mime_type: str | None
        :return: generated image result
        :rtype: GeneratedImage
        """
        ...


@runtime_checkable
class MediaStorage(Protocol):
    """Protocol for media item access and storage.

    Host applications implement this to bridge their specific storage
    infrastructure (S3, database, etc.) to the generic tool interface.
    """

    async def get_media(
        self,
        media_id: UUID,
    ) -> MediaInfo | None:
        """Look up media metadata by ID.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :return: media info or None
        :rtype: MediaInfo | None
        """
        ...

    async def download_media(
        self,
        media_id: UUID,
    ) -> tuple[bytes, str] | None:
        """Download raw media bytes.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :return: (data, mime_type) or None if unavailable
        :rtype: tuple[bytes, str] | None
        """
        ...

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
        *,
        model_name: str | None = None,
    ) -> str | None:
        """Retrieve extracted content for a media item.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :param content_type: type of content to retrieve
        :ptype content_type: str
        :param model_name: optional model name filter
        :ptype model_name: str | None
        :return: extracted content or None
        :rtype: str | None
        """
        ...

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store extracted content for a media item.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :param user_id: user who triggered extraction
        :ptype user_id: UUID
        :param content_type: type of content being stored
        :ptype content_type: str
        :param content: extracted content text
        :ptype content: str
        :param metadata: optional metadata
        :ptype metadata: dict[str, Any] | None
        :return: content ID
        :rtype: str
        """
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for image analysis via a vision-capable model."""

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        """Analyze an image and return a text description.

        :param image_data: raw image bytes
        :ptype image_data: bytes
        :param mime_type: image MIME type
        :ptype mime_type: str
        :param prompt: analysis prompt
        :ptype prompt: str
        :return: analysis result text
        :rtype: str
        """
        ...


@runtime_checkable
class TextProvider(Protocol):
    """Protocol for text-only queries (document QA, summarization).

    Separate from :class:`VisionProvider` so that apps can use different
    models for image analysis vs. text reasoning.  Apps that don't
    handle documents can ignore this entirely.
    """

    async def answer(
        self,
        prompt: str,
    ) -> str:
        """Answer a text prompt.

        :param prompt: text prompt
        :ptype prompt: str
        :return: answer text
        :rtype: str
        """
        ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Protocol for audio/video transcription."""

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str,
    ) -> str:
        """Transcribe audio data to text.

        :param audio_data: raw audio bytes
        :ptype audio_data: bytes
        :param mime_type: audio MIME type
        :ptype mime_type: str
        :return: transcription text
        :rtype: str
        """
        ...


#: the key under which a producing tool places its :class:`ObjectHandle`
#: (as :meth:`ObjectHandle.to_metadata`) in ``ToolResult.metadata`` so the
#: agent's catalog seam can recognise + persist it in the object catalog.
OBJECT_HANDLE_METADATA_KEY = "object_handle"


@dataclass
class ObjectHandle:
    """Handle to a stored object -- the small descriptor that crosses NATS
    in place of the bytes.

    The producing tool returns this (in ``ToolResult.metadata`` under
    :data:`OBJECT_HANDLE_METADATA_KEY`); the agent catalogs it into the
    hub-owned ``objects`` catalog; consumers resolve ``object_id`` back to
    ``s3_key`` (customer-safe, pod-side) and stream the bytes down. The bytes
    themselves never travel with the handle.
    """

    object_id: UUID
    s3_key: str
    mime_type: str
    size_bytes: int
    summary: str | None = None
    category: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Project the handle to a JSON-safe dict for ``ToolResult.metadata``.

        UUIDs are stringified at this border so the descriptor survives the
        NATS/JSON round-trip to the agent intact.

        :return: a JSON-safe representation of this handle
        :rtype: dict[str, Any]
        """
        return {
            "object_id": str(self.object_id),
            "s3_key": self.s3_key,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "summary": self.summary,
            "category": self.category,
        }

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> ObjectHandle:
        """Reconstruct a handle from its :meth:`to_metadata` dict.

        :param data: the JSON-safe handle dict (as produced by
            :meth:`to_metadata`)
        :ptype data: dict[str, Any]
        :return: the reconstructed handle
        :rtype: ObjectHandle
        :raises KeyError: when a required field is absent
        :raises ValueError: when ``object_id`` is not a valid UUID
        """
        return cls(
            object_id=UUID(str(data["object_id"])),
            s3_key=str(data["s3_key"]),
            mime_type=str(data["mime_type"]),
            size_bytes=int(data["size_bytes"]),
            summary=data.get("summary"),
            category=data.get("category"),
        )


@dataclass
class ObjectListing:
    """One entry from an object-store listing: key plus server metadata.

    Yielded by :meth:`ObjectStore.list_entries` so a reconciler can decide
    orphan-eligibility by age (``last_modified``) without a per-key HEAD
    round-trip. ``last_modified`` is the store's own last-modified timestamp
    (timezone-aware); ``size_bytes`` is the stored object size.
    """

    key: str
    last_modified: datetime
    size_bytes: int


@runtime_checkable
class ObjectStore(Protocol):
    """Streaming S3-compatible store for large binary artifacts.

    Host apps / the SDK implement this over their object-store
    infrastructure. STREAMING by contract: writes consume an async byte
    stream and reads yield one, so a multi-GB artifact (pcap, db dump,
    rendered report) never has to sit whole in a pod's memory. The key is
    opaque here; the platform's tenant-scoped key scheme is built above
    this contract.
    """

    async def put(
        self,
        key: str,
        body: AsyncIterator[bytes],
        *,
        content_type: str,
        size: int | None = None,
    ) -> None:
        """Stream ``body`` to ``key`` (multipart for large objects).

        :param key: tenant-scoped object key (opaque to this contract)
        :ptype key: str
        :param body: async iterator yielding the object's bytes in chunks
        :ptype body: AsyncIterator[bytes]
        :param content_type: MIME type stored on the object
        :ptype content_type: str
        :param size: total byte length when known (lets the impl pick a
            single PUT below the multipart threshold); None streams multipart
        :ptype size: int | None
        :return: nothing
        :rtype: None
        """
        ...

    def open_read(self, key: str) -> AsyncIterator[bytes]:
        """Open ``key`` for streaming read, yielding bytes in chunks.

        :param key: object key
        :ptype key: str
        :return: async iterator over the object's bytes
        :rtype: AsyncIterator[bytes]
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete a single object.

        :param key: object key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        ...

    async def delete_many(self, keys: list[str]) -> None:
        """Delete many objects in one batched request (reconciler sweep).

        :param keys: object keys to delete
        :ptype keys: list[str]
        :return: nothing
        :rtype: None
        """
        ...

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        """Yield object keys, optionally restricted to ``prefix``.

        :param prefix: key-prefix filter (e.g. a tenant's ``<customer_id>/``);
            None lists the whole bucket
        :ptype prefix: str | None
        :return: async iterator over object keys
        :rtype: AsyncIterator[str]
        """
        ...

    def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:
        """Yield object listings (key + last-modified + size), optionally by ``prefix``.

        Like :meth:`list_keys` but carries each object's server metadata so a
        reconciler can judge orphan age without a per-key HEAD request.

        :param prefix: key-prefix filter (e.g. a tenant's ``<customer_id>/``);
            None lists the whole bucket
        :ptype prefix: str | None
        :return: async iterator over object listings
        :rtype: AsyncIterator[ObjectListing]
        """
        ...

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        """Presigned GET URL for delivery -- bytes never cross the agent.

        :param key: object key
        :ptype key: str
        :param expires_in: URL validity in seconds
        :ptype expires_in: int
        :return: presigned URL
        :rtype: str
        """
        ...
