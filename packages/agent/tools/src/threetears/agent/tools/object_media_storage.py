"""A :class:`MediaStorage` backed by the produced-object catalog.

Bridges the object catalog (an ``object_id`` addresses a stored artifact) to the
media vocabulary :mod:`~threetears.agent.tools.builtin.analyze_media` speaks (a
``media_id`` addresses a media item). The two are the same id: a catalogued
image IS a media item, so ``media_id == object_id`` and no separate media table
is needed.

Everything is tenant-safe and scope-driven: :func:`resolve_object` and
:func:`open_object_stream` read the verified caller identity off the current
:class:`~threetears.agent.tools.call_scope.ToolCallScope` and let the hub
authorize the id against the owning customer, so this storage holds no
credentials and trusts no request field.

The image path is reference-first: a :class:`ReferenceVisionProvider` never
calls :meth:`download_media`, so a vision question streams no bytes to the pod.
:meth:`download_media` exists for the byte-taking fallback (a document or audio
analyzer, or a bytes-only vision backend). Extracted-content caching
(:meth:`get_content` / :meth:`store_content`) is not backed by the object
catalog -- it has no content column -- so descriptions are recomputed rather
than cached; both methods are honest no-ops documented as such.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.media.contracts import MediaInfo
from threetears.observe import get_logger, traced

from threetears.agent.tools.consume import ConsumeObjectError, open_object_stream, resolve_object
from threetears.agent.tools.object_resolver import ResolveObjectError

log = get_logger(__name__)

__all__ = ["ObjectCatalogMediaStorage", "media_category_for_mime"]


def media_category_for_mime(mime_type: str) -> str:
    """map a MIME type to the media category analyze_media routes on.

    :param mime_type: stored MIME type (e.g. "image/png", "application/pdf")
    :ptype mime_type: str
    :return: one of "image", "audio", "video", "document"
    :rtype: str
    """
    lowered = mime_type.lower()
    if lowered.startswith("image/"):
        result = "image"
    elif lowered.startswith("audio/"):
        result = "audio"
    elif lowered.startswith("video/"):
        result = "video"
    else:
        result = "document"
    return result


class ObjectCatalogMediaStorage:
    """MediaStorage over the object catalog: media_id is the object id."""

    @traced
    async def get_media(self, media_id: UUID) -> MediaInfo | None:
        """resolve an object id to its media metadata, tenant-safely.

        the object id is authorized against the verified customer inside
        :func:`resolve_object`; the category is inferred from the stored MIME
        type. returns ``None`` when the object is not owned by the caller or
        does not exist (resolve raising is treated as absence).

        :param media_id: the media/object UUID to look up
        :ptype media_id: UUID
        :return: media info, or ``None`` when absent / not owned
        :rtype: MediaInfo | None
        """
        result: MediaInfo | None = None
        try:
            handle = await resolve_object(media_id)
        except (ConsumeObjectError, ResolveObjectError) as exc:
            log.info(
                "media not resolvable for caller",
                extra={"extra_data": {"media_id": str(media_id), "reason": type(exc).__name__}},
            )
        else:
            result = MediaInfo(
                media_id=media_id,
                media_category=media_category_for_mime(handle.mime_type),
                mime_type=handle.mime_type,
                extraction_status=None,
                has_downloadable_data=True,
            )
        return result

    @traced
    async def download_media(self, media_id: UUID) -> tuple[bytes, str] | None:
        """stream a stored object's bytes for the byte-taking analysis paths.

        the reference vision path never calls this; it is the fallback for a
        document / audio analyzer or a bytes-only vision backend. the bytes
        stream from the store to this pod and never cross NATS.

        :param media_id: the media/object UUID to download
        :ptype media_id: UUID
        :return: ``(data, mime_type)``, or ``None`` when absent / not owned
        :rtype: tuple[bytes, str] | None
        """
        result: tuple[bytes, str] | None = None
        try:
            handle = await resolve_object(media_id)
            chunks = [chunk async for chunk in open_object_stream(handle.s3_key)]
        except (ConsumeObjectError, ResolveObjectError) as exc:
            log.info(
                "media not downloadable for caller",
                extra={"extra_data": {"media_id": str(media_id), "reason": type(exc).__name__}},
            )
        else:
            result = (b"".join(chunks), handle.mime_type)
        return result

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
        *,
        model_name: str | None = None,
    ) -> str | None:
        """always a cache miss: the object catalog stores no extracted content.

        analyze_media treats ``None`` as "not cached" and recomputes, which on
        the reference vision path is a single Gateway call.

        :param media_id: media/object UUID (unused; no content store)
        :ptype media_id: UUID
        :param content_type: content type requested (unused)
        :ptype content_type: str
        :param model_name: optional model filter (unused)
        :ptype model_name: str | None
        :return: always ``None``
        :rtype: str | None
        """
        return None

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """no-op persistence: the object catalog has no content column.

        returns a deterministic content id so the ``str`` contract holds, but
        stores nothing -- descriptions are recomputed rather than cached. honest
        no-op rather than a silent lie: the docstring and the returned id's
        ``uncached:`` prefix both say so.

        :param media_id: media/object UUID the content describes
        :ptype media_id: UUID
        :param user_id: user who triggered extraction (unused)
        :ptype user_id: UUID
        :param content_type: content type being stored
        :ptype content_type: str
        :param content: extracted content (unused; not persisted)
        :ptype content: str
        :param metadata: optional metadata (unused)
        :ptype metadata: dict[str, Any] | None
        :return: a deterministic, non-persisted content id
        :rtype: str
        """
        return f"uncached:{media_id}:{content_type}"
