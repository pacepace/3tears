"""analyze_media over the object-catalog storage: the two real modules, together.

each module passes its own tests; this file pins what they do as a pair. the only
thing replaced is the hub boundary the storage stands on (``resolve_object`` /
``open_object_stream``), so the storage's own category mapping, its uncached
content, and its byte stream are all the production code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import fitz  # PyMuPDF
import pytest
from threetears.media.contracts import ObjectHandle

from threetears.agent.tools import object_media_storage as oms
from threetears.agent.tools.builtin.analyze_media import AnalyzeMediaTool, AnalyzerConfig
from threetears.agent.tools.consume import ConsumeObjectError
from threetears.agent.tools.object_media_storage import ObjectCatalogMediaStorage


# parity-with: threetears.agent.tools.protocols.TextProvider
class _FakeTextProvider:
    """records the prompts it is asked and answers with a canned reply."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


# parity-with: threetears.agent.tools.protocols.ReferenceVisionProvider
class _FakeReferenceVision:
    """a gateway-shaped vision backend: records the object ids it is sent."""

    def __init__(self) -> None:
        self.calls: list[list[UUID]] = []

    async def analyze_ref(self, object_ids: list[UUID], prompt: str) -> str:
        self.calls.append(list(object_ids))
        return "a referenced image"


def _pdf_saying(text: str) -> bytes:
    """build a one-page PDF whose only text is ``text``."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _catalog(
    monkeypatch: pytest.MonkeyPatch,
    objects: dict[UUID, tuple[bytes, str]],
    *,
    streamed: list[UUID] | None = None,
) -> None:
    """stand the hub boundary up over ``objects``; any other id is not the caller's.

    ``streamed``, when given, records the id of every object whose bytes are opened.
    """

    async def _resolve(object_id: UUID) -> ObjectHandle:
        if object_id not in objects:
            raise ConsumeObjectError("not owned by the caller")
        data, mime = objects[object_id]
        return ObjectHandle(object_id=object_id, s3_key=f"k/{object_id}", mime_type=mime, size_bytes=len(data))

    async def _stream(s3_key: str) -> AsyncIterator[bytes]:
        object_id = UUID(s3_key.rsplit("/", 1)[1])
        if streamed is not None:
            streamed.append(object_id)
        yield objects[object_id][0]

    monkeypatch.setattr(oms, "resolve_object", _resolve)
    monkeypatch.setattr(oms, "open_object_stream", _stream)


class TestCataloguedDocumentsAreAnalyzable:
    """a catalogued document is read from its bytes, since the catalog caches no text."""

    @pytest.mark.parametrize(
        ("data", "mime"),
        [
            (_pdf_saying("The quarterly revenue was 42 widgets."), "application/pdf"),
            (b"The quarterly revenue was 42 widgets.", "text/plain"),
        ],
        ids=["pdf", "text"],
    )
    async def test_the_document_text_reaches_the_analyzer(
        self, monkeypatch: pytest.MonkeyPatch, data: bytes, mime: str
    ) -> None:
        """the analyzer is asked about the document's own words, and its answer comes back."""
        mid = uuid4()
        _catalog(monkeypatch, {mid: (data, mime)})
        text = _FakeTextProvider("Revenue was 42 widgets.")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="What was the revenue?", analyzer="Reader")

        assert result.success, result.content
        assert result.content == "Revenue was 42 widgets."
        assert len(text.prompts) == 1
        assert "quarterly revenue was 42 widgets" in text.prompts[0]

    async def test_a_type_no_parser_reads_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a document of a type nothing can read answers with that, not with a model call."""
        mid = uuid4()
        _catalog(monkeypatch, {mid: (b"\x00\x01\x02", "application/octet-stream")})
        text = _FakeTextProvider("never asked")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="Summarize", analyzer="Reader")

        assert not result.success
        assert "application/octet-stream" in result.content
        assert text.prompts == []

    async def test_a_large_object_no_parser_reads_is_never_downloaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a catalogued packet capture is refused from its type, before a byte of it moves.

        the object store exists for artifacts that must never sit whole in a pod's memory, and every
        type that is not image, audio or video is routed here as a document; downloading one only to
        learn no parser reads it could take the tool pod down with nothing logged that names why.
        """
        mid = uuid4()
        streamed: list[UUID] = []
        _catalog(
            monkeypatch,
            {mid: (b"\xd4\xc3\xb2\xa1" * (16 * 1024 * 1024), "application/vnd.tcpdump.pcap")},
            streamed=streamed,
        )
        text = _FakeTextProvider("never asked")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="Summarize", analyzer="Reader")

        assert not result.success
        assert "application/vnd.tcpdump.pcap" in result.content
        assert streamed == []
        assert text.prompts == []

    async def test_a_readable_object_is_downloaded_and_read_within_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a type a parser reads still downloads, and the analyzer still reads only its window."""
        mid = uuid4()
        streamed: list[UUID] = []
        _catalog(monkeypatch, {mid: (b"The quarterly revenue was 42 widgets. " * 50, "text/plain")}, streamed=streamed)
        text = _FakeTextProvider("Revenue was 42 widgets.")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
            doc_max_chars=100,
        )

        result = await tool.execute(media_ids=[str(mid)], question="What was the revenue?", analyzer="Reader")

        assert result.success, result.content
        assert streamed == [mid]
        assert result.content.startswith("Revenue was 42 widgets.")
        assert result.content != "Revenue was 42 widgets.", "the answer says it read only part of the document"
        assert text.prompts[0].count("quarterly revenue") < 50


class TestReferenceVisionSendsOnlyResolvedIds:
    """the gateway is sent only the ids the caller's storage resolved."""

    async def test_an_id_the_caller_does_not_own_never_reaches_the_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an unowned id is dropped before the reference call, the owned one is sent."""
        owned, foreign = uuid4(), uuid4()
        _catalog(monkeypatch, {owned: (b"png", "image/png")})
        vision = _FakeReferenceVision()
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Eye": AnalyzerConfig(name="Eye", vision=vision)},
        )

        result = await tool.execute(media_ids=[str(owned), str(foreign)], question="compare", analyzer="Eye")

        assert result.success, result.content
        assert vision.calls == [[owned]]

    async def test_no_resolved_id_answers_without_calling_the_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """when nothing resolves, the answer is 'No valid media found' and the gateway is never called."""
        _catalog(monkeypatch, {})
        vision = _FakeReferenceVision()
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Eye": AnalyzerConfig(name="Eye", vision=vision)},
        )

        result = await tool.execute(media_ids=[str(uuid4())], question="what is this", analyzer="Eye")

        assert not result.success
        assert "No valid media found" in result.content
        assert vision.calls == []


class _EndlessDocument:
    """a catalogued readable document whose bytes are produced as they are pulled.

    nothing is allocated up front, so a test can stand for an object far larger than the pod could
    hold and still prove how much of it was read: ``pulled`` counts every byte handed to the reader,
    and ``closed`` says whether the reader released the stream.
    """

    def __init__(self, *, catalogued_size: int, chunk_size: int, chunks: int) -> None:
        self.catalogued_size = catalogued_size
        self.chunk_size = chunk_size
        self.chunks = chunks
        self.pulled = 0
        self.opened = 0
        self.closed = False

    def install(self, monkeypatch: pytest.MonkeyPatch, mid: UUID) -> None:
        """stand the hub boundary up over this one object, of type ``text/plain``."""

        async def _resolve(object_id: UUID) -> ObjectHandle:
            if object_id != mid:
                raise ConsumeObjectError("not owned by the caller")
            return ObjectHandle(
                object_id=object_id, s3_key=f"k/{object_id}", mime_type="text/plain", size_bytes=self.catalogued_size
            )

        async def _stream(s3_key: str) -> AsyncIterator[bytes]:
            self.opened += 1
            try:
                for _ in range(self.chunks):
                    self.pulled += self.chunk_size
                    yield b"a" * self.chunk_size
            finally:
                self.closed = True

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        monkeypatch.setattr(oms, "open_object_stream", _stream)


class TestAReadableDocumentIsSizeCapped:
    """a type a parser reads is still never pulled whole into the pod past the document limit."""

    async def test_a_catalogued_size_over_the_limit_is_refused_before_a_byte_moves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the catalog's size answers first: the object is never opened, and the model is told it is too large."""
        from threetears.agent.tools.builtin.analyze_media import MAX_DOCUMENT_BYTES

        mid = uuid4()
        document = _EndlessDocument(catalogued_size=MAX_DOCUMENT_BYTES + 1, chunk_size=1024 * 1024, chunks=64)
        document.install(monkeypatch, mid)
        text = _FakeTextProvider("never asked")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="Summarize", analyzer="Reader")

        assert not result.success
        assert "too large" in result.content
        assert f"{MAX_DOCUMENT_BYTES + 1:,}" in result.content, "the answer names the document's size"
        assert document.opened == 0, "the object store was opened for a document the catalog says is too large"
        assert text.prompts == []

    async def test_an_unknown_size_stops_reading_at_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a catalog that does not know the size (or understates it) cannot make the read unbounded.

        the stream is three times the limit; the read must stop within one chunk past the limit,
        release the stream, and answer that the document is too large.
        """
        from threetears.agent.tools.builtin.analyze_media import MAX_DOCUMENT_BYTES

        mid = uuid4()
        chunk = 1024 * 1024
        document = _EndlessDocument(catalogued_size=0, chunk_size=chunk, chunks=3 * MAX_DOCUMENT_BYTES // chunk)
        document.install(monkeypatch, mid)
        text = _FakeTextProvider("never asked")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="Summarize", analyzer="Reader")

        assert not result.success
        assert "too large" in result.content
        assert document.pulled <= MAX_DOCUMENT_BYTES + chunk, (
            f"the read pulled {document.pulled} bytes; it must stop within one chunk of the {MAX_DOCUMENT_BYTES}-byte limit"
        )
        assert document.closed, "the stream was not released when the read stopped"
        assert text.prompts == []

    async def test_a_document_under_the_limit_is_read_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a normal document, its size catalogued, is downloaded and answered as before."""
        mid = uuid4()
        document = _EndlessDocument(catalogued_size=4 * 1024, chunk_size=1024, chunks=4)
        document.install(monkeypatch, mid)
        text = _FakeTextProvider("It is all a's.")
        tool = AnalyzeMediaTool(
            storage=ObjectCatalogMediaStorage(),
            analyzers={"Reader": AnalyzerConfig(name="Reader", text=text, supported_categories={"document"})},
        )

        result = await tool.execute(media_ids=[str(mid)], question="What is in it?", analyzer="Reader")

        assert result.success, result.content
        assert document.pulled == 4 * 1024
        assert len(text.prompts) == 1
