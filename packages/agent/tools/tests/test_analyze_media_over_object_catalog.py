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


def _catalog(monkeypatch: pytest.MonkeyPatch, objects: dict[UUID, tuple[bytes, str]]) -> None:
    """stand the hub boundary up over ``objects``; any other id is not the caller's."""

    async def _resolve(object_id: UUID) -> ObjectHandle:
        if object_id not in objects:
            raise ConsumeObjectError("not owned by the caller")
        data, mime = objects[object_id]
        return ObjectHandle(object_id=object_id, s3_key=f"k/{object_id}", mime_type=mime, size_bytes=len(data))

    async def _stream(s3_key: str) -> AsyncIterator[bytes]:
        object_id = UUID(s3_key.rsplit("/", 1)[1])
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
