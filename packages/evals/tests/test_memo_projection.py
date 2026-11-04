"""Where a stored analysis keeps what its retired memo writer authored, and the fields no analysis carries.

The memo writer authored one document per campaign, and the stored analysis keeps it as the JSON it was stored as
(``EvalAnalysis.document``), never validated as a memo and never read: so the stored memo over the toy corpus campaign
reads back byte for byte, a key off the retired contract included. The document is the only carrier of what was
authored: an analysis without the field does not construct, the fields from before it are gone, and a stored read
refuses an old top-level copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_analysis
from threetears.evals.kernel.campaign import EvalAnalysis

#: The analysis the memo writer stored over the toy corpus campaign, as its record holds it.
_STORED_MEMO: Final = Path(__file__).parent / "fixtures" / "retired_writers" / "memo_analysis.json"


class TestTheStoredMemoIsHeldAsStored:
    """A retired memo is JSON the analysis holds, never a model it validates."""

    def test_the_stored_memo_reads_back_as_it_was_stored_whatever_it_holds(self) -> None:
        record: dict[str, Any] = json.loads(_STORED_MEMO.read_text(encoding="utf-8"))
        document = record["analysis"]["document"]
        analysis = EvalAnalysis.model_validate(record["analysis"])
        assert (analysis.basis, analysis.document) == ("memo", document)
        answers = [{"question_id": "q-1", "resolution": "answered", "answer": "Narrow.", "rests_on": [0]}]
        held = EvalAnalysis.model_validate({**record["analysis"], "document": {**document, "questions": answers}})
        assert held.document is not None and held.document["questions"] == answers


class TestTheDocumentIsTheOnlyCarrier:
    """One home for what the memo writer authored, so nothing can hold a second copy."""

    def test_an_analysis_without_a_document_does_not_construct(self):
        """An analysis with nothing to read is unrepresentable."""
        fields = make_analysis().model_dump()
        del fields["document"]
        with pytest.raises(ValidationError, match="document"):
            EvalAnalysis(**fields)

    @pytest.mark.parametrize(
        "field", ["memo", "findings", "methodology", "methods_appendix", "bluf", "decisions", "questions", "bundle"]
    )
    def test_the_pre_document_fields_are_gone_from_the_model(self, field):
        """A ratchet, not a restatement — the removal is total only while nothing re-adds them."""
        assert field not in EvalAnalysis.model_fields

    def test_a_stored_top_level_copy_is_refused_rather_than_carried(self):
        """A document written with an old top-level field does not load at all.

        An old copy cannot return through storage and sit beside the document's own — the two-homes
        state the document exists to make unreachable — because a stored read refuses an unknown key.
        """
        raw = make_analysis().to_dict()
        raw["memo"] = {"bluf": {"title": "Something else entirely."}}
        with pytest.raises(ValidationError, match="memo"):
            EvalAnalysis.from_dict(raw)
