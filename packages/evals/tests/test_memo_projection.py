"""Where the authored document lives, and the memo's contract refusing a list of answers keyed by question.

The memo writer authored one document per campaign (:class:`~threetears.evals.kernel.authored.AuthoredCampaignAnalysis`)
and the stored analysis keeps it verbatim. These pin the two projection rules the stored models still hold it to: the
question is answered by the one recommendation and one summary per aspect, so a list of answers keyed by question is off
the contract, and the document is the only carrier of what was authored. The position links inside the document
(``rests_on``, ``invalidates``) are the stored model's own rule and are pinned beside it in ``test_analysis_models.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_analysis
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalAnalysis

#: The analysis the memo writer stored over the toy corpus campaign, as its record holds it.
_STORED_MEMO: Final = Path(__file__).parent / "fixtures" / "retired_writers" / "memo_analysis.json"


class TestTheQuestionIsAnsweredByTheRecommendation:
    """A campaign asks one question, and the memo answered it with one recommendation and a summary per aspect."""

    def test_a_list_of_answers_keyed_by_question_is_refused_as_off_contract(self) -> None:
        record: dict[str, Any] = json.loads(_STORED_MEMO.read_text(encoding="utf-8"))
        document = record["analysis"]["document"]
        assert AuthoredCampaignAnalysis.model_validate(document).recommendation, "the stored memo validates as stored"
        answers = [{"question_id": "q-1", "resolution": "answered", "answer": "Narrow.", "rests_on": [0]}]
        with pytest.raises(ValidationError, match="questions"):
            AuthoredCampaignAnalysis.model_validate({**document, "questions": answers})


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
