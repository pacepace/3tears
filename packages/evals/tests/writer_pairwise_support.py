"""A writer campaign's pages stored as its results' judge evidence, and a scripted pairwise judge that reads them.

The pairwise pass's tests and the promotion read's over it share one page layout and one judge, so a change to how a
page is laid out, or to how the judge reads which slot holds the candidate's page, moves both at once. The judge reads
the document and the slot off the pair it is shown and answers by a policy the test names. No model is called.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.writer_promotion_support import SCORING_JUDGE
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.schema.completion import ProviderFailure
from threetears.evals.schema.models import EvalTemplate, EvalTrace, JudgedArtifact, JudgeEvidence, eval_trace_doc_id

__all__ = [
    "GARBLED",
    "PAIR_CEILING",
    "PAIR_TEMPLATE",
    "REFUSED",
    "AccountRefused",
    "Pages",
    "Policy",
    "ScriptedPairJudge",
    "describe_account_refusal",
    "every_page",
    "pair_page",
    "prefers_report_a",
    "prefers_the_candidate",
    "prefers_the_control",
    "store_pages",
    "ties",
]

#: The template every run is pointed at, so the judge's prompt carries its intent.
PAIR_TEMPLATE = "writer-template"
#: What one judge call is priced at.
PAIR_CEILING = 0.01
#: A policy's answer that makes the judge's call raise, as an account out of credit does.
REFUSED = "refused"
#: A policy's answer the judge sends as text no parse reads.
GARBLED = "garbled"

#: A page by arm, document and repeat: whether the cell stored one.
Pages = Callable[[str, str, int], bool]
#: The judge's answer, given the document and which slot (A or B) holds the candidate's page: a 1-5 score,
#: CANNOT_TELL, or :data:`REFUSED` or :data:`GARBLED`.
Policy = Callable[[str, str], int | str]


def every_page(_arm: str, _document: str, _repeat: int) -> bool:
    """Every cell stored its page."""
    return True


def prefers_the_candidate(_document: str, slot: str) -> int:
    """Report A clearly better when the candidate is A, Report B when it is B."""
    return 1 if slot == "A" else 5


def prefers_the_control(_document: str, slot: str) -> int:
    """The control's page clearly better, in either slot."""
    return 5 if slot == "A" else 1


def ties(_document: str, _slot: str) -> int:
    """No difference a reader would act on."""
    return 3


def prefers_report_a(_document: str, _slot: str) -> int:
    """A judge that always prefers the page it reads first, whoever wrote it."""
    return 1


def pair_page(arm: str, document: str, repeat: int) -> str:
    """The page an arm's cell stored on a document and repeat: its first line names both, which the judge reads."""
    return f"# {arm} page on {document}, repeat {repeat}\n\nThe {arm} arm's findings."


class AccountRefused(Exception):
    """What the scripted judge raises for a refused call."""


def describe_account_refusal(exc: BaseException) -> ProviderFailure:
    """The host's describer: :class:`AccountRefused` is a refusal for the account, anything else is not."""
    return ProviderFailure(
        type(exc).__name__, payload_withheld=True, owner="account" if isinstance(exc, AccountRefused) else "candidate"
    )


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    temperature: float | None = None
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = SCORING_JUDGE
    served_model: str | None = SCORING_JUDGE


@dataclass
class ScriptedPairJudge:
    """The scoring judge: reads which report is the candidate's off the two pages it is shown, answers by ``policy``.

    :ivar calls: Each call's document, the candidate's slot, and Report A's and Report B's pages, in call order.
    """

    policy: Policy = prefers_the_candidate
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)
    model_name: str = SCORING_JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The most one call costs."""
        return PAIR_CEILING

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        """Record which pages it read, in which slots, and answer."""
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        pages = re.search(r"=== Report A ===\n(.*?)\n\n=== Report B ===\n(.*)$", user, re.S)
        assert pages is not None, user
        first, second = pages.group(1), pages.group(2)
        document = re.search(r" page on (\S+),", first).group(1)  # type: ignore[union-attr]
        slot = "A" if first.startswith("# candidate") else "B"
        self.calls.append((document, slot, first, second))
        answer = self.policy(document, slot)
        if answer == REFUSED:
            raise AccountRefused("out of credit")
        if answer == GARBLED:
            return _Completion(content="not json")
        return _Completion(content=json.dumps({"reasoning": "read both", "criteria_scores": {dim: answer}}))

    async def aclose(self) -> None:
        """Nothing to release."""


def store_pages(host: EvalHost, campaign_id: str, pages: Pages = every_page) -> None:
    """Point the campaign's two runs (control first) at :data:`PAIR_TEMPLATE` and store each result's page as its
    judge evidence, where ``pages`` says the cell stored one. Blocking."""
    storage = host.storage
    storage.save_template(
        EvalTemplate(
            id=PAIR_TEMPLATE,
            scope_id=TOYHOST_SCOPE,
            name="writer",
            intent="Write the campaign's report.",
            candidate_kind=WRITER_KIND,
        )
    )
    campaign = storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert campaign is not None
    for arm, run_id in zip(("control", "candidate"), campaign.run_ids, strict=True):
        run = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert run is not None
        storage.save_eval_run(run.model_copy(update={"template_id": PAIR_TEMPLATE}))
        for result in storage.query_eval_results(TOYHOST_SCOPE, run_id=run_id):
            if not pages(arm, result.test_case_id, result.k_iteration):
                continue
            evidence = JudgeEvidence(
                case_material=f"Computed facts of {result.test_case_id}.",
                artifact=pair_page(arm, result.test_case_id, result.k_iteration),
            )
            trace = EvalTrace(
                id=eval_trace_doc_id(result.id),
                scope_id=TOYHOST_SCOPE,
                result_id=result.id,
                eval_run_id=run_id,
                judge_evidence=evidence,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            storage.save_eval_result(result, trace)
