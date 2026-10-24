"""A reporter template over the toy host's operations host, and a freeze of its corpus campaign into it.

Shared by ``test_reporter_case_actions.py`` and ``test_case_rename.py``: a frozen reporter case is a stored case
of a template beside the toy host's own, which a rename must tell apart by name.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from threetears.evals.analysis import REPORTER_KIND
from threetears.evals.ops import FrozenReporterCase, ReporterCaseFreeze, reporter_case_freeze
from threetears.evals.schema.models import EvalTemplate, RubricDim
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture

GROUNDED = "reporter.groundedness"
USEFUL = "reporter.usefulness"
#: A pass/fail dimension of :func:`scaled_reporter_template`, beside its ordinal :data:`GROUNDED`.
ANSWERS = "reporter.answers_the_question"


def reporter_template(scope_id: str = TOYHOST_SCOPE) -> EvalTemplate:
    """A reporter template scoring two dimensions — what a reporter case is frozen into."""
    return EvalTemplate(
        id="toyhost-reporter",
        scope_id=scope_id,
        name="Memo reporter",
        intent="Judge the analysis memo a campaign got against the evidence its writer read.",
        candidate_kind=REPORTER_KIND,
        rubric=[
            RubricDim(name=GROUNDED, description="Every claim traces to the evidence.", scale="ordinal"),
            RubricDim(name=USEFUL, description="A reader can act on it.", scale="ordinal"),
        ],
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def scaled_reporter_template(scope_id: str = TOYHOST_SCOPE) -> EvalTemplate:
    """A reporter template scoring one ordinal dimension and one pass/fail one, so a label's form can follow either."""
    return EvalTemplate(
        id="toyhost-reporter-scaled",
        scope_id=scope_id,
        name="Memo reporter, two scales",
        intent="Judge the analysis memo a campaign got against the evidence its writer read.",
        candidate_kind=REPORTER_KIND,
        rubric=[
            RubricDim(name=GROUNDED, description="Every claim traces to the evidence.", scale="ordinal"),
            RubricDim(name=ANSWERS, description="The summary answers the question asked.", scale="pass_fail"),
        ],
        created_at=TOYHOST_INSTANT,
        updated_at=TOYHOST_INSTANT,
    )


def reporter_fixture() -> OpsFixture:
    """The toy host's operations fixture with :func:`reporter_template` stored beside its own template."""
    fixture = ops_fixture()
    fixture.host.eval_host.storage.save_template(reporter_template())
    return fixture


#: The reader whose verdicts :func:`freeze_corpus_campaign` relays: a person's own account, which the host attests as
#: itself.
READER = "reviewer@example.com"


def attests_reader(*, scope_id: str, caller: str, person: str) -> str | None:
    """A host's check that attests :data:`READER`'s own account as the reader of its verdicts, and no one else."""
    return person if caller == person == READER else None


def freeze_corpus_campaign(fixture: OpsFixture, scope_id: str = TOYHOST_SCOPE, **fields: Any) -> FrozenReporterCase:
    """Freeze the fixture's corpus campaign into the reporter template, ``fields`` overriding the freeze's own.

    Frozen by :data:`READER` relaying its own verdicts, on a host that attests it for them (:func:`attests_reader`).
    """
    arguments = {"template_id": reporter_template().id, "campaign_id": fixture.campaign.id} | fields
    return reporter_case_freeze(
        replace(fixture.host.eval_host, attests_person=attests_reader),
        ReporterCaseFreeze.model_validate(arguments),
        scope_id,
        collected_by=READER,
    )
