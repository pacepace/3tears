"""A reporter template over the toy host's operations host, and a freeze of its corpus campaign into it.

Shared by ``test_reporter_case_actions.py`` and ``test_case_rename.py``: a frozen reporter case is a stored case
of a template beside the toy host's own, which a rename must tell apart by name.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import REPORTER_KIND
from threetears.evals.ops import FrozenReporterCase, ReporterCaseFreeze, reporter_case_freeze
from threetears.evals.schema.models import EvalTemplate, RubricDim
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture

GROUNDED = "reporter.groundedness"
USEFUL = "reporter.usefulness"


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


def reporter_fixture() -> OpsFixture:
    """The toy host's operations fixture with :func:`reporter_template` stored beside its own template."""
    fixture = ops_fixture()
    fixture.host.eval_host.storage.save_template(reporter_template())
    return fixture


def freeze_corpus_campaign(fixture: OpsFixture, scope_id: str = TOYHOST_SCOPE, **fields: Any) -> FrozenReporterCase:
    """Freeze the fixture's corpus campaign into the reporter template, ``fields`` overriding the freeze's own."""
    arguments = {"template_id": reporter_template().id, "campaign_id": fixture.campaign.id} | fields
    return reporter_case_freeze(fixture.host.eval_host, ReporterCaseFreeze.model_validate(arguments), scope_id)
