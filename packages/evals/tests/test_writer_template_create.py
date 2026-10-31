"""Writing a report writer template: the writer kind, its scoring rubric seeded from the shipped skill, nothing else.

The rubric is copied into the stored template when it is written, so every campaign on the template is scored on what
the shipped ``editor.md`` stated then. Refused on a host that launches no writer kind, and for a name the scope holds.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.fixtures.toyhost.host import with_kinds
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.writer_case_support import writer_host
from threetears.evals.kernel.errors import ConflictError, ValidationFailedError
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.writer_bank import WriterTemplateCreate, writer_template_create
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND, writer_scoring_rubric

_CREATE = WriterTemplateCreate(template_name="report writer", intent="Write a campaign's report.")


def _writer_launching() -> OpsHost:
    """The toy host with a kind registered under the writer's name. Writing a template reads only whether the
    registry has the kind, so the toy host's own launchable stands in for the writer's."""
    host = writer_host().host
    kinds = dict(host.launch.kinds)
    return OpsHost(launch=with_kinds(host.launch, {**kinds, WRITER_KIND: next(iter(kinds.values()))}))


def test_a_writer_template_is_the_writer_kind_scored_on_the_shipped_rubric() -> None:
    host = _writer_launching()
    line = writer_template_create(host, _CREATE, CALLER.scope_id)
    stored = host.eval_host.storage.load_template(line.id, CALLER.scope_id)
    assert stored is not None
    assert (stored.candidate_kind, stored.name, stored.intent) == (WRITER_KIND, _CREATE.template_name, _CREATE.intent)
    assert stored.rubric == writer_scoring_rubric()


def test_a_name_the_scope_holds_is_refused() -> None:
    host = _writer_launching()
    writer_template_create(host, _CREATE, CALLER.scope_id)
    with pytest.raises(ConflictError):
        writer_template_create(host, _CREATE, CALLER.scope_id)


def test_a_host_launching_no_writer_kind_is_refused() -> None:
    host = writer_host().host
    assert WRITER_KIND not in host.launch.kinds
    with pytest.raises(ValidationFailedError, match="launches no 'report_writer' kind"):
        writer_template_create(host, _CREATE, CALLER.scope_id)
