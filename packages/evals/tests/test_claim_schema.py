"""The stored shapes of claim grading (#1287, C1): what a template, a trace and a result may carry, and what they refuse.

**What carries none of it stores, digests and fingerprints exactly as before.** Pinned first, on the build before the
shapes existed: a template's stored document and both its digests, a judge evidence's fingerprint, and a result's
and a trace's stored documents, each less the minor stamp every minor moves.
"""

from __future__ import annotations

from packages.evals.tests.clock_support import FIXED_AT
from packages.evals.tests.factories import make_eval_result, make_eval_trace, make_template
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from threetears.evals.run.definition_digests import measured_template_digest, template_digest
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    EvalResult,
    EvalTemplate,
    EvalTrace,
    JudgedArtifact,
    JudgeEvidence,
    fingerprint_judged_output,
)

#: Taken at ``d1406de56``, the build before claim grading's stored shapes: what a template, an evidence, a result and
#: a trace carrying none of them stored, digested and fingerprinted as. Each must hold after the shapes land.
BEFORE_CLAIMS: dict[str, str] = {
    "template document": "521be221c821e4609d21a35d806fb2152e6f99e2c996a5142de54b935b739153",
    "template_digest": "5792093b03b534d9c03a7bd9324530fce57078062b3fd96083e3242ac2b90c68",
    "measured_template_digest": "c06edf75f254582dfe92d90f1ab694bff90966f147f77fd3e7d96c286a42bfc6",
    "toy template document": "167c4f4759493f4d672fde7b4d6199eeab06593b77e7b89a191300ca79751aa4",
    "toy template_digest": "c7a5125e050278830af521c5e4ff3ec3733dc4107f91b2f161d87692dfd67c0b",
    "toy measured_template_digest": "012be363cc55443558089e29a4048801c77ac185c3027508314c6b81e514fd9b",
    "fingerprint_judged_output": "98b3e32cccb43e7bc6a5e8aa1494fb6da47857dc83784bed7f427c2429e8e95c",
    "trace document": "81b795bc7eb75d906e9e5ab45fd8f6d6cb558192b0f877ed308bab7bcef6485b",
    "result document": "08eb3555a8aa6a350bd2b3484d20fafd9d85dcc8c01d054692378a2571cf3235",
}


def _template() -> EvalTemplate:
    """A template with a rubric, fixed in every field a digest or document reads."""
    return make_template(id="tpl-claims-pin", created_at=FIXED_AT, updated_at=FIXED_AT)


def _evidence() -> JudgeEvidence:
    return JudgeEvidence(
        subject="A research assistant.", case_material="  The source, as given.\n", artifact="The answer.\n"
    )


def _trace() -> EvalTrace:
    return make_eval_trace(
        judge_evidence=_evidence(),
        judged_artifact=JudgedArtifact.DOCUMENT,
    )


def _result() -> EvalResult:
    return make_eval_result(id="res-1", scored_at=FIXED_AT)


def _stored(document: dict[str, object]) -> str:
    """The digest of a stored document less its minor stamp, which every minor moves on every document by design."""
    return canonical_digest({key: value for key, value in document.items() if key != "schema_minor"})


def _taken() -> dict[str, str]:
    template, toy = _template(), toyhost_template()
    return {
        "template document": _stored(template.to_dict()),
        "template_digest": template_digest(template),
        "measured_template_digest": measured_template_digest(template),
        "toy template document": _stored(toy.to_dict()),
        "toy template_digest": template_digest(toy),
        "toy measured_template_digest": measured_template_digest(toy),
        "fingerprint_judged_output": fingerprint_judged_output(_evidence()),
        "trace document": _stored(_trace().to_dict()),
        "result document": _stored(_result().to_dict()),
    }


def test_what_carries_no_claim_shape_stores_digests_and_fingerprints_as_before() -> None:
    assert _taken() == BEFORE_CLAIMS
