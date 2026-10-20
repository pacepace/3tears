"""``admit_template`` is on the public path, so a host test calls one template check without ``create_template``."""

from __future__ import annotations

import dataclasses

import pytest

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.run import admit_template
from packages.evals.tests.fixtures.toyhost.host import TOYHOST_TEMPLATE_CHECKS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template


def _refuse(_checked: object) -> None:
    raise ValidationFailedError("the host's own check refused", remedy="state what the host's world can seed")


def test_a_host_check_is_called_directly_through_the_public_name() -> None:
    template = toyhost_template()
    checks = dataclasses.replace(TOYHOST_TEMPLATE_CHECKS, refuse_undeliverable_template=_refuse)

    admitted = admit_template(
        template, profile=toyhost_profile(), checks=TOYHOST_TEMPLATE_CHECKS, active_judge_config=lambda _dim: None
    )
    assert admitted.id == template.id
    with pytest.raises(ValidationFailedError, match="the host's own check refused"):
        admit_template(template, profile=toyhost_profile(), checks=checks, active_judge_config=lambda _dim: None)
