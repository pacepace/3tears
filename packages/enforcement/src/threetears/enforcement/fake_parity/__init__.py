"""fake-protocol-parity enforcement domain.

every test fake (``Fake<Name>`` / ``_Fake<Name>``) must declare what
production protocol it stands in for, via subclass declaration or a
``# parity-with: <fully.qualified.name>`` marker comment. this domain
catches fakes whose method surface diverges from production -- a
class of bug that bit the typed-NATS migration when fakes silently
rotted behind protocol changes and only failed the consumer suite
far downstream.

per-repo configuration goes through :class:`FakeParityConfig`;
:func:`run_fake_parity_enforcement` is the pytest-friendly entry
point that runs the walker, applies exemptions, emits the report,
and fails in strict mode.
"""

from threetears.enforcement.fake_parity.config import FakeParityConfig
from threetears.enforcement.fake_parity.runner import run_fake_parity_enforcement
from threetears.enforcement.fake_parity.walkers import (
    fake_parity_violations,
    find_fakes_in_tree,
)

__all__ = [
    "FakeParityConfig",
    "fake_parity_violations",
    "find_fakes_in_tree",
    "run_fake_parity_enforcement",
]
