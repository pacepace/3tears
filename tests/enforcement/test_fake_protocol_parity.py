"""thin shell -- actual walker logic in
:mod:`threetears.enforcement.fake_parity`.

every test fake in this repo (class named ``Fake<Name>`` or
``_Fake<Name>``) must declare what production protocol it stands in
for, via subclass declaration or a ``# parity-with: <fqname>``
marker on the line above the class. fakes whose target is too large
to fully implement (``asyncpg.Pool``, ``nats.aio.Client``, etc.) are
exempted with a ``# parity-exempt: <rationale>`` marker on the line
above the class, on ONE line however long -- the walker reads the
first non-blank line above the class and stops.

``_fake_parity_exemptions.txt`` still parses and is deliberately
empty. Its entries are keyed ``path:LINE:symbol``, so an edit
anywhere above a fake silently stops the entry matching and this
test then reports ``no_declaration`` against code nobody touched.
A marker on the class moves with the class.

the test class / method names preserve the canonical shape so any
external CI looking for them continues to find them.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.fake_parity import (
    FakeParityConfig,
    run_fake_parity_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


_CONFIG = FakeParityConfig(
    repo_root=_REPO_ROOT,
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_fake_parity_exemptions.txt",
)


class TestFakeProtocolParity:
    """every fake declares parity (subclass / marker / exemption with rationale)."""

    def test_no_undeclared_fakes(self) -> None:
        """surface fakes whose method surface drifts from the production protocol."""
        run_fake_parity_enforcement(_CONFIG)
