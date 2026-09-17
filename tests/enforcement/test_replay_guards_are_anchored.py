"""every production ``ReplayGuard`` decides about its anchor, rather than defaulting into refusal.

**Three guards existed and two were wrong, in the same way, for months.** A ``ReplayGuard``
without an anchor cannot tell a bucket it never had from one it lost, so it assumes the worse
and applies its creation-time watermark to both. On a FIRST run nothing was ever recorded and
no replay is possible -- yet every artifact issued before the bucket existed is refused, and
refused as ``pop nonce replay``, naming the one thing that did not happen.

Every one of these ledgers is memory-backed on purpose, because a nonce is burned on the
hottest path there is. So the bucket dies with the broker, and "first run" is really "every
NATS restart".

What that cost, concretely:

- ``pop_nonces`` in ``threetears.registry.server``: a 65-second reach, so EVERY tool call
  through the registry was refused for a minute after any restart.
- ``proxy_assertion_nonces`` in ``threetears.agent.tools.server``: a 5-second reach, so every
  proxied call to a pod was refused for five seconds.
- the hub's DPoP guard had the same defect until it was given an anchor, which is the fix this
  test generalises rather than the exception it makes.

None of it was visible. The consumer's CI runs only its unit and enforcement suites, so the
integration tests that catch this had never run there; they were red for months and the
refusal surfaces as a JetStream error naming replay.

**This gate is about the DECISION, not the value.** ``anchor=None`` is legitimate -- a pod with
no registry has nowhere to record first-existence, and fail-closed is the right answer there --
but it must be WRITTEN, because the whole defect was a default nobody chose. A construction
that names ``anchor`` at all passes; one that omits it does not.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import pytest

pytestmark = pytest.mark.enforcement

#: the packages' source roots, which is where a production construction lives. Tests build
#: guards freely and are deliberately out of scope: a test asserting the refusal behaviour has
#: to be able to construct one without an anchor.
_SRC_ROOTS: Final[tuple[Path, ...]] = (Path(__file__).resolve().parents[2] / "packages",)

#: the module that DEFINES the guard, whose own docstring example constructs one to explain it.
_DEFINING_MODULE: Final[str] = "replay_guard.py"


def _constructions() -> list[tuple[str, int, bool]]:
    """every ``ReplayGuard(...)`` call in production source, and whether it names ``anchor``.

    :return: ``(path, line, names_anchor)`` per construction
    :rtype: list[tuple[str, int, bool]]
    """
    found: list[tuple[str, int, bool]] = []
    for root in _SRC_ROOTS:
        for path in sorted(root.rglob("src/**/*.py")):
            if path.name == _DEFINING_MODULE:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name != "ReplayGuard":
                    continue
                names_anchor = any(kw.arg == "anchor" for kw in node.keywords)
                found.append((str(path), node.lineno, names_anchor))
    return found


def test_every_production_replay_guard_names_its_anchor() -> None:
    """a guard built without naming ``anchor`` refuses everything for its whole reach on first run.

    :return: nothing
    :rtype: None
    """
    silent = [(p, line) for p, line, named in _constructions() if not named]
    assert not silent, (
        "these ReplayGuard constructions do not name `anchor`, so they default to refusing every "
        "artifact issued within their reach of the bucket's creation -- on a first run, and after "
        "every NATS restart, because these ledgers are memory-backed. Pass an anchor, or pass "
        "`anchor=None` with a comment saying why this ledger has nowhere to record "
        "first-existence:\n  " + "\n  ".join(f"{p}:{line}" for p, line in silent)
    )


def test_the_gate_has_something_to_check() -> None:
    """a walker that found nothing would pass silently forever.

    The assertion above is satisfied by an empty list, so it would keep passing if the class were
    renamed, the source layout moved, or the glob stopped matching -- which is the failure mode
    every enforcement test in this tree is written to avoid.

    :return: nothing
    :rtype: None
    """
    assert _constructions(), (
        "no ReplayGuard construction found in any package source. Either the class was renamed or "
        "this walker's source glob no longer matches; the gate is inert until it is fixed"
    )
