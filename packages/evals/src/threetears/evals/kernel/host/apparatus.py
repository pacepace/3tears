"""The measuring rig's own failure — the one exception the engine defines and hosts raise.

Two failures look identical at a tool's action boundary and must never be treated alike:

* **In-world failure** — a provider 429, a lookup that found nothing, a malformed argument.
  Legitimate signal: the candidate should see it, handle it, and an eval may score exactly how
  it handles it. Raise an ordinary exception; the host's boundary absorbs it into the failure
  result it has always produced.
* **Apparatus failure** — a replay miss, a malformed seeded payload, a harness bug. Not part of
  the world under test. Raise :class:`ApparatusError`. A host that absorbs this instead would
  complete and score the cell, and what it scored would be the candidate's manner toward a
  broken rig rather than its behaviour in the world the template describes.

**The engine owns this vocabulary; hosts raise it.** The arrow points this way because the
distinction is eval's, not any host's: it is the same one
:data:`~threetears.evals.kernel.host.sweepables.SweepableRole` draws between a swept lever and the rig
that measures it. A host defines its own tools, its own action boundary, and its own rule for
when that boundary re-raises rather than absorbs — but it does not get to define what "the rig
broke" means, because the engine is what changes its behaviour on the answer.

What catching it obliges: the cell must not be scored on it. A rig fault is classified as
infrastructure rather than as the candidate's failure, and the cell stops rather than driving
further turns — otherwise a later candidate error can outrank the infrastructure one in
classification and the harness's own fault gets attributed to the candidate. A broken rig should
subtract a measurement, never invent one.

**Where the engine catches it.** Out of a kind's ``prepare`` or ``invoke``, the runner records THAT
cell excluded under the ``apparatus_failed`` termination — its ``infra_error`` an ``apparatus:``
entry naming the fault, its spend whatever the kind had reported through the cell's sink — and goes
on to the next cell. A run whose every cell an apparatus fault excluded measured nothing and ends
``failed``.

This module imports nothing but ``__future__``, and that is load-bearing rather than incidental: a
host tool layer importing it acquires this one leaf and nothing behind it, and this leaf reaches
back into no host. ``tests/test_extraction_import_boundary.py`` holds every module of the engine to
naming no host package, this one included.
"""

from __future__ import annotations

__all__ = ["ApparatusError"]


class ApparatusError(Exception):
    """A fault in the eval measuring rig, not a failure of the world under test.

    Raised by a host's tool layer, caught by the engine. See the module docstring for the
    distinction it draws and why the engine rather than the host owns it.
    """
