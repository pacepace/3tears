"""The host bridge: an eval's host, the command line, and a measure declared on the function computing it.

The quick path itself — :class:`~threetears.evals.quick.contract.Eval` and what it takes and returns — is the
package root's (``from threetears.evals import Eval, Judge, Model``), loaded on first use. This root holds what a
product adopting the engine as a host reaches for beside it:

- :func:`as_host` is the host for a set of evals: the vocabulary their graders, levers and world need, over the
  product's own store, so ``Eval(...).run(host=as_host([...], store=...), scope=...)`` stores where the product
  reads, and the engine's other surfaces read the same runs through it.
- :func:`run_cli` is ``python -m threetears.evals``: ``run``, ``ls``, ``report``, ``bundle``, ``spend``, ``gate``,
  ``frontier`` and ``judge-temperature`` over a host named ``module:factory``, or mounted under a product's own
  command line with its host factory and any subcommands of its own (:class:`HostCommand`). Handed a ``.py`` or
  ``.toml`` file instead, ``python -m threetears.evals`` runs the evals it declares
  (:mod:`threetears.evals.quick.files`).
- ``@measure(...)`` declares a measure on the function that computes it (a :class:`Measure`, whose ``descriptor`` a
  host registers): an ordinary contract object, mixed freely with ones the host writes by hand, and a function
  grader an eval reads as the measure it declares.

This package composes the others and is composed by nothing in the engine: it may import ``schema``, ``kernel``,
``run``, ``analysis``, ``storage`` and ``testing``, and no package of the engine imports it.

**This module is the package's public root for the host bridge.** A host imports from here and from no module
below it, and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.quick.cli import (
    DEFAULT_PROG,
    ENGINE_COMMANDS,
    EXIT_FAILED,
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_RUN_DID_NOT_COMPLETE,
    HostCommand,
    HostFactory,
    build_parser,
    run_cli,
)
from threetears.evals.quick.contract import as_host
from threetears.evals.quick.measures import Measure, measure

__all__ = [
    "DEFAULT_PROG",
    "ENGINE_COMMANDS",
    "EXIT_FAILED",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RUN_DID_NOT_COMPLETE",
    "HostCommand",
    "HostFactory",
    "Measure",
    "as_host",
    "build_parser",
    "measure",
    "run_cli",
]
