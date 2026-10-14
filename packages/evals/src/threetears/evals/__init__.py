"""The eval engine: templates, cases, runs and results; the trial loop, its judge and simulated user; and the analysis over what runs produced.

The engine is five packages: :mod:`~threetears.evals.schema` (the stored shapes and the ports a host
implements), :mod:`~threetears.evals.kernel` (the behaviour every other package runs on, and the host
contract under :mod:`~threetears.evals.kernel.host`), :mod:`~threetears.evals.run` (launching, executing, judging
and storing runs), :mod:`~threetears.evals.analysis` (campaigns, bundles, memos and reports) and :mod:`~threetears.evals.gen` (generating
cases), plus three that sit beside the engine rather than inside it: :mod:`~threetears.evals.storage`
(the in-memory reference store), :mod:`~threetears.evals.testing` (the store conformance kit) and
:mod:`~threetears.evals.quick` (an eval's host, and the command line). Above the engine sit the surfaces an
agent drives it through: :mod:`~threetears.evals.ops` (typed operations, and one job contract for long
work), :mod:`~threetears.evals.actions` (the action catalogue every transport mounts) and
:mod:`~threetears.evals.transports` (each transport behind its own extra — ``fastmcp``). Two more are for a
completion client: :mod:`~threetears.evals.negotiation` (sending the closest request settings a model
takes, and answering a provider's refusal of one, on the base install) and :mod:`~threetears.evals.providers` (the
shipped adapter over 3tears-models, ``3tears-evals[models]``, which uses it); nothing in the engine imports
either. Runs are launched on demand; there is no scheduled-run surface.

**Start here: an eval is three slots.** ``Eval(cases, runner, grader)`` says what is evaluated — the cases, the thing
under test (a function, or a mapping of them whose first key is the control) and what counts as right (functions,
field names, a :class:`Judge`) — and ``await ev.run()`` runs it, returning a :class:`Result` that prints quietly and
writes its report. ``python -m threetears.evals FILE`` runs every eval a ``.py`` or ``.toml`` file declares. The
names that take it further are this module's own: :class:`World` and :class:`Log` (two more shapes of runner),
:func:`grader` (what a reading declares beside its grade), :class:`Example`, :class:`Model` and :class:`Answer`. Each
is loaded on first use, so importing this package loads none of the engine.

**Import only from a public root**, and only the names its ``__all__`` declares. The roots are
:data:`PUBLIC_ROOTS`, which a consumer can read to check its own imports rather than keep a copy;
``tests/test_package_matrix.py`` holds the tuple equal to the roots it enforces.

**There is no installed host.** A product adopts the engine by building one
:class:`~threetears.evals.kernel.host.EvalHost` — its vocabulary, its storage and its services —
and handing it to every entrypoint that reads any of them; a function that reads only the vocabulary
takes the :class:`~threetears.evals.kernel.host.HostProfile` the host carries. Nothing in the
engine reads a module-level, context-variable or default host, and importing this package installs
and configures nothing, so two hosts in one process — even in one event loop — are two values that
never meet. ``tests/test_two_hosts_one_process.py`` and ``tests/test_no_process_global_state.py``
hold that.

**Which build is installed** is :data:`__version__`, the distribution's version as its metadata states it. No
stored document carries it (each carries its own schema version), so a host that wants to say which build wrote
what records it beside what it stores.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version as _distribution_version
from types import MappingProxyType
from typing import Any

#: The installed distribution's version, as its metadata states it (``importlib.metadata``): the build a host
#: records beside what it stores, since no stored document or analysis bundle carries it. A build from a
#: pre-release tag (``v0.67.0-evaladopt.N``) states the version it prepares, so a host running one records the tag.
#: Read on first use through the module's ``__getattr__`` rather than at import: the source tree imports without
#: distribution metadata (the installability check and a checkout on ``sys.path`` both do), and a read that finds
#: no metadata raises ``importlib.metadata.PackageNotFoundError`` by name rather than answering ``"unknown"``.
__version__: str


#: The quick path's names, each by the module that defines it: served by :func:`__getattr__` on first use, so
#: ``import threetears.evals`` (and ``from threetears.evals import ops``) loads none of the engine until one is asked
#: for. What each is: ``docs/reference.md`` under the package's own root.
_LADDER: Mapping[str, str] = MappingProxyType(
    {
        "Answer": "threetears.evals.quick.answer",
        "Eval": "threetears.evals.quick.contract",
        "Example": "threetears.evals.quick.judged",
        "Judge": "threetears.evals.quick.judged",
        "Log": "threetears.evals.quick.logged",
        "Model": "threetears.evals.quick.client",
        "Result": "threetears.evals.quick.result",
        "World": "threetears.evals.quick.world",
        "grader": "threetears.evals.quick.graders",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve :data:`__version__` from the distribution's metadata, and a quick-path name from its module, on first read.

    Args:
        name: The attribute asked for: ``__version__``, or one of the quick path's names (``Eval``, ``Judge``, ...).

    Returns:
        The installed distribution's version string, or the quick-path object.

    Raises:
        AttributeError: For any other name, as a module normally would.
        importlib.metadata.PackageNotFoundError: When no ``3tears-evals`` distribution is installed, so the
            version cannot be stated; a host records the version only from an installed build.
    """
    if name == "__version__":
        return _distribution_version("3tears-evals")
    if (defined_in := _LADDER.get(name)) is not None:
        return getattr(import_module(defined_in), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Every name the module holds, the quick path's lazily served ones among them."""
    return sorted({*globals(), *_LADDER})


#: The public roots, as absolute module names. A consumer imports from these and from no module
#: below them, and only the names each root's ``__all__`` declares. ``kernel.host`` is a root of its own inside a
#: package: the contract a host implements. Two roots fail to import without an extra
#: (:data:`ROOT_EXTRAS`): ``transports.fastmcp``, which needs ``[fastmcp]``, and ``providers``, the shipped provider
#: adapter over 3tears-models, which needs ``[models]``.
PUBLIC_ROOTS: tuple[str, ...] = (
    "threetears.evals.schema",
    "threetears.evals.kernel",
    "threetears.evals.kernel.host",
    "threetears.evals.run",
    "threetears.evals.analysis",
    "threetears.evals.gen",
    "threetears.evals.storage",
    "threetears.evals.testing",
    "threetears.evals.quick",
    "threetears.evals.ops",
    "threetears.evals.actions",
    "threetears.evals.transports.fastmcp",
    "threetears.evals.negotiation",
    "threetears.evals.providers",
)


@dataclass(frozen=True)
class RootExtra:
    """What one optional public root needs installed before it imports: the extra, and the modules it brings.

    Attributes:
        extra: The extra to install, as ``pip install "3tears-evals[<extra>]"`` names it.
        modules: Every module the root imports that the base install does not provide, by the absolute name an
            ``import`` statement gives it (``anthropic``, ``threetears.models``), sorted: what must be importable for
            the root to import. A gate checks each with ``importlib.util.find_spec`` rather than read the manifest.
            The root may fail on any one of them first, so a gate that checks only the extra's own distribution
            (``threetears.models`` for ``[models]``) misses an install that has it and lacks another.
    """

    extra: str
    modules: tuple[str, ...]


#: Each public root that fails to import without an extra installed, mapped to what it needs (:class:`RootExtra`): the
#: extra's name, and the modules that extra provides which the root imports. What a consumer's import gate reads to
#: import every root its install can, and to say which module is missing, rather than keep its own list. Every root not
#: here imports on the base install, whose only dependency is pydantic.
#: ``tests/test_extraction_import_boundary.py`` proves the map: in fresh interpreters, with every extra's dependencies
#: unimportable exactly these roots fail, with one extra's missing exactly the roots listed under it fail, and with any
#: one listed module missing its root fails; and by an import walk, the modules listed for a root are exactly the
#: non-base modules its source names.
ROOT_EXTRAS: Mapping[str, RootExtra] = MappingProxyType(
    {
        "threetears.evals.transports.fastmcp": RootExtra("fastmcp", ("fastmcp",)),
        "threetears.evals.providers": RootExtra(
            "models", ("anthropic", "httpx", "langchain_core", "openai", "openrouter", "threetears.models")
        ),
    }
)

__all__ = [
    "PUBLIC_ROOTS",
    "ROOT_EXTRAS",
    "Answer",
    "Eval",
    "Example",
    "Judge",
    "Log",
    "Model",
    "Result",
    "RootExtra",
    "World",
    "__version__",
    "grader",
]
