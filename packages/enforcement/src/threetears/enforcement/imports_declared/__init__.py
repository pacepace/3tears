"""every third-party module a repo imports comes from a distribution it DECLARES.

*An import of ours must be declared by us.* A dependency that arrives through somebody
else's dependency list -- an extra of a 3tears package, a transitive of a LangChain partner
package -- is present by accident, and it disappears by accident, in a release note nobody in
the consuming repo is reading. The import then fails at boot with nothing in that repo naming
the cause. The aibots hub paid for it twice: generated Hub-client models began importing
``threetears.iam.connection_types`` with ``3tears-iam`` undeclared, and the gateway imported
Pillow, which resolved only through ``3tears-agent-tools[all]``'s ``vision`` extra.

**Every non-stdlib, non-first-party import is governed**, not only ``threetears.*``. A
namespace-only gate leaves every other import free to free-ride, and the Pillow case was one.

**Checked against the INSTALLED distributions, not a hand-written map.** A map is a second
thing to keep in step, and it goes stale in exactly the silent way the problem it checks for
does. Each import resolves through the distributions' own file lists (``import PIL`` is owned
by ``pillow``, ``import jwt`` by ``pyjwt``), at the import's full module path: ``threetears.agent``
is shipped by six distributions, ``threetears.agent.tools.base_tool`` by one.

**Both computed inputs carry a non-vacuity guard.** The rule is ``offenders = imports whose
owners are all undeclared``; an empty import walk and an empty owner map both yield zero
offenders and a green run. The owner map is the half that silently emptied once: editable
installs list no module files, so :func:`editable_module_files` walks their checkouts.

Residual holes, stated rather than papered over:

- a module provided by several distributions (``google.cloud``) passes when ANY of them is
  declared, since the walk cannot tell which one an import means;
- a declaration in a dependency GROUP satisfies the rule, though a group is not installed at
  runtime -- reporting it as undeclared would send the reader after the wrong fix;
- an editable distribution without a ``src/`` layout contributes only its top-level packages
  that carry an ``__init__.py``.

A consumer adopts it with a thin test::

    run_imports_declared_enforcement(
        ImportsDeclaredConfig(
            repo_root=REPO_ROOT,
            first_party=frozenset({"aibots_agents"}),
            required_import_roots=frozenset({"threetears", "pydantic"}),
        )
    )
"""

from threetears.enforcement.imports_declared.config import ImportsDeclaredConfig
from threetears.enforcement.imports_declared.runner import (
    imports_declared_findings,
    run_imports_declared_enforcement,
)
from threetears.enforcement.imports_declared.walkers import (
    canonical_distribution_name,
    declared_distributions,
    editable_module_files,
    imported_modules,
    is_governed,
    module_name,
    module_owners,
    requirement_name,
    undeclared_imports,
    unresolved_imports,
)

__all__ = [
    "ImportsDeclaredConfig",
    "canonical_distribution_name",
    "declared_distributions",
    "editable_module_files",
    "imported_modules",
    "imports_declared_findings",
    "is_governed",
    "module_name",
    "module_owners",
    "requirement_name",
    "run_imports_declared_enforcement",
    "undeclared_imports",
    "unresolved_imports",
]
