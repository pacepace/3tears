"""pytest-friendly orchestration for the declared-imports gate."""

from __future__ import annotations

import pytest

from threetears.enforcement.imports_declared.config import ImportsDeclaredConfig
from threetears.enforcement.imports_declared.walkers import (
    declared_distributions,
    imported_modules,
    module_owners,
    undeclared_imports,
    unresolved_imports,
)

__all__ = ["imports_declared_findings", "run_imports_declared_enforcement"]


def imports_declared_findings(config: ImportsDeclaredConfig) -> list[str]:
    """every reason the repo fails the gate, each a complete operator-facing message.

    four checks, in the order an operator should read them: the two non-vacuity guards (an
    empty or filtered-out import walk, an empty or blind owner map -- either makes the rule
    pass by having nothing to compare), then imports nothing installed provides (fixed by a
    sync), then the rule itself (fixed by a declaration).

    :param config: the repo's gate configuration
    :ptype config: ImportsDeclaredConfig
    :return: findings; empty when the repo passes
    :rtype: list[str]
    """
    findings: list[str] = []
    imports = imported_modules(config)
    owners = module_owners()
    roots = ", ".join(config.source_roots)
    tops = {module.split(".", 1)[0] for module in imports}
    if not imports:
        findings.append(f"no third-party imports found under {roots} -- this gate is checking nothing.")
    missing_roots = sorted(config.required_import_roots - tops)
    if imports and missing_roots:
        findings.append(
            f"the import walk found {sorted(tops)} but not {missing_roots}, which {roots} imports "
            "throughout -- the stdlib/first-party filter is excluding what it should govern."
        )
    if not owners:
        findings.append(
            "no installed distribution reports a module -- the rule would pass vacuously. "
            "Run `uv sync --all-extras --all-groups`."
        )
    blind = sorted(module for module in config.required_owned_modules if not owners.get(module))
    if owners and blind:
        findings.append(
            f"no installed distribution reports {blind}; if they are installed EDITABLE, "
            "editable_module_files has stopped resolving them and the rule is passing vacuously."
        )
    unresolved = unresolved_imports(imports, owners)
    if owners and unresolved:
        findings.append(
            f"imported but provided by no installed distribution: {unresolved}. "
            "Run `uv sync --all-extras --all-groups` before reading this as a dependency-list problem."
        )
    offenders = undeclared_imports(imports, owners, declared_distributions(config.manifest))
    if offenders:
        findings.append(
            "these imports resolve only through some OTHER package's dependency list "
            f"(module: (owners, importers)): {offenders}. Declare the distribution in "
            f"{config.manifest.name} -- a dependency that arrives by accident disappears by "
            "accident, in a release note nobody here reads."
        )
    return findings


def run_imports_declared_enforcement(config: ImportsDeclaredConfig) -> None:
    """run the gate and fail with every finding.

    :param config: the repo's gate configuration
    :ptype config: ImportsDeclaredConfig
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: when there is any finding
    """
    findings = imports_declared_findings(config)
    if findings:
        pytest.fail("declared-imports gate:\n- " + "\n- ".join(findings))
