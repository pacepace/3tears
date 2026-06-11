"""public-surface fingerprinting for skill attestation.

a component's *public surface* is derived purely from its AST -- no
imports, no execution -- so the fingerprint is deterministic and cheap
enough for every CI run. the surface of one ``.py`` file contributes:

- every top-level public ``def`` / ``async def`` with its full
  argument signature and return annotation;
- every top-level public ``class`` with its base list, plus each
  public method signature and each public annotated class attribute
  (catches dataclass / pydantic field drift);
- the sorted contents of a literal ``__all__``.

docstring and body edits do not move the fingerprint by design:
skill guidance is keyed to the API surface, not the implementation.
private names (leading underscore) never contribute, but private
*modules* do -- lazy ``__init__`` re-exports routinely source public
names from ``_private`` modules, and a module rename legitimately
invalidates import paths the skill may cite.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

__all__ = [
    "COMPONENT_DISTRIBUTIONS",
    "compute_surface_fingerprint",
    "resolve_component_dir",
]

#: component import root -> PyPI distribution name. attestation records
#: the distribution version alongside the fingerprint so a *consumer*
#: (who cannot cheaply recompute) gets a version-comparison staleness
#: heuristic.
COMPONENT_DISTRIBUTIONS: dict[str, str] = {
    "threetears.core": "3tears",
    "threetears.nats": "3tears-nats",
    "threetears.observe": "3tears-observe",
    "threetears.models": "3tears-models",
    "threetears.agent.tools": "3tears-agent-tools",
    "threetears.agent.memory": "3tears-agent-memory",
    "threetears.agent.wake": "3tears-agent-wake",
}

#: hex digest length recorded in attestations. 16 hex chars (64 bits)
#: is ample for drift *detection* (nobody is forging collisions in
#: their own CI) and keeps frontmatter lines well under line-length.
_DIGEST_CHARS = 16


def resolve_component_dir(component: str) -> Path:
    """resolve a component import root to its source directory.

    works in the uv workspace (editable installs point at ``src/``
    trees) and in an installed environment (site-packages) alike.

    :param component: import root, e.g. ``"threetears.agent.wake"``
    :ptype component: str
    :return: directory containing the component's ``__init__.py``
    :rtype: Path
    :raises ModuleNotFoundError: when the component is not installed
    """
    spec = importlib.util.find_spec(component)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"component {component!r} is not importable in this environment")
    return Path(spec.origin).parent


def _signature_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}({ast.unparse(node.args)}){returns}"


def _module_surface_lines(rel_module: str, tree: ast.Module) -> list[str]:
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                lines.append(f"{rel_module}:def {node.name}{_signature_text(node)}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            lines.append(f"{rel_module}:class {node.name}({bases})")
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if not item.name.startswith("_"):
                        lines.append(f"{rel_module}:class {node.name}.{item.name}{_signature_text(item)}")
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if not item.target.id.startswith("_"):
                        lines.append(f"{rel_module}:class {node.name}.{item.target.id}: {ast.unparse(item.annotation)}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        names = ast.literal_eval(node.value)
                    except ValueError:
                        continue
                    lines.extend(f"{rel_module}:__all__:{name}" for name in sorted(names))
    return lines


def compute_surface_fingerprint(package_dir: Path) -> str:
    """fingerprint the public surface of a source tree.

    :param package_dir: directory of the component (the one holding
        its ``__init__.py``); walked recursively
    :ptype package_dir: Path
    :return: ``"sha256:<16 hex chars>"``
    :rtype: str
    """
    lines: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        rel_module = path.relative_to(package_dir).with_suffix("").as_posix().replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines.extend(_module_surface_lines(rel_module, tree))
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:_DIGEST_CHARS]}"
