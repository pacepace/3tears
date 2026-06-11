"""CLI: verify or re-attest a SKILL.md.

verify (default) recomputes every attested component fingerprint and
reports drift; ``--update`` rewrites the frontmatter ``attested:``
block with freshly computed fingerprints and currently installed
distribution versions. consumers in an installed environment can run
verify mode to get a definitive staleness answer (the skill body's
version-comparison heuristic is the no-tooling fallback).
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from threetears.enforcement.integration_skill.skill_doc import parse_skill_document
from threetears.enforcement.integration_skill.surface import (
    COMPONENT_DISTRIBUTIONS,
    compute_surface_fingerprint,
    resolve_component_dir,
)

_ATTESTED_BLOCK = re.compile(r"^attested:\n(?:^\s{2}.+\n)*", flags=re.MULTILINE)


def _render_attested_block() -> str:
    lines = ["attested:"]
    for component, distribution in COMPONENT_DISTRIBUTIONS.items():
        fingerprint = compute_surface_fingerprint(resolve_component_dir(component))
        try:
            dist_version = version(distribution)
        except PackageNotFoundError:
            dist_version = "unreleased"
        lines.append(f'  {component}: "{fingerprint} @ {dist_version}"')
    return "\n".join(lines) + "\n"


def _update(skill_path: Path) -> int:
    text = skill_path.read_text(encoding="utf-8")
    block = _render_attested_block()
    if _ATTESTED_BLOCK.search(text):
        text = _ATTESTED_BLOCK.sub(block, text, count=1)
    else:
        lines = text.splitlines(keepends=True)
        lines.insert(1, block)
        text = "".join(lines)
    skill_path.write_text(text, encoding="utf-8")
    print(f"re-attested {len(COMPONENT_DISTRIBUTIONS)} components in {skill_path}")
    return 0


def _verify(skill_path: Path) -> int:
    doc = parse_skill_document(skill_path)
    if not doc.attested:
        print(f"{skill_path}: no 'attested:' block -- nothing is verified.")
        return 1
    drift = 0
    for component, attested in sorted(doc.attested.items()):
        try:
            computed = compute_surface_fingerprint(resolve_component_dir(component))
        except ModuleNotFoundError:
            print(f"{component:30} attested {attested.fingerprint} @ {attested.version:12} not installed (skipped)")
            continue
        status = "ok" if computed == attested.fingerprint else "STALE"
        drift += status == "STALE"
        print(f"{component:30} attested {attested.fingerprint} @ {attested.version:12} computed {computed} {status}")
    if drift:
        print(f"\n{drift} component(s) drifted -- guidance for those sections is unverified.")
    return 1 if drift else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m threetears.enforcement.integration_skill")
    parser.add_argument("skill_path", type=Path)
    parser.add_argument("--update", action="store_true", help="recompute and rewrite the attested block")
    args = parser.parse_args(argv)
    if args.update:
        return _update(args.skill_path)
    return _verify(args.skill_path)


if __name__ == "__main__":
    sys.exit(main())
