"""Regenerate the current core fixture corpus from ``stored_sample`` and re-pin both digest tables in the pins test.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite the corpus for the current CORE_SCHEMA_VERSION and re-pin the tables:
    uv run python packages/evals/scripts/regenerate_core_fixtures.py

    # Rewrite every fixture, even one that differs from its sample only in a clock reading or a generated id:
    uv run python packages/evals/scripts/regenerate_core_fixtures.py --all

For every core type (``CORE_DOC_TYPES``) it writes ``tests/fixtures/core_documents/v{CORE_SCHEMA_VERSION}/{doc_type}.json``
from :func:`stored_sample` (sorted keys, two-space indent, one trailing newline), then rewrites the two digest tables
in ``tests/test_core_schema_pins.py``: ``CORE_SHAPE_DIGESTS`` (each core type's shape under
:func:`core_shape_digest`) and ``FIXTURE_DIGESTS`` (the sha256 of every ``v*/*.json`` under the corpus). Both are
written with sorted keys. It does not touch
``PINNED_CORE_SCHEMA_VERSION`` or ``PINNED_CORE_SCHEMA_MINOR``: those are the author's statement of what the digests
were taken under, and moving them is part of the version change the pins test's header walks through.

**A released version's corpus is never regenerated.** A ``v<N>`` directory is what an old store holds: it is frozen
so that a build keeps reading it, and regenerating it would make the test that reads it vouch for the new build
rather than for the stored past. Only the current version's directory is written, and only while the version is
unreleased (the baseline of a release not yet published may be regenerated in place, as the pins test's header
says). The script refuses, naming the directory, when a ``v<N>`` newer than ``CORE_SCHEMA_VERSION`` exists: that
means the version constant was moved back or the working tree holds a corpus from a later build, and regenerating
the older directory would overwrite what the newer one was copied from.

**Why a run on an unchanged tree changes nothing.** ``stored_sample`` reads the clock and draws UUIDs, so two
samples of one model never agree byte for byte, and writing each one would move every fixture and every digest on
every run. A fixture is therefore left as it is when the new sample matches it in structure and in every value but
the ones that are a clock reading (an ISO-8601 timestamp) or a generated id (a UUID) at the same place in both: those
are what a sample draws fresh, and a corpus document that differs only in them is the same document. Any other
difference (a field added, removed or retyped, a fixed value changed) is written. A sample whose fixed timestamp or
id was changed on purpose reads as unchanged by that rule; ``--all`` writes every file regardless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import typing
from pathlib import Path
from typing import Any

from public_surface import PACKAGE_ROOT

# The pins test and the sample factories are imported as the suite imports them (``packages.evals.tests...``,
# namespace packages under the repository root), so the root is put on the path before they are.
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.evals.tests import test_core_schema_pins as pins  # noqa: E402
from packages.evals.tests.stored_models import doc_type_of, stored_models, stored_sample  # noqa: E402
from threetears.evals.schema.base import CoreDocumentModel  # noqa: E402
from threetears.evals.schema.versioning import CORE_DOC_TYPES, CORE_SCHEMA_VERSION  # noqa: E402

#: The pins test whose two digest tables this script rewrites.
PINS_PATH = PACKAGE_ROOT / "tests" / "test_core_schema_pins.py"

#: The command that regenerates the corpus, as a refusal tells a reader.
REGENERATE = "uv run python packages/evals/scripts/regenerate_core_fixtures.py"

_VERSION_DIR = re.compile(r"v(\d+)")

#: A whole string value that is a clock reading, as ``utc_now_iso`` writes one.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)")

#: A whole string value that is a generated id (a UUID, any version).
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class SupersededCorpusError(RuntimeError):
    """A corpus directory newer than the current core version exists, so regenerating would write a superseded one."""


def newer_corpus_dirs(root: Path, version: int) -> list[Path]:
    """The ``v<N>`` corpus directories under ``root`` with ``N`` above ``version``.

    Args:
        root: The corpus root (``tests/fixtures/core_documents``).
        version: The current ``CORE_SCHEMA_VERSION``.

    Returns:
        Those directories, oldest first.
    """
    newer = [
        path
        for path in root.glob("v*")
        if path.is_dir() and (match := _VERSION_DIR.fullmatch(path.name)) and int(match.group(1)) > version
    ]
    return sorted(newer, key=lambda path: int(path.name[1:]))


def same_document(kept: Any, sampled: Any) -> bool:
    """Whether two documents agree in structure and in every value but a drawn clock reading or generated id.

    Args:
        kept: The fixture on disk, parsed.
        sampled: The new sample.

    Returns:
        ``True`` when they have the same keys and types throughout and differ only where both hold a timestamp or
        both hold a UUID.
    """
    if isinstance(kept, dict) and isinstance(sampled, dict):
        return kept.keys() == sampled.keys() and all(same_document(kept[key], sampled[key]) for key in kept)
    if isinstance(kept, list) and isinstance(sampled, list):
        return len(kept) == len(sampled) and all(same_document(a, b) for a, b in zip(kept, sampled, strict=True))
    if isinstance(kept, str) and isinstance(sampled, str):
        return kept == sampled or any(
            pattern.fullmatch(kept) and pattern.fullmatch(sampled) for pattern in (_TIMESTAMP, _UUID)
        )
    return type(kept) is type(sampled) and kept == sampled


def write_corpus(root: Path, version: int, *, rewrite_all: bool = False) -> dict[str, str]:
    """Write the ``v{version}`` corpus from ``stored_sample`` and return each core type's shape digest.

    Args:
        root: The corpus root (``tests/fixtures/core_documents``).
        version: The current ``CORE_SCHEMA_VERSION``.
        rewrite_all: Write every fixture, including one :func:`same_document` reads as unchanged.

    Returns:
        ``doc_type`` to :func:`core_shape_digest`, for every core type.

    Raises:
        SupersededCorpusError: A ``v<N>`` directory above ``version`` exists; nothing is written.
        RuntimeError: A core type in ``CORE_DOC_TYPES`` has no stored model, or a stored core model is not in it.
    """
    newer = newer_corpus_dirs(root, version)
    if newer:
        raise SupersededCorpusError(
            f"{', '.join(path.name for path in newer)} exists beside the current core version v{version}: "
            f"regenerating v{version} would rewrite a superseded corpus, and a released version's corpus is never "
            f"regenerated. If {newer[-1].name} is the corpus of a version this build has yet to declare, move "
            f"CORE_SCHEMA_VERSION to it (with its ledger line and upgrader) and run {REGENERATE} again; if it is "
            f"stray, remove it."
        )
    models = {doc_type_of(model): model for model in stored_models() if issubclass(model, CoreDocumentModel)}
    if set(models) != set(CORE_DOC_TYPES):
        raise RuntimeError(
            f"the core stored models {sorted(models)} are not CORE_DOC_TYPES {sorted(CORE_DOC_TYPES)}: "
            "the core's membership moved, which is a decision recorded in CORE_DOC_TYPES and the pins test together; "
            "settle that first, then run the regenerator again"
        )
    target = root / f"v{version}"
    target.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, str] = {}
    for doc_type, model in sorted(models.items()):
        shapes[doc_type] = pins.core_shape_digest(model)
        sample = stored_sample(model)
        path = target / f"{doc_type}.json"
        if not rewrite_all and path.exists() and same_document(json.loads(path.read_text(encoding="utf-8")), sample):
            continue
        path.write_text(json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shapes


def fixture_digests(root: Path) -> dict[str, str]:
    """The sha256 of every ``v*/*.json`` under ``root``, by its path relative to it.

    Args:
        root: The corpus root.

    Returns:
        Relative POSIX path to sha256 hex digest, sorted by path.
    """
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("v*/*.json"))
    }


def render_table(name: str, annotation: str, table: dict[str, str]) -> str:
    """A digest table as the pins test spells it: the annotated name, sorted keys, one entry per line.

    Args:
        name: The module-level name (``CORE_SHAPE_DIGESTS``).
        annotation: Its annotation (``dict[str, str]``).
        table: The entries.

    Returns:
        The assignment's source, without a trailing newline.
    """
    rows = "".join(f'    "{key}": "{value}",\n' for key, value in sorted(table.items()))
    return f"{name}: {annotation} = {{\n{rows}}}"


def repin(source: str, name: str, table: dict[str, str]) -> str:
    """``source`` with the dict literal assigned to ``name`` replaced by ``table``.

    Args:
        source: The pins test's source.
        name: The table to replace.
        table: Its new entries.

    Returns:
        The new source.

    Raises:
        RuntimeError: ``name`` is not assigned a ``dict[str, str]`` literal exactly once.
    """
    pattern = re.compile(rf"^{name}: dict\[str, str\] = \{{\n(?:    .*\n)*?\}}", re.MULTILINE)
    if len(pattern.findall(source)) != 1:
        raise RuntimeError(
            f"{PINS_PATH.name} does not assign {name} a dict[str, str] literal exactly once; "
            "the regenerator rewrites that literal in place, so restore its shape"
        )
    return pattern.sub(lambda _match: render_table(name, "dict[str, str]", table), source)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the current corpus and re-pin the tables."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help="rewrite every fixture, even one that differs from its sample only in a clock reading or a generated id",
    )
    args = parser.parse_args(argv)
    root = pins.FIXTURES
    try:
        shapes = write_corpus(root, CORE_SCHEMA_VERSION, rewrite_all=args.all)
    except SupersededCorpusError as refusal:
        print(refusal, file=sys.stderr)
        return 1
    source = PINS_PATH.read_text(encoding="utf-8")
    source = repin(source, "CORE_SHAPE_DIGESTS", shapes)
    source = repin(source, "FIXTURE_DIGESTS", fixture_digests(root))
    PINS_PATH.write_text(source, encoding="utf-8")
    print(f"regenerated {root.name}/v{CORE_SCHEMA_VERSION}/ ({len(shapes)} documents) and re-pinned both tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
