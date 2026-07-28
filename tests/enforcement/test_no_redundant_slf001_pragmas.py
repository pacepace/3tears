"""thin shell -- actual discovery and scanning in
:mod:`threetears.enforcement.underscore_access.ruff_config` and
:mod:`threetears.enforcement.underscore_access.ledger`.

A path with a per-file ``SLF001`` ignore must not also carry a blanket ``noqa``. The pragma is
redundant to ruff -- the per-file ignore already covers it -- so on its own this would be a
tidiness rule not worth a test. It is here because of what the redundancy does to the exemptions
ledger: ruff honours an inline ``noqa`` even under ``--isolated``, so any access carrying one is
reported by nothing and silently never reaches the ledger. Accesses in the scrape suites went
missing exactly that way, invisible from both ends, since the underscore walker scans
``packages/*/src`` and never enters a ``tests/`` tree.

``scripts/regen-underscore-exemptions.py`` walks the AST instead, so it no longer inherits that
blind spot. This test remains because the pragma is still a hazard for anyone reaching for ruff
directly, and because a redundant suppression on an already-exempted path is a claim about the
code that is not true.

The repo-specific floor below stays here rather than in the package: which configs this
particular repo has is not a property of the domain.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access import (
    all_exempted_files,
    blanket_noqa_offenders,
    exempted_files,
    ruff_configs,
    slf001_globs,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestNoRedundantSlf001Pragmas:
    def test_the_scan_actually_covers_something(self) -> None:
        """A floor, because every failure mode of this discovery is an EMPTY list.

        A bad filter, a renamed config, or a path predicate that excludes the tree all yield
        zero files, and both assertions below then pass on a scan of nothing -- which is the
        exact failure this module exists to end, committed by the module itself.

        The sidecar's nested override is named explicitly because reading only the root config
        is the specific way this went wrong: ``packages/scrape/sidecar/ruff.toml`` is a full
        override, so every path it exempts is unscanned by a checker built on the root.
        """
        configs = ruff_configs(_REPO_ROOT)
        assert (_REPO_ROOT / "pyproject.toml") in configs, "the root ruff config was not discovered"
        assert (_REPO_ROOT / "packages/scrape/sidecar/ruff.toml") in configs, (
            "the sidecar's nested override was not discovered, so every path it exempts is "
            "unscanned while this suite still reports success"
        )

        scanned = all_exempted_files(_REPO_ROOT)
        assert len(scanned) > 25, f"only {len(scanned)} files scanned; discovery has silently collapsed"

    def test_every_slf001_exemption_glob_matches_something(self) -> None:
        """A key matching no file is a stale exemption AND a silent hole in the check below.

        Asserted rather than skipped: treating an unresolvable key as nothing to check lets a
        glob covering zero files keep the suite green.
        """
        empty: list[str] = []
        for config in ruff_configs(_REPO_ROOT):
            for glob in slf001_globs(config):
                if not exempted_files(config, glob, _REPO_ROOT):
                    empty.append(f"{config.relative_to(_REPO_ROOT)}: {glob}")

        assert not empty, (
            "these per-file SLF001 ignores match no Python file, so they exempt nothing -- and "
            f"exempt nothing from the pragma check either, silently: {empty}"
        )

    def test_exempted_paths_carry_no_blanket_noqa(self) -> None:
        """Named per offender, so the failure says which file and which line to fix."""
        offenders = blanket_noqa_offenders(_REPO_ROOT)

        assert not offenders, (
            "these lines carry a `noqa` that suppresses SLF001 on a path that already has a "
            "per-file ignore. The pragma is redundant to ruff AND hides the access from the "
            f"exemptions ledger regeneration, so it silently drops out of the record: {offenders}"
        )
