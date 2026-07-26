"""A path with a per-file ``SLF001`` ignore must not also carry a blanket ``noqa``.

The pragma is redundant to ruff -- the per-file ignore already covers it -- so on its own this
would be a tidiness rule not worth a test. It is here because of what the redundancy does to
the exemptions ledger.

``tests/enforcement/_underscore_exemptions.txt`` is maintained by re-running ruff over an
exempted path and rewriting that path's entries from the output. Ruff honours an inline
``noqa`` even under ``--isolated``, so any access carrying one is reported by nothing and
silently never reaches the ledger. Six accesses in the scrape suites went missing exactly that
way, and the loss is invisible from both ends: the gate passes (the underscore walker scans
``packages/*/src`` only and never enters a ``tests/`` tree), and the regeneration command
produces a short list without complaining.

That is why this guards the ledger rather than style. A bidirectional consistency check --
every entry resolves, every flagged access has an entry -- cannot catch it either, because the
blinded access is never flagged in the first place. This has to be checked at the source.

**Every ruff config, not just the root one.** The first version read only the root
``pyproject.toml``, which left the sidecar uncovered -- and ``packages/scrape/sidecar/ruff.toml``
is a full override declaring its own ``SLF001`` exemptions, with five ledger entries pointing
into it. A pragma added there reproduced the trap in full while this test stayed green: the
failure it exists to end, wearing the guard's own uniform.

**Keys are globs, and one matching nothing is a failure rather than a skip.** ruff resolves
``per-file-ignores`` keys as globs relative to the config declaring them; the first version
compared them as literal paths and skipped any that did not resolve. A glob key would then
cover zero files and pass -- a short list produced without complaining, which is precisely the
shape of the original defect.
"""

from __future__ import annotations

import re
from pathlib import Path

from _ruff_config_discovery import exempted_files, ruff_configs, slf001_globs

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Any comment that would stop ruff reporting an ``SLF001`` on that line, not merely the
#: fully-spelled form. A bare ``# noqa`` suppresses everything and ``# ruff: noqa`` does it for
#: a whole file; both blind the ledger regeneration identically, so matching only the explicit
#: spelling would have left the cheapest one through.
_BLANKET_NOQA = re.compile(
    r"#\s*(?:ruff:\s*)?noqa(?!\s*:)"  # bare `# noqa` / `# ruff: noqa`, no code list
    r"|#\s*(?:ruff:\s*)?noqa\s*:[^#\n]*\bSLF(?:001)?\b"  # a code list naming SLF001 or SLF
)


def _ruff_configs() -> list[Path]:
    return ruff_configs(_REPO_ROOT)


def _slf001_globs(config: Path) -> list[str]:
    return slf001_globs(config)


def _exempted_files(config: Path, pattern: str) -> list[Path]:
    return exempted_files(config, pattern, _REPO_ROOT)


class TestNoRedundantSlf001Pragmas:
    def test_the_scan_actually_covers_something(self) -> None:
        """A floor, because every failure mode of this discovery is an EMPTY list.

        The first version opened with an unconditional read of the root `pyproject.toml`, which
        would have raised if discovery were wrong. Replacing it with globbing plus filters
        removed that accidental floor: a bad filter, a renamed config, or a path predicate that
        excludes the tree all yield zero files, and both assertions below then pass on a scan of
        nothing -- which is the exact failure this module exists to end, committed by the module
        itself.
        """
        configs = _ruff_configs()
        assert (_REPO_ROOT / "pyproject.toml") in configs, "the root ruff config was not discovered"
        assert (_REPO_ROOT / "packages/scrape/sidecar/ruff.toml") in configs, (
            "the sidecar's nested override was not discovered, so every path it exempts is "
            "unscanned while this suite still reports success"
        )

        scanned = {p for c in configs for g in _slf001_globs(c) for p in _exempted_files(c, g)}
        assert len(scanned) > 25, f"only {len(scanned)} files scanned; discovery has silently collapsed"

    def test_every_slf001_exemption_glob_matches_something(self) -> None:
        """A key matching no file is a stale exemption AND a silent hole in the check below.

        Asserted rather than skipped: the previous version treated an unresolvable key as
        nothing to check, so a glob covering zero files kept the suite green.
        """
        empty: list[str] = []
        for config in _ruff_configs():
            for glob in _slf001_globs(config):
                if not _exempted_files(config, glob):
                    empty.append(f"{config.relative_to(_REPO_ROOT)}: {glob}")

        assert not empty, (
            "these per-file SLF001 ignores match no Python file, so they exempt nothing -- and "
            f"exempt nothing from the pragma check either, silently: {empty}"
        )

    def test_exempted_paths_carry_no_blanket_noqa(self) -> None:
        """Named per offender, so the failure says which file and which line to fix."""
        offenders: list[str] = []
        for config in _ruff_configs():
            for glob in _slf001_globs(config):
                for path in _exempted_files(config, glob):
                    for number, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
                        if _BLANKET_NOQA.search(line):
                            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{number}")

        assert not offenders, (
            "these lines carry a `noqa` that suppresses SLF001 on a path that already has a "
            "per-file ignore. The pragma is redundant to ruff AND hides the access from the "
            f"exemptions ledger regeneration, so it silently drops out of the record: {offenders}"
        )
