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
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Any comment that would stop ruff reporting an ``SLF001`` on that line, not merely the
#: fully-spelled form. A bare ``# noqa`` suppresses everything and ``# ruff: noqa`` does it for
#: a whole file; both blind the ledger regeneration identically, so matching only the explicit
#: spelling would have left the cheapest one through.
_BLANKET_NOQA = re.compile(
    r"#\s*(?:ruff:\s*)?noqa(?!\s*:)"  # bare `# noqa` / `# ruff: noqa`, no code list
    r"|#\s*(?:ruff:\s*)?noqa\s*:[^#\n]*\bSLF(?:001)?\b"  # a code list naming SLF001 or SLF
)


def _is_vendored(path: Path) -> bool:
    """Whether *path* sits under a vendored directory INSIDE the repo.

    Relative to the repo root, deliberately. Testing an absolute path's parts meant a checkout
    living under any directory named `.venv` or `.git` -- a plausible place to clone one --
    excluded the entire tree, and both assertions below would then pass on a scan of nothing.
    """
    return any(part in {".venv", "node_modules", ".git", "__pycache__"} for part in path.relative_to(_REPO_ROOT).parts)


def _ruff_configs() -> list[Path]:
    """Every file ruff would read a per-file ignore from.

    An earlier version read the root ``pyproject.toml`` plus ``ruff.toml`` and described itself
    as reading every config -- which is how the sidecar override went unscanned. Being one form
    short is that same failure again, and the likeliest miss in a 27-package workspace is a
    package ``pyproject.toml`` growing a ``[tool.ruff]`` section, since every package already
    ships that file.

    No count in this prose. Two attempts to state one were wrong in opposite directions, and a
    third claimed to have fixed that by stating it once while stating it twice. The loop below
    is the list; it cannot disagree with itself.
    """
    configs: list[Path] = []
    for name in ("pyproject.toml", "ruff.toml", ".ruff.toml"):
        for path in _REPO_ROOT.rglob(name):
            if _is_vendored(path):
                continue
            if path.name == "pyproject.toml" and "[tool.ruff" not in path.read_text(errors="replace"):
                continue  # a pyproject with no ruff section configures nothing
            configs.append(path)
    return sorted(configs)


def _slf001_globs(config: Path) -> list[str]:
    """The ``per-file-ignores`` keys in *config* whose code list includes ``SLF001``."""
    data = tomllib.loads(config.read_text())
    section = (
        data.get("tool", {}).get("ruff", {}).get("lint", {})
        if config.name == "pyproject.toml"
        else data.get("lint", {})
    )
    per_file = section.get("per-file-ignores", {})
    return [key for key, codes in per_file.items() if "SLF001" in codes]


def _exempted_files(config: Path, pattern: str) -> list[Path]:
    """Python files matched by *pattern*, following ruff's own two-way matching.

    ruff matches a pattern containing no separator against the file's BASENAME anywhere beneath
    the config -- which is why its documented ``"__init__.py"`` example covers a whole tree --
    and matches a pattern containing one against the relative path. `Path.glob` only does the
    second, so a bare-name key would have matched whatever single file sat at the config's root
    and left every other file of that name unscanned: a partial match, which the empty-match
    assertion cannot see because it is not empty.
    """
    base = config.parent
    if "/" in pattern:
        return sorted(p for p in base.glob(pattern) if p.suffix == ".py" and not _is_vendored(p))
    return sorted(p for p in base.rglob(pattern) if p.suffix == ".py" and not _is_vendored(p))


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
