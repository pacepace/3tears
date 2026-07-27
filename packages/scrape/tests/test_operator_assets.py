"""The vendored noVNC tree: that it is intact, complete, and shippable.

noVNC is MPL-2.0 and this package is MIT. Redistributing MPL-2.0 files inside an MIT wheel is
permitted, and the obligations that come with it are specific: the licence text and copyright
notice travel with the files, the source stays identifiable, and any modification is marked.
Every assertion here is one of those obligations turned into something that fails loudly rather
than a claim in a document nobody re-reads.

The claim these guard is ``modified: false`` in ``novnc-provenance.json``. A promise that a
vendored tree is unmodified is worth exactly as much as the check that would catch it being
edited, which is why the digest is recorded and recomputed rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _PACKAGE_ROOT / "src" / "threetears" / "scrape" / "operator_assets"
_NOVNC = _ASSETS / "novnc"
_PROVENANCE = _ASSETS / "novnc-provenance.json"


def _record() -> dict:
    """The provenance record, which is the single source of truth for every claim below."""
    return json.loads(_PROVENANCE.read_text(encoding="utf-8"))


def _tree_digest(root: Path) -> str:
    """Digest the tree exactly as ``tree_sha256_recipe`` describes it.

    Path bytes are folded in alongside contents, so a file that is renamed, added or removed
    moves the digest just as an edited one does. Contents alone would miss all three.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class TestTheVendoredTreeIsWhatWeSayItIs:
    """``modified: false`` is a claim, and this is what makes it a checkable one."""

    def test_the_tree_matches_its_recorded_digest(self) -> None:
        """An edited, added or deleted vendored file fails here rather than shipping quietly.

        If this fails because a file was changed on purpose, the fix is NOT to restamp the
        digest. MPL-2.0 requires a modification to be marked: say so in the file and in the
        record's ``modified`` field, then restamp. Restamping alone converts a licence
        obligation into a green test.
        """
        record = _record()
        assert record["modified"] is False, (
            "the record says the tree is modified, so the marking obligation applies -- "
            "this test only covers the unmodified case"
        )
        assert _tree_digest(_NOVNC) == record["tree_sha256"], (
            "the vendored noVNC tree no longer matches its recorded digest; see the docstring"
        )

    def test_every_path_the_record_names_is_present(self) -> None:
        """A record naming files that are not there documents a tree nobody shipped."""
        named = [key for key in _record()["vendored_subset"] if not key.startswith("$")]
        missing = [name for name in named if not (_NOVNC / name).exists()]
        assert not missing, f"the provenance record names paths that are not vendored: {missing}"


class TestTheLicenceNoticeTravelsWithTheFiles:
    """MPL-2.0 is not satisfied by a line in a README, so the texts ship beside the code."""

    def test_the_upstream_notice_and_every_text_it_points_at_are_present(self) -> None:
        """``LICENSE.txt`` names other texts by path, so dropping one breaks the notice itself."""
        for name in ("LICENSE.txt", "AUTHORS", "vendor/pako/LICENSE"):
            assert (_NOVNC / name).is_file(), f"noVNC's {name} is not vendored"
        for name in ("LICENSE.MPL-2.0", "LICENSE.BSD-2-Clause", "LICENSE.BSD-3-Clause"):
            assert (_NOVNC / "docs" / name).is_file(), f"noVNC's LICENSE.txt points at docs/{name}"

    def test_the_wheel_metadata_does_not_claim_the_vendored_files_are_mit(self) -> None:
        """The whole point of the compound expression, held so it cannot quietly revert.

        Every other package in this family declares plain ``MIT``, so this one is the odd entry
        and an editor sweeping them into line would be doing the wrong thing silently.
        """
        project = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        expression = project["project"]["license"]
        assert "MPL-2.0" in expression, (
            f"the wheel declares {expression!r}, which does not cover the vendored MPL-2.0 files"
        )

    def test_every_licence_text_reaches_the_wheel_metadata(self) -> None:
        """Beside the code is the MPL's requirement; in ``dist-info`` is what tooling reads."""
        project = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(project["project"]["license-files"])
        for name in (
            "LICENSE.txt",
            "AUTHORS",
            "docs/LICENSE.MPL-2.0",
            "docs/LICENSE.BSD-2-Clause",
            "docs/LICENSE.BSD-3-Clause",
            "vendor/pako/LICENSE",
        ):
            path = f"src/threetears/scrape/operator_assets/novnc/{name}"
            assert path in declared, f"{path} is vendored but not declared in license-files"


class TestTheAssetsCanActuallyShip:
    """A tree that is perfect in the repository and absent from the wheel is a dead display.

    Both halves of that sentence get a test, because they are different failures with the same
    symptom. The stock Python ``.gitignore`` carries a bare ``lib/`` rule; it has no leading
    slash, so it matches at any depth, and it matches ``vendor/pako/lib/`` -- every zlib module
    ``core/inflator.js`` and ``core/deflator.js`` import. Twelve files. Left ignored they are
    absent from a fresh clone; re-included with a ``!`` rule they are present for git and STILL
    absent from the wheel, because hatchling reads ignore files with its own matcher and does
    not honour the negation. The working tree looks complete either way and the display dies on
    the first compressed framebuffer update.
    """

    def test_no_vendored_file_is_excluded_by_gitignore(self) -> None:
        """An ignored asset is missing from a fresh clone, so nobody can develop against it."""
        git = shutil.which("git")
        if git is None:
            pytest.skip("git is not available, so ignore rules cannot be evaluated")
        paths = "\n".join(str(p) for p in _NOVNC.rglob("*") if p.is_file())
        result = subprocess.run(  # noqa: S603 - fixed argv, repo-local paths on stdin
            [git, "check-ignore", "--stdin"],
            input=paths,
            capture_output=True,
            text=True,
            cwd=_PACKAGE_ROOT,
            check=False,
        )
        # Exit 1 with no output is check-ignore's "nothing matched", which is the passing case.
        ignored = [line for line in result.stdout.splitlines() if line.strip()]
        assert not ignored, f"vendored assets are excluded by a gitignore rule: {ignored}"

    def test_every_vendored_file_reaches_a_built_wheel(self, tmp_path: Path) -> None:
        """The only check that could have caught the hatchling half, so it builds a real wheel.

        Nothing readable from the source tree distinguishes "hatchling will ship this" from
        "hatchling will drop this" -- the passing gitignore test above coexisted with a wheel
        missing all twelve pako modules. Asserting on the config instead would pin today's
        spelling of the fix rather than the property, and the property is that the files are
        in the archive a deployment installs.
        """
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv is not available, so no wheel can be built")
        result = subprocess.run(  # noqa: S603 - fixed argv, repo-local paths
            [uv, "build", "--wheel", str(_PACKAGE_ROOT), "--out-dir", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"the wheel would not build: {result.stderr}"
        wheels = list(tmp_path.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"

        prefix = "threetears/scrape/operator_assets/novnc/"
        with zipfile.ZipFile(wheels[0]) as wheel:
            # `.dist-info/licenses/` holds copies of the same paths, so match on the package
            # prefix rather than a substring, or a missing asset is masked by its licence twin.
            shipped = {name[len(prefix) :] for name in wheel.namelist() if name.startswith(prefix)}
        vendored = {p.relative_to(_NOVNC).as_posix() for p in _NOVNC.rglob("*") if p.is_file()}
        assert not (vendored - shipped), f"vendored assets are missing from the wheel: {sorted(vendored - shipped)}"
