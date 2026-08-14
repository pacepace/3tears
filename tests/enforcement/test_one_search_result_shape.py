"""
enforcement: the search-results border key has ONE construction site.

Success check 14 is "three faces, one contract". The unit half of it lives in
``packages/agent/tools/tests/unit/tools/test_three_faces_one_contract.py``,
which drives one candidate set through the renderings and asserts they agree.
This is the other half, and it guards a failure the unit test structurally
cannot: a **new** construction site, in a module nothing compares against.

The risk §5.5 names is that "a face gets added, someone shapes a response for
it, and the second result shape is born". A test that compares the faces it
knows about cannot see a face it does not. So the shape is protected at its
source instead: ``SEARCH_RESULTS_METADATA_KEY`` may be READ anywhere, and may
be WRITTEN in exactly one place -- ``threetears.search.bind``, whose
``project_metadata`` / ``project_failure_metadata`` own the schema version and
the field names.

This is the ``ObjectHandle.to_metadata`` pattern the projection was modelled
on, held by a test: "the payload is built by a named method that owns its
schema version, not dumped at the call site where the shape would drift per
caller" (``bind.py``).

**It has already caught one.** ``web_fetch`` reimplemented the projection
rather than calling it -- ``SearchResultsMetadata.from_candidate_set(...)``
followed by ``{SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()}``, twice,
once for the success path and once for refusals. The output was identical to
``bind``'s, which is precisely why nothing failed and why it would have stayed
identical only for as long as nobody edited either copy.

Scanned as an AST rather than as text, because the thing being detected is a
dict literal keyed by a particular name, and a regex over source cannot tell
``{SEARCH_RESULTS_METADATA_KEY: ...}`` from a docstring mentioning it -- both
builtins' docstrings mention it, correctly, and must keep being able to.
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The name whose payload shape is being protected. Matched as a *symbol*: a
#: site that spells the string literally instead of importing the constant is
#: caught by :func:`test_no_module_spells_the_border_key_as_a_literal` below.
_BORDER_KEY = "SEARCH_RESULTS_METADATA_KEY"

#: The literal value behind that constant, so the second guard can find a site
#: that hard-codes it. Read from the contract rather than restated here -- a
#: guard that carries its own copy of the value it protects is the bug.
_BORDER_KEY_VALUE = __import__("threetears.search.contracts", fromlist=[_BORDER_KEY]).SEARCH_RESULTS_METADATA_KEY

#: The one module allowed to construct the payload. Not a list that grows: if a
#: second entry is ever proposed, the thing to write is a call to ``bind``, and
#: the review question is why this projection needs two owners.
_SANCTIONED_SITE = Path("packages/search/src/threetears/search/bind.py")

#: Trees scanned. Source only -- tests construct expected payloads on purpose,
#: which is what makes them tests.
_SCANNED_ROOTS = ("packages",)

#: Directory names that end a walk. ``.venv`` and ``site-packages`` matter
#: concretely rather than defensively: the scrape sidecar vendors nodriver into
#: its own virtualenv (AGPL, deliberately outside the workspace venv), and the
#: first run of this guard tried to parse a CDP file that is not even valid
#: UTF-8. Third-party code is not ours to hold to this rule.
_SKIPPED_DIRS = frozenset({".venv", "site-packages", "__pycache__", "node_modules", "tests", "build", "dist"})

#: Deliberate second payloads under the border key, each with a rationale.
_EXEMPTIONS_FILE = Path(__file__).with_name("_one_search_result_shape_exemptions.txt")


def _exempted() -> set[Path]:
    """Paths declared as carrying a second, intentional payload.

    Each entry must be preceded by a rationale line, enforced below -- an
    exemptions file that accepts bare paths becomes a list of things nobody
    remembers deciding.
    """
    exempt: set[Path] = set()
    rationale: str | None = None
    for raw in _EXEMPTIONS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# rationale:"):
            rationale = line.removeprefix("# rationale:").strip()
            continue
        if line.startswith("#"):
            continue
        assert rationale and len(rationale) >= 30, (
            f"{_EXEMPTIONS_FILE.name}: {line!r} has no rationale line, or one too short to "
            f"be a reason. Every exemption declares a second payload on a shared key; say why."
        )
        exempt.add(Path(line))
        rationale = None
    return exempt


def _source_files() -> list[Path]:
    """Every shipped ``.py`` under the scanned roots, vendored trees excluded."""
    found: list[Path] = []
    for root in _SCANNED_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if _SKIPPED_DIRS & set(path.parts):
                continue
            found.append(path)
    return sorted(found)


def _dict_keys_written(tree: ast.AST) -> list[ast.expr]:
    """Every key expression of every dict literal in ``tree``.

    Dict *literals* specifically. ``d[KEY] = value`` on an existing mapping is
    a different act -- merging a projection somebody else built into a wider
    metadata dict, which every caller is supposed to do -- and is not what this
    guard is about.
    """
    keys: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.extend(key for key in node.keys if key is not None)
    return keys


def _relative(path: Path) -> Path:
    return path.relative_to(_REPO_ROOT)


def test_only_bind_constructs_the_border_key() -> None:
    """No module but ``bind`` builds a dict keyed by the border constant."""
    exempt = _exempted()
    offenders: list[str] = []

    for path in _source_files():
        relative = _relative(path)
        if relative == _SANCTIONED_SITE or relative in exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{relative}:{key.lineno}"
            for key in _dict_keys_written(tree)
            if isinstance(key, ast.Name) and key.id == _BORDER_KEY
        )

    assert not offenders, (
        f"these sites construct {{{_BORDER_KEY}: ...}} directly: {offenders}. That is a "
        f"second construction site for the search-results projection, which is how success "
        f"check 14's 'no second result shape' regression starts -- two copies that agree "
        f"until one is edited. Call threetears.search.bind.project_metadata (or "
        f"project_failure_metadata) instead; it owns the schema version and the field "
        f"names. If the payload is deliberately different, say so in "
        f"{_EXEMPTIONS_FILE.name} with a rationale."
    )


def test_no_module_spells_the_border_key_as_a_literal() -> None:
    """The constant is imported, never re-typed as a string.

    Without this, the guard above is trivially defeated by writing the key's
    value instead of its name -- not maliciously, just by someone who did not
    know the constant existed. A hard-coded spelling also decouples the site
    from the contract, so renaming the key would leave it behind, still
    writing the old name, still parsing on the other side, wrong.
    """
    allowed = {
        _SANCTIONED_SITE,
        # where the constant is DEFINED, which necessarily spells it
        Path("packages/search/src/threetears/search/contracts/metadata.py"),
    }
    offenders: list[str] = []

    for path in _source_files():
        relative = _relative(path)
        if relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{relative}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == _BORDER_KEY_VALUE
        )

    assert not offenders, (
        f"these sites hard-code {_BORDER_KEY_VALUE!r}: {offenders}. Import "
        f"{_BORDER_KEY} from threetears.search.contracts instead, so they move when "
        f"the contract does."
    )


def test_the_sanctioned_site_actually_constructs_it() -> None:
    """The allowlist names a real site, not a stale path.

    An enforcement test whose sanctioned entry has moved silently enforces
    nothing: every file passes, including the one that took over the job.
    """
    tree = ast.parse((_REPO_ROOT / _SANCTIONED_SITE).read_text(encoding="utf-8"))

    constructed = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == _BORDER_KEY]

    assert constructed, (
        f"{_SANCTIONED_SITE} no longer constructs {{{_BORDER_KEY}: ...}}. Either the "
        f"projection moved -- update _SANCTIONED_SITE -- or it was inlined into its "
        f"callers, which is the drift this whole file exists to prevent."
    )


def test_the_guard_catches_a_planted_second_site(tmp_path: Path) -> None:
    """Prove the AST walk sees what it claims to see.

    The Gate B sweep shipped an independence test whose first draft passed
    under the exact regression it was written to catch. This runs the walk over
    a planted offender and confirms it is found, so the guard's direction is
    demonstrated rather than assumed.
    """
    planted = tmp_path / "second_shape.py"
    planted.write_text(
        "from threetears.search.contracts import SEARCH_RESULTS_METADATA_KEY\n"
        "def project(x):\n"
        "    return {SEARCH_RESULTS_METADATA_KEY: {'results': x}}\n",
        encoding="utf-8",
    )

    tree = ast.parse(planted.read_text(encoding="utf-8"))
    offenders = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == _BORDER_KEY]

    assert offenders, "the AST walk missed a dict literal keyed by the border constant"


def test_a_docstring_mention_is_not_an_offence(tmp_path: Path) -> None:
    """The complement: prose about the key must stay writable.

    Both builtins document the key they project under, correctly. A guard that
    flagged them would be deleted within a week, which is worse than no guard.
    """
    innocent = tmp_path / "documented.py"
    innocent.write_text(
        '"""Projects under ``{SEARCH_RESULTS_METADATA_KEY: ...}`` (D22)."""\ndef project(x):\n    return None\n',
        encoding="utf-8",
    )

    tree = ast.parse(innocent.read_text(encoding="utf-8"))
    offenders = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == _BORDER_KEY]

    assert not offenders, "a docstring mentioning the key was treated as a construction site"
