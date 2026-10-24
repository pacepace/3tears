"""
enforcement: each structured-result border key has ONE construction site.

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

**One row per border key.** The search key is the first; a grounded answer's
key, ``ANSWER_METADATA_KEY``, is the second, sanctioned at
``threetears.search.bind_answer`` (``answers-requirements-design.md`` AD8,
§10.6). Each row names its symbol, its one construction site, and where its
literal spelling is policed. The search key's value is specific enough to
police everywhere; the answer key's value is the English word ``"answer"``,
which the evals package uses for payloads of its own, so its literal guard
covers the two packages that build or carry it -- search and the agent
family -- and only where the word keys a dict literal, which is what a second
construction site looks like. The exemptions file is shared and its format is
unchanged: an entry declares a second payload on a shared key, whichever key.

Scanned as an AST rather than as text, because the thing being detected is a
dict literal keyed by a particular name, and a regex over source cannot tell
``{SEARCH_RESULTS_METADATA_KEY: ...}`` from a docstring mentioning it -- both
builtins' docstrings mention it, correctly, and must keep being able to.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.enforcement.common import parse_python_path


_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONTRACTS = __import__("threetears.search.contracts", fromlist=["SEARCH_RESULTS_METADATA_KEY"])


@dataclass(frozen=True)
class _BorderKey:
    """One border key whose payload has exactly one construction site."""

    #: the constant's name, matched as a *symbol* in dict-literal keys.
    symbol: str
    #: the module allowed to construct the payload. Not a list that grows: a
    #: second construction site is a call to this one, and the review question
    #: is why the projection needs two owners.
    sanctioned_site: Path
    #: where the literal spelling is policed, as roots under the repo.
    literal_roots: tuple[str, ...]
    #: whether the literal is policed only where it keys a dict literal, for a
    #: value too ordinary to police everywhere.
    literal_in_dict_keys_only: bool

    @property
    def value(self) -> str:
        """The literal behind the constant, read from the contract.

        A guard that carries its own copy of the value it protects is the bug.

        :return: the key's value
        :rtype: str
        """
        return str(getattr(_CONTRACTS, self.symbol))


#: where every border-key constant is DEFINED, which necessarily spells it.
_DEFINING_MODULE = Path("packages/search/src/threetears/search/contracts/metadata.py")

#: the rows: one per border key.
_BORDER_KEYS = (
    _BorderKey(
        symbol="SEARCH_RESULTS_METADATA_KEY",
        sanctioned_site=Path("packages/search/src/threetears/search/bind.py"),
        literal_roots=("packages",),
        literal_in_dict_keys_only=False,
    ),
    _BorderKey(
        symbol="ANSWER_METADATA_KEY",
        sanctioned_site=Path("packages/search/src/threetears/search/bind_answer.py"),
        literal_roots=("packages/search", "packages/agent"),
        literal_in_dict_keys_only=True,
    ),
)

#: the search key's row, which the planted-site controls exercise.
_SEARCH_KEY = _BORDER_KEYS[0]

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


def _source_files(roots: tuple[str, ...] = _SCANNED_ROOTS) -> list[Path]:
    """Every shipped ``.py`` under ``roots``, vendored trees excluded."""
    found: list[Path] = []
    for root in roots:
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


@pytest.mark.parametrize("border", _BORDER_KEYS, ids=lambda border: border.symbol)
def test_only_the_sanctioned_site_constructs_the_border_key(border: _BorderKey) -> None:
    """No module but the key's own bind module builds a dict keyed by its constant."""
    exempt = _exempted()
    offenders: list[str] = []

    for path in _source_files():
        relative = _relative(path)
        if relative == border.sanctioned_site or relative in exempt:
            continue
        tree = parse_python_path(path)
        offenders.extend(
            f"{relative}:{key.lineno}"
            for key in _dict_keys_written(tree)
            if isinstance(key, ast.Name) and key.id == border.symbol
        )

    assert not offenders, (
        f"these sites construct {{{border.symbol}: ...}} directly: {offenders}. That is a "
        f"second construction site for the projection, which is how success check 14's "
        f"'no second result shape' regression starts -- two copies that agree until one is "
        f"edited. Call the projection in {border.sanctioned_site} instead; it owns the schema "
        f"version and the field names. If the payload is deliberately different, say so in "
        f"{_EXEMPTIONS_FILE.name} with a rationale."
    )


@pytest.mark.parametrize("border", _BORDER_KEYS, ids=lambda border: border.symbol)
def test_no_module_spells_the_border_key_as_a_literal(border: _BorderKey) -> None:
    """The constant is imported, never re-typed as a string.

    Without this, the guard above is trivially defeated by writing the key's
    value instead of its name -- not maliciously, just by someone who did not
    know the constant existed. A hard-coded spelling also decouples the site
    from the contract, so renaming the key would leave it behind, still
    writing the old name, still parsing on the other side, wrong.
    """
    allowed = {border.sanctioned_site, _DEFINING_MODULE}
    offenders: list[str] = []

    for path in _source_files(border.literal_roots):
        relative = _relative(path)
        if relative in allowed:
            continue
        tree = parse_python_path(path)
        spelled = _dict_keys_written(tree) if border.literal_in_dict_keys_only else list(ast.walk(tree))
        offenders.extend(
            f"{relative}:{node.lineno}"
            for node in spelled
            if isinstance(node, ast.Constant) and node.value == border.value
        )

    assert not offenders, (
        f"these sites hard-code {border.value!r}: {offenders}. Import "
        f"{border.symbol} from threetears.search.contracts instead, so they move when "
        f"the contract does."
    )


@pytest.mark.parametrize("border", _BORDER_KEYS, ids=lambda border: border.symbol)
def test_the_sanctioned_site_actually_constructs_it(border: _BorderKey) -> None:
    """The allowlist names a real site, not a stale path.

    An enforcement test whose sanctioned entry has moved silently enforces
    nothing: every file passes, including the one that took over the job.
    """
    tree = parse_python_path(_REPO_ROOT / border.sanctioned_site)

    constructed = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == border.symbol]

    assert constructed, (
        f"{border.sanctioned_site} no longer constructs {{{border.symbol}: ...}}. Either the "
        f"projection moved -- update its row -- or it was inlined into its callers, which is "
        f"the drift this whole file exists to prevent."
    )


def test_every_border_key_has_one_row() -> None:
    """Two rows for one key would sanction two sites; a key with no row is unguarded."""
    symbols = [border.symbol for border in _BORDER_KEYS]
    assert len(symbols) == len(set(symbols))
    defined = {name for name in dir(_CONTRACTS) if name.endswith("_METADATA_KEY")}
    assert defined == set(symbols), f"border keys without a row: {sorted(defined - set(symbols))}"


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

    tree = parse_python_path(planted)
    offenders = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == _SEARCH_KEY.symbol]

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

    tree = parse_python_path(innocent)
    offenders = [key for key in _dict_keys_written(tree) if isinstance(key, ast.Name) and key.id == _SEARCH_KEY.symbol]

    assert not offenders, "a docstring mentioning the key was treated as a construction site"


def test_a_hard_coded_answer_key_in_a_dict_is_caught_and_the_word_elsewhere_is_not(tmp_path: Path) -> None:
    """The answer row's literal guard: ``{"answer": ...}`` is a second site; ``Literal["answer"]`` is not."""
    answer_key = _BORDER_KEYS[1]
    planted = tmp_path / "second_answer_shape.py"
    planted.write_text(
        "from typing import Literal\n"
        "kind: Literal['answer'] = 'answer'\n"
        f"def project(x):\n    return {{{answer_key.value!r}: x}}\n",
        encoding="utf-8",
    )

    tree = parse_python_path(planted)
    spelled = [
        node for node in _dict_keys_written(tree) if isinstance(node, ast.Constant) and node.value == answer_key.value
    ]

    assert answer_key.literal_in_dict_keys_only
    assert [node.lineno for node in spelled] == [4], "the dict key is the offence, and only it"
