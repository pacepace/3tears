"""tests for the dict-state-detection walkers."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.dict_state_detection import (
    DictStateAllowlistEntry,
    filter_against_allowlist,
    find_dict_state_violations,
    find_stale_allowlist_entries,
)


_VALID_RATIONALE = (
    "live LangChain ChatModel instances; non-serializable, "
    "process-local by design"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


# ------------------------------------------------------------------
# detection: bad-shape positives
# ------------------------------------------------------------------


class TestDictLiteralDetection:
    def test_empty_dict_literal_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "dict_state_detection.dict_in_init"
        assert v.symbol == "_cache"
        assert v.line == 3
        assert "empty-dict literal" in v.reason
        assert "Foo._cache" in v.reason


class TestDictCallDetection:
    def test_dict_call_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = dict()\n",
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_cache"
        assert "`dict()` call" in violations[0].reason

    def test_dict_call_with_args_not_flagged(self, tmp_path: Path) -> None:
        # initialised dict is a deliberate one-shot use, not a state slot.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = dict(a=1)\n",
        )
        assert find_dict_state_violations((src,), repo) == []


class TestOrderedDictCallDetection:
    def test_ordered_dict_call_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "from collections import OrderedDict\n"
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._cache = OrderedDict()\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_cache"
        assert "`OrderedDict()` call" in violations[0].reason


class TestOrFallbackDetection:
    def test_or_empty_dict_fallback_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self, opts=None) -> None:\n"
                "        self._opts = opts or {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_opts"
        assert "fallback" in violations[0].reason

    def test_or_with_non_dict_fallback_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self, opts=None) -> None:\n"
                "        self._opts = opts or None\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []


class TestAnnotatedDetection:
    def test_dict_typed_annotation_with_empty_value(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._cache: dict[str, int] = {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_cache"
        assert "annotated empty-dict literal" in violations[0].reason

    def test_bare_dict_annotation_with_empty_value(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._cache: dict = {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_cache"

    def test_dict_annotation_with_clean_value_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        # annotation + clean value is fine.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "def make_cache():\n    return {}\n"
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._cache: dict[str, int] = make_cache()\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []


# ------------------------------------------------------------------
# detection: negatives
# ------------------------------------------------------------------


class TestNegatives:
    def test_list_assignment_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._items = []\n",
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_set_assignment_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._seen = set()\n",
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_other_class_construction_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Backend: pass\n"
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._backend = Backend()\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_function_call_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "def make_thing(): return 1\n"
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._thing = make_thing()\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_assignment_outside_init_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def setup(self) -> None:\n"
                "        self._cache = {}\n"
                "    def __init__(self) -> None:\n"
                "        pass\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_module_level_assignment_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "_GLOBAL_CACHE: dict = {}\n")
        assert find_dict_state_violations((src,), repo) == []

    def test_local_assignment_in_init_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        # only self._x flagged; locals are ignored.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        local = {}\n"
                "        self.x = local\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_public_self_attribute_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        # walker only flags self._x (single underscore prefix).
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self.cache = {}\n",
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_other_object_attribute_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self, other) -> None:\n"
                "        other._cache = {}\n"
            ),
        )
        assert find_dict_state_violations((src,), repo) == []

    def test_async_init_scanned(self, tmp_path: Path) -> None:
        # async __init__ is exotic but the canonical handles it.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    async def __init__(self) -> None:  # type: ignore[misc]\n"
                "        self._cache = {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "_cache"


# ------------------------------------------------------------------
# allowlist filter
# ------------------------------------------------------------------


class TestFilterAgainstAllowlist:
    def test_allowlist_removes_matching_violation(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 1
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        true_v, allowed = filter_against_allowlist(
            violations, (entry,), (), repo,
        )
        assert true_v == []
        assert len(allowed) == 1
        assert allowed[0].symbol == "_cache"

    def test_known_violations_remove_matching_violation(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        violations = find_dict_state_violations((src,), repo)
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        true_v, allowed = filter_against_allowlist(
            violations, (), (entry,), repo,
        )
        assert true_v == []
        assert len(allowed) == 1

    def test_non_matching_allowlist_passes_through(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        violations = find_dict_state_violations((src,), repo)
        # wrong line number: walker reports line 3, allowlist references 99.
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=99,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        true_v, allowed = filter_against_allowlist(
            violations, (entry,), (), repo,
        )
        assert len(true_v) == 1
        assert allowed == []

    def test_partial_allowlist_filters_only_matching(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "class Foo:\n"
                "    def __init__(self) -> None:\n"
                "        self._a = {}\n"
                "        self._b = {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 2
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_a",
            rationale=_VALID_RATIONALE,
        )
        true_v, allowed = filter_against_allowlist(
            violations, (entry,), (), repo,
        )
        assert len(true_v) == 1
        assert true_v[0].symbol == "_b"
        assert len(allowed) == 1
        assert allowed[0].symbol == "_a"


# ------------------------------------------------------------------
# stale-entry meta-walker
# ------------------------------------------------------------------


class TestStaleAllowlist:
    def test_stale_allowlist_entry_emits_violation(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # source is clean, so the entry is stale.
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        pass\n",
        )
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        stale = find_stale_allowlist_entries(
            (src,), (entry,), (), repo,
        )
        assert len(stale) == 1
        v = stale[0]
        assert v.category == "dict_state_detection.stale_allowlist"
        assert v.symbol == "_cache"
        assert v.line == 3
        assert "no longer matches" in v.reason

    def test_stale_known_violation_distinct_category(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        pass\n",
        )
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        stale = find_stale_allowlist_entries(
            (src,), (), (entry,), repo,
        )
        assert len(stale) == 1
        v = stale[0]
        assert v.category == "dict_state_detection.stale_known_violation"

    def test_matching_entry_not_stale(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        assert find_stale_allowlist_entries((src,), (entry,), (), repo) == []

    def test_mixed_stale_and_matching(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
        )
        match = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        stale = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=42,
            attr_name="_phantom",
            rationale=_VALID_RATIONALE,
        )
        result = find_stale_allowlist_entries(
            (src,), (match, stale), (), repo,
        )
        assert len(result) == 1
        assert result[0].symbol == "_phantom"

    def test_allowlist_and_known_violations_audited_separately(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        pass\n",
        )
        a = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_a",
            rationale=_VALID_RATIONALE,
        )
        b = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=4,
            attr_name="_b",
            rationale=_VALID_RATIONALE,
        )
        result = find_stale_allowlist_entries(
            (src,), (a,), (b,), repo,
        )
        cats = sorted(v.category for v in result)
        assert cats == [
            "dict_state_detection.stale_allowlist",
            "dict_state_detection.stale_known_violation",
        ]


# ------------------------------------------------------------------
# multi-root walking
# ------------------------------------------------------------------


class TestMultiRoot:
    def test_violations_across_two_roots(self, tmp_path: Path) -> None:
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "class A:\n    def __init__(self) -> None:\n        self._x = {}\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py",
            "class B:\n    def __init__(self) -> None:\n        self._y = dict()\n",
        )
        violations = find_dict_state_violations((a_src, b_src), tmp_path)
        assert len(violations) == 2
        symbols = sorted(v.symbol for v in violations)
        assert symbols == ["_x", "_y"]


# ------------------------------------------------------------------
# multiple violations per init
# ------------------------------------------------------------------


class TestMultipleInOneInit:
    def test_multiple_violations_in_single_init(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            (
                "from collections import OrderedDict\n"
                "class Foo:\n"
                "    def __init__(self, opts=None) -> None:\n"
                "        self._a = {}\n"
                "        self._b = dict()\n"
                "        self._c = OrderedDict()\n"
                "        self._d = opts or {}\n"
                "        self._e: dict[str, int] = {}\n"
            ),
        )
        violations = find_dict_state_violations((src,), repo)
        assert len(violations) == 5
        symbols = sorted(v.symbol for v in violations)
        assert symbols == ["_a", "_b", "_c", "_d", "_e"]
