"""tests for the cache-enforcement walkers.

the central test class is :class:`TestTransitiveCollectionDetection` —
it proves the bug fix from the canonical works: chains like
``MemoriesCollection → SchemaBackedCollection → BaseCollection``
resolve to a Collection across files (and across path-dep package
boundaries via the multi-root fixture). the canonical's
``_class_subclasses_base_collection`` only walked direct AST bases
and missed every intermediate base.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.cache import (
    find_direct_pool_access,
    find_missing_collections,
    find_sqlite_constructions,
    find_wrapper_classes,
)


_DEFAULT_BASE_NAMES = frozenset({"BaseCollection"})
_DEFAULT_CACHE_METHODS = frozenset({"get", "put", "set", "delete", "upsert"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


# ----------------------------------------------------------------------
# walker 1 — find_sqlite_constructions
# ----------------------------------------------------------------------


class TestFindSqliteConstructions:
    def test_bare_construction_in_src_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            "from x import SQLiteBackend\n"
            "def make() -> None:\n"
            "    backend = SQLiteBackend('/tmp/foo')\n",
        )
        violations = find_sqlite_constructions((src,), repo, frozenset())
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "cache.sqlite_construction"
        assert v.symbol == "SQLiteBackend"
        assert v.line == 3

    def test_construction_in_tests_dir_skipped(self, tmp_path: Path) -> None:
        # tests/ trees are always permitted regardless of allowed_sites.
        repo = _make_repo(tmp_path / "repo")
        tests = repo / "tests"
        _write(
            tests / "test_foo.py",
            "from x import SQLiteBackend\n"
            "def test_thing() -> None:\n"
            "    backend = SQLiteBackend('/tmp/foo')\n",
        )
        # scan tests as a src root to verify the walker still skips it.
        assert find_sqlite_constructions((tests,), repo, frozenset()) == []

    def test_allowed_site_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "factory.py",
            "from x import SQLiteBackend\n"
            "def make() -> None:\n"
            "    backend = SQLiteBackend('/tmp/foo')\n",
        )
        allowed = frozenset({"src/factory.py"})
        assert find_sqlite_constructions((src,), repo, allowed) == []

    def test_dotted_call_target_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            "import threetears.core.cache as cache\n"
            "def make() -> None:\n"
            "    backend = cache.SQLiteBackend('/tmp/foo')\n",
        )
        violations = find_sqlite_constructions((src,), repo, frozenset())
        assert len(violations) == 1
        assert violations[0].symbol == "SQLiteBackend"

    def test_unrelated_call_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            "def make() -> None:\n    backend = SomeOther('/tmp/foo')\n",
        )
        assert find_sqlite_constructions((src,), repo, frozenset()) == []


# ----------------------------------------------------------------------
# walker 2 — find_wrapper_classes (transitive walk required)
# ----------------------------------------------------------------------


class TestFindWrapperClasses:
    def test_class_with_sqlite_field_and_cache_method_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend\n"
                "class CacheWrapper:\n"
                "    def __init__(self) -> None:\n"
                "        self._backend = SQLiteBackend('/tmp/foo')\n"
                "    def get(self, key: str):\n"
                "        return None\n"
            ),
        )
        violations = find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "cache.wrapper_class"
        assert v.symbol == "CacheWrapper"
        assert "get" in v.reason

    def test_direct_base_collection_subclass_skipped(
        self, tmp_path: Path,
    ) -> None:
        # class extends BaseCollection directly -> not a wrapper.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend, BaseCollection\n"
                "class CacheWrapper(BaseCollection):\n"
                "    def __init__(self) -> None:\n"
                "        self._backend = SQLiteBackend('/tmp/foo')\n"
                "    def get(self, key: str):\n"
                "        return None\n"
            ),
        )
        assert find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        ) == []

    def test_transitive_collection_subclass_skipped(
        self, tmp_path: Path,
    ) -> None:
        # the bug fix: CacheWrapper(SchemaBackedCollection) where
        # SchemaBackedCollection(BaseCollection) lives in a sibling
        # file; transitive walk must reach BaseCollection.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "bases.py",
            (
                "class BaseCollection: pass\n"
                "class SchemaBackedCollection(BaseCollection): pass\n"
            ),
        )
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend\n"
                "from .bases import SchemaBackedCollection\n"
                "class MemoriesCollection(SchemaBackedCollection):\n"
                "    def __init__(self) -> None:\n"
                "        self._backend = SQLiteBackend('/tmp/foo')\n"
                "    def get(self, key: str):\n"
                "        return None\n"
            ),
        )
        # MUST be 0: transitive walk reaches BaseCollection through
        # SchemaBackedCollection, even though BaseCollection is NEVER
        # a direct AST base of MemoriesCollection.
        assert find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        ) == []

    def test_class_no_sqlite_field_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "class CacheWrapper:\n"
                "    def __init__(self) -> None:\n"
                "        self._items = {}\n"
                "    def get(self, key: str):\n"
                "        return None\n"
            ),
        )
        assert find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        ) == []

    def test_class_no_cache_method_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend\n"
                "class HoldingClass:\n"
                "    def __init__(self) -> None:\n"
                "        self._backend = SQLiteBackend('/tmp/foo')\n"
                "    def helper_method(self):\n"
                "        pass\n"
            ),
        )
        assert find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        ) == []

    def test_constructor_injection_pattern_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend\n"
                "class CacheWrapper:\n"
                "    def __init__(self, l1: SQLiteBackend) -> None:\n"
                "        self._backend = l1\n"
                "    def get(self, key: str):\n"
                "        return None\n"
            ),
        )
        violations = find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "CacheWrapper"

    def test_domain_verb_wrapper_via_data_api_flagged(
        self, tmp_path: Path,
    ) -> None:
        # no public cache verb but calls SQLiteBackend's data api
        # (``upsert`` etc.) directly — fingerprint of a domain-verb
        # wrapper.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "wrapper.py",
            (
                "from x import SQLiteBackend\n"
                "class TokenRevocation:\n"
                "    def __init__(self, l1: SQLiteBackend) -> None:\n"
                "        self._backend = l1\n"
                "    def revoke(self, token: str):\n"
                "        self._backend.upsert(token, {})\n"
            ),
        )
        violations = find_wrapper_classes(
            (src,), repo, (src,), _DEFAULT_BASE_NAMES, _DEFAULT_CACHE_METHODS,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "TokenRevocation"
        assert "data api" in violations[0].reason


# ----------------------------------------------------------------------
# walker 3 — find_direct_pool_access
# ----------------------------------------------------------------------


class TestFindDirectPoolAccess:
    def test_pool_fetch_on_collection_table_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "async def load(pool):\n"
                "    return await pool.fetch('SELECT * FROM memories')\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_direct_pool_access((src,), repo, allowlist)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "cache.pool_access"
        assert v.symbol == "memories"

    def test_pool_fetch_on_unmapped_table_skipped(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "async def load(pool):\n"
                "    return await pool.fetch('SELECT * FROM logs')\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        assert find_direct_pool_access((src,), repo, allowlist) == []

    def test_non_pool_receiver_skipped(self, tmp_path: Path) -> None:
        # ``db.fetch`` doesn't have ``pool``/``store`` in the receiver
        # name, so it's filtered out by the receiver-name check.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "async def load(db):\n"
                "    return await db.fetch('SELECT * FROM memories')\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        assert find_direct_pool_access((src,), repo, allowlist) == []

    def test_call_inside_collection_class_skipped(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "class MemoriesCollection:\n"
                "    async def all(self, pool):\n"
                "        return await pool.fetch('SELECT * FROM memories')\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        assert find_direct_pool_access((src,), repo, allowlist) == []

    def test_cache_bypass_comment_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "async def load(pool):\n"
                "    # cache-bypass: cross-table JOIN constraint\n"
                "    return await pool.fetch('SELECT * FROM memories')\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        assert find_direct_pool_access((src,), repo, allowlist) == []

    def test_executemany_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "foo.py",
            (
                "async def bulk(pool, rows):\n"
                "    await pool.executemany("
                "'INSERT INTO memories VALUES ($1)', rows)\n"
            ),
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_direct_pool_access((src,), repo, allowlist)
        assert len(violations) == 1
        assert violations[0].symbol == "memories"


# ----------------------------------------------------------------------
# walker 4 — find_missing_collections (the bug fix)
# ----------------------------------------------------------------------


class TestTransitiveCollectionDetection:
    """the bug fix: chains through intermediate Collection bases resolve."""

    def test_direct_base_collection_subclass_resolves(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "bases.py",
            "class BaseCollection: pass\n",
        )
        _write(
            src / "leaves.py",
            "from .bases import BaseCollection\n"
            "class Leaf(BaseCollection): pass\n",
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE leaves (id INT)"\n',
        )
        allowlist = {"leaves": "Leaf"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert violations == []

    def test_two_hop_chain_resolves(self, tmp_path: Path) -> None:
        # the canonical bug fix: Foo -> Mid -> BaseCollection. canonical
        # only walked direct bases and would have missed Mid.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "bases.py",
            "class Base: pass\n"
            "class Mid(Base): pass\n",
        )
        _write(
            src / "leaves.py",
            "from .bases import Mid\nclass Leaf(Mid): pass\n",
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE leaves (id INT)"\n',
        )
        # the terminal target is "Base"; transitive walk must reach
        # it through Mid.
        allowlist = {"leaves": "Leaf"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), frozenset({"Base"}),
        )
        # 0 violations -- THE BUG FIX. canonical's direct-only walk
        # would have flagged this as a missing Collection because
        # Leaf's direct AST base is Mid, not Base.
        assert violations == []

    def test_three_hop_chain_resolves(self, tmp_path: Path) -> None:
        # the metallm production chain: MemoriesCollection ->
        # SchemaBackedCollection -> BaseCollection. proves the
        # transitive walk handles arbitrary depth.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "bases.py",
            (
                "class BaseCollection: pass\n"
                "class SchemaBackedCollection(BaseCollection): pass\n"
            ),
        )
        _write(
            src / "memories.py",
            (
                "from .bases import SchemaBackedCollection\n"
                "class MemoriesCollection(SchemaBackedCollection): pass\n"
            ),
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE memories (id INT)"\n',
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        # 0 violations: MemoriesCollection -> SchemaBackedCollection ->
        # BaseCollection. canonical (10 spurious violations).
        assert violations == []

    def test_path_dep_walk_across_packages(self, tmp_path: Path) -> None:
        # cross-package: bases live in package A, leaf in package B,
        # migration in package B. proves the walk works regardless
        # of which src root holds which class.
        repo = _make_repo(tmp_path / "repo")
        a_src = repo / "package_a" / "src"
        b_src = repo / "package_b" / "src"
        _write(
            a_src / "bases.py",
            (
                "class BaseCollection: pass\n"
                "class SchemaBackedCollection(BaseCollection): pass\n"
            ),
        )
        _write(
            b_src / "memories.py",
            (
                "from package_a.bases import SchemaBackedCollection\n"
                "class MemoriesCollection(SchemaBackedCollection): pass\n"
            ),
        )
        _write(
            b_src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE memories (id INT)"\n',
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_missing_collections(
            (a_src, b_src),
            repo,
            (a_src, b_src),
            allowlist,
            frozenset(),
            _DEFAULT_BASE_NAMES,
        )
        # 0 violations: the graph spans both src roots; transitive
        # walk reaches BaseCollection through SchemaBackedCollection
        # in package_a.
        assert violations == []


class TestFindMissingCollectionsViolations:
    def test_table_not_in_allowlist_violation(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE strangers (id INT)"\n',
        )
        violations = find_missing_collections(
            (src,), repo, (src,), {}, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "cache.missing_collection"
        assert v.symbol == "strangers"
        assert "no mapping" in v.reason

    def test_table_in_migration_allowlist_skipped(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE _schema_migrations (id INT)"\n',
        )
        violations = find_missing_collections(
            (src,),
            repo,
            (src,),
            {},
            frozenset({"_schema_migrations"}),
            _DEFAULT_BASE_NAMES,
        )
        assert violations == []

    def test_mapped_class_not_a_collection_violation(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "models.py",
            "class Memory: pass\n",
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE memories (id INT)"\n',
        )
        allowlist = {"memories": "Memory"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "memories"
        assert "BaseCollection" in v.reason

    def test_mapped_class_missing_violation(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE memories (id INT)"\n',
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "memories"

    def test_schema_prefix_stripped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "leaves.py",
            (
                "class BaseCollection: pass\n"
                "class CustomerCollection(BaseCollection): pass\n"
            ),
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE platform.customers (id INT)"\n',
        )
        allowlist = {"customers": "CustomerCollection"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert violations == []

    def test_helper_module_not_a_migration(self, tmp_path: Path) -> None:
        # files like template.py / drift.py / __init__.py are NOT
        # migrations even when they live under migrations/ — the
        # filename pattern excludes them.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "migrations" / "template.py",
            'SQL = "CREATE TABLE mystery (id INT)"\n',
        )
        violations = find_missing_collections(
            (src,), repo, (src,), {}, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert violations == []

    def test_create_table_if_not_exists_parsed(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "leaves.py",
            (
                "class BaseCollection: pass\n"
                "class MemoriesCollection(BaseCollection): pass\n"
            ),
        )
        _write(
            src / "migrations" / "v001_init.py",
            'SQL = "CREATE TABLE IF NOT EXISTS memories (id INT)"\n',
        )
        allowlist = {"memories": "MemoriesCollection"}
        violations = find_missing_collections(
            (src,), repo, (src,), allowlist, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert violations == []

    def test_docstring_create_table_ignored(self, tmp_path: Path) -> None:
        # docstrings don't run; CREATE TABLE in a module/function
        # docstring must NOT trigger detection.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "migrations" / "v001_init.py",
            (
                '"""docstring with CREATE TABLE example_phantom (id INT)"""\n'
                "x = 1\n"
            ),
        )
        violations = find_missing_collections(
            (src,), repo, (src,), {}, frozenset(), _DEFAULT_BASE_NAMES,
        )
        assert violations == []
