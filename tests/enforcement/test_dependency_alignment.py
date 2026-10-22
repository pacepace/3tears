"""thin shell — walker logic in :mod:`threetears.enforcement.dependency_alignment`.

the 3tears self-test consumes the shared ``3tears-enforcement``
workspace package and injects only the per-repo configuration. the
walkers, exemption parser, mode resolver, and report emitter live in
the package; this file declares the knobs and calls the runner.

mode is controlled by ``DEPENDENCY_ALIGNMENT_ENFORCEMENT_MODE`` —
defaults to ``strict``. catches the drift class where the uv workspace
masks undeclared (or stale) cross-package dependencies until a
standalone ``pip install`` of one package ImportErrors in a consumer.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from threetears.enforcement.dependency_alignment import (
    DependencyAlignmentConfig,
    DependencyFloor,
    run_dependency_alignment_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: packages whose hard-dependency list is an argued ruling, not an accretion.
#:
#: ``media-contracts`` and ``tool-schema`` need no entry: their floor is *nothing*, which
#: ``contract_packages`` above already states more strongly. This list is for the
#: packages that cannot be dependency-free but whose floor was still ruled -- where
#: "nothing" is unavailable and "these three" is the promise instead.
_DEPENDENCY_FLOORS = (
    DependencyFloor(
        package="packages/search",
        allowed=("3tears-media-contracts", "3tears-observe", "pydantic", "pydantic-core"),
        rationale=(
            "search-spec.md D24, matching SR-L7's permitted floor exactly. The leaf's whole "
            "value to a constrained host (samsung on a Pi; any embedded consumer) is that "
            "taking it costs pydantic plus two dependency-free family leaves and nothing "
            "else -- so httpx and trafilatura ride the [standalone] and [extract] extras, and "
            "the provider adapters, being pure logic over an injected transport, add no "
            "weight at all. A fourth hard dependency arriving here is exactly the drift that "
            "makes a leaf untakeable, and it would arrive one innocent import at a time"
        ),
    ),
    DependencyFloor(
        package="packages/mcp",
        allowed=("3tears-observe", "mcp", "httpx", "starlette", "uvicorn", "jsonschema"),
        rationale=(
            "owner ruling 2026-10-17: a host that wants an MCP server -- testament, which is outside "
            "the family's platform -- takes the server, the tools, identity and PlatformHttpClient and "
            "nothing else. The grant store (LocalGrantAuthorizer, rbac, migrations) is heavily used by "
            "other apps but is the only part that needs core, epoch, nats and sqlalchemy, so it rides "
            "the [grants] extra; as a hard dependency it put nats-py, asyncpg and sqlalchemy into every "
            "server, though a server with its own Authorizer loads none of them. jsonschema added 2026-10-22 "
            "(pending owner sign-off): mcp 2.x's low-level server no longer checks a call's arguments against "
            "the tool's inputSchema, so McpServer does, with the library mcp itself depends on "
            "(mcp requires jsonschema>=4.20) -- declared, it adds nothing to an install"
        ),
    ),
    DependencyFloor(
        package="packages/evals",
        allowed=("httpx", "pydantic"),
        rationale=(
            "owner ruling 2026-10-11: 3tears-evals depends on no other 3tears package. A base install "
            "with no family dependency installs beside a family pinned at any other version, which is "
            "what let the production host run pre-release tags against a frozen family and what lets a "
            "product outside the family adopt it. The fastmcp transport is the one extra"
        ),
    ),
)

#: why every package below is pinned at the dependencies it had, rather than at an argued floor.
_AS_FOUND_RATIONALE = (
    "owner ruling 2026-10-17: every package's hard dependencies are pinned, so a new one arrives as a "
    "visible edit to tests/enforcement/test_dependency_alignment.py instead of one line in a pyproject. "
    "This floor was recorded as found that day and has not been argued. Before adding to it, ask whether "
    "the new dependency serves the package's core path or belongs behind an extra (family-convergence.md "
    "section 3, principle 7); arguing the floor moves the entry into _DEPENDENCY_FLOORS with its own rationale"
)

#: the packages with hard dependencies and no argued floor yet, pinned at what they declared on
#: 2026-10-17. An entry leaves this list when its floor is argued and moves to ``_DEPENDENCY_FLOORS``.
_RECORDED_AS_FOUND: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "packages/agent/acl",
        (
            "3tears",
            "3tears-observe",
            "pydantic",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/audit",
        (
            "3tears",
            "3tears-nats",
            "3tears-observe",
            "pydantic",
        ),
    ),
    (
        "packages/agent/identity",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-langgraph",
            "3tears-observe",
            "langchain-core",
            "pydantic",
            "uuid-utils",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/intention",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-langgraph",
            "3tears-observe",
            "pgvector",
            "langchain-core",
            "pydantic",
            "uuid-utils",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/knowledge",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-agent-memory",
            "3tears-langgraph",
            "3tears-nats",
            "3tears-observe",
            "pgvector",
            "langchain-core",
            "langchain",
            "langgraph",
        ),
    ),
    (
        "packages/agent/memory",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-langgraph",
            "3tears-nats",
            "3tears-observe",
            "pgvector",
            "uuid-utils",
            "langchain-core",
            "langchain",
            "langgraph",
            "pydantic",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/skills",
        (
            "3tears",
            "3tears-observe",
            "langchain-core",
            "pydantic",
            "uuid-utils",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/tools",
        (
            "3tears",
            "3tears-agent-audit",
            "3tears-agent-memory",
            "3tears-epoch",
            "3tears-langgraph",
            "3tears-media-contracts",
            "3tears-nats",
            "3tears-observe",
            "3tears-search",
            "3tears-tool-schema",
            "langchain-core",
            "httpx",
            "prometheus-client",
            "pydantic",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/wake",
        (
            "3tears",
            "3tears-agent-skills",
            "3tears-observe",
            "3tears-scheduled-jobs",
            "uuid-utils",
            "apscheduler",
            "langchain-core",
            "pydantic",
            "jinja2",
            "sqlalchemy",
        ),
    ),
    (
        "packages/agent/workspace",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-agent-audit",
            "3tears-agent-tools",
            "3tears-nats",
            "3tears-observe",
            "pydantic",
            "ruamel.yaml",
            "jsonpath-ng",
            "watchfiles",
            "asyncpg",
        ),
    ),
    (
        "packages/backup",
        (
            "3tears",
            "3tears-media-contracts",
            "3tears-object-store",
            "3tears-observe",
            "pydantic",
        ),
    ),
    (
        "packages/channels",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-nats",
            "3tears-observe",
            "pydantic",
            "sqlalchemy",
        ),
    ),
    (
        "packages/conversations",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-langgraph",
            "3tears-observe",
            "langchain-core",
        ),
    ),
    (
        "packages/core",
        (
            "3tears-observe",
            "3tears-nats",
            "sqlalchemy",
            "asyncpg",
            "pydantic",
            "pyyaml",
            "pydantic-core",
        ),
    ),
    (
        "packages/datasources",
        (
            "3tears",
            "3tears-nats",
            "3tears-observe",
            "pydantic",
            "asyncpg",
            "pyyaml",
        ),
    ),
    (
        "packages/enforcement",
        (
            "3tears",
            "3tears-observe",
        ),
    ),
    (
        "packages/epoch",
        (
            "3tears",
            "3tears-nats",
            "3tears-observe",
            "pydantic",
            "uuid-utils",
        ),
    ),
    (
        "packages/geo",
        (
            "3tears",
            "3tears-observe",
            "3tears-media-contracts",
            "shapely",
            "mapbox-vector-tile",
        ),
    ),
    (
        "packages/iam",
        (
            "3tears",
            "3tears-observe",
            "3tears-agent-acl",
            "3tears-nats",
            "argon2-cffi",
            "bcrypt",
            "pyjwt",
            "cryptography",
            "httpx",
            "pyotp",
            "joserfc",
            "pydantic",
        ),
    ),
    (
        "packages/langgraph",
        (
            "3tears",
            "3tears-media-contracts",
            "3tears-observe",
            "langchain",
            "langchain-core",
            "langgraph",
            "langgraph-checkpoint",
            "asyncpg",
            "uuid-utils",
            "pydantic",
        ),
    ),
    (
        "packages/models",
        (
            "3tears-media-contracts",
            "3tears-observe",
            "3tears-tool-schema",
            "pydantic",
            "langchain-core",
            "anthropic",
            "jsonschema",
            "langchain-anthropic",
            "langchain-openai",
            "langchain-openrouter",
            "openrouter",
            "uuid-utils",
        ),
    ),
    (
        "packages/nats",
        (
            "3tears-observe",
            "pydantic",
        ),
    ),
    (
        "packages/object-store",
        (
            "3tears",
            "3tears-media-contracts",
            "3tears-observe",
            "aiobotocore",
            "cryptography",
            "pydantic",
            "botocore",
        ),
    ),
    (
        "packages/registry",
        (
            "3tears",
            "3tears-agent-acl",
            "3tears-agent-tools",
            "3tears-epoch",
            "3tears-nats",
            "3tears-observe",
            "nats-py",
            "prometheus-client",
            "pydantic",
            "sqlalchemy",
        ),
    ),
    (
        "packages/scheduled-jobs",
        (
            "3tears",
            "3tears-nats",
            "3tears-observe",
            "uuid-utils",
            "apscheduler",
            "sqlalchemy",
        ),
    ),
    (
        "packages/scrape",
        (
            "3tears",
            "3tears-agent-tools",
            "3tears-models",
            "3tears-observe",
            "3tears-search",
            "httpx",
            "langchain-core",
            "pydantic",
            "pyyaml",
            "uuid-utils",
            "beautifulsoup4",
            "camoufox",
            "playwright",
            "soupsieve",
        ),
    ),
)

_ALL_FLOORS = _DEPENDENCY_FLOORS + tuple(
    DependencyFloor(package=package, allowed=allowed, rationale=_AS_FOUND_RATIONALE)
    for package, allowed in _RECORDED_AS_FOUND
)

_CONFIG = DependencyAlignmentConfig(
    repo_root=_REPO_ROOT,
    package_globs=("packages/*", "packages/agent/*"),
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_dependency_alignment_exemptions.txt",
    contract_packages=("packages/media-contracts", "packages/tool-schema"),
    dependency_floors=_ALL_FLOORS,
)


class TestDependencyAlignment:
    """declared deps match actual imports for every workspace package."""

    def test_declared_dependencies_match_actual_imports(self) -> None:
        """no undeclared module-top imports; no stale 3tears declarations."""
        run_dependency_alignment_enforcement(_CONFIG, walker="alignment")

    def test_contracts_packages_stay_dependency_free(self) -> None:
        """contracts packages import only stdlib + their own namespace."""
        run_dependency_alignment_enforcement(_CONFIG, walker="contract_purity")

    def test_floor_pinned_packages_declare_exactly_their_ruled_floor(self) -> None:
        """a pinned package's hard deps are the ruled list -- no more, no fewer."""
        run_dependency_alignment_enforcement(_CONFIG, walker="dependency_floor")

    def test_every_floor_pin_names_a_real_package(self) -> None:
        """guard the guard: a pin over a path that holds nothing passes vacuously,
        so the pinned directories are asserted to exist as packages here rather
        than only inside the walker's own report."""
        for floor in _ALL_FLOORS:
            assert (_REPO_ROOT / floor.package / "pyproject.toml").is_file(), floor.package
            assert floor.allowed, f"{floor.package} pins an empty floor"
        pinned = [floor.package for floor in _ALL_FLOORS]
        assert len(pinned) == len(set(pinned)), f"a package is pinned twice: {sorted(pinned)}"

    def test_every_package_with_hard_dependencies_has_a_floor(self) -> None:
        """a package without a floor can take a new hard dependency in one unreviewed pyproject line.

        a package that declares no hard dependencies needs no entry: adding its first one leaves it
        with dependencies and no floor, which this test then refuses.
        """
        pinned = {floor.package for floor in _ALL_FLOORS} | set(_CONFIG.contract_packages)
        packages: list[str] = []
        unpinned: list[str] = []
        for pattern in _CONFIG.package_globs:
            for pyproject in sorted(_REPO_ROOT.glob(f"{pattern}/pyproject.toml")):
                package = pyproject.parent.relative_to(_REPO_ROOT).as_posix()
                packages.append(package)
                declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get("dependencies") or []
                if declared and package not in pinned:
                    unpinned.append(package)
        assert len(packages) > 20, f"only {len(packages)} packages found; the workspace layout changed"
        assert not unpinned, (
            f"packages with hard dependencies and no floor: {unpinned}. Argue a floor in _DEPENDENCY_FLOORS, "
            "or, failing that, record the package in _RECORDED_AS_FOUND."
        )
