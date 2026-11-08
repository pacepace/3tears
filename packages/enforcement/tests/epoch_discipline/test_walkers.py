"""The epoch-discipline gates, driven by synthesised modules and ledgers.

Each rule has a positive case (a module that violates it is a finding naming file, line and fix)
and a negative case (the sanctioned shape of the same code is not). The ratchet cases are the ways
a ledger could stop shrinking honestly: a site added beside a listed one, a line left behind after
its site was fixed, a waiver without a reason, a primitive that names nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.epoch_discipline import (
    RULE_EPOCH_BYPASS,
    RULE_L2_SWALLOW,
    RULE_TTL_SAFETY_NET,
    RULE_WHOLE_DROP,
    RULE_WHOLE_REREAD,
    RULES,
    EpochDisciplineConfig,
    allowlist_path_for,
    epoch_discipline_findings,
    epoch_violation_counts,
)

_MODULE = "src/pkg/mod.py"


def _config(
    tmp_path: Path,
    source: str,
    *,
    allowlists: dict[str, str] | None = None,
    waivers: str = "",
    primitives: dict[str, frozenset[str]] | None = None,
) -> EpochDisciplineConfig:
    """write one synthesised module and its ledgers under ``tmp_path``, and configure the gate over them.

    :param tmp_path: pytest temporary directory, used as the repo root
    :ptype tmp_path: Path
    :param source: the module's source
    :ptype source: str
    :param allowlists: rule onto allowlist body
    :ptype allowlists: dict[str, str] | None
    :param waivers: waiver file body
    :ptype waivers: str
    :param primitives: rule onto primitive entries
    :ptype primitives: dict[str, frozenset[str]] | None
    :return: the config
    :rtype: EpochDisciplineConfig
    """
    module = tmp_path / _MODULE
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(source, encoding="utf-8")
    ledgers = tmp_path / "tests"
    ledgers.mkdir(exist_ok=True)
    for rule, body in (allowlists or {}).items():
        allowlist_path_for(ledgers, rule).write_text(body, encoding="utf-8")
    (ledgers / "_epoch_waivers.txt").write_text(waivers, encoding="utf-8")
    return EpochDisciplineConfig(
        repo_root=tmp_path,
        src_roots=(tmp_path / "src",),
        ledger_dir=ledgers,
        primitives=primitives or {},
    )


def _rule_findings(tmp_path: Path, rule: str, source: str) -> list[str]:
    """the findings of one rule over one module with empty ledgers.

    :param tmp_path: pytest temporary directory
    :ptype tmp_path: Path
    :param rule: the rule
    :ptype rule: str
    :param source: the module's source
    :ptype source: str
    :return: the findings
    :rtype: list[str]
    """
    return epoch_discipline_findings(_config(tmp_path, source), rule)


# ---- positive and negative fixtures, one pair or more per rule ----------------------------------

_VIOLATIONS: dict[str, list[str]] = {
    RULE_WHOLE_DROP: [
        "def gap(registry):\n    registry.drop_table('memories', reason='gap')\n",
        "class C:\n    def reset(self):\n        self._scan_cache.clear()\n",
        "async def wipe(conn):\n    await conn.execute('TRUNCATE memories')\n",
        "async def wipe(conn):\n    await conn.execute('DELETE FROM memories')\n",
        "def on_group(cache):\n    cache.invalidate_all()\n",
        "def first_sight(registrations):\n    for each in registrations:\n        each.on_table_dropped()\n",
    ],
    RULE_L2_SWALLOW: [
        "async def get(kv, log):\n    try:\n        return await kv.get('k')\n"
        "    except KvError as exc:\n        log.warning('l2 down', exc=exc)\n        return None\n",
        "async def get(bucket):\n    try:\n        await bucket.put('k', b'v')\n    except NatsClientError:\n        pass\n",
    ],
    RULE_TTL_SAFETY_NET: [
        "async def open_(client):\n    return await client.kv_bucket(name='rows', ttl=600)\n",
        "class MemoriesCollection:\n    l1_ttl = 300\n",
        "async def put(kv, key, value):\n    await kv.put(key=key, value=value, ttl=60)\n",
        "async def put(kv, key, value):\n    timed = {'ttl': 60}\n    await kv.update(key=key, value=value, **timed)\n",
    ],
    RULE_EPOCH_BYPASS: [
        "async def bump(pool, key):\n    await pool.execute('UPDATE agent_skills SET uses = uses + 1 WHERE id = $1', key)\n",
        "_SQL = 'INSERT INTO memories (id) VALUES ($1)'\n\nasync def add(conn, key):\n    await conn.execute(_SQL, key)\n",
        "class Coll:\n    async def remember(self, key, data):\n        await self._save_to_l2(key, data)\n",
        "async def load(backend, files):\n    await backend.load_files('rows', files)\n",
    ],
    RULE_WHOLE_REREAD: [
        "async def all_roles(pool):\n    return await pool.fetch('SELECT id, name FROM roles')\n",
        "class Snap:\n    async def _rebuild_scopes(self, scope):\n        return await self._read_scope(scope)\n",
        "async def warm_cache(pool, table):\n    return await pool.fetch(f'SELECT * FROM {table} WHERE scope = $1', 's')\n",
        "class Coll:\n    async def refill_l2(self):\n        async for page in self.l3_pages(500):\n            yield page\n",
    ],
}

_COMPLIANT: dict[str, list[str]] = {
    RULE_WHOLE_DROP: [
        "def fine(cache, key):\n    cache.pop(key)\n",
        "def local(names):\n    names.clear()\n",
        "async def wipe(conn, announce_change):\n    await conn.execute('TRUNCATE memories')\n"
        "    await announce_change('memories', whole_table=True)\n",
        "async def one(conn, key):\n    await conn.execute('DELETE FROM memories WHERE id = $1', key)\n"
        "    await announce_change('memories', [key])\n",
    ],
    RULE_L2_SWALLOW: [
        "async def get(kv):\n    try:\n        return await kv.get('k')\n"
        "    except KvError as exc:\n        degrade_l2(exc, table='t', op='get', what='reading L3')\n        return None\n",
        "async def get(kv):\n    try:\n        return await kv.get('k')\n    except KvError as exc:\n        raise Refused() from exc\n",
        "async def get(kv):\n    try:\n        return await kv.get('k')\n    except KvKeyNotFoundError:\n        return None\n",
        "async def call(client):\n    try:\n        return await client.request('s', b'')\n    except NatsClientError:\n        return None\n",
    ],
    RULE_TTL_SAFETY_NET: [
        "async def open_(client):\n    return await client.kv_bucket(name='rows')\n",
        "async def open_(client):\n    return await client.kv_bucket(name='rows', ttl=None)\n",
        "def mint(minter):\n    return minter.issue(ttl_seconds=60)\n",
    ],
    RULE_EPOCH_BYPASS: [
        "class Coll:\n    async def save_to_store(self, data):\n        await self.pool.execute('INSERT INTO t (id) VALUES ($1)', data['id'])\n",
        "async def touch(coll, key):\n    async with coll.bypassing_write() as write:\n"
        "        await coll.pool.execute('UPDATE t SET seen = now() WHERE id = $1', key)\n        write.touches(key)\n",
        "async def load(conn, rows):\n    await conn.execute('INSERT INTO t SELECT * FROM unnest($1)', rows)\n"
        "    await announce_change('t', rows)\n",
        "async def read(pool, key):\n    return await pool.fetchrow('SELECT * FROM t WHERE id = $1 FOR UPDATE', key)\n",
    ],
    RULE_WHOLE_REREAD: [
        "async def one(pool, key):\n    return await pool.fetchrow('SELECT * FROM roles WHERE id = $1', key)\n",
        "async def page(pool):\n    return await pool.fetch('SELECT * FROM roles ORDER BY id LIMIT 100')\n",
        "async def count(pool):\n    return await pool.fetchval('SELECT 1')\n",
        "async def dynamic(pool, where):\n    return await pool.fetch(f'SELECT * FROM roles {where}')\n",
    ],
}


@pytest.mark.parametrize(
    ("rule", "source"), [(rule, source) for rule, sources in _VIOLATIONS.items() for source in sources]
)
def test_a_violation_is_a_finding_naming_its_line_and_fix(tmp_path: Path, rule: str, source: str) -> None:
    """each violating shape fails its rule, naming the module, the line and what to do instead."""
    findings = _rule_findings(tmp_path, rule, source)
    assert findings, f"{rule} missed:\n{source}"
    assert all(f"{_MODULE}:" in finding and f"NEW {rule} violation" in finding for finding in findings)


@pytest.mark.parametrize(
    ("rule", "source"), [(rule, source) for rule, sources in _COMPLIANT.items() for source in sources]
)
def test_the_sanctioned_shape_is_not_a_finding(tmp_path: Path, rule: str, source: str) -> None:
    """the shape each rule asks for passes it."""
    assert _rule_findings(tmp_path, rule, source) == []


def test_every_rule_has_both_fixtures() -> None:
    """no rule ships without a case that fails it and a case that passes it."""
    assert set(_VIOLATIONS) == set(RULES) == set(_COMPLIANT)


# ---- the ratchet -------------------------------------------------------------------------------

_TWO_DROPS = "def gap(registry):\n    registry.drop_table('a')\n    registry.drop_table('b')\n"
_LINE = f"{_MODULE} :: gap :: drop_table"
_FIX = "apply the epoch's change set instead of dropping"


def test_a_listed_site_passes_at_its_pinned_count(tmp_path: Path) -> None:
    """an allowlist line pinning exactly the site's hits holds the gate green."""
    config = _config(tmp_path, _TWO_DROPS, allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 2 :: {_FIX}\n"})
    assert epoch_discipline_findings(config, RULE_WHOLE_DROP) == []


def test_a_new_site_beside_a_listed_one_fails(tmp_path: Path) -> None:
    """a hit beyond the pinned count is new, even in a listed function."""
    config = _config(tmp_path, _TWO_DROPS, allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 1 :: {_FIX}\n"})
    findings = epoch_discipline_findings(config, RULE_WHOLE_DROP)
    assert findings and all("NEW whole-drop violation (2 here, 1 allowed)" in finding for finding in findings)


def test_a_line_whose_site_shrank_fails_until_lowered(tmp_path: Path) -> None:
    """a fixed site leaves its count too high, and the gate says to lower it."""
    config = _config(tmp_path, _TWO_DROPS, allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 3 :: {_FIX}\n"})
    assert any("lower the count to 2" in finding for finding in epoch_discipline_findings(config, RULE_WHOLE_DROP))


def test_a_line_whose_site_is_gone_fails_until_deleted(tmp_path: Path) -> None:
    """a line naming a site the code no longer has must go."""
    config = _config(tmp_path, "x = 1\n", allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 1 :: {_FIX}\n"})
    assert any("delete the line" in finding for finding in epoch_discipline_findings(config, RULE_WHOLE_DROP))


def test_a_line_without_a_fix_is_refused(tmp_path: Path) -> None:
    """every allowlist line says which fix removes it."""
    config = _config(tmp_path, _TWO_DROPS, allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 2 :: later\n"})
    assert any("the fix must say" in finding for finding in epoch_discipline_findings(config, RULE_WHOLE_DROP))


def test_a_waiver_with_its_reason_covers_the_site(tmp_path: Path) -> None:
    """a waived site passes; the waiver is pinned like an allowlist line."""
    source = "async def hold(kv):\n    await kv.create(key='lease', value=b'', ttl=30)\n"
    reason = "a lease: the expiry is the lease lapsing when its holder dies"
    waiver = f"{RULE_TTL_SAFETY_NET} :: {_MODULE} :: hold :: create(ttl=) :: 1 :: {reason}\n"
    assert epoch_discipline_findings(_config(tmp_path, source, waivers=waiver), RULE_TTL_SAFETY_NET) == []


def test_a_waiver_without_a_reason_is_refused(tmp_path: Path) -> None:
    """no blanket waivers: the reason must be specific."""
    source = "async def hold(kv):\n    await kv.create(key='lease', value=b'', ttl=30)\n"
    waiver = f"{RULE_TTL_SAFETY_NET} :: {_MODULE} :: hold :: create(ttl=) :: 1 :: fine\n"
    findings = epoch_discipline_findings(_config(tmp_path, source, waivers=waiver), RULE_TTL_SAFETY_NET)
    assert any("the reason must say" in finding for finding in findings)


def test_a_primitive_covers_its_body_and_a_stale_one_fails(tmp_path: Path) -> None:
    """a named primitive's own body is the operation; a primitive naming nothing is a finding."""
    source = "class Registry:\n    def drop_table(self, name):\n        self._l1.drop_cached_table(name)\n"
    live = _config(tmp_path / "a", source, primitives={RULE_WHOLE_DROP: frozenset({f"{_MODULE}::Registry"})})
    assert epoch_discipline_findings(live, RULE_WHOLE_DROP) == []
    stale = _config(tmp_path / "b", source, primitives={RULE_WHOLE_DROP: frozenset({f"{_MODULE}::Gone"})})
    assert any("names no function or class" in finding for finding in epoch_discipline_findings(stale, RULE_WHOLE_DROP))


def test_a_walk_under_its_floor_fails(tmp_path: Path) -> None:
    """a walk that reached too few modules checked nothing, and says so."""
    config = _config(tmp_path, "x = 1\n")
    floored = EpochDisciplineConfig(
        repo_root=config.repo_root, src_roots=config.src_roots, ledger_dir=config.ledger_dir, minimum_files_scanned=5
    )
    assert any("under the floor" in finding for finding in epoch_discipline_findings(floored, RULE_WHOLE_DROP))


def test_counts_measure_an_unadopted_repo_and_ignore_its_ledgers(tmp_path: Path) -> None:
    """a consumer measures every rule before it adopts the gates; a ledger does not hide a hit from the count."""
    config = _config(tmp_path, _TWO_DROPS, allowlists={RULE_WHOLE_DROP: f"{_LINE} :: 2 :: {_FIX}\n"})
    assert epoch_violation_counts(config) == {rule: (2 if rule == RULE_WHOLE_DROP else 0) for rule in RULES}
