"""the memory-only KV walker, run against sources built to break it.

A gate asserted only against a clean tree proves nothing about what it would catch, and this one
guards a reflex rather than a mistake -- it has to be known-good against the exact shape somebody
reaches for when they want durability in a hurry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.memory_only_kv import (
    MemoryOnlyKvConfig,
    file_backed_kv_calls,
    run_memory_only_kv_enforcement,
    scan_for_file_backed_kv,
    source_files,
)


class TestTheWalkerCatchesWhatItMust:
    def test_a_file_backed_open_is_caught(self, tmp_path: Path) -> None:
        source = tmp_path / "opener.py"
        source.write_text('async def go(nc):\n    return await nc.kv_bucket(name="x", storage="file")\n')
        assert file_backed_kv_calls(source) == [(2, "kv_bucket")]

    @pytest.mark.parametrize("callee", ["kv_bucket", "open_kv_stream", "build_kv_stream_config"])
    def test_every_opening_call_shape_is_watched(self, tmp_path: Path, callee: str) -> None:
        """the bucket can be created through any of three names in this estate."""
        source = tmp_path / "shapes.py"
        source.write_text(f'def go(js):\n    return {callee}(js, storage="file")\n')
        assert file_backed_kv_calls(source) == [(2, callee)]

    def test_several_opens_in_one_module_are_all_reported(self, tmp_path: Path) -> None:
        source = tmp_path / "many.py"
        source.write_text(
            "async def go(nc):\n"
            '    await nc.kv_bucket(name="a", storage="file")\n'
            '    await nc.kv_bucket(name="b", storage="memory")\n'
            '    await nc.kv_bucket(name="c", storage="file")\n'
        )
        assert file_backed_kv_calls(source) == [(2, "kv_bucket"), (4, "kv_bucket")]


class TestTheWalkerLeavesTheCompliantAlone:
    def test_memory_and_the_default_both_pass(self, tmp_path: Path) -> None:
        """the default is memory, so an absent keyword is always compliant."""
        source = tmp_path / "fine.py"
        source.write_text(
            'async def go(nc):\n    await nc.kv_bucket(name="a", storage="memory")\n    await nc.kv_bucket(name="b")\n'
        )
        assert file_backed_kv_calls(source) == []

    def test_an_unrelated_storage_keyword_is_not_caught(self, tmp_path: Path) -> None:
        """``storage="file"`` on something that is not a KV open means nothing here."""
        source = tmp_path / "unrelated.py"
        source.write_text('def go(x):\n    return configure_blobs(x, storage="file")\n')
        assert file_backed_kv_calls(source) == []

    def test_a_computed_storage_value_is_left_alone(self, tmp_path: Path) -> None:
        """deliberate: the walker cannot evaluate it, and a guess either fails a legitimate
        caller or reports a location nobody can act on."""
        source = tmp_path / "computed.py"
        source.write_text('async def go(nc, cfg):\n    return await nc.kv_bucket(name="x", storage=cfg.storage)\n')
        assert file_backed_kv_calls(source) == []

    def test_an_unparseable_module_is_skipped_rather_than_crashing_the_gate(self, tmp_path: Path) -> None:
        source = tmp_path / "broken.py"
        source.write_text("def go(:\n")
        assert file_backed_kv_calls(source) == []


class TestTheScanReachesEveryLayout:
    def test_a_nested_package_family_is_scanned(self, tmp_path: Path) -> None:
        """the gap that read green for months.

        The agent packages live at ``packages/agent/tools/src``, which a single ``packages/*/src``
        glob does not match -- so every one of them went unscanned while the gate passed.
        """
        nested = tmp_path / "packages" / "agent" / "tools" / "src" / "pkg"
        nested.mkdir(parents=True)
        (nested / "mod.py").write_text('async def go(nc):\n    await nc.kv_bucket(name="x", storage="file")\n')
        config = MemoryOnlyKvConfig(repo_root=tmp_path)

        assert (nested / "mod.py") in source_files(config)
        with pytest.raises(pytest.fail.Exception, match="packages/agent/tools/src/pkg/mod.py:2"):
            run_memory_only_kv_enforcement(config)

    def test_a_flat_package_and_a_plain_src_layout_are_both_scanned(self, tmp_path: Path) -> None:
        flat = tmp_path / "packages" / "nats" / "src" / "pkg"
        flat.mkdir(parents=True)
        (flat / "mod.py").write_text('async def go(nc):\n    await nc.kv_bucket(name="x", storage="file")\n')
        plain = tmp_path / "src" / "app"
        plain.mkdir(parents=True)
        (plain / "mod.py").write_text('async def go(nc):\n    await nc.kv_bucket(name="y", storage="file")\n')

        found = scan_for_file_backed_kv(source_files(MemoryOnlyKvConfig(repo_root=tmp_path)))
        assert {call.path.name for call in found} == {"mod.py"}
        assert len(found) == 2, "one of the two layouts was not scanned"

    def test_a_clean_tree_passes(self, tmp_path: Path) -> None:
        clean = tmp_path / "src" / "app"
        clean.mkdir(parents=True)
        (clean / "mod.py").write_text('async def go(nc):\n    await nc.kv_bucket(name="x")\n')
        run_memory_only_kv_enforcement(MemoryOnlyKvConfig(repo_root=tmp_path))


class TestTheGateRefusesToPassByScanningNothing:
    def test_globs_that_match_no_file_fail(self, tmp_path: Path) -> None:
        """a gate that scanned nothing passes, which is how a misconfigured consumer reads green."""
        with pytest.raises(pytest.fail.Exception, match="no source files matched"):
            run_memory_only_kv_enforcement(MemoryOnlyKvConfig(repo_root=tmp_path))


class TestThereIsNoExemptionMechanism:
    def test_the_config_carries_no_exemptions_path(self) -> None:
        """the escape hatch went with the work it was holding open.

        Each of the four file-backed buckets this rule removed carried a specific, honest
        rationale naming the work that would remove it, and that is how they stayed for months.
        A future reader adding one back has to add the mechanism first, in the open.
        """
        assert not any("exempt" in name for name in MemoryOnlyKvConfig.__dataclass_fields__)
