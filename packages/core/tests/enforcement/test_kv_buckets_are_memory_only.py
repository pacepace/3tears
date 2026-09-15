"""every NATS KV bucket is memory-backed; durability belongs to L3.

**NATS is L2.** A KV bucket asking for ``storage="file"`` is the cache tier
quietly taking the source-of-truth role -- at ``num_replicas=1``, with no
backups, no migrations and no schema. It reads as prudence and it is a
promotion nobody reviewed.

The rule is one line: **memory, always.** Anything that genuinely cannot be
lost belongs in a ``BaseCollection``, which composes L1, L2 (this same NATS,
memory-backed) and L3 rather than making anyone choose between them.

**Why this is a gate and not a convention.** ``storage`` is an ordinary ``str``
with a ``"memory"`` default, so a literal ``"file"`` type-checks perfectly and
nothing but a reader notices. And a bucket's storage is chosen at CREATE and
never reconciled, so the mistake outlives every later deploy that would have
corrected it -- a wrong value is not a bug you fix by shipping a fix, it is one
you fix by deleting live state. That asymmetry is what makes "we'll catch it in
review" insufficient.

It also has history. File storage was reached for twice in one week as the
answer to an intermittent login failure, once in this repo and once in a
consumer, and neither time was it the cause. The reflex is real and the gate is
aimed at the reflex.

**Precedent, in this repo.** ``threetears.epoch`` refuses file storage for its
own state on the grounds that it would be a FALSE guarantee -- file-backed
JetStream is durable only if the store directory survives, and the failure it
defends against wipes JetStream wholesale -- and keeps a Postgres row instead.
This gate generalises that argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import parse_exemptions_with_rationale
from threetears.enforcement.kv_memory_only import file_backed_kv_calls

_REPO = Path(__file__).resolve().parents[4]
_EXEMPTIONS = Path(__file__).parent / "_kv_memory_only_exemptions.txt"


def _exempt_keys() -> set[str]:
    """return the ``path:line`` keys this repo has exempted.

    :return: exempted call sites
    :rtype: set[str]
    """
    return {f"{item.file}:{item.line}" for item in parse_exemptions_with_rationale(_EXEMPTIONS)}


def test_no_kv_bucket_asks_for_file_storage() -> None:
    """the gate, over every package's source."""
    exempt = _exempt_keys()
    violations: list[str] = []
    for path in sorted(_REPO.glob("packages/*/src/**/*.py")):
        for lineno, callee in file_backed_kv_calls(path):
            rel = path.relative_to(_REPO).as_posix()
            if f"{rel}:{lineno}" in exempt:
                continue
            violations.append(f"{rel}:{lineno} opens a KV bucket with storage='file' via {callee}()")
    assert not violations, (
        "NATS is L2 and these ask it to be L3:\n  "
        + "\n  ".join(violations)
        + "\n\nPut the state in a BaseCollection, which gets L2 AND L3, or add a "
        "`# rationale:` exemption naming the mechanism and the plan to remove it."
    )


def test_every_exemption_still_points_at_a_file_backed_call() -> None:
    """a stale exemption pre-approves whatever is written at that line next.

    These entries are temporary by construction -- each names work that removes
    it -- so an entry whose call site has already moved to memory is not a
    harmless leftover. It is a standing permission at a line number that now
    means something else.
    """
    stale: list[str] = []
    for item in parse_exemptions_with_rationale(_EXEMPTIONS):
        path = _REPO / item.file
        if not path.exists():
            stale.append(f"{item.file}:{item.line} -- file is gone")
            continue
        if item.line not in {lineno for lineno, _ in file_backed_kv_calls(path)}:
            stale.append(f"{item.file}:{item.line} -- no file-backed KV call there any more")
    assert not stale, "exemptions that no longer describe anything:\n  " + "\n  ".join(stale)


class TestThisGateCanFail:
    """run the reader against sources built to break it.

    A gate asserted only against a clean tree proves nothing about what it would
    catch, and this one guards a reflex rather than a mistake -- it has to be
    known-good against the exact shape somebody reaches for.
    """

    def test_a_file_backed_open_is_caught(self, tmp_path: Path) -> None:
        source = tmp_path / "opener.py"
        source.write_text('async def go(nc):\n    return await nc.kv_bucket(name="x", storage="file")\n')
        assert file_backed_kv_calls(source) == [(2, "kv_bucket")]

    def test_memory_and_the_default_both_pass(self, tmp_path: Path) -> None:
        """the default is memory, so an absent keyword is always compliant."""
        source = tmp_path / "fine.py"
        source.write_text(
            'async def go(nc):\n    await nc.kv_bucket(name="a", storage="memory")\n    await nc.kv_bucket(name="b")\n'
        )
        assert file_backed_kv_calls(source) == []

    @pytest.mark.parametrize("callee", ["kv_bucket", "open_kv_stream", "build_kv_stream_config"])
    def test_every_opening_call_shape_is_watched(self, tmp_path: Path, callee: str) -> None:
        """the bucket can be created through any of three names in this estate."""
        source = tmp_path / "shapes.py"
        source.write_text(f'def go(js):\n    return {callee}(js, storage="file")\n')
        assert file_backed_kv_calls(source) == [(2, callee)]

    def test_an_unrelated_storage_keyword_is_not_caught(self, tmp_path: Path) -> None:
        """`storage="file"` on something that is not a KV open means nothing here."""
        source = tmp_path / "unrelated.py"
        source.write_text('def go(x):\n    return configure_blobs(x, storage="file")\n')
        assert file_backed_kv_calls(source) == []

    def test_a_computed_storage_value_is_left_alone(self, tmp_path: Path) -> None:
        """deliberate: the walker cannot evaluate it, and a guess either fails a
        legitimate caller or reports a location nobody can act on."""
        source = tmp_path / "computed.py"
        source.write_text('async def go(nc, cfg):\n    return await nc.kv_bucket(name="x", storage=cfg.storage)\n')
        assert file_backed_kv_calls(source) == []
