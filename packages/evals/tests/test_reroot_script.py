"""The re-root's rewrite script, ``scripts/reroot.py``, run as a host runs it: a subprocess with the standard library
only (``-I -S``: no site-packages, so no ``threetears``), over copies of the trees under ``fixtures/reroot/``.

The trees are stored with ``.txt`` appended to every file name, so the repository's linters, formatters and source
scanners never read them as code; :func:`materialize` strips it. ``map.txt`` is a two-epoch map holding every
record type.

- ``before/`` holds every form the script rewrites, one file per form, plus files it must leave alone (excluded,
  pragma-kept); ``after/`` is what one run makes of it.
- ``half/`` is ``after/`` with every second occurrence put back, a file already holding the ``evals_*`` binding an
  earlier run made, and merge-shaped duplicate imports: one run must make it ``after/``.
- ``hand/`` holds what a person must edit (removed names, unresolvable former roots, a code string, a generated
  file, a taken alias); ``hand_after/`` is the run's output: the rest of each file rewritten, those sites untouched.
- ``edges/before/<item>/`` holds the forms a review found left silently or broken, one directory per item;
  ``edges/after/<item>/`` is what one run makes of it. Links, encodings and failed writes cannot be stored as
  fixture files, so those tests build their trees in ``tmp_path``.

The rule every test here holds: the script never exits 0 or 3 while a reference it recognises but cannot safely
rewrite remains. Each such reference is rewritten or listed (``file:line``, what was found, what to write), and a
later run over a tree that still holds it lists it again.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "reroot.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reroot"
MAP = FIXTURES / "map.txt"
EPOCH_1 = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
EPOCH_2 = "b2c3d4e5f60718293a4b5c6d7e8f901234567890"

EXIT_CLEAN, EXIT_REFUSED, EXIT_HAND_EDITS, EXIT_REWROTE = 0, 1, 2, 3

#: Every site ``hand/`` leaves to a person, as (file, line, kind), on every run.
HAND_SITES = [
    ("host/conflict.py", 9, "unresolved"),
    ("host/dynamic.py", 5, "unresolved"),
    ("host/dynamic.py", 8, "unresolved"),
    ("host/dynamic.py", 12, "unresolved"),
    ("host/dynamic.py", 12, "unresolved"),
    ("host/dynamic.py", 14, "unresolved"),
    ("host/dynamic.py", 15, "unresolved"),
    ("host/removed.py", 7, "removed"),
    ("host/removed.py", 8, "removed"),
    ("host/removed.py", 15, "removed"),
    ("host/removed.py", 16, "removed"),
    ("host/removed.py", 17, "removed"),
    ("host/removed.py", 21, "removed"),
    ("host/removed.py", 22, "removed"),
    ("host/strings.py", 3, "code-string"),
    ("host/strings.py", 4, "ambiguous"),
    ("host/strings.py", 5, "unresolved"),
    ("host/strings.py", 9, "unresolved"),
    ("host/strings.py", 10, "unresolved"),
    ("web/src/ops-types.ts", 1, "generated"),
]

#: The dry run's counts over ``before/``.
BEFORE_COUNTS = {
    "files_scanned": 14,
    "files_changed": 14,
    "files_skipped": 3,
    "statements_rewritten": 25,
    "code_names_rewritten": 11,
    "strings_rewritten": 10,
    "prose_tokens_rewritten": 13,
    "removed_name_uses": 0,
    "unresolved": 0,
    "ambiguous": 0,
    "code_strings": 0,
    "generated_files": 0,
    "renamed": 2,
    "collision": 0,
    "linked": 0,
    "undecodable": 0,
    "kept_lines": 2,
}


def materialize(name: str, into: Path) -> Path:
    """Copy fixture tree ``name`` to ``into``, each file under its real name."""
    for source in sorted((FIXTURES / name).rglob("*.txt")):
        target = into / source.relative_to(FIXTURES / name).with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return into


def snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def run(root: Path, *args: str, map_path: Path = MAP) -> subprocess.CompletedProcess[str]:
    """Run the script as a host does: the standard library only, nothing installed on the path."""
    return subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--map", str(map_path), *args, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def report(root: Path, *args: str, map_path: Path = MAP) -> tuple[int, dict[str, object]]:
    done = run(root, "--json", *args, map_path=map_path)
    assert done.stdout, done.stderr
    return done.returncode, json.loads(done.stdout)


def sites(data: dict[str, object]) -> list[tuple[str, int, str]]:
    return [(s["path"], s["line"], s["kind"]) for s in data["hand_edits"]]  # type: ignore[attr-defined, index]


def write_map(path: Path, rows: list[tuple[str, ...]], *, digest: str | None = None) -> Path:
    body = "".join("\t".join(row) + "\n" for row in rows)
    header = (
        f"# reroot-map 1\n# epoch {EPOCH_1} one\n# epoch {EPOCH_2} two\n"
        f"# body-sha256 {digest or hashlib.sha256(body.encode()).hexdigest()}\n"
    )
    path.write_text(header + body, encoding="utf-8")
    return path


# -- rewriting -----------------------------------------------------------------------------------------------------


def test_one_run_rewrites_every_form_to_after(tmp_path: Path) -> None:
    """Every import form, string form and prose form of ``before/`` comes out as ``after/``, byte for byte, and the
    files the script must leave alone (CHANGELOG, MIGRATIONS.toml, a dot directory, its own fixtures, a pragma line)
    are untouched. The only hand edits are the two renamed names, reported once so their uses are checked."""
    tree = materialize("before", tmp_path / "tree")
    code, data = report(tree)
    assert snapshot(tree) == snapshot(materialize("after", tmp_path / "expected"))
    assert sites(data) == [("host/renamed.py", 3, "renamed"), ("host/renamed.py", 3, "renamed")]
    assert code == EXIT_HAND_EDITS
    assert sorted(data["skipped"]) == [  # type: ignore[call-overload]
        "CHANGELOG.md: excluded",
        "MIGRATIONS.toml: excluded",
        "tests/fixtures/reroot/sample.py: excluded",
    ]
    assert [(k["path"], k["line"]) for k in data["kept"]] == [("host/pragma.py", 3), ("host/pragma.py", 6)]  # type: ignore[attr-defined, index]


def test_a_second_run_over_after_changes_nothing_and_exits_clean(tmp_path: Path) -> None:
    tree = materialize("after", tmp_path / "tree")
    before = snapshot(tree)
    code, data = report(tree)
    assert snapshot(tree) == before
    assert (code, sites(data), data["changed"]) == (EXIT_CLEAN, [], [])


EDGE_FIXTURES = [f"edges/{side}/{item}" for side in ("before", "after") for item in ("t1", "t2", "t5", "t8")]


@pytest.mark.parametrize("fixture", ["before", "half", "hand", "after", "hand_after", *EDGE_FIXTURES])
def test_every_fixture_is_byte_identical_on_a_second_run(fixture: str, tmp_path: Path) -> None:
    """Idempotent: whatever one run makes, a second run leaves byte for byte, and lists the same hand edits."""
    tree = materialize(fixture, tmp_path / "tree")
    first_code, first = report(tree)
    once = snapshot(tree)
    second_code, second = report(tree)
    assert snapshot(tree) == once
    assert second["changed"] == []
    assert [s for s in sites(second)] == [s for s in sites(first) if s[2] != "renamed"]
    assert second_code in (EXIT_CLEAN, EXIT_HAND_EDITS) and first_code != EXIT_REFUSED


def test_a_half_rewritten_tree_converges_in_one_pass(tmp_path: Path) -> None:
    """``half/``: every second occurrence already rewritten, a file already holding ``evals_shapes`` (reused, not
    added again), a module that already imports ``threetears.evals.host`` for its chains, a line epoch 1 already
    wrote that epoch 2 moves on, and duplicate imports a merge left (dropped). One run makes ``after/``."""
    tree = materialize("half", tmp_path / "tree")
    code, data = report(tree)
    assert snapshot(tree) == snapshot(materialize("after", tmp_path / "expected"))
    assert (code, sites(data)) == (EXIT_REWROTE, [])


def test_a_branch_merged_after_the_cut_is_fixed_by_running_again(tmp_path: Path) -> None:
    """A rewritten tree that gains new old-spelling uses (a branch merging after the cut) converges again, and the
    script's own earlier binding is reused rather than a second one appended."""
    tree = materialize("after", tmp_path / "tree")
    routes = tree / "host" / "web" / "routes.py"
    routes.write_text(
        routes.read_text(encoding="utf-8")
        + "\n\nfrom threetears.evals import ops\n\n\ndef listed(host: ops.OpsHost) -> object:\n"
        + "    return ops.runs_list(host)\n",
        encoding="utf-8",
    )
    code, _ = report(tree)
    text = routes.read_text(encoding="utf-8")
    assert code == EXIT_REWROTE
    assert text.count("from threetears.evals import host as evals_host") == 1
    assert "def listed(host: evals_host.OpsHost) -> object:\n    return evals_host.runs_list(host)\n" in text
    assert "import ops" not in text


def test_epochs_apply_in_order_and_a_later_to_finishes_the_job(tmp_path: Path) -> None:
    """``--to <epoch 1>`` applies epoch 1 only; a second run ``--to <epoch 2>`` equals one run with it."""
    tree = materialize("before", tmp_path / "tree")
    first, _ = report(tree, "--to", EPOCH_1[:10])
    later = (tree / "host" / "later.py").read_text(encoding="utf-8")
    assert "from threetears.evals.ops.runs import LaunchSettings" in later  # epoch 2's module line not applied
    assert "from threetears.evals.shapes import PriceRefresh" in later  # epoch 1's name line applied
    assert first == EXIT_HAND_EDITS
    second, _ = report(tree, "--to", EPOCH_2)
    assert snapshot(tree) == snapshot(materialize("after", tmp_path / "expected"))
    assert second == EXIT_REWROTE


def test_a_shadowing_name_binds_the_name_and_its_module_follows_the_module_line(tmp_path: Path) -> None:
    """``run.definition_drift`` is both a public function and a submodule: ``from ... import`` binds the function
    (the name line), ``import ...definition_drift`` and a logger name follow the module line, and nothing appends
    a segment on a re-run."""
    after = materialize("after", tmp_path / "tree")
    shadow = (after / "host" / "shadow.py").read_text(encoding="utf-8")
    strings = (after / "host" / "strings.py").read_text(encoding="utf-8")
    assert (
        "import threetears.evals.run.drift as drift_module\nfrom threetears.evals.host import definition_drift\n"
        in shadow
    )
    assert 'logging.getLogger("threetears.evals.run.drift")' in strings
    assert 'importlib.import_module("threetears.evals.analysis.judge_drift_read")' in strings


# -- what a person must do -----------------------------------------------------------------------------------------


def test_removed_names_and_unresolvable_forms_are_listed_and_never_changed(tmp_path: Path) -> None:
    """Each use of a removed name is listed as ``file:line old -> replacement`` and left as written; the rest of its
    file is rewritten; exit 2. The same list comes back on a re-run, with nothing redone."""
    tree = materialize("hand", tmp_path / "tree")
    code, data = report(tree)
    assert snapshot(tree) == snapshot(materialize("hand_after", tmp_path / "expected"))
    assert (code, sites(data)) == (EXIT_HAND_EDITS, HAND_SITES)
    removed = [s["text"] for s in data["hand_edits"] if s["path"] == "host/removed.py"]  # type: ignore[attr-defined, index]
    assert removed[0] == "threetears.evals.ops.await_job -> host.job_manager.wait_for(ids), or the job_poll action"
    text = (tree / "host" / "removed.py").read_text(encoding="utf-8")
    assert "from threetears.evals.ops import await_job\n" in text and "    ops.await_job(host, ids)\n" in text
    assert "from threetears.evals.testing import start_run\n" in text  # the rest of the statement moved

    again_code, again = report(tree)
    assert (again_code, sites(again), again["changed"]) == (EXIT_HAND_EDITS, HAND_SITES, [])
    printed = run(tree).stdout
    assert "host/removed.py:7: removed: threetears.evals.ops.await_job -> host.job_manager.wait_for(ids)" in printed


def test_strings_paired_roots_resolve_and_the_rest_is_reported_or_left(tmp_path: Path) -> None:
    """A bare former root resolves when the call or tuple pairs it with one of its names; unpaired it is reported;
    a string holding code is reported, never rewritten; a module path the map does not name is untouched."""
    after = (materialize("after", tmp_path / "after") / "host" / "strings.py").read_text(encoding="utf-8")
    assert after.count('seed_module="threetears.evals.shapes"') == 2
    assert 'PROVISIONAL_NAME = ("threetears.evals.shapes", "BUDGET_SHARES")' in after
    assert 'owner_module="threetears.evals.storage.routed"' in after
    assert 'PROFILE_LOGGER = "threetears.evals.kernel.host.profile"' in after
    assert '"threetears.evals.run.judge.MeteredCallLedger"' in after
    assert 'mock.patch("threetears.evals.host.OpsHost.campaign_export_max_bytes", 1)' in after
    assert 'harness = "threetears.evals.host:OpsHost"' in after

    hand_tree = materialize("hand", tmp_path / "hand")
    _, data = report(hand_tree)
    texts = {(s["path"], s["line"]): s["text"] for s in data["hand_edits"]}  # type: ignore[attr-defined, index]
    assert "former root and nothing beside it" in texts[("host/strings.py", 10)]
    assert "string holding code" in texts[("host/strings.py", 3)]
    hand = (hand_tree / "host" / "strings.py").read_text(encoding="utf-8")
    assert hand == (FIXTURES / "hand" / "host" / "strings.py.txt").read_text(encoding="utf-8")


def test_a_generated_file_is_reported_with_its_header_and_left(tmp_path: Path) -> None:
    tree = materialize("hand", tmp_path / "tree")
    _, data = report(tree)
    generated = [s for s in data["hand_edits"] if s["kind"] == "generated"]  # type: ignore[attr-defined, index]
    assert len(generated) == 1 and "GENERATED by web/scripts/gen-ops-types.mjs" in generated[0]["text"]
    assert "web/src/ops-types.ts: generated" in data["skipped"]  # type: ignore[operator]
    assert (tree / "web/src/ops-types.ts").read_bytes() == (FIXTURES / "hand/web/src/ops-types.ts.txt").read_bytes()


def test_an_excluded_path_is_left_alone(tmp_path: Path) -> None:
    tree = materialize("before", tmp_path / "tree")
    held = (tree / "host" / "strings.py").read_bytes()
    report(tree, "--exclude", "host/strings.py")
    assert (tree / "host" / "strings.py").read_bytes() == held
    assert (tree / "host" / "nested.py").read_bytes() != (FIXTURES / "before/host/nested.py.txt").read_bytes()


# -- targets -------------------------------------------------------------------------------------------------------


def test_home_for_left_rewrites_removed_names_to_their_home(tmp_path: Path) -> None:
    """In-repo tests take a removed name from its home module instead of being listed."""
    tree = materialize("hand", tmp_path / "tree")
    _, data = report(tree, "--home-for-left")
    text = (tree / "host" / "removed.py").read_text(encoding="utf-8")
    assert (
        "from threetears.evals.ops.jobs import await_job\nfrom threetears.evals.run.regrading import regrade\n" in text
    )
    assert "evals_ops_jobs.await_job(host, ids)" in text
    assert [s for s in sites(data) if s[2] == "removed"] == []


def test_home_imports_from_the_home_module_and_prose_keeps_the_public_spelling(tmp_path: Path) -> None:
    """Package source: imports go to each name's home module; docstrings name the public spelling."""
    tree = materialize("before", tmp_path / "tree")
    report(tree, "--home")
    nested = (tree / "host" / "nested.py").read_text(encoding="utf-8")
    prose = (tree / "host" / "prose.py").read_text(encoding="utf-8")
    assert "from threetears.evals.schema.models import EvalRun" in nested
    assert "from threetears.evals.run.launch import start_run" in nested
    assert ":class:`threetears.evals.host.OpsHost`" in prose


# -- dry run and refusals ------------------------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_counts_by_kind(tmp_path: Path) -> None:
    tree = materialize("before", tmp_path / "tree")
    held = snapshot(tree)
    code, data = report(tree, "--dry-run")
    assert snapshot(tree) == held
    assert data["counts"] == BEFORE_COUNTS
    assert code == EXIT_HAND_EDITS
    printed = run(tree, "--dry-run").stdout
    assert "dry run: nothing was written" in printed
    for label in ("import statements rewritten", "strings rewritten", "uses of a removed name", "files skipped"):
        assert label in printed


def test_exit_codes_distinguish_nothing_rewrote_and_hand_edits(tmp_path: Path) -> None:
    assert report(materialize("after", tmp_path / "a"))[0] == EXIT_CLEAN
    assert report(materialize("half", tmp_path / "b"))[0] == EXIT_REWROTE
    assert report(materialize("hand", tmp_path / "c"))[0] == EXIT_HAND_EDITS


def _good_rows() -> list[tuple[str, ...]]:
    return [
        (EPOCH_1, "name", "threetears.evals.ops.OpsHost", "threetears.evals.host.OpsHost", ""),
        (EPOCH_1, "root", "threetears.evals.ops", "threetears.evals.host", ""),
    ]


@pytest.mark.parametrize(
    ("rows", "digest", "message"),
    [
        (
            [*_good_rows(), (EPOCH_1, "name", "threetears.evals.host.OpsHost", "threetears.evals.shapes.OpsHost", "")],
            None,
            "equals the old value",
        ),
        (
            [(EPOCH_1, "name", "threetears.evals.run.regrade", "threetears.evals.run.regrade.regrade", "")],
            None,
            "has as a dotted prefix",
        ),
        (
            [(EPOCH_1, "module", "threetears.evals.run.regrade", "threetears.evals.run.regrade.inner", "")],
            None,
            "has as a dotted prefix",
        ),
        (_good_rows(), "0" * 64, "body-sha256 does not match"),
        (
            [(EPOCH_1, "demoted", "threetears.evals.ops.await_job", "threetears.evals.ops.jobs.await_job", "")],
            None,
            "replacement",
        ),
        ([(EPOCH_1, "moved", "threetears.evals.ops.X", "threetears.evals.host.X", "")], None, "unknown record"),
    ],
)
def test_a_bad_map_is_refused_and_nothing_is_written(
    rows: list[tuple[str, ...]], digest: str | None, message: str, tmp_path: Path
) -> None:
    tree = materialize("before", tmp_path / "tree")
    held = snapshot(tree)
    done = run(tree, map_path=write_map(tmp_path / "map.txt", rows, digest=digest))
    assert done.returncode == EXIT_REFUSED
    assert message in done.stderr
    assert snapshot(tree) == held


def test_a_later_epoch_may_move_what_an_earlier_one_wrote(tmp_path: Path) -> None:
    rows = [
        *_good_rows(),
        (EPOCH_2, "name", "threetears.evals.host.OpsHost", "threetears.evals.shapes.OpsHost", ""),
    ]
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.py").write_text("from threetears.evals.ops import OpsHost\n", encoding="utf-8")
    code, _ = report(tree, map_path=write_map(tmp_path / "map.txt", rows))
    assert code == EXIT_REWROTE
    assert (tree / "a.py").read_text(encoding="utf-8") == "from threetears.evals.shapes import OpsHost\n"


def test_an_unknown_epoch_is_refused(tmp_path: Path) -> None:
    done = run(materialize("before", tmp_path / "tree"), "--to", "ffffffff")
    assert done.returncode == EXIT_REFUSED and "names no epoch" in done.stderr


def test_a_file_that_does_not_parse_refuses_the_run_and_nothing_is_written(tmp_path: Path) -> None:
    tree = materialize("before", tmp_path / "tree")
    (tree / "host" / "broken.py").write_text("from threetears.evals.ops import (OpsHost\n", encoding="utf-8")
    held = snapshot(tree)
    done = run(tree)
    assert done.returncode == EXIT_REFUSED
    assert "host/broken.py:1: does not parse before rewriting" in done.stdout
    assert snapshot(tree) == held


def test_a_rewrite_that_would_not_parse_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    rows = [(EPOCH_1, "renamed", "threetears.evals.ROOT_EXTRAS", "threetears.evals.class", "a keyword")]
    tree = materialize("before", tmp_path / "tree")
    held = snapshot(tree)
    done = run(tree, map_path=write_map(tmp_path / "map.txt", rows))
    assert done.returncode == EXIT_REFUSED
    assert "host/renamed.py" in done.stdout and "does not parse after rewriting" in done.stdout
    assert snapshot(tree) == held


# -- how a host runs it --------------------------------------------------------------------------------------------


def test_the_script_imports_the_standard_library_only() -> None:
    """One file, the standard library only, and never ``threetears``: a host runs it before upgrading."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported - {"__future__"} <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
    assert "threetears" not in imported


def test_it_runs_with_nothing_installed_and_says_how_in_its_help(tmp_path: Path) -> None:
    """Under ``-I -S`` the package is not importable, and the script still runs; ``--help`` says how to fetch and
    run it at a pinned commit."""
    blocked = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import threetears.evals"], capture_output=True, check=False
    )
    assert blocked.returncode != 0
    shown = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert shown.returncode == 0
    help_text = " ".join(shown.stdout.split())
    assert "never imports threetears" in help_text and "Python 3.14" in help_text
    assert "git show <pin>:packages/evals/scripts/reroot.py" in help_text
    assert "python3.14 reroot.py --map reroot-map.txt --dry-run ." in help_text
    assert "an epoch SHA from the map's header" in help_text and "not a release pin" in help_text
    assert "--to <pinned SHA>" not in SCRIPT.read_text(encoding="utf-8")


# -- what a review found left silently or broken (one test per item; each fails on the script before the fix) -----


def edge(item: str, tmp_path: Path) -> tuple[Path, int, dict[str, object]]:
    """Run once over ``edges/before/<item>`` and check the tree is ``edges/after/<item>``; then run again and check
    nothing changes and the same sites come back."""
    tree = materialize(f"edges/before/{item}", tmp_path / "tree")
    code, data = report(tree)
    assert snapshot(tree) == snapshot(materialize(f"edges/after/{item}", tmp_path / "expected"))
    again_code, again = report(tree)
    assert snapshot(tree) == snapshot(tmp_path / "expected")
    assert again["changed"] == [] and sites(again) == [s for s in sites(data) if s[2] != "renamed"]
    assert again_code == (EXIT_HAND_EDITS if sites(again) else EXIT_CLEAN)
    return tree, code, data


def test_a_former_root_binding_used_as_an_object_is_listed_on_every_run_and_kept(tmp_path: Path) -> None:
    """T-1 (a, b): ``monkeypatch.setattr(providers, ...)`` (the shape at test_harness_writer_turns.py:280),
    ``mock.patch.object(kernel, ...)``, ``vars(kernel)`` and ``__all__ = ["ops"]`` keep their binding and are listed
    with what moved where; the binding's ``binding.attr`` reads are still rewritten."""
    tree, code, data = edge("t1", tmp_path)
    assert code == EXIT_HAND_EDITS
    listed = {(s["path"], s["line"]): s["text"] for s in data["hand_edits"]}  # type: ignore[attr-defined, index]
    assert "monkeypatch.setattr(providers, ...)" in listed[("object_use.py", 10)]
    assert "PriceRefresh went to threetears.evals.shapes" in listed[("object_use.py", 10)]
    assert "mock.patch.object(kernel, ...)" in listed[("object_use.py", 11)]
    assert "EvalServiceError went to threetears.evals.host" in listed[("object_use.py", 11)]
    assert "vars(kernel" in listed[("object_use.py", 13)]
    assert "__all__ exports ops" in listed[("exports.py", 6)]
    object_use = (tree / "object_use.py").read_text(encoding="utf-8")
    assert "from threetears.evals import kernel, providers\n" in object_use
    assert "assert evals_host.ValidationFailedError" in object_use
    assert "from threetears.evals import ops\n" in (tree / "exports.py").read_text(encoding="utf-8")


def test_string_annotations_through_a_binding_are_rewritten_and_the_binding_goes(tmp_path: Path) -> None:
    """T-1 (b): ``"ops.OpsHost"`` in an annotation or ``cast`` is code the type checker reads; it is rewritten with
    the reads, so removing the former-root binding leaves no string naming an unbound name."""
    tree, _, data = edge("t1", tmp_path)
    text = (tree / "annotations.py").read_text(encoding="utf-8")
    assert "ops." not in text and "import ops" not in text
    assert 'runs: "list[evals_shapes.LaunchArguments] | None" = None' in text
    assert 'cast("evals_host.OpsHost", held or host)' in text
    assert not [s for s in sites(data) if s[0] == "annotations.py"]


def test_a_bare_import_inside_a_function_has_its_chains_rewritten(tmp_path: Path) -> None:
    """T-1 (c): a function's own ``import threetears.evals.kernel`` and the chains read through it, a nested
    function's included."""
    tree, _, _ = edge("t1", tmp_path)
    text = (tree / "function_import.py").read_text(encoding="utf-8")
    assert "    import threetears.evals.host\n\n    return threetears.evals.host.EvalServiceError\n" in text
    assert "        return threetears.evals.host.OpsHost\n" in text
    assert "kernel" not in text and "evals.ops" not in text


def test_reads_through_an_alias_of_the_package_are_listed(tmp_path: Path) -> None:
    """T-1 (d): ``import threetears.evals as te`` then ``te.ops.OpsHost``, and ``from threetears import evals``
    then ``evals.kernel.X``, are listed with the import to write; a read of a new root through the alias is not."""
    tree, _, data = edge("t1", tmp_path)
    listed = [s["text"] for s in data["hand_edits"] if s["path"] == "package_alias.py"]  # type: ignore[attr-defined, index]
    assert len(listed) == 2
    assert any("te.ops.OpsHost" in t and "from threetears.evals.host import OpsHost" in t for t in listed)
    assert any("evals.kernel.EvalServiceError" in t for t in listed)
    assert (tree / "package_alias.py").read_bytes() == (FIXTURES / "edges/before/t1/package_alias.py.txt").read_bytes()


def test_a_merge_duplicate_after_a_use_keeps_the_first_binding(tmp_path: Path) -> None:
    """T-2: the rewritten first import is kept although an identical one comes later; the use between them still
    has its binding."""
    tree, code, _ = edge("t2", tmp_path)
    text = (tree / "merge.py").read_text(encoding="utf-8")
    assert code == EXIT_REWROTE
    assert text.index("from threetears.evals.host import OpsHost as Host\n") < text.index("print(Host)")


def test_a_symlink_is_never_written_through_and_is_listed(tmp_path: Path) -> None:
    """T-3: a link to a file or directory outside the root is never written and is listed on every run; a link to
    a file the run reaches at its own path is covered there and not listed."""
    outside = tmp_path / "outside"
    (outside / "pkg").mkdir(parents=True)
    old = "from threetears.evals.ops import OpsHost\n"
    (outside / "target.py").write_text(old, encoding="utf-8")
    (outside / "pkg" / "module.py").write_text(old, encoding="utf-8")
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "real.py").write_text(old, encoding="utf-8")
    (root / "link.py").symlink_to(Path("..") / "outside" / "target.py")
    (root / "linkdir").symlink_to(Path("..") / "outside" / "pkg")
    (root / "inner.py").symlink_to(Path("sub") / "real.py")
    for _ in range(2):
        code, data = report(root)
        assert (code, [s[:3] for s in sites(data)]) == (
            EXIT_HAND_EDITS,
            [("link.py", 1, "linked"), ("linkdir", 1, "linked")],
        )
        assert (outside / "target.py").read_text(encoding="utf-8") == old
        assert (outside / "pkg" / "module.py").read_text(encoding="utf-8") == old
        assert (root / "link.py").is_symlink() and (root / "inner.py").is_symlink()
    assert (root / "sub" / "real.py").read_text(encoding="utf-8") == "from threetears.evals.host import OpsHost\n"


def test_to_takes_an_epoch_and_a_release_pin_is_refused_with_the_accepted_values(tmp_path: Path) -> None:
    """T-4: the help says to fetch the map at the pin and run without ``--to``; ``--to`` with a commit that is not an
    epoch is refused, and the refusal names the epochs it accepts and what to do instead."""
    tree = materialize("after", tmp_path / "tree")
    done = run(tree, "--to", "0123456789abcdef0123456789abcdef01234567")
    message = " ".join(done.stderr.split())
    assert done.returncode == EXIT_REFUSED
    assert EPOCH_1 in message and EPOCH_2 in message and "run without --to" in message
    assert report(tree, "--to", EPOCH_1[:4])[0] == EXIT_CLEAN


def test_a_comment_naming_an_old_spelling_on_an_import_is_rewritten_not_refused(tmp_path: Path) -> None:
    """T-5: a trailing comment and a comment inside a parenthesised import, each naming an old spelling, are
    rewritten with the statement; the run is not refused."""
    tree, code, _ = edge("t5", tmp_path)
    text = (tree / "comments.py").read_text(encoding="utf-8")
    assert code == EXIT_REWROTE
    assert "from threetears.evals.host import OpsHost  # was threetears.evals.host.OpsHost\n" in text
    assert "# also threetears.evals.host.EvalServiceError upstream\n" in text


def _limit_file_size() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (16384, 16384))


def test_a_failed_write_is_reported_and_leaves_every_file_whole(tmp_path: Path) -> None:
    """T-6: each file is replaced whole (a temporary file beside it, then a rename); when one write fails (here the
    file-size limit, which holds for any user), the report says which files were written and which were not, exit 1,
    no temporary file is left, and running again finishes the job."""
    tree = tmp_path / "tree"
    tree.mkdir()
    old = "from threetears.evals.ops import OpsHost\n"
    (tree / "a.py").write_text(old, encoding="utf-8")
    (tree / "b.py").write_text(old + "# padding\n" * 4000, encoding="utf-8")
    (tree / "c.py").write_text(old, encoding="utf-8")
    os.chmod(tree / "c.py", 0o640)
    held_b = (tree / "b.py").read_bytes()
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--map", str(MAP), "--json", str(tree)],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=_limit_file_size,  # noqa: PLW1509 -- a resource limit in the child only; nothing else runs there
    )
    data = json.loads(done.stdout)
    assert done.returncode == EXIT_REFUSED
    assert data["written"] == ["a.py", "c.py"]
    assert [line.split(":")[0] for line in data["not_written"]] == ["b.py"]  # type: ignore[attr-defined]
    assert (tree / "b.py").read_bytes() == held_b
    assert (tree / "a.py").read_text(encoding="utf-8") == "from threetears.evals.host import OpsHost\n"
    assert os.stat(tree / "c.py").st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tree.iterdir()) == ["a.py", "b.py", "c.py"]
    assert report(tree)[0] == EXIT_REWROTE
    assert (tree / "b.py").read_text(encoding="utf-8").startswith("from threetears.evals.host import OpsHost\n")
    assert report(tree)[0] == EXIT_CLEAN


def test_a_byte_order_mark_and_a_declared_encoding_are_kept_and_undecodable_files_are_listed(tmp_path: Path) -> None:
    """T-7: a UTF-8 file with a byte-order mark is rewritten and keeps the mark and its CRLF endings; a Python file
    with a PEP 263 coding line is read and written in it; a file that names the package but is not text in its
    encoding is listed, on every run, never skipped silently."""
    tree = tmp_path / "tree"
    tree.mkdir()
    bom = b"\xef\xbb\xbf"
    (tree / "bom.py").write_bytes(bom + b"from threetears.evals.ops import OpsHost\r\nprint(OpsHost)\r\n")
    (tree / "bom.md").write_bytes(bom + b"See `threetears.evals.ops.OpsHost`.\n")
    (tree / "latin.py").write_bytes(b"# -*- coding: latin-1 -*-\n# caf\xe9\nfrom threetears.evals.ops import OpsHost\n")
    (tree / "bad.py").write_bytes(b"# caf\xe9\nfrom threetears.evals.ops import OpsHost\n")
    (tree / "notes.md").write_bytes("See threetears.evals.ops.OpsHost.\n".encode("utf-16"))
    held = {name: (tree / name).read_bytes() for name in ("bad.py", "notes.md")}
    for expected in (EXIT_HAND_EDITS, EXIT_HAND_EDITS):
        code, data = report(tree)
        assert (code, sites(data)) == (expected, [("bad.py", 1, "undecodable"), ("notes.md", 1, "undecodable")])
    assert (tree / "bom.py").read_bytes() == bom + b"from threetears.evals.host import OpsHost\r\nprint(OpsHost)\r\n"
    assert (tree / "bom.md").read_bytes() == bom + b"See `threetears.evals.host.OpsHost`.\n"
    assert (tree / "latin.py").read_bytes() == (
        b"# -*- coding: latin-1 -*-\n# caf\xe9\nfrom threetears.evals.host import OpsHost\n"
    )
    assert {name: (tree / name).read_bytes() for name in held} == held


def test_a_renamed_name_that_collides_with_a_binding_is_listed_on_every_run(tmp_path: Path) -> None:
    """T-8: ``ROOT_EXTRAS`` is not rewritten into a file that already binds ``EXTRAS``; the collision is listed with
    what to write, on every run. A file that already imports ``EXTRAS`` from the same root (a merge) is the same
    binding, not a collision."""
    tree, code, data = edge("t8", tmp_path)
    assert code == EXIT_HAND_EDITS
    collisions = [s for s in data["hand_edits"] if s["kind"] == "collision"]  # type: ignore[attr-defined, index]
    assert [(c["path"], c["line"]) for c in collisions] == [("collision.py", 3)]
    assert "already bound at line 5" in collisions[0]["text"]
    assert "from threetears.evals import EXTRAS as ROOT_EXTRAS" in collisions[0]["text"]
    text = (tree / "collision.py").read_text(encoding="utf-8")
    assert "print(ROOT_EXTRAS, EXTRAS, Extra)" in text
    assert (
        (tree / "merged.py")
        .read_text(encoding="utf-8")
        .endswith("from threetears.evals import EXTRAS\n\nprint(EXTRAS, EXTRAS)\n")
    )
