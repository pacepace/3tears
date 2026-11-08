"""A campaign exported by the engine, then minted from the files: producer and consumer held together.

``python -m threetears.evals.harness export --host module:factory --scope SCOPE --campaign ID --out DIR`` writes the
three files ``python -m threetears.evals.harness mint`` reads. Over the courier's capture and the toy host's corpus
campaign, exporting and then minting at the held-out export's instant writes the case the held-out export freezes
through the host directly, byte for byte, and the committed one by its snapshot id.

The committed Discodon case (``held_out/discodon-dj-character-01a1b3d4.json.gz``) is not re-minted here: the
repository holds its frozen snapshot, not the stored documents it was frozen from, and its profile factory is
Discodon's code.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE
from packages.evals.tests.fixtures.writer_cases.export_held_out_cases import HOST_CAMPAIGNS, held_out_path, host_case
from packages.evals.tests.fixtures.writer_cases.export_writer_cases import FROZEN_AT, FROZEN_BY, PACKAGE
from packages.evals.tests.hot_path_support import campaign_id_of, courierhost_from
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.harness import WriterCaseFile
from threetears.evals.harness.__main__ import main
from threetears.evals.kernel.host import EvalHost, HostProfile
from threetears.evals.ops import OpsHost, campaign_export
from threetears.evals.quick.cli import EXIT_FAILED, EXIT_OK, EXIT_REFUSED

#: The courier's capture: every document of its planner bake-off, as stored.
CAPTURE = PACKAGE / "tests" / "fixtures" / "hot_path_oracle" / "inputs" / "courierhost.json"

#: The profile factories the mint rebuilds each host's profile from, as each host names its own.
COURIER_PROFILE = "packages.evals.tests.fixtures.courierhost:courier_profile"
TOY_PROFILE = "packages.evals.tests.fixtures.toyhost.profile:toyhost_profile"

#: The three files, in the order the command names them.
FILES = ("documents.json", "profile.json", "manifest.json")


def courier_export_host() -> EvalHost:
    """The courier host over its capture: what a host's export factory returns, its store and its profile."""
    return courierhost_from(json.loads(CAPTURE.read_text()))


def toy_export_host() -> EvalHost:
    """The toy host over its corpus campaign."""
    return ops_fixture(generation=False).host.eval_host


def profile_only() -> HostProfile:
    """A factory returning a profile, which the mint takes and an export cannot read a store from."""
    return courier_export_host().profile


def broken_host() -> EvalHost:
    """A host's export factory that raises, as one whose config is missing does."""
    raise ValueError("the host's config file is missing")


def _export_args(out: Path, *, host: str, scope: str, campaign: str, built_by: str | None = None) -> list[str]:
    args = ["export", "--host", host, "--scope", scope, "--campaign", campaign, "--out", str(out)]
    return [*args, "--built-by", built_by] if built_by is not None else args


def _courier_campaign() -> str:
    return campaign_id_of(json.loads(CAPTURE.read_text()))


def _mint_args(out: Path, case: Path, *, campaign: str, name: str, group: str) -> list[str]:
    return [
        "mint",
        *(arg for file in FILES for arg in (f"--{file.removesuffix('.json')}", str(out / file))),
        "--campaign",
        campaign,
        "--name",
        name,
        "--group",
        group,
        "--stratum",
        "host",
        "--created-by",
        FROZEN_BY,
        "--as-of",
        FROZEN_AT,
        "--out",
        str(case),
    ]


# --- the command line ------------------------------------------------------------------------------------------------


def test_the_command_line_writes_the_three_files_the_mint_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export"
    campaign = _courier_campaign()
    host = f"{__name__}:courier_export_host"
    assert (
        main(_export_args(out, host=host, scope=COURIER_SCOPE, campaign=campaign, built_by=COURIER_PROFILE)) == EXIT_OK
    )
    exported = campaign_export(courier_export_host(), COURIER_SCOPE, campaign, built_by=COURIER_PROFILE)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == exported.files()
    assert json.loads((out / "profile.json").read_text())["built_by"] == COURIER_PROFILE
    counts = ", ".join(
        f"{doc_type} {count}" for doc_type, count in sorted(exported.manifest.doc_type_counts.items()) if count
    )
    assert counts == "eval_campaign 1, eval_result 12, eval_run 2, eval_template 1, eval_trace 12"
    assert exported.manifest.unresolved, "the capture holds no test case, so the runs' cases are unresolved"
    assert capsys.readouterr().out == (
        f"{out}: {len(exported.documents)} documents of campaign {campaign} ({counts}); "
        f"{len(exported.manifest.unresolved)} links unresolved, listed in manifest.json\n"
    )

    # A second export into the same directory is refused, and leaves the first untouched.
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    assert main(_export_args(out, host=host, scope=COURIER_SCOPE, campaign=campaign)) == EXIT_REFUSED
    assert "documents.json, profile.json, manifest.json already in" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_built_by_defaults_to_the_host_factory_named(tmp_path: Path) -> None:
    """The factory that builds the host for the export builds it for the mint too: the mint reads the host's profile."""
    out = tmp_path / "export"
    host = f"{__name__}:courier_export_host"
    assert main(_export_args(out, host=host, scope=COURIER_SCOPE, campaign=_courier_campaign())) == EXIT_OK
    assert json.loads((out / "profile.json").read_text())["built_by"] == host


def test_a_campaign_the_scope_lacks_is_refused_and_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export"
    host = f"{__name__}:courier_export_host"
    assert main(_export_args(out, host=host, scope=COURIER_SCOPE, campaign="no-such-campaign")) == EXIT_REFUSED
    assert "campaign 'no-such-campaign' not found" in capsys.readouterr().err
    assert not out.exists() or list(out.iterdir()) == []


@pytest.mark.parametrize(
    ("given", "said"),
    [
        (["--host", "x:y", "--scope", "s", "--campaign", "c"], "the following arguments are required: --out"),
        (["--host", "builtins", "--scope", "s", "--campaign", "c", "--out"], "module:factory"),
        (["--host", "nowhere_host.evals:build", "--scope", "s", "--campaign", "c", "--out"], "cannot import"),
    ],
)
def test_a_command_line_the_export_cannot_follow_is_refused(
    given: list[str], said: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["export", *given, *([str(tmp_path / "export")] if given[-1] == "--out" else [])]
    assert main(argv) == EXIT_REFUSED
    assert said in capsys.readouterr().err
    assert not (tmp_path / "export").exists()


def test_a_factory_returning_a_profile_alone_is_refused_naming_what_an_export_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export"
    args = _export_args(out, host=f"{__name__}:profile_only", scope=COURIER_SCOPE, campaign=_courier_campaign())
    assert main(args) == EXIT_REFUSED
    assert "returned a HostProfile, not an EvalHost: an export reads the host's store" in capsys.readouterr().err


def test_a_factory_raising_exits_three_with_its_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "export"
    args = _export_args(out, host=f"{__name__}:broken_host", scope=COURIER_SCOPE, campaign=_courier_campaign())
    assert main(args) == EXIT_FAILED
    err = capsys.readouterr().err
    assert "failed on an unanticipated error" in err and "ValueError: the host's config file is missing" in err
    assert not out.exists() or list(out.iterdir()) == []


# --- the round trip --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "host", "scope", "built_by"),
    [
        ("courierhost", "courier_export_host", COURIER_SCOPE, COURIER_PROFILE),
        ("toyhost", "toy_export_host", TOYHOST_SCOPE, TOY_PROFILE),
    ],
)
async def test_an_export_minted_is_the_case_the_host_freezes_directly(
    name: str, host: str, scope: str, built_by: str, tmp_path: Path
) -> None:
    (campaign,) = [entry for entry in HOST_CAMPAIGNS if entry.name == name]
    exported = await asyncio.to_thread(globals()[host])
    (campaign_id,) = [
        document["id"]
        for document in await asyncio.to_thread(exported.storage.stored_documents, "eval_campaign", scope)
    ]
    out, case = tmp_path / "export", tmp_path / f"{name}.json.gz"

    def round_trip() -> tuple[int, int]:
        return (
            main(_export_args(out, host=f"{__name__}:{host}", scope=scope, campaign=campaign_id, built_by=built_by)),
            main(_mint_args(out, case, campaign=campaign_id, name=name, group=campaign.group)),
        )

    assert await asyncio.to_thread(round_trip) == (EXIT_OK, EXIT_OK)
    direct = (await host_case(campaign)).write(tmp_path / f"direct-{name}.json.gz")
    assert case.read_bytes() == direct.read_bytes()
    assert WriterCaseFile.read(case).snapshot.id == WriterCaseFile.read(held_out_path(name)).snapshot.id


def test_the_manifest_the_export_writes_is_the_one_the_mint_reads(tmp_path: Path) -> None:
    """The export's reasoned_on is the mint's own model, and a manifest of another campaign is refused by it."""
    out = tmp_path / "export"
    campaign = _courier_campaign()
    host = f"{__name__}:courier_export_host"
    assert (
        main(_export_args(out, host=host, scope=COURIER_SCOPE, campaign=campaign, built_by=COURIER_PROFILE)) == EXIT_OK
    )
    manifest: dict[str, Any] = json.loads((out / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps({**manifest, "campaign_id": "another"}))
    assert main(_mint_args(out, tmp_path / "case.json.gz", campaign=campaign, name="c", group="g")) == EXIT_REFUSED


def test_a_write_failing_part_way_leaves_no_directory_and_the_same_command_runs_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export"
    args = _export_args(out, host=f"{__name__}:courier_export_host", scope=COURIER_SCOPE, campaign=_courier_campaign())
    written: list[str] = []
    write_bytes = Path.write_bytes

    def second_write_fails(path: Path, data: bytes) -> int:
        if written:
            raise OSError(28, "No space left on device")
        written.append(path.name)
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", second_write_fails)
    assert main(args) == EXIT_REFUSED
    assert written == ["documents.json"], "the first file was written before the second failed"
    assert "No space left on device" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [], "neither the export nor its partial directory is left behind"

    monkeypatch.undo()
    assert main(args) == EXIT_OK
    assert sorted(path.name for path in out.iterdir()) == sorted(FILES)
    assert [path.name for path in tmp_path.iterdir()] == ["export"]


def test_an_empty_out_directory_is_filled_and_one_holding_other_files_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host, campaign = f"{__name__}:courier_export_host", _courier_campaign()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(_export_args(empty, host=host, scope=COURIER_SCOPE, campaign=campaign)) == EXIT_OK
    assert sorted(path.name for path in empty.iterdir()) == sorted(FILES)

    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("the host's own")
    assert main(_export_args(busy, host=host, scope=COURIER_SCOPE, campaign=campaign)) == EXIT_REFUSED
    assert f"--out {busy} already holds other files" in capsys.readouterr().err
    assert [path.name for path in busy.iterdir()] == ["notes.txt"]


async def test_the_command_line_exports_a_closure_past_the_actions_ceiling(tmp_path: Path) -> None:
    """The action refuses what its host's ceiling does not hold; the command line, reading once in its own process,
    exports the same campaign whole."""
    host = await asyncio.to_thread(courier_export_host)
    ceilinged = dataclasses.replace(OpsHost.reading(host), campaign_export_max_bytes=10_000)
    campaign = _courier_campaign()
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))
    refused = await evals.call(
        {"action": "campaign_export", "campaign_id": campaign},
        host=ceilinged,
        caller=Caller(scope_id=COURIER_SCOPE, identity="agent:test"),
    )
    assert refused.is_error and "export ceiling of 10,000 bytes" in refused.text

    out = tmp_path / "export"
    args = _export_args(out, host=f"{__name__}:courier_export_host", scope=COURIER_SCOPE, campaign=campaign)
    assert await asyncio.to_thread(main, args) == EXIT_OK
    assert len((out / "documents.json").read_bytes()) > ceilinged.campaign_export_max_bytes
