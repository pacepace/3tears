"""A run says which build launched it: the package version and the git commit, stamped at launch (engine item 1).

Contracts held here:

- **where they come from**: the version is the installed distribution's (``importlib.metadata``); the commit is the
  ``vcs_info.commit_id`` its installer recorded in ``direct_url.json`` (PEP 610) for an install from git. Neither is
  read from a shell call or a checkout;
- **None where unknown, never a guess**: a wheel from an index, an editable or local-directory install, an installer
  record that does not parse (logged as a warning) and a build with no distribution metadata each stamp ``None``;
- **the launch stamps both** on every run it starts, as the build reads them;
- **an old run reads**: a run stored before the fields existed reads both as None, dumps with neither key, and its
  context key and case-set fingerprint do not move when a build stamps them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel import build_stamp
from threetears.evals.kernel.build_stamp import package_rev, package_version
from threetears.evals.kernel.identity import case_set_fingerprint, derive_context_identity
from threetears.evals.run import LaunchHost, start_run
from threetears.evals.schema import EvalRun
from threetears.evals.storage import InMemoryDocumentStore

#: A run stored at the core's first v15 minor, before any stamp existed.
_STORED_RUN = Path(__file__).parent / "fixtures" / "core_documents" / "v15" / "eval_run.json"

#: A commit an installer recorded.
_COMMIT = "5772d7ac85529985cfeb7ecd7a8eb65dfe5ccf0a"


class _Installed:
    """An installed distribution as ``importlib.metadata`` hands it back: its version and its metadata files."""

    def __init__(self, version: str, direct_url: str | None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a distribution under the stamp's reader for one test, the stamp's cache cleared around it."""

    def install(version: str | None, direct_url: str | None = None) -> None:
        def distribution(name: str) -> _Installed:
            assert name == "3tears-evals"
            if version is None:
                raise PackageNotFoundError(name)
            return _Installed(version, direct_url)

        monkeypatch.setattr(build_stamp, "distribution", distribution)
        package_version.cache_clear()
        package_rev.cache_clear()

    yield install
    package_version.cache_clear()
    package_rev.cache_clear()


# --- where they come from ----------------------------------------------------------------------------------------


def test_an_install_from_git_stamps_its_version_and_the_commit_its_installer_recorded(installed: Any) -> None:
    installed(
        "0.81.0",
        json.dumps(
            {
                "url": "https://github.com/pacepace/3tears",
                "vcs_info": {"vcs": "git", "commit_id": _COMMIT, "requested_revision": _COMMIT},
                "subdirectory": "packages/evals",
            }
        ),
    )
    assert (package_version(), package_rev()) == ("0.81.0", _COMMIT)


@pytest.mark.parametrize(
    "direct_url",
    [
        None,  # a wheel from an index records no URL
        json.dumps({"url": "file:///src/packages/evals", "dir_info": {"editable": True}}),  # an editable checkout
        json.dumps({"url": "file:///src/evals-0.81.0.tar.gz", "archive_info": {}}),  # a local archive
        json.dumps({"url": "https://github.com/pacepace/3tears", "vcs_info": {"vcs": "git", "commit_id": ""}}),
    ],
)
def test_an_install_that_recorded_no_commit_stamps_none_for_it(installed: Any, direct_url: str | None) -> None:
    installed("0.81.0", direct_url)
    assert (package_version(), package_rev()) == ("0.81.0", None)


def test_an_installer_record_that_does_not_parse_stamps_no_commit_and_says_so(
    installed: Any, caplog: pytest.LogCaptureFixture
) -> None:
    installed("0.81.0", "{not json")
    with caplog.at_level("WARNING", logger="threetears.evals.kernel.build_stamp"):
        assert (package_version(), package_rev()) == ("0.81.0", None)
    assert any("direct_url.json does not parse" in record.getMessage() for record in caplog.records)


def test_a_build_with_no_distribution_metadata_stamps_neither(installed: Any) -> None:
    installed(None)
    assert (package_version(), package_rev()) == (None, None)


# --- the launch stamps both --------------------------------------------------------------------------------------


def _launching() -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    host, _client = toyhost_launch_host(storage=storage)
    return host, storage


async def _launched(host: LaunchHost, storage: EvalStorage) -> EvalRun:
    (run,) = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
    )
    async with asyncio.timeout(10):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    stored = off_loop(storage.load_eval_run, run.id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


async def test_a_launch_stamps_the_version_and_the_commit_its_build_reads(installed: Any) -> None:
    installed("0.81.0", json.dumps({"url": "https://github.com/pacepace/3tears", "vcs_info": {"commit_id": _COMMIT}}))
    host, storage = off_loop(_launching)

    stored = await _launched(host, storage)

    assert (stored.package_version, stored.package_rev) == ("0.81.0", _COMMIT)


async def test_a_launch_by_an_install_that_recorded_no_commit_stores_no_rev_key(installed: Any) -> None:
    installed("0.81.0")
    host, storage = off_loop(_launching)

    stored = await _launched(host, storage)

    assert (stored.package_version, stored.package_rev) == ("0.81.0", None)
    assert "package_rev" not in stored.to_dict()


# --- an old run reads, its digests unmoved -----------------------------------------------------------------------


def test_a_run_stored_before_the_stamp_reads_neither_and_dumps_as_it_was_read() -> None:
    stored = json.loads(_STORED_RUN.read_text())
    run = EvalRun.from_dict(stored)
    assert (run.package_version, run.package_rev) == (None, None)
    dumped = run.to_dict()
    assert "package_version" not in dumped and "package_rev" not in dumped


def test_stamping_a_run_moves_none_of_its_digests() -> None:
    plain = make_eval_run(id="run-stamped")
    stamped = plain.model_copy(update={"package_version": "0.81.0", "package_rev": _COMMIT})
    profile = toyhost_profile()
    assert derive_context_identity(stamped, profile) == derive_context_identity(plain, profile)
    assert case_set_fingerprint(stamped) == case_set_fingerprint(plain)
    assert set(stamped.to_dict()) - set(plain.to_dict()) == {"package_version", "package_rev"}
