"""the nats-py private surface ``threetears.nats`` depends on is present in the installed nats-py.

Nothing else fails when it is not: every use of it is behind a call that a health probe, a
subscribe or a pull-consumer stop catches and degrades. A nats-py release that renames one of
these attributes must fail here, by name, before anything ships against it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import importlib.metadata
import shutil
import string
import subprocess
import sys
import tomllib
import types
from datetime import timedelta
from pathlib import Path

import pytest
from nats.aio.client import Client

from threetears.nats import _asyncio_send_path
from threetears.nats._asyncio_send_path import send_path_gaps
from threetears.nats._nats_py_internals import (
    NUID_LENGTH,
    PRIVATE_SURFACE,
    CsprngNuid,
    PrivateAttribute,
    bound_transport,
    missing_private_attributes,
    send_bound_gaps,
    take_queued_messages,
)

from .wire_server import wire_server


def test_the_installed_nats_py_has_every_private_attribute_the_wrapper_uses() -> None:
    """each declared attribute exists, with the declared kind, on a real nats-py object."""
    missing = missing_private_attributes()
    assert not missing, (
        f"nats-py {importlib.metadata.version('nats-py')} no longer provides what "
        f"threetears.nats._nats_py_internals depends on: {missing}. Read the module docstring "
        "before moving the nats-py cap."
    )


def test_every_declared_nats_py_range_admits_only_the_minor_the_suites_run() -> None:
    """Flips if a pyproject admits a nats-py the suites never run: a floor below the installed minor let a
    consumer resolve 2.15.0, which lacks a private attribute the wrapper reads (every large reply failed there),
    while this repo's lock ran 2.16 and hid it. Each declaration must be ``>=<installed minor>,<<next minor>``."""
    major, minor = (int(part) for part in importlib.metadata.version("nats-py").split(".")[:2])
    wanted = f">={major}.{minor},<{major}.{minor + 1}"
    root = Path(__file__).resolve().parents[4]
    found: dict[str, str] = {}
    for path in sorted([root / "pyproject.toml", *root.glob("packages/**/pyproject.toml")]):
        if ".venv" in path.parts:
            continue
        project = tomllib.loads(path.read_text(encoding="utf-8"))
        declared = [
            *project.get("project", {}).get("dependencies", []),
            *(d for extra in project.get("project", {}).get("optional-dependencies", {}).values() for d in extra),
            *(d for group in project.get("dependency-groups", {}).values() for d in group if isinstance(d, str)),
            *project.get("tool", {}).get("uv", {}).get("dev-dependencies", []),
        ]
        for text in declared:
            named = re.match(r"^\s*nats-py\s*(?P<specifier>[^;\[]*)", text)
            if named is not None:
                found[path.relative_to(root).as_posix()] = re.sub(r"\s", "", named.group("specifier"))
    assert found, "no pyproject declares nats-py"
    wrong = {where: specifier for where, specifier in found.items() if specifier != wanted}
    assert not wrong, f"nats-py must be declared {wanted} (the minor the suites run): {wrong}"


def test_the_check_reports_an_attribute_nats_py_does_not_have() -> None:
    """non-vacuity: the check above can fail, for an absent attribute and for a changed kind."""
    surface = (
        PrivateAttribute("Client", "_renamed_in_a_future_release", list),
        PrivateAttribute("Subscription", "_id", str),
        PrivateAttribute("Client", "_pending", "method"),
        PrivateAttribute("_SelectorSocketTransport", "_buffer", "callable"),
        PrivateAttribute("TcpTransport", "_io_writer_renamed", "declared"),
    )
    assert missing_private_attributes(surface) == [
        "Client._renamed_in_a_future_release: absent",
        "Subscription._id: expected str, found int",
        "Client._pending: no longer a coroutine method",
        "_SelectorSocketTransport._buffer: no longer callable",
        "TcpTransport._io_writer_renamed: absent",
    ]


def test_the_declared_surface_covers_every_class_the_wrapper_reaches_into() -> None:
    """the surface names the four nats-py classes and the one asyncio class the wrapper reaches into."""
    assert {attribute.owner for attribute in PRIVATE_SURFACE} == {
        "Client",
        "Subscription",
        "PullSubscription",
        "TcpTransport",
        "_SelectorSocketTransport",
    }


def test_the_running_python_sends_through_the_path_the_send_bound_replaces() -> None:
    """asyncio's socket transport still sends a large write through ``_get_sendmsg_buffer``, the method the bound sets."""
    assert send_path_gaps() == []


def test_the_send_path_check_names_each_step_a_python_release_could_change() -> None:
    """non-vacuity: a transport without ``sendmsg``, with a missing method, or whose send no longer reads the
    replaced method is named, step by step."""

    class _ChangedTransport:
        def __init__(self) -> None:
            self.ready = self.send_plainly

        def send_plainly(self) -> None:
            return None

        def writelines(self) -> None:
            return None

    changed = types.ModuleType("selector_events")
    setattr(changed, _asyncio_send_path.TRANSPORT_NAME, _ChangedTransport)
    assert send_path_gaps(changed) == [
        "asyncio.selector_events._HAS_SENDMSG: not True, so the socket transport sends with send() and never "
        "reads _get_sendmsg_buffer",
        "_SelectorSocketTransport.__init__: no longer reads _write_sendmsg",
        "_SelectorSocketTransport.writelines: no longer reads _write_ready",
        "_SelectorSocketTransport._write_sendmsg: absent",
        "_SelectorSocketTransport._write_sendmsg: absent",
        "_SelectorSocketTransport._get_sendmsg_buffer: absent",
    ]
    assert send_path_gaps(types.ModuleType("selector_events")) == [
        "asyncio.selector_events._SelectorSocketTransport: absent"
    ]


def _declared_python_minors() -> list[str]:
    """every released CPython minor the workspace's ``requires-python`` admits, floor to the newest uv knows."""
    root = Path(__file__).resolve().parents[4]
    floors: set[tuple[int, int]] = set()
    for path in sorted([root / "pyproject.toml", *root.glob("packages/**/pyproject.toml")]):
        if ".venv" in path.parts:
            continue
        declared = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {}).get("requires-python")
        if declared is None:
            continue
        found = re.fullmatch(r">=\s*3\.(\d+)", declared.strip())
        assert found is not None, f"{path.relative_to(root)}: requires-python {declared!r} is not an open >=3.N floor"
        floors.add((3, int(found.group(1))))
    assert floors, "no pyproject declares requires-python"
    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH: the send path must be checked under every declared Python minor"
    listed = subprocess.run(
        [uv, "python", "list", "--all-versions", "--output-format", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    released = {
        (entry["version_parts"]["major"], entry["version_parts"]["minor"])
        for entry in json.loads(listed.stdout)
        if entry["implementation"] == "cpython" and re.fullmatch(r"\d+\.\d+\.\d+", entry["version"])
    }
    floor = min(floors)
    return [f"{major}.{minor}" for major, minor in sorted(released) if (major, minor) >= floor]


def test_every_declared_python_minor_sends_through_the_path_the_send_bound_replaces() -> None:
    """Flips if a Python the packages admit changed asyncio's send path: the bound would silently stop applying there
    while the suites, on one interpreter, stayed green. The check imports only the standard library, so it runs by
    path in a bare interpreter of each minor (uv fetches one that is not installed)."""
    minors = _declared_python_minors()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in minors, minors
    uv = shutil.which("uv")
    assert uv is not None
    gaps: dict[str, object] = {}
    for minor in minors:
        ran = subprocess.run(
            [uv, "run", "--no-project", "--python", minor, "python", "-I", _asyncio_send_path.__file__],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert ran.returncode == 0, f"Python {minor}: the send path check did not run: {ran.stderr[-2000:]}"
        found = json.loads(ran.stdout.strip().splitlines()[-1])
        if found:
            gaps[minor] = found
    assert not gaps, f"asyncio's send path differs on a declared Python minor: {gaps}"


def test_a_changed_send_path_refuses_the_import() -> None:
    """an interpreter whose socket transport no longer sends with sendmsg cannot import the bound, rather than run
    every large publish unbounded."""
    ran = subprocess.run(
        [
            sys.executable,
            "-c",
            "from asyncio import selector_events; selector_events._HAS_SENDMSG = False; "
            "import threetears.nats._nats_py_internals",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode != 0
    assert "ImportError: asyncio's socket send path changed" in ran.stderr
    assert "_HAS_SENDMSG: not True" in ran.stderr


def test_one_send_on_a_real_socket_takes_no_more_than_the_bound() -> None:
    """over a loopback socket whose kernel buffers hold all of an 8 MB write, the bounded transport's one send takes at
    most a megabyte; left as asyncio built it, the same send takes it all (so the check can fail)."""
    assert send_bound_gaps() == []
    assert send_bound_gaps(bounded=False) == [
        "one send of a 8388608-byte write took 8388608 bytes; the bound is 1048576"
    ]


def test_a_transport_the_bound_cannot_reach_is_logged_once_never_skipped_unseen(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """another event loop's plain transport (uvloop's, say) is left as it is, with one warning per transport class."""

    class _OtherLoopTransport(asyncio.Transport):
        pass

    with caplog.at_level(logging.WARNING, logger="threetears.nats._nats_py_internals"):
        assert not bound_transport(_OtherLoopTransport())
        assert not bound_transport(_OtherLoopTransport())
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 1, warned
    assert "_OtherLoopTransport transport is not bounded per send" in warned[0]


def test_a_tls_transport_is_left_alone_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """TLS writes through asyncio's SSL transport, which is not bounded by design: nothing is logged for it."""

    class _TlsTransport(asyncio.Transport):
        pass

    with caplog.at_level(logging.WARNING, logger="threetears.nats._nats_py_internals"):
        assert not bound_transport(_TlsTransport(extra={"sslcontext": object()}))
    assert caplog.records == []


def test_taking_the_queue_keeps_nats_py_pending_bytes_right() -> None:
    """taking queued messages empties the queue and the byte count nats-py tracks beside it.

    The messages arrive the way they do in production: a server delivers them to a subscription
    with no callback, and nats-py's read loop queues them and counts their bytes. The queue and the
    count are then read through nats-py's public ``pending_msgs`` and ``pending_bytes``.
    """

    async def scenario() -> None:
        async with wire_server() as server:
            connection = Client()
            await connection.connect(server.url, allow_reconnect=False)
            try:
                subscription = await connection.subscribe("inbox")
                await connection.flush()
                for data in (b"one", b"three"):
                    await server.deliver("inbox", data)
                async with asyncio.timeout(2.0):
                    while subscription.pending_msgs < 2:
                        await asyncio.sleep(0.01)
                assert subscription.pending_bytes == len(b"one") + len(b"three")

                taken = take_queued_messages(subscription)

                assert [msg.data for msg in taken] == [b"one", b"three"]
                assert subscription.pending_msgs == 0
                assert subscription.pending_bytes == 0
            finally:
                await connection.close()

    asyncio.run(scenario())


_BASE62 = set((string.digits + string.ascii_uppercase + string.ascii_lowercase).encode())


def test_a_csprng_token_is_what_nats_py_slices_and_extends() -> None:
    """22 base62 bytes, in a fresh mutable bytearray: nats-py slices at 22 and extends the result."""
    nuid = CsprngNuid()
    first, second = nuid.next(), nuid.next()

    assert isinstance(first, bytearray)
    assert len(first) == NUID_LENGTH == 22
    assert set(first) <= _BASE62
    assert first is not second
    first.extend(b"ab")  # nats-py appends token_hex(2) to a request token
    assert len(second) == NUID_LENGTH


def test_csprng_tokens_share_no_prefix_and_no_step() -> None:
    """10,000 draws: no token repeats, no 12-character prefix repeats, no constant step between them.

    nats-py's NUID holds one 12-character prefix and adds a fixed increment, so one observed token
    predicts the next; these must not.
    """
    nuid = CsprngNuid()
    tokens = [bytes(nuid.next()) for _ in range(10_000)]

    assert len(set(tokens)) == len(tokens)
    assert len({token[:12] for token in tokens}) == len(tokens)
    alphabet = sorted(_BASE62)

    def value(token: bytes) -> int:
        number = 0
        for char in token[12:]:
            number = number * 62 + alphabet.index(char)
        return number

    steps = {value(b) - value(a) for a, b in zip(tokens, tokens[1:], strict=False)}
    assert len(steps) > 9_000


def test_nats_py_own_nuid_fails_the_same_checks() -> None:
    """non-vacuity: the sequential NUID these tests replace shares its prefix across draws."""
    from nats.nuid import NUID

    nuid = NUID()
    tokens = [bytes(nuid.next()) for _ in range(100)]
    assert len({token[:12] for token in tokens}) == 1


def test_the_declared_surface_names_the_nuid_override() -> None:
    """the NUID override is declared, so a nats-py release that renames ``_nuid`` fails by name."""
    assert PrivateAttribute("Client", "_nuid", __import__("nats.nuid", fromlist=["NUID"]).NUID) in PRIVATE_SURFACE


def test_a_threetears_connection_names_its_inboxes_from_the_csprng() -> None:
    """every inbox a NatsClient's connection names is unpredictable from any other it named.

    Through the front door: a NatsClient connects (to an in-process server, so no JetStream check)
    and its connection's inboxes are read. Under nats-py's own NUID all of them share one prefix.
    """
    from threetears.nats import NatsClient

    async def scenario() -> None:
        async with wire_server() as server:
            client = await NatsClient.connect(
                nats_url=server.url,
                nats_subject_namespace="3tears",
                client_name="csprng-check",
                verify_jetstream=False,
            )
            try:
                tokens = [client.raw.new_inbox().removeprefix("_INBOX.") for _ in range(200)]
            finally:
                await client.shutdown(drain_timeout=timedelta(seconds=1))
            assert all(len(token) == NUID_LENGTH for token in tokens)
            assert len({token[:12] for token in tokens}) == len(tokens)

    asyncio.run(scenario())
