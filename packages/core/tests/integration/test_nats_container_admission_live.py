"""Integration test: which clients the session NATS container admits, with and without its SYSTEM account.

Declaring a SYSTEM account makes the server create a hidden no-auth user for the global account, and
nats-server applies a no-auth user only to a client presenting NO credential at all. A client that
presents a connect token -- every tool pod, which opens its own connection with a minted token -- is
then checked against token auth nobody configured, and refused with ``Authorization Violation``. Before
the SYSTEM account existed the server had no auth and ignored the token.

So the account is a choice a consuming suite makes: ``nats_system_account`` (default true, which a
control plane under test needs to kick and ping) or false (a suite whose code under test presents
connect tokens to a bus that, like this one, verifies none). Both shapes are driven here through the
fixture's own body against a real server, because the property is what nats-server answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest

from threetears.core.testing import fixtures
from threetears.nats import NatsClient, NatsClientError

pytestmark = pytest.mark.integration

_NAMESPACE = "admissionlive"
_TOKEN = "a-connect-token-this-bus-does-not-verify"  # noqa: S105 - not a credential, nothing verifies it


def _bus(system_account: bool, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """start one NATS container through the fixture's own body.

    :param system_account: whether the server declares the SYSTEM account
    :ptype system_account: bool
    :param tmp_path_factory: where the server's configuration is written
    :ptype tmp_path_factory: pytest.TempPathFactory
    :return: generator yielding the container's url, stopping it when closed
    :rtype: Iterator[str]
    """
    return fixtures.nats_container.__wrapped__(True, system_account, tmp_path_factory)


async def _admits(url: str, token: str | None) -> bool:
    """report whether the server at ``url`` admits a client presenting ``token``.

    :param url: the bus
    :ptype url: str
    :param token: connect token to present, or ``None`` for none
    :ptype token: str | None
    :return: True when the connection opened (with JetStream reachable)
    :rtype: bool
    """
    admitted = False
    try:
        client = await NatsClient.connect(
            nats_url=url,
            nats_subject_namespace=_NAMESPACE,
            client_name="admission-probe",
            auth_token=(lambda: token) if token is not None else None,
            startup_timeout=timedelta(seconds=5),
        )
    except NatsClientError:
        admitted = False
    else:
        async with client:
            admitted = True
    return admitted


@pytest.mark.parametrize("system_account", [True, False])
async def test_a_client_presenting_no_credential_is_admitted(
    system_account: bool, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """either shape admits an anonymous client to the global account, JetStream included.

    :param system_account: whether the server declares the SYSTEM account
    :ptype system_account: bool
    :param tmp_path_factory: where the server's configuration is written
    :ptype tmp_path_factory: pytest.TempPathFactory
    :param monkeypatch: clears any external bus, so the fixture starts its own
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    monkeypatch.delenv("THREETEARS_TEST_NATS_URL", raising=False)
    bus = _bus(system_account, tmp_path_factory)
    try:
        assert await _admits(next(bus), None) is True
    finally:
        bus.close()


async def test_without_the_system_account_a_client_presenting_a_token_is_admitted(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the shape a suite chooses when its code under test presents connect tokens.

    :param tmp_path_factory: where the server's configuration is written
    :ptype tmp_path_factory: pytest.TempPathFactory
    :param monkeypatch: clears any external bus, so the fixture starts its own
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    monkeypatch.delenv("THREETEARS_TEST_NATS_URL", raising=False)
    bus = _bus(False, tmp_path_factory)
    try:
        assert await _admits(next(bus), _TOKEN) is True
    finally:
        bus.close()


async def test_with_the_system_account_a_client_presenting_a_token_is_refused(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE CONTROL: the server-side rule the opt-out exists for, so the test above discriminates.

    :param tmp_path_factory: where the server's configuration is written
    :ptype tmp_path_factory: pytest.TempPathFactory
    :param monkeypatch: clears any external bus, so the fixture starts its own
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    monkeypatch.delenv("THREETEARS_TEST_NATS_URL", raising=False)
    bus = _bus(True, tmp_path_factory)
    try:
        assert await _admits(next(bus), _TOKEN) is False
    finally:
        bus.close()
