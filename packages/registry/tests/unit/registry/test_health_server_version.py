"""the registry's health server names the release that is answering.

``python -m threetears.registry`` runs inside a host's image, so the image tag says which HOST
build is deployed while the registry code is whatever ``3tears-registry`` that image installed.
the probe body and the startup line carry that installed version, read from the distribution's
own metadata rather than a second hardcoded string.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from threetears.observe import HealthTier, InflightRequestsGauge
from threetears.registry import server as server_module
from threetears.registry.auth import AllowAllAuthorizer
from threetears.registry.server import RegistryServer


class TestTheRegistryHealthServerCarriesItsVersion:
    """the version passthrough from the server to its health surface."""

    @pytest.mark.asyncio
    async def test_the_built_health_server_echoes_the_constructor_version(self) -> None:
        """the status the probe renders carries the version the server was given.

        :return: nothing
        :rtype: None
        """
        server = RegistryServer(authorizer=AllowAllAuthorizer(), health_port=0, version="9.9.9")
        gauge = InflightRequestsGauge("test_registry_version_inflight")

        health = server._build_health_server(gauge)  # noqa: SLF001 - the builder under test

        assert health.version == "9.9.9"
        status = await health.get_status(HealthTier.LIVE)
        assert status.version == "9.9.9"

    def test_an_embedding_caller_that_passes_no_version_still_builds(self) -> None:
        """the parameter is optional: an embedding host passing none keeps working.

        :return: nothing
        :rtype: None
        """
        server = RegistryServer(authorizer=AllowAllAuthorizer(), health_port=0)
        gauge = InflightRequestsGauge("test_registry_no_version_inflight")

        health = server._build_health_server(gauge)  # noqa: SLF001 - the builder under test

        assert health.version is None

    def test_the_entry_point_passes_the_installed_distribution_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``python -m threetears.registry`` reports the installed ``3tears-registry`` version.

        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv("THREETEARS_REGISTRY_ALLOW_ALL_TOOLS", "true")
        captured: dict[str, object] = {}

        class _CapturingServer:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def serve(self) -> None:
                return None

        monkeypatch.setattr(server_module, "RegistryServer", _CapturingServer)

        server_module._run_server()  # noqa: SLF001 - the module entry point under test

        installed = importlib.metadata.version("3tears-registry")
        assert installed
        assert captured["version"] == installed
