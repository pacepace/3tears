"""the registry's health server names the release that is answering.

``python -m threetears.registry`` runs inside a host's image, so the image tag says which HOST
build is deployed while the registry code is whatever ``3tears-registry`` that image installed.
the probe body and the startup line carry that installed version, read from the distribution's
own metadata rather than a second hardcoded string.

that the running server's probe body carries the version it was constructed with is pinned
against a registry started with ``serve()``, in
``tests/integration/test_registry_server_lifecycle_live.py``.
"""

from __future__ import annotations

import importlib.metadata
import runpy

import pytest

from threetears.registry import server as server_module


class TestTheRegistryHealthServerCarriesItsVersion:
    """the version passthrough from the server to its health surface."""

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

        # `python -m threetears.registry`, the entry point a deployment runs.
        runpy.run_module("threetears.registry", run_name="__main__")

        installed = importlib.metadata.version("3tears-registry")
        assert installed
        assert captured["version"] == installed
