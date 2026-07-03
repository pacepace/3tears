"""tests for ToolProvisioningStrategy protocol and BootstrapContext."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from threetears.agent.tools.strategy import (
    BootstrapContext,
    ToolProvisioningStrategy,
)


class _FakeStrategy:
    """fake strategy that records provisioning + teardown calls.

    used to prove the Protocol is runtime_checkable and that an
    arbitrary object satisfying the four-method shape passes
    ``isinstance`` against it.
    """

    def __init__(self) -> None:
        """initialize recording buffers."""
        self.provisioned: list[BootstrapContext] = []
        self.ready_calls: list[float] = []
        self.reload_calls: list[tuple[Any, Any]] = []
        self.teardown_count = 0

    async def provision(self, bootstrap_context: BootstrapContext) -> None:
        """record the context and return.

        :param bootstrap_context: shared bootstrap handles
        :ptype bootstrap_context: BootstrapContext
        :return: nothing
        :rtype: None
        """
        self.provisioned.append(bootstrap_context)

    async def await_ready(self, timeout: float) -> None:
        """record the timeout and return.

        :param timeout: maximum seconds to wait
        :ptype timeout: float
        :return: nothing
        :rtype: None
        """
        self.ready_calls.append(timeout)

    async def reload_workspace_tools(
        self,
        workspace_runtime: Any,
        workspace_config: Any,
    ) -> None:
        """record the reload call.

        :param workspace_runtime: runtime for the reloaded bundle
        :ptype workspace_runtime: Any
        :param workspace_config: config for the reloaded bundle
        :ptype workspace_config: Any
        :return: nothing
        :rtype: None
        """
        self.reload_calls.append((workspace_runtime, workspace_config))

    async def teardown(self) -> None:
        """record the teardown invocation.

        :return: nothing
        :rtype: None
        """
        self.teardown_count += 1


class TestBootstrapContextIsFrozen:
    """BootstrapContext is a frozen dataclass; mutation fails at runtime."""

    def test_mutation_raises_frozen_instance_error(self) -> None:
        """assigning to any field raises FrozenInstanceError."""
        context = BootstrapContext(
            nats_client=MagicMock(),
            agent_id=uuid4(),
            namespace="3tears",
        )
        with pytest.raises(FrozenInstanceError):
            context.namespace = "changed"  # type: ignore[misc]

    def test_default_optional_fields_are_none(self) -> None:
        """workspace_runtime, registry_client, bootstrap_token default to None."""
        context = BootstrapContext(
            nats_client=MagicMock(),
            agent_id=uuid4(),
            namespace="3tears",
        )
        assert context.workspace_runtime is None
        assert context.registry_client is None
        assert context.bootstrap_token is None

    def test_populated_optional_fields_preserve_values(self) -> None:
        """supplied optional fields are accessible on the frozen instance."""
        workspace_runtime = MagicMock(name="workspace_runtime")
        registry_client = MagicMock(name="registry_client")
        context = BootstrapContext(
            nats_client=MagicMock(),
            agent_id=uuid4(),
            namespace="3tears",
            bootstrap_token="token-abc",
            workspace_runtime=workspace_runtime,
            registry_client=registry_client,
        )
        assert context.bootstrap_token == "token-abc"
        assert context.workspace_runtime is workspace_runtime
        assert context.registry_client is registry_client


class TestToolProvisioningStrategyIsRuntimeCheckable:
    """the protocol is @runtime_checkable; isinstance works on fakes."""

    def test_fake_satisfies_protocol(self) -> None:
        """any object with the three async methods is a strategy."""
        fake = _FakeStrategy()
        assert isinstance(fake, ToolProvisioningStrategy)

    def test_object_missing_method_fails_isinstance(self) -> None:
        """object missing any of the three methods fails the check."""

        class _Incomplete:
            """partial strategy missing teardown."""

            async def provision(self, bootstrap_context: BootstrapContext) -> None:
                """noop."""
                return None

            async def await_ready(self, timeout: float) -> None:
                """noop."""
                return None

        incomplete = _Incomplete()
        assert not isinstance(incomplete, ToolProvisioningStrategy)


class TestFakeStrategyRecordsInvocations:
    """the fake strategy captures what was provisioned so tests can assert."""

    async def test_provision_records_context(self) -> None:
        """provision appends the received context to the provisioned list."""
        fake = _FakeStrategy()
        context = BootstrapContext(
            nats_client=MagicMock(),
            agent_id=uuid4(),
            namespace="3tears",
        )
        await fake.provision(context)
        assert len(fake.provisioned) == 1
        assert fake.provisioned[0] is context

    async def test_await_ready_records_timeout(self) -> None:
        """await_ready records the timeout argument."""
        fake = _FakeStrategy()
        await fake.await_ready(5.0)
        assert fake.ready_calls == [5.0]

    async def test_teardown_increments_counter(self) -> None:
        """teardown bumps the teardown counter so tests can assert it ran."""
        fake = _FakeStrategy()
        await fake.teardown()
        await fake.teardown()
        assert fake.teardown_count == 2
