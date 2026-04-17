"""tests for threetears.core.security.sandbox abstract base and helpers.

covers :class:`SandboxDecision` enum values, :class:`SandboxDenied`
attribute exposure, and :class:`Sandbox.enforce` DENY/ALLOW dispatch via
a mock subclass.
"""

from __future__ import annotations

from abc import ABC

import pytest

from threetears.core.security.sandbox import (
    Sandbox,
    SandboxDecision,
    SandboxDenied,
)


class _AllowSandbox(Sandbox):
    """sandbox subclass whose ``check`` always returns ALLOW.

    used to exercise :meth:`Sandbox.enforce` success path.
    """

    def check(self, action: str, target: str) -> SandboxDecision:
        """return ALLOW for every input.

        :param action: action verb being evaluated
        :ptype action: str
        :param target: target identifier being evaluated
        :ptype target: str
        :return: always ALLOW
        :rtype: SandboxDecision
        """
        return SandboxDecision.ALLOW


class _DenySandbox(Sandbox):
    """sandbox subclass whose ``check`` always returns DENY.

    used to exercise :meth:`Sandbox.enforce` raise path.
    """

    def check(self, action: str, target: str) -> SandboxDecision:
        """return DENY for every input.

        :param action: action verb being evaluated
        :ptype action: str
        :param target: target identifier being evaluated
        :ptype target: str
        :return: always DENY
        :rtype: SandboxDecision
        """
        return SandboxDecision.DENY


class _DenyWithReasonSandbox(Sandbox):
    """sandbox subclass with rich ``_deny_reason`` override.

    verifies subclass extension point for actionable error text.
    """

    def check(self, action: str, target: str) -> SandboxDecision:
        """return DENY for every input.

        :param action: action verb being evaluated
        :ptype action: str
        :param target: target identifier being evaluated
        :ptype target: str
        :return: always DENY
        :rtype: SandboxDecision
        """
        return SandboxDecision.DENY

    def _deny_reason(self, action: str, target: str) -> str:
        """return rich reason referencing both action and target.

        :param action: action verb being evaluated
        :ptype action: str
        :param target: target identifier being evaluated
        :ptype target: str
        :return: actionable reason string
        :rtype: str
        """
        return f"denied {action} on {target} by test policy"


class TestSandboxDecision:
    """value checks on the :class:`SandboxDecision` str-enum."""

    def test_allow_value(self) -> None:
        assert SandboxDecision.ALLOW.value == "allow"

    def test_deny_value(self) -> None:
        assert SandboxDecision.DENY.value == "deny"

    def test_str_enum_behavior(self) -> None:
        assert SandboxDecision.ALLOW == "allow"
        assert SandboxDecision.DENY == "deny"

    def test_members(self) -> None:
        names = {member.name for member in SandboxDecision}
        assert names == {"ALLOW", "DENY"}


class TestSandboxDenied:
    """attribute exposure and message format for :class:`SandboxDenied`."""

    def test_is_permission_error(self) -> None:
        exc = SandboxDenied("read", "foo", "no")
        assert isinstance(exc, PermissionError)

    def test_attributes_round_trip(self) -> None:
        exc = SandboxDenied("write", "bar/baz.yaml", "no matching glob")
        assert exc.action == "write"
        assert exc.target == "bar/baz.yaml"
        assert exc.reason == "no matching glob"

    def test_message_format(self) -> None:
        exc = SandboxDenied("read", "x", "bad policy")
        assert str(exc) == "sandbox denied 'read' on 'x': bad policy"


class TestSandboxAbcContract:
    """`Sandbox` must be an ABC with abstract ``check`` method."""

    def test_sandbox_is_abc(self) -> None:
        assert issubclass(Sandbox, ABC)

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Sandbox()  # type: ignore[abstract]


class TestSandboxEnforce:
    """`Sandbox.enforce` dispatches on ``check`` result."""

    def test_allow_does_not_raise(self) -> None:
        sb = _AllowSandbox()
        sb.enforce("read", "anything")

    def test_deny_raises_sandbox_denied(self) -> None:
        sb = _DenySandbox()
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "anything")
        assert info.value.action == "read"
        assert info.value.target == "anything"
        assert info.value.reason == "policy denied"

    def test_deny_uses_subclass_reason(self) -> None:
        sb = _DenyWithReasonSandbox()
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("write", "foo.yaml")
        assert info.value.reason == "denied write on foo.yaml by test policy"

    def test_default_deny_reason_on_base(self) -> None:
        sb = _DenySandbox()
        assert sb._deny_reason("x", "y") == "policy denied"
