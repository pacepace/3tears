"""test-isolation support for hosts that run threetears.nats under pytest.

The subject namespace set by :func:`threetears.nats.set_default_namespace` (and by
:meth:`threetears.nats.NatsClient.connect`) is a process-wide module global by design: a
``ContextVar`` set at connect time is invisible to the sibling task trees that serve later work,
which once broke subject resolution in production. A process-wide value does not clear itself
between tests, so a test that connects a client, or sets the namespace, would shadow the
namespace of every test that runs after it.

This module is the published way to restore the unconfigured state between tests, the same role
:mod:`threetears.core.testing` plays for collections. Production code has no reason to unconfigure
a deployment constant and never imports it; host test suites (this workspace's root
``conftest.py``, the hub's and the SDK's) call :func:`reset_default_namespace` from an autouse
fixture.
"""

from __future__ import annotations

from threetears.nats.subjects import _reset_default_namespace

__all__ = ["reset_default_namespace"]


def reset_default_namespace() -> None:
    """clear the process-wide subject namespace back to unconfigured.

    After this, :func:`threetears.nats.get_default_namespace` resolves from the
    ``THREETEARS_NATS_SUBJECT_NAMESPACE`` environment variable again, or raises
    :class:`~threetears.nats.errors.NamespaceNotConfiguredError` when that is unset too.

    :return: nothing
    :rtype: None
    """
    _reset_default_namespace()
