"""integration tests for agent-workspace.

the tests in this package wire multiple real components together (real
:class:`WorkspaceFileLease` + real :class:`KVLease` + fake NATS KV) and
exercise multi-pod behaviours that cannot be unit-tested with pure
mocks. they are marked ``integration`` so the unit-test suite remains
fast by default; CI runs this directory separately.
"""
