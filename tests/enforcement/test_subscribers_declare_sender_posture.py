"""enforcement: every NATS receiver in 3tears says how it authenticates whoever sent its messages.

a subject proves nothing about its sender. any connection can have its own bytes delivered on
any subject, through a push consumer's ``deliver_subject`` or a request's reply subject, so a
receiver that acts on what arrives has to say what it checked. the walker and the posture
vocabulary live in :mod:`threetears.enforcement.sender_posture`; this file is 3tears'
configuration of it, and ``_sender_posture_allowlist.txt`` beside it is the shrinking list of
receivers still open under an audit row.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.sender_posture import (
    SenderPostureConfig,
    find_subscribe_sites,
    sender_posture_findings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: every workspace member's source tree, per the root pyproject's workspace globs.
_SOURCE_ROOTS = tuple(
    sorted(
        path
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)

_CONFIG = SenderPostureConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SOURCE_ROOTS,
    allowlist_path=_REPO_ROOT / "tests" / "enforcement" / "_sender_posture_allowlist.txt",
    # the functions that implement subscribing itself; their callers declare the posture. named one
    # by one because client.py also holds real receivers (the result waiter, the refusal and renewal
    # handlers) that must declare what they check.
    primitive_functions=frozenset(
        {
            "packages/nats/src/threetears/nats/client.py::Subscription.subscribe_on",
            "packages/nats/src/threetears/nats/client.py::NatsClient.renew_connection",
            "packages/nats/src/threetears/nats/client.py::NatsClient._subscribe_internal",
            "packages/nats/src/threetears/nats/client.py::NatsClient.jetstream_subscribe_durable._bind",
            "packages/nats/src/threetears/nats/client.py::NatsClient.jetstream_pull_subscribe._bind",
            "packages/nats/src/threetears/nats/forward.py::serve_owner",
            "packages/nats/src/threetears/nats/pipe.py::PipeStream.open",
            "packages/nats/src/threetears/nats/pipe.py::serve_pipe",
        }
    ),
)


class TestSubscribersDeclareSenderPosture:
    """No receiver ships without saying how it knows who sent its messages."""

    def test_the_walker_sees_the_receivers(self) -> None:
        """
        a walk that found no receivers checked nothing, so the gate refuses to read it as green.

        :return: nothing
        :rtype: None
        """
        assert len(_SOURCE_ROOTS) > 20, f"the workspace globs found only {_SOURCE_ROOTS}"
        sites = find_subscribe_sites(_CONFIG)
        assert len(sites) >= 25, f"found only {len(sites)} receiving sites; the walker is not seeing the source"
        paths = {site.path for site in sites}
        assert "packages/agent/tools/src/threetears/agent/tools/server.py" in paths
        assert "packages/registry/src/threetears/registry/proxy.py" in paths

    def test_every_receiver_declares_its_posture_and_the_allowlist_matches(self) -> None:
        """
        every site declares a valid posture, and the allowlist carries exactly the open rows.

        :return: nothing
        :rtype: None
        """
        findings = sender_posture_findings(_CONFIG)
        assert not findings, "\n".join(findings)
