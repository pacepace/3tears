"""3tears' configuration of the sender-posture walker, shared by the gates that walk NATS receivers.

``test_subscribers_declare_sender_posture.py`` holds every receiver to a declared posture;
``test_subject_grants_match_the_code.py`` places each receiver it finds against the principals that
run it. Both read the one configuration here, so they walk the same surface.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.sender_posture import SenderPostureConfig

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

SENDER_POSTURE_CONFIG = SenderPostureConfig(
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
