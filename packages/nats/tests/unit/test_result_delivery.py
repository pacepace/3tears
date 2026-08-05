"""unit tests for :mod:`threetears.nats.result_delivery`."""

from __future__ import annotations

import pytest

from threetears.nats import (
    RESULT_ACK_TIMEOUT_SECONDS,
    SYNC_REPLY_BUDGET_SECONDS,
    Subjects,
    reply_subject_is_owned_by_agent,
    requires_async_result,
    result_stream_name,
    result_subject_is_owned_by_pod,
    set_default_namespace,
)

_NS = "3tears"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    """pin the process-wide namespace so subject builders render predictably."""
    set_default_namespace(_NS)


def test_short_calls_stay_on_the_reply_inbox() -> None:
    """a call that fits inside the drain grace keeps the fast synchronous path."""
    assert requires_async_result(1.0) is False
    assert requires_async_result(SYNC_REPLY_BUDGET_SECONDS) is False


def test_long_calls_require_durable_delivery() -> None:
    """past the drain grace the responder can no longer promise it may still answer."""
    assert requires_async_result(SYNC_REPLY_BUDGET_SECONDS + 0.001) is True
    assert requires_async_result(1200.0) is True


def test_unknown_timeout_is_treated_as_long() -> None:
    """guessing "short" silently discards a result; guessing "long" costs one round trip."""
    assert requires_async_result(None) is True


def test_ack_timeout_is_far_below_the_sync_budget() -> None:
    """the accept is answered before any work starts, so a dead pod is detected fast.

    if the accept could take as long as a synchronous call, the registry's failover to a sibling
    endpoint would arrive only after the whole tool budget had elapsed -- which is the dead-pod
    behaviour the failover loop exists to avoid.
    """
    assert RESULT_ACK_TIMEOUT_SECONDS < SYNC_REPLY_BUDGET_SECONDS


def test_result_stream_name_is_namespace_prefixed() -> None:
    """the stream name reads the SAME namespace source the subject factory does.

    a stream named from one namespace and subjects built from another would leave a publisher
    addressing a stream that does not hold its subjects, which surfaces as a publish that never acks
    rather than as a naming mistake.
    """
    assert result_stream_name() == f"{_NS}-tools-results"


def test_pod_owns_its_own_result_subject() -> None:
    """the subject the factory builds for a pod passes that pod's ownership check."""
    subject = Subjects.tools_result("pod-A", "call-1").path
    assert result_subject_is_owned_by_pod(subject, pod_id="pod-A") is True


def test_pod_refuses_a_peers_result_subject() -> None:
    """a responder will not publish under another pod's identity even when told to.

    the broker denies it too, but that denial arrives as an opaque publish failure AFTER the tool has
    run; refusing up front names the offending subject while the call can still be rejected cleanly.
    """
    subject = Subjects.tools_result("pod-B", "call-1").path
    assert result_subject_is_owned_by_pod(subject, pod_id="pod-A") is False


def test_inprocess_composite_pod_owns_its_result_subject() -> None:
    """an agent's in-process pod-id keeps its structural dot on both sides of the check.

    the composite renders as two subject tokens; a prefix built by collapsing it to one would reject
    every in-process tool result, and the failure would look like a permissions problem.
    """
    composite = Subjects.agent_inprocess_pod_id("agent-A", "inst-1")
    subject = Subjects.tools_result(composite, "call-1").path
    assert result_subject_is_owned_by_pod(subject, pod_id=composite) is True


def test_result_subject_must_be_exactly_one_token_deep() -> None:
    """a deeper tail is refused: it would push the publish outside the family the pod owns."""
    prefix = f"{_NS}.tools.result.pod-A."
    assert result_subject_is_owned_by_pod(f"{prefix}call-1.extra", pod_id="pod-A") is False
    assert result_subject_is_owned_by_pod(prefix, pod_id="pod-A") is False


def test_result_subject_may_not_smuggle_a_wildcard() -> None:
    """a wildcard tail would make one publish land on every waiter's consumer at once."""
    assert result_subject_is_owned_by_pod(f"{_NS}.tools.result.pod-A.*", pod_id="pod-A") is False
    assert result_subject_is_owned_by_pod(f"{_NS}.tools.result.pod-A.>", pod_id="pod-A") is False


def test_result_subject_of_a_different_namespace_is_refused() -> None:
    """namespace is part of the identity: a staging pod cannot deliver into production."""
    subject = "prod14.tools.result.pod-A.call-1"
    assert result_subject_is_owned_by_pod(subject, pod_id="pod-A") is False


def test_reply_subject_must_name_the_calling_agent() -> None:
    """the registry's two-token wildcard grant is contained by this check, not by the broker."""
    mine = Subjects.tools_reply("agent-A", "call-1").path
    peer = Subjects.tools_reply("agent-B", "call-1").path
    assert reply_subject_is_owned_by_agent(mine, agent_id="agent-A") is True
    assert reply_subject_is_owned_by_agent(peer, agent_id="agent-A") is False


def test_reply_subject_cannot_be_redirected_into_the_result_family() -> None:
    """a caller cannot name a pod-result subject and have the registry publish there."""
    subject = Subjects.tools_result("pod-A", "call-1").path
    assert reply_subject_is_owned_by_agent(subject, agent_id="agent-A") is False
