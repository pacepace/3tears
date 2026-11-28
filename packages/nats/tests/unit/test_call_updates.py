"""a call's updates: the two subject families, who owns which subject, the rate bound, and the grants that carry them."""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from threetears.nats import (
    Principal,
    Subjects,
    UpdateBudget,
    UpdateDrain,
    build_permissions,
    progress_subject_is_owned_by_pod,
    set_default_namespace,
    updates_subject_is_owned_by_caller,
)

_NS = "test"
_POD = "01947100-0000-7000-8000-000000000002"
_AGENT = "019470a8-b5c3-7def-8123-000000000001"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


class TestTheSubjects:
    def test_the_pod_publishes_under_its_own_id_and_the_caller_reads_under_its_own(self) -> None:
        call = uuid7()
        assert Subjects.tools_progress(_POD, call).path == f"{_NS}.tools.progress.{_POD}.{call}"
        assert Subjects.tools_updates(_AGENT, call).path == f"{_NS}.tools.updates.{_AGENT}.{call}"

    def test_an_in_process_pod_nests_under_its_agents_subtree(self) -> None:
        inprocess = Subjects.agent_inprocess_pod_id(UUID(_AGENT), "instance-1")
        subject = Subjects.tools_progress(inprocess, uuid7()).path
        assert subject.startswith(str(Subjects.tools_progress_agent_subtree(_AGENT)).removesuffix(">"))

    def test_the_families_do_not_overlap_the_result_families(self) -> None:
        assert not Subjects.tools_progress(_POD, uuid7()).path.startswith(f"{_NS}.tools.result.")
        assert not Subjects.tools_updates(_AGENT, uuid7()).path.startswith(f"{_NS}.tools.reply.")


class TestOwnership:
    def test_a_pod_owns_its_own_subject(self) -> None:
        assert progress_subject_is_owned_by_pod(Subjects.tools_progress(_POD, uuid7()).path, pod_id=_POD)

    def test_a_pod_does_not_own_another_pods_subject(self) -> None:
        other = Subjects.tools_progress("another-pod", uuid7()).path
        assert not progress_subject_is_owned_by_pod(other, pod_id=_POD)

    def test_a_pod_owns_nothing_in_another_namespace(self) -> None:
        subject = Subjects.tools_progress(_POD, uuid7()).path
        assert not progress_subject_is_owned_by_pod(subject, pod_id=_POD, namespace="elsewhere")

    @pytest.mark.parametrize("suffix", [".deeper", "*", ">"], ids=["one-token-too-deep", "wildcard", "tail-wildcard"])
    def test_anything_but_one_plain_token_below_the_prefix_is_refused(self, suffix: str) -> None:
        base = f"{_NS}.tools.progress.{_POD}."
        token = "call" if suffix.startswith(".") else ""
        assert not progress_subject_is_owned_by_pod(f"{base}{token}{suffix}", pod_id=_POD)
        assert not updates_subject_is_owned_by_caller(f"{_NS}.tools.updates.{_AGENT}.{token}{suffix}", caller_id=_AGENT)

    def test_a_caller_owns_its_own_updates_and_no_peers(self) -> None:
        own = Subjects.tools_updates(_AGENT, uuid7()).path
        assert updates_subject_is_owned_by_caller(own, caller_id=_AGENT)
        assert not updates_subject_is_owned_by_caller(own, caller_id=_POD)
        assert not updates_subject_is_owned_by_caller(f"{_NS}.tools.updates.{_AGENT}.", caller_id=_AGENT)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestTheBudget:
    def test_it_spends_its_burst_then_refills_at_its_rate(self) -> None:
        clock = _Clock()
        budget = UpdateBudget(rate=10.0, burst=3, clock=clock)

        assert [budget.take() for _ in range(4)] == [True, True, True, False]
        assert budget.wait() == pytest.approx(0.1)
        clock.now += 0.11
        assert budget.wait() == 0.0
        assert budget.take() is True

    def test_it_never_holds_more_than_its_burst(self) -> None:
        clock = _Clock()
        budget = UpdateBudget(rate=10.0, burst=2, clock=clock)
        clock.now += 3600
        assert [budget.take() for _ in range(3)] == [True, True, False]

    @pytest.mark.parametrize(("rate", "burst"), [(0.0, 1), (-1.0, 1), (1.0, 0)])
    def test_a_budget_that_could_never_send_is_refused(self, rate: float, burst: int) -> None:
        with pytest.raises(ValueError, match="positive rate"):
            UpdateBudget(rate=rate, burst=burst)


class TestTheGrants:
    def test_a_tool_pod_publishes_its_own_progress_and_reads_its_own_updates(self) -> None:
        permissions = build_permissions(Principal.TOOL_POD, pod_id=_POD)
        assert str(Subjects.tools_progress_pod_wildcard(_POD)) in permissions.publish
        assert str(Subjects.tools_updates_caller_subtree(_POD)) in permissions.subscribe
        assert str(Subjects.tools_progress_wildcard()) not in permissions.subscribe

    def test_an_agent_pod_publishes_its_in_process_progress_under_its_own_subtree(self) -> None:
        permissions = build_permissions(Principal.AGENT_POD, agent_id=_AGENT, pod_id=_POD)
        assert str(Subjects.tools_progress_agent_subtree(_AGENT)) in permissions.publish

    def test_the_registry_reads_every_pods_progress_and_writes_every_callers_updates(self) -> None:
        permissions = build_permissions(Principal.REGISTRY, conn_id="registry-1")
        assert str(Subjects.tools_progress_wildcard()) in permissions.subscribe
        assert str(Subjects.tools_updates_wildcard()) in permissions.publish


class _ManualTime:
    """a clock and a sleep that advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestTheDrain:
    @pytest.mark.asyncio
    async def test_it_settles_once_nothing_has_arrived_for_the_quiet(self) -> None:
        time = _ManualTime()
        drain = UpdateDrain(clock=time.clock, sleep=time.sleep)
        with drain:
            pass

        assert await drain.settled(quiet=0.05, limit=1.0) is True
        assert 0.05 <= time.now < 0.1

    @pytest.mark.asyncio
    async def test_an_update_being_handled_holds_it_until_the_limit(self) -> None:
        time = _ManualTime()
        drain = UpdateDrain(clock=time.clock, sleep=time.sleep)
        drain.__enter__()

        assert await drain.settled(quiet=0.05, limit=0.5) is False
        assert time.now == pytest.approx(0.5)
