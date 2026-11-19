"""``BuildOnce``: a value built at most once per key, however many threads ask for it first."""

from __future__ import annotations

import threading
import time

import pytest

from threetears.observe import BuildOnce

_THREADS = 8


def _race(cache: BuildOnce[str, object], build: object) -> list[object]:
    """have several threads, released together, ask *cache* for one key.

    :param cache: the cache under test
    :ptype cache: BuildOnce[str, object]
    :param build: the build callable every thread passes
    :ptype build: object
    :return: what each thread got
    :rtype: list[object]
    """
    start = threading.Barrier(_THREADS)
    got: list[object] = []
    record = threading.Lock()

    def ask() -> None:
        start.wait()
        value = cache.get("key", build)  # type: ignore[arg-type]
        with record:
            got.append(value)

    workers = [threading.Thread(target=ask) for _ in range(_THREADS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    return got


class TestBuiltOnce:
    def test_threads_asking_first_at_once_share_one_build(self) -> None:
        """the build is slow enough that every thread arrives while it runs."""
        builds: list[object] = []

        def build() -> object:
            time.sleep(0.05)
            value = object()
            builds.append(value)
            return value

        got = _race(BuildOnce(), build)

        assert len(builds) == 1
        assert len(got) == _THREADS
        assert all(value is builds[0] for value in got)

    def test_each_key_is_built_separately(self) -> None:
        cache: BuildOnce[str, list[str]] = BuildOnce()

        first = cache.get("a", lambda: ["a"])
        second = cache.get("b", lambda: ["b"])

        assert first == ["a"]
        assert second == ["b"]
        assert cache.get("a", lambda: ["rebuilt"]) is first

    @pytest.mark.parametrize("value", [None, False, 0, ""])
    def test_a_falsy_value_is_stored_not_rebuilt(self, value: object) -> None:
        """a probe that answers ``False`` is an answer; only absence means "build"."""
        cache: BuildOnce[str, object] = BuildOnce()
        builds: list[object] = []

        def build() -> object:
            builds.append(value)
            return value

        cache.get("key", build)
        cache.get("key", build)

        assert builds == [value]

    def test_a_build_that_raises_stores_nothing(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce()

        def broken() -> str:
            raise RuntimeError("could not build")

        with pytest.raises(RuntimeError):
            cache.get("key", broken)

        assert cache.peek("key") is None
        assert cache.get("key", lambda: "built") == "built"


class TestStaleValues:
    def test_a_value_that_is_no_longer_current_is_rebuilt(self) -> None:
        alive = {"first": True}
        cache: BuildOnce[str, str] = BuildOnce(is_current=lambda value: alive.get(value, True))

        first = cache.get("key", lambda: "first")
        alive["first"] = False
        second = cache.get("key", lambda: "second")

        assert (first, second) == ("first", "second")
        assert cache.get("key", lambda: "third") == "second"

    def test_peek_returns_a_stale_value_without_building(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce(is_current=lambda value: False)
        cache.get("key", lambda: "stale")

        assert cache.peek("key") == "stale"


class TestTakingValuesOut:
    def test_pop_takes_the_value_so_the_next_get_builds(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce()
        cache.get("key", lambda: "first")

        assert cache.pop("key") == "first"
        assert cache.pop("key") is None
        assert cache.get("key", lambda: "second") == "second"

    def test_clear_disposes_of_every_value_before_dropping_it(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce()
        cache.get("a", lambda: "one")
        cache.get("b", lambda: "two")
        disposed: list[str] = []

        cache.clear(dispose=disposed.append)

        assert sorted(disposed) == ["one", "two"]
        assert cache.peek("a") is None
        assert cache.peek("b") is None


class TestACostBound:
    """``cost`` + ``max_cost``: stored values cost at most the cap together; what does not fit is
    handed out unstored, and nothing stored is evicted for it."""

    def test_values_are_stored_until_the_next_one_would_pass_the_cap(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce(cost=len, max_cost=5)

        for key in ("abc", "de", "f"):
            assert cache.get(key, lambda key=key: key) == key  # type: ignore[misc]

        assert cache.peek("abc") == "abc"
        assert cache.peek("de") == "de"
        assert cache.peek("f") is None

    def test_a_value_that_does_not_fit_is_built_on_every_get(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce(cost=len, max_cost=3)
        cache.get("held", lambda: "abc")
        builds: list[int] = []

        def build() -> str:
            builds.append(1)
            return "xyz"

        assert [cache.get("over", build) for _ in range(3)] == ["xyz"] * 3
        assert len(builds) == 3
        assert cache.peek("held") == "abc"

    def test_a_walk_in_the_same_order_again_keeps_finding_what_was_stored(self) -> None:
        cache: BuildOnce[int, int] = BuildOnce(cost=lambda value: 1, max_cost=4)
        builds: list[int] = []

        def build(key: int) -> int:
            builds.append(key)
            return key

        for _ in range(3):
            for key in range(10):
                cache.get(key, lambda key=key: build(key))  # type: ignore[misc]

        assert builds.count(0) == 1
        assert builds.count(3) == 1
        assert builds.count(9) == 3

    def test_a_stale_value_gives_its_cost_back(self) -> None:
        live = {"old": True}
        cache: BuildOnce[str, str] = BuildOnce(is_current=lambda value: live.get(value, True), cost=len, max_cost=3)
        cache.get("key", lambda: "old")
        live["old"] = False

        assert cache.get("key", lambda: "new") == "new"
        assert cache.peek("key") == "new"

    def test_a_stale_value_whose_rebuild_does_not_fit_is_dropped(self) -> None:
        live = {"old": True}
        cache: BuildOnce[str, str] = BuildOnce(is_current=lambda value: live.get(value, True), cost=len, max_cost=3)
        cache.get("key", lambda: "old")
        live["old"] = False

        assert cache.get("key", lambda: "longer") == "longer"
        assert cache.peek("key") is None
        assert cache.get("other", lambda: "abc") == "abc"
        assert cache.peek("other") == "abc"

    def test_pop_and_clear_give_the_cost_back(self) -> None:
        cache: BuildOnce[str, str] = BuildOnce(cost=len, max_cost=3)
        cache.get("a", lambda: "abc")
        cache.pop("a")
        cache.get("b", lambda: "def")
        assert cache.peek("b") == "def"

        cache.clear()
        cache.get("c", lambda: "ghi")
        assert cache.peek("c") == "ghi"

    @pytest.mark.parametrize(("cost", "max_cost"), [(len, None), (None, 3)])
    def test_cost_and_max_cost_come_together(self, cost: object, max_cost: int | None) -> None:
        with pytest.raises(ValueError, match="both or neither"):
            BuildOnce(cost=cost, max_cost=max_cost)  # type: ignore[arg-type]

    def test_a_negative_cap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="0 or more"):
            BuildOnce(cost=len, max_cost=-1)
