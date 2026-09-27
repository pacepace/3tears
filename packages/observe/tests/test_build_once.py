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
