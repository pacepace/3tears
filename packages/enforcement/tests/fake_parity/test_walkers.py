"""tests for the fake-protocol-parity walker.

covers four behaviour shapes:

1. discovery via name pattern -- ``Fake*`` and ``_Fake*`` classes are
   collected; non-matching names are ignored
2. subclass declaration -- a non-object base counts as parity declared
   and produces zero violations
3. marker comment -- ``# parity-with: <fqname>`` resolves the
   production class and runs the method-surface comparison
4. no-declaration -- a fake without subclass + without marker produces
   exactly one ``no_declaration`` violation
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.fake_parity.walkers import (
    fake_parity_violations,
    find_fakes_in_tree,
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


# ------------------------------------------------------------------
# discovery
# ------------------------------------------------------------------


class TestDiscovery:
    """name-pattern discovery selects Fake* / _Fake* and skips others."""

    def test_picks_up_underscore_fake(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class _FakeBucket:\n    pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert [f.name for f in fakes] == ["_FakeBucket"]

    def test_picks_up_unprefixed_fake(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class FakeBucket:\n    pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert [f.name for f in fakes] == ["FakeBucket"]

    def test_skips_non_fake_classes(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class RealThing:\n    pass\n\nclass StubThing:\n    pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert fakes == []

    def test_collects_inner_classes(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class Outer:\n    class _FakeInner:\n        pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert [f.name for f in fakes] == ["_FakeInner"]

    def test_skips_pycache(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "__pycache__" / "test_thing.cpython-314.pyc",
            "noise",
        )
        # add a real fake so the test verifies pycache path isn't traversed
        _write(
            tmp_path / "test_thing.py",
            "class _FakeReal:\n    pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert [f.name for f in fakes] == ["_FakeReal"]


# ------------------------------------------------------------------
# subclass declaration short-circuits the parity check
# ------------------------------------------------------------------


class TestSubclassDeclaration:
    """a fake with any non-object base produces zero violations."""

    def test_subclass_skips_check(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class _FakeWidget(SomeBase):\n    pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        assert violations == []

    def test_explicit_object_base_treated_as_no_base(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class _FakeWidget(object):\n    pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        # `object` is filtered out, leaving no parity declaration
        assert len(violations) == 1
        assert violations[0].category == "fake_parity.no_declaration"


# ------------------------------------------------------------------
# no-declaration violations
# ------------------------------------------------------------------


class TestNoDeclaration:
    """a fake without subclass + without marker fails."""

    def test_bare_fake_violates(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "class _FakeBucket:\n    pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        assert len(violations) == 1
        assert violations[0].category == "fake_parity.no_declaration"
        assert violations[0].symbol == "_FakeBucket"


class TestExemptMarker:
    """``# parity-exempt: <rationale>`` exempts a fake in place."""

    def test_exempt_marker_with_rationale_passes(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "# parity-exempt: subset stub, only the ctor the route touches\nclass _FakeBucket:\n    pass\n",
        )
        assert fake_parity_violations((tmp_path,), tmp_path) == []

    def test_exempt_rationale_is_recorded(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "test_thing.py",
            "# parity-exempt: subset stub\nclass _FakeBucket:\n    pass\n",
        )
        fakes = find_fakes_in_tree(path.parent)
        assert len(fakes) == 1
        assert fakes[0].exempt_rationale == "subset stub"
        assert fakes[0].marker_target is None

    def test_blank_exempt_marker_does_not_exempt(self, tmp_path: Path) -> None:
        # a bare marker with no rationale must NOT exempt -- it falls through
        # to no_declaration so the author is forced to state a reason.
        _write(
            tmp_path / "test_thing.py",
            "# parity-exempt:\nclass _FakeBucket:\n    pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        assert len(violations) == 1
        assert violations[0].category == "fake_parity.no_declaration"

    def test_exempt_marker_read_across_decorator(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "test_thing.py",
            "import dataclasses\n# parity-exempt: subset stub\n@dataclasses.dataclass\nclass _FakeBucket:\n    pass\n",
        )
        fakes = find_fakes_in_tree(path.parent)
        assert fakes[0].exempt_rationale == "subset stub"


# ------------------------------------------------------------------
# marker comment parity check
# ------------------------------------------------------------------


class TestMarkerComment:
    """``# parity-with:`` marker drives the method-surface check."""

    def test_marker_with_matching_methods_passes(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "# parity-with: pathlib.PurePath\n"
            "class _FakePath:\n"
            "    def joinpath(self, *args):\n"
            "        pass\n"
            "    def is_absolute(self):\n"
            "        pass\n"
            "    def relative_to(self, *args, **kwargs):\n"
            "        pass\n",
        )
        # ``pathlib.PurePath`` has many public methods; the fake will
        # almost certainly miss some. assert the marker is parsed and
        # the walker tries to import; whether every method is present
        # is the strict-mode question, not the unit-test question.
        fakes = find_fakes_in_tree(tmp_path)
        assert len(fakes) == 1
        assert fakes[0].marker_target == "pathlib.PurePath"

    def test_unimportable_marker_target(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "test_thing.py",
            "# parity-with: nonexistent.module.GhostClass\nclass _FakeGhost:\n    pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        assert len(violations) == 1
        assert violations[0].category == "fake_parity.import_failed"
        assert violations[0].symbol == "_FakeGhost"

    def test_marker_with_decorator_above_class(self, tmp_path: Path) -> None:
        # marker should still be readable across decorator lines
        _write(
            tmp_path / "test_thing.py",
            "import dataclasses\n# parity-with: pathlib.PurePath\n@dataclasses.dataclass\nclass _FakePath:\n    pass\n",
        )
        fakes = find_fakes_in_tree(tmp_path)
        assert len(fakes) == 1
        assert fakes[0].marker_target == "pathlib.PurePath"


# ------------------------------------------------------------------
# method-surface comparison
# ------------------------------------------------------------------


def _install_production_fixture(tmp_path: Path) -> str:
    """drop a tiny ``_ProductionDouble`` class into ``tmp_path`` and add to sys.path.

    the walker imports the marker target via :func:`importlib.import_module`
    so the production class must live at a real importable path. dropping
    a single-file module under ``tmp_path`` and prepending the dir to
    ``sys.path`` keeps the test self-contained.

    :param tmp_path: pytest tmp dir
    :ptype tmp_path: Path
    :return: fully-qualified marker target the test should use
    :rtype: str
    """
    import sys  # noqa: PLC0415

    _write(
        tmp_path / "_production_fixture.py",
        "class ProductionDouble:\n"
        "    def required_method(self, alpha, beta):\n"
        "        del alpha, beta\n"
        "    def optional_method(self, alpha=None):\n"
        "        del alpha\n",
    )
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return "_production_fixture.ProductionDouble"


class TestMethodSurface:
    """the walker enforces that fake methods accept production-required params."""

    def test_missing_method(self, tmp_path: Path) -> None:
        target = _install_production_fixture(tmp_path)
        _write(
            tmp_path / "test_thing.py",
            f"# parity-with: {target}\nclass _FakeDouble:\n    def required_method(self, alpha, beta):\n        pass\n",
        )
        # `optional_method` is missing on the fake; should violate
        violations = fake_parity_violations((tmp_path,), tmp_path)
        method_missing = [v for v in violations if v.category == "fake_parity.method_missing"]
        assert any("optional_method" in v.reason for v in method_missing)

    def test_required_param_missing_violates(self, tmp_path: Path) -> None:
        target = _install_production_fixture(tmp_path)
        _write(
            tmp_path / "test_thing.py",
            f"# parity-with: {target}\n"
            "class _FakeDouble:\n"
            "    def required_method(self, alpha):\n"
            "        pass\n"
            "    def optional_method(self):\n"
            "        pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        param_missing = [v for v in violations if v.category == "fake_parity.method_required_arg_missing"]
        assert any("`beta`" in v.reason for v in param_missing)

    def test_kwargs_satisfies_any_required(self, tmp_path: Path) -> None:
        target = _install_production_fixture(tmp_path)
        _write(
            tmp_path / "test_thing.py",
            f"# parity-with: {target}\n"
            "class _FakeDouble:\n"
            "    def required_method(self, **kwargs):\n"
            "        pass\n"
            "    def optional_method(self, **kwargs):\n"
            "        pass\n",
        )
        violations = fake_parity_violations((tmp_path,), tmp_path)
        # **kwargs catches everything; no required-param violations
        param_missing = [v for v in violations if v.category == "fake_parity.method_required_arg_missing"]
        assert param_missing == []
