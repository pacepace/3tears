"""tests for public-surface fingerprinting.

the contract under test: the fingerprint moves exactly when the
*public surface* moves -- public defs/classes/method signatures,
public annotated class attributes, ``__all__`` -- and stays put for
implementation-only edits (bodies, docstrings, private names).
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.integration_skill import compute_surface_fingerprint


def _write_pkg(tmp_path: Path, module_source: str) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text('__all__ = ["greet"]\n')
    (pkg / "mod.py").write_text(module_source)
    return pkg


_BASE = '''
def greet(name: str) -> str:
    """say hello."""
    return f"hello {name}"


class Widget:
    score: int = 0

    def spin(self, times: int = 1) -> None:
        pass

    def _internal(self) -> None:
        pass


def _helper() -> None:
    pass
'''


class TestSurfaceFingerprint:
    def test_body_and_docstring_edits_are_invisible(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        edited = _BASE.replace('"""say hello."""', '"""say hello loudly."""').replace(
            'return f"hello {name}"', 'return f"HELLO {name}!"'
        )
        _write_pkg(tmp_path, edited)
        assert compute_surface_fingerprint(pkg) == before

    def test_private_additions_are_invisible(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        _write_pkg(tmp_path, _BASE + "\n\ndef _another_helper() -> int:\n    return 1\n")
        assert compute_surface_fingerprint(pkg) == before

    def test_new_public_function_moves_the_fingerprint(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        _write_pkg(tmp_path, _BASE + "\n\ndef shout(name: str) -> str:\n    return name.upper()\n")
        assert compute_surface_fingerprint(pkg) != before

    def test_signature_change_moves_the_fingerprint(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        _write_pkg(
            tmp_path,
            _BASE.replace("def greet(name: str) -> str:", "def greet(name: str, *, loud: bool = False) -> str:"),
        )
        assert compute_surface_fingerprint(pkg) != before

    def test_public_method_signature_change_moves_the_fingerprint(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        _write_pkg(tmp_path, _BASE.replace("def spin(self, times: int = 1)", "def spin(self, times: int = 2)"))
        assert compute_surface_fingerprint(pkg) != before

    def test_annotated_class_attribute_change_moves_the_fingerprint(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        _write_pkg(tmp_path, _BASE.replace("score: int = 0", "score: float = 0.0"))
        assert compute_surface_fingerprint(pkg) != before

    def test_all_change_moves_the_fingerprint(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        before = compute_surface_fingerprint(pkg)
        (pkg / "__init__.py").write_text('__all__ = ["greet", "Widget"]\n')
        assert compute_surface_fingerprint(pkg) != before

    def test_fingerprint_is_deterministic(self, tmp_path: Path) -> None:
        pkg = _write_pkg(tmp_path, _BASE)
        assert compute_surface_fingerprint(pkg) == compute_surface_fingerprint(pkg)
