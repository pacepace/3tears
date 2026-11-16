"""The browser-side helpers that ship with the package: ``pwa.js`` and ``safe-area.css``.

They are plain files for the app to serve beside its own scripts and styles, not something this
package serves: :func:`export_static` copies them into the app's static directory (at image build,
and before the app's own JavaScript tests, which import them by path).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Final

__all__ = ["STATIC_FILES", "export_static"]

#: the shipped files: an ES module of helpers, and the safe-area stylesheet
STATIC_FILES: Final[tuple[str, ...]] = ("pwa.js", "safe-area.css")


def _static_file(name: str) -> bytes:
    """One shipped file's bytes.

    :raises KeyError: ``name`` is not one of :data:`STATIC_FILES`
    """
    if name not in STATIC_FILES:
        raise KeyError(name)
    return (resources.files("threetears.pwa") / "static" / name).read_bytes()


def export_static(dest: Path) -> list[Path]:
    """Copy every shipped file into ``dest`` (created if missing), rewriting only what differs.

    :return: the files written, in :data:`STATIC_FILES` order
    :rtype: list[Path]
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in STATIC_FILES:
        body = _static_file(name)
        target = dest / name
        if not target.is_file() or target.read_bytes() != body:
            target.write_bytes(body)
            written.append(target)
    return written
