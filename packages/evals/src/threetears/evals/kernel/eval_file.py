"""Where an eval file's relative paths are read from: beside the file, while ``python -m threetears.evals`` loads it.

An eval file names its cases, its recordings and its labels by paths relative to itself, as a TOML file does, so it
runs the same from any working directory. :func:`loading_eval_file` holds the loading file's directory for as long as
the file executes, and :func:`beside_eval_file` reads a path against it: ``Eval``'s cases, ``Model``'s and
``World``'s ``recording=``, ``Judge``'s ``labels=`` and a ``Log``'s file each go through it when they are built.
Outside a load, a relative path is the working directory's, as anywhere else in Python, and an absolute path is
always taken as given.

A context variable rather than a change of directory: nothing else the process reads moves under it (the store and
the report stay under the working directory), and two loads in two tasks never see each other's directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

#: The directory of the eval file being loaded, or ``None`` outside a load.
_LOADING_DIR: ContextVar[Path | None] = ContextVar("threetears_evals_loading_dir", default=None)


@contextmanager
def loading_eval_file(path: Path) -> Iterator[None]:
    """Read every relative path built inside the block against ``path``'s directory.

    :param path: The eval file being loaded.
    :ptype path: Path
    :yield: Nothing; the directory is unset when the block ends, however it ends.
    :rtype: Iterator[None]
    """
    token = _LOADING_DIR.set(path.parent)
    try:
        yield
    finally:
        _LOADING_DIR.reset(token)


def beside_eval_file(path: str | os.PathLike[str]) -> Path:
    """A path as the eval file being loaded names it: relative to that file, else to the working directory.

    :param path: The path as written.
    :ptype path: str | os.PathLike[str]
    :return: An absolute path unchanged; a relative one under the loading file's directory, or as given outside a load.
    :rtype: Path
    """
    given = Path(path)
    base = _LOADING_DIR.get()
    return given if base is None or given.is_absolute() else base / given


__all__ = ["beside_eval_file", "loading_eval_file"]
