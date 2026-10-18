"""what the git backend's unit suite and its live suite share: the test layout and a bare origin."""

from __future__ import annotations

import subprocess
from pathlib import Path

from threetears.core.backends.git import JsonCodec, TableLayout, TemplateLayout


def character_layout() -> TemplateLayout:
    """a character-shaped layout: keepers per tile, one recipe, notes without a version, numbered rejects.

    :return: the layout
    :rtype: TemplateLayout
    """
    codec = JsonCodec()
    return TemplateLayout(
        [
            TableLayout("keepers", "{character}/keepers/{batch}/{tile}.json", ("character", "batch", "tile"), codec),
            TableLayout("recipes", "{character}/recipe.json", ("character",), codec),
            TableLayout("notes", "{character}/notes.json", ("character",), codec, version_column=None),
            TableLayout(
                "rejects",
                "{character}/rejects/{batch}/{tile}-{n}.json",
                ("character", "batch", "tile", "n"),
                codec,
                pk_types={"n": int},
            ),
        ]
    )


def bare_origin(tmp_path: Path) -> Path:
    """make an empty bare repository to push to.

    :param tmp_path: the folder to make it in
    :ptype tmp_path: Path
    :return: the bare repository's path
    :rtype: Path
    """
    path = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(path)], check=True)
    return path
