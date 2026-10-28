"""Which build of the package is installed: its version and, where its install recorded one, the git commit it came from.

Both are read from the installed distribution's metadata, never from a shell call or a checkout at runtime:

- **The version** is the distribution's (``importlib.metadata``), what ``threetears.evals.__version__`` states.
- **The commit** is what the installer recorded when it installed the package from a git URL: the
  ``vcs_info.commit_id`` of the distribution's ``direct_url.json`` (PEP 610), which pip and uv write for an install
  from ``git+https://...@<rev>``. A wheel from an index, an editable install of a checkout and an install from a local
  directory record none, and the commit is then ``None``: not known, never a guess read off whatever checkout happens
  to sit beside the code.

A source tree on the path with no distribution metadata at all (the installability check, a vendored checkout) has
neither, and says so with ``None``.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution

__all__ = ["package_rev", "package_version"]

#: The distribution this package is installed as.
_DISTRIBUTION = "3tears-evals"

#: The metadata file an installer writes for a package installed from a URL (PEP 610).
_DIRECT_URL = "direct_url.json"


@cache
def package_version() -> str | None:
    """The installed distribution's version, or None when this build carries no distribution metadata.

    :return: The version, as its metadata states it; None with no metadata.
    :rtype: str | None
    """
    try:
        return distribution(_DISTRIBUTION).version
    except PackageNotFoundError:
        # NOSILENT: no metadata is a build fact stated as None, which every reader prints as not recorded
        return None


@cache
def package_rev() -> str | None:
    """The git commit the installed distribution was built from, as its installer recorded it, or None.

    :return: ``vcs_info.commit_id`` of the distribution's ``direct_url.json`` when it was installed from git; None for
        an install that recorded no commit (a wheel from an index, an editable or local-directory install) or a build
        with no distribution metadata.
    :rtype: str | None
    """
    try:
        recorded = distribution(_DISTRIBUTION).read_text(_DIRECT_URL)
    except PackageNotFoundError:
        # NOSILENT: no metadata is a build fact stated as None, which every reader prints as not recorded
        return None
    if recorded is None:
        return None
    try:
        direct_url = json.loads(recorded)
    except json.JSONDecodeError:
        # NOSILENT: an installer's record that does not parse names no commit; the stamp says not recorded
        return None
    vcs = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
    return commit if isinstance(commit, str) and commit else None
