"""pytest-friendly orchestration for the import-floor gate.

One :func:`run_import_floor_enforcement` entry point so each consumer's shell is a few lines. No
exemptions file: a module that loads more than the floor is either moved behind an extra or the
floor itself changes, in the package's dependencies and its config together.
"""

from __future__ import annotations

import pytest

from threetears.enforcement.import_floor.config import ImportFloorConfig
from threetears.enforcement.import_floor.walkers import ProbeError, modules_loaded_by_importing, outside_floor

__all__ = ["run_import_floor_enforcement"]


def run_import_floor_enforcement(config: ImportFloorConfig, module: str) -> None:
    """fail when importing ``module`` on the base install loads anything outside its floor.

    Also fails when the probe did not load ``module`` itself: an allowlist over an empty set passes
    for the wrong reason.

    :param config: the package's floor
    :ptype config: ImportFloorConfig
    :param module: one of ``config.base_modules``
    :ptype module: str
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: when the import fails, did not happen, or loads past the floor
    :raises ValueError: when ``module`` is not one of the config's base modules
    """
    if module not in config.base_modules:
        raise ValueError(f"import floor: {module} is not one of the floor's base modules")
    try:
        loaded = modules_loaded_by_importing(module, refused=config.refused_roots)
    except ProbeError as exc:
        pytest.fail(
            f"{exc}\nA base module imports what the base install leaves out without the guard naming its extra."
        )
    if module not in loaded:
        pytest.fail(f"import floor: the probe did not load {module}, so it measured nothing")
    if offenders := outside_floor(config, loaded):
        pytest.fail(
            f"importing {module} on the base install loaded modules outside its floor: {offenders}. Either "
            f"the import belongs behind TYPE_CHECKING or behind an extra's guard at the module's top, or the "
            f"floor itself is changing -- in which case the package's dependencies and its ImportFloorConfig "
            f"move together"
        )
