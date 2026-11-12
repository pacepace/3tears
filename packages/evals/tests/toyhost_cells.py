"""The toy campaign's cells and arms, addressed by chunk width: read off the assembled bundle, never pasted.

Harness support for the suites that read the toy host's bundle, not part of the toy host itself. A cell ref is two
content hashes, and a literal would stop matching the moment either derivation moved under it, so each helper reads the
coordinate off the bundle the test assembled.
"""

from __future__ import annotations

from typing import Any

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS
from threetears.evals.analysis import CampaignBundle
from threetears.evals.analysis.cells import cell_ref, variant_of_cell_ref
from threetears.evals.analysis.references import cell_aliases

__all__ = ["alias_at", "arm_at", "cell_at"]


def cell_at(bundle: Any, level: int) -> str:
    """The ``cell_ref`` of the arm run at one chunk width, read off the assembled bundle.

    Joined through the arm's run, because the run summary is where the width is recorded and the
    cell is where the run was pooled.

    :param bundle: The assembled context, or a campaign bundle of one aspect.
    :ptype bundle: Any
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The cell's ``<variant_key>:<apparatus_class_id>``.
    :rtype: str
    """
    bundle = bundle.aspects[0] if isinstance(bundle, CampaignBundle) else bundle
    (run_id,) = [summary.run_id for summary in bundle.run_summaries if summary.config[TOYHOST_AXIS] == str(level)]
    (cell,) = [cell for cell in bundle.cell_measures if run_id in cell.run_ids]
    return cell_ref(cell.variant_key, cell.apparatus_class_id)


def alias_at(bundle: Any, level: int) -> str:
    """The alias the writer names the cell at one chunk width by.

    :param bundle: The assembled context.
    :ptype bundle: Any
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The cell's alias, which stands for :func:`cell_at`'s ``cell_ref``.
    :rtype: str
    """
    aspect = bundle.aspects[0] if isinstance(bundle, CampaignBundle) else bundle
    (alias,) = [alias for alias, ref in cell_aliases(aspect.cell_measures).items() if ref == cell_at(aspect, level)]
    return alias


def arm_at(bundle: CampaignBundle, level: int) -> str:
    """The label the campaign bundle names the arm at one chunk width by.

    :param bundle: The campaign bundle, of one aspect.
    :ptype bundle: CampaignBundle
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The arm's label.
    :rtype: str
    """
    variant = variant_of_cell_ref(cell_at(bundle, level))
    assert variant is not None
    return bundle.label_of(variant)
