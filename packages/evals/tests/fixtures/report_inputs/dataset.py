"""The report inputs beside this file, and the Kairo round 2 campaign bundle the r5 input was adapted from.

- **Inputs.** ``r1`` to ``r5`` each hold ``input.json``, a report input (:func:`report_input_of`), which
  ``build_inputs.py`` reproduces byte for byte from the kit's references.
- **The Kairo bundle.** The Kairo round 2 campaign bundle is committed gzipped (``kairo-round2-bundle.json.gz``,
  :func:`kairo_bundle`), so ``report_input(bundle)`` runs against the real stored campaign. It has no generator: it is
  real data. It is moved across an aspect bundle shape by a migration script that recomputes only what the shape
  changes from what the bundle holds, and refuses when any other field moves. At shape 65 that was
  ``migrate_bundle_shape_65.py kairo-round2-bundle.json.gz``, which moved eight rate contrasts' intervals (the pass
  rate, a goal check and two pass/fail judged dimensions, two arms each) onto their stored bounded intervals, every
  verdict holding, named each corrected comparison's interval basis, and moved the shape.

The plans the retired report planner stored beside these inputs, and its scoring dataset, left with it. The charts a
few of those plans stored are kept for the chart resolver's tests (``fixtures/plan_charts/``).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report_input import ReportInput

__all__ = ["HERE", "KAIRO_BUNDLE", "kairo_bundle", "report_input_of"]

HERE = Path(__file__).resolve().parent
#: The Kairo round 2 campaign bundle, as the writer read it, gzipped.
KAIRO_BUNDLE = HERE / "kairo-round2-bundle.json.gz"


def kairo_bundle() -> CampaignBundle:
    """The Kairo round 2 campaign bundle, read from its gzipped fixture."""
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        return CampaignBundle.from_dict(json.load(stream))


def report_input_of(name: str) -> ReportInput:
    """One fixture input (``r1`` to ``r5``)."""
    return ReportInput.model_validate_json((HERE / name / "input.json").read_text(encoding="utf-8"))
