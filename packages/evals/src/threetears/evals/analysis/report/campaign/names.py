"""What a campaign's page calls an aspect: the one place a name is chosen, so every part names an aspect alike.

Today an aspect is called by its label, as declared. A name is declared, never derived: the fallback is always the
full label, never a shortening of it.
"""

from __future__ import annotations

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle

__all__ = ["aspect_name"]


def aspect_name(bundle: AnalysisContextBundle) -> str:
    """What the page calls the aspect ``bundle`` reads: its label, as declared.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: The aspect's name on the page.
    :rtype: str
    """
    return bundle.aspect
