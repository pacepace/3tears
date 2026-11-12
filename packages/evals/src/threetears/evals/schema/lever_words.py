"""A lever's id read as words: the one rule, so no caller keeps a copy.

A lever is named by the reader name stored for it (``Sweepable.reader_name``, a campaign's ``SweptAxis.reader_name``).
Where none is stored anywhere, its id is read as words, by :func:`lever_id_words` alone: the name a page prints for a
lever no declaration names (:func:`~threetears.evals.analysis.style.lever_reader_name`, which a campaign's derived arm
label spells each lever by), and the name the durable v16 to v17 step gives a stored axis that has none
(:func:`~threetears.evals.schema.campaign_upgrade.upgrade_campaign_v16`, which first drops the quick path's kind
namespace, a part of the id no author wrote).
"""

from __future__ import annotations

import re

__all__ = ["lever_id_words"]


def lever_id_words(lever: str) -> str:
    """A lever's id read as words: its whitespace collapsed, and each run of ``_`` and ``.`` a space
    (``max_output_tokens`` to ``max output tokens``, ``extractor.prompt_style`` to ``extractor prompt style``); the id
    itself where nothing else is left.

    :param lever: The lever, by id.
    :ptype lever: str
    :return: Its id in words.
    :rtype: str
    """
    return re.sub(r"[_.]+", " ", " ".join(lever.split())).strip() or lever
