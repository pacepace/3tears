"""module-level validator callable used by ``test_validator_rejection.py``.

the validator dispatcher resolves validators by dotted import path, so
the test's strict rejector must live at an importable address. this
module is test-only and documents that constraint.
"""

from __future__ import annotations


def reject_any_audience_units(relative_path: str, content: bytes) -> None:
    """
    reject any audience_settings.yaml whose bytes reference audience_units.

    the heuristic is intentionally blunt: "audience_units" appearing as
    a top-level key (anywhere in the payload) fails validation. the
    fixture audience_settings.yaml carries this key so the test's write
    is guaranteed to be rejected.

    :param relative_path: workspace-relative path being validated
    :ptype relative_path: str
    :param content: bytes about to land on disk
    :ptype content: bytes
    :return: None on pass
    :rtype: None
    :raises ValueError: when the content contains ``audience_units:``
    """
    del relative_path
    if b"audience_units:" in content:
        raise ValueError(
            "strict test validator rejects any audience_units: payload"
        )
