"""One place that decides which log records belong to which driver.

Extracted because the decision is where a bug lived: filtering with
``r.name.endswith("document")`` also matches ``multi_document`` -- the wrapping driver that
forwards a human's solve to the inner document driver -- so the loose form was wrong precisely
in the case the filter exists for.

Sharing it is what makes the rule guardable. Every driver suite that asserts on driver log
records goes through this function -- the others never touch ``caplog`` -- and
``test_the_filter_does_not_confuse_a_wrapper_for_its_inner_driver`` drives it directly, so
loosening the match fails there rather than quietly weakening the assertions elsewhere.

No count of suites or call sites here, deliberately. This docstring carried "four suites",
which was accurate when written and wrong one commit later, when a different file gained an
import and nothing brought anyone back here. That is the whole hazard: a tally is true at the
moment of writing and rots without being touched, so neither re-reading the diff nor grepping
the prose catches it.

The previous attempt at that guard asserted ``"...multi_document".endswith("document")``,
which is a property of :class:`str`: it held whether production was exact or loose, and so
guarded nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["driver_warnings"]


def driver_warnings(caplog: Any, module: str) -> list[Any]:
    """Records emitted by EXACTLY ``threetears.scrape.drivers.<module>``.

    :param caplog: pytest's ``caplog`` fixture
    :ptype caplog: Any
    :param module: driver module name, e.g. ``"document"``
    :ptype module: str
    :return: matching records, in emission order
    :rtype: list[Any]
    """
    return [r for r in caplog.records if r.name == f"threetears.scrape.drivers.{module}"]
