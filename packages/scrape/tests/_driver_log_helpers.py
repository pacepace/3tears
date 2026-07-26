"""One place that decides which log records belong to which driver.

Extracted because the decision is where a bug lived: filtering with
``r.name.endswith("document")`` also matches ``multi_document`` -- the wrapping driver that
forwards a human's solve to the inner document driver -- so the loose form was wrong precisely
in the case the filter exists for.

Sharing it is what makes the rule guardable. EVERY driver suite asserts through this function,
and ``test_the_filter_does_not_confuse_a_wrapper_for_its_inner_driver`` drives it directly, so
loosening the match fails there rather than quietly weakening the assertions elsewhere.

No count of suites or call sites here, deliberately: the sentence above this one was itself
rewritten to fix a false claim, and that rewrite introduced a "four suites" figure its own
commit made wrong. A number in prose is a fact that has to be maintained, and this file has
now proved twice that it will not be.

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
