"""The answers conformance suite, run against Brave's own recorded streams (AR-I2).

The cited case is the live capture, unchanged; the uncited and malformed
cases are synthetic streams in its framing, since no capture exhibits them.
"""

from __future__ import annotations

from threetears.search.adapters.brave_answers import BraveAnswersAdapter
from threetears.search.contracts import AnswerProvider, Criterion, StreamingTransport
from threetears.search.testing import AnswerConformanceCase, AnswerConformanceSuite
from packages.search.tests.brave_answers_payloads import (
    INVALID_TOKEN_BODY,
    RATE_LIMITED_BODY,
    SINGLE,
    stream,
    usage_tag,
)


def _brave(transport: StreamingTransport) -> AnswerProvider:
    """A Brave adapter over the suite's transport.

    :param transport: the recorded transport
    :ptype transport: StreamingTransport
    :return: the provider under test
    :rtype: AnswerProvider
    """
    return BraveAnswersAdapter(api_key="conformance-key", transport=transport, provider_instance="brave-conformance")


class TestBraveAnswersConformance(AnswerConformanceSuite):
    case = AnswerConformanceCase(
        provider_factory=_brave,
        cited_stream=SINGLE,
        uncited_stream=stream("Capybaras are rodents.", usage_tag()),
        malformed_stream=stream("Capybaras are rodents.", "<usage>{}</usage>"),
        rate_limited=(429, RATE_LIMITED_BODY),
        auth_failed=(422, INVALID_TOKEN_BODY),
        pushdown_criterion=Criterion.language("en"),
        unsatisfiable_criterion=Criterion.domains_include(["example.org"]),
        pushdown_parameter="language",
    )
