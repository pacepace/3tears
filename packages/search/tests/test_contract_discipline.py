"""Contract-level behavioral pins from search-spec.md §2/§3.1.

The open-vocabulary rules, the score-comparability rule, spend arithmetic,
zero-results-is-success, transport satisfiability by shape, import
cleanliness, and the no-layer-names naming rule.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import JsonValue, ValidationError

import threetears.search.contracts as contracts
from threetears.search.contracts import (
    EGRESS_DIRECT,
    WELL_KNOWN_CRITERIA,
    CandidateSet,
    Criterion,
    FetchTransport,
    Provenance,
    ScoreEntry,
    SearchTransport,
    Spend,
    TransportResponse,
)
from _search_instances import PROVENANCE


class FakeSearchTransport(SearchTransport):  # parity-with: threetears.search.contracts.SearchTransport
    """Minimal in-memory transport satisfying the injected seam by shape."""

    def __init__(self, body: bytes = b"{}") -> None:
        """Hold canned response bytes.

        :param body: bytes every request answers with
        :ptype body: bytes
        """
        self._body = body

    @property
    def egress_name(self) -> str:
        """Report the configured exit's name.

        :return: always ``direct`` for this fake
        :rtype: str
        """
        return EGRESS_DIRECT

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Answer every request with the canned body.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL
        :ptype url: str
        :param headers: ignored
        :ptype headers: Mapping[str, str] | None
        :param params: ignored
        :ptype params: Mapping[str, str] | None
        :param json_body: ignored
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :return: a canned successful exchange
        :rtype: TransportResponse
        """
        return TransportResponse(
            status_code=200,
            body=self._body,
            final_url=url,
            egress=self.egress_name,
            elapsed_seconds=0.001,
            attempts=1,
        )


async def test_transport_is_satisfiable_by_shape() -> None:
    """SR-N1/P9: the seam is structural -- a host adapter needs no base class."""
    transport = FakeSearchTransport()
    assert isinstance(transport, SearchTransport)
    response = await transport.request("GET", "https://searx.example/search", params={"q": "capybara"})
    assert response.status_code == 200
    assert response.egress == EGRESS_DIRECT
    assert response.attempts == 1


class FakeFetchTransport(FetchTransport):  # parity-with: threetears.search.contracts.FetchTransport
    """Minimal in-memory implementer of Extract's byte-capped fetch seam."""

    @property
    def egress_name(self) -> str:
        """Report the configured exit's name.

        :return: always ``direct`` for this fake
        :rtype: str
        """
        return EGRESS_DIRECT

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Answer every fetch with an empty capped body.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL
        :ptype url: str
        :param headers: ignored
        :ptype headers: Mapping[str, str] | None
        :param max_bytes: the per-call cap the seam requires
        :ptype max_bytes: int
        :param allowed_content_types: ignored
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :return: a canned successful exchange, body under the cap
        :rtype: TransportResponse
        """
        return TransportResponse(
            status_code=200,
            body=b"x" * min(4, max_bytes),
            final_url=url,
            egress=self.egress_name,
            elapsed_seconds=0.001,
            attempts=1,
        )


async def test_fetch_transport_is_a_second_seam_not_a_widening() -> None:
    """the Gate A ruling: Extract's byte-capped read is its own protocol, so a
    search-only transport stays conformant forever -- the two shapes are
    independently satisfiable and neither implies the other."""
    fetcher = FakeFetchTransport()
    assert isinstance(fetcher, FetchTransport)
    assert not isinstance(FakeSearchTransport(), FetchTransport), (
        "a Phase-1 SearchTransport implementer must never be retroactively non-conformant, which is "
        "exactly what folding fetch() into SearchTransport would have done"
    )
    response = await fetcher.fetch("GET", "https://example.org/page", max_bytes=4)
    assert len(response.body) <= 4


def test_unknown_plain_criterion_key_is_rejected() -> None:
    """a plain key outside the well-known set must be namespaced instead."""
    with pytest.raises(ValidationError, match="namespaced"):
        Criterion(key="search-depth", value="advanced")


def test_namespaced_criterion_key_is_open() -> None:
    """foreign vocabulary rides '<namespace>:<name>' -- never a closed enum."""
    criterion = Criterion.namespaced("tavily", "search-depth", "advanced")
    assert criterion.key == "tavily:search-depth"
    assert Criterion(key="tavily:search-depth", value="advanced") == criterion


def test_well_known_constructors_use_well_known_keys() -> None:
    """every typed constructor lands inside the declared well-known set."""
    built = [
        Criterion.time_range(start=PROVENANCE.retrieved_at),
        Criterion.domains_include(["example.org"]),
        Criterion.domains_exclude(["example.net"]),
        Criterion.language("en"),
        Criterion.carrier("image"),
        Criterion.min_resolution(width=800, height=600),
        Criterion.rights_class("public-domain"),
        Criterion.max_results(5),
    ]
    assert {criterion.key for criterion in built} == WELL_KNOWN_CRITERIA


def test_degenerate_constructor_arguments_are_refused() -> None:
    """constructors validate what a raw dict could not."""
    with pytest.raises(ValueError, match="at least one"):
        Criterion.time_range()
    with pytest.raises(ValueError, match="positive"):
        Criterion.max_results(0)


def test_time_range_rejects_naive_datetimes() -> None:
    """the Provenance stance applied to the one constructor that lacked it:
    a naive bound has unknown-timezone semantics and is refused at the
    border, never embedded (Gate A, 2026-08-10)."""
    with pytest.raises(ValueError, match="timezone-aware"):
        Criterion.time_range(start=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        Criterion.time_range(end=datetime(2026, 8, 1))


def test_time_range_normalizes_equal_instants_to_one_form() -> None:
    """criterion values participate in the canonical form (D26, SR-F1):
    12:00+00:00 and 14:00+02:00 are one instant and must be one criterion
    value, or one search splits into two replay keys."""
    utc_bound = Criterion.time_range(start=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    offset_bound = Criterion.time_range(start=datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2))))
    assert utc_bound == offset_bound
    assert utc_bound.value == {"start": "2026-01-01T12:00:00+00:00"}


def test_provider_native_scores_are_never_comparable() -> None:
    """SR-A4/D1: provider-native scores are non-comparable by construction."""
    score = ScoreEntry.provider_native(name="relevance", value=0.9, scale="unit-interval", provider_instance="tavily")
    assert score.comparable is False


def test_zero_results_is_a_success_value() -> None:
    """SR-J2: an empty candidate set constructs and round-trips as success."""
    empty = CandidateSet()
    assert empty.candidates == ()
    assert CandidateSet.model_validate_json(empty.model_dump_json()) == empty


def test_spend_sums_dimension_wise() -> None:
    """spend aggregation is add-per-dimension, money in one currency."""
    total = Spend(money=Decimal("0.10"), calls=1, provider_units=Decimal("2")) + Spend(
        money=Decimal("0.05"), calls=2, wall_clock_seconds=0.5, bytes_transferred=100
    )
    assert total == Spend(
        money=Decimal("0.15"),
        calls=3,
        provider_units=Decimal("2"),
        wall_clock_seconds=0.5,
        bytes_transferred=100,
    )


def test_spend_refuses_cross_currency_money() -> None:
    """summing money across currencies would fabricate a bill."""
    with pytest.raises(ValueError, match="currencies"):
        _ = Spend(money=Decimal("1")) + Spend(money=Decimal("1"), currency="EUR")


def test_spend_refuses_cross_provider_units() -> None:
    """SR-E4: one provider's credits and another's are not one quantity.

    The same fabrication the currency guard prevents, in the dimension that
    had no guard: a cost surface summing Tavily credits into another
    provider's units would display a number nothing spent.
    """
    with pytest.raises(ValueError, match="provider units"):
        _ = Spend(provider_units=Decimal("2"), provider_unit="tavily:credits") + Spend(
            provider_units=Decimal("1"), provider_unit="brave:requests"
        )


def test_spend_sums_matching_provider_units() -> None:
    """two calls to the same provider aggregate, label intact."""
    total = Spend(provider_units=Decimal("2"), provider_unit="tavily:credits", calls=1) + Spend(
        provider_units=Decimal("1"), provider_unit="tavily:credits", calls=1
    )
    assert total.provider_units == Decimal("3")
    assert total.provider_unit == "tavily:credits"
    assert total.calls == 2


def test_a_zero_unit_spend_adopts_the_other_side_s_label() -> None:
    """Spend() is the aggregation identity, in this dimension too.

    A free or per-request call carries no unit, and summing one into a
    weighted call's spend must neither refuse nor drop the label -- the
    total still counts Tavily credits, and a reader with no label cannot
    tell what the number is.
    """
    assert (Spend(calls=1) + Spend(provider_units=Decimal("2"), provider_unit="tavily:credits")).provider_unit == (
        "tavily:credits"
    )
    assert (Spend(provider_units=Decimal("2"), provider_unit="tavily:credits") + Spend(calls=1)).provider_unit == (
        "tavily:credits"
    )


def test_naive_retrieval_time_is_rejected() -> None:
    """provenance timestamps are timezone-aware by construction."""
    data = PROVENANCE.model_dump()
    data["retrieved_at"] = PROVENANCE.retrieved_at.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        Provenance.model_validate(data)


def test_no_layer_name_is_a_type_name() -> None:
    """search-spec.md §2: Adapter/Call/Aggregate/Extract/Select/Bind are module
    vocabulary; no exported name carries one."""
    layers = ("Adapter", "Call", "Aggregate", "Extract", "Select", "Bind")
    offenders = [name for name in contracts.__all__ for layer in layers if layer in name]
    assert offenders == []


def test_contracts_import_is_clean() -> None:
    """importing the contracts pulls no core, agent, langchain, NATS, httpx,
    or observe module (search-spec.md §2 -- the leaf floor)."""
    probe = (
        "import sys; import threetears.search.contracts; "
        "banned = [m for m in sys.modules if m.startswith("
        "('threetears.core', 'threetears.agent', 'langchain', 'nats', 'httpx', 'trafilatura', "
        "'threetears.observe'))]; "
        "sys.exit(repr(banned) if banned else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"contracts import pulled banned modules: {result.stderr}"
