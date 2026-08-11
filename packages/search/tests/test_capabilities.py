"""Capability declarations are queryable, consistent, and registered (SR-B4).

The point of a declaration is that a consumer branches *before* sending. So
the pins are: the answer is available without constructing a provider, it
cannot contradict itself, and it covers the namespaced vocabulary as well as
the well-known one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.search.adapters.searxng import SEARXNG_CAPABILITIES, SEARXNG_PARAM_ENGINES, SEARXNG_PROVIDER
from threetears.search.contracts import (
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_LANGUAGE,
    CRITERION_TIME_RANGE,
    ProviderCapabilities,
    get_capabilities,
    list_capabilities,
    register_capabilities,
)


def test_importing_an_adapter_registers_its_declaration() -> None:
    """A consumer queries the shape without a base URL or a transport."""
    assert get_capabilities(SEARXNG_PROVIDER) == SEARXNG_CAPABILITIES
    assert SEARXNG_PROVIDER in list_capabilities()


def test_an_unregistered_provider_answers_none_rather_than_empty() -> None:
    """'no adapter imported' and 'the provider can do nothing' are different."""
    assert get_capabilities("no-such-provider") is None


def test_disposition_for_answers_every_vocabulary() -> None:
    """Well-known, namespaced, and unrecognised keys all get an answer."""
    assert SEARXNG_CAPABILITIES.disposition_for(CRITERION_LANGUAGE) == "pushdown"
    assert SEARXNG_CAPABILITIES.disposition_for(CRITERION_DOMAINS_INCLUDE) == "local"
    assert SEARXNG_CAPABILITIES.disposition_for(CRITERION_TIME_RANGE) == "unsatisfied"
    assert SEARXNG_CAPABILITIES.disposition_for(SEARXNG_PARAM_ENGINES) == "pushdown"
    assert SEARXNG_CAPABILITIES.disposition_for("tavily:search-depth") == "ignored-unknown"


def test_a_contradictory_declaration_is_refused() -> None:
    """A criterion has exactly one disposition, so a clash cannot be silent."""
    with pytest.raises(ValidationError, match="more than one"):
        ProviderCapabilities(
            provider="contradictory",
            pushdown_criteria=(CRITERION_LANGUAGE,),
            unsatisfiable_criteria=(CRITERION_LANGUAGE,),
        )


def test_a_repeated_key_within_one_bucket_is_fine() -> None:
    """Only cross-bucket clashes are contradictions; a duplicate is not."""
    declared = ProviderCapabilities(provider="repetitive", pushdown_criteria=(CRITERION_LANGUAGE, CRITERION_LANGUAGE))
    assert declared.disposition_for(CRITERION_LANGUAGE) == "pushdown"


def test_registration_replaces_wholesale_and_the_registry_copy_is_inert() -> None:
    """A registry read cannot mutate the registry."""
    original = get_capabilities(SEARXNG_PROVIDER)
    assert original is not None
    try:
        register_capabilities(ProviderCapabilities(provider=SEARXNG_PROVIDER))
        assert get_capabilities(SEARXNG_PROVIDER) == ProviderCapabilities(provider=SEARXNG_PROVIDER)
        snapshot = list_capabilities()
        snapshot.pop(SEARXNG_PROVIDER)
        assert get_capabilities(SEARXNG_PROVIDER) is not None
    finally:
        register_capabilities(original)
    assert get_capabilities(SEARXNG_PROVIDER) == SEARXNG_CAPABILITIES


def test_searxng_declares_what_searxng_can_actually_do() -> None:
    """The declaration is the provider's API, not an aspiration.

    SearXNG has no domain allow-list and no absolute date filter, and both
    facts are load-bearing for a consumer choosing between providers.
    """
    assert CRITERION_DOMAINS_INCLUDE not in SEARXNG_CAPABILITIES.pushdown_criteria
    assert CRITERION_TIME_RANGE in SEARXNG_CAPABILITIES.unsatisfiable_criteria
    assert SEARXNG_CAPABILITIES.relative_time_ranges == ("day", "week", "month", "year")
    assert SEARXNG_CAPABILITIES.pricing_model == "free-self-hosted"
