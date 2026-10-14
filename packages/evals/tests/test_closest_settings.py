"""``closest_settings``: what a client sends for what it was asked, per provider family, table-tested.

The shared negotiation (``threetears.evals.negotiation``, which the shipped adapter and a host's own client both use)
never refuses a setting a model merely does not take (owner ruling, 2026-10-13): it sends the closest one the model
accepts and reports it. This holds the pure choice, row by row of the per-provider mapping table in the
design of record (``design-sent-settings.md`` §3): word to closest word, on the same side of the on/off line with
ties going lower; a budget to the word its share of the cap maps to on a model that takes words only, and a word to
that share of the cap on a model that takes budgets only; Anthropic's effort words over adaptive thinking with no
temperature beside any thinking; ``max`` and ``none`` per provider; the strict reply format sent unenforced where it
cannot be enforced; the output cap clamped to the model's own. Facts left unknown are sent as asked. The shares are
one constant, pinned here. What reaches the wire is ``test_models_adapter*.py``; what is learned from a refusal,
``test_negotiation_loop.py``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from threetears.evals.negotiation import (
    BUDGET_SHARES,
    EFFORT_LADDER,
    SettingFacts,
    WireSettings,
    closest_settings,
)
from threetears.evals.schema import REASONING_EFFORTS, ClientRequestSettings

#: The judge's asked settings (``JUDGE_REQUEST_SETTINGS``), as the adapter is asked them, with the judge temperature.
JUDGE = {"max_tokens": 10240, "reasoning_effort": "high", "strict_output": True, "temperature": 0.0}

#: The simulator's asked settings (``SIMULATOR_REQUEST_SETTINGS``).
SIMULATOR = {"max_tokens": 5120, "reasoning_effort": "low", "strict_output": True}

#: Facts of models the design names, as 3tears-models' capability fields would declare them.
SONNET_5 = {
    "accepts_temperature": False,
    "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"),
    "reasoning_budget": False,
    "adaptive_reasoning": True,
    "reasoning_disable": True,
    "structured_output": True,
}
HAIKU_4_5 = {"reasoning_efforts": (), "reasoning_budget": True, "adaptive_reasoning": False, "reasoning_disable": True}
ALWAYS_ON = {"reasoning_disable": False}
GPT_5 = {"reasoning_efforts": ("minimal", "low", "medium", "high")}
GPT_5_1 = {"reasoning_efforts": ("none", "low", "medium", "high")}
GPT_4O = {"reasoning_efforts": (), "structured_output": True}

#: Every row: (id, provider, asked, facts, sent). ``sent`` names every field of the result.
ROWS: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = [
    # --- unknown facts: everything as asked -------------------------------------------------------------------------
    ("anthropic-unknown-judge", "anthropic", JUDGE, {}, {**JUDGE, "temperature": None}),
    ("openai-unknown-judge", "openai", JUDGE, {}, JUDGE),
    ("openrouter-unknown-judge", "router", JUDGE, {}, JUDGE),
    # --- temperature ------------------------------------------------------------------------------------------------
    ("anthropic-temperature-alone", "anthropic", {"temperature": 0.2}, {}, {"temperature": 0.2}),
    ("anthropic-temperature-refused", "anthropic", {"temperature": 0.2}, {"accepts_temperature": False}, {}),
    (
        "anthropic-no-temperature-beside-a-budget",
        "anthropic",
        {"temperature": 0.2, "max_tokens": 4096, "reasoning_max_tokens": 2048},
        {},
        {"max_tokens": 4096, "reasoning_max_tokens": 2048},
    ),
    (
        "anthropic-no-temperature-beside-adaptive",
        "anthropic",
        {"temperature": 0.2, "reasoning_effort": "low"},
        {},
        {"reasoning_effort": "low"},
    ),
    (
        "anthropic-temperature-beside-effort-alone",
        "anthropic",
        {"temperature": 0.2, "reasoning_effort": "low"},
        {"adaptive_reasoning": False},
        {"temperature": 0.2, "reasoning_effort": "low"},
    ),
    (
        "anthropic-temperature-beside-thinking-off",
        "anthropic",
        {"temperature": 0.2, "reasoning_effort": "none"},
        {},
        {"temperature": 0.2, "reasoning_effort": "none"},
    ),
    ("openai-temperature-refused", "openai", {"temperature": 0.2}, {"accepts_temperature": False}, {}),
    ("openrouter-temperature-not-catalogued", "router", {"temperature": 0.2}, {"accepts_temperature": False}, {}),
    # --- Anthropic effort words -------------------------------------------------------------------------------------
    ("anthropic-judge-on-sonnet-5", "anthropic", JUDGE, SONNET_5, {**JUDGE, "temperature": None}),
    ("anthropic-minimal-is-low", "anthropic", {"reasoning_effort": "minimal"}, {}, {"reasoning_effort": "low"}),
    (
        "anthropic-simulator-on-sonnet-4-6",
        "anthropic",
        SIMULATOR,
        {"reasoning_efforts": ("low", "medium", "high", "max")},
        SIMULATOR,
    ),
    ("anthropic-none-is-thinking-off", "anthropic", {"reasoning_effort": "none"}, {}, {"reasoning_effort": "none"}),
    (
        "anthropic-none-on-always-on-is-lowest",
        "anthropic",
        {"reasoning_effort": "none"},
        ALWAYS_ON,
        {"reasoning_effort": "low"},
    ),
    (
        "anthropic-max-without-max-is-xhigh",
        "anthropic",
        {"reasoning_effort": "max"},
        {"reasoning_efforts": ("low", "medium", "high", "xhigh")},
        {"reasoning_effort": "xhigh"},
    ),
    (
        "anthropic-xhigh-tie-goes-lower",
        "anthropic",
        {"reasoning_effort": "xhigh"},
        {"reasoning_efforts": ("low", "medium", "high", "max")},
        {"reasoning_effort": "high"},
    ),
    # A budget-only model (Haiku 4.5): a word is its share of the cap, at least 1024 and below the cap.
    (
        "anthropic-judge-on-budget-only",
        "anthropic",
        JUDGE,
        HAIKU_4_5,
        {"max_tokens": 10240, "reasoning_max_tokens": 8192, "strict_output": True},
    ),
    (
        "anthropic-simulator-minimal-on-budget-only",
        "anthropic",
        {**SIMULATOR, "reasoning_effort": "minimal"},
        HAIKU_4_5,
        {"max_tokens": 5120, "reasoning_max_tokens": 1024, "strict_output": True},
    ),
    (
        "anthropic-max-on-budget-only",
        "anthropic",
        {"max_tokens": 10240, "reasoning_effort": "max"},
        HAIKU_4_5,
        {"max_tokens": 10240, "reasoning_max_tokens": 9728},
    ),
    (
        "anthropic-none-on-budget-only",
        "anthropic",
        {"max_tokens": 10240, "reasoning_effort": "none"},
        HAIKU_4_5,
        {"max_tokens": 10240, "reasoning_effort": "none"},
    ),
    (
        "anthropic-no-room-for-a-budget",
        "anthropic",
        {"max_tokens": 1024, "reasoning_effort": "high"},
        HAIKU_4_5,
        {"max_tokens": 1024},
    ),
    (
        "anthropic-takes-no-reasoning",
        "anthropic",
        {"reasoning_effort": "high"},
        {"reasoning_efforts": (), "reasoning_budget": False},
        {},
    ),
    (
        "anthropic-budget-only-with-no-cap-to-size-one",
        "anthropic",
        {"reasoning_effort": "high"},
        HAIKU_4_5,
        {},
    ),
    # --- Anthropic budgets ------------------------------------------------------------------------------------------
    (
        "anthropic-budget-as-asked",
        "anthropic",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        {},
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
    ),
    (
        "anthropic-budget-on-adaptive-only",
        "anthropic",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        SONNET_5,
        {"max_tokens": 10240, "reasoning_effort": "high"},
    ),
    (
        "anthropic-budget-under-a-clamped-cap",
        "anthropic",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        {"max_output_tokens": 8000},
        {"max_tokens": 8000, "reasoning_max_tokens": 7999},
    ),
    (
        "anthropic-budget-with-no-room-under-the-cap",
        "anthropic",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        {"max_output_tokens": 1024},
        {"max_tokens": 1024, "reasoning_effort": "high"},
    ),
    # --- OpenAI -----------------------------------------------------------------------------------------------------
    ("openai-max-is-xhigh", "openai", {"reasoning_effort": "max"}, {}, {"reasoning_effort": "xhigh"}),
    ("openai-max-without-xhigh-is-high", "openai", {"reasoning_effort": "max"}, GPT_5, {"reasoning_effort": "high"}),
    ("openai-none-on-gpt-5-is-minimal", "openai", {"reasoning_effort": "none"}, GPT_5, {"reasoning_effort": "minimal"}),
    ("openai-minimal-on-gpt-5-1-is-low", "openai", SIMULATOR | {"reasoning_effort": "minimal"}, GPT_5_1, SIMULATOR),
    ("openai-none-where-taken", "openai", {"reasoning_effort": "none"}, GPT_5_1, {"reasoning_effort": "none"}),
    ("openai-non-reasoning-model", "openai", JUDGE, GPT_4O, {**JUDGE, "reasoning_effort": None}),
    (
        "openai-refused-word-tie-goes-lower",
        "openai",
        {"reasoning_effort": "high"},
        {"refused_efforts": frozenset({"high"})},
        {"reasoning_effort": "medium"},
    ),
    (
        "openai-judge-budget-is-high",
        "openai",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        {},
        {"max_tokens": 10240, "reasoning_effort": "high"},
    ),
    (
        "openai-analysis-budget-is-low",
        "openai",
        {"max_tokens": 40960, "reasoning_max_tokens": 8192},
        {},
        {"max_tokens": 40960, "reasoning_effort": "low"},
    ),
    ("openai-cap-clamped", "openai", {"max_tokens": 10240}, {"max_output_tokens": 4096}, {"max_tokens": 4096}),
    (
        "openai-budget-with-no-cap-to-take-a-share-of",
        "openai",
        {"reasoning_max_tokens": 4096},
        {},
        {},
    ),
    # --- OpenRouter -------------------------------------------------------------------------------------------------
    *[
        (f"openrouter-{word}-as-asked", "router", {"reasoning_effort": word}, {}, {"reasoning_effort": word})
        for word in EFFORT_LADDER
    ],
    (
        "openrouter-budget-is-a-word",
        "router",
        {"max_tokens": 10240, "reasoning_max_tokens": 8192},
        {},
        {"max_tokens": 10240, "reasoning_effort": "high"},
    ),
    (
        "openrouter-no-reasoning-catalogued",
        "router",
        JUDGE,
        {"reasoning_efforts": ()},
        {**JUDGE, "reasoning_effort": None},
    ),
    # --- strict output ----------------------------------------------------------------------------------------------
    ("anthropic-no-structured-output", "anthropic", {"strict_output": True}, {"structured_output": False}, {}),
    ("openai-no-structured-output", "openai", {"strict_output": True}, {"structured_output": False}, {}),
    ("openrouter-no-structured-output", "router", {"strict_output": True}, {"structured_output": False}, {}),
    (
        "openrouter-no-endpoint-under-enforcement",
        "router",
        JUDGE,
        {"enforcement": False},
        {**JUDGE, "strict_output": False},
    ),
    (
        "anthropic-enforcement-is-openrouter-s",
        "anthropic",
        {"strict_output": True},
        {"enforcement": False},
        {"strict_output": True},
    ),
    ("unstrict-stays-unstrict", "openai", {"strict_output": False}, {"structured_output": True}, {}),
]


@pytest.mark.parametrize(("provider", "asked", "facts", "sent"), [row[1:] for row in ROWS], ids=[r[0] for r in ROWS])
def test_closest_settings_sends_the_closest_setting_the_model_takes(
    provider: Any, asked: dict[str, Any], facts: dict[str, Any], sent: dict[str, Any]
) -> None:
    chosen, notes = closest_settings(WireSettings(**asked), SettingFacts(**facts), provider)
    assert chosen == WireSettings(**sent)
    assert bool(notes) == (chosen != WireSettings(**asked)), "every setting that moved is noted, and only those"


def test_the_rows_are_unique_and_cover_every_provider() -> None:
    """Non-vacuity: the table names each row once and exercises all three families."""
    assert len({row[0] for row in ROWS}) == len(ROWS)
    assert {row[1] for row in ROWS} == {"anthropic", "openai", "router"}


def test_the_ladder_is_the_engine_s_seven_words_lowest_first() -> None:
    assert set(EFFORT_LADDER) == set(REASONING_EFFORTS) and len(EFFORT_LADDER) == len(REASONING_EFFORTS)
    assert EFFORT_LADDER[0] == "none" and EFFORT_LADDER[-1] == "max"


def test_the_budget_shares_are_openrouter_s_documented_ratios() -> None:
    """The one constant both directions read: a budget to a word, and a word to a budget."""
    assert BUDGET_SHARES == {
        "minimal": Fraction(1, 10),
        "low": Fraction(1, 5),
        "medium": Fraction(1, 2),
        "high": Fraction(4, 5),
        "xhigh": Fraction(19, 20),
        "max": Fraction(19, 20),
    }


@pytest.mark.parametrize(
    ("budget", "word"),
    [
        (100, "minimal"),
        (150, "minimal"),  # the minimal/low midpoint goes lower
        (151, "low"),
        (350, "low"),
        (351, "medium"),
        (650, "medium"),
        (651, "high"),
        (800, "high"),
        (875, "high"),
        (876, "xhigh"),
        (999, "xhigh"),
    ],
)
def test_a_budget_s_share_maps_to_the_nearest_word_and_a_tie_goes_lower(budget: int, word: str) -> None:
    sent, _ = closest_settings(WireSettings(max_tokens=1000, reasoning_max_tokens=budget), SettingFacts(), "openai")
    assert sent.reasoning_effort == word and sent.reasoning_max_tokens is None


def test_a_word_sent_back_as_a_budget_is_the_budget_asked() -> None:
    """The judge's 8192 of 10240 goes to OpenRouter as ``high``; ``high`` on a budget model is 8192 again."""
    to_word, _ = closest_settings(WireSettings(max_tokens=10240, reasoning_max_tokens=8192), SettingFacts(), "router")
    back, _ = closest_settings(to_word, SettingFacts(**HAIKU_4_5), "anthropic")
    assert to_word.reasoning_effort == "high" and back.reasoning_max_tokens == 8192


def test_the_sent_settings_report_names_a_cap_or_nothing() -> None:
    """A request with no output cap cannot be reported as a settings value, which names one."""
    assert WireSettings(reasoning_effort="high").reported() is None
    assert WireSettings(max_tokens=10240, reasoning_effort="high", strict_output=True).reported() == (
        ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)
    )


def test_learned_facts_win_over_declared_ones() -> None:
    """A refusal outranks the registry (``SettingFacts.over``): the provider's answer is the live fact."""
    learned = SettingFacts(accepts_temperature=False, refused_efforts=frozenset({"max"}))
    declared = SettingFacts(accepts_temperature=True, reasoning_efforts=("low", "max"), refused_efforts=frozenset())
    combined = learned.over(declared)
    assert combined.accepts_temperature is False and combined.reasoning_efforts == ("low", "max")
    sent, _ = closest_settings(WireSettings(temperature=0.2, reasoning_effort="max"), combined, "openai")
    assert sent == WireSettings(reasoning_effort="low")
