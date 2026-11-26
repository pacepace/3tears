"""The key pool's per-provider classifiers, against recorded error responses.

Every response is served through the provider's own SDK and LangChain chat model, built by the
public factory, so each classifier sees the exception a caller of ``model.invoke`` really gets --
LangChain's re-raised subclass, the SDK's own body parsing, the openrouter SDK's typed errors -- and
not one constructed by hand. The fixtures, and where each shape comes from, are in
``key_error_fixtures.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone

import anthropic
import httpx
import openai
import openrouter
import openrouter.errors
import pytest

from threetears.models.errors import is_rate_limit
from threetears.models.key_pool.classify import Verdict, VerdictKind, VerdictScope, classifier_for
from threetears.models.providers.openai import create_openai_chat
from threetears.models.providers.openrouter import create_openrouter_chat

from . import key_error_fixtures as fx
from .provider_wire import anthropic_model, serve_http_handler

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)

#: Every error each provider's chat model raises for a failed call, so a test catches exactly that.
_ANTHROPIC_RAISES = (anthropic.APIError,)
_OPENAI_RAISES = (openai.APIError,)
_OPENROUTER_RAISES = (openrouter.errors.OpenRouterError, ValueError, httpx.TransportError)

Answer = Callable[[httpx.Request], httpx.Response]


def _answer(recorded: fx.RecordedError | fx.RecordedStream) -> Answer:
    return lambda request: recorded.response()


def anthropic_raises(answer: Answer, *, stream: bool = False) -> BaseException:
    """what an Anthropic model raises when its API answers with ``answer``."""
    with serve_http_handler(answer) as base_url:
        model = anthropic_model(base_url)
        with pytest.raises(_ANTHROPIC_RAISES) as raised:
            if stream:
                list(model.stream("hi"))
            else:
                model.invoke("hi")
    return raised.value


def openai_raises(answer: Answer, *, stream: bool = False) -> BaseException:
    """what an OpenAI model raises when its API answers with ``answer``."""
    transport = httpx.MockTransport(answer)
    model = create_openai_chat(
        "gpt-4o",
        "sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
    )
    with pytest.raises(_OPENAI_RAISES) as raised:
        if stream:
            list(model.stream("hi"))
        else:
            model.invoke("hi")
    return raised.value


def openrouter_raises(answer: Answer, *, stream: bool = False) -> BaseException:
    """what an OpenRouter model raises when its API answers with ``answer``."""
    transport = httpx.MockTransport(answer)
    # a caller's own SDK client keeps its own retry policy, and the SDK's default retries a 5xx or a
    # dropped connection for an hour (openrouter/chat.py), so this one retries nothing
    sdk = openrouter.OpenRouter(
        api_key="sk-or-test",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        retry_config=None,
    )
    model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-or-test", client=sdk)
    with pytest.raises(_OPENROUTER_RAISES) as raised:
        if stream:
            list(model.stream("hi"))
        else:
            model.invoke("hi")
    return raised.value


def _verdict(kind: VerdictKind, scope: VerdictScope, *, billed: bool, until: datetime | None = None) -> Verdict:
    return Verdict(kind, scope, may_have_billed=billed, until=until)


EXHAUSTED_ACCOUNT = _verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, billed=False)
EXHAUSTED_KEY = _verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, billed=False)
REVOKED = _verdict(VerdictKind.REVOKED, VerdictScope.KEY, billed=False)
FORBIDDEN = _verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, billed=False)
NOT_KEYS_UNBILLED = _verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, billed=False)
NOT_KEYS_BILLED = _verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, billed=True)
OVERSIZED = _verdict(VerdictKind.OVERSIZED, VerdictScope.KEY, billed=False)

#: The pool's default threshold: a rate limit asking a longer wait than this counts as exhaustion.
THRESHOLD = timedelta(seconds=120)


class TestAnthropic:
    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [
            pytest.param(fx.ANTHROPIC_BILLING_402, EXHAUSTED_ACCOUNT, id="402-billing_error"),
            pytest.param(fx.ANTHROPIC_CREDIT_TOO_LOW_400, EXHAUSTED_ACCOUNT, id="400-credit-too-low"),
            pytest.param(fx.ANTHROPIC_SPEND_LIMIT_400, EXHAUSTED_ACCOUNT, id="400-spend-limit"),
            pytest.param(fx.ANTHROPIC_WORKSPACE_SPEND_LIMIT_400, EXHAUSTED_ACCOUNT, id="400-workspace-spend-limit"),
            pytest.param(fx.ANTHROPIC_SPEND_CAP_429, EXHAUSTED_ACCOUNT, id="429-enforced_spend_limit_reached"),
            pytest.param(
                fx.ANTHROPIC_CONTRACT_SPEND_LIMIT_429, EXHAUSTED_ACCOUNT, id="429-contract_spend_limit_reached"
            ),
            pytest.param(
                fx.ANTHROPIC_RAMP_LIMIT_429,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(seconds=5)
                ),
                id="429-ramp_limit_exceeded-is-a-rate",
            ),
            pytest.param(fx.ANTHROPIC_AUTHENTICATION_401, REVOKED, id="401"),
            pytest.param(fx.ANTHROPIC_PERMISSION_403, FORBIDDEN, id="403"),
            pytest.param(
                fx.ANTHROPIC_RATE_LIMIT_429,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(seconds=17)
                ),
                id="429-retry-after",
            ),
            pytest.param(
                fx.ANTHROPIC_RATE_LIMIT_429_RESET_ONLY,
                _verdict(
                    VerdictKind.RATE_LIMITED,
                    VerdictScope.ACCOUNT,
                    billed=False,
                    until=datetime(2026, 11, 26, 12, 0, 45, tzinfo=UTC),
                ),
                id="429-latest-reset-of-a-used-up-limit",
            ),
            pytest.param(fx.ANTHROPIC_OVERLOADED_529, NOT_KEYS_BILLED, id="529"),
            pytest.param(fx.ANTHROPIC_API_ERROR_500, NOT_KEYS_BILLED, id="500"),
            pytest.param(fx.ANTHROPIC_INVALID_REQUEST_400, NOT_KEYS_UNBILLED, id="other-400"),
        ],
    )
    def test_each_recorded_response(self, recorded: fx.RecordedError, expected: Verdict) -> None:
        raised = anthropic_raises(_answer(recorded))
        assert classifier_for("anthropic").classify(raised, now=NOW) == expected

    def test_an_error_event_mid_stream_may_have_billed_whatever_the_pool_saw(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_MID_STREAM_OVERLOADED), stream=True)
        assert classifier_for("anthropic").classify(raised, now=NOW) == NOT_KEYS_BILLED

    def test_a_rate_limit_mid_stream_is_still_the_accounts(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_MID_STREAM_RATE_LIMIT), stream=True)
        assert classifier_for("anthropic").classify(raised, now=NOW) == _verdict(
            VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=True
        )

    def test_a_refused_connection_was_not_billed(self) -> None:
        with serve_http_handler(_answer(fx.ANTHROPIC_API_ERROR_500)) as base_url:
            pass  # the server is gone once the block ends, so its port refuses the connection
        model = anthropic_model(base_url)
        with pytest.raises(anthropic.APIConnectionError) as raised:
            model.invoke("hi")
        assert classifier_for("anthropic").classify(raised.value, now=NOW) == NOT_KEYS_UNBILLED


class TestOpenAI:
    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [
            pytest.param(fx.OPENAI_INSUFFICIENT_QUOTA_429, EXHAUSTED_ACCOUNT, id="429-insufficient_quota"),
            pytest.param(fx.OPENAI_CREDIT_BALANCE_EXHAUSTED_429, EXHAUSTED_ACCOUNT, id="429-credit_balance_exhausted"),
            pytest.param(fx.OPENAI_PROJECT_SPEND_LIMIT_429, EXHAUSTED_ACCOUNT, id="429-project_spend_limit_exceeded"),
            pytest.param(
                fx.OPENAI_ORGANIZATION_SPEND_LIMIT_429, EXHAUSTED_ACCOUNT, id="429-organization_spend_limit_exceeded"
            ),
            pytest.param(
                fx.OPENAI_ORGANIZATION_USAGE_LIMIT_429, EXHAUSTED_ACCOUNT, id="429-organization_usage_limit_exceeded"
            ),
            pytest.param(fx.OPENAI_BILLING_HARD_LIMIT_429, EXHAUSTED_ACCOUNT, id="429-billing_hard_limit_reached"),
            pytest.param(
                fx.OPENAI_SLOW_DOWN_429,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(seconds=2)
                ),
                id="429-slow_down-is-a-rate",
            ),
            pytest.param(
                fx.OPENAI_RATE_LIMIT_429,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(seconds=56)
                ),
                id="429-retry-after",
            ),
            pytest.param(
                fx.OPENAI_RATE_LIMIT_429_RESET_ONLY,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(minutes=6)
                ),
                id="429-latest-reset-of-a-used-up-limit",
            ),
            pytest.param(fx.OPENAI_INVALID_API_KEY_401, REVOKED, id="401"),
            pytest.param(fx.OPENAI_REGION_403, FORBIDDEN, id="403"),
            pytest.param(fx.OPENAI_COMPATIBLE_BALANCE_402, EXHAUSTED_KEY, id="402-compatible-endpoint"),
            pytest.param(fx.OPENAI_SERVER_500, NOT_KEYS_BILLED, id="500"),
        ],
    )
    def test_each_recorded_response(self, recorded: fx.RecordedError, expected: Verdict) -> None:
        raised = openai_raises(_answer(recorded))
        assert classifier_for("openai").classify(raised, now=NOW) == expected

    def test_a_quota_error_mid_stream_is_the_accounts_and_may_have_billed(self) -> None:
        raised = openai_raises(_answer(fx.OPENAI_MID_STREAM_QUOTA), stream=True)
        assert classifier_for("openai").classify(raised, now=NOW) == _verdict(
            VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, billed=True
        )

    def test_a_rate_limit_mid_stream_is_read_by_its_type(self) -> None:
        raised = openai_raises(_answer(fx.OPENAI_MID_STREAM_RATE_LIMIT), stream=True)
        assert classifier_for("openai").classify(raised, now=NOW) == _verdict(
            VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=True
        )

    def test_a_refused_connection_was_not_billed(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        raised = openai_raises(refuse)
        assert isinstance(raised, openai.APIConnectionError)
        assert classifier_for("openai").classify(raised, now=NOW) == NOT_KEYS_UNBILLED

    def test_a_timeout_may_have_billed(self) -> None:
        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out", request=request)

        raised = openai_raises(stall)
        assert isinstance(raised, openai.APITimeoutError)
        assert classifier_for("openai").classify(raised, now=NOW) == NOT_KEYS_BILLED


class TestOpenRouter:
    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [
            pytest.param(fx.OPENROUTER_INVALID_KEY_401, REVOKED, id="401"),
            pytest.param(fx.OPENROUTER_KEY_LIMIT_402, EXHAUSTED_KEY, id="402-key-limit"),
            pytest.param(fx.OPENROUTER_CREDITS_402, EXHAUSTED_ACCOUNT, id="402-account-credits"),
            pytest.param(
                fx.OPENROUTER_IN_FLIGHT_402,
                _verdict(
                    VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(seconds=3)
                ),
                id="402-in-flight-budget-with-retry-after",
            ),
            pytest.param(
                fx.OPENROUTER_IN_FLIGHT_402_NO_RETRY_AFTER, EXHAUSTED_ACCOUNT, id="402-in-flight-budget-no-retry-after"
            ),
            pytest.param(fx.OPENROUTER_WEIGHT_EXCEEDS_BUDGET_402, OVERSIZED, id="402-weight_exceeds_budget"),
            pytest.param(fx.OPENROUTER_BARE_402, EXHAUSTED_KEY, id="402-no-limit-named"),
            pytest.param(fx.OPENROUTER_MODERATION_403, NOT_KEYS_UNBILLED, id="403-moderation-flag"),
            pytest.param(fx.OPENROUTER_CONTENT_POLICY_403, NOT_KEYS_UNBILLED, id="403-content_policy_violation"),
            pytest.param(fx.OPENROUTER_REFUSAL_403, NOT_KEYS_UNBILLED, id="403-refusal"),
            pytest.param(
                fx.OPENROUTER_GUARDRAIL_BUDGET_403,
                _verdict(
                    VerdictKind.EXHAUSTED,
                    VerdictScope.ACCOUNT,
                    billed=False,
                    until=datetime(2026, 10, 12, tzinfo=UTC),
                ),
                id="403-guardrail-budget-until-its-reset",
            ),
            pytest.param(
                fx.OPENROUTER_KEY_BUDGET_403,
                _verdict(
                    VerdictKind.EXHAUSTED, VerdictScope.KEY, billed=False, until=datetime(2026, 11, 27, tzinfo=UTC)
                ),
                id="403-key-budget-until-its-reset",
            ),
            pytest.param(fx.OPENROUTER_FORBIDDEN_403, FORBIDDEN, id="403"),
            pytest.param(
                fx.OPENROUTER_RATE_LIMIT_429,
                _verdict(
                    VerdictKind.RATE_LIMITED,
                    VerdictScope.KEY_AND_MODEL,
                    billed=False,
                    until=datetime(2026, 11, 26, 12, 1, 30, tzinfo=UTC),
                ),
                id="429-retry-after-http-date",
            ),
            pytest.param(
                fx.OPENROUTER_RATE_LIMIT_429_RESET_ONLY,
                _verdict(
                    VerdictKind.RATE_LIMITED,
                    VerdictScope.KEY_AND_MODEL,
                    billed=False,
                    until=datetime(2026, 11, 26, 12, 0, 45, tzinfo=UTC),
                ),
                id="429-x-ratelimit-reset-in-epoch-milliseconds",
            ),
            pytest.param(fx.OPENROUTER_UPSTREAM_502, NOT_KEYS_BILLED, id="502"),
        ],
    )
    def test_each_recorded_response(self, recorded: fx.RecordedError, expected: Verdict) -> None:
        raised = openrouter_raises(_answer(recorded))
        assert classifier_for("openrouter").classify(raised, now=NOW) == expected

    @pytest.mark.parametrize(
        ("scope", "expected"),
        [
            ("key_limit", VerdictScope.KEY),
            ("api_key", VerdictScope.KEY),
            ("member", VerdictScope.ACCOUNT),
            ("workspace", VerdictScope.ACCOUNT),
            ("workspace_default_key", VerdictScope.ACCOUNT),
        ],
    )
    @pytest.mark.parametrize(
        "source", ["openrouter_guardrail_budget", "openrouter_workspace_budget", "openrouter_key_limit"]
    )
    def test_a_403_budget_is_the_keys_only_for_a_keys_own_scope(
        self, source: str, scope: str, expected: VerdictScope
    ) -> None:
        metadata = {"limit_source": source, "scope": scope, "resets_at": None}
        recorded = fx.RecordedError(
            403, {"error": {"code": 403, "message": "Budget limit exceeded", "metadata": metadata}}
        )
        verdict = classifier_for("openrouter").classify(openrouter_raises(_answer(recorded)), now=NOW)
        # a lifetime budget (resets_at null) names no reset: the pool holds it to the period's end
        assert verdict == _verdict(VerdictKind.EXHAUSTED, expected, billed=False)

    def test_the_mid_stream_429_is_one_models_rate_limit_and_may_have_billed(self) -> None:
        raised = openrouter_raises(_answer(fx.OPENROUTER_MID_STREAM_RATE_LIMIT), stream=True)
        assert isinstance(raised, ValueError)
        assert is_rate_limit(raised)  # the breaker's reading agrees
        assert classifier_for("openrouter").classify(raised, now=NOW) == _verdict(
            VerdictKind.RATE_LIMITED, VerdictScope.KEY_AND_MODEL, billed=True
        )

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (402, _verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, billed=True)),
            (401, _verdict(VerdictKind.REVOKED, VerdictScope.KEY, billed=True)),
            (403, NOT_KEYS_BILLED),  # no metadata survives to tell a moderation flag from a forbidden key
            (502, NOT_KEYS_BILLED),
        ],
    )
    def test_the_other_codes_an_in_band_error_carries(self, code: int, expected: Verdict) -> None:
        raised = ValueError(f"OpenRouter API returned an error during streaming: upstream said no (code: {code})")
        assert classifier_for("openrouter").classify(raised, now=NOW) == expected

    def test_a_refused_connection_was_not_billed(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        raised = openrouter_raises(refuse)
        assert isinstance(raised, httpx.ConnectError)
        assert classifier_for("openrouter").classify(raised, now=NOW) == NOT_KEYS_UNBILLED

    def test_a_read_timeout_may_have_billed(self) -> None:
        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out", request=request)

        raised = openrouter_raises(stall)
        assert isinstance(raised, httpx.ReadTimeout)
        assert classifier_for("openrouter").classify(raised, now=NOW) == NOT_KEYS_BILLED


class TestAnUnknownErrorNeverBurnsAKey:
    @pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
    @pytest.mark.parametrize(
        "raised",
        [
            RuntimeError("something else broke"),
            KeyError("missing"),
            ValueError("(code: 429) but not from OpenRouter"),
        ],
    )
    def test_it_is_not_key_related_and_may_have_billed(self, provider: str, raised: BaseException) -> None:
        assert classifier_for(provider).classify(raised, now=NOW) == NOT_KEYS_BILLED

    def test_another_providers_rate_limit_is_not_this_providers_key(self) -> None:
        raised = openai_raises(_answer(fx.OPENAI_RATE_LIMIT_429))
        for provider in ("anthropic", "openrouter"):
            assert not classifier_for(provider).classify(raised, now=NOW).moves_key


class TestRetryAfter:
    """``retry-after`` in each form it comes in, read the same by every classifier that reads it."""

    @staticmethod
    def _until(headers: dict[str, str]) -> datetime | None:
        recorded = fx.RecordedError(429, fx.ANTHROPIC_RATE_LIMIT_429.body, headers=headers)
        raised = anthropic_raises(_answer(recorded))
        return classifier_for("anthropic").classify(raised, now=NOW).until

    @pytest.mark.parametrize(
        ("headers", "until"),
        [
            pytest.param({"retry-after": "30"}, NOW + timedelta(seconds=30), id="seconds"),
            pytest.param({"retry-after": "1.5"}, NOW + timedelta(seconds=1.5), id="fractional-seconds"),
            pytest.param(
                {"retry-after": "Thu, 26 Nov 2026 12:02:00 GMT"},
                datetime(2026, 11, 26, 12, 2, tzinfo=UTC),
                id="http-date",
            ),
            pytest.param({"retry-after": "Thu, 26 Nov 2026 11:00:00 GMT"}, NOW, id="http-date-already-past"),
            pytest.param(
                {"retry-after-ms": "2500", "retry-after": "60"}, NOW + timedelta(seconds=2.5), id="milliseconds-first"
            ),
            pytest.param({}, None, id="absent"),
            pytest.param({"retry-after": "soon"}, None, id="unreadable"),
        ],
    )
    def test_each_form(self, headers: dict[str, str], until: datetime | None) -> None:
        assert self._until(headers) == until

    def test_absent_leaves_no_until_for_every_provider(self) -> None:
        bare_anthropic = fx.RecordedError(429, fx.ANTHROPIC_RATE_LIMIT_429.body)
        bare_openai = fx.RecordedError(429, fx.OPENAI_RATE_LIMIT_429.body)
        bare_openrouter = fx.RecordedError(429, fx.OPENROUTER_RATE_LIMIT_429.body)
        verdicts = [
            classifier_for("anthropic").classify(anthropic_raises(_answer(bare_anthropic)), now=NOW),
            classifier_for("openai").classify(openai_raises(_answer(bare_openai)), now=NOW),
            classifier_for("openrouter").classify(openrouter_raises(_answer(bare_openrouter)), now=NOW),
        ]
        assert [(v.kind, v.until) for v in verdicts] == [(VerdictKind.RATE_LIMITED, None)] * 3

    def test_a_retry_after_in_seconds_counts_from_the_callers_now_in_utc(self) -> None:
        raised = openai_raises(_answer(fx.OPENAI_RATE_LIMIT_429))
        elsewhere = NOW.astimezone(timezone(timedelta(hours=-5)))
        until = classifier_for("openai").classify(raised, now=elsewhere).until
        assert until == NOW + timedelta(seconds=56)
        assert until is not None and until.tzinfo is UTC


class TestMayHaveBilled:
    @pytest.mark.parametrize(
        ("provider", "raised"),
        [
            ("anthropic", lambda: anthropic_raises(_answer(fx.ANTHROPIC_RATE_LIMIT_429))),
            ("openai", lambda: openai_raises(_answer(fx.OPENAI_INVALID_API_KEY_401))),
            ("openrouter", lambda: openrouter_raises(_answer(fx.OPENROUTER_KEY_LIMIT_402))),
        ],
    )
    def test_a_refusal_is_unbilled_until_the_pool_says_a_stream_had_begun(
        self, provider: str, raised: Callable[[], BaseException]
    ) -> None:
        error = raised()
        classifier = classifier_for(provider)
        assert classifier.classify(error, now=NOW).may_have_billed is False
        after_output = classifier.classify(error, now=NOW, stream_started=True)
        assert after_output.may_have_billed is True
        assert after_output.kind is classifier.classify(error, now=NOW).kind  # the verdict itself is unchanged


class TestTheContract:
    def test_a_provider_with_no_classifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="voyageai"):
            classifier_for("voyageai")

    def test_now_must_carry_a_timezone(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            classifier_for("anthropic").classify(RuntimeError("x"), now=datetime(2026, 11, 26, 12))  # noqa: DTZ001

    @pytest.mark.parametrize(
        "kind", [VerdictKind.REVOKED, VerdictKind.FORBIDDEN, VerdictKind.OVERSIZED, VerdictKind.NOT_KEY_RELATED]
    )
    def test_only_a_rate_limit_or_an_exhaustion_carries_an_until(self, kind: VerdictKind) -> None:
        with pytest.raises(ValueError, match="until"):
            Verdict(kind, VerdictScope.ACCOUNT, may_have_billed=False, until=NOW)
        for timed in (VerdictKind.RATE_LIMITED, VerdictKind.EXHAUSTED):
            assert Verdict(timed, VerdictScope.KEY, may_have_billed=False, until=NOW).until == NOW

    def test_an_until_carries_a_timezone(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            Verdict(
                VerdictKind.RATE_LIMITED,
                VerdictScope.KEY,
                may_have_billed=False,
                until=datetime(2026, 11, 26, 12),  # noqa: DTZ001
            )

    @pytest.mark.parametrize("kind", list(VerdictKind))
    def test_only_a_used_up_dead_or_forbidden_key_moves(self, kind: VerdictKind) -> None:
        verdict = Verdict(kind, VerdictScope.KEY, may_have_billed=False)
        moving = {VerdictKind.EXHAUSTED, VerdictKind.REVOKED, VerdictKind.FORBIDDEN}
        assert verdict.moves_key is (kind in moving)
        assert verdict.tries_another_key is (kind in moving | {VerdictKind.OVERSIZED})

    def test_only_openrouters_sdk_leaves_rate_limits_to_the_pool(self) -> None:
        assert [classifier_for(p).sdk_retries_rate_limits for p in ("anthropic", "openai", "openrouter")] == [
            True,
            True,
            False,
        ]


class TestLongWait:
    """A rate limit whose wait runs past the pool's threshold counts as exhaustion until the wait ends."""

    def test_a_wait_past_the_threshold_is_exhaustion_until_it_ends(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT))
        verdict = classifier_for("anthropic").classify(raised, now=NOW, exhaustion_wait=THRESHOLD)
        assert verdict == _verdict(
            VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, billed=False, until=NOW + timedelta(hours=1)
        )
        assert verdict.moves_key

    def test_a_wait_within_the_threshold_stays_a_rate_limit(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT))
        verdict = classifier_for("anthropic").classify(raised, now=NOW, exhaustion_wait=timedelta(hours=2))
        assert (verdict.kind, verdict.until) == (VerdictKind.RATE_LIMITED, NOW + timedelta(hours=1))

    def test_a_wait_of_exactly_the_threshold_stays_a_rate_limit(self) -> None:
        recorded = fx.RecordedError(429, fx.ANTHROPIC_RATE_LIMIT_429.body, headers={"retry-after": "120"})
        raised = anthropic_raises(_answer(recorded))
        verdict = classifier_for("anthropic").classify(raised, now=NOW, exhaustion_wait=THRESHOLD)
        assert verdict.kind is VerdictKind.RATE_LIMITED

    def test_with_no_threshold_every_rate_limit_is_one(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT))
        assert classifier_for("anthropic").classify(raised, now=NOW).kind is VerdictKind.RATE_LIMITED

    def test_a_long_reset_header_counts_as_a_long_wait(self) -> None:
        raised = anthropic_raises(_answer(fx.ANTHROPIC_RATE_LIMIT_429_RESET_ONLY))
        verdict = classifier_for("anthropic").classify(raised, now=NOW, exhaustion_wait=timedelta(seconds=30))
        assert (verdict.kind, verdict.until) == (VerdictKind.EXHAUSTED, datetime(2026, 11, 26, 12, 0, 45, tzinfo=UTC))

    def test_a_per_model_rate_limit_held_long_takes_the_whole_key(self) -> None:
        raised = openrouter_raises(_answer(fx.OPENROUTER_RATE_LIMIT_429))  # retry-after 12:01:30
        verdict = classifier_for("openrouter").classify(raised, now=NOW, exhaustion_wait=timedelta(seconds=60))
        assert verdict == _verdict(
            VerdictKind.EXHAUSTED, VerdictScope.KEY, billed=False, until=datetime(2026, 11, 26, 12, 1, 30, tzinfo=UTC)
        )

    def test_a_rate_limit_with_no_wait_named_is_never_long(self) -> None:
        bare = fx.RecordedError(429, fx.OPENAI_RATE_LIMIT_429.body)
        verdict = classifier_for("openai").classify(
            openai_raises(_answer(bare)), now=NOW, exhaustion_wait=timedelta(microseconds=1)
        )
        assert verdict.kind is VerdictKind.RATE_LIMITED


def _wire(recorded: fx.RecordedError) -> tuple[int, httpx.Headers, bytes]:
    """a recorded response as an SDK hook sees it: status, headers, body bytes."""
    return recorded.status, httpx.Headers(recorded.headers), json.dumps(recorded.body).encode()


class TestRefusesRetry:
    """The rule the SDK hook asks: raise at once what is the key's, leave a rate limit to the SDK."""

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
    @pytest.mark.parametrize("status", [401, 402, 403])
    def test_a_key_refusal_is_never_retried_whatever_its_body(self, provider: str, status: int) -> None:
        assert classifier_for(provider).refuses_retry(status, httpx.Headers(), None, now=NOW, exhaustion_wait=THRESHOLD)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
    @pytest.mark.parametrize("status", [200, 400, 408, 500, 529])
    def test_any_other_status_is_left_to_the_sdk(self, provider: str, status: int) -> None:
        headers = httpx.Headers({"retry-after": "3600"})
        rule = classifier_for(provider)
        assert not rule.refuses_retry(status, headers, None, now=NOW, exhaustion_wait=THRESHOLD)

    @pytest.mark.parametrize(
        ("provider", "recorded", "refused"),
        [
            pytest.param("anthropic", fx.ANTHROPIC_SPEND_CAP_429, True, id="anthropic-spend-cap"),
            pytest.param("anthropic", fx.ANTHROPIC_CONTRACT_SPEND_LIMIT_429, True, id="anthropic-contract-limit"),
            pytest.param("anthropic", fx.ANTHROPIC_RAMP_LIMIT_429, False, id="anthropic-ramp-limit"),
            pytest.param("anthropic", fx.ANTHROPIC_RATE_LIMIT_429, False, id="anthropic-rate-limit"),
            pytest.param("anthropic", fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT, True, id="anthropic-long-wait"),
            pytest.param("openai", fx.OPENAI_INSUFFICIENT_QUOTA_429, True, id="openai-insufficient_quota"),
            pytest.param("openai", fx.OPENAI_CREDIT_BALANCE_EXHAUSTED_429, True, id="openai-credit-balance"),
            pytest.param("openai", fx.OPENAI_PROJECT_SPEND_LIMIT_429, True, id="openai-project-spend"),
            pytest.param("openai", fx.OPENAI_ORGANIZATION_SPEND_LIMIT_429, True, id="openai-organization-spend"),
            pytest.param("openai", fx.OPENAI_ORGANIZATION_USAGE_LIMIT_429, True, id="openai-organization-usage"),
            pytest.param("openai", fx.OPENAI_BILLING_HARD_LIMIT_429, True, id="openai-billing-hard-limit"),
            pytest.param("openai", fx.OPENAI_RATE_LIMIT_429, False, id="openai-rate-limit"),
            pytest.param("openai", fx.OPENAI_SLOW_DOWN_429, False, id="openai-slow-down"),
            pytest.param("openrouter", fx.OPENROUTER_RATE_LIMIT_429, False, id="openrouter-rate-limit"),
            pytest.param("openrouter", fx.OPENROUTER_RATE_LIMIT_429_RESET_ONLY, False, id="openrouter-reset-only"),
        ],
    )
    def test_a_429_by_its_body_and_its_wait(self, provider: str, recorded: fx.RecordedError, refused: bool) -> None:
        status, headers, body = _wire(recorded)
        assert (
            classifier_for(provider).refuses_retry(status, headers, body, now=NOW, exhaustion_wait=THRESHOLD) is refused
        )

    @pytest.mark.parametrize(
        ("provider", "recorded", "answer"),
        [
            ("anthropic", fx.ANTHROPIC_SPEND_CAP_429, anthropic_raises),
            ("anthropic", fx.ANTHROPIC_CONTRACT_SPEND_LIMIT_429, anthropic_raises),
            ("anthropic", fx.ANTHROPIC_RAMP_LIMIT_429, anthropic_raises),
            ("anthropic", fx.ANTHROPIC_RATE_LIMIT_429, anthropic_raises),
            ("anthropic", fx.ANTHROPIC_RATE_LIMIT_429_LONG_WAIT, anthropic_raises),
            ("openai", fx.OPENAI_INSUFFICIENT_QUOTA_429, openai_raises),
            ("openai", fx.OPENAI_BILLING_HARD_LIMIT_429, openai_raises),
            ("openai", fx.OPENAI_RATE_LIMIT_429, openai_raises),
            ("openai", fx.OPENAI_SLOW_DOWN_429, openai_raises),
        ],
    )
    def test_the_hook_and_the_classifier_read_a_429_alike(
        self, provider: str, recorded: fx.RecordedError, answer: Callable[[Answer], BaseException]
    ) -> None:
        classifier = classifier_for(provider)
        status, headers, body = _wire(recorded)
        refused = classifier.refuses_retry(status, headers, body, now=NOW, exhaustion_wait=THRESHOLD)
        verdict = classifier.classify(answer(_answer(recorded)), now=NOW, exhaustion_wait=THRESHOLD)
        assert refused is verdict.moves_key

    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    @pytest.mark.parametrize("body", [b"<html>Too Many Requests</html>", b"\xff\xfe", b"", None])
    def test_a_429_whose_body_is_not_json_stays_retriable(self, provider: str, body: bytes | None) -> None:
        headers = httpx.Headers({"retry-after": "3"})
        assert not classifier_for(provider).refuses_retry(429, headers, body, now=NOW, exhaustion_wait=THRESHOLD)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
    def test_a_429_asking_a_long_wait_is_raised_at_once(self, provider: str) -> None:
        headers = httpx.Headers({"retry-after": "121"})
        assert classifier_for(provider).refuses_retry(429, headers, b"{}", now=NOW, exhaustion_wait=THRESHOLD)
        headers = httpx.Headers({"retry-after": "120"})
        assert not classifier_for(provider).refuses_retry(429, headers, b"{}", now=NOW, exhaustion_wait=THRESHOLD)

    def test_now_must_carry_a_timezone(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            classifier_for("openai").refuses_retry(
                429,
                httpx.Headers(),
                None,
                now=datetime(2026, 11, 26, 12),
                exhaustion_wait=THRESHOLD,  # noqa: DTZ001
            )
