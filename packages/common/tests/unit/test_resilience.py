"""Unit tests for retry, circuit breaking, and timeouts.

Sleep, clock, and randomness are injected throughout so backoff schedules and
state transitions are asserted exactly, with no real waiting.
"""

from __future__ import annotations

import asyncio

import pytest

from fie_common.errors import (
    CircuitOpenError,
    DatabaseError,
    OperationTimeoutError,
    RateLimitError,
    ValidationError,
)
from fie_common.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResiliencePolicy,
    RetryPolicy,
    default_retryable,
    retry_async,
    with_timeout,
)

pytestmark = pytest.mark.unit


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SleepRecorder:
    """Records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class TestRetryPolicy:
    def test_first_attempt_has_no_backoff(self) -> None:
        policy = RetryPolicy(jitter=0.0)

        assert policy.backoff_for(1) == 0.0

    def test_backoff_grows_exponentially(self) -> None:
        policy = RetryPolicy(initial_backoff_seconds=1.0, multiplier=2.0, jitter=0.0)

        assert policy.backoff_for(2) == 1.0
        assert policy.backoff_for(3) == 2.0
        assert policy.backoff_for(4) == 4.0

    def test_backoff_is_capped(self) -> None:
        policy = RetryPolicy(
            initial_backoff_seconds=1.0, multiplier=10.0, max_backoff_seconds=5.0, jitter=0.0
        )

        assert policy.backoff_for(5) == 5.0

    def test_jitter_stays_within_the_configured_band(self) -> None:
        policy = RetryPolicy(initial_backoff_seconds=10.0, jitter=0.5, multiplier=1.0)

        assert policy.backoff_for(2, rand=lambda: 0.0) == pytest.approx(5.0)
        assert policy.backoff_for(2, rand=lambda: 1.0) == pytest.approx(15.0)
        assert policy.backoff_for(2, rand=lambda: 0.5) == pytest.approx(10.0)

    def test_backoff_is_never_negative(self) -> None:
        policy = RetryPolicy(initial_backoff_seconds=1.0, jitter=1.0, multiplier=1.0)

        assert policy.backoff_for(2, rand=lambda: 0.0) >= 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_attempts": 0},
            {"initial_backoff_seconds": -1.0},
            {"max_backoff_seconds": 0.01, "initial_backoff_seconds": 1.0},
            {"multiplier": 0.5},
            {"jitter": 1.5},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)


class TestRetryAsync:
    async def test_returns_immediately_on_success(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        sleeper = SleepRecorder()
        result = await retry_async(operation, sleep=sleeper)

        assert result == "ok"
        assert calls == 1
        assert sleeper.delays == []

    async def test_retries_a_retryable_failure_then_succeeds(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise DatabaseError("transient")
            return "recovered"

        sleeper = SleepRecorder()
        result = await retry_async(
            operation,
            policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=1.0, jitter=0.0),
            sleep=sleeper,
        )

        assert result == "recovered"
        assert calls == 3
        assert sleeper.delays == [1.0, 2.0]

    async def test_non_retryable_error_is_raised_without_retrying(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise ValidationError("bad input")

        sleeper = SleepRecorder()
        with pytest.raises(ValidationError):
            await retry_async(operation, sleep=sleeper)

        assert calls == 1, "a caller error must never be retried"
        assert sleeper.delays == []

    async def test_exhausted_attempts_reraise_the_last_error(self) -> None:
        async def operation() -> str:
            raise DatabaseError("always down")

        with pytest.raises(DatabaseError, match="always down"):
            await retry_async(
                operation,
                policy=RetryPolicy(max_attempts=3, jitter=0.0),
                sleep=SleepRecorder(),
            )

    async def test_rate_limit_retry_after_overrides_a_shorter_backoff(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("slow down", retry_after_seconds=30.0)
            return "ok"

        sleeper = SleepRecorder()
        await retry_async(
            operation,
            policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.1, jitter=0.0),
            sleep=sleeper,
        )

        assert sleeper.delays == [30.0], "provider-supplied retry-after must win"

    async def test_retry_after_does_not_shorten_a_longer_backoff(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("slow down", retry_after_seconds=0.5)
            return "ok"

        sleeper = SleepRecorder()
        await retry_async(
            operation,
            policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=10.0, jitter=0.0),
            sleep=sleeper,
        )

        assert sleeper.delays == [10.0]

    async def test_on_retry_callback_receives_attempt_error_and_delay(self) -> None:
        observed: list[tuple[int, str, float]] = []

        async def operation() -> str:
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await retry_async(
                operation,
                policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=1.0, jitter=0.0),
                on_retry=lambda attempt, error, delay: observed.append(
                    (attempt, type(error).__name__, delay)
                ),
                sleep=SleepRecorder(),
            )

        assert observed == [(1, "DatabaseError", 1.0)]

    async def test_custom_retryable_predicate_is_honored(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValidationError("normally not retried")
            return "ok"

        result = await retry_async(
            operation,
            policy=RetryPolicy(max_attempts=2, jitter=0.0),
            retryable=lambda error: isinstance(error, ValidationError),
            sleep=SleepRecorder(),
        )

        assert result == "ok"

    async def test_single_attempt_policy_disables_retry(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await retry_async(operation, policy=RetryPolicy(max_attempts=1))

        assert calls == 1


class TestDefaultRetryable:
    @pytest.mark.parametrize(
        "error", [DatabaseError("x"), RateLimitError("x"), ConnectionError("x"), TimeoutError()]
    )
    def test_transient_errors_are_retryable(self, error: BaseException) -> None:
        assert default_retryable(error) is True

    @pytest.mark.parametrize("error", [ValidationError("x"), ValueError("x")])
    def test_deterministic_errors_are_not_retryable(self, error: BaseException) -> None:
        assert default_retryable(error) is False


class TestWithTimeout:
    async def test_returns_result_within_deadline(self) -> None:
        async def operation() -> str:
            return "fast"

        assert await with_timeout(operation, seconds=1.0) == "fast"

    async def test_exceeding_the_deadline_raises_operation_timeout(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(1.0)
            return "never"

        with pytest.raises(OperationTimeoutError) as exc_info:
            await with_timeout(operation, seconds=0.01, description="slow call")

        assert exc_info.value.details["operation"] == "slow call"
        assert exc_info.value.retryable is True

    async def test_non_positive_timeout_is_rejected(self) -> None:
        async def operation() -> str:
            return "x"

        with pytest.raises(ValueError):
            await with_timeout(operation, seconds=0)


class TestCircuitBreaker:
    async def test_starts_closed_and_passes_calls_through(self) -> None:
        breaker = CircuitBreaker("test")

        async def operation() -> str:
            return "ok"

        assert await breaker.call(operation) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_opens_after_the_failure_threshold(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test", config=CircuitBreakerConfig(failure_threshold=3), clock=clock
        )

        async def failing() -> str:
            raise DatabaseError("down")

        for _ in range(3):
            with pytest.raises(DatabaseError):
                await breaker.call(failing)

        assert breaker.state is CircuitState.OPEN

    async def test_open_circuit_rejects_without_calling_the_dependency(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test", config=CircuitBreakerConfig(failure_threshold=1), clock=clock
        )
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await breaker.call(operation)

        with pytest.raises(CircuitOpenError):
            await breaker.call(operation)

        assert calls == 1, "an open circuit must not reach the dependency"

    async def test_success_resets_the_failure_count(self) -> None:
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        async def failing() -> str:
            raise DatabaseError("down")

        async def succeeding() -> str:
            return "ok"

        for _ in range(2):
            with pytest.raises(DatabaseError):
                await breaker.call(failing)
        await breaker.call(succeeding)

        with pytest.raises(DatabaseError):
            await breaker.call(failing)

        assert breaker.state is CircuitState.CLOSED

    async def test_transitions_to_half_open_after_the_recovery_window(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=30.0),
            clock=clock,
        )

        async def failing() -> str:
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        assert breaker.state is CircuitState.OPEN

        clock.advance(30.0)

        assert breaker.state is CircuitState.HALF_OPEN

    async def test_half_open_success_closes_the_circuit(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=10.0),
            clock=clock,
        )

        async def failing() -> str:
            raise DatabaseError("down")

        async def succeeding() -> str:
            return "recovered"

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        clock.advance(10.0)

        assert await breaker.call(succeeding) == "recovered"
        assert breaker.state is CircuitState.CLOSED

    async def test_half_open_failure_reopens_and_restarts_the_clock(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=10.0),
            clock=clock,
        )

        async def failing() -> str:
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        clock.advance(10.0)

        with pytest.raises(DatabaseError):
            await breaker.call(failing)

        assert breaker.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing)

    async def test_half_open_limits_concurrent_trial_calls(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout_seconds=5.0, half_open_max_calls=1
            ),
            clock=clock,
        )
        release = asyncio.Event()

        async def failing() -> str:
            raise DatabaseError("down")

        async def blocking() -> str:
            await release.wait()
            return "ok"

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        clock.advance(5.0)

        trial = asyncio.create_task(breaker.call(blocking))
        await asyncio.sleep(0)

        with pytest.raises(CircuitOpenError, match="trial capacity"):
            await breaker.call(blocking)

        release.set()
        assert await trial == "ok"

    async def test_caller_errors_do_not_trip_the_breaker(self) -> None:
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=2))

        async def invalid() -> str:
            raise ValidationError("bad input")

        for _ in range(5):
            with pytest.raises(ValidationError):
                await breaker.call(invalid)

        assert breaker.state is CircuitState.CLOSED, (
            "a 4xx-class error says nothing about dependency health"
        )

    async def test_rejections_are_counted(self) -> None:
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        async def failing() -> str:
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing)

        assert breaker.stats.total_rejections == 1

    async def test_reset_returns_the_breaker_to_closed(self) -> None:
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        async def failing() -> str:
            raise DatabaseError("down")

        with pytest.raises(DatabaseError):
            await breaker.call(failing)
        await breaker.reset()

        assert breaker.state is CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"failure_threshold": 0},
            {"recovery_timeout_seconds": 0},
            {"half_open_max_calls": 0},
            {"success_threshold": 0},
        ],
    )
    def test_invalid_breaker_configuration_is_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(**kwargs)


class TestResiliencePolicy:
    async def test_combines_timeout_retry_and_breaker(self) -> None:
        calls = 0

        async def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise DatabaseError("transient")
            return "ok"

        policy = ResiliencePolicy(
            timeout_seconds=1.0,
            retry=RetryPolicy(max_attempts=3, jitter=0.0),
            breaker=CircuitBreaker("test"),
        )

        assert await policy.execute(flaky, sleep=SleepRecorder()) == "ok"
        assert calls == 2

    async def test_timeout_applies_per_attempt(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1.0)
            return "never"

        policy = ResiliencePolicy(
            timeout_seconds=0.01, retry=RetryPolicy(max_attempts=2, jitter=0.0), breaker=None
        )

        with pytest.raises(OperationTimeoutError):
            await policy.execute(slow, sleep=SleepRecorder())

    async def test_open_breaker_short_circuits_the_retry_loop(self) -> None:
        calls = 0

        async def failing() -> str:
            nonlocal calls
            calls += 1
            raise DatabaseError("down")

        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))
        policy = ResiliencePolicy(
            timeout_seconds=1.0, retry=RetryPolicy(max_attempts=5, jitter=0.0), breaker=breaker
        )

        with pytest.raises((DatabaseError, CircuitOpenError)):
            await policy.execute(failing, sleep=SleepRecorder())

        assert calls == 1, "the breaker must stop the retry loop from hammering a dead dependency"

    async def test_works_without_a_breaker(self) -> None:
        async def operation() -> str:
            return "ok"

        policy = ResiliencePolicy(timeout_seconds=1.0, breaker=None)

        assert await policy.execute(operation) == "ok"
