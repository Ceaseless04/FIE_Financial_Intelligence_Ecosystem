"""Resilience primitives: retry with backoff, circuit breaking, and timeouts.

Phase 8 of the platform plan requires that a failure in one service cannot
cascade through the ecosystem. These helpers are the shared mechanism: every
outbound call (AI provider, database, event bus, service-to-service) is wrapped
in a timeout, a bounded retry, and a circuit breaker.

The clock, sleep, and randomness are injectable so tests assert exact backoff
schedules and state transitions instead of sleeping in real time.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from fie_common.errors import CircuitOpenError, FIEError, OperationTimeoutError, RateLimitError

T = TypeVar("T")

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]
RandFn = Callable[[], float]


def default_retryable(error: BaseException) -> bool:
    """Retry FIE errors flagged retryable, plus raw connection/timeout errors."""
    if isinstance(error, FIEError):
        return error.retryable
    return isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError))


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with optional jitter.

    Attributes:
        max_attempts: Total attempts including the first. ``1`` disables retry.
        initial_backoff_seconds: Delay before the second attempt.
        max_backoff_seconds: Ceiling applied to every computed delay.
        multiplier: Growth factor between successive attempts.
        jitter: Fraction of the delay randomized (``0.0`` = deterministic).
        respect_retry_after: Honor ``RateLimitError.retry_after_seconds`` when
            the provider tells us how long to wait.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 10.0
    multiplier: float = 2.0
    jitter: float = 0.1
    respect_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must not be negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be between 0.0 and 1.0")

    def backoff_for(self, attempt: int, *, rand: RandFn = random.random) -> float:
        """Delay in seconds before ``attempt`` (1-based; attempt 1 has no delay)."""
        if attempt <= 1:
            return 0.0
        raw = self.initial_backoff_seconds * (self.multiplier ** (attempt - 2))
        delay = min(raw, self.max_backoff_seconds)
        if self.jitter:
            delay *= 1.0 + self.jitter * (2.0 * rand() - 1.0)
        return max(0.0, delay)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    retryable: Callable[[BaseException], bool] = default_retryable,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: SleepFn = asyncio.sleep,
    rand: RandFn = random.random,
) -> T:
    """Run ``operation``, retrying failures the ``retryable`` predicate accepts.

    Non-retryable errors propagate immediately — a 401 or a schema violation
    must not be hammered. The final failure is re-raised with its original
    traceback once attempts are exhausted.
    """
    policy = policy or RetryPolicy()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except BaseException as error:
            if not retryable(error) or attempt == policy.max_attempts:
                raise
            last_error = error

            delay = policy.backoff_for(attempt + 1, rand=rand)
            if (
                policy.respect_retry_after
                and isinstance(error, RateLimitError)
                and error.retry_after_seconds is not None
            ):
                delay = max(delay, error.retry_after_seconds)

            if on_retry is not None:
                on_retry(attempt, error, delay)
            if delay > 0:
                await sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error if last_error else RuntimeError("retry_async exited without result")


async def with_timeout(
    operation: Callable[[], Awaitable[T]],
    *,
    seconds: float,
    description: str = "operation",
) -> T:
    """Await ``operation`` with a deadline, normalizing to ``OperationTimeoutError``."""
    if seconds <= 0:
        raise ValueError("timeout seconds must be positive")
    try:
        return await asyncio.wait_for(operation(), timeout=seconds)
    except TimeoutError as exc:
        raise OperationTimeoutError(
            f"{description} timed out after {seconds}s",
            details={"timeout_seconds": seconds, "operation": description},
            cause=exc,
        ) from exc


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    success_threshold: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be positive")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")


@dataclass
class _CircuitStats:
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    half_open_calls: int = 0
    opened_at: float | None = None
    total_rejections: int = 0


class CircuitBreaker:
    """Per-dependency circuit breaker.

    Closed -> Open after ``failure_threshold`` consecutive failures.
    Open -> Half-open once ``recovery_timeout_seconds`` elapses; half-open
    admits at most ``half_open_max_calls`` trial calls. A trial success closes
    the circuit, a trial failure re-opens it and restarts the recovery clock.
    """

    def __init__(
        self,
        name: str,
        *,
        config: CircuitBreakerConfig | None = None,
        clock: ClockFn = time.monotonic,
        failure_predicate: Callable[[BaseException], bool] = default_retryable,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._clock = clock
        self._is_failure = failure_predicate
        self._state = CircuitState.CLOSED
        self._stats = _CircuitStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for an elapsed recovery window."""
        if self._state is CircuitState.OPEN and self._recovery_elapsed():
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def stats(self) -> _CircuitStats:
        return self._stats

    def _recovery_elapsed(self) -> bool:
        opened_at = self._stats.opened_at
        if opened_at is None:
            return False
        return (self._clock() - opened_at) >= self.config.recovery_timeout_seconds

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute ``operation`` under the breaker.

        Raises:
            CircuitOpenError: if the circuit rejects the call without running it.
        """
        await self._before_call()
        try:
            result = await operation()
        except BaseException as error:
            if self._is_failure(error):
                await self._record_failure()
            else:
                # A caller error (bad input, auth) says nothing about dependency
                # health — do not let it trip the breaker.
                await self._record_success()
            raise
        else:
            await self._record_success()
            return result

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                if not self._recovery_elapsed():
                    self._stats.total_rejections += 1
                    raise CircuitOpenError(
                        f"circuit '{self.name}' is open",
                        details={
                            "circuit": self.name,
                            "consecutive_failures": self._stats.consecutive_failures,
                            "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                        },
                    )
                self._transition_to_half_open()

            if self._state is CircuitState.HALF_OPEN:
                if self._stats.half_open_calls >= self.config.half_open_max_calls:
                    self._stats.total_rejections += 1
                    raise CircuitOpenError(
                        f"circuit '{self.name}' is half-open and at trial capacity",
                        details={"circuit": self.name, "state": str(CircuitState.HALF_OPEN)},
                    )
                self._stats.half_open_calls += 1

    def _transition_to_half_open(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._stats.half_open_calls = 0
        self._stats.consecutive_successes = 0

    async def _record_success(self) -> None:
        async with self._lock:
            self._stats.consecutive_failures = 0
            if self._state is CircuitState.HALF_OPEN:
                self._stats.consecutive_successes += 1
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._close()
            else:
                self._close()

    async def _record_failure(self) -> None:
        async with self._lock:
            self._stats.consecutive_successes = 0
            if self._state is CircuitState.HALF_OPEN:
                self._open()
                return
            self._stats.consecutive_failures += 1
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._open()

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._stats.consecutive_failures = 0
        self._stats.half_open_calls = 0
        self._stats.opened_at = None

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._stats.opened_at = self._clock()
        self._stats.half_open_calls = 0
        self._stats.consecutive_successes = 0

    async def reset(self) -> None:
        """Force the circuit closed. Intended for tests and operator tooling."""
        async with self._lock:
            self._stats = _CircuitStats()
            self._close()


@dataclass
class ResiliencePolicy:
    """Bundle of timeout + retry + circuit breaker applied to one dependency."""

    timeout_seconds: float = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker | None = None

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        description: str = "operation",
        sleep: SleepFn = asyncio.sleep,
        rand: RandFn = random.random,
    ) -> T:
        """Run ``operation`` under the full policy.

        Ordering matters: the breaker wraps each individual attempt (so a trial
        call in half-open state is a single attempt, not a full retry budget),
        and the timeout applies per attempt rather than to the whole retry loop.
        """

        async def attempt() -> T:
            async def timed() -> T:
                return await with_timeout(
                    operation, seconds=self.timeout_seconds, description=description
                )

            if self.breaker is not None:
                return await self.breaker.call(timed)
            return await timed()

        return await retry_async(attempt, policy=self.retry, sleep=sleep, rand=rand)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ResiliencePolicy",
    "RetryPolicy",
    "default_retryable",
    "retry_async",
    "with_timeout",
]
