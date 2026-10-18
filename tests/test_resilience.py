# tests/test_resilience.py
"""韧性层单元测试：限流、熔断、重试语义。

不依赖任何 LLM 与网络，全部可在离线环境运行。
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from langgraphcode.common.llm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    is_retryable,
)


class TestIsRetryable:
    """可重试错误分类。"""

    def test_network_errors_are_retryable(self) -> None:
        assert is_retryable(httpx.ConnectError("boom"))
        assert is_retryable(httpx.ReadTimeout("slow"))

    def test_sdk_style_errors_are_retryable(self) -> None:
        class FakeRateLimitError(Exception):
            pass

        assert is_retryable(FakeRateLimitError())

    def test_client_errors_are_not_retryable(self) -> None:
        with pytest.raises(ValueError):
            raise ValueError("bad request")

        assert not is_retryable(ValueError("bad request"))
        assert not is_retryable(PermissionError("invalid api key"))


class TestRateLimiter:
    """限流器按间隔放行。"""

    def test_high_rpm_does_not_block(self) -> None:
        limiter = RateLimiter(rpm=60000)  # 每分钟 6 万次 ≈ 无限制
        start = time.monotonic()
        for _ in range(10):
            limiter.acquire()
        assert time.monotonic() - start < 0.5

    def test_low_rpm_throttles(self) -> None:
        limiter = RateLimiter(rpm=30)  # 每次调用间隔 2 秒
        limiter.acquire()
        start = time.monotonic()
        limiter.acquire()
        assert time.monotonic() - start >= 1.9

    def test_async_acquire(self) -> None:
        async def run() -> float:
            limiter = RateLimiter(rpm=60000)
            start = time.monotonic()
            await limiter.aacquire()
            return time.monotonic() - start

        assert asyncio.run(run()) < 0.5


class TestCircuitBreaker:
    """熔断器三态转换。"""

    def _make_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(failure_threshold=2, recovery_seconds=0.1, name="test")

    def test_opens_after_threshold(self) -> None:
        breaker = self._make_breaker()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        assert breaker.state == "open"
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "ok")

    def test_recovers_after_recovery_window(self) -> None:
        breaker = self._make_breaker()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        time.sleep(0.15)  # 超过 recovery_seconds，进入半开放探测
        assert breaker.call(lambda: "ok") == "ok"
        assert breaker.state == "closed"

    def test_success_resets_failure_count(self) -> None:
        breaker = self._make_breaker()
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        assert breaker.call(lambda: "ok") == "ok"
        # 失败后成功一次，连续失败计数清零，还需 2 次失败才熔断
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        assert breaker.state == "closed"

    def test_async_path(self) -> None:
        async def run() -> None:
            breaker = self._make_breaker()

            async def fail() -> str:
                raise RuntimeError("down")

            for _ in range(2):
                with pytest.raises(RuntimeError):
                    await breaker.acall(fail)
            assert breaker.state == "open"

        asyncio.run(run())
