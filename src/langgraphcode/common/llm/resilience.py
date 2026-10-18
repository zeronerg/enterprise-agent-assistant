"""LLM 调用韧性层：限流、熔断、应用层重试。

职责边界：

- 传输层韧性（timeout / max_retries）由工厂在创建模型时注入；
- 本模块提供**应用层**韧性：按 preset 共享的速率限制（保护供应商配额）、
  熔断器（故障供应商快速失败，避免拖垮整个编排图）、
  仅针对可重试错误的指数退避重试。

设计为**包装器**而非工厂内联：创建实例（factory）与调用策略（本层）
分离——测试可注入裸模型绕过本层，韧性参数可按 preset 差异化。
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ConfigDict

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """熔断器打开，调用被快速拒绝。"""


# 按类名后缀识别可重试错误：不同 SDK 的异常类不同（openai / anthropic / ...），
# 但限流、连接、超时、5xx 的命名高度收敛，后缀匹配是跨 provider 的务实做法。
_RETRYABLE_SUFFIXES: tuple[str, ...] = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "TimeoutException",
    "ConnectError",
    "ReadError",
)


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    429 限流、网络抖动、服务端 5xx 可重试；4xx 参数/认证错误立即失败——
    盲目重试会把配置错误放大成雪崩。

    :param exc: 待判断的异常。
    :return: 可重试返回 True。
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return any(exc.__class__.__name__.endswith(suffix) for suffix in _RETRYABLE_SUFFIXES)


class RateLimiter:
    """令牌桶限流器（按调用间隔实现），保护供应商 RPM 配额。

    同步/异步共用一把 ``threading.Lock``：临界区内只做纳秒级数值计算，
    不会阻塞事件循环；实际等待发生在锁外（sleep / asyncio.sleep）。
    """

    def __init__(self, rpm: int) -> None:
        """初始化限流器。

        :param rpm: 每分钟允许的调用次数。
        """
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _take(self) -> float:
        """分配下一个可用时间点，返回需要等待的时长（秒）。"""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        return max(0.0, slot - time.monotonic())

    def acquire(self) -> None:
        """同步获取一个调用许可。"""
        delay = self._take()
        if delay > 0:
            time.sleep(delay)

    async def aacquire(self) -> None:
        """异步获取一个调用许可。"""
        delay = self._take()
        if delay > 0:
            await asyncio.sleep(delay)


class _BreakerState(StrEnum):
    """熔断器三态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """按目标维度的熔断器。

    连续失败达到阈值后打开，拒绝调用一段时间；恢复期转半开放，
    放一次探测调用，成功则关闭，失败则重新计时。
    """

    def __init__(self, failure_threshold: int, recovery_seconds: float, name: str) -> None:
        """初始化熔断器。

        :param failure_threshold: 连续失败多少次后打开。
        :param recovery_seconds: 打开后多久进入半开放探测。
        :param name: 标识（用于日志与排错）。
        """
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._name = name
        self._lock = threading.Lock()
        self._state = _BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        """当前状态（只读，便于观测与测试断言）。"""
        return self._state.value

    def _before_request(self) -> None:
        """请求前检查：打开期快速拒绝；恢复期转半开放放行探测。"""
        with self._lock:
            if self._state is _BreakerState.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = _BreakerState.HALF_OPEN
                else:
                    raise CircuitOpenError(f"熔断器 [{self._name}] 已打开，调用被快速拒绝")

    def _on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = _BreakerState.CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state is _BreakerState.HALF_OPEN
                or self._consecutive_failures >= self._threshold
            ):
                self._state = _BreakerState.OPEN
                self._opened_at = time.monotonic()

    def call(self, fn: Callable[[], T]) -> T:
        """同步执行受熔断保护的调用。"""
        self._before_request()
        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    async def acall(self, fn: Callable[[], Awaitable[T]]) -> T:
        """异步执行受熔断保护的调用。"""
        self._before_request()
        try:
            result = await fn()
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result


def _retry_sync(  # noqa: UP047 - 与 CallGuard 的 TypeVar 风格保持一致
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_delay: float,
) -> T:
    """同步重试：指数退避 + 随机抖动，仅对可重试错误生效。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def _retry_async(  # noqa: UP047 - 与 CallGuard 的 TypeVar 风格保持一致
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
) -> T:
    """异步重试：指数退避 + 随机抖动，仅对可重试错误生效。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class CallGuard:
    """限流 + 熔断 + 重试的组合守卫。

    每个 preset 共享一个实例（工厂缓存），保证限流与熔断状态跨调用累积。
    """

    def __init__(
        self,
        name: str,
        rpm: int,
        *,
        failure_threshold: int,
        recovery_seconds: float,
        max_attempts: int,
        base_delay: float,
    ) -> None:
        """初始化守卫。

        :param name: 守卫标识（通常为 preset 名）。
        :param rpm: 每分钟调用上限。
        :param failure_threshold: 熔断连续失败阈值。
        :param recovery_seconds: 熔断恢复探测间隔（秒）。
        :param max_attempts: 应用层最大尝试次数（含首次）。
        :param base_delay: 重试基础延迟（秒）。
        """
        self._limiter = RateLimiter(rpm)
        self._breaker = CircuitBreaker(failure_threshold, recovery_seconds, name)
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    def call(self, fn: Callable[[], T]) -> T:
        """同步守卫：限流 → 熔断 → 重试。"""
        self._limiter.acquire()
        return self._breaker.call(
            lambda: _retry_sync(fn, max_attempts=self._max_attempts, base_delay=self._base_delay)
        )

    async def acall(self, fn: Callable[[], Awaitable[T]]) -> T:
        """异步守卫：限流 → 熔断 → 重试。"""
        await self._limiter.aacquire()
        return await self._breaker.acall(
            lambda: _retry_async(fn, max_attempts=self._max_attempts, base_delay=self._base_delay)
        )


class _ResilientRunnable(Runnable):
    """为任意 Runnable（如 ``with_structured_output`` 的结果）套上守卫。

    保留内层 Runnable 的配置语义（config_specs 透传），
    仅拦截 invoke/ainvoke 两个入口。
    """

    def __init__(self, inner: Runnable, guard: CallGuard) -> None:
        """初始化包装器。

        :param inner: 被包装的 Runnable。
        :param guard: 共享的调用守卫。
        """
        self._inner = inner
        self._guard = guard

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """同步调用，受守卫保护。"""
        return self._guard.call(lambda: self._inner.invoke(input, config=config, **kwargs))

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步调用，受守卫保护。"""
        return await self._guard.acall(
            lambda: self._inner.ainvoke(input, config=config, **kwargs)
        )

    @property
    def config_specs(self) -> list[Any]:
        """透传内层 Runnable 的配置规格。"""
        return self._inner.config_specs


class ResilientChatModel(BaseChatModel):
    """为 ChatModel 增加限流、熔断与应用层重试的透明包装器。

    对调用方完全透明：``invoke`` / ``ainvoke`` 受守卫保护；
    ``bind_tools`` 返回新的受保护模型；``with_structured_output``
    返回套了守卫的 Runnable。直接调用 ``generate`` 的路径
    （部分 Runnable 默认实现）也经由 ``_generate`` 兜底保护。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bound: BaseChatModel
    """被包装的原始模型。"""

    guard: CallGuard
    """共享的调用守卫。"""

    @property
    def _llm_type(self) -> str:
        return "resilient-chat-model"

    # ---- 主要入口：public invoke/ainvoke ----

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """同步调用，受守卫保护。"""
        return self.guard.call(
            lambda: self.bound.invoke(input, config=config, stop=stop, **kwargs)
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """异步调用，受守卫保护。"""
        return await self.guard.acall(
            lambda: self.bound.ainvoke(input, config=config, stop=stop, **kwargs)
        )

    # ---- 透明委托：工具绑定与结构化输出 ----

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> ResilientChatModel:
        """绑定工具：返回仍受守卫保护的新包装器。"""
        return ResilientChatModel(bound=self.bound.bind_tools(tools, **kwargs), guard=self.guard)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """结构化输出：内层逻辑不变，外层套守卫。"""
        inner = self.bound.with_structured_output(schema, **kwargs)
        return _ResilientRunnable(inner, self.guard)

    # ---- 兜底：直接走 generate 的代码路径 ----

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """BaseChatModel.generate 的底层入口，同样受守卫保护。"""
        def _do() -> ChatResult:
            result = self.bound.generate([messages], stop=stop, **kwargs)
            generation = result.generations[0][0]
            if not isinstance(generation, ChatGeneration):
                raise TypeError("底层模型返回了非 ChatGeneration 结果")
            return ChatResult(generations=[generation], llm_output=result.llm_output)

        return self.guard.call(_do)

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """BaseChatModel.agenerate 的底层入口，同样受守卫保护。"""
        async def _do() -> ChatResult:
            result = await self.bound.agenerate([messages], stop=stop, **kwargs)
            generation = result.generations[0][0]
            if not isinstance(generation, ChatGeneration):
                raise TypeError("底层模型返回了非 ChatGeneration 结果")
            return ChatResult(generations=[generation], llm_output=result.llm_output)

        return await self.guard.acall(_do)
