"""LLM 工厂：preset → 配置 → 模型实例。

全系统唯一的模型出口。业务代码声明角色（preset），不声明具体模型——
换模型只改配置，不改代码、不重新部署。

职责边界：工厂只做"按配置创建实例"；超时/重试参数在此注入，
限流/熔断/应用层重试由 :class:`ResilientChatModel` 包装承担。
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from langgraphcode.common.llm.presets import LLMPreset
from langgraphcode.common.llm.resilience import CallGuard, ResilientChatModel
from langgraphcode.config.settings import ModelProfile, settings

# 每个 preset 共享一个守卫：限流与熔断状态必须在多次调用间累积，
# 若随模型实例创建则失去意义。
_GUARDS: dict[LLMPreset, CallGuard] = {}


def _resolve_profile(preset: LLMPreset) -> ModelProfile:
    """从全局配置解析 preset 对应的模型档位。"""
    return getattr(settings.llm, preset.value)


def _get_guard(preset: LLMPreset, profile: ModelProfile) -> CallGuard:
    """获取（或惰性创建）preset 共享的调用守卫。"""
    if preset not in _GUARDS:
        _GUARDS[preset] = CallGuard(
            name=preset.value,
            rpm=profile.rpm_limit,
            failure_threshold=settings.llm.breaker_failure_threshold,
            recovery_seconds=settings.llm.breaker_recovery_seconds,
            max_attempts=settings.llm.max_attempts,
            base_delay=settings.llm.retry_base_delay,
        )
    return _GUARDS[preset]


def create_llm(
    preset: LLMPreset = LLMPreset.AGENT,
    *,
    temperature: float | None = None,
    enable_resilience: bool | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """创建指定档位的聊天模型实例。

    :param preset: 模型档位（ROUTER / AGENT / WRITER）。
    :param temperature: 覆盖配置的温度；None 使用配置值。
    :param enable_resilience: 覆盖韧性开关；测试可传 False 拿裸模型。
    :param overrides: 透传给 ``init_chat_model`` 的其他参数。
    :return: 配置好的模型实例（默认带韧性包装）。
    """
    profile = _resolve_profile(preset)
    model = init_chat_model(
        model=profile.model,
        model_provider=profile.provider,
        base_url=profile.base_url,
        api_key=profile.api_key.get_secret_value(),
        temperature=profile.temperature if temperature is None else temperature,
        timeout=profile.timeout,
        max_retries=profile.max_retries,
        max_tokens=profile.max_tokens,
        **overrides,
    )

    resilience = settings.llm.enable_resilience if enable_resilience is None else enable_resilience
    if not resilience:
        return model
    return ResilientChatModel(bound=model, guard=_get_guard(preset, profile))
