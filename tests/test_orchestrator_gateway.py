# tests/test_orchestrator_gateway.py
"""编排层节点测试：dispatch 节点的依赖注入与路由语义。

用 FakeGateway 替代真实网关，验证节点逻辑不依赖 LLM 与网络——
这正是 make_dispatch_node(gateway) 依赖注入设计的可测试性回报。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraphcode.orchestrator.gateway import (
    AGENT_SYSTEM_PROMPTS,
    INTENT_AGENT_URLS,
    InlineGateway,
)
from langgraphcode.orchestrator.nodes import format_response, handle_error, make_dispatch_node
from langgraphcode.orchestrator.state import OrchestratorState


class FakeGateway:
    """测试替身：按意图返回确定性结果，并记录调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def dispatch(self, intent: str, query: str) -> str:
        self.calls.append((intent, query))
        return f"fake-result-for-{intent}"


def _state(**overrides: Any) -> OrchestratorState:
    base: OrchestratorState = {
        "query": "测试请求",
        "messages": [],
        "intent": "data_query",
        "agent_result": "",
        "final_response": "",
        "error": "",
    }
    base.update(overrides)
    return base


class TestDispatchNode:
    """分发节点的注入与状态更新语义。"""

    def test_dispatch_writes_agent_result(self) -> None:
        fake = FakeGateway()
        node = make_dispatch_node(fake)

        result = asyncio.run(node(_state(intent="ticket", query="VPN 连不上")))

        assert result == {"agent_result": "fake-result-for-ticket"}
        assert fake.calls == [("ticket", "VPN 连不上")]

    def test_empty_result_triggers_error_branch(self) -> None:
        """条件边的判断依据：空 agent_result → handle_error。"""
        empty = FakeGateway()
        empty.dispatch = lambda intent, query: ""  # type: ignore[assignment]
        node = make_dispatch_node(empty)  # type: ignore[arg-type]

        result = asyncio.run(node(_state()))
        # 空字符串是假值 → 图的条件边会路由到 handle_error
        assert result == {"agent_result": ""}


class TestFormatAndErrorNodes:
    """结果格式化与错误收敛。"""

    def test_format_response_passthrough(self) -> None:
        result = format_response(_state(agent_result="好的", intent="report"))
        assert result["final_response"] == "好的"
        assert len(result["messages"]) == 1

    def test_format_response_empty_result_falls_back(self) -> None:
        result = format_response(_state(agent_result=""))
        assert "抱歉" in result["final_response"]

    def test_handle_error_produces_user_facing_message(self) -> None:
        result = handle_error(_state(error="网关超时"))
        assert "网关超时" in result["final_response"]
        assert "联系管理员" in result["final_response"]


class TestGatewayRegistry:
    """注册表数据完整性：每个意图都有端点与提示词。"""

    def test_intent_urls_cover_all_valid_intents(self) -> None:
        assert set(INTENT_AGENT_URLS) == {"data_query", "ticket", "report"}

    def test_prompts_cover_all_valid_intents(self) -> None:
        assert set(AGENT_SYSTEM_PROMPTS) == {"data_query", "ticket", "report"}

    def test_inline_gateway_is_default_constructible(self) -> None:
        gateway = InlineGateway()
        assert hasattr(gateway, "dispatch")
