"""主编排图构建与编译。

将 LangGraph 的编排能力、Agent 网关、LLM 工厂整合为统一工作流：

    START → classify → dispatch → format → END
                 │          │
                 │    （条件边：有结果→format，无结果→handle_error）
                 ▼
           handle_error → END

Checkpointer 与 Gateway 均为注入参数：测试用内存实现，
生产替换为 PostgresSaver / A2AGateway 而图构建逻辑不变。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraphcode.common.logging import get_logger
from langgraphcode.orchestrator.gateway import AgentGateway, InlineGateway
from langgraphcode.orchestrator.nodes import (
    classify_intent,
    format_response,
    handle_error,
    make_dispatch_node,
)
from langgraphcode.orchestrator.state import OrchestratorState

logger = get_logger(__name__)


def build_orchestrator_graph(
    gateway: AgentGateway | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建并编译企业助手编排图。

    :param gateway: Agent 调用网关；None 时使用 InlineGateway（进程内模拟）。
    :param checkpointer: 状态持久化器；None 时使用 MemorySaver。
        生产环境应替换为 PostgresSaver——本函数不感知具体实现。
    :return: 编译后的图实例，支持 invoke / ainvoke。
    """
    gateway = gateway or InlineGateway()
    checkpointer = checkpointer or MemorySaver()

    graph = StateGraph(OrchestratorState)

    # ---- 注册节点 ----
    graph.add_node("classify", classify_intent)
    graph.add_node("dispatch", make_dispatch_node(gateway))
    graph.add_node("format", format_response)
    graph.add_node("handle_error", handle_error)

    # ---- 配置边 ----
    graph.add_edge(START, "classify")

    # classify 用 Command 自路由，不需要显式条件边；
    # dispatch 之后判断结果有效性，纯路由用条件边更轻。
    graph.add_conditional_edges(
        "dispatch",
        lambda state: "format" if state.get("agent_result") else "handle_error",
        {"format": "format", "handle_error": "handle_error"},
    )

    graph.add_edge("format", END)
    graph.add_edge("handle_error", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("编排图已编译", gateway=type(gateway).__name__)
    return compiled
