"""编排节点函数。

包含编排图中的所有节点实现：意图分类、Agent 分发、结果格式化。
每个节点接收状态，返回状态更新字典或 Command 对象。

路由机制的选择原则：

- :func:`classify_intent` 用 ``Command(update=..., goto=...)``——
  它既要更新状态（intent）又要决定路由，两者合并在一个返回值中，
  路由逻辑内聚在节点函数内（LangGraph 推荐的单 Agent 路由模式）；
- dispatch 之后用条件边判断结果有效性——纯路由、不更新状态，
  条件边更轻。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from langgraphcode.common.context import bind_request_context
from langgraphcode.common.llm import LLMPreset, create_llm
from langgraphcode.common.logging import get_logger
from langgraphcode.orchestrator.gateway import AgentGateway
from langgraphcode.orchestrator.state import OrchestratorState

logger = get_logger(__name__)

VALID_INTENTS: tuple[str, ...] = ("data_query", "ticket", "report")


class IntentClassification(BaseModel):
    """意图分类的结构化输出。

    Attributes:
        intent: 意图类别。
        confidence: 分类置信度（0 到 1）。
        reason: 分类理由。
    """

    intent: str = Field(description="意图类别，必须是 data_query、ticket、report 之一")
    confidence: float = Field(description="分类置信度", ge=0.0, le=1.0)
    reason: str = Field(description="分类理由，一句话说明")


def classify_intent(
    state: OrchestratorState,
) -> Command[Literal["dispatch", "handle_error"]]:
    """对用户请求进行意图分类，并通过 Command 路由到下一个节点。

    双重防御：Pydantic schema 约束是第一道防线，
    代码兜底（非法意图回退）是第二道——LLM 的输出永远不可全信。

    :param state: 当前编排状态，包含 query 字段。
    :return: 包含状态更新和路由目标的 Command 对象。
    """
    query = state["query"]
    logger.info("意图分类", query=query[:100])

    llm = create_llm(LLMPreset.ROUTER)
    structured_llm = llm.with_structured_output(IntentClassification)

    result = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是一个意图分类器。根据用户请求判断属于以下哪一类：\n"
                    "- data_query：查询数据、统计信息、客户信息、销售数据\n"
                    "- ticket：报告问题、故障、需要 IT 支持、提交工单\n"
                    "- report：写报告、生成文档、总结分析\n"
                    "只输出类别名称，不要输出其他内容。"
                )
            ),
            HumanMessage(content=query),
        ]
    )

    intent = result.intent
    if intent not in VALID_INTENTS:
        logger.warning("LLM 返回了无效意图，回退到 data_query", raw_intent=intent)
        intent = "data_query"

    # 分类结果写入日志上下文，后续节点的日志自动携带 intent 字段。
    bind_request_context(intent=intent)
    logger.info("分类完成", intent=intent, confidence=result.confidence, reason=result.reason)

    return Command(update={"intent": intent}, goto="dispatch")


def make_dispatch_node(
    gateway: AgentGateway,
) -> Callable[[OrchestratorState], Awaitable[dict[str, str]]]:
    """创建 dispatch 节点：路由决策不变，Agent 调用委托给注入的网关。

    依赖注入而非模块级单例：测试可传入 FakeGateway，
    生产通过 create_gateway() 选择 A2A 或进程内实现，
    节点函数本身不感知下游形态。

    :param gateway: 实现了 AgentGateway 协议的网关实例。
    :return: 符合 LangGraph 节点签名的异步函数。
    """

    async def dispatch_to_agent(state: OrchestratorState) -> dict[str, str]:
        """根据意图分发到对应的处理 Agent。

        :param state: 当前编排状态，包含 intent 和 query 字段。
        :return: 包含 agent_result 的更新字典。
        """
        intent = state["intent"]
        query = state["query"]
        logger.info("分发到 Agent", intent=intent)

        result = await gateway.dispatch(intent, query)
        return {"agent_result": result}

    return dispatch_to_agent


def format_response(state: OrchestratorState) -> dict[str, Any]:
    """格式化最终响应：包装 Agent 结果并追加到消息历史。

    :param state: 当前编排状态，包含 agent_result 字段。
    :return: 包含 final_response 和 messages 的更新字典。
    """
    result = state.get("agent_result") or "抱歉，处理您的请求时出现了问题。"
    intent = state.get("intent", "unknown")

    logger.info("格式化响应完成", intent=intent)
    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
    }


def handle_error(state: OrchestratorState) -> dict[str, Any]:
    """错误处理节点：把异常路径收敛为统一的用户可读回复。

    :param state: 当前编排状态。
    :return: 包含 final_response 和 messages 的更新字典。
    """
    error = state.get("error") or "未知错误"
    message = f"系统处理您的请求时遇到问题：{error}\n请稍后重试或联系管理员。"

    logger.error("进入错误处理节点", error=error)
    return {
        "final_response": message,
        "messages": [AIMessage(content=message)],
    }
