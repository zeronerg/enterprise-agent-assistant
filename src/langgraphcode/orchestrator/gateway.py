"""Agent 网关：编排层与专业 Agent 之间的调用抽象。

编排节点只依赖 :class:`AgentGateway` 协议，两种实现可自由切换：

- :class:`InlineGateway`：进程内用 LLM 模拟各 Agent 的行为，
  保证项目零依赖启动即可运行，也用于集成测试；
- :class:`A2AGateway`：通过 A2A 协议调用远程 Agent Server
  （需先 ``serve`` 启动各 Agent），是生产环境的完整形态。

切换由配置 ``APP_ORCHESTRATOR__USE_REAL_AGENTS`` 驱动，
编排节点代码零改动——依赖注入取代模块级单例。
"""

from __future__ import annotations

import uuid
from typing import Protocol

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from langgraphcode.common.llm import LLMPreset, create_llm
from langgraphcode.common.logging import get_logger
from langgraphcode.config.settings import settings

logger = get_logger(__name__)

# 意图 → Agent Server 端点的映射（数据驱动的注册表，而非 if-elif 链）。
INTENT_AGENT_URLS: dict[str, str] = {
    "data_query": settings.a2a.data_agent_url,
    "ticket": settings.a2a.ticket_agent_url,
    "report": settings.a2a.report_agent_url,
}

# InlineGateway 使用的各 Agent 角色提示词（与对应 A2A Agent 的
# 系统提示词语义对齐，保证两种实现行为一致）。
AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "data_query": (
        "你是一个企业数据查询助手。根据用户的查询请求，"
        "提供相关的数据分析和统计信息。"
        "如果用户指定了月份，使用该月份的数据。"
        "数据展示要清晰，使用表格或列表格式。"
    ),
    "ticket": (
        "你是一个 IT 工单处理助手。根据用户描述的问题，"
        "提供解决方案或排查步骤。"
        "回答要具体、可操作，包含明确的步骤编号。"
    ),
    "report": (
        "你是一个报告撰写助手。根据用户的报告请求，"
        "生成结构化的 Markdown 格式文档。"
        "包含标题、摘要、正文和结论。"
    ),
}

DEFAULT_INTENT = "data_query"


class AgentGateway(Protocol):
    """Agent 调用协议：intent + query → 处理结果文本。"""

    async def dispatch(self, intent: str, query: str) -> str:
        """把查询分发给对应意图的专业 Agent。

        :param intent: 意图类别（data_query / ticket / report）。
        :param query: 用户原始请求。
        :return: Agent 处理结果文本。
        """
        ...


class InlineGateway:
    """进程内 LLM 模拟实现。

    用于独立演示（不启动 A2A Server 也能跑通编排流程）与集成测试。
    """

    async def dispatch(self, intent: str, query: str) -> str:
        """在进程内用 LLM 按角色提示词模拟对应 Agent 的回答。

        :param intent: 意图类别。
        :param query: 用户原始请求。
        :return: 模拟的 Agent 处理结果。
        """
        logger.info("InlineGateway 分发（进程内模拟）", intent=intent)
        llm = create_llm(LLMPreset.AGENT)
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=AGENT_SYSTEM_PROMPTS.get(intent, AGENT_SYSTEM_PROMPTS[DEFAULT_INTENT])
                ),
                HumanMessage(content=query),
            ]
        )
        return response.content


class A2AGateway:
    """通过 A2A 协议调用远程 Agent Server。

    真实 A2A 调用流程：AgentCard 发现（/.well-known/agent.json）→
    构造 Message 请求 → send_message → 从响应中提取最终 artifact。
    """

    def __init__(self, timeout: float | None = None) -> None:
        """初始化网关。

        :param timeout: 调用远程 Agent 的超时（秒），None 使用配置值。
        """
        self._timeout = timeout or settings.a2a.request_timeout

    async def dispatch(self, intent: str, query: str) -> str:
        """向对应意图的远程 Agent 发送 A2A 请求并提取结果。

        :param intent: 意图类别。
        :param query: 用户原始请求。
        :return: 远程 Agent 返回的处理结果文本。
        :raises ValueError: 意图无对应 Agent 端点。
        :raises RuntimeError: Agent 任务失败或无结果。
        """
        from a2a.client import A2AClient, A2ACardResolver
        from a2a.types import Message, MessageSendParams, Part, Role, Task, TaskState, TextPart

        base_url = INTENT_AGENT_URLS[intent]
        logger.info("A2AGateway 分发", intent=intent, target=base_url)

        async with httpx.AsyncClient(timeout=self._timeout) as httpx_client:
            resolver = A2ACardResolver(httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client, agent_card=agent_card)

            response: Message | Task = await client.send_message(
                MessageSendParams(
                    message=Message(
                        role=Role.user,
                        parts=[Part(root=TextPart(text=query))],
                        message_id=uuid.uuid4().hex,
                    )
                )
            )

        return _extract_response_text(response)


def _extract_response_text(response: "Message | Task") -> str:
    """从 A2A 响应（Message 或最终态 Task）中提取文本结果。

    :param response: A2A send_message 的返回。
    :return: 拼接后的文本内容。
    :raises RuntimeError: 任务失败或没有可提取的内容。
    """
    from a2a.types import Message, Task, TaskState, TextPart

    if isinstance(response, Message):
        return _parts_to_text(response.parts)

    if response.status.state == TaskState.failed:
        raise RuntimeError(f"A2A 任务执行失败：{response.status.message or '无错误详情'}")
    if not response.artifacts:
        raise RuntimeError("A2A 任务完成但没有返回 artifact")
    return _parts_to_text(response.artifacts[-1].parts)


def _parts_to_text(parts: list) -> str:
    """提取 Part 列表中的全部文本并拼接。"""
    from a2a.types import TextPart

    return "".join(
        part.root.text for part in parts if isinstance(part.root, TextPart)
    )


def create_gateway() -> AgentGateway:
    """按配置创建网关实现。

    :return: ``use_real_agents=True`` 时返回 A2AGateway，否则 InlineGateway。
    """
    if settings.orchestrator.use_real_agents:
        return A2AGateway()
    return InlineGateway()
