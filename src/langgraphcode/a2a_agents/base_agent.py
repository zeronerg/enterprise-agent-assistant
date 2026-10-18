"""A2A Agent 基类与执行器。

封装 A2A Agent 的通用逻辑：AgentCard 定义、任务执行模板、
LLM 调用和 MCP 工具加载。子类只需实现 :meth:`get_skills` 和
:meth:`get_system_prompt` 两个抽象方法，即可获得完整的 A2A Agent 能力
（模板方法模式——基类冻结流程骨架，子类只提供个性）。

执行器把业务逻辑适配到 A2A 协议的 AgentExecutor 接口，
并负责在任务入口绑定日志上下文（task_id / agent），
保证一次任务的所有日志可按 task_id 过滤串联。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

from langgraphcode.common.context import bind_request_context, clear_request_context, new_trace_id
from langgraphcode.common.llm import LLMPreset, create_llm
from langgraphcode.common.logging import get_logger

logger = get_logger(__name__)


class BaseA2AAgent(ABC):
    """A2A Agent 抽象基类。

    Attributes:
        name: Agent 名称。
        description: Agent 功能描述。
        port: Agent 监听端口。
        mcp_server_paths: MCP Server 脚本路径列表，为空表示不加载 MCP 工具。
    """

    def __init__(
        self,
        name: str,
        description: str,
        port: int,
        mcp_server_paths: list[str] | None = None,
    ) -> None:
        """初始化 A2A Agent。

        :param name: Agent 名称。
        :param description: Agent 功能描述。
        :param port: Agent 监听端口。
        :param mcp_server_paths: MCP Server 脚本路径列表。
        """
        self.name = name
        self.description = description
        self.port = port
        self.mcp_server_paths = mcp_server_paths or []

    @abstractmethod
    def get_skills(self) -> list[AgentSkill]:
        """返回 Agent 的技能列表（用于构建 AgentCard）。"""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """返回 Agent 的系统提示词。"""

    async def _load_mcp_tools(self) -> list[Any]:
        """加载所有 MCP Server 提供的工具。

        通过 LangChain 的 MCPAdapter 以 stdio 连接每个 MCP Server，
        发现工具并适配为 LangChain 工具。

        .. note::
            stdio 传输意味着每次任务都 fork 新的子进程，适合演示与低频
            调用；生产高频场景应让 MCP Server 常驻（HTTP 传输）后改为
            ``MCPAdapter("http://...")`` 并复用连接。

        :return: LangChain 工具列表。
        """
        all_tools: list[Any] = []
        for server_path in self.mcp_server_paths:
            async with MCPAdapter(f"stdio:{server_path}") as adapter:
                tools = await adapter.list_tools()
                all_tools.extend(tools)
                logger.info(
                    "从 MCP Server 加载了工具",
                    server=server_path,
                    tool_count=len(tools),
                )
        return all_tools

    async def execute_task(self, query: str) -> str:
        """执行任务的核心逻辑。

        有 MCP 工具时创建带工具的 Agent 走 ReAct 循环；
        无工具时直接调用 LLM（如报告生成 Agent）。

        :param query: 用户查询或任务描述。
        :return: Agent 处理结果文本。
        """
        llm = create_llm(LLMPreset.AGENT)

        if self.mcp_server_paths:
            tools = await self._load_mcp_tools()
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=self.get_system_prompt(),
            )
            result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            last_message = result["messages"][-1]
            # 鸭子类型优先：只要有 content 属性就使用它，不依赖具体消息类，
            # 对 LangChain 内部消息类型的演进更健壮。
            return last_message.content if hasattr(last_message, "content") else str(last_message)

        response = await llm.ainvoke(
            [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
            ]
        )
        return response.content

    def to_agent_card(self, host: str = "localhost") -> AgentCard:
        """构建 A2A AgentCard——Agent 的"名片"。

        通过 ``/.well-known/agent.json`` 端点暴露，让其他 Agent
        （包括编排层）能够发现本 Agent 的能力。

        :param host: Agent 主机地址。
        :return: 描述本 Agent 能力的 AgentCard 实例。
        """
        return AgentCard(
            name=self.name,
            description=self.description,
            url=f"http://{host}:{self.port}/",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=self.get_skills(),
            default_input_modes=["text"],
            default_output_modes=["text"],
        )


class BaseAgentExecutor(AgentExecutor):
    """A2A Agent 执行器。

    遵循 A2A 标准任务生命周期：submit → execute → add_artifact →
    complete / failed，并用 TaskUpdater 把每一步状态推给客户端。

    Attributes:
        agent: 具体的 A2A Agent 实例。
    """

    def __init__(self, agent: BaseA2AAgent) -> None:
        """初始化执行器。

        :param agent: 要执行的 A2A Agent 实例。
        """
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """处理传入的 A2A 请求。

        :param context: 请求上下文，包含用户消息和任务信息。
        :param event_queue: 事件队列，用于发送状态更新和结果。
        """
        task_id = context.task_id or new_trace_id()
        # 绑定日志上下文：该任务整条调用链的日志自动携带 task_id / agent。
        bind_request_context(task_id=task_id, agent=self.agent.name)
        try:
            updater = TaskUpdater(event_queue, task_id, context.context_id)
            await updater.submit()

            try:
                query = context.get_user_input()
                logger.info("收到任务", query=query[:100])

                result = await self.agent.execute_task(query)

                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=result))],
                    name=f"{self.agent.name}_result",
                )
                await updater.complete()
                logger.info("任务完成")
            except Exception as exc:
                logger.error("任务执行失败", error=str(exc))
                await updater.failed()
        finally:
            clear_request_context()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """取消正在执行的任务。

        :param context: 请求上下文。
        :param event_queue: 事件队列。
        """
        task_id = context.task_id or new_trace_id()
        updater = TaskUpdater(event_queue, task_id, context.context_id)
        await updater.cancel()
