"""A2A Server 启动器。

提供将 BaseA2AAgent 实例启动为 A2A Server 的通用函数。
每个 Agent 独立运行在自己的端口上，通过 AgentCard 对外声明能力。

create_a2a_app 与 run_agent_server 分离（关注点分离）：
测试可以只创建 ASGI 应用而不真正起端口。
"""

from __future__ import annotations

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from langgraphcode.a2a_agents.base_agent import BaseA2AAgent, BaseAgentExecutor
from langgraphcode.common.logging import get_logger

logger = get_logger(__name__)


def create_a2a_app(agent: BaseA2AAgent) -> A2AStarletteApplication:
    """为 A2A Agent 创建 Starlette 应用。

    :param agent: 要启动的 A2A Agent 实例。
    :return: 配置好的 A2A Starlette 应用。
    """
    agent_card = agent.to_agent_card()
    executor = BaseAgentExecutor(agent)
    task_store = InMemoryTaskStore()

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


def run_agent_server(agent: BaseA2AAgent, host: str = "0.0.0.0") -> None:
    """启动 A2A Agent Server（阻塞运行）。

    :param agent: 要启动的 A2A Agent 实例。
    :param host: 监听地址，默认 0.0.0.0。
    """
    app = create_a2a_app(agent)
    logger.info("启动 Agent Server", agent=agent.name, host=host, port=agent.port)
    uvicorn.run(app.build(), host=host, port=agent.port, log_level="info")
