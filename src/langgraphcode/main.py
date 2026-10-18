"""智能企业助手系统入口。

演示完整的请求处理流程：
用户输入 → LangGraph 编排 → 意图分类 → Agent 分发 → 结果返回。

三种运行模式：

- ``demo``：跑编排器演示（三种意图各一个请求）；
- ``agent``：独立运行一个 A2A Agent 进行测试（绕过 HTTP 验证业务逻辑）；
- ``serve``：启动 A2A Agent Server（生产协作形态）。
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
from typing import Any

from langgraphcode.common.context import bind_request_context, clear_request_context, new_trace_id
from langgraphcode.common.logging import get_logger, setup_logging
from langgraphcode.orchestrator.gateway import create_gateway
from langgraphcode.orchestrator.graph import build_orchestrator_graph
from langgraphcode.orchestrator.state import OrchestratorState

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Agent 注册表：新增 Agent 只需加一行，消灭 if-elif 链；
# 值为 "模块路径:类名" 字符串，配合 importlib 延迟导入——
# 不启动 Agent 时不加载对应模块，避免启动开销与循环导入。
# ---------------------------------------------------------------------------

_AGENT_LOADERS: dict[str, str] = {
    "data": "langgraphcode.a2a_agents.data_agent:DataAgent",
    "ticket": "langgraphcode.a2a_agents.ticket_agent:TicketAgent",
    "report": "langgraphcode.a2a_agents.report_agent:ReportAgent",
}

_DEMO_QUERIES: list[str] = [
    "帮我查一下 2025 年 3 月华东地区的销售数据",
    "VPN 连接不上了，怎么排查？",
    "帮我写一份关于 AI Agent 技术趋势的技术简报",
]


def _load_agent(agent_type: str) -> type:
    """动态加载指定类型的 A2A Agent 类。

    :param agent_type: Agent 类型（data/ticket/report）。
    :return: 未实例化的 Agent 类。
    :raises ValueError: 当 agent_type 不在注册表中。
    """
    if agent_type not in _AGENT_LOADERS:
        raise ValueError(
            f"未知的 Agent 类型: {agent_type}。支持的类型: {list(_AGENT_LOADERS)}"
        )
    module_path, class_name = _AGENT_LOADERS[agent_type].split(":")
    module = import_module(module_path)
    return getattr(module, class_name)


def _initial_state(query: str) -> OrchestratorState:
    """构造编排图的初始状态。"""
    return {
        "query": query,
        "messages": [],
        "intent": "",
        "agent_result": "",
        "final_response": "",
        "error": "",
    }


async def _run_demo_queries(app: Any, queries: list[str]) -> None:
    """依次执行演示请求，每个请求绑定独立 trace_id 便于日志串联。"""
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"[请求 {i}] {query}")
        print("-" * 60)

        bind_request_context(trace_id=new_trace_id(), source="demo")
        try:
            config = {"configurable": {"thread_id": f"demo-thread-{i}"}}
            result = await app.ainvoke(_initial_state(query), config)
            print(f"意图分类: {result['intent']}")
            print(f"处理结果:\n{result['final_response']}")
        finally:
            clear_request_context()


def run_demo() -> None:
    """运行编排器演示（默认 InlineGateway，设置 USE_REAL_AGENTS=true 走真实 A2A）。"""
    setup_logging()
    app = build_orchestrator_graph(gateway=create_gateway())
    asyncio.run(_run_demo_queries(app, _DEMO_QUERIES))


async def run_agent_standalone(agent_type: str, query: str) -> None:
    """独立运行一个 A2A Agent 进行测试（绕过 HTTP，直接调业务逻辑）。

    :param agent_type: Agent 类型（data/ticket/report）。
    :param query: 测试查询。
    """
    setup_logging()
    bind_request_context(trace_id=new_trace_id(), source="standalone")
    try:
        agent_cls = _load_agent(agent_type)
        agent = agent_cls()

        print(f"直接调用 {agent.name}...")
        result = await agent.execute_task(query)
        print(f"结果:\n{result}")
    finally:
        clear_request_context()


def run_agent_server(agent_type: str) -> None:
    """启动指定类型的 A2A Agent Server。

    :param agent_type: Agent 类型（data/ticket/report）。
    """
    setup_logging()
    from langgraphcode.a2a_agents.server_runner import run_agent_server as serve

    agent_cls = _load_agent(agent_type)
    serve(agent_cls())


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="智能企业助手系统")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="运行编排器演示")

    agent_parser = subparsers.add_parser("agent", help="独立运行 A2A Agent（绕过 HTTP）")
    agent_parser.add_argument("agent_type", choices=list(_AGENT_LOADERS), help="Agent 类型")
    agent_parser.add_argument("query", help="测试查询")

    serve_parser = subparsers.add_parser("serve", help="启动 A2A Agent Server")
    serve_parser.add_argument("agent_type", choices=list(_AGENT_LOADERS), help="Agent 类型")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "agent":
        asyncio.run(run_agent_standalone(args.agent_type, args.query))
    elif args.command == "serve":
        run_agent_server(args.agent_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
