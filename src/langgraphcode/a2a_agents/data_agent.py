# data_agent.py
"""数据查询 A2A Agent。

负责处理销售数据查询、客户信息查询等数据类请求，
通过 MCP 连接数据库 Server 获取数据。
"""

from __future__ import annotations

from pathlib import Path

from a2a.types import AgentSkill

from langgraphcode.a2a_agents.base_agent import BaseA2AAgent

# 路径可移植性：MCP Server 路径始终相对于当前文件解析，
# 不依赖进程工作目录，项目部署到任何位置都能找到。
_DB_SERVER_PATH = str(Path(__file__).parent.parent / "mcp_servers" / "database_server.py")


class DataAgent(BaseA2AAgent):
    """数据查询 Agent。

    连接数据库 MCP Server，提供销售数据、客户信息的查询能力。
    """

    def __init__(self) -> None:
        """初始化数据查询 Agent。"""
        super().__init__(
            name="DataAgent",
            description="企业数据查询助手，支持销售数据和客户信息查询",
            port=8011,
            mcp_server_paths=[_DB_SERVER_PATH],
        )

    def get_skills(self) -> list[AgentSkill]:
        """返回数据查询 Agent 的技能列表。"""
        return [
            AgentSkill(
                id="query_sales",
                name="销售数据查询",
                description="查询指定月份和区域的销售数据",
                tags=["data", "sales"],
                examples=["查询 2025-03 华北地区的销售数据"],
            ),
            AgentSkill(
                id="query_customer",
                name="客户信息查询",
                description="查询客户的详细信息和健康分",
                tags=["data", "sales"],
                examples=["查询客户 C-1001 的信息"],
            ),
            AgentSkill(
                id="sales_trend",
                name="销售趋势分析",
                description="获取最近几个月的销售趋势摘要",
                tags=["data", "sales"],
                examples=["分析最近 3 个月的销售趋势"],
            ),
        ]

    def get_system_prompt(self) -> str:
        """返回数据查询 Agent 的系统提示词。"""
        return (
            "你是一个企业数据查询助手。你可以使用以下工具：\n"
            "- query_sales：查询销售数据（需要月份参数，格式 YYYY-MM）\n"
            "- query_customer：查询客户信息（需要客户编号）\n"
            "- get_sales_trend：获取销售趋势摘要\n\n"
            "回答要求：\n"
            "1. 先调用工具获取数据，再基于数据回答\n"
            "2. 数据展示要清晰，使用表格或列表格式\n"
            "3. 如果用户没有指定月份，默认查询最近可用的月份\n"
            "4. 不要编造数据，只使用工具返回的真实数据"
        )
