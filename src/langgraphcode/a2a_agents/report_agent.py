# report_agent.py
"""报告生成 A2A Agent。

负责根据用户需求生成结构化报告或文档。
不依赖 MCP 工具，直接通过 LLM 生成内容——
演示了 Agent 的两种形态：有工具（Data/Ticket）与无工具（Report）。
"""

from __future__ import annotations

from a2a.types import AgentSkill

from langgraphcode.a2a_agents.base_agent import BaseA2AAgent


class ReportAgent(BaseA2AAgent):
    """报告生成 Agent。

    根据用户提供的主题和数据，生成结构化的技术报告或业务简报。
    """

    def __init__(self) -> None:
        """初始化报告生成 Agent。"""
        super().__init__(
            name="ReportAgent",
            description="报告撰写助手，生成结构化技术报告和业务简报",
            port=8013,
            mcp_server_paths=[],
        )

    def get_skills(self) -> list[AgentSkill]:
        """返回报告生成 Agent 的技能列表。"""
        return [
            AgentSkill(
                id="write_report",
                name="报告撰写",
                description="根据主题生成结构化技术报告",
                examples=["写一份关于 AI Agent 技术趋势的报告"],
            ),
            AgentSkill(
                id="summarize_data",
                name="数据总结",
                description="将数据整理为可读的摘要",
                examples=["总结这份销售数据的关键发现"],
            ),
        ]

    def get_system_prompt(self) -> str:
        """返回报告生成 Agent 的系统提示词。"""
        return (
            "你是一个专业的报告撰写助手。收到报告请求后，请：\n"
            "1. 使用 Markdown 格式输出\n"
            "2. 包含标题、摘要、正文（分章节）和结论\n"
            "3. 正文使用分点论述，逻辑清晰\n"
            "4. 语言简洁专业，适合企业管理层阅读\n"
            "5. 如果用户提供了数据，基于数据进行分析\n"
            "6. 不要编造不存在的数据"
        )
