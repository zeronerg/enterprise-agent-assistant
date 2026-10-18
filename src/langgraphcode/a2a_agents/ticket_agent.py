# ticket_agent.py
"""工单处理 A2A Agent。

负责处理用户提交的技术问题、IT 支持请求等工单类任务，
通过 MCP 连接文档 Server 检索知识库。
"""

from __future__ import annotations

from pathlib import Path

from a2a.types import AgentSkill

from langgraphcode.a2a_agents.base_agent import BaseA2AAgent

_DOC_SERVER_PATH = str(Path(__file__).parent.parent / "mcp_servers" / "document_server.py")


class TicketAgent(BaseA2AAgent):
    """工单处理 Agent。

    连接文档 MCP Server，检索知识库来解决用户的技术问题。
    """

    def __init__(self) -> None:
        """初始化工单处理 Agent。"""
        super().__init__(
            name="TicketAgent",
            description="IT 工单处理助手，检索知识库解决技术问题",
            port=8012,
            mcp_server_paths=[_DOC_SERVER_PATH],
        )

    def get_skills(self) -> list[AgentSkill]:
        """返回工单处理 Agent 的技能列表。"""
        return [
            AgentSkill(
                id="search_knowledge",
                name="知识库检索",
                description="在企业知识库中搜索解决方案",
                examples=["VPN 连接不上怎么办"],
            ),
            AgentSkill(
                id="get_document",
                name="文档查看",
                description="获取知识库文档的完整内容",
                examples=["查看文档 KB-001 的完整内容"],
            ),
            AgentSkill(
                id="submit_ticket",
                name="提交工单",
                description="当知识库无法解决时，生成工单提交建议",
                examples=["帮我提交一个关于网络问题的工单"],
            ),
        ]

    def get_system_prompt(self) -> str:
        """返回工单处理 Agent 的系统提示词。"""
        return (
            "你是一个 IT 工单处理助手。你可以使用以下工具：\n"
            "- search_knowledge：在企业知识库中搜索解决方案\n"
            "- get_document：获取指定文档的完整内容\n\n"
            "处理流程：\n"
            "1. 用户描述问题后，先用 search_knowledge 搜索知识库\n"
            "2. 如果找到相关文档，用 get_document 获取完整内容\n"
            "3. 基于文档内容给出解决方案\n"
            "4. 如果知识库中没有相关方案，建议用户提交工单\n\n"
            "回答要具体、可操作，包含明确的步骤。"
        )
