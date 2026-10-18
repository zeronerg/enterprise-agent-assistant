# tests/test_mcp_servers.py
"""MCP Server 集成测试。

通过 FastMCP 的内存内 Client 直接调用工具，
验证工具契约（输入 → JSON 输出 / 错误 JSON），不依赖 stdio 子进程。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langgraphcode.mcp_servers.database_server import mcp as database_mcp
from langgraphcode.mcp_servers.document_server import mcp as document_mcp


async def _call_tool(mcp: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """内存内调用 MCP 工具并解析 JSON 返回。"""
    from fastmcp import Client

    async with Client(mcp) as client:
        result = await client.call_tool(tool, args)
        payload = json.loads(result.content[0].text)
    return payload


class TestDatabaseServer:
    """数据库查询工具契约。"""

    def test_query_sales_happy_path(self) -> None:
        payload = asyncio.run(_call_tool(database_mcp, "query_sales", {"period": "2025-03"}))
        assert payload["period"] == "2025-03"
        assert payload["total_revenue"] > 0
        assert len(payload["details"]) == 4

    def test_query_sales_region_filter(self) -> None:
        payload = asyncio.run(
            _call_tool(database_mcp, "query_sales", {"period": "2025-03", "region": "华东"})
        )
        assert len(payload["details"]) == 1
        assert payload["details"][0]["region"] == "华东"

    def test_query_sales_unknown_period_returns_error_json(self) -> None:
        payload = asyncio.run(_call_tool(database_mcp, "query_sales", {"period": "2030-01"}))
        assert "error" in payload
        assert "available_periods" in payload  # 错误信息携带可修正线索，供 LLM 自我纠正

    def test_query_customer_happy_path(self) -> None:
        payload = asyncio.run(_call_tool(database_mcp, "query_customer", {"customer_id": "C-1001"}))
        assert payload["name"] == "星辰科技有限公司"

    def test_query_customer_unknown_id_returns_error_json(self) -> None:
        payload = asyncio.run(_call_tool(database_mcp, "query_customer", {"customer_id": "C-9999"}))
        assert "error" in payload


class TestDocumentServer:
    """文档检索工具契约。"""

    def test_search_knowledge_finds_vpn_doc(self) -> None:
        payload = asyncio.run(_call_tool(document_mcp, "search_knowledge", {"query": "VPN"}))
        assert payload["total"] >= 1
        assert any(r["doc_id"] == "KB-001" for r in payload["results"])

    def test_search_knowledge_category_filter(self) -> None:
        payload = asyncio.run(
            _call_tool(document_mcp, "search_knowledge", {"query": "流程", "category": "财务"})
        )
        assert all(r["category"] == "财务" for r in payload["results"])

    def test_search_knowledge_no_match(self) -> None:
        payload = asyncio.run(
            _call_tool(document_mcp, "search_knowledge", {"query": "不存在的词汇xyz"})
        )
        assert payload["total"] == 0

    def test_get_document_happy_path(self) -> None:
        payload = asyncio.run(_call_tool(document_mcp, "get_document", {"doc_id": "KB-001"}))
        assert "VPN" in payload["title"]
