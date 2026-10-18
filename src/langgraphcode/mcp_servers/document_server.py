"""文档检索 MCP Server。

提供企业内部知识库的模拟检索能力，演示 MCP Server 的工具（Tools）、
资源模板（Resource Templates）和提示模板（Prompts）。
"""

from __future__ import annotations

import json

from fastmcp import FastMCP

mcp = FastMCP("document-server")

# ---------------------------------------------------------------------------
# 模拟知识库
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, dict[str, object]] = {
    "KB-001": {
        "title": "VPN 连接故障排查指南",
        "category": "IT支持",
        "content": (
            "VPN 连接失败的常见原因及解决方案：\n"
            "1. 检查网络连接是否正常，尝试 ping 网关\n"
            "2. 确认 VPN 客户端版本是否为最新\n"
            "3. 检查证书是否过期，过期需联系 IT 部门续期\n"
            "4. 尝试切换 VPN 服务器节点\n"
            "5. 如果以上均无效，请提交工单并附上错误日志"
        ),
        "tags": ["VPN", "网络", "故障排查"],
    },
    "KB-002": {
        "title": "报销流程说明",
        "category": "财务",
        "content": (
            "员工报销流程：\n"
            "1. 在 OA 系统中填写报销单\n"
            "2. 上传发票原件扫描件（PDF 格式）\n"
            "3. 直属主管审批（1 个工作日内）\n"
            "4. 财务审核（2 个工作日内）\n"
            "5. 打款至员工工资卡（审核通过后 3 个工作日内）\n"
            "注意事项：单笔超过 5000 元需部门总监额外审批。"
        ),
        "tags": ["报销", "财务", "流程"],
    },
    "KB-003": {
        "title": "新产品 API 接入文档",
        "category": "技术文档",
        "content": (
            "数据平台 API v2.0 接入指南：\n"
            "- 认证方式：Bearer Token（在控制台生成）\n"
            "- 基础 URL：https://api.example.com/v2\n"
            "- 主要端点：\n"
            "  GET /datasets — 列出可用数据集\n"
            "  POST /query — 提交查询请求\n"
            "  GET /jobs/{id} — 查询任务状态\n"
            "- 速率限制：每分钟 100 次请求\n"
            "- 错误码：400 参数错误，401 认证失败，429 限流，500 服务异常"
        ),
        "tags": ["API", "技术文档", "接入"],
    },
}


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------


@mcp.tool()
def search_knowledge(query: str, category: str = "") -> str:
    """在企业知识库中搜索相关文档。

    Args:
        query: 搜索关键词，会在标题、内容和标签中匹配。
        category: 可选的分类筛选，如 IT支持、财务、技术文档。

    Returns:
        JSON 格式的搜索结果列表，每项包含文档 ID、标题和摘要。
    """
    query_lower = query.lower()
    results = [
        {
            "doc_id": doc_id,
            "title": doc["title"],
            "category": doc["category"],
            "summary": f"{doc['content'][:100]}...",
            "tags": doc["tags"],
        }
        for doc_id, doc in _KNOWLEDGE_BASE.items()
        if (not category or doc["category"] == category)
        and query_lower
        in f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}".lower()  # type: ignore[arg-type]
    ]

    if not results:
        return json.dumps(
            {"message": f"未找到与「{query}」相关的文档", "total": 0},
            ensure_ascii=False,
        )

    return json.dumps(
        {"query": query, "total": len(results), "results": results},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def get_document(doc_id: str) -> str:
    """获取指定文档的完整内容。

    Args:
        doc_id: 文档编号，格式为 KB-XXX，例如 KB-001。

    Returns:
        JSON 格式的文档完整内容。
    """
    doc = _KNOWLEDGE_BASE.get(doc_id)
    if not doc:
        return json.dumps(
            {"error": f"文档 {doc_id} 不存在", "known_ids": list(_KNOWLEDGE_BASE)},
            ensure_ascii=False,
        )
    return json.dumps(
        {"doc_id": doc_id, **doc},
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# 资源定义
# ---------------------------------------------------------------------------


@mcp.resource("doc://categories")
def list_categories() -> str:
    """列出知识库中所有文档分类。

    Returns:
        JSON 格式的分类列表及每个分类下的文档数量。
    """
    categories: dict[str, int] = {}
    for doc in _KNOWLEDGE_BASE.values():
        cat = str(doc["category"])
        categories[cat] = categories.get(cat, 0) + 1

    return json.dumps(
        {"categories": [{"name": k, "count": v} for k, v in categories.items()]},
        ensure_ascii=False,
    )


@mcp.resource("doc://{doc_id}/content")
def get_document_content(doc_id: str) -> str:
    """资源模板：通过 URI 直接读取文档内容。

    Args:
        doc_id: 文档编号，从 URI 模板中提取。

    Returns:
        文档的纯文本内容，文档不存在时返回错误提示。
    """
    doc = _KNOWLEDGE_BASE.get(doc_id)
    if not doc:
        return f"文档 {doc_id} 不存在"
    return f"# {doc['title']}\n\n分类：{doc['category']}\n\n{doc['content']}"


# ---------------------------------------------------------------------------
# 提示模板
# ---------------------------------------------------------------------------


@mcp.prompt("summarize-doc")
def summarize_doc_prompt(doc_id: str) -> str:
    """生成文档摘要的提示模板。

    Args:
        doc_id: 要摘要的文档编号。

    Returns:
        引导 LLM 生成文档摘要的提示文本。
    """
    doc = _KNOWLEDGE_BASE.get(doc_id)
    if not doc:
        return f"请查找文档 {doc_id} 并生成摘要。"
    return (
        f"请阅读以下文档内容，生成一段不超过 200 字的摘要，"
        f"突出关键操作步骤和注意事项：\n\n{doc['content']}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
