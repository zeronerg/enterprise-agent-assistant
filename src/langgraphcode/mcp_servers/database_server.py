"""数据库查询 MCP Server。

提供企业内部数据的模拟查询能力，演示 MCP Server 的工具（Tools）
和资源（Resources）两种核心能力。

在生产环境中，工具函数内部会连接真实数据库（PostgreSQL、MySQL 等），
通过 ORM 或原生 SQL 查询数据。本模块使用内存模拟数据，保证项目可独立运行。

设计要点：
- ``@mcp.tool()`` 从类型注解生成 JSON Schema、从 docstring 生成工具描述，
  零样板代码；``Literal`` 枚举让 schema 本身约束 LLM 的入参。
- 工具统一返回 JSON 字符串：跨语言兼容，且错误也是带 ``error`` 字段的
  JSON——LLM 读到错误信息后可以自我修正（换月份重查）。
"""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from fastmcp import FastMCP

mcp = FastMCP("database-server")

Region = Literal["华北", "华东", "华南", "西南", "全部"]

# ---------------------------------------------------------------------------
# 模拟数据层（生产环境替换为真实数据库连接）
# ---------------------------------------------------------------------------

_SALES_DATA: dict[str, list[dict[str, object]]] = {
    "2025-01": [
        {"region": "华北", "revenue": 1280000, "orders": 3420, "growth": 0.12},
        {"region": "华东", "revenue": 2150000, "orders": 5680, "growth": 0.18},
        {"region": "华南", "revenue": 980000, "orders": 2610, "growth": 0.08},
        {"region": "西南", "revenue": 720000, "orders": 1950, "growth": 0.05},
    ],
    "2025-02": [
        {"region": "华北", "revenue": 1150000, "orders": 3080, "growth": 0.09},
        {"region": "华东", "revenue": 2280000, "orders": 5920, "growth": 0.22},
        {"region": "华南", "revenue": 1050000, "orders": 2790, "growth": 0.11},
        {"region": "西南", "revenue": 680000, "orders": 1820, "growth": 0.02},
    ],
    "2025-03": [
        {"region": "华北", "revenue": 1420000, "orders": 3750, "growth": 0.15},
        {"region": "华东", "revenue": 2450000, "orders": 6310, "growth": 0.25},
        {"region": "华南", "revenue": 1180000, "orders": 3120, "growth": 0.14},
        {"region": "西南", "revenue": 850000, "orders": 2280, "growth": 0.10},
    ],
}

_CUSTOMER_DATA: dict[str, dict[str, object]] = {
    "C-1001": {
        "name": "星辰科技有限公司",
        "tier": "战略客户",
        "arr": 580000,
        "health_score": 92,
        "contract_end": "2026-06-30",
    },
    "C-1002": {
        "name": "云图信息技术",
        "tier": "核心客户",
        "arr": 320000,
        "health_score": 78,
        "contract_end": "2025-12-31",
    },
    "C-1003": {
        "name": "蓝海数据服务",
        "tier": "标准客户",
        "arr": 150000,
        "health_score": 65,
        "contract_end": "2026-03-15",
    },
}


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------


@mcp.tool()
def query_sales(period: str, region: Region = "全部") -> str:
    """查询指定时间段的销售数据。

    Args:
        period: 查询月份，格式为 YYYY-MM，例如 2025-03。
        region: 区域筛选，可选值：华北、华东、华南、西南、全部。

    Returns:
        JSON 格式的销售数据，包含各地区收入、订单量和增长率；
        无数据时返回带 error 字段的 JSON。
    """
    if period not in _SALES_DATA:
        return json.dumps(
            {"error": f"无 {period} 的数据", "available_periods": sorted(_SALES_DATA)},
            ensure_ascii=False,
        )

    records: list[dict[str, object]] = _SALES_DATA[period]  # type: ignore[assignment]
    if region != "全部":
        records = [r for r in records if r["region"] == region]
        if not records:
            return json.dumps(
                {"error": f"区域 {region} 无数据", "period": period},
                ensure_ascii=False,
            )

    return json.dumps(
        {
            "period": period,
            "region_filter": region,
            "total_revenue": sum(int(r["revenue"]) for r in records),
            "total_orders": sum(int(r["orders"]) for r in records),
            "details": records,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def query_customer(customer_id: str) -> str:
    """查询客户详细信息。

    Args:
        customer_id: 客户编号，格式为 C-XXXX，例如 C-1001。

    Returns:
        JSON 格式的客户信息，包含名称、等级、年收入、健康分和合同到期日。
    """
    customer = _CUSTOMER_DATA.get(customer_id)
    if not customer:
        return json.dumps(
            {"error": f"客户 {customer_id} 不存在", "known_ids": list(_CUSTOMER_DATA)},
            ensure_ascii=False,
        )
    return json.dumps(
        {"customer_id": customer_id, **customer},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def get_sales_trend(months: int = 3) -> str:
    """获取最近 N 个月的销售趋势摘要。

    Args:
        months: 查询月数，默认 3 个月。

    Returns:
        文本格式的销售趋势摘要，包含总收入变化和增长最快的区域。
    """
    periods = sorted(_SALES_DATA.keys())[-months:]
    summaries = [
        f"{p}: 总收入 ¥{sum(int(r['revenue']) for r in _SALES_DATA[p]):,}" for p in periods
    ]
    best_region = max(_SALES_DATA[periods[-1]], key=lambda r: float(r["growth"]))
    lines = [
        f"近 {months} 个月销售趋势：",
        *summaries,
        f"\n增长最快区域：{best_region['region']}（增长率 {float(best_region['growth']):.1%}）",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 资源定义
# ---------------------------------------------------------------------------


@mcp.resource("config://database-status")
def get_database_status() -> str:
    """返回数据库连接状态和可用数据集信息。

    Returns:
        描述数据库状态的 JSON 字符串。
    """
    return json.dumps(
        {
            "status": "healthy",
            "available_tables": ["sales_data", "customer_data"],
            "available_periods": sorted(_SALES_DATA.keys()),
            "customer_count": len(_CUSTOMER_DATA),
            "last_updated": date.today().isoformat(),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
