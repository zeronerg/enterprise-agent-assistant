"""LLM 预设档位定义。

业务代码声明"角色"而非"具体模型"：
路由分类是高频低难度任务，报告生成是低频高难度任务，
两者的成本和时延应差一个数量级。换模型只改配置，不改代码。
"""

from enum import StrEnum


class LLMPreset(StrEnum):
    """模型档位，与 ``Settings.llm`` 的三个配置段一一对应。"""

    ROUTER = "router"
    """意图分类等高频简单任务：便宜、快、低温度。"""

    AGENT = "agent"
    """Agent 工具循环主力：工具调用能力强的中等成本模型。"""

    WRITER = "writer"
    """报告生成等输出质量敏感任务：强模型。"""
