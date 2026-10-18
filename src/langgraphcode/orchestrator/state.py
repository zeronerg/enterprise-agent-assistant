"""编排状态定义。

定义 LangGraph 编排图的状态结构，贯穿整个请求处理流程。

关键机制：``Annotated[list, add_messages]`` 让 LangGraph 对 messages 字段
使用追加语义而非默认的覆盖语义——reducer 函数决定同名字段被多次更新时
如何合并，这是 LangGraph 状态管理的核心。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """企业助手编排状态。

    每个字段的写入者/读取者：

    - messages：各节点追加（add_messages reducer），供多轮对话与调试追溯；
    - query：入口写入，全图只读；
    - intent：classify 写入，dispatch 读取；
    - agent_result：dispatch 写入，format 读取；
    - final_response：format / handle_error 写入，出口读取；
    - error：异常路径填充。
    """

    messages: Annotated[list, add_messages]
    query: str
    intent: str
    agent_result: str
    final_response: str
    error: str
