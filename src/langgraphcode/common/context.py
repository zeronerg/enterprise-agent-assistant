"""请求级上下文绑定。

在请求入口（``main`` / A2A executor）调用 :func:`bind_request_context` 后，
trace_id / task_id 等字段通过 contextvars 自动附加到该请求整条异步
调用链（编排节点、工具调用、LLM 请求）的所有日志，无需人肉传参。
"""

from __future__ import annotations

import uuid

import structlog.contextvars

__all__ = ["bind_request_context", "clear_request_context", "new_trace_id"]


def new_trace_id() -> str:
    """生成短格式 trace id（16 位十六进制），用于串联一次请求的所有日志。"""
    return uuid.uuid4().hex[:16]


def bind_request_context(**fields: str) -> None:
    """绑定字段到当前上下文，对该请求后续所有异步任务生效。

    :param fields: 任意键值对，常见用法::

        bind_request_context(trace_id=new_trace_id(), task_id=task_id, intent="ticket")
    """
    structlog.contextvars.bind_contextvars(**fields)


def clear_request_context() -> None:
    """清除当前上下文绑定的字段。

    请求结束时调用，避免字段泄漏到复用的 worker 线程/协程。
    """
    structlog.contextvars.clear_contextvars()
