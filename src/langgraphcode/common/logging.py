"""统一日志：structlog 结构化输出 + contextvars 上下文绑定。

设计要点：

- 同一个 logger，按配置渲染为 JSON（生产/日志采集）或彩色控制台（开发）。
- ``bind_request_context`` 绑定的字段（trace_id / task_id / intent）通过
  contextvars 传播，自动附加到该请求后续所有异步任务产生的日志，
  无需在每个调用点手工传参——这是 Agent 系统可观测性的关键。
- 内置密钥脱敏 processor，防止 API Key 等敏感信息进入日志系统。
- ``setup_logging`` 幂等：以 root logger 上的标记属性判断，
  测试可通过 ``reset_logging()`` 恢复未配置状态。
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path

import structlog

from langgraphcode.config.settings import LoggingSettings, settings

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
)
_REDACTED = "***"
_CONFIG_ATTR = "_langgraphcode_logging_configured"


def _redact_secrets(
    _logger: logging.Logger,
    _method: str,
    event_dict: dict[str, object],
) -> dict[str, object]:
    """脱敏 processor：扫描字符串字段中的密钥模式并替换。"""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern in _SECRET_PATTERNS:
                value = pattern.sub(_REDACTED, value)
            event_dict[key] = value
    return event_dict


def _foreign_pre_chain() -> list[object]:
    """非 structlog 产生的日志（uvicorn/httpx 等）进入 JSON 渲染前的标准化链。"""
    return [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def setup_logging(config: LoggingSettings | None = None) -> None:
    """初始化全局日志配置（幂等，多次调用只有第一次生效）。

    :param config: 日志配置，缺省时使用全局 ``settings.logging``。
    """
    config = config or settings.logging
    root = logging.getLogger()
    if getattr(root, _CONFIG_ATTR, False):
        return

    root.setLevel(config.level)
    root.handlers.clear()

    # stdout：始终输出；格式由 structlog ProcessorFormatter 统一接管。
    stdout_handler = logging.StreamHandler(sys.stdout)
    root.addHandler(stdout_handler)

    file_handler: logging.handlers.RotatingFileHandler | None = None
    if config.file_path is not None:
        path = Path(config.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
            encoding="utf-8",
        )
        root.addHandler(file_handler)

    if config.suppress_noisy_loggers:
        for name in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
            logging.getLogger(name).setLevel(logging.WARNING)

    renderer: structlog.types.Processor
    if config.json_format:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_secrets,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(config.level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_foreign_pre_chain(),
        processors=[renderer],
    )
    stdout_handler.setFormatter(formatter)
    if file_handler is not None:
        file_handler.setFormatter(formatter)

    setattr(root, _CONFIG_ATTR, True)


def reset_logging() -> None:
    """清除配置标记（供测试在 setup 之间重置状态）。"""
    root = logging.getLogger()
    root.handlers.clear()
    if hasattr(root, _CONFIG_ATTR):
        delattr(root, _CONFIG_ATTR)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """获取绑定 structlog 的 logger（未初始化时按默认配置自动初始化）。

    :param name: logger 名称，通常传 ``__name__``。
    :return: 带级别过滤的 BoundLogger。
    """
    if not getattr(logging.getLogger(), _CONFIG_ATTR, False):
        setup_logging()
    return structlog.get_logger(name)
