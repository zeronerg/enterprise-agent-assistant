"""全局配置。

所有环境差异通过环境变量注入（12-factor）：进程启动时构造 ``settings``
单例即完成全部校验，非法配置立即 fail-fast，而不是在首次调用时才暴露。

环境变量命名规则：``APP_`` 前缀 + ``__`` 分隔嵌套字段，例如::

    APP_LLM__ROUTER__API_KEY=sk-xxx
    APP_LLM__AGENT__MODEL=qwen-plus
    APP_LOGGING__LEVEL=DEBUG
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ModelScope 的 OpenAI 兼容推理接口，作为快速启动的合理缺省。
_DEFAULT_BASE_URL = "https://api-inference.modelscope.cn/v1"


def _default_profile(model: str) -> "ModelProfile":
    """构造带缺省值的模型档位（api_key 留空，由 Settings 统一校验）。"""
    return ModelProfile(model=model, base_url=_DEFAULT_BASE_URL, api_key=SecretStr(""))


class ModelProfile(BaseModel):
    """单个模型档位的完整连接与行为配置。

    Attributes:
        provider: ``init_chat_model`` 的 provider 标识（openai / anthropic / ...）。
        model: 模型名称。
        base_url: OpenAI 兼容接口地址，None 表示使用该 provider 官方地址。
        api_key: API 密钥（SecretStr，任何意外打印都是掩码）。
        temperature: 采样温度。
        max_tokens: 最大输出 token，None 表示不限制。
        timeout: 单次请求超时（秒），传输层韧性。
        max_retries: 传输层自动重试次数（429/5xx），应用层重试见 LLMSettings。
        rpm_limit: 每分钟调用上限，应用层令牌桶限流。
    """

    provider: str = Field(default="openai", description="init_chat_model 的 provider 标识")
    model: str
    base_url: str | None = None
    api_key: SecretStr
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=2, ge=0)
    rpm_limit: int = Field(default=60, ge=1)


class LLMSettings(BaseModel):
    """LLM 配置：三个 preset 档位 + 应用层韧性策略。

    Attributes:
        router: 意图分类等高频简单任务（便宜、快、低温度）。
        agent: Agent 工具循环主力（工具调用能力强）。
        writer: 报告生成等质量敏感任务。
        enable_resilience: 是否给模型套韧性包装。
        max_attempts: 应用层最大尝试次数（含首次调用）。
        retry_base_delay: 重试基础延迟（秒），实际延迟按指数退避 + 抖动。
        breaker_failure_threshold: 熔断器连续失败阈值。
        breaker_recovery_seconds: 熔断器打开后的恢复探测间隔（秒）。
    """

    router: ModelProfile = Field(default_factory=lambda: _default_profile("qwen-turbo"))
    agent: ModelProfile = Field(default_factory=lambda: _default_profile("qwen-plus"))
    writer: ModelProfile = Field(default_factory=lambda: _default_profile("qwen-max"))
    enable_resilience: bool = True
    max_attempts: int = Field(default=3, ge=1)
    retry_base_delay: float = Field(default=1.0, gt=0.0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_recovery_seconds: float = Field(default=30.0, gt=0.0)


class LoggingSettings(BaseModel):
    """日志配置。"""

    level: LogLevel = "INFO"
    json_format: bool = True
    file_path: Path | None = None
    file_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    file_backup_count: int = Field(default=5, ge=0)
    suppress_noisy_loggers: bool = True


class A2ASettings(BaseModel):
    """A2A 协作层配置：各 Agent Server 的端点。"""

    data_agent_url: str = "http://localhost:8011"
    ticket_agent_url: str = "http://localhost:8012"
    report_agent_url: str = "http://localhost:8013"
    request_timeout: float = Field(default=60.0, ge=1.0)


class OrchestratorSettings(BaseModel):
    """编排层配置。

    Attributes:
        use_real_agents: True 时编排层通过 A2A 协议调用远程 Agent Server；
            False 时使用进程内 LLM 模拟（InlineGateway），保证零依赖可运行。
    """

    use_real_agents: bool = False


class LLMPresetNames:
    """preset 名称常量（settings 不依赖 common.llm，防止循环导入）。"""

    ROUTER = "router"
    AGENT = "agent"
    WRITER = "writer"

    ITERABLE = (ROUTER, AGENT, WRITER)


class Settings(BaseSettings):
    """应用全局配置根节点。"""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    a2a: A2ASettings = Field(default_factory=A2ASettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)

    @model_validator(mode="after")
    def _require_api_keys(self) -> "Settings":
        """启动即校验：任一 preset 缺 API Key 立即 fail-fast。"""
        for name in LLMPresetNames.ITERABLE:
            profile: ModelProfile = getattr(self.llm, name)
            if not profile.api_key.get_secret_value():
                env_var = f"APP_LLM__{name.upper()}__API_KEY"
                raise ValueError(f"LLM preset '{name}' 未配置 API Key，请设置环境变量 {env_var}")
        return self


# 模块级单例：import 即校验，进程启动时 fail-fast。
settings = Settings()
