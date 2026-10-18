# 智能企业助手原型 · LangGraph + MCP + A2A

**一个可运行的企业智能助手原型**：用户输入自然语言 → 意图分类 → 路由到对应专业 Agent → 返回结构化结果。
按「编排层 / 工具层 / 协作层」三层解耦，配套 LLM 韧性层与可观测性。

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-green)
![MCP](https://img.shields.io/badge/MCP-FastMCP-orange)
![A2A](https://img.shields.io/badge/A2A-1.2+-purple)

> ⚠️ **项目性质**：个人工程实践原型。业务数据为模拟数据，未在生产环境部署；知识库检索为关键词匹配实现，尚未接入向量检索（见 [Roadmap](#roadmap)）。

---

## 一、它解决什么问题

企业内部有三类高频需求：**数据查询**（销售/客户）、**IT 工单**（故障排查）、**报告生成**。

传统做法是每个场景各做一个机器人，导致：

- 工具重复实现，新增一个数据源要改多处；
- Agent 之间无法互相调用，跨系统任务只能靠人工串联；
- 模型超时、限流、供应商故障直接暴露给用户，没有兜底。

本项目用三层解耦解决上述问题：

| 层 | 职责 | 技术选型 | 解耦带来的收益 |
|---|---|---|---|
| 编排层 | 意图分类、路由、结果格式化 | LangGraph StateGraph | 流程显式化，每个节点可单独测试、可断点恢复 |
| 工具层 | 数据查询、知识库检索 | MCP（FastMCP） | 工具与 Agent 解耦，加工具不改 Agent 代码 |
| 协作层 | 多 Agent 跨服务协作 | A2A 协议 | 跨团队/跨系统互操作，无需定制集成 |

---

## 二、架构

```
                         ┌──────────────────────────────────┐
   用户请求  ──────────▶ │     Orchestrator (LangGraph)      │
                         │  classify → dispatch → format     │
                         └───────────────┬──────────────────┘
                                         │ AgentGateway (Protocol)
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                   InlineGateway                 A2AGateway
                （进程内模拟 / 测试）      （真实 A2A 协议调用）
                                                        │
              ┌─────────────────────┬───────────────────┴────────────┐
              ▼                     ▼                                ▼
        DataAgent :8011       TicketAgent :8012               ReportAgent :8013
              │                     │
              ▼                     ▼
   MCP: database_server     MCP: document_server
     （销售 / 客户数据）          （知识库检索）
```

**一次请求的完整链路**：`classify` 判定意图 → `dispatch` 通过网关调用对应 Agent → Agent 经 MCP 加载工具并执行 ReAct 循环 → 结果回到 `format` 节点 → 输出。

---

## 三、核心设计（Why，而不只是 What）

### 1. 确定的路由，用状态图显式化

三类意图的路由是**确定的**，不需要 Agent 自主规划。用 LangGraph 把流程画成状态图，收益是：每个节点可单独测试、可断点续跑、成本可预测。纯自主 Agent 在这种场景反而不可控。

### 2. 两种路由机制各得其所

- `classify` 用 **`Command(update=..., goto=...)`**：它既要更新状态（intent）又要决定去哪，两者耦合在一个返回值里更内聚；
- `dispatch` 之后用**条件边**：只判断结果是否为空，纯路由用条件边更轻。

同一个图里混用两种机制，是为了让每种机制只承担它擅长的事。

### 3. MCP 让工具与 Agent 解耦

`@mcp.tool()` 从**类型注解生成 JSON Schema**、从 **docstring 生成工具描述**，零样板代码；用 `Literal` 枚举约束模型入参，让 schema 本身少犯错。

工具出错时**返回带 `error` 和可选值清单的 JSON**，模型读到后能换参数自我修正——这比抛异常更有用。

### 4. A2A 让 Agent 之间互操作

每个 Agent 通过 `/.well-known/agent.json` 暴露 **AgentCard**（能力声明），编排层先发现再调用；任务按标准生命周期推进：`submit → execute → add_artifact → complete`。换一个 Agent 实现，编排层无需改动。

### 5. LLM 韧性层：把供应商故障挡在业务之外

- **三档 preset**（router / agent / writer）：高频低难度与低频高质量任务用不同模型，成本差一个数量级；
- **令牌桶限流**：保护供应商 RPM 配额，同步/异步共用一把锁，临界区只做数值计算，等待在锁外；
- **三态熔断**：CLOSED → OPEN（快速失败）→ HALF_OPEN（放一次探测）→ CLOSED；
- **指数退避 + 抖动的重试**：仅对可重试错误（限流/超时/连接/5xx）生效，**4xx 立即失败**——盲目重试会把配置错误放大成雪崩。

以**透明包装器**实现，不侵入业务代码：`invoke` / `ainvoke` / `bind_tools` / `with_structured_output` 全部受保护。

### 6. 可观测性与可测试性

- `structlog` + `contextvars`：入口 `bind` 一次 `trace_id`，整条异步调用链的日志自动携带，无需逐层传参；内置密钥脱敏 processor。
- **依赖注入**：网关、Checkpointer 均为注入参数，测试可用 `FakeGateway` 替代，节点逻辑不依赖 LLM 与网络。

---

## 四、快速开始

```bash
git clone https://github.com/zeronerg/enterprise-agent-assistant.git
cd enterprise-agent-assistant

uv sync

cp .env.example .env
# 在 .env 中填入三个 preset 的 API Key：
# APP_LLM__ROUTER__API_KEY=...
# APP_LLM__AGENT__API_KEY=...
# APP_LLM__WRITER__API_KEY=...

# 运行编排器演示（进程内网关，零外部依赖）
uv run python -m langgraphcode.main demo

# 启动单个 Agent 服务
uv run python -m langgraphcode.main serve data

# 启用真实 A2A 调用
# APP_ORCHESTRATOR__USE_REAL_AGENTS=true
```

---

## 五、项目结构

```
langgraphcode/
├── src/langgraphcode/
│   ├── config/settings.py          # 全局配置（fail-fast 校验）
│   ├── common/
│   │   ├── logging.py              # structlog + 脱敏
│   │   ├── context.py              # contextvars 链路追踪
│   │   └── llm/
│   │       ├── presets.py          # 三档模型枚举
│   │       ├── resilience.py       # 限流 / 熔断 / 重试
│   │       └── factory.py          # 全系统唯一模型出口
│   ├── mcp_servers/                # 工具层
│   │   ├── database_server.py      # 销售 / 客户数据
│   │   └── document_server.py      # 知识库检索
│   ├── a2a_agents/                 # 协作层
│   │   ├── base_agent.py           # 模板方法基类 + 执行器
│   │   ├── data_agent.py           # :8011
│   │   ├── ticket_agent.py         # :8012
│   │   └── report_agent.py         # :8013
│   ├── orchestrator/               # 编排层
│   │   ├── state.py                # 状态定义（reducer 语义）
│   │   ├── gateway.py              # Gateway 协议 + 双实现
│   │   ├── nodes.py                # 节点函数
│   │   └── graph.py                # 图构建与编译
│   └── main.py                     # 入口（demo / agent / serve）
└── tests/
    ├── test_resilience.py          # 韧性层离线单测
    ├── test_mcp_servers.py         # MCP 工具契约测试
    └── test_orchestrator_gateway.py# 编排节点测试（FakeGateway）
```

---

## 六、测试

```bash
uv run pytest -q
```

| 测试文件 | 验证什么 | 设计要点 |
|---|---|---|
| `test_resilience.py` | 限流、熔断三态、可重试错误分类 | 纯离线，不依赖 LLM 与网络 |
| `test_mcp_servers.py` | 工具契约：正常返回 / 错误 JSON / 参数筛选 | 用 FastMCP 内存内 Client，不起 stdio 子进程 |
| `test_orchestrator_gateway.py` | 分发节点的依赖注入与路由语义 | `FakeGateway` 注入，验证节点不依赖 LLM |

---

## 七、已知局限

诚实说明，也是下一步的方向：

- **数据是内存模拟的**，未接真实数据库；
- **知识库检索是关键词子串匹配**，不是向量检索，没有召回率指标；
- **MCP 使用 stdio 传输**，每次任务 fork 子进程；高频场景应让 MCP Server 常驻并改用 HTTP 传输复用连接；
- **Checkpointer 是内存实现**（`MemorySaver`），进程重启状态丢失，生产应换 `PostgresSaver`；
- **没有评测集**，效果未量化；
- 未做流式输出与前端界面。

---

## 八、Roadmap

- [ ] **RAG**：embedding + 向量库 + rerank，替换关键词检索
- [ ] **评测集**：意图分类准确率 / 工具选择正确率 / 任务完成率，形成回归基线
- [ ] **Docker Compose**：一键启动编排器 + 三个 Agent
- [ ] **流式输出**：SSE 返回中间状态
- [ ] **MCP 常驻化**：HTTP 传输 + 连接复用

---

## License

MIT
