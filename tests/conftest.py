"""测试环境准备：在导入应用代码之前注入占位配置。

``langgraphcode.config.settings`` 是模块级单例，import 即完成校验，
因此必须在任何 ``langgraphcode`` 模块被导入之前设置好环境变量。

这样 ``pytest`` 无需真实 API Key 即可跑通全部离线用例。
"""

import os

for _preset in ("ROUTER", "AGENT", "WRITER"):
    os.environ.setdefault(f"APP_LLM__{_preset}__API_KEY", "test-key")
    os.environ.setdefault(f"APP_LLM__{_preset}__MODEL", "test-model")
