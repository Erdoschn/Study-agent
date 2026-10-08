# StudyAgent Web UI

Framework-free local frontend served by deepseek_web_study_agent_api.py.

Features: Markdown, code highlighting, LaTeX/MathJax, local multi-conversation history, SSE status updates, API-key settings, health status, responsive layout, and cancel-waiting behavior.

Run:
python examples/deepseek_web_study_agent_api.py

Open http://127.0.0.1:8001/

## Coder Web UI

独立的 Coder 前端提供 workspace 文件查看、拖拽/选择文件上传、实时 Agent 步骤、测试结果和 Notebook 支持。

Run:
python examples/coder_web_api.py

Open http://127.0.0.1:8002/coder

Environment:
- CODER_WORKSPACE：Coder 工作区
- CODER_WEB_HOST / CODER_WEB_PORT：服务地址
- CODER_WEB_API_KEY：可选 API key
- CODER_MAX_RUNTIME_SECONDS：单次 Coder 最大运行时间
- CODER_UPLOAD_MAX_BYTES：单文件上传上限（不会超过 workspace 文件上限）

上传只接受 UTF-8 代码/文本及合法 .ipynb，并继续经过 WorkspaceFS 的路径、敏感文件和大小检查。
