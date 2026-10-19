# StudyAgent Web UI

Framework-free local frontend served by `examples/deepseek_web_study_agent_api.py`.

On every page load, choose one mode:
- **Web Model** — uses the local Edge browser session for DeepSeek Web and supports original-file attachment uploads.
- **API Model** — routes only to enabled `openai_compatible` models in `config/providers.json`. The selection may incur provider charges. Text/code attachments are embedded into the prompt with a size bound; PDF, Word and image attachments must use Web Model.

Both modes use the same StudyAgent Agent Loop but keep separate runtime Agent state per user profile. The Web and API modes for the same username share that user's SQLite knowledge graph; different usernames use separate files under `data/users/<sha256-of-normalized-username>/knowledge_graph.sqlite3`. Usernames are normalized case-insensitively and are never used directly as filesystem path components.

On first open, enter a username (letters, numbers, dot, underscore, hyphen; Chinese letters are supported). The UI partitions its conversation history by username and sends the profile with every request. The API rejects requests without a username unless `STUDY_AGENT_DEFAULT_USERNAME` is explicitly configured for a single-user legacy client. `STUDY_AGENT_MAX_USER_PROFILES` controls the profile directory limit (default 256).

Important: a username is a data-partition label, not proof of identity. Do not expose a multi-user server with a shared API key and assume usernames provide access control; use separate authenticated identities. The old shared `data/knowledge_graph.sqlite3` file is left untouched and is not automatically assigned to any profile, because its owner is unknown.

Edge is opened lazily when the Web Model is first used, rather than on server startup.

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
- CODER_UPLOAD_MAX_BYTES：Coder 单文件上传上限，默认 48 MiB（不会超过 workspace 文件上限）

Coder 上传只接受 UTF-8 代码/文本及合法 .ipynb，单文件默认上限为 48 MiB，并继续经过 WorkspaceFS 的路径、敏感文件和大小检查。Study Agent 网页端附件单文件上限为 48 MiB，单次附件总量上限为 48 MiB。
