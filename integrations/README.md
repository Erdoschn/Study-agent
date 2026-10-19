# Study Agent multi-adapter integration

The existing Python Study Agent remains the canonical implementation. The adapters below must not duplicate the reasoning loop, teacher, learner state, evidence store, or knowledge graph.

- `integrations/study_agent_client.py`: shared HTTP client for the existing OpenAI-compatible API.
- `integrations/mcp_server.py`: optional MCP stdio server.
- `integrations/dsh-plugin/`: native DeepSeek Harness tool plugin.
- `skills/study-agent/SKILL.md`: reusable workflow and privacy instructions.

## Start the existing API

From the repository root:

```powershell
python examples/deepseek_web_study_agent_api.py
```

Default endpoint: `http://127.0.0.1:8001/v1/chat/completions`.

Configure the adapter environment in the process that launches it:

```powershell
$env:STUDY_AGENT_API_URL = "http://127.0.0.1:8001/v1/chat/completions"
$env:STUDY_AGENT_WEB_API_KEY = "<same key as the API, if enabled>"
```

Keep local services bound to loopback. Remote endpoints must use HTTPS. Do not commit API keys.

## MCP adapter

```powershell
python -m pip install -r requirements-mcp.txt
python -m integrations.mcp_server
```

Configure the Harness MCP client to launch `python -m integrations.mcp_server` over stdio from the repository root. Use the virtual-environment Python's absolute path if needed. Stdout is reserved for MCP messages.

Tools:
- `study_agent_ask(username, question, context?)` — username is required for profile isolation.
- `study_agent_health()`

## Native Harness plugin

The out-of-tree plugin lives in `dsh-plugin/`. Load it with the plugin/bundle workflow supported by your installed Harness version. The Harness plugin API is in developer preview, so verify it in an isolated profile before enabling it in your primary profile.

It reads `STUDY_AGENT_API_URL` and `STUDY_AGENT_WEB_API_KEY` from the Harness process environment. The plugin does not persist credentials.

## Skill

Copy `skills/study-agent/` into a skill directory recognized by the target runtime. The Skill documents when and how to use the tools; it does not replace the MCP server or native plugin.

## Security scope

The API requires `username` on each request unless `STUDY_AGENT_DEFAULT_USERNAME` is explicitly set for a single-user compatibility setup. Usernames select separate graph files and separate in-memory Agent state, but are not authentication credentials. Do not expose a multi-user server behind a single shared API key and treat profile names as access control. This integration exposes learning requests and health checks only; it does not expose arbitrary shell execution, filesystem mutation, browser control, cookies, or browser/device identity changes. Context is bounded and should not contain secrets or unrelated personal data.
