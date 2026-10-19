# DeepSeek Harness Study Agent plugin

This out-of-tree plugin registers two tools in the Harness tool registry:

- `study_agent_ask`: delegates a learning request to the existing Study Agent API.
- `study_agent_health`: checks API reachability.

## Environment

Set these variables in the process that launches DeepSeek Harness:

- `STUDY_AGENT_API_URL`: defaults to `http://127.0.0.1:8001/v1/chat/completions`.
- `STUDY_AGENT_WEB_API_KEY`: optional; when set, must match the key configured for the Study Agent API.

Keep local endpoints bound to loopback. Remote endpoints require HTTPS.

## Install / load

The plugin uses the Cordis tool registry and `defineTool` API. DeepSeek Harness is currently in developer preview, and its plugin contracts can change. Load this directory through the plugin or bundle mechanism documented by the exact Harness version you installed. Test in an isolated profile first; this folder is source code, not a universal one-click installer.

Start the Study Agent API separately from the repository root:

```powershell
python examples/deepseek_web_study_agent_api.py
```

Then verify `study_agent_health` before invoking `study_agent_ask`.

The plugin intentionally does not expose shell commands, file writes, browser control, cookies, or other unrestricted local-machine capabilities.
