# Study Agent: independent core with multiple adapters

## Decision

Keep the existing Python Study Agent as the canonical implementation. Do not port its reasoning loop into DeepSeek Harness and do not reduce it to a Skill. Expose the same service through multiple adapters:

```text
                  Existing Python Study Agent
          Teacher / Evidence / Learner State / Knowledge Graph
                              |
                 Existing OpenAI-compatible API
                              |
               +--------------+---------------+
               |              |               |
         Harness plugin    MCP Server      Other clients
               |              |
               +---- Study Agent Skill ------+
                    workflow guidance
```

The HTTP API is the transport boundary, not a second copy of the agent. Adapters must stay thin and delegate learning decisions to the existing core.

## Adapter contracts

### Ask
- Input `question`: required non-empty string, maximum 8,000 characters.
- Input `context`: optional relevant context, bounded to 4,000 characters.
- Output `answer`: non-empty final answer.
- Output `model`: model identifier reported by the API, or `study-agent`.

### Health
Returns the existing API's health JSON. This is a connectivity check, not a guarantee that every model provider is healthy.

## Why both native plugin and MCP?

The native plugin integrates directly into DeepSeek Harness and registers tools in its tool registry. MCP makes the same limited operations available to other compatible clients. The Skill describes when and how to use the tools; it is not a substitute for either adapter.

## Security boundary

- The adapters expose learning requests and health checks only.
- They do not expose arbitrary shell execution, filesystem writes, browser control, cookies, or browser/device identity changes.
- Remote API endpoints require HTTPS. Use API authentication and network controls for remote deployments.
- The client does not follow redirects, bounds response size, and avoids returning raw server error bodies to the model.
- Do not send secrets or unrelated personal data in context.

## Validation

Run:

```powershell
python -m pytest tests/test_study_agent_client.py -q
```

For MCP, install `requirements-mcp.txt`, start the existing API, and verify `study_agent_health` followed by one non-sensitive `study_agent_ask` request.

For the Harness plugin, load it in an isolated profile, verify both tools register, then run a health check before asking a test question. DeepSeek Harness is in developer preview, so validate the plugin against the exact installed version before using it in a main profile.
