# Study Agent multi-adapter plan

Keep the Python Study Agent as the source of truth. Expose it through a thin native DeepSeek Harness plugin, an optional MCP server, and a reusable Skill. The adapters must delegate reasoning and teaching to the existing API instead of copying core logic.
