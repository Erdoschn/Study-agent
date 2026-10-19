# Role context isolation and prompt-cache strategy

## Guarantees

- A context scope is bound to a canonical user profile and a conversation/task ID. HTTP requests without a task ID receive a fresh ID; the UI sends its conversation ID for continuity.
- Each model call is tagged with a role (task_analyzer, reasoner, teacher, or teaching_validator). The bounded in-memory role ledger stores that role's own prompt/response turns; it never merges one role's history into another role's history.
- Shared task facts and handoffs are explicit records in RoleContextManager. They are keyed by user and task and are not global knowledge. The current AgentState remains the source of truth for the live execution's task facts and evidence.
- Token usage events can include role, user-profile and task identifiers alongside provider-reported input/output/cache tokens. These are identifiers/metadata only; prompts, responses, API keys and file contents are not added to telemetry.
- stable_prefix_fingerprint fingerprints only caller-supplied stable prompt components. Keep system policy, role instructions, output schema and tool protocol in stable prefixes; put changing task facts and user inputs later in the request.
- application_cache_key includes user, task, role, provider, model, prompt version, input digest, evidence/tool versions and permission scope. It is an identity helper, not a result cache or a guarantee that a provider supports prompt caching.

## Cache and invalidation rules

1. Provider prompt caching is provider/model-specific. Preserve byte-identical stable prefixes and consult the provider's cache rules; a fingerprint alone cannot force a cache hit.
2. Never reuse private application results across users. User identity and permission scope are part of the cache key.
3. Invalidate application results when the prompt, model/provider, task input, evidence, tool implementation or permission scope changes.
4. Do not inject the whole role ledger back into prompts just to claim continuity. AgentReasoner already receives the current task state and recent observations; Teacher receives a structured teaching payload; Validator receives its own validation payload. Role histories are bounded diagnostics/context records and are not blindly replayed into every request, which would increase tokens and undermine stable-prefix reuse.
5. The role ledger is in-memory and bounded; it is not a durable transcript store. Persistent learning knowledge remains in the existing user-isolated storage layer.

## Validation

Run python -m pytest test/test_context_management.py test/test_token_usage.py, followed by the full test suite. The GitHub Actions result is authoritative for the branch; no passing result should be assumed while checks are queued.
