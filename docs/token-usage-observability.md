# Token usage and cache observability

The OpenAI-compatible API client records usage metadata from successful provider responses. The telemetry is deliberately separate from agent reasoning and never stores prompts, completions, API keys, or raw response bodies.

## What is recorded

Each API request writes one JSONL event containing the UTC timestamp, provider, model, status, duration, input/output/total token counts, available prompt-cache hit/miss counts, and an optional cost estimate. A provider that does not return token usage is marked as unavailable; its usage is not misreported as zero. Browser-backed model sessions do not expose the same structured usage object and therefore are not assigned invented token counts.

The cache-hit rate is token-weighted:

    total prompt-cache hit tokens / total prompt tokens for requests that report cache counters

Only requests whose provider reports cache counters contribute to the cache-rate denominator. This avoids treating missing cache metrics as cache misses.

## Configuration

The defaults live in config/observability.json. The local config/providers.json may override the observability object without changing the tracked defaults.

- enabled: enable or disable usage logging.
- path: JSONL output path; relative paths are resolved from the repository root.
- console_log: emit a short metadata-only line for each event.
- max_file_bytes and backup_count: rotate the JSONL file to bounded numbered backups.
- Optional model pricing belongs in each local model entry under pricing_usd_per_million, with input, cache_hit_input, and output prices expressed in USD per one million tokens. No prices are assumed or embedded in the repository. Estimates are only emitted when the required configured prices and token counters are available.

Usage logs are runtime data and are ignored by Git. They can still contain usage patterns (provider/model, time, and volume), so do not share them publicly without reviewing them.

## Viewing the report

Run this from the repository root:

    python scripts/token_usage_report.py

To choose a specific log:

    python scripts/token_usage_report.py --input data/token_usage.jsonl

For machine-readable output:

    python scripts/token_usage_report.py --json

The report combines the current log with the configured number of rotated backups. Token totals are cumulative only within the retained log files; rotation intentionally bounds local disk use.
