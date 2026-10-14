# Security, Privacy, and Maintainability Review

Date: 2026-10-14  
Branch: `audit/security-readability-hardening`  
Status: review changes are in PR #23; not merged.

## Review scope

This pass reviewed the HTTP entry points, request parsing, local browser lifecycle, Coder workspace and project path boundaries, sandbox invocation, backup and persistent-memory stores, SQLite knowledge graphs, Coder-to-Study-Agent bridge, logging, and the two browser UIs. The existing sandbox command construction was also inspected for shell use, resource limits, network isolation, and filesystem access.

This is a targeted repository review with regression tests, not a formal penetration test or a claim that every code path is vulnerability-free. The browser/provider paths still need controlled integration testing on the target Windows host.

## Findings addressed

| Area | Finding | Change |
| --- | --- | --- |
| Local Coder API | Wildcard CORS allowed any website to attempt browser access to the local API. | Removed wildcard allow-origin; allow only the matching loopback UI origin and reject other browser origins. |
| HTTP request framing | Request handlers parsed unbounded or ambiguous bodies; a slow client could hold a request open. | Shared bounded body parser rejects duplicate/invalid Content-Length, Transfer-Encoding, truncation, oversized lengths, and reads exceeding a deadline. |
| Model API exposure | The standalone DeepSeek Web API did not require a key before binding to a non-loopback address. | Check exposure policy before browser/server startup; require an API key for non-loopback binds. |
| Browser ownership | A synchronous Playwright-backed client could be used by multiple HTTP request threads. | Use the dedicated browser-session wrapper and serialize browser requests in the standalone API. |
| Error disclosure | Some API error responses included raw provider or exception text. | Return generic errors with reference IDs; keep diagnostic logs to error types/references. |
| Diagnostic privacy | Credentials and host paths could appear in logs and exception strings. | Shared redaction is applied before debug sinks; Coder task failures and bridge context are also redacted. |
| Persistent Coder memory | Per-instance locking and a shared temporary filename could race; corrupt JSON could be treated as empty and overwritten. | Use per-root locks, unique atomic temp files, restrictive POSIX file modes, size and symlink/reparse checks, and fail closed on corrupt histories. Memory failures no longer prevent the coding run from producing its result or releasing its run lock. |
| Project and backup paths | Resolving a path before checking it could conceal an internal symlink; archive reads were not fully bounded. | Check lexical project/backup paths before resolving, reject unsafe project names and linked archive files, and cap archive entry counts, total size, manifest size, and per-file reads. |
| SQLite persistence | A database or SQLite sidecar could be a symlink; linked parent directories could redirect storage elsewhere. | Centralized validation checks the database, WAL/shared-memory/journal files, and every parent directory component before opening the DB. |
| Browser profile privacy | Browser profiles contain login cookies and had no explicit path-link or POSIX permission check. | Reject linked profile directories and set POSIX profile-directory permissions to 0700. Windows relies on the account/parent ACL; see residual risks below. |
| Bridge egress | Redirects could send a request and its Authorization header to a different endpoint; context may contain secrets. | Validate bridge URLs, require HTTPS for non-loopback destinations, disable redirects, cap response size, and redact credentials/host paths before sending context. |
| Repeated security code | HTTP framing/origin checks, credential patterns, and some failure handling were duplicated. | Extract shared helpers in `core/http_security.py`, `core/redaction.py`, and `core/storage_security.py`; reuse them from API, harness, bridge, logging, and persistence code. |

## Defense in depth retained

- The Coder sandbox uses fixed argument arrays with `shell=False`, a container with network disabled, read-only root filesystem, dropped capabilities, no-new-privileges, and process/memory/CPU limits.
- The workspace filesystem rejects path traversal and reparse points, restricts readable/writable file types, bounds file sizes, and atomically publishes writes.
- The browser UI renders untrusted model Markdown through DOMPurify when available and falls back to HTML-escaped text if the sanitizer is unavailable. Dynamic history/file/status text is rendered with `textContent`.
- Persisted feedback is treated as a preference signal, not as an instruction that can override the current user request or harness boundary.

## Verification

The latest successful GitHub Actions run for this branch reported:

- Python 3.11: **617 passed**
- Python 3.13: **617 passed**
- Coder sandbox security job: **passed**

Regression coverage includes HTTP Origin and framing validation, bridge redirect/URL safety, credentials and path redaction, concurrent memory updates, corrupt-memory fail-closed behavior, project/backup/SQLite/profile symlink checks, and recovery when memory storage fails.

## Residual risks and operational guidance

1. **HTTP is not encryption.** A non-loopback API key protects against unauthenticated callers but does not encrypt prompts, responses, or credentials. Keep the service bound to loopback, or place a remotely accessible service behind a correctly configured TLS reverse proxy. Do not expose these development APIs directly to the public Internet.
2. **Local OS trust still matters.** These controls do not defend against a malicious process already running as the same Windows user or an administrator. The browser profile’s Windows ACL is inherited from the parent directory; this pass rejects links but does not rewrite Windows DACLs. Keep the workspace/profile under a user-only directory and protect the Windows account.
3. **Redaction is heuristic.** Common credential formats and host paths are masked, but no regex can recognize every secret. Avoid entering credentials into prompts, and rotate any credential that may have been exposed.
4. **Provider data flow is intentional.** When the user explicitly asks the model to analyze code or attachments, task content must be sent to the configured model provider. Redaction reduces accidental leakage of obvious credentials; it is not a guarantee that all proprietary or personal content is removed.
5. **Runtime integration still matters.** CI validates deterministic code paths and sandbox policy, but it cannot fully simulate Edge/DeepSeek login state, Windows ACL inheritance, antivirus locks, enterprise proxies, or every browser DOM change. Run the Coder/Study Agent locally with non-sensitive sample files before processing private projects.

No changes in this review should be described as a guarantee of absolute security. The goal is to close identified failure modes, reduce duplicated security logic, fail closed at sensitive boundaries, and give failures safe, testable behavior.
