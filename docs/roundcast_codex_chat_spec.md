# ROUNDCAST local Codex chat v1

## Objective and assumptions
The user requested a dialogue box below the model and asked to continue implementation.
Add a gray/white, local-only explanatory chat panel below the existing model dashboard.
Each explicit send shares the question, up to six prior user/assistant exchanges, and
the currently selected round snapshot and last successful prediction with local Codex CLI.
This is a separate ROUNDCAST conversation, not an embedding of the desktop daily task.
No model retraining, repository editing, arbitrary commands, uploads, or new predictions
are authorized by this panel. Do not send hidden outcomes or frozen reference predictions.

## Stack, structure, and commands
Use the existing Python stdlib HTTP server and plain HTML/CSS/JavaScript. No dependency changes.
- `src/csdemo/roundcast_chat.py`: bounded job manager, trusted context, Codex subprocess.
- `src/csdemo/roundcast_server.py`: loopback/same-origin chat routes and prediction cache.
- `web/roundcast/chat.js`, `index.html`, `styles.css`, `app.js`: panel and state integration.
- `tests/test_roundcast_chat.py`, `tests/roundcast_chat.test.cjs`: failure and state checks.
- Test: `C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests`
- Run: `C:\Users\admin\11\envs\game\python.exe -m src.csdemo.roundcast_server --host 127.0.0.1 --port 8765`

## Style
Follow existing functions/classes and explicit validation, e.g. `raise ValueError("Invalid selection")`.
Render text with `textContent`, never HTML supplied by users or Codex. Fail closed with
short Chinese errors; never return subprocess diagnostics or credentials to the browser.

## HTTP contract
- GET `/api/chat/status`: `{available: boolean, message: string}` checks installed/logged-in CLI; not proof of live network.
- POST `/api/chat`: exact `{message, example_id, stage, algorithm, request_id, history}`.
  Message 1–2000 characters. `request_id` is null or a cached matching successful prediction UUID.
  History is at most 12 alternating `{role: 'user'|'assistant', text: string}` messages,
  at most 24000 characters total; it is untrusted conversation, never application instructions.
  Returns 202 `{job_id, status: 'running'}`. At most one active Codex job per server.
- GET `/api/chat/jobs/<uuid>`: `{job_id, status: 'running'|'success'|'error'|'cancelled', reply?, message?}`.
- POST `/api/chat/cancel`: `{job_id}`. Cancels only the active child owned by this manager.
Messages/results remain in memory only (bounded cache, expire after 30 minutes).

## Plan and checkpoints
1. Add contract and failing backend/frontend tests.
2. Implement independent chat job manager + UI in parallel, then integrate server.
3. Verify context spoofing, no spoilers, validation/limits, concurrency, cancellation,
   timeout, login/network errors, safe rendering and repeat-send prevention.
4. Run existing model regressions; perform a real local Codex send and browser preview.

## Boundaries and success criteria
- Always: read-only sandbox, approval never, disable shell/browser/plugins/connectors/agents,
  ignore user configuration, temporary empty working directory, fixed executable arguments,
  stdin prompt, explicit server-side data allowlist. Credentials stay in existing Codex storage.
- Ask first: public deployment, account changes, new dependencies, code-writing chat mode.
- Never: copy/read auth tokens into application code, return logs/secrets, fabricate AI replies,
  silently substitute reference probabilities or another model provider.
- A send attaches the current context; a failed/new prediction does not reuse an older result.
- Chat failure/cancellation must not break model predictions. Show send/disclosure/wait/error/clear
  states and keep an answer's original context label even if a later model run differs.
- The existing model remains T05 backend / T04 UI; this feature does not silently complete T06.

## Sources
Official non-interactive execution and authentication behavior:
https://learn.chatgpt.com/docs/non-interactive-mode
Local CLI help and `codex features list` are checked against the installed version.
