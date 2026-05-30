# TUI QA Report — 2026-05-30 (claude-code-mini run)

## Summary

1 new finding, **P0 — FIXED**. The studio could not execute any workflow that
contained a `type: human` node. The full claude-code-mini demo
(`sirenspec-demos/claude-code-mini/workflow.yaml`) is unusable inside
`sirenspec launch` for this reason; agent + tool nodes themselves are wired
correctly (Python tools import via the workflow-dir `sys.path` shim in
[app.py:626](src/sirenspec/session/app.py:626), and the recent
`interpolate_tool_config` change does propagate `{{ propose_edits.output }}`
into `apply_edits` / `run_command` tool args), so the *only* thing blocking
the demo from working seamlessly is the missing human-input bridge.

## Findings

### [P0 — FIXED] `type: human` approval nodes hang / fail silently inside the studio

**Fix landed in:**
- [`runtime.py`](src/sirenspec/session/runtime.py): `stream_execution`,
  `take_turn`, and `run_full` now accept and forward `human_input_fn` to
  `execute_streaming`.
- [`app.py`](src/sirenspec/session/app.py): new
  `LaunchApp.await_human_input(prompt_text)` renders the HumanNode prompt into
  the transcript, focuses the chat box, and resolves a pending future on the
  next `Input.Submitted`. `on_input_submitted` checks `_pending_approval`
  first so the next submission answers the prompt instead of starting a new
  chat turn. `handle_turn` / `command_run` wire the coroutine in.
- Regression: [`tests/test_launch_human_in_loop.py`](tests/test_launch_human_in_loop.py)
  now drives a real key-press approval flow through Pilot (no `xfail`) and
  pins that `human_input_fn` is forwarded.


- **What**: As soon as a turn reaches a HumanNode (`approve_edits` /
  `approve_command` in claude-code-mini) the TUI either blocks forever or —
  when the node sets `on_timeout: use_default` — silently injects the default
  response without ever showing the prompt to the operator. There is no UI at
  all for the approval prompt: no inline question in the transcript, no modal,
  no rebinding of the chat input. The "live" indicator stays lit while the
  user has no way to answer.

- **Repro** (against the real claude-code-mini workflow):
  1. `sirenspec launch sirenspec-demos/claude-code-mini/workflow.yaml`
  2. In the chat box: `change fibonacci.py to use an iterative implementation`
     (anything the triage agent classifies as `edit`).
  3. Workflow walks `classify → read_files → propose_edits → approve_edits`.
  4. At `approve_edits` the executor calls
     [`stdin_input`](src/sirenspec/core/human_runner.py:34), which writes the
     prompt to stderr (invisible — Textual owns the screen) and blocks on
     `sys.stdin.readline` (Textual owns stdin too).
  5. After the configured `timeout: 300`, `on_timeout: use_default` fires with
     `default_output: "no"`, the workflow skips `apply_edits`, and the user
     sees a final summary saying *(no files were changed)* with no indication
     that an approval was ever requested.

- **Mechanism**: Both entry points in the runtime invoke `execute_streaming`
  **without** a `human_input_fn`:
  - [`stream_execution`](src/sirenspec/session/runtime.py:156) — called by
    `WorkflowSession.take_turn` (chat turns).
  - [`WorkflowSession.run_full`](src/sirenspec/session/runtime.py:326) —
    called by `/run`.

  The executor's HumanNode branch
  ([executor.py:1096](src/sirenspec/core/executor.py:1096)) therefore falls
  back to `stdin_input`, which is incompatible with Textual's input loop. The
  TUI itself defines no command, transcript widget, or focus shim to surface
  the rendered prompt and capture a reply via the existing `CommandInput`.

- **Evidence**:
  - Failing regression:
    `tests/test_launch_human_in_loop.py::test_chat_turn_with_human_node_completes_without_blocking`
    (marked `xfail(strict=True)`) — drives the chat box through real key
    presses, monkeypatches `stdin_input` to assert it is never called, and
    fails today.
  - Pinning test:
    `tests/test_launch_human_in_loop.py::test_runtime_does_not_forward_human_input_fn_today`
    spies on `execute_streaming` and confirms `human_input_fn` is absent from
    its kwargs. Invert this assertion once the bug is fixed.

- **Triage / suggested fix**:
  1. Thread `human_input_fn` through `WorkflowSession.take_turn` and
     `run_full` into `execute_streaming`.
  2. Have `LaunchApp` provide a coroutine that:
     - posts the rendered prompt to the transcript as a notice (or a
       dedicated "approval card" widget),
     - flips `CommandInput` into "approval mode" so the next
       `Input.Submitted` resolves the coroutine's future instead of starting a
       new chat turn,
     - shows an explicit "awaiting approval" indicator in the status bar so
       the live state is honest about what it's waiting on.
  3. Respect `HumanNode.timeout` by surfacing a countdown and routing the
     default-output fallback through the same notice mechanism, so users see
     why a turn moved on without their input.

- **Regression test**: `tests/test_launch_human_in_loop.py` (added). Strict
  `xfail` flips to a hard failure the moment the fix lands, forcing the marker
  to come off.

## Out-of-scope observations (not bugs, surfaced while debugging)

- The Python tool adapter and `{{ … }}` interpolation through tool-node args
  both work correctly in the TUI — `apply_edits` and `run_shell` *would*
  execute fine if the approval gate ever opened. The cwd → `sys.path` shim at
  [app.py:626](src/sirenspec/session/app.py:626) is what makes the demo's
  `import tools` resolve.
- The summarizer's `{{ inputs.history }}` template variable is never populated
  by the TUI runtime (which threads history into `user_input` instead via
  [`build_turn_input`](src/sirenspec/session/runtime.py:93)). The demo's
  `workflow.yaml` was written for the demo's `main.py`, which passes
  `initial_inputs={"history": transcript}`. This is a workflow-vs-runtime
  contract mismatch worth documenting, but per the goal we treat the workflow
  as canonical — flagged here for follow-up only.
