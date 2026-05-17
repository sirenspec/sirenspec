# Additive execute_streaming() alongside execute() for per-node CLI output

A new `execute_streaming()` async generator is added alongside the existing `execute()` function. It yields a typed `NodeEvent` after each node completes and a final `SummaryEvent`. The CLI uses `execute_streaming()`; all other callers (tests, programmatic API) continue using `execute()` unchanged.

Modifying `execute()` to accept an `on_node_complete` callback was rejected because it mixes rendering concerns into the executor and makes the function signature harder to reason about. Replacing `execute()` with a generator was rejected because it would break all existing callers and tests. The additive approach keeps `execute()` stable and lets the two callpaths diverge without coupling.
