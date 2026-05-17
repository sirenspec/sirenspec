# Provider-layer mock for sirenspec test cassettes

`sirenspec test --mock` intercepts LLM calls at the provider registry boundary, not at the HTTP level. When mock mode is active, the executor's provider registry is replaced with a mock registry that returns responses from a cassette file keyed by `(model_uri, sha256(messages))`. `--record` runs live and writes the cassette; `--mock` replays it.

HTTP-level interception (VCR-style via `vcrpy`) was rejected because cassettes would embed raw HTTP details (headers, auth tokens, wire encoding) that are internal to each provider SDK and change independently of workflow behaviour. Provider-layer mocking was also preferred over executor-level static injection because it validates that the real prompt construction produces the expected LLM input, not just that routing and assertions work.
