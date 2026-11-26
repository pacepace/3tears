---
artifact: build-plan
version: 1
scope: mcp-drop-sdk1
branch: feat/mcp-drop-sdk1
depends_on: []
partition: "serial -- three packages drop one SDK major together; their floor run and CI wiring are shared"
last_validated: 2026-11-25
---

# Build plan: 3tears drops MCP SDK 1.x

Off origin/develop d0ce6f31c. One PR into develop, areas mcp, models, evals. Critic mode: cumulative
at the end.

## Goals

3tears supports both MCP SDK majors. `3tears-mcp` (`mcp>=1.27,<3`) picks its handler registration at
import (`_SDK_V1 = hasattr(Server, "call_tool")`) and builds results and tool listings from wire-name
dicts that validate on either major; `3tears-models`' CLI pool (`mcp>=1.27,<3`) rebinds a copied
tool server's handlers one way per major; the `3tears-evals` `fastmcp` extra (`fastmcp>=3.4.2,<5`)
admits fastmcp 3, which needs mcp 1; and `scripts/test-mcp-v1.sh` runs their tests on mcp 1.30 in CI.
The features coming next (a call context for handlers, progress and client logging, an error model,
resources, stateful Streamable HTTP) would each be built and tested twice. The owner ruled
(2026-11-25) that 3tears-mcp drops 1.x first, and then (2026-11-25, later the same day) that
3tears-models and the evals `fastmcp` extra drop it too, on this branch.

Success: 3tears-mcp and the models `claude-cli` extra declare `mcp>=2.0,<3`, and the evals `fastmcp`
extra `fastmcp>=4.0,<5`; no 1.x branch, shim or 1.x comment is left in `packages/mcp`,
`packages/models` or `packages/evals`; `scripts/test-mcp-v1.sh` and its CI step are gone; the MCP
tests of the three packages pass on the locked versions and, in a recurring check, on the floors
(mcp 2.0.0, fastmcp 4.0.0); the unit suite is green.

Out of scope: no version bump (work PRs never bump).

## Decisions

Owner rulings (2026-11-25):

- [DECISION: 3tears-mcp drops mcp 1.x | owner ruling]
- [DECISION: 3tears-models and the evals `fastmcp` extra drop mcp 1.x too, and
  `scripts/test-mcp-v1.sh` and its CI step go | owner ruling, superseding the earlier inferred
  decision that they keep it]

Technical decisions taken here (the owner left the floor to the builder):

- [DECISION: floor `mcp>=2.0` for 3tears-mcp and models, `fastmcp>=4.0` and `fastmcp-slim>=4.0` for
  the evals extra, kept true by `scripts/test-mcp-floor.sh` | the lowest versions the floor run
  proves: the mcp, models and evals MCP tests pass on mcp 2.0.0 with fastmcp 4.0.0, and fastmcp 4.0.0
  resolves with mcp 2.0.0. A known consumer resolves mcp 2.2, so raising the floor to the locked 2.3
  would move it. The script reads the floors from the pyprojects, overlays exactly those versions on
  the locked environment, checks the overlay took, and runs the tests; CI's `check` job and
  `check-all.sh` run it, and `tests/enforcement/test_mcp_floor.py` keeps the models floor equal to
  3tears-mcp's, the fastmcp bounds readable and shared with the dev install, and the CI step present]
- [DECISION: no test is deleted | none exists only for the 1.x path. The tests carried 1.x shims
  (`streams[0], streams[1]` for 1.x's three-value client, `McpError`/`MCPError` lookup, a
  `skipif(not hasattr(mcp, "Client"))` on the 2026-07-28 protocol class, the evals `wire()` reader
  by wire name, the fastmcp 3 `initialize_result` fallback, the models recorder's 1.x handler
  lookups); the shims go and every assertion stays, reading 2.x's field names.
  `TestTheSdkDecoratorsDutiesHoldOnEitherSdk` is renamed `TestMcpServerDoesWhatTheLowLevelSdkDoesNot`,
  same tests]
- [DECISION: results and listings are built with 2.x's constructor keywords; a listing sets only the
  optional fields the tool sets | the wire-name dicts existed only to validate on both majors]

## Chunk 01 (the only one)

- `packages/mcp/src/threetears/mcp/server.py`: remove `_SDK_V1` and the 1.x registration branch;
  build `CallToolResult` and `Tool` with keywords; drop 1.x rationale from comments, keeping the
  present reasons (the low-level SDK checks no input schema, wraps no content list, checks no output
  schema).
- `packages/mcp/src/threetears/mcp/tool.py`, `packages/mcp/README.md`, `packages/mcp/pyproject.toml`,
  `uv.lock`.
- `packages/models/src/threetears/models/claude_cli_pool.py`,
  `packages/models/src/threetears/models/_mcp_sdk_internals.py`, `packages/models/pyproject.toml`,
  `packages/models/tests/unit/models/claude_cli_recorder.py`: the 2.x path only.
- `packages/evals/pyproject.toml`, `pyproject.toml` (dev install), `packages/evals/tests/test_fastmcp_transport.py`,
  `packages/evals/tests/test_apparatus_settings.py`: fastmcp 4 only; `mcp_wire.py` removed.
- mcp tests: shims out, as above.
- `scripts/test-mcp-floor.sh`, `.github/workflows/ci.yml`, `scripts/check-all.sh`,
  `tests/test_check_all_script.py`, `tests/enforcement/test_mcp_floor.py`, `docs/testing.md`,
  `README.md`, `CLAUDE.md` (merged onto the 0.95.0 line, where the scripts table lives in `docs/dev-workflow.md`,
  that file carries the change): the floor run covers the three packages; `scripts/test-mcp-v1.sh` removed.
- `CHANGELOG.md` under Unreleased, with the consumer check: of the known applications that install
  3tears-mcp, 3tears-models or the evals extra, three resolve mcp 2.2 or 2.3; four still resolve mcp
  1.x and move before they take this release; none resolves fastmcp 3.

Done when: lint, typecheck, `test.sh` for mcp, models and evals, `test-integration.sh mcp`,
`test-mcp-floor.sh`, `tests/test_check_all_script.py` and the unit suite pass; Critic run.
