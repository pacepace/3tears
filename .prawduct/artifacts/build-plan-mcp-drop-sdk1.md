---
artifact: build-plan
version: 1
scope: mcp-drop-sdk1
branch: feat/mcp-drop-sdk1
depends_on: []
partition: "serial -- one package's SDK path and its tests; a second agent would collide in server.py"
last_validated: 2026-11-25
---

# Build plan: 3tears-mcp drops MCP SDK 1.x

Off origin/develop d0ce6f31c. One PR into develop, area mcp. Critic mode: cumulative at the end.

## Goals

`3tears-mcp` supports both MCP SDK majors (`mcp>=1.27,<3`): `McpServer` picks its handler
registration at import (`_SDK_V1 = hasattr(Server, "call_tool")`), builds results and tool listings
from wire-name dicts that validate on either major, and `scripts/test-mcp-v1.sh` runs its tests on
mcp 1.30 in CI. The features coming next (a call context for handlers, progress and client logging,
an error model, resources, stateful Streamable HTTP) would each be built and tested twice. The owner
ruled (2026-11-25) that 3tears-mcp drops 1.x first, so each is built once against the current SDK.

Success: `packages/mcp/pyproject.toml` declares `mcp>=2.0,<3`; no 1.x branch, shim or 1.x comment is
left in `packages/mcp`; the mcp unit and integration tests pass on the locked 2.3, and the mcp unit
tests pass on 2.0.0 (the floor) in a recurring check; the unit suite is green.

Out of scope: `3tears-models` (`mcp>=1.27,<3`, the CLI pool's runtime switch) and the `3tears-evals`
`fastmcp` extra (`fastmcp>=3.4.2,<5`). Neither depends on 3tears-mcp, and the ruling names only
3tears-mcp. No version bump (work PRs never bump).

## Decisions taken (inferred; put to the owner in the PR body, not ruled)

These three are not owner rulings. Until the owner rules on them they describe the current state of
the branch, and the PR body asks for a ruling on each.

- [DECISION: models and evals keep their 1.x support; `scripts/test-mcp-v1.sh` and its CI step stay
  and run only their tests | the ruling is about 3tears-mcp; dropping 1.x from models (which
  claude-agent-sdk brings mcp into) is a separate call for the owner]
- [DECISION: floor `>=2.0`, kept true by `scripts/test-mcp-floor.sh` | the mcp unit suite passes on
  mcp 2.0.0 unchanged, and a known consumer resolves mcp 2.2, so raising the floor to the locked 2.3
  would move it. The floor script reads the floor from the pyproject, overlays exactly that mcp on
  the locked environment and runs the mcp unit tests; CI's `check` job and `check-all.sh` run it,
  and `tests/enforcement/test_mcp_floor.py` fails if the CI step goes away. So a later use of a name
  added after 2.0 goes red here rather than at a consumer]
- [DECISION: no test is deleted | none exists only for the 1.x path. The tests carried 1.x shims
  (`streams[0], streams[1]` for 1.x's three-value client, `McpError`/`MCPError` lookup, a
  `skipif(not hasattr(mcp, "Client"))` on the 2026-07-28 protocol class); the shims go and every
  assertion stays. `TestTheSdkDecoratorsDutiesHoldOnEitherSdk` is renamed
  `TestMcpServerDoesWhatTheLowLevelSdkDoesNot`, same tests]
- [DECISION: results and listings are built with 2.x's constructor keywords; a listing sets only the
  optional fields the tool sets | the wire-name dicts existed only to validate on both majors]

## Chunk 01 (the only one)

- `server.py`: remove `_SDK_V1` and the 1.x registration branch; build `CallToolResult` and `Tool`
  with keywords; drop 1.x rationale from comments, keeping the present reasons (the low-level SDK
  checks no input schema, wraps no content list, checks no output schema).
- `tool.py`, `README.md`, `pyproject.toml` (floor, jsonschema comment), `uv.lock` (one specifier).
- mcp tests: shims out, as above.
- `scripts/test-mcp-v1.sh`, `.github/workflows/ci.yml`, `scripts/check-all.sh`, `docs/testing.md`:
  the 1.x suite runs models and evals only.
- `scripts/test-mcp-floor.sh` (new), its CI `check` step and `check-all.sh` line,
  `tests/enforcement/test_mcp_floor.py`, `docs/testing.md`: the mcp unit tests run at the floor.
- `CHANGELOG.md` under Unreleased, with the consumer check: of the known applications that install
  3tears-mcp, two resolve mcp 2.2 or 2.3; three still resolve mcp 1.x and move before they take
  this release; none combines 3tears-mcp with fastmcp 3.

Done when: lint, `typecheck.sh mcp`, `test.sh mcp`, `test-integration.sh mcp`,
`test-mcp-floor.sh` (the mcp unit tests on mcp 2.0.0), `test-mcp-v1.sh` and the unit suite pass;
Critic run.
