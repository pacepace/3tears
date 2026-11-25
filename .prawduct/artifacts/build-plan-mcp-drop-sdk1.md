# Build plan: 3tears-mcp drops MCP SDK 1.x

Branch: `feat/mcp-drop-sdk1` (off origin/develop d0ce6f31c). One PR into develop, area mcp.
Critic mode: cumulative at the end.

## Goals

`3tears-mcp` supports both MCP SDK majors (`mcp>=1.27,<3`): `McpServer` picks its handler
registration at import (`_SDK_V1 = hasattr(Server, "call_tool")`), builds results and tool listings
from wire-name dicts that validate on either major, and `scripts/test-mcp-v1.sh` runs its tests on
mcp 1.30 in CI. The features coming next (a call context for handlers, progress and client logging,
an error model, resources, stateful Streamable HTTP) would each be built and tested twice. The owner
ruled (2026-11-25) that 3tears-mcp drops 1.x first, so each is built once against the current SDK.

Success: `packages/mcp/pyproject.toml` declares `mcp>=2.0,<3`; no 1.x branch, shim or 1.x comment is
left in `packages/mcp`; the mcp unit and integration tests pass on the locked 2.3 and on 2.0.0 (the
floor); the unit suite is green.

Out of scope: `3tears-models` (`mcp>=1.27,<3`, the CLI pool's runtime switch) and the `3tears-evals`
`fastmcp` extra (`fastmcp>=3.4.2,<5`). Neither depends on 3tears-mcp, and the ruling names only
3tears-mcp. No version bump (work PRs never bump).

## Decisions taken (inferred; flag at the boundary)

- [DECISION: models and evals keep their 1.x support; `scripts/test-mcp-v1.sh` and its CI step stay
  and run only their tests | the ruling is about 3tears-mcp; dropping 1.x from models (which
  claude-agent-sdk brings mcp into) is a separate call for the owner]
- [DECISION: floor `>=2.0` | the mcp unit suite passes on mcp 2.0.0 unchanged]
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
- `CHANGELOG.md` under Unreleased.

Done when: lint, `typecheck.sh mcp`, `test.sh mcp`, `test-integration.sh mcp`, mcp unit tests on
mcp 2.0.0, `test-mcp-v1.sh` and the unit suite pass; Critic run.
