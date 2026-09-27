# Testing 3tears

The three suites, why two of them sit outside `./scripts/test.sh`, how to make the integration
suite run everything it can, and how test fakes declare what they stand in for. The rules are in
`CLAUDE.md`; this is the detail behind them.

## The sidecar suite

`./scripts/test-sidecar.sh` runs the nodriver sidecar's own tests. nodriver is AGPL-3.0 and never
enters the workspace venv, so `test.sh` carries `--ignore` for the sidecar and cannot run these.
Separate but not optional: `check-all.sh` runs it. Until it existed, a ruff autofix wrote a syntax
error into `hitl.py` that passed lint, mypy, and the entire workspace suite.

## The integration suite

`./scripts/test-integration.sh` runs the integration tests. `test.sh` excludes them with
`-m "not integration"`, and `check-all.sh` does not run them either: they spin real NATS and
Postgres containers and need Docker, so folding them into the default gate would break it
wherever Docker is absent. **CI cannot run them at all -- GitHub Actions has no Docker -- so
nothing but a developer, locally, ever executes them.**

**Run them before any PR.** Cross-pod behaviour lives entirely there, and a green
`check-all.sh` says nothing about it. `project-state.yaml` lists this as the third declared test
command, so recorded evidence that omits it covers two suites out of three.

### A skip is not a pass

The suite exits 0 with tests skipped, so a skipped test reads exactly like a passing one in the
summary line. Run it as `./scripts/test-integration.sh -rs` and account for every skip.

This is not hypothetical: `test_a_real_display_is_driven_through_the_pipe` was unpassable on
every machine for three weeks after the 2026-08-18 change that defaulted the sidecar's
`BIND_HOST` to loopback (correct for the shipping Kubernetes shape, wrong for a testcontainer with
its own network namespace). Nobody saw it, because the test skips when the sidecar image is absent
and CI never runs the suite at all -- the skip that hid the breakage was also the reason nobody
noticed. It surfaced only when a release stopped to ask why 35 tests were skipping.

### Build the sidecar image first

Or its integration tests skip:

```bash
docker buildx bake --file docker-bake.hcl nodriver-sidecar \
  --set nodriver-sidecar.platform=linux/$(docker version --format '{{.Server.Arch}}') --load
```

The bake target is multi-platform and the local `docker` driver refuses that ("Multi-platform
build is not supported for the docker driver"), hence `--set ... platform` and `--load`. A bare
`docker buildx bake nodriver-sidecar` exits non-zero having built nothing -- and piping it through
`tee` masks that exit code, which is how it looked like it had worked.

**Build for the Docker host's own architecture**, which is what the `docker version`
substitution picks (`arm64` on Apple silicon, `amd64` on an Intel box). A `linux/amd64` image on
an arm64 host builds and loads, then runs under emulation and never becomes healthy, so its
integration tests fail rather than skip. A native build passes.

### Legitimate skips on a dev box

They need credentials or tools this repo does not ship: the Redshift live tests
(`OTS_REDSHIFT_PASSWORD`), the backup suites (`pg_dump`/`pg_restore`/`psql` on PATH), and one
deliberate manual microbenchmark. Anything else is a test turned off by accident. When reporting
results, state the pass count AND the remaining skips with their reasons -- "integration green"
on its own is not a report.

## Test fakes

A test fake is any class named `Fake<Name>` or `_Fake<Name>` under a `tests/` directory. Every one
declares what production protocol it stands in for. Three routes, in order of preference:

1. **Subclass it.** `class _FakeKv(KvBucketLike):`. The walker accepts any non-`object` base and
   checks nothing further, on the theory that a type checker covers it. **In this repo it does
   not:** mypy runs over `packages/*/src` only, so a subclassed fake's surface is unverified by
   anything. Prefer it anyway for a real Protocol, because the base documents intent and an IDE
   follows it. Reach for route 2 when you want the surface actually compared.
2. **`# parity-with: <fully.qualified.name>`** on the line above the class. The walker imports
   the target and compares method surfaces. This is the only route that verifies anything.
3. **`# parity-exempt: <rationale>`** on the line above the class, for a hand-rolled subset stub
   with no single production protocol to name. The rationale must be at least 30 characters and
   must not be a blanket phrase like "tests need this" or "temporary". **Keep it on one line,
   however long.** The walker reads the first non-blank line above the class and stops, so a
   wrapped rationale exempts nothing.

Workspace tests centralise their asyncpg and workspace-entity shells under
`packages/agent/workspace/tests/_helpers/`, so per-test inline fakes need only a one-line
subclass declaration.

**Exempt in place, not in `tests/enforcement/_fake_parity_exemptions.txt`.** That file still
parses and is deliberately empty. Its entries are keyed `path:LINE:symbol`, so one added import
shifts every fake below it and the gate fails with `no_declaration` for a fake nobody touched. A
marker on the class moves with the class.

`tests/enforcement/test_fake_protocol_parity.py` enforces this. It is a thin shell over the
canonical walker in `packages/enforcement/src/threetears/enforcement/fake_parity/`.
`FAKE_PARITY_ENFORCEMENT_MODE` defaults to `strict`. It catches the drift class where production
protocols evolve while test fakes rot silently, until some downstream test happens to call the
missing method.
