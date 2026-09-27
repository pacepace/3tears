# Releasing 3tears

How the family is versioned, how a release is cut, and how an already-tagged version is
republished. The hard rules are in `CLAUDE.md`; this is the reasoning and the procedure behind
them.

## Why the family versions in lockstep

Every `3tears*` package releases at the same version, and every intra-family dependency carries
the bound `>=<major>.<minor>.0,<major>.<minor+1>.0`. Unbounded siblings let pip resolve a mixed
family, and that produces two failure modes, both brutal to diagnose.

**A mixed install that builds clean and breaks at runtime.** pip paired `3tears-object-store`
0.18.0 with an otherwise-0.19.0 family in the hub image. 0.18.0 predates `build_object_key`'s
`path=` parameter. Nothing failed at build time.

**A resolution failure that names the wrong package.** With about 17 published versions across
about 25 mutually-unbounded packages, pip backtracks the cross-product and dies with
`ResolutionImpossible` or `resolution-too-deep` against whatever node it was holding. One real
failure reported `no matching distributions available for your environment: 3tears-agent-tools`.
The actual cause was a stale `protobuf` pin in a consumer's constraints file, three levels away.
That message cost most of a day. It sends you hunting registry access, private indexes, and
extras, none of which were the problem.

Bounding makes a mixed family unresolvable rather than merely unlikely. It also collapses the
search space, so pip blames the package that actually conflicts.

`tests/enforcement/test_intra_family_version_bounds.py` enforces the bounds; when the family
version is bumped, that test names every bound to move. Consumers pin the whole family to one
exact version too -- see the matching warning in `14-eng-ai-bot/CLAUDE.md`.

An intra-family API addition ships in a MINOR bump
(`tests/enforcement/test_api_growth_requires_a_minor_bump.py`).

## Cutting a release

1. Bump the version.
2. PR into `develop`.
3. PR `develop` into `main`, with no version bump on that second PR.
4. Tag from `main`.

Do not cut a release tag on a plain develop-to-main sync that is not meant to ship.

**"Tag from main" means push a tag.** Run `git tag -a vX.Y.Z <commit on main>` then
`git push origin vX.Y.Z`. The tag push is the trigger. It is the only path that creates the
GitHub Release.

**Do not run `gh workflow run release.yml` to cut a release.** That is the republish command
below. It deliberately creates no release. Using it to cut a new one publishes to PyPI while
leaving no tag and no release behind, with every job green. That happened on 2026-08-01 with
0.22.5.

**A green release run does not mean a release exists.** Confirm with
`git ls-remote --tags origin`, and check that `github-release` did not report `skipped`.
`tag-on-main` verifies the ref is on main and creates nothing, whatever its name suggests.

## Republishing an already-tagged version

Use this when a package missed the upload, or a partial publish needs completing. Do not move
the tag. Do not bump the version to carry one artifact.

1. **Land the fix on `main` via a hotfix branch.** A release is cut from `main`, and `develop`
   usually holds unreleased work that must not ship.
2. **Merge it to `develop` before dispatching.** GitHub only offers `workflow_dispatch` for a
   workflow whose file is on the repo's default branch, which here is `develop`. Land it on
   `main` alone and `gh workflow run` returns 422 with the trigger apparently missing. This is a
   hotfix, so it goes to both branches anyway. The ordering is what matters.
3. Run `gh workflow run release.yml --ref main -f version=X.Y.Z`. The `--ref` decides which
   version of the workflow file runs and which tree is built, so it must carry both the fix and
   the version being published. Do not dispatch against the old tag: that tree predates the fix.
4. Approve the `pypi` environment gate.

`skip-existing` means everything already on PyPI is skipped. The only possible effect is that a
genuinely absent artifact uploads.

This procedure lives in a document rather than only in `release.yml` because v0.18.0 shipped 26
of 27 packages while the instruction that would have prevented it sat in a comment inside the
step it was telling you to delete.
