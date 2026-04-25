# Docker Bake orchestration for the 3tears + aibots platform.
#
# This file is canonically version-controlled inside the 3tears repo because
# 3tears owns the framework + its base image. The build contexts address
# every sibling repo via relative path (`../14-eng-ai-bot`, etc.) so all
# four repos must live as siblings of 3tears under the same parent
# directory (the layout used in development and CI).
#
# Invocation MUST be from the parent directory containing every repo so the
# `../<repo>` relative paths resolve. A convenience symlink at the parent
# (`<parent>/docker-bake.hcl -> 3tears/docker-bake.hcl`) lets `docker buildx
# bake` resolve the file automatically when invoked from the parent. See
# scripts/dev-build.sh in the hub repo for the canonical wrapper.
#
# Targets and groups:
#   docker buildx bake threetears-base   # just the framework base
#   docker buildx bake aibots-base       # framework + SDK base
#   docker buildx bake base              # both bases
#   docker buildx bake hub               # hub consumer image
#   docker buildx bake admin             # admin agent consumer image
#   docker buildx bake schema            # schema agent consumer image
#   docker buildx bake all               # everything
#
# The cross-target `contexts` wiring (target:threetears-base, target:aibots-base)
# means local-dev `bake all` builds the bases first and consumers consume the
# in-flight target output without round-tripping through the registry.
# CI uses `--push` to publish bases first, then consumers via the registry tag.

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "VERSION" {
  default = "v0.5.0"
}

variable "REGISTRY" {
  default = "ghcr.io/pacepace"
}

# Multi-arch: linux/amd64 (CI runners, x86 prod) + linux/arm64 (Apple Silicon
# dev, ARM cloud nodes). PLATFORMS is a comma-separated string so it can be
# overridden as a single CLI flag (--set "*.platforms=linux/amd64").
variable "PLATFORMS" {
  default = "linux/amd64,linux/arm64"
}

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

group "default" {
  targets = ["all"]
}

group "base" {
  targets = ["threetears-base", "aibots-base"]
}

group "consumers" {
  targets = ["hub", "admin", "schema"]
}

group "all" {
  targets = ["threetears-base", "aibots-base", "hub", "admin", "schema"]
}

# ---------------------------------------------------------------------------
# Common settings (HCL inheritance via target "common")
# ---------------------------------------------------------------------------

target "common" {
  platforms = split(",", PLATFORMS)
}

# ---------------------------------------------------------------------------
# Base images
# ---------------------------------------------------------------------------

target "threetears-base" {
  inherits   = ["common"]
  context    = "../3tears"
  dockerfile = "docker/Dockerfile"
  tags = [
    "${REGISTRY}/threetears-base:${VERSION}",
    "${REGISTRY}/threetears-base:latest",
  ]
}

target "aibots-base" {
  inherits   = ["common"]
  context    = "../14-eng-ai-bot-agents"
  dockerfile = "docker/Dockerfile"
  contexts = {
    # Wires the in-flight threetears-base target as a build context, so
    # `bake all` builds the framework base then immediately consumes it
    # without a registry round-trip during local development.
    "ghcr.io/pacepace/threetears-base:v0.5.0" = "target:threetears-base"
  }
  args = {
    THREETEARS_BASE = "ghcr.io/pacepace/threetears-base:v0.5.0"
  }
  tags = [
    "${REGISTRY}/aibots-base:${VERSION}",
    "${REGISTRY}/aibots-base:latest",
  ]
}

# ---------------------------------------------------------------------------
# Consumer images
# ---------------------------------------------------------------------------

target "hub" {
  inherits   = ["common"]
  context    = "../14-eng-ai-bot"
  dockerfile = "Dockerfile"
  contexts = {
    "ghcr.io/pacepace/aibots-base:v0.5.0" = "target:aibots-base"
  }
  args = {
    AIBOTS_BASE = "ghcr.io/pacepace/aibots-base:v0.5.0"
  }
  tags = [
    "${REGISTRY}/aibots-hub:${VERSION}",
    "${REGISTRY}/aibots-hub:latest",
  ]
}

target "admin" {
  inherits   = ["common"]
  context    = "../14-eng-ai-bot-agent-admin"
  dockerfile = "Dockerfile"
  contexts = {
    "ghcr.io/pacepace/aibots-base:v0.5.0" = "target:aibots-base"
  }
  args = {
    AIBOTS_BASE = "ghcr.io/pacepace/aibots-base:v0.5.0"
  }
  tags = [
    "${REGISTRY}/aibots-admin:${VERSION}",
    "${REGISTRY}/aibots-admin:latest",
  ]
}

target "schema" {
  inherits   = ["common"]
  context    = "../14-eng-ai-bot-agents"
  dockerfile = "docker/schema-agent/Dockerfile"
  contexts = {
    "ghcr.io/pacepace/aibots-base:v0.5.0" = "target:aibots-base"
  }
  args = {
    AIBOTS_BASE = "ghcr.io/pacepace/aibots-base:v0.5.0"
  }
  tags = [
    "${REGISTRY}/aibots-schema:${VERSION}",
    "${REGISTRY}/aibots-schema:latest",
  ]
}
