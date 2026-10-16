# Sourced by lint.sh and typecheck.sh; not run directly.
#
# run_tool <name> [args...] runs a dev tool (ruff, mypy) from a venv instead of through
# ``uv run``, which re-resolves the workspace on every call. Lookup order: the active
# ``$VIRTUAL_ENV``, then the repository's own ``.venv``; the first that holds the binary wins.
# When neither does (a fresh clone before ``uv sync``) it falls back to ``uv run <name>``.
run_tool() {
    local name="$1" venv
    shift
    for venv in "${VIRTUAL_ENV:-}" "$REPO_ROOT/.venv"; do
        if [ -n "$venv" ] && [ -x "$venv/bin/$name" ]; then
            "$venv/bin/$name" "$@"
            return
        fi
    done
    uv run "$name" "$@"
}
