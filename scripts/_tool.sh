# Sourced by lint.sh and typecheck.sh; not run directly.
#
# run_tool <name> [args...] runs a dev tool (ruff, mypy) from a venv instead of through
# ``uv run``, which re-resolves the workspace on every call. Lookup order: the repository's own
# ``.venv``, then the active ``$VIRTUAL_ENV``; the first that holds the binary wins. The checkout's
# ``.venv`` comes first because that is the environment uv itself uses here (it ignores, with a
# warning, a ``$VIRTUAL_ENV`` that is not it), so a shell left activated in another venv does not
# check this tree with that venv's tools. When neither holds the binary (a fresh clone before
# ``uv sync``) it falls back to ``uv run <name>``.
run_tool() {
    local name="$1" venv
    shift
    for venv in "$REPO_ROOT/.venv" "${VIRTUAL_ENV:-}"; do
        if [ -n "$venv" ] && [ -x "$venv/bin/$name" ]; then
            "$venv/bin/$name" "$@"
            return
        fi
    done
    uv run "$name" "$@"
}
