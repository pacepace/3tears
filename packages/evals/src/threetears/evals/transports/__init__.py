"""Transports: thin adapters that mount the action catalogue on a server.

Each transport is its own public root, behind its own optional extra, so importing one never pulls in
another's server stack: :mod:`threetears.evals.transports.fastmcp` (extra ``fastmcp``). The 3tears
``mcp`` transport follows once that server can host a package's action group (pacepace/3tears#531).
Nothing is exported here.
"""
