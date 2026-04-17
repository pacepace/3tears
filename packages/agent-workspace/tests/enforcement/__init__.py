"""enforcement tests for the agent-workspace package.

AST-based tests that lock in architectural invariants: tool count and
naming, atomic-write usage, sandbox-enforce ordering, and validator
dispatch. every test is import-cheap and parse-only so the whole suite
runs well under the EAD 15-second budget.
"""
