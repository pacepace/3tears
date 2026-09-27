"""Agent-tools test configuration.

Shared helpers (``testing_utils``, ``unit/tools/_pod_auth``) are imported by their repo-root
name -- ``from packages.agent.tools.tests.testing_utils import ...`` -- the same name pytest
gives every test module. This directory is deliberately NOT put on ``sys.path``: its
``unit`` / ``integration`` / ``enforcement`` directories would then be importable as
top-level packages, and the workspace suite has packages of the same names, so whichever
suite a combined run imported first shadowed the other's.
"""
