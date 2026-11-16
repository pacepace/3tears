"""Keeps pytest out of the agent test's run directories.

``runs/`` (git-ignored) holds every trial's copy of an app as the agent left it, and agents write files named
``test_*.py`` there; a sweep of ``packages/`` would collect them as the repo's tests. The harness's own tests are
``packages/evals/tests/test_onramp_*.py``.
"""

collect_ignore = ["runs"]
