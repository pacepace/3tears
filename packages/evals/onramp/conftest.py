"""Keeps pytest out of the agent test's run directories and out of its apps.

``runs/`` (git-ignored) holds every trial's copy of an app as the agent left it, and agents write files named
``test_*.py`` there; a sweep of ``packages/`` would collect them as the repo's tests. The harness's own tests are
``packages/evals/tests/test_onramp_*.py``.

``apps/`` and ``twins/`` hold the apps an agent is given. An app's own ``tests/`` are the app's, written to run from
the app's directory (``python -m pytest tests``), where the app's modules import by their bare names, as they do in a
trial; collected from the repo root they would not import.
"""

collect_ignore = ["runs", "apps", "twins"]
