"""shared test setup for the 3tears-search package test suite.

shared helpers (``search_instances``, the payload modules) are imported by
their repo-root name -- ``from packages.search.tests.search_instances import``.
a bare ``from tests.x import`` cannot work: several workspace packages own a
``tests`` package, and whichever one a run imported first would shadow this
one; putting this directory on ``sys.path`` instead leaks its module names
into every other suite in the same process.
"""
