"""
shared pytest setup for the 3tears-conversations test suite.

the fake NATS KV these tests use for L2 parity is published as
:mod:`threetears.core.testing.kv` and imported normally; nothing here puts
another suite's test directory on ``sys.path``.
"""
