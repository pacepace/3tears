"""A described provider failure is logged through one helper, which alone decides the traceback.

When a describer withholds something, a traceback's last line is ``str(exc)`` — the provider
envelope, account id included — so the log line must carry the description and no traceback.
That decision had been hand-written at each caller of ``describe_failure``; it now lives in
``log_provider_failure`` and nowhere else, and the canary below keeps it there.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path


from threetears.evals.contracts.completion import ProviderFailure
from threetears.evals.contracts.provider import describe_and_log_failure, log_provider_failure


_REPO = Path(__file__).resolve().parents[1] / "src"
_ENVELOPE = '{"error": {"message": "Insufficient credits", "metadata": {"account_id": "acct-SECRET"}}}'
#: The one module that may read the flag: it is where the logging decision is made.
_READER = _REPO / "threetears" / "evals" / "contracts" / "provider.py"


class _ProviderDown(Exception):
    pass


def _raised() -> _ProviderDown:
    try:
        raise _ProviderDown(_ENVELOPE)
    except _ProviderDown as exc:
        return exc


class TestTheHelperDecidesTheTraceback:
    def test_a_withheld_description_is_logged_with_no_traceback(self, caplog):
        exc = _raised()
        failure = ProviderFailure("HTTP 402: Insufficient credits", payload_withheld=True, account_refused=True)
        with caplog.at_level(logging.ERROR):
            log_provider_failure(logging.getLogger("t"), failure, exc, "judge call failed for case %s", "c1")

        (record,) = caplog.records
        assert record.getMessage() == "judge call failed for case c1: HTTP 402: Insufficient credits"
        assert record.exc_info is None
        assert "acct-SECRET" not in caplog.text

    def test_an_honest_description_keeps_its_traceback(self, caplog):
        exc = _raised()
        failure = ProviderFailure("TimeoutError: read timed out", payload_withheld=False, account_refused=False)
        with caplog.at_level(logging.WARNING):
            log_provider_failure(logging.getLogger("t"), failure, exc, "call failed", level=logging.WARNING)

        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert record.exc_info is not None and record.exc_info[1] is exc

    def test_describe_and_log_describes_through_the_describer_then_logs_its_answer(self, caplog):
        exc = _raised()
        seen: list[BaseException] = []

        def describer(e: BaseException) -> ProviderFailure:
            seen.append(e)
            return ProviderFailure("HTTP 402: Insufficient credits", payload_withheld=True, account_refused=True)

        with caplog.at_level(logging.ERROR):
            failure = describe_and_log_failure(
                describer,
                exc,
                logger=logging.getLogger("t"),
                where="judge",
                message="judge call failed for case %s",
                args=("c1",),
            )

        assert seen == [exc] and failure.account_refused
        (record,) = caplog.records
        assert record.exc_info is None and "acct-SECRET" not in caplog.text


class TestNoCallerDecidesTheTracebackItself:
    def test_only_the_helper_module_reads_payload_withheld(self):
        """A caller branching on the flag is a copy of the helper's rule, free to drift from it."""
        readers: list[str] = []
        assert _READER.is_file(), "the helper module moved; the walk below excuses a file that is not there"
        for path in sorted((_REPO / "threetears").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            # Only a file naming the flag can read it; parsing the rest would cost seconds for nothing.
            if path == _READER or "payload_withheld" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            readers.extend(
                f"{path.relative_to(_REPO)}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr == "payload_withheld"
                and isinstance(node.ctx, ast.Load)
            )
        assert readers == [], f"read ProviderFailure.payload_withheld through log_provider_failure instead: {readers}"

    def test_the_canary_sees_a_read_when_there_is_one(self):
        """Positive control: the walk above finds the shape it is looking for."""
        tree = ast.parse("if failure.payload_withheld:\n    pass\n")
        assert any(isinstance(n, ast.Attribute) and n.attr == "payload_withheld" for n in ast.walk(tree))
