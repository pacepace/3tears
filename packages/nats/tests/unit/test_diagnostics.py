"""unit tests for :mod:`threetears.nats._diagnostics`.

The condition under test is the one that produces no useful signal on its own: a
KV bucket the connection's user JWT does not grant. The server refuses the
JetStream request and never answers it, so the caller sees a timeout and reads
"unreachable broker" -- while the connection it is using stays up and healthy for
every other subject.

These tests pin the two places the truth is recoverable: the server's own
``permissions violation`` frame (which reaches the error callback and leaves the
connection open), and the deadline the refused operation eventually blows.
"""

from __future__ import annotations

import logging

import pytest
from nats import errors as nats_errors

from threetears.nats._diagnostics import (  # noqa: SLF001 - module is private by design; this is its test
    kv_grant_remedy,
    kv_timeout_remedy,
    permissions_violation_remedy,
)


class TestReadingTheServersRefusal:
    """``permissions_violation_remedy`` over the frames ``nats-py`` actually delivers.

    Every string here is lower-cased, because ``nats-py``'s protocol parser
    lower-cases an ``-ERR`` frame in full before dispatching it. A matcher written
    against the server's own capitalisation would never fire in production and
    would pass a test that used the server's spelling.
    """

    def test_an_unrelated_error_is_left_alone(self) -> None:
        """Ordinary errors keep the ordinary error path.

        :return: nothing
        :rtype: None
        """
        assert permissions_violation_remedy(nats_errors.StaleConnectionError()) is None

    def test_a_refused_kv_data_subject_names_its_bucket_and_the_grant(self) -> None:
        """``$KV.{bucket}.{key}`` yields the bucket and the declaration to change.

        :return: nothing
        :rtype: None
        """
        exc = nats_errors.Error('nats: permissions violation for publish to "$kv.prod-epochs.cfg.agents.epoch"')

        remedy = permissions_violation_remedy(exc)

        assert remedy is not None
        assert "'prod-epochs'" in remedy
        assert "js_resources" in remedy
        assert '"$KV.prod-epochs.>"' in remedy
        assert '"KV_prod-epochs"' in remedy

    def test_a_refused_jetstream_control_subject_names_its_bucket(self) -> None:
        """The control plane is refused FIRST, so it must resolve to a bucket too.

        Opening a bucket is a ``STREAM.CREATE`` against ``KV_{bucket}``; a fully
        ungranted bucket never reaches a ``$KV`` data subject at all, so a matcher
        that only understood the data plane would stay silent for the whole of the
        symptom an operator actually meets first.

        :return: nothing
        :rtype: None
        """
        exc = nats_errors.Error('nats: permissions violation for publish to "$js.api.stream.create.kv_prod-epochs"')

        remedy = permissions_violation_remedy(exc)

        assert remedy is not None
        assert "'prod-epochs'" in remedy

    def test_a_stream_merely_containing_kv_is_not_read_as_a_bucket(self) -> None:
        """``kv_`` must open a subject token, not appear anywhere in one.

        A stream named ``events-kv_v2`` is not a KV bucket. Naming one would send
        the reader to grant a bucket nothing will ever open, which is worse than
        saying nothing: the generic remedy at least describes the real situation.

        :return: nothing
        :rtype: None
        """
        exc = nats_errors.Error('nats: permissions violation for publish to "$js.api.stream.create.events-kv_v2"')

        remedy = permissions_violation_remedy(exc)

        assert remedy is not None
        assert "v2" not in remedy.replace(str(exc), "")
        assert "KV bucket" not in remedy

    def test_a_refusal_naming_no_kv_subject_still_explains_itself(self) -> None:
        """An unnameable bucket is not a reason to fall back to the bland line.

        The load-bearing fact -- the connection stays up, so this warning is the
        only one -- holds for every refused subject, not just KV ones.

        :return: nothing
        :rtype: None
        """
        exc = nats_errors.Error('nats: permissions violation for subscription to "prod.agent.>"')

        remedy = permissions_violation_remedy(exc)

        assert remedy is not None
        assert "PERMISSIONS VIOLATION" in remedy
        assert "prod.agent.>" in remedy

    def test_the_servers_own_words_survive_into_the_message(self) -> None:
        """The remedy adds to the raw text; it never replaces it.

        Whatever this module fails to parse, the operator can still read.

        :return: nothing
        :rtype: None
        """
        raw = 'nats: permissions violation for publish to "$kv.prod-epochs.k"'

        remedy = permissions_violation_remedy(nats_errors.Error(raw))

        assert remedy is not None
        assert raw in remedy


class TestTheRemedyText:
    """What the two remediation strings must contain to be actionable."""

    def test_the_grant_remedy_names_the_declaration_not_only_the_wire_subjects(self) -> None:
        """A reader who patches only the subjects leaves the declaration wrong.

        ``mint_user_jwt`` derives both wire grants from one ``JsResource`` entry, so
        the entry is the fix and the subjects are the explanation.

        :return: nothing
        :rtype: None
        """
        remedy = kv_grant_remedy("prod-epochs")

        assert "js_resources" in remedy
        assert "threetears.nats.subject_permissions" in remedy

    def test_the_remedy_does_not_instruct_the_reader_to_reopen_the_hole(self) -> None:
        """The old text told operators to add ``$KV`` on pub AND sub. That is now a hole.

        Nothing in nats-py subscribes a ``$KV.`` subject -- a read is a ``$JS.API`` request and a
        watch delivers to an inbox -- so the subscribe half conferred no capability and handed the
        holder every write's full value. A remediation string that is actionable and WRONG is worse
        than none: it is followed, and it is followed by whoever is already lost.

        :return: nothing
        :rtype: None
        """
        remedy = kv_grant_remedy("prod-epochs")

        assert "pub+sub" not in remedy
        assert "PUBLISH-ONLY" in remedy
        assert "<scope>" in remedy  # the scoped shape is named, not only the whole subtree

    def test_an_uncertain_remedy_does_not_assert_the_cause(self) -> None:
        """Where a grant is only the leading candidate, the text must say so.

        A failed bucket bind has no refusal behind it: a bucket nobody created
        fails identically. An error that asserts the wrong cause sends the reader
        further away than one that admits it is ranking two.

        :return: nothing
        :rtype: None
        """
        certain = kv_grant_remedy("prod-epochs")
        hedged = kv_grant_remedy("prod-epochs", certain=False)

        assert certain.startswith("FIX:")
        assert hedged.startswith("MOST LIKELY FIX")
        assert "never created" in hedged
        assert "js_resources" in hedged

    def test_the_timeout_remedy_puts_the_grant_ahead_of_the_network(self) -> None:
        """Both causes are named, and the silent one comes first.

        A reader checks the network unprompted; nothing prompts them to check a
        grant, which is why ordering is part of the contract rather than prose.

        :return: nothing
        :rtype: None
        """
        remedy = kv_timeout_remedy("prod-epochs")

        assert remedy.index("user JWT does not grant") < remedy.index("broker is unreachable")
        assert "'prod-epochs'" in remedy


class TestTheClientErrorCallback:
    """The remedy reaches the log, through the callback ``nats-py`` actually calls."""

    @pytest.mark.asyncio
    async def test_a_permissions_violation_is_logged_as_a_remedy(self, caplog: pytest.LogCaptureFixture) -> None:
        """The refusal that leaves the connection up gets the loud line.

        :param caplog: pytest log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        from threetears.nats import client as client_module

        client_module._last_error_log.clear()  # noqa: SLF001 - module-level rate-limit state
        exc = nats_errors.Error('nats: permissions violation for publish to "$kv.prod-epochs.k"')

        with caplog.at_level(logging.ERROR):
            await client_module._on_error(exc)  # noqa: SLF001 - the callback under test

        assert any("PERMISSIONS VIOLATION" in record.getMessage() for record in caplog.records)
        assert any("js_resources" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_an_ordinary_error_keeps_the_ordinary_line(self, caplog: pytest.LogCaptureFixture) -> None:
        """Nothing else acquires a remediation it does not have.

        :param caplog: pytest log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        from threetears.nats import client as client_module

        client_module._last_error_log.clear()  # noqa: SLF001 - module-level rate-limit state

        with caplog.at_level(logging.ERROR):
            await client_module._on_error(nats_errors.StaleConnectionError())  # noqa: SLF001

        messages = [record.getMessage() for record in caplog.records]
        assert any("NATS error:" in message for message in messages)
        assert not any("PERMISSIONS VIOLATION" in message for message in messages)
