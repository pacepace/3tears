"""enforcement: every NATS receiver in 3tears says how it authenticates whoever sent its messages.

a subject proves nothing about its sender. any connection can have its own bytes delivered on
any subject, through a push consumer's ``deliver_subject`` or a request's reply subject, so a
receiver that acts on what arrives has to say what it checked. the walker and the posture
vocabulary live in :mod:`threetears.enforcement.sender_posture`; 3tears' configuration of it is
``sender_posture_support.py`` beside this file, and ``_sender_posture_allowlist.txt`` is the
shrinking list of receivers still open under an audit row.
"""

from __future__ import annotations

from tests.enforcement.sender_posture_support import SENDER_POSTURE_CONFIG as _CONFIG
from threetears.enforcement.sender_posture import find_subscribe_sites, sender_posture_findings

_SOURCE_ROOTS = _CONFIG.src_roots


class TestSubscribersDeclareSenderPosture:
    """No receiver ships without saying how it knows who sent its messages."""

    def test_the_walker_sees_the_receivers(self) -> None:
        """
        a walk that found no receivers checked nothing, so the gate refuses to read it as green.

        :return: nothing
        :rtype: None
        """
        assert len(_SOURCE_ROOTS) > 20, f"the workspace globs found only {_SOURCE_ROOTS}"
        sites = find_subscribe_sites(_CONFIG)
        assert len(sites) >= 25, f"found only {len(sites)} receiving sites; the walker is not seeing the source"
        paths = {site.path for site in sites}
        assert "packages/agent/tools/src/threetears/agent/tools/server.py" in paths
        assert "packages/registry/src/threetears/registry/proxy.py" in paths

    def test_every_receiver_declares_its_posture_and_the_allowlist_matches(self) -> None:
        """
        every site declares a valid posture, and the allowlist carries exactly the open rows.

        :return: nothing
        :rtype: None
        """
        findings = sender_posture_findings(_CONFIG)
        assert not findings, "\n".join(findings)
