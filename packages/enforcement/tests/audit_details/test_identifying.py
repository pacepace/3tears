"""no producer passes a details key that names a person (``identifying_detail_findings``), on synthesised modules.

The rule's data is synthetic here (two identifying keys, one family allowed ``name``), so the gate is tested without
the audit package a shell hands it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from threetears.enforcement.audit_details import IdentifyingDetailsConfig, identifying_detail_findings

_IDENTIFYING = frozenset({"name", "email"})


def _refused(event_type: str) -> frozenset[str]:
    return _IDENTIFYING - ({"name"} if event_type.startswith("thing.") else frozenset())


_PRODUCERS = """
    from threetears.agent.audit import AuditEvent

    def emit_widget(*, details):
        return AuditEvent(event_type="widget.change", details=details)

    def ok():
        return AuditEvent(event_type="widget.change", details={"widget_id": "w", "count": 2, "reason": "free text"})

    def top():
        return AuditEvent(event_type="widget.change", details={"widget_id": "w", "email": "a@b.c"})

    def nested():
        return AuditEvent(event_type="widget.change", details={"target": {"member": {"email": "a@b.c"}}})

    def allowed_family():
        return AuditEvent(event_type="thing.create", details={"name": "my thing"})

    def other_family():
        return AuditEvent(event_type="widget.create", details={"name": "Alice"})

    def through_an_emitter():
        emit_widget(details={"widget_id": "w", "email": "a@b.c"})
"""


def _config(tmp_path: Path) -> IdentifyingDetailsConfig:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "producers.py").write_text(textwrap.dedent(_PRODUCERS), encoding="utf-8")
    return IdentifyingDetailsConfig(repo_root=tmp_path, src_roots=(tmp_path / "src",), identifying_keys_for=_refused)


def test_every_identifying_key_a_producer_writes_is_found_at_any_depth(tmp_path: Path) -> None:
    """Flips if a key naming a person passes at the top, nested, through an ``emit_*`` wrapper, or under a family
    not allowed it; or if ids, free text, or a family's allowed resource name are flagged."""
    findings = identifying_detail_findings(_config(tmp_path))
    lines = {int(finding.split(":")[1]) for finding in findings}
    source = textwrap.dedent(_PRODUCERS).splitlines()
    flagged = {source[line - 1].strip().split("(")[0] for line in lines}
    assert any("details key email names a person" in finding for finding in findings), findings
    assert any("details key target.member.email names a person" in finding for finding in findings), findings
    assert any("details key name names a person (widget.create)" in finding for finding in findings), findings
    assert len(findings) == 4, findings
    assert "allowed_family" not in "".join(findings) and all("thing.create" not in f for f in findings), findings
    assert flagged, findings


def test_no_source_roots_is_itself_a_finding(tmp_path: Path) -> None:
    """Flips if a shell scanning nothing reports what a clean repo reports."""
    config = IdentifyingDetailsConfig(repo_root=tmp_path, src_roots=(), identifying_keys_for=_refused)
    assert identifying_detail_findings(config)
