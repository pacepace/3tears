"""The sender-posture gate, driven by synthesised modules and allowlists.

Every case is a way a subscriber could reach production without saying how it authenticates
whoever sent its messages, or a way the shrinking allowlist could stop shrinking honestly:
an entry left behind after its site was fixed or deleted, a site that claims an open row the
list does not carry, or a marker that no longer sits on any site.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.sender_posture import (
    SenderPostureConfig,
    find_subscribe_sites,
    sender_posture_findings,
)

_ALLOWLIST_HEADER = "# path :: row :: chunk :: why it is still open\n"


def _config(
    tmp_path: Path,
    sources: dict[str, str],
    allowlist: str = "",
    *,
    primitive_modules: frozenset[str] = frozenset(),
) -> SenderPostureConfig:
    """
    write synthesised modules and allowlist under tmp_path and answer gate config over them.

    :param tmp_path: pytest temporary directory, used as repo root
    :ptype tmp_path: Path
    :param sources: repo-relative module path onto module source
    :ptype sources: dict[str, str]
    :param allowlist: allowlist body written after header line
    :ptype allowlist: str
    :param primitive_modules: repo-relative modules allowed to declare primitive posture
    :ptype primitive_modules: frozenset[str]
    :return: gate configuration over synthesised tree
    :rtype: SenderPostureConfig
    """
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    allowlist_path = tmp_path / "tests" / "_sender_posture_allowlist.txt"
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text(_ALLOWLIST_HEADER + allowlist, encoding="utf-8")
    return SenderPostureConfig(
        repo_root=tmp_path,
        src_roots=(tmp_path / "src",),
        allowlist_path=allowlist_path,
        primitive_modules=primitive_modules,
    )


def test_an_undeclared_subscribe_fails(tmp_path: Path) -> None:
    """a subscribe call with no marker is a finding naming its file and line."""
    config = _config(
        tmp_path,
        {"src/pkg/door.py": "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)\n"},
    )
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "src/pkg/door.py:2" in findings[0]
    assert "declares no sender posture" in findings[0]


def test_a_declared_subscribe_passes(tmp_path: Path) -> None:
    """a marker on the call line declares the site."""
    config = _config(
        tmp_path,
        {
            "src/pkg/door.py": (
                "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: verified-token\n"
            )
        },
    )
    assert sender_posture_findings(config) == []


def test_a_marker_inside_a_multiline_call_declares_it(tmp_path: Path) -> None:
    """a marker on any line of the call's own span counts."""
    source = (
        "async def start(nc, cb):\n"
        "    self_sub = await nc.jetstream_pull_subscribe(\n"
        "        subject='x',\n"
        "        # sender-posture: own-consumer\n"
        "        cb=cb,\n"
        "    )\n"
    )
    config = _config(tmp_path, {"src/pkg/door.py": source})
    assert sender_posture_findings(config) == []


def test_a_marker_in_the_comment_block_directly_above_declares_it(tmp_path: Path) -> None:
    """contiguous comment lines straight above the statement belong to it."""
    source = (
        "async def start(nc, cb):\n"
        "    # sender-posture: trigger-only -- refetches the catalogue, reads nothing from the body\n"
        "    # the reload is cheap and idempotent\n"
        "    await nc.subscribe_typed(subject='x', cb=cb)\n"
    )
    config = _config(tmp_path, {"src/pkg/door.py": source})
    assert sender_posture_findings(config) == []


def test_a_marker_separated_by_a_blank_line_does_not_declare_it(tmp_path: Path) -> None:
    """a blank line ends the comment block, so the marker is an orphan and the site undeclared."""
    source = "async def start(nc, cb):\n    # sender-posture: read-only\n\n    await nc.subscribe(subject='x', cb=cb)\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert any("declares no sender posture" in finding for finding in findings)
    assert any("sits on no subscribe site" in finding for finding in findings)


def test_a_marker_spelled_inside_a_string_is_not_a_declaration(tmp_path: Path) -> None:
    """only a real comment declares; text inside a string literal is ignored."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='# sender-posture: read-only', cb=cb)\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "declares no sender posture" in findings[0]


def test_an_unknown_posture_fails(tmp_path: Path) -> None:
    """a posture outside the declared vocabulary is refused, not read as a declaration."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: trusted\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "unknown sender posture 'trusted'" in findings[0]


def test_a_malformed_marker_fails(tmp_path: Path) -> None:
    """a marker that names the gate but does not parse is a finding, never silently ignored."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture verified-token\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert any("malformed sender-posture marker" in finding for finding in findings)


def test_unauthenticated_needs_an_open_row(tmp_path: Path) -> None:
    """unauthenticated without the audit row it is tracked under is refused."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: unauthenticated\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "unauthenticated must name the open audit row" in findings[0]


def test_an_open_row_listed_in_the_allowlist_passes(tmp_path: Path) -> None:
    """an unauthenticated site whose row the allowlist carries is green."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: unauthenticated open H5\n"
    config = _config(
        tmp_path,
        {"src/pkg/door.py": source},
        "src/pkg/door.py :: H5 :: 03 :: owner and customer come from the body\n",
    )
    assert sender_posture_findings(config) == []


def test_an_open_row_missing_from_the_allowlist_fails(tmp_path: Path) -> None:
    """a site cannot grant itself an exception the allowlist does not record."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: unauthenticated open H5\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "open row H5" in findings[0]
    assert "not in the allowlist" in findings[0]


def test_a_declared_posture_may_also_carry_an_open_row(tmp_path: Path) -> None:
    """a site that verifies a bearer token but is still open under a row lists both."""
    source = (
        "async def start(nc, cb):\n"
        "    await nc.subscribe(subject='x', cb=cb)  # sender-posture: verified-token open H10,H11\n"
    )
    config = _config(
        tmp_path,
        {"src/pkg/door.py": source},
        "src/pkg/door.py :: H10 :: 10 :: bearer replay\nsrc/pkg/door.py :: H11 :: 10 :: bearer replay\n",
    )
    assert sender_posture_findings(config) == []


def test_an_allowlist_entry_whose_site_is_gone_fails(tmp_path: Path) -> None:
    """the list only shrinks honestly: a line whose site was fixed or deleted must go too."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: verified-token\n"
    config = _config(
        tmp_path,
        {"src/pkg/door.py": source},
        "src/pkg/door.py :: H5 :: 03 :: owner and customer come from the body\n"
        "src/pkg/gone.py :: H8 :: 03 :: ids from the body\n",
    )
    findings = sender_posture_findings(config)
    assert len(findings) == 2
    assert all("no site in it declares that row open" in finding for finding in findings)


def test_a_malformed_allowlist_line_fails(tmp_path: Path) -> None:
    """every allowlist line names path, row, chunk and reason, or it is a finding."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: read-only\n"
    config = _config(tmp_path, {"src/pkg/door.py": source}, "src/pkg/door.py :: H5\n")
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "malformed allowlist line" in findings[0]


def test_a_duplicate_allowlist_line_fails(tmp_path: Path) -> None:
    """one (path, row) pair is listed once."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: unauthenticated open H5\n"
    entry = "src/pkg/door.py :: H5 :: 03 :: owner from the body\n"
    config = _config(tmp_path, {"src/pkg/door.py": source}, entry + entry)
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "listed twice" in findings[0]


def test_a_missing_allowlist_file_fails(tmp_path: Path) -> None:
    """an absent allowlist is a finding, never an empty list that passes."""
    config = _config(tmp_path, {"src/pkg/door.py": "x = 1\n"})
    config.allowlist_path.unlink()
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "allowlist not found" in findings[0]


def test_primitive_is_refused_outside_a_primitive_module(tmp_path: Path) -> None:
    """only the modules implementing the subscribe primitive may defer posture to their callers."""
    source = "async def start(nc, cb):\n    await nc.subscribe(subject='x', cb=cb)  # sender-posture: primitive\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "primitive is allowed only in" in findings[0]


def test_primitive_passes_inside_a_primitive_module(tmp_path: Path) -> None:
    """the configured primitive module may declare primitive."""
    source = (
        "async def start(connection, path):\n    return await connection.subscribe(path)  # sender-posture: primitive\n"
    )
    config = _config(
        tmp_path,
        {"src/pkg/client.py": source},
        primitive_modules=frozenset({"src/pkg/client.py"}),
    )
    assert sender_posture_findings(config) == []


def test_an_orphan_marker_fails(tmp_path: Path) -> None:
    """a marker left behind after its subscribe was removed is a finding."""
    source = "def nothing():\n    # sender-posture: read-only\n    return 1\n"
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "sits on no subscribe site" in findings[0]


def test_a_bare_name_callee_is_a_site(tmp_path: Path) -> None:
    """a primitive called by bare name, such as serve_owner, is a site like a method call."""
    source = "async def run(nats, key, handler):\n    async with serve_owner(nats, key, handler):\n        pass\n"
    config = _config(tmp_path, {"src/pkg/owner.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "serve_owner" in findings[0]


def test_every_named_receiver_shape_is_a_site(tmp_path: Path) -> None:
    """subscribe, the jetstream consumers, the result waiter and the kv watches are all sites."""
    callees = (
        "subscribe",
        "subscribe_typed",
        "jetstream_subscribe_durable",
        "jetstream_pull_subscribe",
        "jetstream_result_waiter",
        "pull_subscribe",
        "watch_key",
        "watch_prefix",
        "renew_on_request",
        "abandon_on_refusal",
    )
    body = "".join(f"    await nc.{callee}(x)\n" for callee in callees)
    config = _config(tmp_path, {"src/pkg/door.py": "async def start(nc, x):\n" + body})
    sites = find_subscribe_sites(config)
    assert sorted(site.callee for site in sites) == sorted(callees)


def test_a_definition_is_not_a_site(tmp_path: Path) -> None:
    """defining a method named subscribe is not subscribing."""
    source = "class Fake:\n    async def subscribe(self, subject, cb):\n        return None\n"
    config = _config(tmp_path, {"src/pkg/fake.py": source})
    assert find_subscribe_sites(config) == []
    assert sender_posture_findings(config) == []


def test_an_unparseable_module_is_a_finding(tmp_path: Path) -> None:
    """a module the walker cannot read is reported, never skipped as if it held no sites."""
    config = _config(tmp_path, {"src/pkg/broken.py": "def (:\n"})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "could not be parsed" in findings[0]


def test_two_sites_need_two_markers(tmp_path: Path) -> None:
    """a marker declares the statement it sits on, not its neighbour."""
    source = (
        "async def start(nc, cb):\n"
        "    # sender-posture: read-only\n"
        "    await nc.subscribe(subject='a', cb=cb)\n"
        "    await nc.subscribe(subject='b', cb=cb)\n"
    )
    config = _config(tmp_path, {"src/pkg/door.py": source})
    findings = sender_posture_findings(config)
    assert len(findings) == 1
    assert "src/pkg/door.py:4" in findings[0]


def test_the_site_reports_its_posture_and_open_rows(tmp_path: Path) -> None:
    """a declared site carries its posture and rows, for a repo gate to count and report."""
    source = (
        "async def start(nc, cb):\n"
        "    await nc.subscribe(subject='x', cb=cb)  # sender-posture: verified-token open H10\n"
    )
    config = _config(tmp_path, {"src/pkg/door.py": source}, "src/pkg/door.py :: H10 :: 10 :: bearer replay\n")
    (site,) = find_subscribe_sites(config)
    assert site.path == "src/pkg/door.py"
    assert site.line == 2
    assert site.posture == "verified-token"
    assert site.open_rows == ("H10",)
