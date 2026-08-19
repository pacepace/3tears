"""enforcement: exactly one place in this workspace composes or sends email.

The mail capability in `threetears.channels.mail` exists because a working
implementation was found in `14-eng-ai-bot-identity` and PROMOTED rather than copied --
the estate's rule being that a second consumer is the trigger to move something upstream,
never to grow a second version of it. A guard is what keeps that true after the promotion,
because the cheapest way to need mail in a new package is to `import smtplib` and write
twenty lines, and twenty lines is exactly small enough to look like it is not a decision.

Two families are refused outside the sanctioned module:

**SMTP clients** (`smtplib`, `aiosmtplib`). A second one means a second answer to
STARTTLS, to authenticating over an unencrypted connection, to timeouts, and to whether a
failure is reported or swallowed -- and the first three are security decisions the
promoted transport already makes and documents.

**MIME composition** (`email.message`, `email.mime`). Header construction is where
injection lives: a subject or display name carrying a newline appends headers of the
caller's choosing. One composer means one place that gets that right. `email.utils` is
NOT refused -- date and address parsing are ordinary utilities with no send behind them,
and `threetears.search` uses `parsedate_to_datetime` for an HTTP header.

**Provider SDKs** are refused everywhere, including inside the sanctioned module. The
transport reaches SendGrid, SES, a self-hosted relay and whatever replaces them because
they all speak SMTP; a vendor SDK buys templates, campaigns and analytics this path does
not want, in exchange for a rewrite per provider.

Static parsing only -- no imports executed, no network -- consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")

#: The one module allowed to speak SMTP and to build a MIME message. Repo-relative posix.
_SANCTIONED_TRANSPORT = "packages/channels/src/threetears/channels/mail/smtp.py"

#: Modules whose import means "this file sends or composes mail".
_SMTP_MODULES = frozenset({"smtplib", "aiosmtplib"})
_MIME_MODULES = frozenset({"email.message", "email.mime"})

#: Vendor mail SDKs, refused workspace-wide including in the sanctioned module.
_PROVIDER_SDKS = frozenset({"sendgrid", "postmarker", "mailgun", "resend", "mailjet", "sib_api_v3_sdk"})


def _imported_modules(tree: ast.Module) -> set[str]:
    """Return every module name a file imports, by dotted root path.

    A submodule import is reported under its own dotted name so
    ``from email.message import EmailMessage`` is distinguishable from
    ``from email.utils import parsedate_to_datetime``.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: dotted module names this file imports from
    :rtype: set[str]
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _matches(imported: set[str], refused: frozenset[str]) -> set[str]:
    """Return the refused modules `imported` names, exactly or as a package prefix.

    :param imported: dotted module names a file imports
    :ptype imported: set[str]
    :param refused: dotted module names that are not allowed
    :ptype refused: frozenset[str]
    :return: the refused names actually imported
    :rtype: set[str]
    """
    return {
        candidate for candidate in refused for name in imported if name == candidate or name.startswith(f"{candidate}.")
    }


def _source_files() -> list[Path]:
    """Return every shipped python file across the workspace's source trees.

    :return: python files under each package's ``src``
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for root in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))
    return found


def test_the_source_trees_were_discovered() -> None:
    """The globs must actually match -- a silent zero would pass every test below.

    :return: none
    :rtype: None
    :raises AssertionError: if almost nothing was discovered
    """
    assert len(_source_files()) > 100, (
        f"only {len(_source_files())} source files matched {_PACKAGE_GLOBS}. The layout changed "
        "and this guard is now inspecting almost nothing."
    )


def test_only_the_promoted_transport_speaks_smtp_or_builds_mime() -> None:
    """No second mailer grows anywhere in the workspace.

    :return: none
    :rtype: None
    :raises AssertionError: if a file outside the sanctioned transport imports an SMTP
        client or a MIME composer
    """
    violations: list[str] = []
    for path in _source_files():
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative == _SANCTIONED_TRANSPORT:
            continue
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        offending = _matches(imported, _SMTP_MODULES) | _matches(imported, _MIME_MODULES)
        violations.extend(f"{relative}: imports {name}" for name in sorted(offending))
    assert not violations, (
        "these files send or compose email outside the promoted transport:\n  "
        + "\n  ".join(sorted(violations))
        + f"\n\nUse `threetears.channels.mail` instead. Only {_SANCTIONED_TRANSPORT} may hold an "
        "SMTP conversation or build a MIME message: a second one is a second answer to STARTTLS, "
        "to authenticating in clear, to timeouts, and to header injection. If the promoted "
        "package genuinely cannot express what you need, extend it there."
    )


def test_no_package_reaches_a_mail_provider_sdk() -> None:
    """A relay is reached by SMTP, never by a vendor's client library.

    :return: none
    :rtype: None
    :raises AssertionError: if any file imports a mail provider SDK
    """
    violations: list[str] = []
    for path in _source_files():
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        relative = path.relative_to(_REPO_ROOT).as_posix()
        violations.extend(f"{relative}: imports {name}" for name in sorted(_matches(imported, _PROVIDER_SDKS)))
    assert not violations, (
        "these files reach a mail provider through its own SDK:\n  "
        + "\n  ".join(sorted(violations))
        + "\n\nEvery provider this platform sends through speaks SMTP, so one transport reaches "
        "all of them and changing provider is a settings edit. An SDK makes it a rewrite, and "
        "buys templates, campaigns and analytics that this path does not use."
    )
