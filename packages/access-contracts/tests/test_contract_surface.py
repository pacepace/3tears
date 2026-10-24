"""The access contracts: what they export, and the values both enforcement points rely on."""

from __future__ import annotations

import json
import subprocess
import sys

import threetears.access.contracts as contracts
from threetears.access.contracts import ImpersonationCategory


def test_every_exported_name_resolves() -> None:
    for name in contracts.__all__:
        assert getattr(contracts, name) is not None


def test_the_six_categories_and_their_wire_values() -> None:
    """The values are a wire vocabulary: renaming one breaks every peer that stored it."""
    assert {member.name: member.value for member in ImpersonationCategory} == {
        "CREDENTIAL_PASSKEY_MFA_MANAGEMENT": "credential_passkey_mfa_management",
        "ACCOUNT_DELETION": "account_deletion",
        "APIKEY_MINT_ROTATE_FOR_TARGET": "apikey_mint_rotate_for_target",
        "EMAIL_CHANGE": "email_change",
        "RBAC_GRANT_CHANGE": "rbac_grant_change",
        "AUDIT_EDITING": "audit_editing",
    }


def test_a_category_is_its_string() -> None:
    assert ImpersonationCategory("email_change") is ImpersonationCategory.EMAIL_CHANGE
    assert ImpersonationCategory.EMAIL_CHANGE == "email_change"


def test_importing_the_contracts_loads_nothing_outside_the_stdlib() -> None:
    """The leaf's promise is an empty closure; checked in a fresh interpreter, which nothing preloaded."""
    probe = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import threetears.access.contracts\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr
    loaded: list[str] = json.loads(result.stdout)
    assert "threetears.access.contracts" in loaded
    family = {"threetears", "threetears.access"}
    offenders = [
        module
        for module in loaded
        if module.split(".")[0] not in sys.stdlib_module_names
        and not module.startswith(("_sysconfigdata", "threetears.access.contracts"))
        and module not in family
    ]
    assert offenders == []
