"""The access interfaces: what they export, and the values both enforcement points rely on."""

from __future__ import annotations

import threetears.interfaces.access as access
from threetears.interfaces.access import ImpersonationCategory


def test_every_exported_name_resolves() -> None:
    for name in access.__all__:
        assert getattr(access, name) is not None


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
