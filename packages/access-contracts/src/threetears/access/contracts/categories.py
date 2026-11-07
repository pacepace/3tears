"""The sensitive-action categories authentication and authorization both name.

Two checks guard the same six actions. Authorization (``3tears-agent-acl``'s
``authorize_from_claims``) refuses them outright under impersonation. Authentication
(``3tears-iam``'s ``stepup``) lets them through only on a recent sign-in. Each check is
only as good as its agreement with the other about what the six are, so the set is
declared once, here, where neither side pays for the other's dependencies to import it.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ImpersonationCategory"]


class ImpersonationCategory(StrEnum):
    """The fixed, non-tenant-configurable sensitive-action category set.

    security-model.md's Impersonation paragraph names these six categories verbatim:
    "credential, passkey, and MFA management; account deletion; API-key mint/rotate for the
    target; email change; RBAC grant changes; audit editing."

    Two enforcement points read it. Under impersonation every member is deny-listed: there is
    no partial or tenant-configurable subset. Outside impersonation every member requires
    step-up re-authentication: the same taxonomy, enforced by freshness instead of subtraction.

    The string values are a wire vocabulary as well as a type: a token, an audit record or a
    peer repository may carry them, so a member is added, never renamed.
    """

    CREDENTIAL_PASSKEY_MFA_MANAGEMENT = "credential_passkey_mfa_management"
    ACCOUNT_DELETION = "account_deletion"
    APIKEY_MINT_ROTATE_FOR_TARGET = "apikey_mint_rotate_for_target"
    EMAIL_CHANGE = "email_change"
    RBAC_GRANT_CHANGE = "rbac_grant_change"
    AUDIT_EDITING = "audit_editing"
