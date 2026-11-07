# 3tears-access-contracts

Dependency-free identity and authorization vocabulary shared by
`3tears-iam` (who a caller is) and `3tears-agent-acl` (what they may do).

Neither of those packages depends on the other. A service that only signs
people in should not install an RBAC evaluator and its database stack to name
a sensitive action, and an evaluator should not install the login protocols to
name the same one. So the values both sides must agree on are declared once,
here, in a package that depends on nothing.

```python
from threetears.access.contracts import ImpersonationCategory
```

`ImpersonationCategory` is the fixed set of six sensitive actions: credential,
passkey and MFA management; account deletion; API-key mint or rotate for the
target; email change; RBAC grant changes; audit editing. agent-acl deny-lists
all six under impersonation, and iam's `stepup` requires a recent sign-in for
each. `threetears.agent.acl.ImpersonationCategory` is this same class, so
existing imports keep working and compare equal by identity.

The package holds only vocabulary both sides name. A value only one side uses
stays with that side.

## Install

```bash
pip install 3tears-access-contracts
```

## Versioning policy

`3tears-access-contracts` versions in lockstep with the rest of the 3tears
monorepo: every package shares one version, tracking the framework git tag.
