"""agent-acl's ``ImpersonationCategory`` is the access-contracts leaf's class, not a copy.

The deny-list here and iam's step-up check name the same six actions. A second declaration
would compare equal by value and still be a second type, so an ``isinstance`` check or a
mapping keyed by one would quietly miss the other's members.
"""

from __future__ import annotations

import importlib

import threetears.access.contracts as contracts
import threetears.agent.acl as acl

#: the module, not the ``authorize`` function the package re-exports under the same name
authorize_module = importlib.import_module("threetears.agent.acl.authorize")


def test_the_package_export_is_the_leaf_s_class() -> None:
    assert acl.ImpersonationCategory is contracts.ImpersonationCategory


def test_the_authorize_module_s_name_is_the_leaf_s_class() -> None:
    """``from threetears.agent.acl.authorize import ImpersonationCategory`` keeps working too."""
    assert authorize_module.ImpersonationCategory is contracts.ImpersonationCategory


def test_the_package_still_lists_it_as_public() -> None:
    assert "ImpersonationCategory" in acl.__all__
    assert "ImpersonationCategory" in authorize_module.__all__
