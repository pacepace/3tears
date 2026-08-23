"""Every epoch subject must have a DECIDED storage substrate.

The two substrates are not interchangeable and the wrong choice is not
symmetric. An ephemeral counter that should have been durable re-issues the
same version number for different content, and if that number has escaped to a
CDN there is nothing inside this system that can repair it. A durable row where
ephemeral would have done costs a Postgres round trip.

So the dangerous default is ephemeral, and ephemeral is exactly what a new
subject gets for free: :func:`~threetears.epoch.client._is_durable` answers
``False`` for anything it does not recognise. That is correct as a runtime
fallback and useless as a policy -- nothing makes the author of a new
``*_epoch`` builder notice they had a decision to make.

This test is what makes them notice. It enumerates the real ``Subjects``
factory and requires every ``*_epoch`` builder to appear in one of the two
declared tables in ``client.py``. Adding a sixth epoch subject fails here until
someone writes down which substrate it takes and why.

It deliberately drives the REAL factory rather than literal paths: the
classifier reads a subject's shape, that shape is produced in another package,
and a hand-written literal would keep matching after the builder changed.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from threetears.epoch.client import (  # noqa: SLF001 - this module IS the policy's test
    _DURABLE_FAMILIES,
    _EPHEMERAL_FAMILIES,
    _is_durable,
)
from threetears.nats.subjects import Subject, Subjects, set_default_namespace


@pytest.fixture(autouse=True)
def _namespace() -> None:
    """Bind a namespace so the builders render.

    No teardown, deliberately. The root ``conftest.py``'s autouse
    ``_bind_test_subject_namespace`` calls ``_reset_default_namespace()`` both
    before and after every test in the workspace, precisely so a test that sets
    the process-wide global cannot shadow its neighbours. Restoring here would
    duplicate that, and an earlier attempt to do so was worse than nothing: it
    read the env-derived value and wrote it back as an explicitly-set global,
    which is the state the root fixture keeps clear.

    :return: nothing
    :rtype: None
    """
    set_default_namespace("polprobe")


def _epoch_builders() -> dict[str, Any]:
    """Every ``*_epoch`` classmethod the subject factory publishes.

    :return: builder name -> callable
    :rtype: dict[str, Any]
    """
    return {
        name: getattr(Subjects, name)
        for name in dir(Subjects)
        if name.endswith("_epoch") and callable(getattr(Subjects, name, None)) and not name.startswith("_")
    }


def _build(builder: Any) -> Subject:
    """Call ``builder``, supplying a placeholder for every required argument.

    :param builder: an epoch subject classmethod
    :ptype builder: Any
    :return: a concrete subject of that family
    :rtype: Subject
    """
    params = [
        p
        for p in inspect.signature(builder).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    built: Subject = builder(*["polprobe" for _ in params])
    return built


class TestEveryEpochSubjectHasADecidedSubstrate:
    """The enumeration guard. A new epoch subject cannot default in silence."""

    def test_the_factory_actually_yields_builders(self) -> None:
        """Guard the guard: a rename that empties the sweep must not read as a pass.

        Every assertion below is vacuous if `_epoch_builders()` returns nothing,
        and a rename in another package is exactly how that would happen.

        :return: nothing
        :rtype: None
        """
        builders = _epoch_builders()

        assert len(builders) >= 5, f"expected the known epoch builders, found {sorted(builders)}"
        assert "datasource_tile_epoch" in builders

    def test_every_builder_is_declared_in_exactly_one_table(self) -> None:
        """No epoch subject may be absent from both tables, or present in both.

        :return: nothing
        :rtype: None
        """
        durable = {name for name, _marker, _why in _DURABLE_FAMILIES}
        ephemeral = {name for name, _why in _EPHEMERAL_FAMILIES}

        undeclared = sorted(set(_epoch_builders()) - durable - ephemeral)
        assert not undeclared, (
            f"epoch subject(s) with no declared substrate: {undeclared}. Add each to "
            "_DURABLE_FAMILIES or _EPHEMERAL_FAMILIES in threetears.epoch.client, with the "
            "reason. Ephemeral is the safe default only when someone has decided it is."
        )
        assert not (durable & ephemeral), f"declared as both: {sorted(durable & ephemeral)}"

    def test_neither_table_names_a_builder_that_no_longer_exists(self) -> None:
        """A deleted subject must not leave a decision behind that reads as live.

        :return: nothing
        :rtype: None
        """
        known = set(_epoch_builders())
        declared = {name for name, _m, _w in _DURABLE_FAMILIES} | {name for name, _w in _EPHEMERAL_FAMILIES}

        stale = sorted(declared - known)
        assert not stale, f"declared substrate for subject(s) that no longer exist: {stale}"


class TestTheDeclarationMatchesTheClassifier:
    """A table nothing consults is documentation, not policy."""

    def test_every_durable_declaration_classifies_durable(self) -> None:
        """Built through the real factory, so a changed subject shape fails here.

        :return: nothing
        :rtype: None
        """
        builders = _epoch_builders()
        for name, _marker, _why in _DURABLE_FAMILIES:
            assert _is_durable(_build(builders[name])), f"{name} is declared durable but classifies ephemeral"

    def test_every_ephemeral_declaration_classifies_ephemeral(self) -> None:
        """The other direction, which is the one a stray marker would break.

        :return: nothing
        :rtype: None
        """
        builders = _epoch_builders()
        for name, _why in _EPHEMERAL_FAMILIES:
            assert not _is_durable(_build(builders[name])), f"{name} is declared ephemeral but classifies durable"

    def test_a_non_epoch_subject_is_never_durable(self) -> None:
        """The classifier keys on the epoch suffix, not on the marker alone.

        A tile DATA subject shares the `.tiles.` segment and must not be routed
        to a Postgres counter it has no row in.

        The one literal in this module, and deliberately: no builder produces a
        non-epoch tile subject, so there is nothing in the factory to drive.

        :return: nothing
        :rtype: None
        """
        assert not _is_durable(Subject(path="polprobe.datasource.ds1.tiles.parcels.render", kind="point"))

    def test_every_declaration_carries_a_reason(self) -> None:
        """A table entry with no reason is a rubber stamp. BOTH tables.

        The durable table used to keep its reason in an inline comment, which
        this test cannot read -- so the guarantee covered one table of two while
        the prose claimed both.

        :return: nothing
        :rtype: None
        """
        both = [(name, why) for name, _marker, why in _DURABLE_FAMILIES]
        both += list(_EPHEMERAL_FAMILIES)

        assert len(both) == len(_DURABLE_FAMILIES) + len(_EPHEMERAL_FAMILIES)
        for name, why in both:
            assert len(why) >= 20, f"{name}'s rationale is too thin: {why!r}"
