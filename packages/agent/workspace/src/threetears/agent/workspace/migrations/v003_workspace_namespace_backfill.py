"""
agent-workspace v003: RETIRED. the workspace namespace backfill no longer runs.

**What it did.** workspace-task-19 (WS-ACL-11) made every workspace a
namespace via shared primary key -- ``workspaces.workspace_id ==
namespaces.namespace_id``, ``namespace_type = 'workspace'``. v003 healed
pre-task-19 history by reading ``<agent_schema>.workspaces`` and inserting a
matching row into the hub's ``namespaces`` table, joining the hub's
``agents`` table for the customer, from a connection whose ``search_path``
was bound to the AGENT schema. To reach a second schema from there, SQL has
to name it -- and the statement named ``platform``.

**Why the body is gone rather than corrected.** Three separate reasons, each
sufficient on its own:

1. **The schema name it hardcoded is not a fact about any deployment.** The
   hub writes to whatever ``FOURTEENAIBOTS_DB_SCHEMA`` names -- ``aibots`` on
   the compose stack. The hardcoded ``platform`` default was removed from the
   hub precisely because it is correct on one deployment and wrong on every
   other, and this statement was the last executable copy of it left in this
   repo. Against a deployment naming its schema anything else, the statement
   does not degrade: it raises a ``relation ... does not exist`` error naming
   the removed default, and fails agent provisioning.

2. **The correct name cannot be threaded in.** A migration body's only
   argument is a :class:`~threetears.core.data.store.DataStore`, and a
   DataStore knows one schema -- its own ``agent_<hex>``. The hub's schema
   name lives in a hub setting this library must not read, the migration
   runner deliberately hardcodes no schema names ("that stays the caller's
   responsibility"), and the hub's own agent-migration connection sets
   ``search_path`` to the agent schema plus ``public``, so an unqualified
   ``namespaces`` cannot resolve either. Threading it would mean changing the
   signature of every migration callable in the family.

3. **The write moved, and an agent must not do it.** The paired namespace row
   for a workspace is now emitted by ``workspace_create`` as an event and
   WRITTEN BY THE HUB, which owns direct database access; the agent-side L3
   proxy cannot reach the hub's tables at all. A migration that heals toward
   an invariant somebody else maintains, through a door that is closed, has
   nothing to do.

**Why the version number survives.** Renumbering is what
``_verify_ledger_identity`` exists to catch: shift v004 down into 3 and any
database carrying the old ledger reads the shifted version as already applied
and never runs its body. The runner walks ``sorted(package.versions)`` and
tolerates gaps, but a number that has ever meant one thing must not come to
mean another. So 3 stays claimed, by a body that does nothing and says so.

**What it means to apply this today.** Nothing, and that is not a silence --
it is logged. Nothing applies it either: the hub retired the per-package
agent-scope chains in favour of one squashed package (``hub_agent_squash``),
and this package's ``register`` has no production caller in any repo -- its
only importer is a test. Whether an older hub release ever composed it is not
knowable from here and does not matter, because that retirement was taken
pre-GA, when there were no production agent schemas to have run it against.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["workspace_namespace_backfill"]

log = get_logger(__name__)


async def workspace_namespace_backfill(store: DataStore) -> None:
    """
    no-op. the workspace namespace backfill was retired; see the module docstring.

    executes no statement. the row this once inserted is written by the hub
    off ``workspace_create``'s emitted event, and the cross-schema statement
    it used to issue could only name the hub's schema by hardcoding one.

    :param store: DataStore bound to per-agent schema via search_path;
        deliberately unused
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    _ = store
    log.info(
        "agent-workspace v003 is retired and applies nothing: the paired namespace row for a "
        "workspace is written by the hub off workspace_create's emitted event",
    )
