"""A :class:`~threetears.evals.schema.DocumentStore` over several stores, each document type placed by its facts.

A host that keeps its evidence in one database and what the engine can remake in a cheaper one, or what people
author apart from what runs record, routes each ``doc_type`` to a store of its own. :class:`RoutedDocumentStore`
does that behind the one port the engine stores through, so nothing above it learns there is more than one store.

**The rule is over facts, never names.** :class:`DocTypeRoute` keys on one fact of
:data:`~threetears.evals.schema.doc_types.DOC_TYPE_FACTS` (``origin`` or ``lifecycle``) and maps every value of
that fact to a store, and must map every one: a rule that leaves a value out is refused, so a type the engine adds
later lands where its facts say rather than nowhere. Types the registry does not hold (a host's own documents) go
to the one store the rule names for them. The rule is resolved into a table, one store per registered type, when
the router is built, so every mistake a rule can hold is a refusal at construction rather than a misplaced write:
a store name that is not a child, a value left unmapped, and a tombstone placed apart from the type it suppresses
(the seed reads them together, so a definition in one store and its tombstone in another would write a deleted
key back).

**Typed calls route; id-only calls fan out.** ``upsert`` (by the document's ``doc_type``), ``get_many``,
``by_doc_type`` and ``iter_by_doc_type`` go to the one store their type is placed in, and a conditional write's
``if_match`` reaches that store unchanged, so each store's etags mean what they always meant. ``get``,
``get_with_etag``, ``merge_fields`` and ``delete`` carry only an id, so they ask every store. **One id, one
store**: a read that finds the id in two stores, or in a store its type is not placed in, raises
:class:`MisplacedDocument` rather than picking one, because whichever it picked, a typed read of the same type
reads the other. A store that does not hold the id answers "not here" (``None``, ``False``), which is the port's
miss, never a refusal; so an id-only write reports ``False`` only when no store holds it.

**Changing the rule is an operation.** Each store holds a record of the placements it was last started under,
in a scope reserved for it (:data:`ROUTING_SCOPE`), under a ``doc_type`` no engine call names. The router refuses to start
(:class:`RouteChanged`) when any type would now be read from a store other than the one an earlier start wrote it
to, when a store's record names another store (two stores passed under each other's names), when the records
disagree, and when no store holds a record at all, since a store adopted with documents already in it cannot be
told from an empty one through the port. :meth:`RoutedDocumentStore.relocate` is the operation: it moves every
document of the scopes its host names to its placement, then writes the record. A type the record does not name
(new in this release) is read where the record's store for unregistered types is, so a new type changes nothing
unless it is placed elsewhere.

The record is the router's bookkeeping about the stores, not an engine document: it is not routed by facts, it
is written to every store alike, and it is not in :data:`~threetears.evals.schema.models.EVAL_DOC_TYPES` because
no operator wipe should remove it and no adapter's conformance run should be asked to store it. It lives in its
own scope, under its own ``doc_type``, so no typed read of a document scope meets it. The router does not police
the scope (tenancy is opaque: nothing in the engine reads a scope's value), so a host must never use
:data:`ROUTING_SCOPE` as a scope of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from threetears.evals.schema.doc_types import DOC_TYPE_FACTS, DocLifecycle, DocOrigin, DocTypeFacts
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.store_port import DocumentStore

__all__ = ["ROUTING_SCOPE", "DocTypeRoute", "MisplacedDocument", "RouteChanged", "RoutedDocumentStore"]

#: The scope each store's placement record lives in, reserved: a host never uses it as a scope of its own. An
#: adapter that isolates by scope must be able to hold it.
ROUTING_SCOPE: Final[str] = "threetears.evals.routing"

#: The ``doc_type`` and ``id`` of a store's placement record, inside :data:`ROUTING_SCOPE`.
_RECORD_DOC_TYPE: Final[str] = "eval_routing_record"
_RECORD_ID: Final[str] = "placements"

#: The facts a rule may key on, and the values each takes.
_FACT_VALUES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"origin": get_args(DocOrigin), "lifecycle": get_args(DocLifecycle)}
)


class RouteChanged(RuntimeError):
    """The router was started under placements its stores were not written under; :meth:`relocate` first."""


class MisplacedDocument(RuntimeError):
    """One id is held by two stores, or by a store its type is not placed in; the router never picks one."""


@dataclass(frozen=True)
class DocTypeRoute:
    """Which store each document type is kept in, decided by one of its facts.

    Attributes:
        by: The fact the rule keys on: ``origin`` or ``lifecycle``.
        to: Every value of that fact, to the name of the store its types are kept in. Exhaustive: a value left out
            is refused, so no registered type is ever unplaced.
        unregistered: The store a ``doc_type`` the registry does not hold is kept in — a host's own documents.
            Required, so where they go is a decision rather than a default.
    """

    by: Literal["origin", "lifecycle"]
    to: Mapping[str, str]
    unregistered: str

    def __post_init__(self) -> None:
        """Refuse a fact that is not one, and a mapping that misses or invents a value of it.

        Raises:
            ValueError: ``by`` names no fact, ``to`` leaves a value of it unmapped or names one it does not
                take, or a store name is blank.
        """
        values = _FACT_VALUES.get(self.by)
        if values is None:
            raise ValueError(f"a route keys on one of {sorted(_FACT_VALUES)}, not {self.by!r}")
        missing = sorted(set(values) - set(self.to))
        unknown = sorted(set(self.to) - set(values))
        if missing or unknown:
            raise ValueError(
                f"a route by {self.by} maps every value of it, {list(values)}, and nothing else: "
                f"missing {missing}, unknown {unknown}"
            )
        if not self.unregistered or not all(self.to.values()):
            raise ValueError("a route names a store for every value and for unregistered types; a name is never blank")
        object.__setattr__(self, "to", MappingProxyType(dict(self.to)))

    def placements(self, facts: Mapping[str, DocTypeFacts]) -> Mapping[str, str]:
        """Resolve the rule over a registry: the store each registered type is kept in.

        Args:
            facts: The registry (:data:`~threetears.evals.schema.doc_types.DOC_TYPE_FACTS`).

        Returns:
            ``doc_type`` to store name, for every registered type.

        Raises:
            ValueError: A tombstone would be kept in another store than the type it suppresses.
        """
        table = {doc_type: self.to[getattr(fact, self.by)] for doc_type, fact in facts.items()}
        for doc_type, fact in facts.items():
            if fact.suppresses is None:
                continue
            suppressed = table.get(fact.suppresses, self.unregistered)
            if table[doc_type] != suppressed:
                raise ValueError(
                    f"this route keeps {doc_type} in {table[doc_type]!r} and the {fact.suppresses} it suppresses in "
                    f"{suppressed!r}; a tombstone is read with the type it suppresses, so they are kept in one store"
                )
        return MappingProxyType(table)


def _fingerprint(placements: Mapping[str, str], unregistered: str) -> str:
    """The digest of a set of placements: equal exactly when every type is kept in the same store."""
    return canonical_digest({"placements": dict(placements), "unregistered": unregistered})


def _record(child: str, placements: Mapping[str, str], unregistered: str) -> dict[str, Any]:
    """The placement record written to the store named ``child``: the placements whole, and their fingerprint.

    The placements are kept whole rather than as the digest alone so a start can tell a type that moved (refused)
    from a type that is merely new (placed where unregistered types were, and recorded).
    """
    return {
        "id": _RECORD_ID,
        "scope_id": ROUTING_SCOPE,
        "doc_type": _RECORD_DOC_TYPE,
        "child": child,
        "placements": dict(sorted(placements.items())),
        "unregistered": unregistered,
        "fingerprint": _fingerprint(placements, unregistered),
    }


def _checked_children(children: Mapping[str, DocumentStore], route: DocTypeRoute) -> Mapping[str, str]:
    """Resolve ``route`` over the registry and refuse a route or a set of stores the router cannot run on.

    Returns:
        The placement table.

    Raises:
        ValueError: No stores, one store passed under two names, or a route naming a store that is not a child.
    """
    if not children:
        raise ValueError("a routed store needs at least one store")
    if len({id(store) for store in children.values()}) != len(children):
        raise ValueError("one store is passed under two names; every id would read as held twice")
    placements = route.placements(DOC_TYPE_FACTS)
    unknown = sorted({*placements.values(), route.unregistered} - set(children))
    if unknown:
        raise ValueError(f"the route names stores {unknown} that are not among {sorted(children)}")
    return placements


class RoutedDocumentStore:
    """One :class:`~threetears.evals.schema.DocumentStore` over several, each ``doc_type`` kept where its facts say.

    See the module docstring for the routing, the fan-out, and why the router refuses to start after its rule
    changed.

    Args:
        children: The stores, by name. A store no type is placed in is still asked by every id-only call, so one
            being drained stays reachable until :meth:`relocate` has emptied it.
        route: Where each type is kept.

    Raises:
        ValueError: The route or the stores are refused (see :class:`DocTypeRoute` and :meth:`relocate`).
        RouteChanged: The stores were written under other placements, or under none.
    """

    def __init__(self, children: Mapping[str, DocumentStore], route: DocTypeRoute) -> None:
        """Resolve the route, and refuse to start unless every store was written under these placements."""
        self._children: Mapping[str, DocumentStore] = MappingProxyType(dict(children))
        self._route = route
        self._placements = _checked_children(self._children, route)
        self._check_records()

    @property
    def placements(self) -> Mapping[str, str]:
        """The store each registered ``doc_type`` is kept in, by name."""
        return self._placements

    @property
    def unregistered(self) -> str:
        """The store a ``doc_type`` the registry does not hold is kept in."""
        return self._route.unregistered

    def _check_records(self) -> None:
        """Refuse to start when a store's record says documents were placed otherwise, or no store has one.

        When nothing moved but the record is not this router's whole — a type new in this release, placed where
        unregistered types were, or a store added that holds no record yet — every store's record is rewritten.

        Raises:
            RouteChanged: See the module docstring.
        """
        records = {name: store.get(_RECORD_ID, ROUTING_SCOPE) for name, store in self._children.items()}
        held = {name: record for name, record in records.items() if record is not None}
        remedy = "run RoutedDocumentStore.relocate over every scope with the engine stopped, then start again"
        if not held:
            raise RouteChanged(
                f"none of the stores {sorted(self._children)} records the placements it was written under, so "
                f"documents already in one cannot be told from an empty store; {remedy}"
            )
        for name, record in held.items():
            if record.get("child") != name:
                raise RouteChanged(f"the store passed as {name!r} was written as {record.get('child')!r}; {remedy}")
        if len({record.get("fingerprint") for record in held.values()}) != 1:
            raise RouteChanged(f"the stores record different placements (an interrupted relocation); {remedy}")
        first = next(iter(held.values()))
        was: Mapping[str, str] = first["placements"]
        was_unregistered: str = first["unregistered"]
        moved = sorted(
            doc_type
            for doc_type in {*was, *self._placements}
            if was.get(doc_type, was_unregistered) != self._placements.get(doc_type, self.unregistered)
        )
        if was_unregistered != self.unregistered:
            moved.append("(unregistered types)")
        if moved:
            raise RouteChanged(f"this route keeps {moved} in another store than they were written to; {remedy}")
        if len(held) != len(self._children) or first["fingerprint"] != _fingerprint(
            self._placements, self.unregistered
        ):
            self._write_records()

    def _write_records(self) -> None:
        """Write this router's placements to every store's record."""
        for name, store in self._children.items():
            store.upsert(_record(name, self._placements, self.unregistered))

    # --- routing ---------------------------------------------------------------------------------

    def _place(self, doc_type: str) -> str:
        """The name of the store ``doc_type`` is kept in."""
        return self._placements.get(doc_type, self.unregistered)

    def _routed(self, doc_type: str) -> DocumentStore:
        """The store ``doc_type`` is kept in."""
        return self._children[self._place(doc_type)]

    def _holders(self, doc_id: str, scope_id: str) -> list[tuple[str, dict[str, Any], str | None]]:
        """Every store holding ``doc_id`` in ``scope_id``, with the document and its etag."""
        found = []
        for name, store in self._children.items():
            document, etag = store.get_with_etag(doc_id, scope_id)
            if document is not None:
                found.append((name, document, etag))
        return found

    def _one_holder(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """The one document ``doc_id`` names and its etag, refusing two holders or a misplaced one.

        Raises:
            MisplacedDocument: The id is held by two stores, or by one its type is not placed in.
        """
        found = self._holders(doc_id, scope_id)
        if not found:
            return None, None
        if len(found) > 1:
            raise MisplacedDocument(
                f"document {doc_id!r} in scope {scope_id!r} is held by {[name for name, _, _ in found]}; one id is "
                "kept in one store, so neither is read — relocate settles it, or remove the copy that is wrong"
            )
        name, document, etag = found[0]
        expected = self._place(str(document.get("doc_type")))
        if name != expected:
            raise MisplacedDocument(
                f"document {doc_id!r} ({document.get('doc_type')}) in scope {scope_id!r} is held by {name!r}, and "
                f"its type is kept in {expected!r}, where a typed read would not find it; relocate moves it"
            )
        return document, etag

    def _fan_out_write(self, doc_id: str, scope_id: str, outcomes: Iterable[tuple[str, bool]], what: str) -> bool:
        """Settle an id-only write asked of every store: True when one held it, a refusal when several did.

        Raises:
            MisplacedDocument: More than one store held the id, so the write reached each of them.
        """
        landed = [name for name, wrote in outcomes if wrote]
        if len(landed) > 1:
            raise MisplacedDocument(
                f"{what} of {doc_id!r} in scope {scope_id!r} reached {landed}, which each held it; one id is kept "
                "in one store — relocate settles it"
            )
        return bool(landed)

    # --- the port --------------------------------------------------------------------------------

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Return the one document ``doc_id`` names in a scope, from whichever store holds it.

        Raises:
            MisplacedDocument: Two stores hold it, or one its type is not placed in.
        """
        return self._one_holder(doc_id, scope_id)[0]

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return the one document and the etag of the store holding it, which a conditional write presents there.

        Raises:
            MisplacedDocument: Two stores hold it, or one its type is not placed in.
        """
        return self._one_holder(doc_id, scope_id)

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return the documents of one type whose ids are in ``doc_ids``, from the store the type is kept in."""
        return self._routed(doc_type).get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Return documents of one type within a scope, from the store the type is kept in."""
        return self._routed(doc_type).by_doc_type(
            doc_type, scope_id, order_by=order_by, descending=descending, limit=limit, exclude=exclude, **field_eq
        )

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Yield the id of every document of one type within a scope, from the store the type is kept in."""
        return self._routed(doc_type).iter_by_doc_type(doc_type, scope_id)

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        """Write a document to the store its ``doc_type`` is kept in, passing ``if_match`` through unchanged.

        Raises:
            ValueError: The document carries no ``doc_type``, which is what places it.
            StoreConflict: Raised by the store the type is kept in.
        """
        doc_type = document.get("doc_type")
        if not isinstance(doc_type, str):
            raise ValueError("a routed store places a document by its doc_type, and this one carries none")
        self._routed(doc_type).upsert(document, if_match=if_match)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Set top-level fields on the one document ``doc_id`` names, in whichever store holds it.

        Returns:
            ``True`` when a store held it; ``False`` when none did.

        Raises:
            MisplacedDocument: Several stores held it (the fields were set in each).
            ValueError: Raised by the stores for empty or locating fields.
        """
        outcomes = [(name, store.merge_fields(doc_id, scope_id, fields)) for name, store in self._children.items()]
        return self._fan_out_write(doc_id, scope_id, outcomes, "merge_fields")

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete the one document ``doc_id`` names, from whichever store holds it.

        Returns:
            ``True`` when a store held it; ``False`` when none did.

        Raises:
            MisplacedDocument: Several stores held it (each copy was deleted).
        """
        outcomes = [(name, store.delete(doc_id, scope_id)) for name, store in self._children.items()]
        return self._fan_out_write(doc_id, scope_id, outcomes, "delete")

    # --- changing the rule -----------------------------------------------------------------------

    @classmethod
    def relocate(
        cls,
        children: Mapping[str, DocumentStore],
        route: DocTypeRoute,
        *,
        scope_ids: Iterable[str],
        host_doc_types: Sequence[str] = (),
    ) -> dict[str, int]:
        """Move every document of the named scopes to the store ``route`` keeps it in, then record the placements.

        The operation that changes a router's rule, and that adopts stores into one. Run it with the engine
        stopped: a write landing in a store while its documents are moved out of it can be left behind. It is
        safe to run again after an interruption: a document already copied is found identical where it belongs
        and its other copy removed, and the record is written only once every document is in place. Like the
        operator wipe, it reaches the scopes and the host types it is told of: the port has no scope-free read,
        and a type it is not named holds documents it cannot find.

        Args:
            children: The stores, by name, as the router will be built over them.
            route: Where each type is to be kept.
            scope_ids: Every scope with documents in these stores.
            host_doc_types: The host's own ``doc_type`` values in these stores, kept in ``route.unregistered``.

        Returns:
            How many documents were moved, by ``doc_type``.

        Raises:
            ValueError: The route or the stores are refused.
            MisplacedDocument: Two stores hold different documents under one id; which is right is an
                operator's decision, and nothing of that id is moved.
        """
        placements = _checked_children(children, route)
        known: set[str] = set(placements) | set(host_doc_types)
        for store in children.values():
            record = store.get(_RECORD_ID, ROUTING_SCOPE)
            if record is not None:
                known |= set(record.get("placements", {}))
        moved: dict[str, int] = {}
        for scope_id in scope_ids:
            for doc_type in sorted(known):
                target_name = placements.get(doc_type, route.unregistered)
                target = children[target_name]
                for name, store in children.items():
                    if name == target_name:
                        continue
                    for doc_id in list(store.iter_by_doc_type(doc_type, scope_id)):
                        document = store.get(doc_id, scope_id)
                        if document is None:
                            continue
                        there = target.get(doc_id, scope_id)
                        if there is None:
                            target.upsert(document)
                        elif there != document:
                            raise MisplacedDocument(
                                f"document {doc_id!r} in scope {scope_id!r} is held by {name!r} and by "
                                f"{target_name!r}, and the two differ; decide which is right and remove the other"
                            )
                        store.delete(doc_id, scope_id)
                        moved[doc_type] = moved.get(doc_type, 0) + 1
        for name, store in children.items():
            store.upsert(_record(name, placements, route.unregistered))
        return moved
