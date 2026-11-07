"""per-package configuration for the import-floor gate."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ImportFloorConfig"]


@dataclass(frozen=True)
class ImportFloorConfig:
    """what one package promises a consumer importing its base install.

    Every collection is enumerated rather than derived, so a change in what a dependency drags
    in is a visible event to read rather than weight to absorb.

    :ivar base_modules: every module the base install imports, by dotted name. Listed rather than
        discovered, so a new module is a deliberate addition to the promise; a module behind an
        extra is left out of it on purpose
    :ivar third_party_roots: top-level import roots the base install pays for: its declared
        distributions, and the support modules each loads of its own accord
    :ivar family_prefixes: dotted prefixes of the family packages the base install declares,
        the package itself among them (``threetears.iam``, ``threetears.observe``)
    :ivar namespace_modules: the namespace packages on the way down to them (``threetears``),
        permitted by exact name, so permitting the root does not permit every package under it
    :ivar refused_roots: what the base install leaves out (``threetears.core``, ``sqlalchemy``),
        refused at import in the probe as an install without them refuses them. An optional
        extra's module imports under a guard at its top, so the probe sees the cost of the base
        install, not of the development environment that has every extra installed
    :raises ValueError: when ``base_modules`` is empty, or a refused root is also permitted
    """

    base_modules: tuple[str, ...]
    third_party_roots: frozenset[str]
    family_prefixes: tuple[str, ...]
    namespace_modules: frozenset[str] = field(default=frozenset({"threetears"}))
    refused_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.base_modules:
            raise ValueError("import floor: no base modules to probe -- the gate would pass by probing nothing")
        permitted = {*self.third_party_roots, *self.family_prefixes}
        if both := sorted(permitted.intersection(self.refused_roots)):
            raise ValueError(f"import floor: {both} are both refused and permitted")
