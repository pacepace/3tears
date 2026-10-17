"""The declaration types the eval engine ships its seed prompts in.

The engine owns the prompts its own LLM calls are steered by, and a host
registers them. That direction needs a vocabulary the engine can speak without
naming a host type:
:class:`SeedTemplate` and :class:`SeedSection` mirror the shape a sectioned
prompt template has, in plain values, and :class:`SeedPrompt` says where each
shipped default lives so a host's registry can find it by import rather than by
holding a source path of its own into this package.

Nothing here imports a host model. That is the constraint that made these types
exist rather than reusing a host's own template model: this package serves hosts
that have none of it, and a seed declaration that could only be written in one
host's model would travel nowhere. The host adapts — building its own template from
:meth:`SeedTemplate.to_dict` is the whole adaptation, and that payload is the contract
it reads.

Two prompt shapes are declared with these types, and they differ in what a
registry does with them. A ``template`` seed is a sectioned template a template
registry seeds; a ``text`` seed is a single string a shared-prompt registry
seeds. Both seed
**empty slots only** — the store is master once set, so what these
constants govern is a fresh deployment, not a running one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A seed whose constant is a sectioned template, registered with a template registry.
KIND_TEMPLATE = "template"

#: A seed whose constant is a single string, registered with a shared-prompt registry.
KIND_TEXT = "text"


@dataclass(frozen=True)
class SeedSection:
    """One section of a seeded prompt template.

    Field names and value vocabulary match the payload a host template model
    validates, so :meth:`SeedTemplate.to_dict` needs no translation table.

    Provisional: the zones (``primacy``, ``reference``, ``recency``) and the section's shape were taken from the
    first host that adopted the engine, and are kept or removed by whether a second host's adoption uses them
    (``docs/adopting-a-host.md``, "Provisional mechanisms").

    :ivar section_name: Stable identifier for the section within its template.
    :ivar zone: Ordering zone — ``primacy``, ``reference`` or ``recency``.
    :ivar order: Rank within the zone; non-negative.
    :ivar condition: ``always`` to include unconditionally, ``has_data`` to include
        only when ``data_key`` resolves.
    :ivar formatter: Name of the host-registered function that renders the section.
        ``text_template`` renders ``content_template`` literally.
    :ivar cache_class: ``stable`` or ``dynamic`` — whether the rendered text is
        expected to change between calls.
    :ivar content_template: The literal body, for a ``text_template`` section.
    :ivar data_key: The context key a ``has_data`` condition tests.
    :ivar description: One line about what the section carries, shown to an operator.
    :ivar formatter_params: Extra arguments for a parameterised formatter.
    """

    section_name: str
    zone: str
    order: int
    condition: str
    formatter: str
    cache_class: str
    content_template: str = ""
    data_key: str = ""
    description: str = ""
    formatter_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the section as the payload a host template model validates.

        :return: Every field, by its declared name, with no host types in it.
        :rtype: dict[str, Any]
        """
        return {
            "section_name": self.section_name,
            "zone": self.zone,
            "order": self.order,
            "condition": self.condition,
            "data_key": self.data_key,
            "formatter": self.formatter,
            "content_template": self.content_template,
            "cache_class": self.cache_class,
            "description": self.description,
            "formatter_params": self.formatter_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeedSection:
        """Rebuild a section from :meth:`to_dict`'s payload.

        :param data: A payload carrying at least the required fields.
        :ptype data: dict[str, Any]
        :return: The section.
        :rtype: SeedSection
        :raises KeyError: When a required field is absent.
        """
        return cls(
            section_name=data["section_name"],
            zone=data["zone"],
            order=data["order"],
            condition=data["condition"],
            formatter=data["formatter"],
            cache_class=data["cache_class"],
            content_template=data.get("content_template", ""),
            data_key=data.get("data_key", ""),
            description=data.get("description", ""),
            formatter_params=data.get("formatter_params"),
        )


@dataclass(frozen=True)
class SeedTemplate:
    """A seeded prompt template: an ordered set of sections and its identity.

    :ivar template_name: Preset name within the type — ``default`` for a shipped seed.
    :ivar template_type: The registry type this seeds, as its wire value.
    :ivar scope: ``base``; a shipped seed is never a per-subject override.
    :ivar description: Catalog blurb shown where an operator picks a preset.
    :ivar sections: The sections, in declaration order.
    :ivar version: Revision of the shipped text, bumped by a promotion.
    """

    template_name: str
    template_type: str
    scope: str
    description: str
    sections: tuple[SeedSection, ...] = ()
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return the template as the payload a host template model validates.

        :return: The declared fields, with ``sections`` flattened to payload dicts.
            Timestamps and provenance are deliberately absent — a shipped
            constant has neither, and a host fills its own defaults.
        :rtype: dict[str, Any]
        """
        return {
            "template_name": self.template_name,
            "template_type": self.template_type,
            "scope": self.scope,
            "description": self.description,
            "sections": [section.to_dict() for section in self.sections],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeedTemplate:
        """Rebuild a template from :meth:`to_dict`'s payload.

        This is the shape a promotion writes back to this package: a tuned
        preset leaves an operator's store as a payload, and the constant it
        replaces is re-emitted as ``SeedTemplate.from_dict({...})``.

        :param data: A payload carrying the declared fields.
        :ptype data: dict[str, Any]
        :return: The template.
        :rtype: SeedTemplate
        :raises KeyError: When a required field is absent.
        """
        return cls(
            template_name=data["template_name"],
            template_type=data["template_type"],
            scope=data["scope"],
            description=data.get("description", ""),
            sections=tuple(SeedSection.from_dict(section) for section in data.get("sections", ())),
            version=data.get("version", 1),
        )


@dataclass(frozen=True)
class SeedPrompt:
    """Where one shipped default lives, so a host can register it by import.

    The host's registries used to hold this themselves — a module path and a
    source path, written host-side, pointing into this package. Reversing that
    is the point of the declaration: a prompt that moves within the engine moves
    its own entry with it, and no host edit is required.

    :ivar kind: :data:`KIND_TEMPLATE` or :data:`KIND_TEXT` — which registry seeds it.
    :ivar prompt_type: The registry type's wire value.
    :ivar seed_module: Importable module holding the constant. A host's registry
        imports this to SEED a fresh deployment.
    :ivar constant: The constant's name in that module.
    """

    kind: str
    prompt_type: str
    seed_module: str
    constant: str

    def __post_init__(self) -> None:
        """Refuse a declaration no registry could seed from: an unknown kind, or a blank field.

        Checked where the declaration is written, because a host registers its own prompts through this
        type too, and a typo here otherwise surfaces as a slot a fresh deployment silently never seeds.

        :raises ValueError: ``kind`` is neither :data:`KIND_TEMPLATE` nor :data:`KIND_TEXT`, or a field is blank.
        """
        if self.kind not in (KIND_TEMPLATE, KIND_TEXT):
            raise ValueError(
                f"seed prompt {self.prompt_type!r} declares kind {self.kind!r}; a seed is {KIND_TEMPLATE!r} (a sectioned "
                f"template, for a template registry) or {KIND_TEXT!r} (one string, for a shared-prompt registry)"
            )
        if blank := [name for name in ("prompt_type", "seed_module", "constant") if not getattr(self, name).strip()]:
            raise ValueError(
                f"seed prompt {self.prompt_type!r} leaves {', '.join(blank)} blank; a registry finds the seed by all three"
            )

    @property
    def seed_ref(self) -> tuple[str, str]:
        """``(module, attribute)`` — how a registry resolves this seed by import."""
        return (self.seed_module, self.constant)


__all__ = [
    "KIND_TEMPLATE",
    "KIND_TEXT",
    "SeedPrompt",
    "SeedSection",
    "SeedTemplate",
]
