"""enforcement: every request that carries a proof of possession leaves an absent proof off the wire.

A hub from before the door proofs (0.66.0 to 0.68.0) reads some door requests with ``extra="forbid"``,
so a body carrying ``"identity_proof": null`` is refused there: a new pod's every query fails against
an older hub. :class:`threetears.core.security.door_request.ProvenDoorRequest` holds the field and the
serializer that omits it together. This gate holds the construction:

- no class declares ``identity_proof`` itself; it comes from the base, with its serializer;
- every subclass of the base, a subclass that replaces the serializer included, puts no proof key on
  the wire when it has no proof, and keeps one it has.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from threetears.core.security.door_request import DOOR_PROOF_FIELD, ProvenDoorRequest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_ROOTS = tuple(
    sorted(
        path
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)

#: the one class that declares the field
_BASE = "ProvenDoorRequest"


def hand_declared_proofs(source: str) -> list[str]:
    """the classes in ``source`` that declare the proof field themselves rather than inherit it.

    :param source: a module's text
    :ptype source: str
    :return: the class names, the base itself left out
    :rtype: list[str]
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name != _BASE:
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == DOOR_PROOF_FIELD
                ):
                    found.append(node.name)
    return found


def proof_on_the_wire_when_absent(model: type[BaseModel]) -> bool:
    """whether ``model`` serializes a proof key when it holds no proof.

    Every field is set to ``None`` without validation, so a model with required fields is checked
    too; the python dump runs the model's serializer, as the JSON one does.

    :param model: a proof-carrying request model
    :ptype model: type[BaseModel]
    :return: ``True`` when the key reaches the wire
    :rtype: bool
    """
    built = model.model_construct(**dict.fromkeys(model.model_fields))
    return DOOR_PROOF_FIELD in built.model_dump(warnings=False)


def proof_dropped_when_present(model: type[BaseModel]) -> bool:
    """whether ``model`` loses a proof it holds.

    :param model: a proof-carrying request model
    :ptype model: type[BaseModel]
    :return: ``True`` when a held proof does not reach the wire
    :rtype: bool
    """
    values: dict[str, Any] = dict.fromkeys(model.model_fields)
    values[DOOR_PROOF_FIELD] = "proof"
    built = model.model_construct(**values)
    return built.model_dump(warnings=False).get(DOOR_PROOF_FIELD) != "proof"


def _modules_naming_the_base() -> list[str]:
    """the source modules that name the base, so importing them registers every subclass.

    :return: module names
    :rtype: list[str]
    """
    names: list[str] = []
    for root in _SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if _BASE in path.read_text(encoding="utf-8"):
                names.append(".".join(path.relative_to(root).with_suffix("").parts))
    return names


def _subclasses(model: type[BaseModel]) -> list[type[BaseModel]]:
    """every subclass of ``model``, at any depth.

    :param model: the base
    :ptype model: type[BaseModel]
    :return: its subclasses
    :rtype: list[type[BaseModel]]
    """
    found: list[type[BaseModel]] = []
    for sub in model.__subclasses__():
        found.append(sub)
        found.extend(_subclasses(sub))
    return found


def _proof_carrying_models() -> list[type[BaseModel]]:
    """every model in the source that carries the proof field, through the base.

    :return: the models
    :rtype: list[type[BaseModel]]
    """
    for name in _modules_naming_the_base():
        importlib.import_module(name)
    return [model for model in _subclasses(ProvenDoorRequest) if model.__module__.startswith("threetears.")]


def test_no_model_declares_the_proof_field_by_hand() -> None:
    """the field comes only from the base, so its serializer always comes with it.

    :return: nothing
    :rtype: None
    """
    declared = [
        f"{path.relative_to(_REPO_ROOT)}::{name}"
        for root in _SOURCE_ROOTS
        for path in sorted(root.rglob("*.py"))
        for name in hand_declared_proofs(path.read_text(encoding="utf-8"))
    ]
    assert not declared, f"declare identity_proof through ProvenDoorRequest, never by hand: {declared}"


def test_every_proof_carrying_model_omits_an_absent_proof_and_keeps_a_present_one() -> None:
    """no proof, no key; a proof, its key.

    :return: nothing
    :rtype: None
    """
    models = _proof_carrying_models()
    assert len(models) >= 5, f"found {len(models)} proof-carrying models; the walk is not seeing the source"
    leaking = [f"{m.__module__}.{m.__qualname__}" for m in models if proof_on_the_wire_when_absent(m)]
    assert not leaking, f"these put a null identity_proof on the wire: {leaking}"
    dropping = [f"{m.__module__}.{m.__qualname__}" for m in models if proof_dropped_when_present(m)]
    assert not dropping, f"these drop a proof they hold: {dropping}"


def test_a_hand_declared_proof_field_is_reported() -> None:
    """the source reader names a class that declares the field itself, and not the base.

    :return: nothing
    :rtype: None
    """
    source = (
        "class ProvenDoorRequest(BaseModel):\n"
        "    identity_proof: str | None = None\n"
        "class Sneaky(BaseModel):\n"
        "    identity_token: str\n"
        "    identity_proof: str | None = None\n"
    )
    assert hand_declared_proofs(source) == ["Sneaky"]


def test_a_subclass_whose_own_serializer_forgets_the_rule_is_reported() -> None:
    """a subclass that replaces the base's serializer without omitting the proof leaks it; one that keeps it does not.

    :return: nothing
    :rtype: None
    """

    class _Forgets(ProvenDoorRequest):
        identity_token: str

        @model_serializer(mode="wrap")
        def _own(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            data: dict[str, Any] = handler(self)
            return data

    class _Keeps(ProvenDoorRequest):
        identity_token: str

    assert proof_on_the_wire_when_absent(_Forgets)
    assert not proof_on_the_wire_when_absent(_Keeps)
    assert not proof_dropped_when_present(_Keeps)
