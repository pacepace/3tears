"""The fidelity checker as an adopter uses it: a host's source canary over its own registry.

A host proves its eval calls the production constructor by registering a
:class:`~threetears.evals.run.FidelityContract` and asserting, in its own test suite, that
:func:`~threetears.evals.run.callers_missing_the_constructor` names no caller. This file is that
canary for the toy host (``fixtures/toyhost/fidelity.py``), written against the public root alone —
an adopter imports nothing below ``threetears.evals.run``.

The canary is proven from both sides: it passes on the toy host as shipped, and it names the eval
caller once that caller's source stops reaching the constructor. The red half rewrites a COPY of the
toy kind with the constructor call replaced by an inline prompt — the exact divergence the pattern
exists to catch — because a check that only ever passed is evidence of nothing.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.evals.run import FidelityContract, callers_missing_the_constructor, resolve_constructor
from packages.evals.tests.fixtures.toyhost import kind as toy_kind
from packages.evals.tests.fixtures.toyhost import product
from packages.evals.tests.fixtures.toyhost.fidelity import TOYHOST_FIDELITY_CONTRACTS
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_DOCUMENTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.run import execute_toyhost_run

#: The kind's one call to the constructor, as the shipped source spells it.
_CONSTRUCTOR_CALL = "request = extraction_request(model=instance.model, document=document)"
#: The divergence: the eval builds its own request, as an eval written beside a product so often does.
_INLINE_REQUEST = 'request = ExtractionRequest(model=instance.model, prompt="extract the fields", document=document)'


@pytest.mark.parametrize("contract", TOYHOST_FIDELITY_CONTRACTS, ids=lambda contract: contract.behavior)
def test_every_declared_caller_reaches_its_constructor(contract: FidelityContract) -> None:
    assert callable(resolve_constructor(contract))
    assert callers_missing_the_constructor(contract) == []


def test_the_registered_constructor_is_the_one_production_and_the_eval_both_import() -> None:
    (contract,) = TOYHOST_FIDELITY_CONTRACTS
    constructor = resolve_constructor(contract)
    assert constructor is product.extraction_request
    assert toy_kind.extraction_request is constructor, "the eval side must import the product's constructor"


@pytest.fixture
def diverged_kind_module(tmp_path: Path) -> Iterator[str]:
    """A copy of the toy kind whose invoke builds its own request instead of calling the constructor."""
    source = Path(toy_kind.__file__).read_text(encoding="utf-8")
    assert source.count(_CONSTRUCTOR_CALL) == 1, "the toy kind no longer spells its constructor call as expected"
    diverged = source.replace(_CONSTRUCTOR_CALL, _INLINE_REQUEST).replace(
        "import ExtractionRequest, extraction_request", "import ExtractionRequest"
    )
    # The mutation must have LANDED: a copy that still calls or imports the constructor would pass
    # for the wrong reason. (Prose naming it in a docstring is no reference, and the walk ignores it.)
    assert "extraction_request(" not in diverged and "extraction_request\n" not in diverged
    name = f"diverged_toy_kind_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text(diverged, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)


def test_the_canary_names_an_eval_caller_that_stops_calling_the_constructor(diverged_kind_module: str) -> None:
    (contract,) = TOYHOST_FIDELITY_CONTRACTS
    production, _eval = contract.callers
    diverged = FidelityContract(
        behavior=contract.behavior,
        constructor=contract.constructor,
        callers=(production, diverged_kind_module),
        why=contract.why,
    )
    assert callers_missing_the_constructor(diverged) == [diverged_kind_module]


@pytest.fixture
def import_only_kind_module(tmp_path: Path) -> Iterator[str]:
    """A copy of the toy kind that still IMPORTS the constructor but builds its own request."""
    source = Path(toy_kind.__file__).read_text(encoding="utf-8")
    assert source.count(_CONSTRUCTOR_CALL) == 1, "the toy kind no longer spells its constructor call as expected"
    diverged = source.replace(_CONSTRUCTOR_CALL, _INLINE_REQUEST)
    # Landed, and landed the way this case needs: the call is gone and the import is still there.
    assert "extraction_request(" not in diverged
    assert "import ExtractionRequest, extraction_request" in diverged
    name = f"import_only_toy_kind_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text(diverged, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)


def test_an_import_left_behind_does_not_count_as_reaching_the_constructor(import_only_kind_module: str) -> None:
    """The drift a canary most often misses: the call was replaced and the import never cleaned up."""
    (contract,) = TOYHOST_FIDELITY_CONTRACTS
    production, _eval = contract.callers
    diverged = FidelityContract(
        behavior=contract.behavior,
        constructor=contract.constructor,
        callers=(production, import_only_kind_module),
        why=contract.why,
    )
    assert callers_missing_the_constructor(diverged) == [import_only_kind_module]


async def test_production_and_the_eval_hand_the_provider_identical_requests() -> None:
    """The boundary-equality half: the canary proves the eval REACHES the constructor, this the same bytes."""
    path = await execute_toyhost_run(host=toyhost_host())
    eval_requests = {(request.model, request.document.document_id): request for request in path.client.requests}
    assert eval_requests, "the toy run sent no extraction request"

    production_client = ScriptedExtractionClient()
    documents = {document.document_id: document for document in TOY_DOCUMENTS}
    for (model, document_id), sent_by_eval in eval_requests.items():
        await product.extract_invoice(production_client, model=model, document=documents[document_id])
        assert production_client.requests[-1] == sent_by_eval
