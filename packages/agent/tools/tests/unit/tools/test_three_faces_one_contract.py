"""success check 14 -- three faces, one contract (in-repo half).

The check asks that an HTTP API operation and an MCP tool be two more
*renderings* of the same candidate set, never a second result shape. Structure
is the primitive and rendering is a Bind (SR-A1); the risk the check names is
the ordinary one -- "a face gets added, someone shapes a response for it, and
the second result shape is born".

**Why this file verifies two faces and not three, and why that is the whole
in-repo half.** The face flags govern *reach*, and nothing in this repository
reads them: ``publish_registration`` copies them onto ``ToolManifestEntry``
(``server.py``), the ACL ``namespaces`` table persists them, and the surfaces
that act on them -- the API namespace stamp, the MCP export, the face-flip
re-stamp -- are hub-side. So:

* the **platform tool** face renders here, in ``TearsTool.execute`` ->
  ``ToolResult``;
* the **MCP** face renders here, in ``threetears.mcp.server``, which splits a
  structured handler result into prose ``content`` and the spec's
  ``structuredContent``;
* the **API** face renders in the hub (``map_tool_result_to_http``
  reconstructing from ``metadata["http"]``), and is verified where it is
  exposed -- out-of-repo in exactly the way success checks 1, 2, 3, 9 and 10
  already are, and excluded by Gate B's own "verifiable in-repo" wording.

Turning the flags on would not move that boundary by one line: it changes what
the hub is *told*, not what this repository renders. So the property is pinned
here against the renderings, and the reach is pinned where the reach lives.

**Every assertion compares one face to another**, never a face to the constant
it was built from. That direction is deliberate and was learned the expensive
way: ``test_egress_independence.py``'s first draft compared each side against
the value it had been configured with, so collapsing two exits into one left
every test green. A file that compares each face against a literal would pass
just as happily with two shapes as with one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import mcp.types as mcp_types
import pytest

from threetears.agent.tools.base_tool import ToolResult
from threetears.search.bind import (
    bind_candidate_set,
    bind_failure,
    project_failure_metadata,
    project_metadata,
)
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Candidate,
    CandidateSet,
    Locator,
    Provenance,
    QuotaExhausted,
    Spend,
)

_QUERY = "who ruled that structure is the primitive"


def _candidate(identity: str, url: str) -> Candidate:
    """One candidate, enough of it to be worth comparing across faces."""
    return Candidate(
        identity=identity,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query=_QUERY,
            provider_instance="searxng:test",
            retrieved_at=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
        ),
    )


def _candidate_set() -> CandidateSet:
    """A multi-candidate set, so a face that truncates is visibly different.

    More candidates than the prose binding renders would be pointless here --
    what matters is that the STRUCTURED projection carries all of them on every
    face, which a single-candidate fixture could not distinguish from a face
    that silently kept only the first.
    """
    return CandidateSet(
        candidates=tuple(_candidate(f"result-{n}", f"https://example.test/{n}") for n in range(3)),
        spend=Spend(calls=1),
    )


def _render_platform_face(rendered: Any) -> dict[str, Any]:
    """The platform tool face: what a ``ToolResult`` carries on the mesh.

    Built the way a builtin builds it -- ``RenderedSearch``'s four fields are
    exactly a ``ToolResult``'s, which is the seam's whole point -- so this is
    the real rendering rather than a restatement of it.
    """
    result = ToolResult(
        success=rendered.success,
        content=rendered.content,
        error=rendered.error,
        metadata=dict(rendered.metadata),
    )
    assert result.metadata is not None
    return result.metadata


def _render_mcp_face(rendered: Any) -> dict[str, Any]:
    """The MCP face: prose in ``content``, structure in ``structuredContent``.

    Mirrors ``threetears.mcp.server``'s normalization of a structured handler
    result rather than calling the dispatcher, because the dispatcher needs a
    registered tool, an identity and a permission check -- none of which bear
    on the question this file asks. The branch reproduced is the one that
    matters: a dict-shaped result rides BOTH registers.
    """
    payload: dict[str, Any] = dict(rendered.metadata)
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
        structuredContent=payload,
        isError=not rendered.success,
    )
    assert call_result.structuredContent is not None
    return dict(call_result.structuredContent)


class TestOneSuccessfulSearchAcrossFaces:
    """One candidate set; the faces must not disagree about what it was."""

    def test_both_faces_carry_the_named_key(self) -> None:
        rendered = bind_candidate_set(_QUERY, _candidate_set())

        platform = _render_platform_face(rendered)
        mcp_face = _render_mcp_face(rendered)

        assert SEARCH_RESULTS_METADATA_KEY in platform
        assert SEARCH_RESULTS_METADATA_KEY in mcp_face

    def test_the_two_faces_carry_the_same_structure(self) -> None:
        """The check itself: no second result shape per face.

        Compared face-to-face. If a future rendering reshapes, re-keys, drops a
        field or flattens the typed details for "its" consumer, these two stop
        being equal -- which is the only way this test can fail, and exactly the
        regression check 14 exists to make visible.
        """
        rendered = bind_candidate_set(_QUERY, _candidate_set())

        platform = _render_platform_face(rendered)
        mcp_face = _render_mcp_face(rendered)

        assert platform[SEARCH_RESULTS_METADATA_KEY] == mcp_face[SEARCH_RESULTS_METADATA_KEY]

    def test_every_candidate_survives_on_every_face(self) -> None:
        """Structure is unbounded where prose is bounded (PROSE_MAX_CANDIDATES).

        A face is allowed to render fewer candidates as PROSE. It is not allowed
        to carry fewer of them as STRUCTURE, because a program reading the
        structured face would then be reading a silently truncated answer.
        """
        candidate_set = _candidate_set()
        rendered = bind_candidate_set(_QUERY, candidate_set)

        platform = _render_platform_face(rendered)[SEARCH_RESULTS_METADATA_KEY]
        mcp_face = _render_mcp_face(rendered)[SEARCH_RESULTS_METADATA_KEY]

        assert len(platform["candidates"]) == len(candidate_set.candidates)
        assert len(mcp_face["candidates"]) == len(platform["candidates"])

    def test_prose_differs_from_structure_without_either_face_losing_it(self) -> None:
        """The registers are different; which register a face gets is not.

        Guards the inverse mistake: "one contract" satisfied by collapsing the
        two registers into one, so a model gets JSON or a program gets prose.
        """
        rendered = bind_candidate_set(_QUERY, _candidate_set())

        assert rendered.content
        assert rendered.content != json.dumps(rendered.metadata, default=str)
        assert _render_platform_face(rendered)[SEARCH_RESULTS_METADATA_KEY]
        assert _render_mcp_face(rendered)[SEARCH_RESULTS_METADATA_KEY]


class TestOneFailureAcrossFaces:
    """A failure is a rendering too, and carries spend on every face (D10)."""

    def test_the_two_faces_carry_the_same_failure_structure(self) -> None:
        rendered = bind_failure(_QUERY, QuotaExhausted("out of quota", spend=Spend(calls=1)))

        platform = _render_platform_face(rendered)
        mcp_face = _render_mcp_face(rendered)

        assert platform[SEARCH_RESULTS_METADATA_KEY] == mcp_face[SEARCH_RESULTS_METADATA_KEY]

    def test_a_failure_is_not_a_missing_key_on_either_face(self) -> None:
        """SR-E3: the failure path is where spend is most worth having."""
        rendered = bind_failure(_QUERY, QuotaExhausted("out of quota", spend=Spend(calls=1)))

        for face in (_render_platform_face(rendered), _render_mcp_face(rendered)):
            assert SEARCH_RESULTS_METADATA_KEY in face
            assert face[SEARCH_RESULTS_METADATA_KEY]["spend"] is not None


class TestOneProjectionAcrossProducers:
    """Two tools, one border shape -- the same property one axis over.

    ``web_search`` answers with many candidates and ``web_fetch`` with one, but
    a consumer reading structure off a tool result reads ONE shape either way
    (D22). Before this file existed ``web_fetch`` reimplemented the projection
    rather than calling it: identical output, two construction sites, and
    nothing keeping them identical.
    """

    def test_success_projection_has_one_construction_site(self) -> None:
        candidate_set = _candidate_set()

        via_bind = bind_candidate_set(_QUERY, candidate_set).metadata
        via_projection = project_metadata(_QUERY, candidate_set)

        assert via_bind == via_projection

    def test_failure_projection_has_one_construction_site(self) -> None:
        failure = QuotaExhausted("out of quota", spend=Spend(calls=1))

        via_bind = bind_failure(_QUERY, failure).metadata
        via_projection = project_failure_metadata(_QUERY, failure.to_record())

        assert via_bind == via_projection

    def test_the_two_builtins_agree_on_the_border_key(self) -> None:
        """A one-candidate fetch and a many-candidate search key the same way.

        Not "both are dicts" -- both must place their payload under the SAME
        named key with the SAME top-level field names, which is what lets a
        consumer read either without asking which tool ran.
        """
        search_side = project_metadata(_QUERY, _candidate_set())
        fetch_side = project_metadata(
            "https://example.test/0",
            CandidateSet(
                candidates=(_candidate("only", "https://example.test/0"),),
                spend=Spend(calls=1),
            ),
        )

        assert set(search_side) == set(fetch_side) == {SEARCH_RESULTS_METADATA_KEY}
        assert set(search_side[SEARCH_RESULTS_METADATA_KEY]) == set(fetch_side[SEARCH_RESULTS_METADATA_KEY])


class TestTheseAssertionsCanFail:
    """Guard tests: prove the comparisons above are load-bearing.

    ``test_egress_independence.py`` shipped a first draft in which the real
    regression left every assertion green. These reproduce the two regressions
    this file claims to catch and confirm the comparison notices, so a reader
    does not have to take the direction of the assertions on trust.
    """

    def test_a_reshaped_face_is_caught(self) -> None:
        rendered = bind_candidate_set(_QUERY, _candidate_set())
        platform = _render_platform_face(rendered)

        second_shape = dict(platform)
        second_shape[SEARCH_RESULTS_METADATA_KEY] = {
            **second_shape[SEARCH_RESULTS_METADATA_KEY],
            "results": second_shape[SEARCH_RESULTS_METADATA_KEY]["candidates"],
        }

        assert platform[SEARCH_RESULTS_METADATA_KEY] != second_shape[SEARCH_RESULTS_METADATA_KEY]

    def test_a_truncated_face_is_caught(self) -> None:
        rendered = bind_candidate_set(_QUERY, _candidate_set())
        platform = _render_platform_face(rendered)

        truncated = {
            **platform[SEARCH_RESULTS_METADATA_KEY],
            "candidates": platform[SEARCH_RESULTS_METADATA_KEY]["candidates"][:1],
        }

        assert len(truncated["candidates"]) != len(platform[SEARCH_RESULTS_METADATA_KEY]["candidates"])


@pytest.mark.parametrize("flag", ["face_api", "face_mcp"])
def test_the_faces_this_file_pins_do_not_depend_on_the_reach_flags(flag: str) -> None:
    """Renderings do not consult the flags, which is why the split is honest.

    If a rendering ever DID branch on a face flag, "one contract, three faces"
    would stop being verifiable without turning reach on, and this file's
    premise would be wrong. Pinned so that change cannot land quietly.
    """
    from threetears.agent.tools.builtin import web_fetch, web_search
    from threetears.search import bind

    for module in (bind, web_search, web_fetch):
        source = (module.__file__ or "").strip()
        assert source
        with open(source, encoding="utf-8") as handle:
            body = handle.read()
        assert flag not in body, (
            f"{module.__name__} now reads {flag}. A rendering that branches on reach "
            "means the faces can disagree, and success check 14 can no longer be "
            "verified without turning the flag on -- re-read this file's docstring."
        )
