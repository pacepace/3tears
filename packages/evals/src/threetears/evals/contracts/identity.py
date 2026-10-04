"""Stable identity keys for eval observations — variant, measurement context, generated content.

Reporting has to answer "are these two numbers comparable?" without the operator
hand-carrying run ids.  That needs coordinates that are *keys*, not rendered
strings: a **variant** (the resolved contestant stack — everything that would
ship if this cell wins) and a **measurement context** (the conditions pinned
around the contest).  A cell is variant x context; one ``EvalResult`` is one
observation in it.

Three rules shape everything here.

**Identity is computed over RESOLVED config, never over the request that asked
for it.** An override that restates the system default names the same variant,
so it must produce the same key. That is why the variant hashes the merged
snapshot rather than the override dict, and why the context hashes the resolved
judge and simulator models rather than the launch parameters — a parameter of
``None`` meaning "role default" would compare equal across a change to that
default and assert a comparability that does not hold.

The corollary is that **how** a resolved value was arrived at is not part of any
key. A run that named the default explicitly and a run that inherited it were
measured under identical conditions; splitting them would assert a difference
that does not exist. That distinction is still worth having — it is what says
whether re-running today would pick the same model — so it is recorded beside
the pins as ``EvalRun.model_role_provenance`` and never hashed.

**A predicate change is a version bump, not a silent regrouping.**
:data:`IDENTITY_VERSION` is stored beside every key rather than only hashed into
it, so old keys stay queryably distinct instead of quietly merging into new
ones. Golden-vector tests pin the digests, so changing the hashed inputs breaks
a test and forces the bump to be deliberate.

**Keys are stamped once, at the moment the thing they describe comes into
existence, and never rewritten afterwards.** A run stamps its context key at
launch and a result stamps its variant key as it is built — those are the
deliberate writes, and they happen where every input is known and frozen.

**A key is stamped beside its PRE-IMAGE, never alone.** A digest says two
observations are the same thing and refuses to say what that thing was, so a
stored key on its own leaves a reader reconstructing the inputs by replaying the
current predicate — which is only valid while the predicate is frozen, and the
predicate above is designed to move. ``EvalRun.context_components`` is the
context key's pre-image and ``EvalRun.variant_levers`` is the variant
key's. Keeping them is what makes a version bump
cost only comparability, which is what it is for, instead of also erasing every
description of what was measured.

**Everything after that is a read.** A run its host assembled without the launch
carries no context stamp, and a run stamped under an older ``IDENTITY_VERSION``
carries one this build does not compose; both still have the inputs sitting on the
stored run, so a key is recoverable — but recovering it must never turn into a backfill. The functions
here return the identity as a *value* instead of assigning it onto the run or
result, which is what makes that structural rather than a rule someone has to
remember: nothing a read produces can be picked up by a later save of the
document. The value carries ``source`` so a reconstructed key is never mistaken
for a recorded one, and names the components it could not know rather than
papering over them. See :class:`DerivedContextIdentity`.

Two coordinates that distinguished nothing were found while the predicate still read the first
host's subject record field by field, and both were removed rather than disclosed: a graph-version
hash no production code ever wrote (retired at v15), and a system-prompt field that was always
empty yet read as though it carried the subject's prose, so two subjects differing only in that
prose merged (replaced at v2). Neither can recur in that form: since v8 the predicate reads no
subject field at all, and hashes only what a host resolves for the levers it registered.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.hashing import canonical_digest
from threetears.evals.contracts.host.sweepables import SweepableValue
from threetears.evals.contracts.models import ContextComponents

if TYPE_CHECKING:
    from threetears.evals.contracts.host.profile import HostProfile
    from threetears.evals.contracts.models import EvalRun

#: One counter covers both predicates, so a bump on either side re-derives the other's
#: keys — conservative in the safe direction, since re-deriving unchanged inputs
#: reproduces the same digest. What moves is the label, not the key.
#:
#: **How to read the entries before v8.** Until v8 the predicate read the first host's subject
#: record field by field, so those entries describe that record's fields; they are named here by
#: what each field WAS (the subject's prose, the model its background tool ran on) because the
#: record is the host's and the engine no longer reads it. From v8 on, the predicate hashes what a
#: host resolves for its own registered levers and state levels, and v13 to v19 record bumps that
#: the first host's registrations caused — a host growing a coordinate moves the pre-image exactly
#: as an engine change does.
#:
#: - **v2** — the variant predicate swapped an always-empty system-prompt field for the subject's
#:   four identity-prose fields. The context predicate did not change.
#: - **v3** — the ``roles`` component now composes the **resolved** judge instead of the
#:   judge as declared at launch. Under v2 a run that inherited its judge stored ``None``
#:   there, so every inheriting run hashed alike no matter which judge each actually used.
#:   The bump is what makes those keys re-derive: derivation now treats an absent judge as
#:   an unrecoverable role pin and drops the whole component, so such a run reads as
#:   *partial* rather than silently sitting in the same namespace as a complete one.
#: - **v4** — the ``catalog`` component now composes the run's frozen seeded findings — what a
#:   background tool was seeded to return — beside ``resolved_world_seed``. A seeded payload is
#:   part of what the candidate was handed, exactly as the seeded world is, so two runs differing
#:   only in what that tool returned were previously sharing a context key and being pooled as
#:   repetitions of one condition. Under v3 nothing could seed, so no stored key is wrong —
#:   the bump is what keeps the first seeded run from landing in a live run's namespace.
#: - **v5** — the ``roles`` component now composes the run's ``effective_judges`` beside
#:   the two role pins. ``judge_model`` names the run's PIN, but under the judge-model
#:   cascade a dim whose ``JudgeConfig`` names a model is scored by that instead — so two
#:   runs whose rubric dims were judged by DIFFERENT models hashed alike and were pooled as
#:   repetitions of one condition, which is the confound the roles component exists to
#:   prevent, one level down from the one v3 closed. Derivation treats absent
#:   ``effective_judges`` the way v3 treats an absent judge — the whole component drops and
#:   the run reads as *partial* — because a run predating the field genuinely cannot say
#:   whether its dims diverged. Attribution reconstructed after the fact (``derived``) is
#:   NOT eligible: it INFERS what scored each dim from stored records rather than recording
#:   it, so hashing it would put an inference in the namespace of measurements.
#: - **v6** — the ``roles`` component now composes the run's pinned ``judge_config_ids``
#:   beside its attribution. v5 closed the case of two runs whose dims were scored by
#:   different judge MODELS; this closes the one where the model is identical and the
#:   *instrument* is not — a different judge prompt, or the same prompt at a different
#:   temperature, is a different measurement, and two such runs were sharing a key and
#:   pooling as repetitions of one condition. That case stopped being hypothetical when a
#:   run gained the ability to pin a configuration set at launch, since selecting a set is
#:   precisely how an operator produces two runs identical but for their judges. Config
#:   ids are hashed rather than config content because configs are immutable-versioned —
#:   re-authoring one mints a new id — which also means a cosmetic re-authoring splits the
#:   namespace, the cost immutable versioning already charges wherever else it is read.
#:   A run that recorded no set drops the whole component and reads as *partial*, on v5's
#:   reasoning; note that an EMPTY set is a recording, not an absence (it says no scored
#:   dim carried a config), so the test is on ``is not None``, never on truthiness.
#: - **v7** — one audit, three changes, one bump.
#:   **(a)** The variant predicate gained the resolved model of the subject's background tool. Only
#:   the ``inherited`` tier was ever invisible — the chosen and subject-level tiers already rode
#:   ``resolved_tool_configs`` — so two runs that both inherited the system default for it,
#:   spanning a change to it, were identical on both keys and pooled as repetitions of one
#:   condition. It lands in the VARIANT key rather than the context key deliberately: that model
#:   is a lever, and a lever in the condition key would give a bake-off's arms different context
#:   keys and report a deliberate sweep as not-comparable. Putting it here also keeps all three
#:   tiers of one value in one place instead of hashing the same field into both keys.
#:   **(b)** ``case_basis`` gained the snapshot's subject id. Two DIFFERENT subjects could
#:   produce an identical context key whenever both carried no memory and no catalog — the shape
#:   of a fresh subject or a memory-stripped probe — and pool as one condition, while
#:   ``compare_two_runs`` withholds cross-subject composites as invalid measurements. The guard
#:   existed only at the grouping layer, so the key was not self-sufficient. The subject's display name is
#:   deliberately NOT hashed: a rename must not split a cohort.
#:   **(c)** ``k_runs`` was REMOVED from the context key, and ``ContextComponents.k`` with it.
#:   ``k`` is not a condition — it is how many repetitions were taken under one — so two runs at
#:   k=1 and k=3 were measured identically and differ only in precision. Keeping it made a k=1
#:   pilot badge ``context_differs`` against the k=3 run that followed, a caveat about nothing.
#:   Depth is disclosed by the surfaces that own it (``Iters/case``, ``scored_iterations_min``/
#:   ``max``, the completeness sentence), which is where a pass^k mixture belongs.
#:   **Read this before assuming (c) changed pooling: it did not.** ``context_key`` gates no
#:   grouping anywhere today — ``comparison_sets`` groups by ``(subject_id, template_id)`` and
#:   uses the key only to raise ``BADGE_CONTEXT_DIFFERS``, and the frontier pools by
#:   ``variant_key``. So (c) removes a false badge and nothing else. In particular it does NOT
#:   make three k=1 runs compose a pass^3 finding; ``compute_pass_k`` groups attempts by
#:   ``(model, eval_run_id, test_case_id)``, so cross-run depth cannot compose regardless of any
#:   key. That capability is unbuilt and unscoped — see the audit.
#: - **v8** — the variant predicate stopped naming its inputs. It hashed ten fields of the first
#:   host's subject snapshot plus a prompt-override map, so only a host with that subject shape
#:   could produce a key; it now hashes whatever the host resolved for its own registered levers,
#:   checked against that host's registry. Two changes ride in the same bump because both move
#:   what is hashed.
#:   **(a)** A subject's components enter as content hashes rather than as prose read off named
#:   fields, so the same predicate serves a host whose subjects share nothing with the first's.
#:   **(b)** The prompt-override lever is addressed by its RESOLVED TEXT rather than by the preset
#:   names that select it. Under v7 two runs naming the same preset across an edit to that
#:   preset's content produced one key over two different prompts — the live half of R9's defect,
#:   and a wrong merge rather than a wrong split, which is the direction that cannot be corrected
#:   downstream. Resolution happens at capture, where the registry is in hand; the key sees only
#:   the hash.
#:   No stored key survives this bump: the corpus it would have re-derived over was dropped in
#:   the same commit, deliberately, rather than migrated.
#: - **v9** — the context predicate gained a ``world`` component: which of the host's declared
#:   world dimensions this run SEEDED, and which its subject merely witnessed. The world a subject
#:   is placed in is stimulus, and stimulus belongs to the conditions a cell holds constant — so
#:   two runs that placed one world differently were sharing a key and pooling as repetitions of
#:   one condition while one of them had not controlled what the other did. It is the same
#:   distinction ``family-convergence.md`` §4.3 already draws between a declared apparatus and a
#:   witnessed one, one axis over: apparatus is the rig, this is the world.
#:   The component composes the PLACEMENTS and not the seeded values, deliberately: the values are
#:   already composed into ``catalog_hash`` as it then was (``resolved_world_seed`` since v4,
#:   the seeded findings with it — both under the ``seeded_world`` mapping that replaced them at
#:   v10), and hashing them again here would split nothing further while making the reader of a
#:   badge unable to say which of two components moved. A run that recorded no placements drops the
#:   whole component and reads as *partial*, on v5's reasoning — a run predating the record
#:   genuinely cannot say what it placed, and an empty MAP is a recording rather than an absence
#:   (it says this host's world holds nothing this run could place), so the test is on ``is not
#:   None`` and never on truthiness.
#: - **v10** — the context predicate stopped naming one host tool in its signature.
#:   ``resolved_world_seed`` and the seeded findings were two parameters for one concept —
#:   what a run froze into the world before the subject's first turn — and the second was the first
#:   host's tool name sitting in the pre-image of every context key this engine has ever minted.
#:   They are now one ``seeded_world`` mapping, keyed by carrier, composed by the caller. What is
#:   hashed is the same information: the values still entered ``catalog_hash``, sorted, so no
#:   run's condition is described more loosely than before. The keys around them moved, which is
#:   why this is a bump and not a rename.
#:   It is deliberately NOT a move into the ``world`` component, which is where the design note
#:   that asked for this put it: ``world`` drops whole when a run recorded no placements, so
#:   seeded values living there would stop being hashed for every run predating that record —
#:   two runs that faced different frozen payloads pooling as repetitions of one condition,
#:   which is a wrong merge and uncorrectable downstream.
#:   No stored key survives this bump either, and for the same reason v8's did not: a bump drops
#:   and rebuilds the corpus rather than migrating it, in every deployment — production holds
#:   eval rows and they are expendable on the same terms. So landing a bump before a deploy that
#:   would drop the corpus anyway saves exactly one drop-and-rebuild: the ordering is a
#:   convenience and never a constraint on when a predicate may move.
#: - **v11** — the context predicate stops being shaped like one host's subject, and one condition
#:   it never hashed joins it. Four changes ride one bump because each moves a context key's
#:   pre-image, and taking them separately would split one corpus four times over.
#:   **(a)** The variant predicate's prompt-override component stops dropping empty resolved
#:   bodies. A slot that resolved to ``""`` was filtered out of the digest, so overlaying a slot
#:   with nothing hashed identically to never having overlaid it, and a run whose every body was
#:   empty produced no hash at all — which that lever declares as "the run predates the capture".
#:   Both are wrong merges: an operator who substituted an empty prompt did something to the
#:   candidate, and a run that recorded doing it is not a run nobody recorded.
#:   **(b)** The component carrying the storage scope is named ``scope``, not the first host's word
#:   for its storage partition. :func:`compute_context_key` digests ``model_dump()``, so a
#:   component's field NAME sits in the pre-image of every context key this engine has ever
#:   minted. Carrying the identical value under a different name is therefore a new predicate
#:   rather than a rename: every stored key stops matching, which is what a bump is for.
#:   **(c)** The context predicate stops reading the first host's subject record. It composed a
#:   subject's carried memory, its catalogue and its record id off a host snapshot the engine had to
#:   reach into the host adapter to recover, so only that host could produce a key. It now
#:   composes what the host supplies: ``subject_state``, the snapshot's own map of what the subject
#:   CARRIED IN, kept as the map rather than digested so a badge can name which of the host's state
#:   names moved. ``memory_hash`` and ``catalog_hash`` are gone — the first is that map, the host
#:   half of the second is one name inside it, and the frozen world the second also carried
#:   becomes its own ``seeded_world`` component, because the two halves had different availability
#:   and sharing a component hid that: a catalogue is a host value a capture can fail to record,
#:   while a run that froze nothing passes an empty value per carrier and is always composable. It
#:   is deliberately NOT a move into ``world``, on v10's own reasoning, which survives unchanged.
#:   ``case_basis`` reads the engine's own ``subject_snapshot.subject_id``, which is the same value
#:   the host record's own id was, so v7's reasoning is served identically by a field the engine
#:   already requires and refuses to let be blank. An absent state map drops the component and
#:   reads *partial*, on v5's rule; an empty MAP is a recording — this subject carries nothing
#:   outside its variant components — so the test is ``is not None`` and never truthiness.
#:   **(d)** The context predicate gains a ``tool_permissions`` component over the allowlist frozen
#:   onto the run at launch. The allowlist is frozen there BECAUSE the template is mutable, so
#:   ``template_id`` inside ``case_basis`` cannot stand for it: two runs on one template spanning an
#:   edit to its allowlist shared a key while one candidate could reach for capabilities the other
#:   could not, and pooled as repetitions of one condition. Unlike every other absence in this
#:   signature, no allowlist means UNBOUNDED — a recorded level, hashed as one — so the component is
#:   always composable and two unbounded runs hash equal.
#: - **v12** — the variant predicate's lever NAMES change, because the engine stopped holding two
#:   vocabularies for one knob. :func:`compute_variant_key` digests the resolved map, so a lever's
#:   name sits in the pre-image of every variant key this engine has minted; carrying the identical
#:   level under a different name is a new predicate rather than a rename.
#:   **(a)** The candidate model's lever is ``model``, not ``models``. The core declared one name
#:   while every reporting lens — the coverage map, ``RunSummary.config``, a finding's scope —
#:   used the other, so a campaign that DECLARED the candidate model as its swept axis got a
#:   coverage row nothing could match the declaration to, and the completeness check read the
#:   answer as silent about an axis it had answered.
#:   **(b)** The background tool's model lever took its dotted spelling (the tool's name, then the
#:   field), for the same reason from the other side: the dotted spelling is the one three lenses
#:   and the generator prompt already used, and it is now the declared name rather than a carrier
#:   path a lens flattened for itself.
#:   The levels are unchanged — the same candidate model, the same resolved background-tool model —
#:   so nothing about what a variant IS moved. Only what it is CALLED did, which is exactly the
#:   class of change this counter exists to make visible rather than silent.
#: - **NOT bumped when the roles component's MEMBERSHIP became host-scoped**.
#:   ``derive_context_identity`` now drops a roles input the host DECLARED it
#:   does not have, so the pre-image can be a proper subset or empty — which is a change in how a
#:   component is composed, and the rule above would ordinarily bump for it. It does not, because a
#:   host that declares no inapplicability omits nothing: the pre-image is the same four keys in the
#:   same shape, every key the first host had stored still derives to itself, and only a declaring
#:   host's keys move — of which none has stored data. Pinned by
#:   ``test_the_roles_pre_image_is_unchanged_for_a_host_that_declares_no_inapplicability``, so the
#:   claim is checked rather than asserted. Recorded here because this ledger is the authority a
#:   later reader consults before deciding their own bump, and a rule with a silent exception is a
#:   rule the next person follows differently.
#: - **v13** — the first host's variant map gains a coordinate: the subject's RESOLVED output-token
#:   cap for its own model call. The cap was a hardcoded constant on the client constructor at first,
#:   so no observation had ever varied on it and its absence from the key was correct;
#:   the day it became configurable — system tier, subject tier, and a run-scoped overlay — two runs
#:   at different caps became two contestants with one key, which is a wrong MERGE, the direction
#:   nothing downstream undoes. The coordinate is the resolved value off the snapshot, not the
#:   request: an arm that asked for the default cap and an arm that inherited it ran the same
#:   candidate. A snapshot written before the field carries an explicit not-recorded level rather
#:   than today's config — the candidate ran under some cap and nobody recorded which, and
#:   re-reading the system tier now would name one that may not have been in force. Every stored
#:   variant key re-derives (the map gained a name, and the name is in the pre-image); no live eval
#:   data is valued, and the context predicate is untouched. The cap's bisection reader is declared
#:   beside it in the host's registrations.
#: - **v14** — the first host's variant map gains a second subject coordinate: the RESOLVED
#:   reasoning effort the subject's own model call ran at. Its absence from the key was correct
#:   while no observation could vary on it: the subject's client sent no ``reasoning`` parameter at
#:   all at first. The day it became configurable — system tier, subject tier, and a
#:   run-scoped overlay — two runs at different efforts became two contestants with one key, the
#:   wrong MERGE that nothing downstream undoes. It is a SEPARATE coordinate from v13's cap rather
#:   than a refinement of it: the provider expresses effort as a fraction of ``max_tokens``, so the
#:   two vary independently and a run may move either. Three levels, kept apart on purpose — a named
#:   effort; a recorded run that sent no reasoning parameter (its own sentinel, so it cannot arrive
#:   looking like an absence); and a snapshot written before the field, which carries an explicit
#:   not-recorded level rather than today's config. Every stored variant key re-derives; no live
#:   eval data is valued, and the context predicate and its golden vector are untouched.
#: - **v15** — the first host's variant map LOSES a coordinate: a prompt-graph version hash read
#:   off its subject snapshot. Removing a factor is a bump on this ledger's own rule, and it is the
#:   one direction the rule exists for: every stored map named the coordinate, so no stored key
#:   derives to itself under the new predicate.
#:   **It is a retirement and not a deferral, which is the opposite of what the note above used
#:   to promise.** The field was kept in the hashed tuple against the day a producer was wired;
#:   that day was never coming. The host's prompt graph was an AUTHORING and visualisation surface
#:   that only its admin layer imported, while a subject's system prompt was assembled from
#:   templates through a registry — so no observation the host could produce was rendered by a
#:   prompt graph, and the only hash available to stamp mirrored the DEFAULT template rather than
#:   the resolved one a subject ran under. Writing it would have put a fabricated apparatus
#:   condition in the pre-image of every key.
#:   **The cost of leaving it declared was real and observed**, which is what forced the
#:   decision rather than another deferral: a coordinate that is always ``None`` distinguishes
#:   nothing, and its apparatus twin (``EvalResult.graph_version_hash``, retired in the same
#:   commit) made every analysis over this corpus hedge its findings on a prompt-graph version
#:   nobody recorded — a caveat no campaign could ever discharge by running. What an observation
#:   genuinely varies on here is the resolved prompt TEXT, which this predicate has hashed since
#:   v11(a). Nothing that could distinguish two variants stops being hashed.
#:   Losing a coordinate is a wrong-MERGE risk in general, and it is not one here: the coordinate
#:   held one level for every observation ever taken, so no two variants were ever kept apart by
#:   it. Every stored variant key re-derives; no live eval data is valued, and the context
#:   predicate and its golden vector are untouched.
#: - **v16** — the first host's variant map gains a coordinate: the subject's own statement of what
#:   it is ABOUT, its domain. Its absence was correct while nothing read it; from the commit that
#:   interpolated it into the measured classifier's resolved prompt, two subjects differing only
#:   here sent DIFFERENT prompts to the classifier while deriving the same subject key — a wrong
#:   MERGE, the direction nothing downstream undoes. The host's classifier launch made it reachable
#:   rather than theoretical: it captured the resolved prompt for the candidate but built the run's
#:   identity from the subject snapshot, so the two halves of one run disagreed about whether the
#:   arms differed, and two such runs had to be kept apart by hand. Empty is a level like any
#:   other — a subject that declares no domain runs a classifier that infers one, which is a
#:   different prompt from one that is told. Every stored variant key re-derives (the map gained a
#:   name, and the name is in the pre-image); no live eval data is valued, and the context
#:   predicate and its golden vector are untouched. **What this bump does NOT cover**, stated
#:   because the next reader will assume it does: a subject's stored CLASSIFIER PROMPT OVERRIDE was
#:   still absent from the snapshot and from the components, so switching a subject between a
#:   forked prompt and the shared default moved no key. That is the same class and was tracked
#:   separately rather than folded in here, because the override
#:   is a whole prompt rather than a field and capturing it raises a retention question this
#:   coordinate does not.
#: - **v17** — the first host's variant map gains a coordinate: WHETHER the subject has an
#:   optional speech capability (named apart from ``style``; WHICH capability configuration is not
#:   hashed, since the candidate is offered the same with any).
#:   It decides what the candidate is OFFERED, not only how it sounds: the host grants the matching
#:   tools, and the state sections and perceptions that go with them, only to a subject that has
#:   the capability. Before this, the subject snapshot did not carry it at all, so every eval
#:   subject ran without it whatever its live self had, and a capable subject and an incapable one
#:   derived the same subject key. That was invisible while the snapshot dropped it everywhere; once
#:   it is carried (the same change), the two send different tool catalogs to the candidate and
#:   would be a wrong MERGE without this coordinate. Empty is a level (not capable), and a snapshot
#:   predating the field reads as not recorded.
#:   The same bump adds the subject's persisted settings: the snapshot had dropped every persisted
#:   key it did not name, so an eval subject ran at the code defaults for tool rounds and attention
#:   per turn, the ground-rules and tool-reaction presets, role, attributes, timezone and memory
#:   windows — all of which shape the turn. They travel now, in the persisted form, and are one
#:   coordinate. The accumulated turn state that travelled with them (todos, turn summaries, recent
#:   spoken lines, turn counter) is a state level of its own, ``turn_state``, not a variant
#:   coordinate — and because state levels are CONTEXT inputs, this bump moves the context key as
#:   well. Every stored variant and context key re-derives; no live eval data is valued.
#: - **v18** — a subject state level stops composing over one of the entries it hashed: the host's
#:   ``turn_state`` no longer carries what the subject last said aloud, which became a world
#:   dimension the run's world seed states instead — so the same fact is still hashed into the
#:   context key, once, through the ``world_seed`` component, where a run SETS it rather than
#:   inheriting whatever the captured subject last did. The predicate's shape is unchanged and both
#:   golden vectors hold, since neither fixture carries that entry; what moves is the pre-image of a
#:   real subject's ``turn_state`` level, so a stored run whose subject had spoken derives a
#:   different context key under v18.
#: - **v19** — the subject's carried state stops composing over its recent turns and turn number: what it
#:   remembers of its last turns, the number its next turn runs as and the time it reads became host world dimensions
#:   a run's world seed states (each defaulted into the resolved seed when a template states none), so they are
#:   hashed once, through the ``world_seed`` component, where a run SETS them rather than inheriting the live
#:   subject's last hour. The predicate's shape is unchanged; what moves is the pre-image of a real subject's
#:   ``carried_memory`` level (no recent turns in it) and every resolved world seed (a session namespace in it),
#:   so every stored run derives a different context key under v19.
#: - **v20** — a kind owns what a launch may turn and what a judge is asked. **(a)** The four
#:   named overlay fields a run carried are gone: a launch's overlays are one map the kind's own model
#:   validates, frozen on the run whole, defaults included (``EvalRun.overlays``), and a host reads its
#:   overlay levers off that map through the levers its kind contract derives. The variant predicate's
#:   arithmetic is unchanged; what moves is every overlay lever's pre-image, which is now the resolved
#:   model rather than the request — a launch restating a default and one naming nothing hash alike.
#:   **(b)** The ``roles`` component's per-dim attribution names only the dims a kind's judge is asked
#:   (``scored_dim_ids``): a document run no longer records a judge for the transcript and outcome
#:   axes no judge scores it on, so every document run's context key moves. **(c)** The
#:   ``seeded_world`` component composes the run's frozen ``kind_spec`` — what its template stated
#:   for the kind, re-validated and frozen at launch (``EvalRun.kind_spec``) — under the key
#:   ``kind_spec``, in place of the seeded background-tool findings a template could carry: those
#:   leave the template with the rest of one host's kind content, and a kind that seeds findings
#:   states them in its spec. **(d)** An unjudged run's ``roles`` component composes without its judge
#:   inputs rather than dropping whole: the runner refuses to execute a judged run that names no judge,
#:   so ``judge_model=None`` records that the run was not judged — a level, where it used to be read as
#:   an unrecorded pin and left every unjudged run on a judge-capable host ``partial``. **(e)** The
#:   ``cassette`` component composes the corpus a replay serves (``EvalRun.cassette_corpus_id``) in
#:   place of a ``cassette_version`` label nothing bound: two replays of different corpora were served
#:   different tool output, and a capture run's own corpus is not a condition its live tools ran under.
#:   One bump for (a)-(e), since nothing between them was released.
#: - **v21** — the variant map gains the candidate KIND, and the engine composes the map itself. A run's
#:   kind is now a core lever (``candidate_kind``) resolved for every run, so two runs of different kinds
#:   are two variants however alike everything else is: before, runs of two kinds at one model with no
#:   overlays (or overlays all at ``None``) derived one key and pooled into one arm — a wrong MERGE. The
#:   engine also resolves the candidate model and every kind contract's levers itself (a host's reader
#:   returns only its own levers), and a kind contract's "not a run of this kind" level is addressed so it
#:   can equal no value a field holds, where it used to hash as ``None``. Every stored variant key
#:   re-derives (each map gained a name). The context predicate composes v20's inputs, with the
#:   ``seeded_world`` component's seed now keyed ``world_seed`` (the engine's one word for the seeded
#:   world, beside ``WorldRegistry``), so every context key re-derives too. One bump for both, since
#:   nothing between them was released.
IDENTITY_VERSION: int = 21
"""Version of the key-derivation predicate below.

Bump whenever the hashed inputs of *any* key change — adding a factor, removing
one, or changing how a component is composed. Keys carrying different versions
are never comparable, which is the whole point: a predicate that grows silently
would regroup history behind the operator's back.
"""


# =============================================================================
# Generated-artifact identity
# =============================================================================


def compute_content_hash(variation_params: dict[str, str] | None) -> str | None:
    """Digest a generated test case's content so its identity is pinned.

    A generated artifact that is not content-addressed lets "the same cell"
    silently differ between runs, which confounds every downstream comparison.
    ``variation_params`` is the whole of a test case's generated content — the
    opening utterance is rendered from the template at run time, not stored.

    Args:
        variation_params: The case's frozen axis values.

    Returns:
        The digest, or ``None`` when there is no generated content to pin. An
        empty mapping returns ``None`` rather than the digest of ``{}``: such a
        case was never generated from axes at all, so its identity is its
        template plus its id, and giving it a content hash would invent a
        provenance it does not have.
    """
    if not variation_params:
        return None
    return canonical_digest(variation_params)


# =============================================================================
# Variant identity — the resolved contestant stack
# =============================================================================


class LeverCoordinateError(ValueError):
    """A host's per-observation lever map disagrees with its own registry, in either direction.

    Raised rather than tolerated because the alternative is the exact drift the sweepables module
    was written to make unexpressible: two lists of the same thing, authored separately, where one
    quietly grows or loses an entry the other never hears about. The registry is the authority and
    the map is checked against it **both ways** — an axis the registry never declared must not
    enter a key, and a lever the registry DOES declare must not silently drop out of one.

    The second direction is the one that bites hardest, and it is why this is not one-way: a
    forgotten reader removes a coordinate with no ``IDENTITY_VERSION`` movement, so two genuinely
    different variants come to share a key. That is a wrong MERGE, and nothing downstream can
    undo it. A lever that legitimately carries no coordinate says so on its declaration
    (:attr:`~threetears.evals.contracts.host.sweepables.Sweepable.no_own_coordinate`), which is what makes the
    intended omission distinguishable from the forgotten one.
    """


def compute_variant_key(levers: Mapping[str, SweepableValue]) -> str:
    """Digest one observation's resolved contestant stack — everything that would ship if it wins.

    The whole predicate, and it names nothing: a sorted map from axis name to that axis's
    **content hash**, over every lever the host resolved for this observation. The names are the
    host's; the arithmetic is the engine's. What the axes mean, how many there are, and whether
    any of them is a subject is not something this function can be asked.

    Rendering is deliberately excluded. Two observations that swept the same content under
    different labels are the same variant, and hashing :attr:`SweepableValue.display` would split
    a cohort on a rename — the same reason a subject's label has never been part of a key.

    Args:
        levers: Axis name → the level this observation carried. Every value is already
            content-addressed by the host, which is what lets a host the engine cannot see into
            supply one at all (R9).

    Returns:
        A 64-character hex digest.
    """
    return canonical_digest({name: value.content_hash for name, value in levers.items()})


# =============================================================================
# Measurement-context identity
# =============================================================================


#: What the ``tool_permissions`` component records for a run that bound nothing. A LEVEL rather
#: than an absence, and the opposite of every other ``None`` reaching this predicate: a run with no
#: allowlist really did let the candidate reach for everything its snapshot carried, so two such
#: runs must hash equal instead of reading as two runs nobody recorded.
#:
#: A host's own confound scan renders the same state in the same words, and the two are deliberately
#: separate literals rather than one shared constant: this one is a hash pre-image, that one is a
#: display, and coupling them would make a cosmetic edit to a reader's wording re-key the corpus.
UNBOUNDED_TOOL_PERMISSIONS = "(unbounded)"


class StateCoordinateError(ValueError):
    """A host's subject-state map names something its own registry does not declare as apparatus.

    Raised for the reason :class:`LeverCoordinateError` is raised in its first direction: a name
    entering a key that no reader of the registry can see is a coordinate nobody can place on an
    axis, and the registry is the authority for what an input is.

    **The reverse direction is deliberately not checked, and the asymmetry is the point.** For the
    variant key, a declared lever missing from the map is unambiguously a forgotten reader, so it
    raises. Here it is ambiguous — a subject that honestly carried nothing under the name reads
    exactly the same way — and a check that cannot tell a forgotten writer from an honest absence
    fires on the wrong one. So that direction is held structurally instead: a host writes its state map and declares its
    sweepables from one table, and a name cannot exist on one side and not the other because there
    is only one side.
    """


def compute_context_components(
    *,
    subject_id: str,
    subject_state: Mapping[str, SweepableValue] | None,
    seeded_world: Mapping[str, Any],
    template_id: str | None,
    test_case_ids: list[str],
    judge_model: str | None,
    simulator_model: str | None,
    effective_judges: dict[str, str] | None,
    judge_config_ids: dict[str, str] | None,
    cassette_mode: str,
    cassette_corpus_id: str | None,
    tools_allowed: list[str] | None,
    scope_id: str,
    world_placements: Mapping[str, str] | None,
    include_roles: bool = True,
    omitted_roles: Collection[str] = (),
) -> ContextComponents:
    """Compute each pinned condition separately so a mismatch names itself.

    Args:
        subject_id: Who was measured — the engine's own pooling boundary, which
            :class:`~threetears.evals.contracts.host.subject.SubjectSnapshot` requires and refuses to let be
            blank. In the key because two subjects carrying nothing otherwise produce an identical
            context key and pool as one condition, while a cross-subject composite is not a valid
            measurement. The subject's LABEL is deliberately not here: a rename is a label change
            and must not split a cohort.
        subject_state: What the subject carried INTO the run, keyed by the host's own names and
            already content-addressed by it. The symmetric sibling of the variant key's component
            map: that one is what the subject *is*, this one is what it brought, and the split is
            the one every host draws, since state hashed into the variant would mint a new variant
            every time the subject remembered something. Stored as the MAP rather than digested,
            because a comparison surface exists to badge WHICH condition moved and one hash over
            the host's names cannot. ``{}`` is a recording — this subject carries nothing outside
            its variant components — and ``None`` is an absence, which drops the component.
        seeded_world: What this run froze before the subject's first turn — the world it
            seeded, keyed by carrier, and the spec its kind validated. Every entry is part of what the subject faced, so part of the
            condition its results describe; a run that froze nothing passes an empty value per
            carrier rather than omitting the argument, because "seeded nothing" is a condition
            and not an absence. That is also why this is its own component rather than riding
            beside a host value that a capture can fail to record: the two halves had different
            availability, and sharing a component hid it. One mapping rather than a parameter per
            carrier: a second consumer's async tool seeds a carrier this engine has never heard
            of, and a named parameter for each would make the predicate grow a signature entry
            per host tool.
        template_id: The run's template, or ``None`` for an ad-hoc run.
        test_case_ids: The frozen case set. Sorted before hashing so the order
            they were resolved in cannot split an otherwise identical basis.
        judge_model: The RESOLVED run-level judge — the model any dimension
            without its own ``JudgeConfig`` is scored by. Resolved for the same
            reason the simulator is: a declared value's ``None`` means "role
            default", so two runs that both inherited would compare equal across
            a change to that default. This names the run's PIN; which model
            each dim was requested from is ``effective_judges``.
        effective_judges: Per-dim attribution, ``{dim_id: resolved_model}`` — the
            model each dim was REQUESTED from once the cascade (role default <
            run pin < per-dim ``JudgeConfig.model``) is applied. A floating alias
            can be served by a different model; ``RubricScore.served_model``
            records the one that scored, and this key does not. Folded in from
            v5, because the pin alone let two runs whose rubric dims were judged
            by different models share a key and pool as one condition. Only
            ``recorded`` attribution is eligible: a reconstruction infers what
            each dim was requested from, from stored records, so hashing it would put an
            inference in the namespace of measurements. ``None`` for a run that recorded none, which is a
            reason to drop the component, not to hash around it.
        judge_config_ids: The ``JudgeConfig`` set the run pinned at launch,
            ``{dim_id: config_id}`` over the dims that carry one. Folded in from
            v6, because ``effective_judges`` names the model each dim was
            requested from and says nothing about the prompt it was scored with — so two runs
            differing only in their judge configuration hashed alike, which is the
            difference a judge A/B exists to create. ``{}`` is a value here, not an
            absence: it records that no scored dim carried a config. ``None`` is
            the absence, and drops the component.
        simulator_model: The RESOLVED simulator model.
        cassette_mode: Whether tool output was live, captured, or replayed.
        cassette_corpus_id: The corpus a replay run serves — ``None`` for a capture run, whose tools ran
            live, and for a run with cassettes off.
        tools_allowed: The tool allowlist frozen onto the run at launch, or ``None`` for a run that
            bound nothing. It is frozen on the run BECAUSE the template is mutable, so
            ``template_id`` in the case basis cannot stand for it: two runs on one template
            spanning an edit to its allowlist would otherwise share a key while one candidate
            could reach for capabilities the other could not — one name over two contents, which
            is a wrong merge. ``None`` here is the one absence in this signature that is a
            recorded LEVEL rather than a gap, so this component is always composed and two
            unbounded runs hash equal.
        scope_id: The storage scope the run executed in, carried into the ``scope``
            component verbatim. Uninterpreted but **not** unused: it is part of the
            context key's pre-image, so two scopes do not pool as one condition. The
            analysis layer's scope really is inert metadata; this one is not, and the
            two are the same value wearing different jobs.
        world_placements: What the run did with each dimension the host's world declares —
            seeded-and-perceived, seeded-and-unperceived, perceived-and-unseeded, or neither.
            Folded in from v9, because the world a subject is placed in is stimulus and two runs
            that placed it differently are not repetitions of one condition. ``{}`` is a value
            here, not an absence: it records that this host's world held nothing this run could
            place. ``None`` is the absence — a run whose writer placed nothing — and drops the component.
        omitted_roles: The declared apparatus dimensions this host does not HAVE, by
            :meth:`~threetears.evals.contracts.host.profile.HostProfile.omits_apparatus` over what the run
            recorded. Their inputs leave the roles pre-image entirely rather than hashing as
            ``None``, because a host that grades with code has no judge pin — it did not fail to
            record one. A host omitting every role input still composes the component, over an
            empty pre-image: "this host pins no model roles" is a real and stable condition that
            all its runs share, and it is distinct in shape from the absent component
            ``include_roles=False`` produces.
        include_roles: ``False`` omits the roles component entirely, for a run
            missing any of its inputs — a simulator model its writer never recorded,
            attribution that was never recorded or was only reconstructed, or a
            judged run with no configuration set recorded. A roles digest built from whichever inputs
            survived would be a *different* component wearing the same name, so
            the honest move is to leave it absent.

    Returns:
        The populated components. Time is deliberately absent — it is a
        recorded coordinate for bucketing, and hashing it would make every key
        unique.

    Note:
        ``subject_state`` needs no ``include_`` gate beside it, and that is not an omission.
        ``include_roles`` exists because a caller declares whether the roles component is
        composable separately from the composer that builds it, so the two can disagree; the state
        map's own nullity IS the declaration, so there is nothing for it to disagree with.
    """
    return ContextComponents(
        # The MAP, reduced to its identities. Not a digest of it: `ContextComponents` exists so a
        # surface can badge which condition moved, and one hash over the host's names would answer
        # "the subject's state moved" where this answers which of them did, in the host's own
        # vocabulary. The same argument that keeps `scope` raw one field down.
        subject_state=None
        if subject_state is None
        else {name: value.content_hash for name, value in subject_state.items()},
        # Sorted for the reason ``effective_judges`` is: the order the caller happened to assemble
        # the carriers in is incidental and must not split an otherwise identical world.
        seeded_world=canonical_digest(dict(sorted(seeded_world.items()))),
        case_basis=canonical_digest(
            {
                # WHO was measured, not just what against. Without it two different subjects
                # produced an identical context key whenever both carried nothing — a fresh
                # subject, or a stripped-snapshot probe — and pooled as repetitions of one
                # condition, while ``compare_two_runs`` withholds cross-subject composites
                # precisely because rubric dims derive from each subject's own description. The
                # guard existed only in ``comparison_sets``' grouping, so the key was not
                # self-sufficient. The subject's LABEL is deliberately absent: a rename is a label
                # change, and splitting a cohort on it would be the same error ``bisect_runs``
                # avoids by reporting the pair.
                "subject_id": subject_id,
                "template_id": template_id,
                "test_case_ids": sorted(test_case_ids),
            }
        ),
        roles=(
            canonical_digest(
                _roles_payload(judge_model, simulator_model, effective_judges, judge_config_ids, omitted_roles)
            )
            if include_roles
            else None
        ),
        cassette=canonical_digest({"cassette_mode": cassette_mode, "cassette_corpus_id": cassette_corpus_id}),
        # Sorted because the order a template happened to list its tools in is incidental, and
        # never ``None``: an unbounded run is a run that recorded letting the candidate reach for
        # everything, so it hashes the sentinel and pools with every other unbounded run.
        tool_permissions=canonical_digest(
            {"tools_allowed": UNBOUNDED_TOOL_PERMISSIONS if tools_allowed is None else sorted(tools_allowed)}
        ),
        # Sorted for the reason ``effective_judges`` is: the order dimensions were declared in is
        # the registry's business and must not split an otherwise identical world.
        world=None if world_placements is None else canonical_digest(dict(sorted(world_placements.items()))),
        scope=scope_id,
    )


def _roles_payload(
    judge_model: str | None,
    simulator_model: str | None,
    effective_judges: dict[str, str] | None,
    judge_config_ids: dict[str, str] | None,
    omitted: Collection[str] = (),
) -> dict[str, Any]:
    """Compose the roles pre-image, refusing to build a partial one.

    ``include_roles`` is the single gate on whether this component exists, and the
    caller sets it from whether ALL of its inputs are recorded. Reaching here with
    a missing input therefore means the two disagree — a programming error, not an
    unrecorded run — so it raises rather than hashing a component that would wear the
    same name as a complete one while meaning something narrower. That is the
    failure v3 was created to stop being silent.

    Args:
        judge_model: The resolved run-level pin.
        simulator_model: The resolved simulator model.
        effective_judges: Per-dim attribution, ``recorded`` provenance only.
        judge_config_ids: The pinned config set. An empty dict is a value and
            hashes; only ``None`` is absent.
        omitted: The declared dimensions this host does not have. Their entries leave the
            pre-image rather than hashing as ``None``, and are exempt from the refusal above:
            an input a host never had is not an input it failed to record, and the two must not
            produce the same digest. Named by DIMENSION, since that is what a host declares; the
            attribution map is the value of ``judge_dim_divergence``, whose declared reader is
            the one that reads it.

    Returns:
        The mapping to digest — ``{}`` when this host has none of the roles inputs, which is a
        recording rather than an absence and pools that host's runs with each other.

    Raises:
        ValueError: An input this host HAS is absent.
    """
    # ONE enumeration of the four, carrying the dimension a host declares, the pre-image key, and
    # the value. Building the refusal from one list and the payload from another is how the two
    # come to disagree about which inputs the component covers — and this component's whole
    # contract is that it is composed from ALL of its inputs or from none.
    #
    # The maps are sorted here so the order dims were resolved in cannot split an otherwise
    # identical basis — the same reason ``test_case_ids`` is sorted into the case basis.
    declared = (
        ("judge_model", "judge_model", judge_model),
        ("simulator_model", "simulator_model", simulator_model),
        (
            "judge_dim_divergence",
            "effective_judges",
            None if effective_judges is None else dict(sorted(effective_judges.items())),
        ),
        (
            "judge_config_ids",
            "judge_config_ids",
            None if judge_config_ids is None else dict(sorted(judge_config_ids.items())),
        ),
    )
    present: dict[str, Any] = {key: value for dimension, key, value in declared if dimension not in omitted}
    if absent := sorted(key for key, value in present.items() if value is None):
        raise ValueError(
            f"roles component requested with an unrecorded input ({', '.join(absent)} absent"
            + (f"; omitted as inapplicable: {', '.join(sorted(omitted))}" if omitted else "")
            + ") — pass include_roles=False so the component is dropped instead"
        )
    return present


#: Role inputs that describe ANOTHER role input rather than standing on their own, as
#: ``{dependent: the input it is a fact about}``.
#:
#: Per-dim attribution is divergence FROM a judge pin and a config set is the configuration OF one,
#: so a host with no judge model has neither — they are facts about a judge that does not exist.
#: Without this they read as absences the host failed to fill, and a code-grading host could never
#: compose a roles component however many dimensions it declared inapplicable: ``judge_config_ids``
#: alone kept the predicate False on every run. **Propagation still loses to the data** — a
#: dependent the run actually recorded stays in the pre-image, on the same rule
#: ``omits_apparatus`` applies to a declaration the runs refute.
#:
#: **It is not a second applicability map.** Nothing here can declare anything inapplicable; it
#: only propagates an omission the HOST declared and the runs did not refute, which is why it names
#: dependencies rather than dimensions. ``judge_config_ids`` is also the case that cannot be
#: recovered any other way: its declared reader answers the OBSERVED set, which for a host with no
#: judge legitimately records ``NO_JUDGE_CONFIGS`` — so declaring it inapplicable would be a claim
#: that host's own data refutes, logged as a contradiction on every call.
_ROLE_INPUT_DEPENDENCIES: Mapping[str, str] = {
    "judge_dim_divergence": "judge_model",
    "judge_config_ids": "judge_model",
}


#: Components composed from ALL of their inputs or none of them, in the order a reader meets them.
#:
#: Each is dropped rather than built from whichever inputs survived, because a digest over a subset
#: would be a *different* component wearing the same name — so ``None`` on one of these means "this
#: run could not say", never "this run's value was empty". Every other component is either always
#: computable or carries its own emptiness as a real value.
_ALL_OR_NONE_COMPONENTS: tuple[str, ...] = ("subject_state", "roles", "world")


def missing_context_components(components: ContextComponents) -> list[str]:
    """Which all-or-none components this context could not compose.

    One predicate for both branches of :func:`resolve_context_identity`. They answered the same
    question separately once, and the second time a component joined the set only one of them
    learned about it — so a stamped key read as complete where the identical derived key read as
    partial, and the badge an operator sees is built from whichever branch they happened to hit.

    Args:
        components: The components, stamped or freshly computed.

    Returns:
        The absent component names, in :data:`_ALL_OR_NONE_COMPONENTS` order. Empty means the
        context is complete.
    """
    return [name for name in _ALL_OR_NONE_COMPONENTS if getattr(components, name) is None]


def compute_context_key(components: ContextComponents) -> str:
    """Compose the component hashes into the single groupable context key.

    Components that are ``None`` are dropped from the pre-image rather than
    encoded as null, so a partial context (one whose roles could not be known)
    can never collide with a complete one that happens to match on everything
    else — the two have different shapes, not merely different values.

    Args:
        components: The separately-computed pieces.

    Returns:
        A 64-character hex digest.
    """
    present = {k: v for k, v in components.model_dump(mode="json").items() if v is not None}
    return canonical_digest(present)


# =============================================================================
# Read-time derivation for runs carrying no current stamp
# =============================================================================


class DerivedContextIdentity(EvalBaseModel):
    """A run's context key as a value, never written back onto the run.

    Returned rather than assigned on purpose. Filling the run's own fields
    would make a key that was merely inferred indistinguishable from one
    recorded at launch, and would let any later save of that document persist
    it — turning a read into a backfill.

    ``source`` is the field a reader must not skip. A ``stamped`` key was
    recorded when the run launched and is trustworthy about conditions as they
    were; a ``derived`` key was reconstructed later by replaying the current
    predicate over whatever happens to still be stored. They are both keys, and
    they are not equally good evidence.
    """

    context_key: str
    context_components: ContextComponents
    identity_version: int
    source: Literal["stamped", "derived"]
    partial: bool
    missing_components: list[str] = []


class DerivedVariantIdentity(EvalBaseModel):
    """A computed variant key plus what it was computable from.

    Unlike :class:`DerivedContextIdentity`, this type carries the key on BOTH paths: the
    run loop computes it once for the run's candidate and stamps it onto every result, and a reader
    answers the same question from the run when it needs the lever map behind a stamped key. There is
    no partial variant identity: the engine resolves the candidate model, the candidate kind and every
    kind contract's levers itself, and a host lever nobody resolves is refused rather than left out.

    **The map reaches this type two ways, and they are not equally good evidence.**
    :func:`derive_variant_identity` RESOLVES it through the host as it is today;
    :func:`resolve_variant_identity` prefers the one the run RECORDED when it was launched.
    The arithmetic over the map is identical either way — that is why one type serves both
    — but a resolved map describes the run through this build's predicate and a recorded one
    describes it through the predicate it actually ran under. Prefer the recorded path
    wherever a run is in hand.
    """

    variant_key: str
    identity_version: int
    levers: dict[str, SweepableValue]
    """The resolved lever map the key was digested from — axis name to the level carried.

    Returned rather than discarded because a key alone cannot be placed on an axis. A
    digest says two observations are the same contestant stack and refuses to say what
    that stack WAS, so anything joining a variant to a declared level — an arm table, a
    pivot, a marginal — has to hold the map beside the key or invent one.
    """


def _refuse_undeclared_state(state: Mapping[str, SweepableValue] | None, profile: HostProfile) -> None:
    """Refuse a subject-state map naming anything the host's registry does not declare as apparatus.

    The context key's half of the check :func:`derive_variant_identity` makes against ``lever``
    declarations, and it exists for the same reason: a name entering a key that no reader of the
    registry can see is a coordinate nobody can place on an axis.

    **One direction only.** A declared name absent from the map is NOT an error here, because a
    subject that honestly carried nothing under the name reads identically to a forgotten writer and
    a check that cannot tell them apart fires on the honest one. That direction is held by construction on the host side
    instead — one table drives both the snapshot writer and the declarations — which is why this
    function can be narrow without leaving the hole its variant sibling closes by checking.

    Args:
        state: The subject's carried-state map, or ``None`` for a capture that has none.
        profile: The host whose registry declares the apparatus inputs.

    Raises:
        StateCoordinateError: A name in ``state`` carries no ``apparatus`` declaration.
    """
    # ``is None`` and never truthiness, the rule this component states everywhere it is
    # described: ``{}`` is a recording — the subject carried no state outside its variant
    # components — and a check that skips it on falsiness reads that recording as an absence.
    # Nothing undeclared can hide in an empty map, so the two agree today; they stop agreeing
    # the moment this function grows a check that an empty map should also answer.
    if state is None:
        return
    declared = {d.name for d in profile.sweepables.declarations if d.role == "apparatus"}
    if undeclared := sorted(set(state) - declared):
        raise StateCoordinateError(
            f"host '{profile.host_id}' captured subject state it never registered as apparatus: {', '.join(undeclared)}. "
            f"Registered apparatus inputs: {', '.join(sorted(declared)) or '(none)'}"
        )


def derive_context_identity(run: EvalRun, profile: HostProfile) -> DerivedContextIdentity:
    """Derive a run's context key from what was persisted, writing nothing.

    Args:
        run: The run to derive for, stamped or not.
        profile: The host whose declarations decide which role inputs it has and which state
            names are apparatus.

    Returns:
        The derived identity. ``partial`` is ``True`` and the component is absent
        for each of two independently-recoverable groups. ``roles`` is absent
        when a roles input **this host HAS** is unrecorded: a run whose writer
        recorded no simulator model, no judge model, no per-dim attribution (or
        only a reconstructed one), or no judge configuration set. Each is
        genuinely lost — no reconstruction from the results can recover which
        model the launch cascade chose, and reading today's records would report
        an apparatus the run never used. An input the host DECLARED it
        does not have is a different case and does not make the component absent:
        it leaves the pre-image instead, so a host that grades with code composes
        a roles component over what it actually pins, and one that declares
        nothing composes exactly the four it always did. ``world`` is absent when the run recorded
        no placements, which is lost for a sharper reason: the world registry is
        host CODE and moves under the record, so deriving placements now would
        report today's world for a run measured under an older one.
        ``subject_state`` is absent when the snapshot recorded no state map,
        which means one thing only: a host that wired no state reader. What a
        state-less host's subject carried is not recoverable from anything
        stored now, so the component drops rather than being built from a
        subset.

    Raises:
        StateCoordinateError: The snapshot's state map names something the host's registry does
            not declare as apparatus.

    Note:
        Derivation replays the *current* predicate over the *stored* inputs, which is
        why derived keys are labelled rather than stamped: a key the launch recorded is
        evidence about conditions as they were, and a derivation is a reading of them.
    """
    # All the roles inputs THIS HOST HAS, or none: the component is "the roles this run held
    # fixed", and a digest over a partial set answers a question nobody asked while looking like
    # the whole answer. A `judge_model` of None is never an absence: the runner refuses to execute
    # a judged run that names no judge, so a run naming none was not judged, and its judge inputs
    # leave the pre-image the way a host's declared inapplicability takes them out — see the
    # membership block below.
    #
    # ``recorded`` attribution only. A run carrying DERIVED effective_judges is one whose
    # attribution was INFERRED from stored config records rather than recorded as it ran —
    # eligible to be displayed, never to be hashed, because a key is what pools two runs
    # as repetitions of one condition and an inference cannot certify that. The
    # source test lives on the model as ``hashable_effective_judges`` so this predicate
    # and ``comparison_sets``' roles badge cannot disagree about what a reconstruction
    # may assert — they did, and the badge was the one asserting comparability that this
    # function refuses to hash.
    #
    # ``judge_config_ids is not None`` rather than a truthiness test, and the distinction
    # is the whole reason the field is nullable: ``{}`` records that no scored dim carried
    # a config, which is a real and stable condition — every dim was scored by the judge's
    # built-in prompt. Reading that as absent would drop the roles component for exactly
    # the runs whose judging is most uniform, and stop them pooling with each other.
    hashable_judges = run.hashable_effective_judges
    # ALL the roles inputs THIS HOST HAS, or none. The set used to be the four core names
    # unconditionally, which made the predicate permanently False for a host that grades with
    # code: it declared `judge_model` and `simulator_model` inapplicable, could never record one,
    # and so every comparison carried `context_incomplete` — glossed "at least one run never
    # recorded a pinned role model". That is a false sentence about a host that never had the
    # role, and all three surfaces hang off this one predicate.
    #
    # The DECLARATION decides membership and the VALUE still decides the claim: `omits_apparatus`
    # takes what this run recorded, so a host that declared a role inapplicable and then recorded
    # one is refuted by its own data and the input stays required.
    role_inputs: tuple[tuple[str, Any], ...] = (
        ("simulator_model", run.simulator_model),
        ("judge_model", run.judge_model),
        # The attribution map is the value of `judge_dim_divergence` — that declaration's reader
        # is what reads `hashable_effective_judges`, so a host declaring it inapplicable is
        # declaring this input away.
        ("judge_dim_divergence", hashable_judges),
        ("judge_config_ids", run.judge_config_ids),
    )
    declared_omitted = {name for name, value in role_inputs if profile.omits_apparatus(name, value)}
    # An unjudged run has no judge, whatever its host declares: the runner refuses to execute a
    # judged run that names none (``execute_run``), so this blank is a recorded fact about the run
    # rather than a gap, and the judge's dependents follow it out exactly as they follow a declaration.
    if run.judge_model is None:
        declared_omitted.add("judge_model")
    role_values = dict(role_inputs)
    # A dependent is carried out with its owner ONLY when the run recorded nothing for it. The
    # rest of this mechanism rests on the value beating the declaration, and propagation is the
    # one route that could bypass it: a host declaring `judge_model` away, on a run whose launch
    # pinned a real `judge_config_ids`, would otherwise drop a RECORDED input from the pre-image
    # with nothing logged — and two runs whose judge configuration genuinely differed would hash
    # alike. `is None` rather than `is_indeterminate`, on this predicate's own rule: `{}` here
    # records that no scored dim carried a config, which is a level and not an absence.
    omitted_roles = frozenset(
        declared_omitted
        | {
            name
            for name, owner in _ROLE_INPUT_DEPENDENCIES.items()
            if owner in declared_omitted and role_values.get(name) is None
        }
    )
    has_roles = all(value is not None for name, value in role_inputs if name not in omitted_roles)
    _refuse_undeclared_state(run.subject_snapshot.state, profile)
    components = compute_context_components(
        # Both halves of what was measured come off the engine's own snapshot. The subject id is a
        # field the engine requires and refuses to let be blank, and the state map is
        # content-addressed by the host, so nothing here has to know what a subject is made of.
        subject_id=run.subject_snapshot.subject_id,
        subject_state=run.subject_snapshot.state,
        # What the run froze before the subject's first turn: the world it seeded, and the spec its
        # kind validated — part of what the candidate was handed, so two runs whose specs differ
        # are not repetitions of one condition. Both are frozen resolved forms, so a spec stating a
        # default and one leaving it out hash alike.
        seeded_world={
            "world_seed": run.resolved_world_seed,
            "kind_spec": run.kind_spec,
        },
        template_id=run.template_id,
        test_case_ids=run.test_case_ids,
        judge_model=run.judge_model,
        simulator_model=run.simulator_model,
        effective_judges=hashable_judges if has_roles else None,
        judge_config_ids=run.judge_config_ids if has_roles else None,
        cassette_mode=run.cassette_mode,
        cassette_corpus_id=run.cassette_corpus_id,
        # No ``is not None`` gate, and this is the one field in this call where that would be
        # wrong: an absent allowlist is a run that bound nothing, which is a level the predicate
        # records rather than an absence it drops.
        tools_allowed=run.resolved_tools_allowed,
        scope_id=run.scope_id,
        # ``is not None`` rather than truthiness, for the reason ``judge_config_ids`` uses it: an
        # empty MAP is a recording — this host's world held nothing this run could place — and
        # reading it as absent would drop the component for exactly the runs whose world is
        # simplest, and stop them pooling with each other.
        world_placements=run.world_placements,
        include_roles=has_roles,
        omitted_roles=omitted_roles,
    )
    missing = missing_context_components(components)
    return DerivedContextIdentity(
        context_key=compute_context_key(components),
        context_components=components,
        identity_version=IDENTITY_VERSION,
        source="derived",
        partial=bool(missing),
        missing_components=missing,
    )


def resolve_context_identity(run: EvalRun, profile: HostProfile) -> DerivedContextIdentity:
    """Return the identity a read surface should show for ``run``.

    The single place the stamped-or-derived decision is made, so every read
    surface answers it the same way. Writes nothing either way.

    A run stamped under an *older* version is re-derived rather than returned
    as-is, and quietly mixing the two is the regrouping :data:`IDENTITY_VERSION`
    exists to prevent. Note the gate is on the version, not on this predicate:
    the counter is shared with the variant predicate, so a bump on **either**
    side re-derives every stamped run, including those whose own inputs never
    moved — conservative in the safe direction, since re-deriving unchanged
    inputs reproduces the same key. What moves for those is only the *label*:
    ``source`` becomes ``derived`` and the returned ``identity_version`` is the
    current one, so an older run renders as "derived at read" at today's version
    while its key is byte-identical to what was stamped. "Derived" therefore
    does not by itself imply this predicate changed. The stored key stays on the
    document, still honest about its own era.

    Where the context predicate itself moved, re-derivation is the point rather
    than a side effect: it is what lets a run recorded under weaker inputs report
    its key as ``partial`` instead of sitting alongside complete ones. Which
    version did which is on :data:`IDENTITY_VERSION`.

    Args:
        run: The run to resolve.
        profile: The host a re-derivation reads; unread for a run stamped at this version.

    Returns:
        The identity, with ``source`` recording whether it was recorded at
        launch or reconstructed here.
    """
    if run.context_key is not None and run.context_components is not None and run.identity_version == IDENTITY_VERSION:
        # The same predicate the derived branch uses, not a second reading of the same question:
        # this function is one public answer to "is this key partial", and two implementations of
        # it drift the first time a component joins the set.
        missing = missing_context_components(run.context_components)
        return DerivedContextIdentity(
            context_key=run.context_key,
            context_components=run.context_components,
            identity_version=run.identity_version,
            source="stamped",
            partial=bool(missing),
            missing_components=missing,
        )
    return derive_context_identity(run, profile)


def derive_variant_identity(*, run: EvalRun, profile: HostProfile) -> DerivedVariantIdentity:
    """Derive a run's variant key through the host's profile, writing nothing.

    The engine resolves the levels every run has — its candidate model, its candidate kind and
    every kind contract's levers (:meth:`~threetears.evals.contracts.host.profile.HostProfile.engine_levels`)
    — and the host's reader resolves its own; this composes the two, checks the map against the
    host's registry and hashes it. The check is what makes the registry the single authority for what
    a lever is: without it, a host could put an axis into the key that it never declared, and no
    reader of the registry would be able to tell.

    Args:
        run: The run whose observations the key describes; every one of them shares it, since a
            run is one arm.
        profile: The host whose lever reader resolves the levels and whose registry checks them.

    Returns:
        The derived identity, carrying the resolved lever map beside the key so a caller can
        place the variant on an axis rather than holding an opaque digest.

    Raises:
        LeverCoordinateError: The host's reader resolved a lever the engine resolves itself, the
            composed map named an axis the registry does not declare as a lever, or it omitted a
            declared one that carries no waiver.
    """
    levers = profile.engine_levels(run)
    hosts = profile.variant_levers(run) if profile.variant_levers is not None else {}
    if claimed := sorted(set(hosts) & set(levers)):
        raise LeverCoordinateError(
            f"host '{profile.host_id}' resolved variant levers the engine resolves itself: {', '.join(claimed)}. "
            "The candidate model, the candidate kind and every kind contract's levers are the engine's to place; "
            "a host's variant-lever reader returns only the levers the host declares beyond them"
        )
    levers |= hosts
    # Resolved against THIS RUN, not against the declaration list alone: an open family's members
    # are whatever the run carried, so the set of names a host may legitimately resolve a
    # coordinate for is run-dependent by construction. Fixed declarations are always in it.
    declared = set(profile.sweepables.resolve_levers(run).values)
    fixed = [d for d in profile.sweepables.declarations if d.role == "lever" and d.open_family is None]

    if undeclared := sorted(set(levers) - declared):
        raise LeverCoordinateError(
            f"host '{profile.host_id}' resolved variant levers it never registered as levers: {', '.join(undeclared)}. "
            f"Registered levers: {', '.join(sorted(declared)) or '(none)'}"
        )

    # The reverse, and the direction that produces a wrong MERGE rather than a wrong split: a
    # lever whose reader was forgotten drops out of the key with no version movement, so two
    # different variants come to share one. A lever that legitimately carries no coordinate has
    # said so on its declaration, which is what tells the two cases apart.
    #
    # Read over the FIXED declarations only. An open family has one expander rather than one
    # reader per member, so there is no per-member reader to forget — the failure this check
    # exists for is not expressible there, and requiring a coordinate for every member a run
    # happened to overlay would refuse every ad-hoc knob an operator reaches for.
    if forgotten := sorted(d.name for d in fixed if d.name not in levers and d.no_own_coordinate is None):
        raise LeverCoordinateError(
            f"host '{profile.host_id}' registered levers its variant map does not resolve: {', '.join(forgotten)}. "
            "A lever missing from the map contributes no coordinate, so two observations differing only on it share "
            'a key — declare `no_own_coordinate="<why>"` on it if that is intended, or resolve it.'
        )

    return DerivedVariantIdentity(
        variant_key=compute_variant_key(levers), identity_version=IDENTITY_VERSION, levers=levers
    )


def resolve_variant_identity(*, run: EvalRun, profile: HostProfile) -> DerivedVariantIdentity:
    """Return the variant identity a reader should show for ``run``'s observations.

    The single place the recorded-or-derived decision is made, so the runner stamping results
    and a bundle reading them back years later answer it identically. Writes nothing either way.

    **A recorded pre-image wins at ANY version, and that inverts**
    :func:`resolve_context_identity`. That function re-derives a run stamped under an older
    :data:`IDENTITY_VERSION`, because what it holds is a DIGEST and a set of components the
    predicate composed — re-deriving is the only way to say what today's predicate makes of it.
    Here the run holds the pre-image itself: the map is what was digested, so nothing is being
    reconstructed and there is no predicate to be out of date. Re-deriving it would replay
    today's lever map over an older run and either disagree with the stored key or describe the
    arm with a stack it never carried — which is precisely how a bump used to erase a whole
    campaign's arm table while its keys stayed authoritative for pooling.

    The version reported is therefore the one that RECORDED the map, not this build's: the map
    was resolved by that predicate and saying otherwise would date it wrongly.

    Args:
        run: The run whose observations the identity describes.
        profile: The host the derived path reads; unread when the run recorded a map.

    Returns:
        The identity, carrying the recorded lever map when the run has one and the freshly
        derived map otherwise.

    Raises:
        ValueError: The run records a lever map but no ``identity_version``.
        LeverCoordinateError: Only on the derived path — see :func:`derive_variant_identity`. A
            recorded map is not checked against today's registry, because the registry describes
            what a lever is NOW and the map records what one WAS; refusing the second against the
            first is the same version coupling this function exists to remove.
    """
    recorded = run.variant_levers
    if recorded is not None:
        # The map and the version that recorded it are stamped at one site, so a run carrying the
        # map without the version is a document no writer produces — and dating the map with this
        # build's version would claim a predicate nobody can say it ran under.
        if run.identity_version is None:
            raise ValueError(
                f"run {run.id} records a variant lever map but no identity_version; the launch stamps both "
                "together, so this run's map cannot be dated"
            )
        return DerivedVariantIdentity(
            variant_key=compute_variant_key(recorded), identity_version=run.identity_version, levers=recorded
        )
    return derive_variant_identity(run=run, profile=profile)


def variant_levers_of_run(run: EvalRun, profile: HostProfile) -> dict[str, SweepableValue]:
    """Resolve the pre-image to stamp on ``run`` — the lever map at its candidate model.

    The launch-time writer of :attr:`~threetears.evals.contracts.models.EvalRun.variant_levers`, and
    the only one: stamping is a single deliberate write at the moment the run is assembled and
    every input is frozen, which is the rule this module opens with.

    Args:
        run: The assembled run, after its world placements are stamped.
        profile: The host whose lever reader resolves the map.

    Returns:
        The resolved lever map.

    Raises:
        LeverCoordinateError: The host's map disagrees with its own registry — see
            :func:`derive_variant_identity`. Raised at LAUNCH, where it belongs: the same
            registry would refuse every observation this run is about to write.
    """
    return derive_variant_identity(run=run, profile=profile).levers


__all__ = [
    "IDENTITY_VERSION",
    "UNBOUNDED_TOOL_PERMISSIONS",
    "DerivedContextIdentity",
    "DerivedVariantIdentity",
    "LeverCoordinateError",
    "StateCoordinateError",
    "compute_content_hash",
    "compute_context_components",
    "compute_context_key",
    "compute_variant_key",
    "derive_context_identity",
    "derive_variant_identity",
    "missing_context_components",
    "resolve_context_identity",
    "resolve_variant_identity",
    "variant_levers_of_run",
]
