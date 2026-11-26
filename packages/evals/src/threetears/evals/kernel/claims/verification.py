"""Checking each listed claim against numbered evidence: the verify step of claims grading, and the verdict it gives.

:class:`ClaimVerifier` takes the claims :class:`~threetears.evals.kernel.claims.extraction.ClaimExtractor` listed and
asks a model, in batches, which numbered pieces of evidence support or contradict each one. It never asks for a
verdict: the verdict is derived in code from the citations that pass the checks (:func:`truth_verdict`).

**Two steps, two kinds of evidence.** The *truth* step sees only what is the same for every arm: the case key's facts,
numbered ``K1..``, and passages of the case's own documents, numbered ``S1..``. It never sees a document the candidate
was given or fetched itself (a ``run`` document): those never nominate a claim and never decide one. The *ground* step,
asked only when the criterion names ``grounded_in``, sees only passages of documents of those provenances, the
candidate's own, and cites support alone. A key fact never grounds a claim, since the candidate was never shown it.

**Passages are chosen by code, under the page budget, the same way every time.** Each document is cut into tiles of
``window_chars`` from its start. Each tile is ranked by the anchors it holds: the folded words, of at least four
characters or holding a digit, of the cell's world claims and the key's names and aliases, each weighted by one over
the number of the document's tiles that hold it, so a rare name outweighs a common word. Tiles are then taken in rounds
across the documents, in their order, each document's best remaining tile per round, while the document's
``per_document_chars`` and the step's ``total_chars`` allow; adjacent tiles join into one passage. A document not read
whole is recorded as truncated. Each step's selection is bounded on its own, so the documents a candidate fetched never
take the case's share: truth reads the same case passages whatever an arm fetched, given the same claims.

**A citation names a value and its subject, in one sentence or row.** Every citation quotes at most
:data:`~threetears.evals.schema.models.CLAIM_QUOTE_CHARS` characters, found by
:func:`~threetears.evals.kernel.claims.verbatim.verbatim_span` inside the evidence the call was shown and cited: a
passage (never elsewhere in its document), or a key fact's line as shown after its ref (``A key fact about <entity>:
<fact>``, so the entity it names can stand as the subject of a fact that does not name it). Inside that quote it names
the words that name the claim's subject and the words that state the value. Both are found inside the stored quote,
and no sentence end (``.``, ``!`` or ``?``, with any closing quotes or brackets after it, before a space or the end)
or line break may lie between them, so 'Ann Lee said, "I stay in Leeds." Her brother moved to York.' parts Ann Lee from
York. That
narrows the neighbouring-value error, a value taken from the sentence or row next to the one about this subject, to
one sentence or row. A full stop after an abbreviation ends no sentence: after initials or a dotted acronym ("J. R.
R. Tolkien", "the U.S. capital"), or after a capitalised token of at most three letters followed by a capitalised word
or a digit ("Dr. John", "St. Louis", "Vol. 2", "No. 5"). A "!" or "?" followed by spaces and then "(" or a
lower-case letter ends none either: it stands inside a title or a phrase ("Revolting Cocks were a Wax Trax! (Chicago)
act.", "on Wax Trax! in the early 1980s"), as does a "?" before a lower-case letter. Before a capitalised word, a
digit or a line break each still does ("He left! Then ...", "He left? No."), and so does a "?" before a bracket ("Did
Ann Lee stay? (Her brother moved to York.)"). The cost, taken so that a title ending in "!" keeps its bracketed
gloss: an exclamation before a bracketed sentence reads as one sentence with it ("Wow! (Her brother moved to York.)"),
and an exclamation or question before a lower-case word ("Stop! he said") reads as one, as it is spoken. Each break costs a repair round, so the rule takes the
cheaper error: a real sentence that ends in one of those shapes before a capitalised word or a digit ("printed in
May. The bound volume followed in 1932", "in the U.S. The ...") is read as one sentence, and its binding is left to
the model, as within any sentence. The rule is English-shaped: case is read in any cased script, but the shapes are
how English abbreviates, and a language that capitalises nouns ends more sentences on a short capitalised word. A
year or any other number before the full stop is always a sentence end, so "The pamphlet appeared in 1931. The bound
volume followed in 1932." still parts the bound volume from 1931. Clauses joined by a semicolon count as one
sentence, so a subject and a value either side of one ("members may renew a loan once; doing so costs two dollars")
pass with no repair; binding across the semicolon is not proved by code. That is the accepted trade-off: fewer forced
repairs on rule-style sentences, for a weaker code check. Nor does the check prove the value belongs to that subject
within any sentence: one stating two values for two subjects ("the pamphlet in 1931 and the bound volume in 1932", or
"the pamphlet appeared in 1931; the bound volume followed in 1932") stays the model's to bind, and calibration
measures what remains. What is stored is the evidence's own slice, never the model's strings. A key fact is held to
the rule as a passage is, since its support is what reads ``supported`` first: a reply citing it with no quote, with no
value, or with a value the fact does not state (the claim's own, where the fact gives another) is asked again. What
the check cannot catch is a value the fact does state, cited as support for a claim that states another: binding the
stance to the value stays the model's, as for a passage, and the stored value span shows which words it read. The
known limit: a key fact of several sentences is cited for the sentence that names its subject. Its line's heading (``A
key fact about <entity>:``) takes the one-sentence rule as any text does, so a subject quoted from the heading binds
only to a value before the fact's first sentence end; a later sentence that never names the entity ("It followed in
1932.") cannot be cited alone. Binding the heading's entity across sentences would let "Ann Lee was born in Leeds. Her
brother moved to York." support "Ann Lee moved to York", and a false ``supported`` costs more than a repair.

**Precedence among case documents.** A case document may override others (``EvidenceDocument.overrides``: house
rules over a rulebook, an erratum over its text, a policy update over the old policy), following the chain. The truth
call is shown which passage's document overrides which, beneath the passages, and told to cite both sides, since code
applies the precedence and the model never decides a verdict. A citation is overruled when a passage of a document
that overrides its own cites the claim with the opposite stance. So a claim supported only from an overridden document
that an overriding one contradicts reads ``contradicted``; support from the overriding document stands over a
contradiction from the one it overrides; and where the overriding document is silent, the overridden one's citation
stands. A key fact is no document: it overrides nothing and is never overruled, so a key fact an overriding document
contradicts is a conflict, asked again in the re-ask (below), and code cannot detect the fact as stale. A host keeps
its key facts to the values in force. The ground step reads no precedence, since it cites support alone.

**A support states every value.** A claim may state several values ("drums and vocals", "in a band and on its
album"). The truth and re-ask rules say each piece cited as support states its values for the claim's own subject
and in the relation the claim gives them, and the pieces cited as support, one or several, state every value the claim
states between them: a value stated in another relation (another role, attribute, time or place) supports nothing,
evidence that states some of the values and is silent on the rest is not cited as support alone, and evidence that
gives a different value for one of them contradicts it. A value joint by meaning (a song two people wrote together) is
one value. Code cannot read whether evidence covers a value or in which relation it states it, so the rule binds the
model and no check holds it. What code does check of each support is what it checks of any citation: its quote is a
stretch of evidence the call was shown for that claim, and its subject and value spans lie inside that quote, in one
sentence. So a reply citing a partial support, or pooling facts about other properties that between them name every
value, passes every code check and reads ``supported``; pooled beside a contradiction, they make a conflict, never a
support. A partial
support beside a contradiction of another of its values is a conflict, and goes to the re-ask. The cost, taken for the
false rate: a true compound the evidence states only part of reads ``unverifiable``, never ``supported``.

**Batches.** One call per unit, holding every world claim of that unit, and one more for the claims outside any unit.
A restatement is not asked again: it rides in its original's batch, shown beside it, and takes the original's
citations, suspicion and verdict, since it states the same fact. Session claims are about the output itself and are
never verified here; their finding is None.

**Repair, not refusal.** A reply that will not parse, or does not answer exactly the claims asked, is asked again in
full, naming why, up to three times, then fails (``unreadable``). A claim whose answer fails a check (a ref to no
evidence shown, a quote not in the passage cited or over the length, a missing or misplaced span, a subject and value
in different sentences, a suspect claim that cites evidence) is asked again alone, with its failure named, within the
repair budget: its rounds, and the most items one cell may ask again, which each step holds on its own. Only failing
claims are asked again. Still failing, or past the budget, fails (``repair_budget_spent``); a reply cut short fails at
once (``cut_short``).

**Two kinds of claim are asked again, all in one call per cell (the re-ask),** once the truth step's repairs pass.

- *A doubted claim, with the key facts about what it names.* The truth batch is shown every key fact, but no check can
  make a model cite one, so a world claim it answers with no citation and marks suspect, though a key fact contradicts
  it, would read ``unverifiable`` wherever the web is off, past the cell's web budget or inconclusive. So every such
  claim that names a key entity is listed with the refs of its own entities' facts. Code decides which those are,
  never the model: a claim names a key entity when the entity's name, or one of its aliases, stands in the claim's
  proposition or a restatement's as whole words in order, under the verbatim fold. So "Ann Lee" stands in "Ann Lee
  lives in York" and in "Ann Lee's home is York", and neither the entity "Annabel Lee", "Lee Ann" nor "Ann Lee's
  brother" stands in "Ann Lee moved to York". An item citing evidence replaces the batch's answer: a cited
  contradiction reads ``contradicted`` and a cited support ``supported``, graded against the case, so the web never
  checks it. **One citing nothing leaves the batch's answer standing**, and the claim goes on to the web as before.
- *A conflict, with the evidence it cited.* A world claim whose citations that stand once precedence is read both
  support and contradict it is listed with the refs of exactly the evidence it cited on both sides, key facts and case
  passages, and never its entities' other facts. Its item replaces the batch's citations whatever it cites: only
  support reads ``supported`` and only contradiction ``contradicted``, both from the case; both reads ``unverifiable``
  with ``evidence_conflict``, since the verdict table never reads evidence still standing on both sides as support.
  **One citing nothing never falls back to the batch's answer**: the batch's citations stay on record, both sides, so
  the claim reads ``unverifiable`` with ``evidence_conflict``. That is the opposite of a doubted claim's rule, since a
  conflict's batch answer is itself what is in doubt. A conflict cites evidence, so it is never nominated, and the web
  never decides one: it is the case's own evidence that disagrees.

The re-ask has its own prompt, which asks whether any of a claim's evidence states it true or false and says nothing of
the batch's answer. Each item is checked as a truth answer is, against that claim's own refs alone, so a ref listed
for another claim is no evidence shown for it. A failing item is repaired as a truth batch's is, shown its own
evidence; a restatement takes its original's outcome. No claim's outcome turns on its place: every claim of either
kind is in the one call. A re-ask reply cut short or unreadable in its attempts, an item past the repair budget, a
provider's failure or a protocol defect fails as any truth call does, and never leaves a claim ``unverifiable``.

**What the re-ask costs.** One call per cell, asked again in full only when its reply cannot be read (at most
:data:`_ATTEMPTS` calls), whatever the number of claims it holds. A conflict rides in the call the doubted claims make,
adding none; a cell with a conflict and no doubted claim naming a key entity makes the one call where it made none. The
price ceiling prices it wherever the truth step can be shown evidence (a key fact or a case document), at the truth
step's whole evidence. Its failing items are repaired within what the truth step's repairs left of
``max_items_per_cell``, so ``RepairBudget``'s arithmetic is unchanged: the truth step, its re-ask items included, asks
at most ``max_items_per_cell`` items again, and a cell at most four times that.

**Its limits.** The match reads names, not reference. A claim naming the entity inside a phrase about someone else
("Ann Lee's brother moved to York") is listed with Ann Lee's facts. A fact filed under Ann Lee's own entry that is
about someone else ("Her brother moved to York.") is shown, and cited as support for "Ann Lee moved to York", with its
heading's "Ann Lee" as the subject, it passes every code check: only the model's reading stands between it and a false
``supported``. A short name or alias inside a longer one ("Lee" for Ann Lee, an entity named "Ann") matches every claim
naming the longer. A claim that names its subject only by a pronoun or a description the key lists as no alias, or in
a script written without spaces between words, is asked nothing again as a doubted claim. Nor does code hold the
relation: evidence about another attribute of the same subject ("Ann Lee was born in Leeds", against "Ann Lee lives in
York"), cited as a contradiction that passes every check, reads ``contradicted``, and cited as the only support of a
conflict asked again, reads ``supported``. A conflict is shown again the very evidence it misread, so a model that
binds it the same way twice keeps that reading; and whatever a conflict's re-ask cites alone stands, so one that drops
the contradiction an overriding document gave reads the support it kept. The re-ask's prompt says that such evidence bears on nothing, and that
binding stays the model's, as in the truth batch.

**Truth and groundedness fail apart.** A truth failure raises
:class:`~threetears.evals.kernel.claims.failures.ClaimGradingFailed`, costing the cell this criterion. A ground
failure never touches truth: the documents an arm fetched differ by arm, so they must not decide whether its cell
keeps a truth reading. That holds for every way a ground call can fail: a reply cut short, unreadable or past its
repairs, the run's cost cap, the provider failing the call, and a completion that breaks the protocol on it. The
verification comes back with every truth finding, no groundedness, and the ground step's loss on
:attr:`Verification.ground_loss`, for the judge phase to record against the groundedness reading alone. Only a call
the rig could not serve (a replay's miss) is raised from the ground step, since that fails the run. A cell whose
evidence declares no documents (None, not empty) asks no ground step at all: nothing says what its candidate was given,
so its ground step is lost as ``documents_undeclared`` and truth is checked as for any cell.

**It records what it spent, and how it read.** Every completion is folded into the cell's ``judge`` ledger and
recorded as a :class:`~threetears.evals.schema.models.ClaimCall`. :attr:`ClaimVerifier.identity` names the model and a
digest of everything that shapes a verdict: the prompts with the host's guidance, the reply schemas, the page budget
and the selection rule, the batch rule, the repair budget, the re-ask rule, ``grounded_in``, the quote cap, the
attempts and the verbatim fold's identity.

Engine-internal: the judge phase builds and calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.capped_completion import call_ceiling
from threetears.evals.kernel.claims.extraction import ExtractedClaim, claim_call, echo_chars, filler
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.claims.verbatim import VERBATIM_FOLD, verbatim_span
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.claims import Segment
from threetears.evals.schema.completion import CompletionClient, CompletionProtocolDefect
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.models import (
    CLAIM_QUOTE_CHARS,
    ClaimCall,
    ClaimCitation,
    EvidenceDocument,
    EvidenceProvenance,
    EvidenceRef,
    GradedClaim,
    KnownClaimLoss,
    document_precedence,
    PageBudget,
    RepairBudget,
    TextSpan,
)
from threetears.evals.schema.reference import ReferenceFact, ReferenceKey

log = get_logger(__name__)

#: How many times one batch is asked in full: the first call and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: The ledger every verify call is billed to: it reads the candidate's output, as a judge does.
_ROLE: Final = "judge"

#: A ref as wide as any a step can number: what a priced prompt prints where a passage's ref will stand.
_WIDEST_REF: Final = 999_999_999

#: Something other than whitespace.
_NOT_BLANK: Final = r"\S"

#: The fewest characters a folded word needs to anchor a passage, unless it holds a digit.
_ANCHOR_MIN_CHARS: Final = 4

#: A word, once folded.
_WORD: Final = re.compile(r"\w+")

#: The page-selection rule, by revision and in words: part of the identity, raised with any change to the code below.
_SELECTION_RULE: Final = (
    "passages-v1: tiles of window_chars from each document's start; each ranked by the anchors it holds (folded words "
    "of the world claims and the key's names and aliases, of at least 4 characters or holding a digit), each weighted "
    "1 / the number of the document's tiles holding it, ties to the earlier tile; tiles taken in rounds across the "
    "documents in order, one per document per round, while per_document_chars and the step's total_chars allow; "
    "adjacent tiles joined; each step selected on its own"
)

#: The batch rule, in words: part of the identity.
_BATCH_RULE: Final = (
    "one batch per unit in unit order, then one for claims outside any unit; a restatement rides with its original "
    "and takes its answer; session claims are not verified"
)

#: The re-ask rule, by revision and in words: part of the identity, raised with any change to which claims are asked
#: again, what each is shown (:func:`_names_subject`, :func:`_standing`) or how its answer is read.
_REASK_RULE: Final = (
    "reask-v2: once the truth step's repairs pass, two kinds of original world claim are asked again, all in one call "
    "per cell: a doubted claim, whose answer cites nothing and is suspect, and whose proposition or a restatement's "
    "names a key entity (the entity's name or an alias standing in it as whole folded words in order), listed with the "
    "refs of the key facts of the entities it names; and a conflict, whose citations standing after precedence both "
    "support and contradict it, listed with the refs of exactly the key facts and case passages it cited; each is "
    "checked against its own refs alone, and its failing items are repaired within the truth step's repair budget; a "
    "doubted claim's answer citing evidence replaces the batch's, and one citing nothing leaves the batch's standing; "
    "a conflict's answer replaces the batch's whatever it cites, and one citing nothing keeps the batch's citations, "
    "so it never falls back to the batch's supported; the verdict table reads citations still standing on both sides "
    "unverifiable, with evidence_conflict"
)

#: What a claim is, for a check: the same fact about the same subject. Shared by every verify prompt.
_CITING_RULES: Final = (
    # one line: the only placeholder is CLAIM_QUOTE_CHARS, an int; every rule's line is written here in full
    "- Bind the value to its subject. Evidence that states a value for a neighbouring subject (another item, edition, "
    "date, person or place in the same passage) bears on nothing here, even when its value matches the claim's.\n"
    "- An omission contradicts nothing: evidence that does not state the claim's fact, or states other facts about "
    "its subject, is not cited.\n"
    "- Quote the one sentence or table row of the cited evidence that states it, copied exactly as it is written, "
    f"at most {CLAIM_QUOTE_CHARS} characters, never shortened with an ellipsis. Give subject: the words of that quote "
    "that name the claim's subject; and value: the words of that quote that state the value at issue.\n"
    "- Cite each piece of evidence at most once for a claim."
)

#: A claim that states several values is supported only where its support states every one: the truth and re-ask
#: rules', and no code check's, since code cannot read whether evidence covers a value.
_EVERY_VALUE_RULE: Final = (
    "- A claim may state several values (drums and vocals; in a band and on its album). Each piece you cite as "
    "support states its values for this same subject and in the relation the claim gives them: a value that evidence "
    "states in another relation (another role, attribute, time or place of the subject) supports nothing here. Cite "
    "support only when the pieces you cite as support, one or several, state every value the claim states between "
    "them; when no evidence given states one of its values that way, cite nothing as support, not even evidence that "
    "states the rest. Evidence that gives a different value for one of them contradicts the claim, even when it "
    "states the others. Values joined by meaning (a song two people wrote together) are one value.\n"
)

_TRUTH_RULES: Final = (
    # one line: the every-value rule is one list item, written here in full with no placeholder
    "Read each claim against the numbered evidence given, and nothing else: not what you remember.\n"
    "- Cite every piece of evidence that bears on a claim, by its ref (K1, S1, ...), with its stance: supports when "
    "it states this same fact about this same subject; contradicts when it states a different value for this same "
    "subject and relation. A contradiction's value is the value the evidence gives instead.\n"
    + _EVERY_VALUE_RULE
    + _CITING_RULES
    + "\n- A key fact (K) is quoted the same way, from its line as shown after its ref: the entity that line names "
    "may stand as the subject.\n"
    "- suspect: true only when no evidence given bears on the claim and you doubt that it is true; false otherwise. "
    "A claim you cite evidence for is never suspect."
)

_REASK_RULES: Final = (
    # one line: the every-value rule is one list item, written here in full with no placeholder
    "Read each claim against the case evidence listed for it, and nothing else: not what you remember. Each piece is "
    "a key fact (K) about an entity the claim names, or a passage (S) of a case document.\n"
    "- Does any of it state that the claim is true or false? Cite each piece that does, by its ref, with its stance: "
    "supports when it states this same fact about this same subject; contradicts when it states a different value for "
    "this same subject and relation. A contradiction's value is the value the evidence gives instead.\n"
    "- A piece may state another attribute of the entity, or something about someone or something else (a relative, a "
    "work, a place): such a piece bears on nothing here. When none of a claim's evidence bears on it, cite nothing for "
    "it.\n"
    + _EVERY_VALUE_RULE
    + _CITING_RULES
    + "\n- A key fact (K) is quoted the same way, from its line as shown after its ref: the entity that line names "
    "may stand as the subject."
)

_GROUND_RULES: Final = (
    "Read each claim against the numbered passages given, which come from documents the text's author was given, and "
    "nothing else: not what you remember.\n"
    "- Cite every passage that supports a claim, by its ref (S1, S2, ...): one that states this same fact about this "
    "same subject. Never cite a passage that only mentions the subject, or that states something else.\n"
    + _CITING_RULES
)

_GUIDANCE_HEADING: Final = "\n\n# Notes on reading claims here\n\n"

_ANSWER_LINE: Final = "\n\nAnswer with JSON only, in the shape the response format states."

_TRUTH_SYSTEM: Final = (
    "You check the factual claims a text states against numbered evidence. Each claim is numbered; answer for every "
    "claim asked, by its number.\n\n" + _TRUTH_RULES + _ANSWER_LINE
)

_REASK_SYSTEM: Final = (
    "You check factual claims a text states against the case evidence listed for each. Each claim is numbered and "
    "followed by the refs of its evidence; answer for every claim asked, by its number.\n\n"
    + _REASK_RULES
    + _ANSWER_LINE
)

_GROUND_SYSTEM: Final = (
    "You check which factual claims a text states are supported by numbered passages. Each claim is numbered; answer "
    "for every claim asked, by its number.\n\n" + _GROUND_RULES + _ANSWER_LINE
)

_TRUTH_REPAIR_SYSTEM: Final = (
    "You correct your answer for one claim checked against numbered evidence. You are given the evidence, the claim, "
    "your answer and what is wrong with it. Answer for the claim again with that fixed.\n\n"
    + _TRUTH_RULES
    + _ANSWER_LINE
)

_GROUND_REPAIR_SYSTEM: Final = (
    "You correct your answer for one claim checked against numbered passages. You are given the passages, the claim, "
    "your answer and what is wrong with it. Answer for the claim again with that fixed.\n\n"
    + _GROUND_RULES
    + _ANSWER_LINE
)

_USER_TEMPLATE: Final = "# Evidence\n\n{evidence}\n\n# Claims\n\n{claims}"

_FACT_TEMPLATE: Final = "{ref}. {fact}"

_REASK_TEMPLATE: Final = "# Case evidence\n\n{evidence}\n\n# Claims\n\n{claims}"

#: Beneath a claim the re-ask lists: the refs of its evidence, the only pieces it is read against (a doubted claim's,
#: the key facts about the entities it names; a conflict's, what it cited on both sides).
_REASK_REFS: Final = "Its evidence: {refs}"

#: A key fact as a call is shown it after its ref: the text a key citation quotes.
_FACT_LINE: Final = "A key fact about {entity}: {claim}"

_PASSAGE_TEMPLATE: Final = "{ref}. From {title}, characters {start} to {end}:\n{text}"

_NO_EVIDENCE: Final = "None."

#: What failed in the ground step of a cell whose evidence declares no documents under ``grounded_in``.
_UNDECLARED: Final = (
    "the cell's claim evidence declares no documents (None, not empty), so nothing says what the candidate was given "
    "and no claim can be grounded"
)

#: Beneath the passages, when a shown passage's document overrides another's that is shown too: the precedence code
#: applies to the citations, stated so the call cites both sides.
_PRECEDENCE_HEADING: Final = (
    "Where passages of these documents disagree, the first named in a line below overrides the others in it. Cite "
    "every passage that bears on a claim, from either side: the precedence is applied to your citations, so never "
    "leave a passage out because another overrides it."
)

_PRECEDENCE_LINE: Final = "- {overriding} overrides {overridden}."

_PRECEDENCE_DOCUMENT: Final = "{title} ({refs})"

_CLAIM_TEMPLATE: Final = 'Claim {position}, {place}: {claim}\nIt is said as: "{excerpt}"'

_RESTATED_TEMPLATE: Final = 'It is said again, as claim {position}: "{excerpt}"'

_AGAIN_TEMPLATE: Final = "{user}\n\n# Your last reply\n\nIt could not be used: {failure}. Answer again, in full."

_REPAIR_TEMPLATE: Final = (
    "# Evidence\n\n{evidence}\n\n# The claim\n\n{claim}\n\n# Your answer\n\n{answer}\n\n# What is wrong\n\n{failure}"
)

#: What breaks one sentence or table row from the next, between a citation's subject and its value: a sentence end
#: (``.``, ``!`` or ``?``, and any closing quotes or brackets after it, before a space or the end) or a line break,
#: unless :func:`_ends_no_sentence` says a full stop marks an abbreviation, or a ``!`` or ``?`` stands inside a sentence.
#: A semicolon joins clauses of one sentence, and is no break.
_SENTENCE_END: Final = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s|$)|[\n\r]")

#: The token a full stop follows: the letters, and full stops between them, up to it.
_TOKEN_BEFORE: Final = re.compile(r"[^\W\d_](?:[^\W\d_]|\.)*\Z")

#: Initials and dotted acronyms, the full stop after the last letter aside: ``J``, ``U.S``, ``e.g``.
_INITIALS: Final = re.compile(r"(?:[^\W\d_]\.)*[^\W\d_]")

#: The first character of the word after a full stop and its spaces.
_NEXT_WORD: Final = re.compile(r"\s+(\S)")

#: The most letters a capitalised abbreviation (``No``, ``St``, ``Vol``) holds.
_ABBREVIATION_LETTERS: Final = 3

#: The sentence-break rule, by revision and in words: part of the identity, raised with any change to
#: :func:`_ends_no_sentence` or the patterns above.
_SENTENCE_RULE: Final = (
    "sentences-v3: a break is a '.', '!' or '?', with any closing quotes or brackets after it, before a space or the "
    "end, or a line break; a '.' is no break after initials or a dotted acronym (single letters, each but the last "
    "followed by '.'), whatever follows, nor after a token of 2 to 3 letters, the first upper case and the rest lower "
    "case, followed by spaces and an upper-case letter or a digit; a '!' is no break when followed by spaces and then "
    "'(' or a lower-case letter, a '?' when followed by spaces and then a lower-case letter"
)

#: What may follow a ``!`` and its spaces inside one sentence, besides a lower-case letter.
_INSIDE_AFTER_BANG: Final = "("

#: Every failure a repair or a full re-ask names to the model, as the templates it is written from: what the judge is
#: shown, so part of the identity. ``{kind}`` names what a quote was cut from: a passage, or a key fact.
_FAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "twice": "it cites {refs} more than once; cite each piece of evidence once",
        "suspect": "it is marked suspect and cites evidence; suspect means no evidence given bears on the claim",
        "unknown_ref": "it cites {ref}, which is no evidence shown here; cite the refs listed under Evidence",
        "no_quote": "its citation of {ref} quotes nothing; a {kind} is cited with the sentence or row that states it",
        "not_shown": (
            "its quote of {ref} is not in {kind} {ref} as shown; a quote is one unbroken stretch of the {kind} it "
            "cites, copied as it is written"
        ),
        "too_long": "its quote of {ref} runs {length} characters; a quote is one sentence or row of at most {cap}",
        "no_span": "its citation of {ref} names no {what}; give the words of its quote that state the {what}",
        "span_outside": "its {what} for {ref} is not in its quote; the {what} is words of the quoted sentence itself",
        "apart": (
            "its subject and value for {ref} stand in different sentences or rows of its quote; quote the one sentence "
            "or row that states this value for this subject"
        ),
        "correction": "{failure}; and the last correction could not be used: {why}",
        "shape": "it is not the shape the response format states ({problems})",
        "answers": "it must answer claims {asked} once each, and it {problems}",
        "missing": "leaves out claims {missing}",
        "extra": "answers claims {extra} that were not asked, or more than once",
    }
)

_NULLABLE_STRING: Final[dict[str, Any]] = {"anyOf": [{"type": "string"}, {"type": "null"}]}

_TRUTH_CITATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "ref": {"type": "string"},
        "stance": {"type": "string", "enum": ["supports", "contradicts"]},
        "quote": _NULLABLE_STRING,
        "subject": _NULLABLE_STRING,
        "value": _NULLABLE_STRING,
    },
}

_GROUND_CITATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "ref": {"type": "string"},
        "quote": _NULLABLE_STRING,
        "subject": _NULLABLE_STRING,
        "value": _NULLABLE_STRING,
    },
}

type _Step = Literal["truth", "ground"]

#: The steps in the order their batches run and their failing items are asked again.
_STEPS: Final[tuple[_Step, ...]] = ("truth", "ground")

#: A web check's outcome as the verdict table reads it: what the page said; that the cell's budget, the cost cap or the
#: deadline left it unchecked (``skipped_cap``); or that the web step failed before deciding it (``lost``).
type WebOutcome = Literal["true", "false", "inconclusive", "skipped_cap", "lost"]

type Verdict = Literal["supported", "contradicted", "unverifiable", "true_ungrounded"]

#: What a checked quote was cut from, as a failure names it to the model.
type QuotedKind = Literal["passage", "key fact"]


class _Cited(EvalBaseModel):
    """One citation as a truth reply gives it."""

    ref: str = Field(pattern=_NOT_BLANK)
    stance: Literal["supports", "contradicts"]
    quote: str | None
    subject: str | None
    value: str | None


class _GroundCited(EvalBaseModel):
    """One citation as a ground reply gives it: support, so it carries no stance."""

    ref: str = Field(pattern=_NOT_BLANK)
    quote: str | None
    subject: str | None
    value: str | None

    def as_cited(self) -> _Cited:
        return _Cited(stance="supports", **self.model_dump())


class _Answer(EvalBaseModel):
    """One claim's answer under either step. A ground answer is never suspect."""

    citations: list[_Cited]
    suspect: bool


class _TruthItem(EvalBaseModel):
    claim: int = Field(ge=0)
    citations: list[_Cited]
    suspect: bool


class _TruthBatch(EvalBaseModel):
    claims: list[_TruthItem]


class _ReaskItem(EvalBaseModel):
    claim: int = Field(ge=0)
    citations: list[_Cited]


class _ReaskBatch(EvalBaseModel):
    claims: list[_ReaskItem]


class _GroundItem(EvalBaseModel):
    claim: int = Field(ge=0)
    citations: list[_GroundCited]


class _GroundBatch(EvalBaseModel):
    claims: list[_GroundItem]


class _TruthRepaired(EvalBaseModel):
    citations: list[_Cited]
    suspect: bool


class _GroundRepaired(EvalBaseModel):
    citations: list[_GroundCited]


@dataclass(frozen=True)
class ClaimFinding:
    """What the verify step found for one world claim: the citations that passed their checks, and the call behind.

    :ivar truth_citations: Case evidence cited for or against it, each checked.
    :ivar suspect: No case evidence bears on it, and the model doubts it: the judge's nomination for the web check.
    :ivar truth_call: The call whose answer stands for it, by its place in :attr:`Verification.calls`: its batch's,
        the repair that fixed it, or the re-ask (or the repair of its item) that decided it: a doubted claim's whose
        citation stands, or a conflict's, whatever it cited. A restatement's is its original's.
    :ivar ground_citations: Documents the candidate was given that support it, each checked; empty when not asked.
    :ivar grounded: Whether one supports it; None when the criterion names no ``grounded_in``, or the ground step was
        lost.
    """

    truth_citations: tuple[ClaimCitation, ...]
    suspect: bool
    truth_call: int
    ground_citations: tuple[ClaimCitation, ...] = ()
    grounded: bool | None = None


@dataclass(frozen=True)
class Verification:
    """What the verify step read and found for one cell.

    :ivar evidence: Every piece of evidence a call was shown, numbered as the calls cited it.
    :ivar documents_truncated: The documents a shown selection did not read whole, by id, in document order.
    :ivar calls: Every call made, in order.
    :ivar findings: One per claim, in the claims' order: a :class:`ClaimFinding` for a world claim, None for a session
        claim.
    :ivar ground_loss: Why the ground step was lost (``cut_short``, ``unreadable``, ``repair_budget_spent``,
        ``cost_cap``, ``hold_exceeded``, ``provider_error``, ``protocol_defect`` or, never asked, ``documents_undeclared``), its
        own loss and never the criterion's: the truth findings stand, and every finding's ``grounded`` reads None.
        None when the step was not lost, or not asked.
    :ivar ground_failure: What failed in the ground step, named, beside :attr:`ground_loss`; None with it.
    :ivar outranks: Each truth passage's ref whose document overrides another shown document, and the refs of that
        document's passages: the precedence :func:`truth_verdict` applies. Empty when no shown document overrides one.
    """

    evidence: tuple[EvidenceRef, ...]
    documents_truncated: tuple[str, ...]
    calls: tuple[ClaimCall, ...]
    findings: tuple[ClaimFinding | None, ...]
    ground_loss: KnownClaimLoss | None = None
    ground_failure: str | None = None
    outranks: Mapping[str, frozenset[str]] = MappingProxyType({})


def _undeclared(undeclared: bool) -> tuple[KnownClaimLoss | None, str | None]:
    """The ground step's loss and its failure for a cell whose documents are undeclared under ``grounded_in``; else
    neither."""
    return ("documents_undeclared", _UNDECLARED) if undeclared else (None, None)


@dataclass(frozen=True)
class TruthVerdict:
    """A world claim's truth verdict, derived from its case citations, its suspicion and a web outcome.

    :ivar verdict: ``supported``, ``contradicted``, ``unverifiable`` or ``true_ungrounded``.
    :ivar graded_against: ``case`` or ``web``, what decided it; None for ``unverifiable``.
    :ivar evidence_conflict: Case evidence both supports and contradicts it, and no override settles which stands: it
        reads ``unverifiable``.
    :ivar nominated_by: ``judge`` when no case evidence bears on it and the model doubts it; else None.
    """

    verdict: Verdict
    graded_against: Literal["case", "web"] | None
    evidence_conflict: bool
    nominated_by: Literal["judge"] | None


def truth_verdict(
    citations: Sequence[ClaimCitation],
    *,
    suspect: bool,
    web: WebOutcome | None = None,
    outranks: Mapping[str, frozenset[str]] | None = None,
) -> TruthVerdict:
    """The verdict table: case support alone reads supported, case contradiction alone contradicted, both
    unverifiable with ``evidence_conflict``, and the web decides only a claim no case evidence bears on that the
    model doubts.

    Precedence comes first (:func:`_standing`): the table reads the citations that stand. So support from an
    overridden document that an overriding one contradicts reads contradicted, the overriding document's support
    stands over the overridden one's contradiction, and where no overriding document cites the claim, every citation
    stands. Overrides run in no cycle, so some citation always stands. Citations still standing on both sides are a
    conflict, which the verify step asks again (the re-ask): what this reads is what that left, and it never reads
    evidence on both sides as support.

    :param citations: The claim's checked case citations.
    :param suspect: Whether the model doubts it, with no case evidence bearing on it.
    :param web: What the web check found for it; None when it was not checked (web off, or not yet run).
    :param outranks: Each passage ref, and the refs of the passages whose documents its document overrides
        (:attr:`Verification.outranks`); None or empty where no document overrides another.
    :return: The verdict, what decided it, whether case evidence conflicts, and who nominated it.
    :raises ValueError: A web outcome is given for a claim nobody nominated: the web never overrules case evidence and
        never checks a claim the model did not doubt.
    """
    standing = _standing(citations, outranks)
    supports = any(citation.stance == "supports" for citation in standing)
    contradicts = any(citation.stance == "contradicts" for citation in standing)
    nominated = not citations and suspect
    if web is not None and not nominated:
        raise ValueError(
            "a web outcome was given for a claim nobody nominated: the web checks only a claim no case evidence bears "
            "on that the judge doubts, and never overrules case evidence"
        )
    if supports and contradicts:
        return TruthVerdict("unverifiable", None, True, None)
    if supports:
        return TruthVerdict("supported", "case", False, None)
    if contradicts:
        return TruthVerdict("contradicted", "case", False, None)
    if not nominated:
        return TruthVerdict("unverifiable", None, False, None)
    if web == "true":
        return TruthVerdict("true_ungrounded", "web", False, "judge")
    if web == "false":
        return TruthVerdict("contradicted", "web", False, "judge")
    return TruthVerdict("unverifiable", None, False, "judge")


def _standing(citations: Sequence[ClaimCitation], outranks: Mapping[str, frozenset[str]] | None) -> list[ClaimCitation]:
    """The citations that stand once precedence is read: each but those another, of the opposite stance, overrules by
    citing a passage whose document overrides the first one's (``outranks``)."""
    above = outranks or {}
    return [
        citation
        for citation in citations
        if not any(
            other.stance != citation.stance and citation.ref in above.get(other.ref, frozenset()) for other in citations
        )
    ]


def _in_conflict(citations: Sequence[ClaimCitation], outranks: Mapping[str, frozenset[str]]) -> bool:
    """Whether the citations that stand both support and contradict: a conflict, which the re-ask asks again."""
    stances = {citation.stance for citation in _standing(citations, outranks)}
    return stances == {"supports", "contradicts"}


@dataclass(frozen=True)
class WebGraded:
    """What the web check left for one nominated claim, as its grade reads it.

    :ivar outcome: What the page said (``true``, ``false``, ``inconclusive``); ``skipped_cap``, the cell's web budget,
        the run's cost cap or the cell's deadline left it unchecked; or ``lost``, the web step failed before deciding
        it. The last two read ``unverifiable``.
    :ivar call: The call whose answer fixed its verdict, by its place in the grade's calls; None where no verdict
        call was asked (skipped, or no window read), and the truth call stands.
    """

    outcome: WebOutcome
    call: int | None = None


def graded_claims(
    claims: Sequence[ExtractedClaim],
    verification: Verification,
    *,
    calls_before: int,
    web: Mapping[int, WebGraded] | None = None,
) -> tuple[GradedClaim, ...]:
    """Each claim as the grade stores it, its verdict read from its case citations, its suspicion and, for a
    nominated claim, what the web check found (:func:`truth_verdict`).

    A restatement takes its original's web outcome, as it takes its citations: it states the same fact, and the web
    check is made once, for the original. With no web outcome (web off), a nominated claim reads ``unverifiable``.

    :param claims: The cell's claims, as listed.
    :param verification: What :meth:`ClaimVerifier.verify` found for them.
    :param calls_before: How many of the grade's calls precede the verify step's (the extraction's), so each
        ``decided_by`` is a place in the grade's calls.
    :param web: What the web check left, by the position of each nominated claim that restates nothing; None or
        empty when the criterion checks nothing on the web.
    :return: One graded claim per claim, in order.
    :raises ValueError: The verification holds a finding for another number of claims; or a web outcome is given for
        a session claim, a restatement, or a claim nobody nominated.
    """
    if len(verification.findings) != len(claims):
        raise ValueError(
            f"a verification holds {len(verification.findings)} findings, and there are {len(claims)} claims"
        )
    outcomes = dict(web or {})
    if stray := sorted(
        position
        for position in outcomes
        if position >= len(claims) or claims[position].restates is not None or verification.findings[position] is None
    ):
        raise ValueError(
            f"web outcomes are given for claims {stray}, which are no world claims that restate nothing: a "
            "restatement takes its original's"
        )
    graded: list[GradedClaim] = []
    for position, (claim, finding) in enumerate(zip(claims, verification.findings, strict=True)):
        # The extractor's fields are the stored claim's own: where it stands, the proposition and its tags.
        base: dict[str, Any] = claim.model_dump()
        if finding is None:
            graded.append(GradedClaim.model_validate(base))
            continue
        checked = outcomes.get(position if claim.restates is None else claim.restates)
        verdict = truth_verdict(
            finding.truth_citations,
            suspect=finding.suspect,
            web=None if checked is None else checked.outcome,
            outranks=verification.outranks,
        )
        decided = calls_before + finding.truth_call if checked is None or checked.call is None else checked.call
        graded.append(
            GradedClaim.model_validate(
                {
                    **base,
                    "truth_citations": finding.truth_citations,
                    "ground_citations": finding.ground_citations,
                    "suspect": finding.suspect,
                    "nominated_by": verdict.nominated_by,
                    "verdict": verdict.verdict,
                    "graded_against": verdict.graded_against,
                    "grounded": finding.grounded,
                    "evidence_conflict": verdict.evidence_conflict,
                    "web": "none"
                    if checked is None
                    else checked.outcome
                    if checked.outcome in {"skipped_cap", "lost"}
                    else "checked",
                    "decided_by": decided,
                }
            )
        )
    return tuple(graded)


def redecided_claims(claims: Sequence[GradedClaim], web: Mapping[int, WebGraded]) -> tuple[GradedClaim, ...]:
    """A stored grade's claims with the web outcomes of a web step finished again: each claim in ``web``, and each
    restatement of it, decided anew by the verdict table (:func:`truth_verdict`) from its stored case citations and
    suspicion. Every other claim, and everything else about these, stands as stored.

    :param claims: The grade's claims, as stored.
    :param web: What the web step left, by the position of each claim it checked again; each one the grade left
        ``lost``, nominated, of the world and restating nothing.
    :return: One graded claim per claim, in order.
    :raises ValueError: A web outcome is given for a claim the stored grade did not leave lost, or one that restates.
    """
    if stray := sorted(
        position
        for position in web
        if position >= len(claims) or claims[position].web != "lost" or claims[position].restates is not None
    ):
        raise ValueError(
            f"web outcomes are given for claims {stray}, which the stored grade did not leave lost, or which restate "
            "another"
        )
    redone: list[GradedClaim] = []
    for position, claim in enumerate(claims):
        checked = web.get(position if claim.restates is None else claim.restates)
        if checked is None:
            redone.append(claim)
            continue
        verdict = truth_verdict(claim.truth_citations, suspect=claim.suspect, web=checked.outcome)
        redone.append(
            GradedClaim.model_validate(
                {
                    **claim.model_dump(),
                    "nominated_by": verdict.nominated_by,
                    "verdict": verdict.verdict,
                    "graded_against": verdict.graded_against,
                    "evidence_conflict": verdict.evidence_conflict,
                    "web": checked.outcome if checked.outcome in {"skipped_cap", "lost"} else "checked",
                    # A claim no verdict call decided keeps the call it was decided by before: its truth call.
                    "decided_by": claim.decided_by if checked.call is None else checked.call,
                }
            )
        )
    return tuple(redone)


@dataclass(frozen=True)
class _Passage:
    """One numbered passage: a stretch of a document, by offsets in its text."""

    document: EvidenceDocument
    start: int
    end: int


@dataclass(frozen=True)
class _Shown:
    """What one step's calls are shown: its numbered facts and passages, rendered once, and which case documents
    override which (``precedence``, by document id, following the chain), shown beneath the passages."""

    facts: tuple[tuple[str, str, ReferenceFact], ...]
    passages: tuple[tuple[str, _Passage], ...]
    precedence: Mapping[str, frozenset[str]] = MappingProxyType({})

    @property
    def refs(self) -> frozenset[str]:
        return frozenset([ref for ref, _, _ in self.facts] + [ref for ref, _ in self.passages])

    def passage(self, ref: str) -> _Passage | None:
        return next((passage for shown, passage in self.passages if shown == ref), None)

    def quotable(self, ref: str) -> tuple[str, QuotedKind]:
        """The text a citation of ``ref`` quotes, and what it is: a key fact's line as shown after its ref, or a
        passage's text."""
        fact = next(((entity, fact) for shown, entity, fact in self.facts if shown == ref), None)
        if fact is not None:
            return _fact_line(*fact), "key fact"
        passage = self.passage(ref)
        if passage is None:
            raise KeyError(ref)
        return passage.document.text[passage.start : passage.end], "passage"

    def numbered(self) -> str:
        """The facts and passages as a verify prompt numbers them, one block each: evidence, not a tool's result."""
        lines = [_FACT_TEMPLATE.format(ref=ref, fact=_fact_line(entity, fact)) for ref, entity, fact in self.facts]
        lines += [
            _PASSAGE_TEMPLATE.format(
                ref=ref,
                title=json.dumps(passage.document.title, ensure_ascii=False),
                start=passage.start,
                end=passage.end,
                text=passage.document.text[passage.start : passage.end],
            )
            for ref, passage in self.passages
        ]
        if precedence := self._precedence_lines():
            lines.append("\n".join([_PRECEDENCE_HEADING, *precedence]))
        return "\n\n".join(lines) or _NO_EVIDENCE

    def _precedence_lines(self) -> list[str]:
        """One line per shown document that overrides a shown other, in the order their passages were numbered."""
        refs: dict[str, list[str]] = {}
        titles: dict[str, str] = {}
        for ref, passage in self.passages:
            refs.setdefault(passage.document.id, []).append(ref)
            titles[passage.document.id] = passage.document.title

        def named(document: str) -> str:
            title = json.dumps(titles[document], ensure_ascii=False)
            return _PRECEDENCE_DOCUMENT.format(title=title, refs=", ".join(refs[document]))

        lines: list[str] = []
        for document in refs:
            overridden = [other for other in refs if other in self.precedence.get(document, frozenset())]
            if overridden:
                lines.append(
                    _PRECEDENCE_LINE.format(
                        overriding=named(document), overridden=", ".join(named(other) for other in overridden)
                    )
                )
        return lines

    def outranks(self) -> Mapping[str, frozenset[str]]:
        """Each shown passage's ref, and the refs of the shown passages whose documents its document overrides."""
        outranked: dict[str, frozenset[str]] = {}
        for ref, passage in self.passages:
            overridden = self.precedence.get(passage.document.id, frozenset())
            if below := frozenset(other for other, shown in self.passages if shown.document.id in overridden):
                outranked[ref] = below
        return MappingProxyType(outranked)


def anchors_of(texts: Iterable[str]) -> frozenset[str]:
    """The folded words of ``texts`` that anchor a passage: at least :data:`_ANCHOR_MIN_CHARS` long, or holding a
    digit. A retrieval key, never a check of what any text says."""
    words: set[str] = set()
    for text in texts:
        folded = VERBATIM_FOLD.fold(text)[0]
        words.update(
            word for word in _WORD.findall(folded) if len(word) >= _ANCHOR_MIN_CHARS or any(c.isdigit() for c in word)
        )
    return frozenset(words)


def select_passages(
    documents: Sequence[EvidenceDocument], anchors: frozenset[str], budget: PageBudget
) -> tuple[tuple[_Passage, ...], tuple[str, ...]]:
    """The passages one step reads under ``budget``, and the documents it did not read whole (the selection rule,
    :data:`_SELECTION_RULE`).

    :param documents: The step's documents, in order.
    :param anchors: The folded words a passage is ranked by.
    :param budget: The criterion's page budget.
    :return: The passages in document order and, within one, by offset; and the ids of the documents truncated.
    """
    spans = select_spans([document.text for document in documents], anchors, budget)
    passages: list[_Passage] = []
    truncated: list[str] = []
    for document, joined in zip(documents, spans, strict=True):
        passages.extend(_Passage(document, start, end) for start, end in joined)
        if sum(end - start for start, end in joined) < len(document.text):
            truncated.append(document.id)
    return tuple(passages), tuple(truncated)


def select_spans(
    texts: Sequence[str], anchors: frozenset[str], budget: PageBudget
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The stretches of each of ``texts`` one step reads under ``budget``, by the selection rule
    (:data:`_SELECTION_RULE`): what :func:`select_passages` cuts its passages from, and the web check its windows.

    :param texts: The texts, in order.
    :param anchors: The folded words a tile is ranked by.
    :param budget: How much is read: of one text, of all of them, and in tiles of what length.
    :return: For each text, its stretches as ``(start, end)`` offsets, adjacent tiles joined, in offset order.
    """
    width = budget.window_chars
    ranked: list[list[tuple[int, int]]] = []
    for text in texts:
        tiles = [(start, min(start + width, len(text))) for start in range(0, len(text), width)]
        held = [anchors_of([text[start:end]]) & anchors for start, end in tiles]
        spread = Counter(word for words in held for word in words)
        weight = [sum((Fraction(1, spread[word]) for word in words), Fraction(0)) for words in held]
        order = sorted(range(len(tiles)), key=lambda at: (-weight[at], at))
        ranked.append([tiles[at] for at in order])
    taken: list[list[tuple[int, int]]] = [[] for _ in texts]
    document_left = [budget.per_document_chars] * len(texts)
    total_left = budget.total_chars
    for rank in range(max((len(tiles) for tiles in ranked), default=0)):
        for place, tiles in enumerate(ranked):
            if rank >= len(tiles):
                continue
            start, end = tiles[rank]
            if end - start <= min(document_left[place], total_left):
                taken[place].append((start, end))
                document_left[place] -= end - start
                total_left -= end - start
    selected: list[tuple[tuple[int, int], ...]] = []
    for spans in taken:
        joined: list[list[int]] = []
        for start, end in sorted(spans):
            if joined and joined[-1][1] == start:
                joined[-1][1] = end
            else:
                joined.append([start, end])
        selected.append(tuple((start, end) for start, end in joined))
    return tuple(selected)


def _batches(claims: Sequence[ExtractedClaim]) -> list[tuple[int, ...]]:
    """The world claims asked, by position, one batch per unit in unit order and one for claims outside any unit.

    A restatement is not asked: it takes its original's answer.
    """
    units: dict[int, list[int]] = {}
    outside: list[int] = []
    for position, claim in enumerate(claims):
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        if claim.scope != "world" or claim.restates is not None:
            continue
        if claim.unit is None:
            outside.append(position)
        else:
            units.setdefault(claim.unit, []).append(position)
    batches = [tuple(units[unit]) for unit in sorted(units)]
    return [*batches, tuple(outside)] if outside else batches


def _unreadable(refused: ValueError) -> str:
    """Why a reply could not be read, in a line a model can act on."""
    if isinstance(refused, ValidationError):
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or 'the reply'}: {error['msg']}"
            for error in refused.errors()[:5]
        )
        return _FAULTS["shape"].format(problems=problems)
    return str(refused)


@dataclass
class _Checked:
    """One claim's answer under one step: the citations that passed, what is wrong with the rest, its call, and the
    evidence it was checked against, which a repair of it is shown."""

    answer: _Answer
    call: int
    citations: tuple[ClaimCitation, ...]
    fault: str | None
    shown: _Shown


@dataclass(frozen=True)
class ClaimVerifier:
    """A model step that checks a cell's world claims against numbered evidence, in batches, and repairs what fails.

    :ivar client: The completion client the step calls. It must name its model (``model_name``).
    :ivar pages: How much of the documents each step reads.
    :ivar repairs: How often, and how many items, a cell may ask again.
    :ivar grounded_in: The provenances of the documents the candidate itself was given, whose support grounds a claim;
        None asks no ground step.
    :ivar guidance: The host's own notes on reading claims in its domain, appended to every prompt. Empty by default.
    """

    client: CompletionClient
    pages: PageBudget
    repairs: RepairBudget
    grounded_in: tuple[EvidenceProvenance, ...] | None = None
    guidance: str = ""

    def __post_init__(self) -> None:
        """Refuse a client that names no model, or an empty ``grounded_in``.

        :raises ValueError: One of those.
        """
        model = getattr(self.client, "model_name", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"a claim verifier's client names its model (model_name), and {type(self.client).__name__} names "
                "none: the grade records which model checked the claims, so an unnamed one would pool with any other"
            )
        if self.grounded_in is not None and not self.grounded_in:
            raise ValueError("grounded_in names at least one provenance; None asks no ground step")

    @property
    def model_name(self) -> str:
        """The model the client calls."""
        # Checked present and a non-blank string at construction.
        return str(getattr(self.client, "model_name", ""))

    @property
    def truth_response_format(self) -> dict[str, Any]:
        """The strict directive a truth batch sends: ``{"claims": [{"claim", "citations", "suspect"}]}``."""
        item = {
            "type": "object",
            "properties": {
                "claim": {"type": "integer"},
                "citations": {"type": "array", "items": _TRUTH_CITATION},
                "suspect": {"type": "boolean"},
            },
        }
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_truth", schema)

    @property
    def ground_response_format(self) -> dict[str, Any]:
        """The strict directive a ground batch sends: ``{"claims": [{"claim", "citations"}]}``, supports only."""
        item = {
            "type": "object",
            "properties": {
                "claim": {"type": "integer"},
                "citations": {"type": "array", "items": _GROUND_CITATION},
            },
        }
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_ground", schema)

    @property
    def truth_repair_response_format(self) -> dict[str, Any]:
        """The strict directive a truth repair sends: one claim's ``citations`` and ``suspect``."""
        schema = {
            "type": "object",
            "properties": {
                "citations": {"type": "array", "items": _TRUTH_CITATION},
                "suspect": {"type": "boolean"},
            },
        }
        return strict_response_format("claim_truth_repair", schema)

    @property
    def ground_repair_response_format(self) -> dict[str, Any]:
        """The strict directive a ground repair sends: one claim's ``citations``."""
        schema = {"type": "object", "properties": {"citations": {"type": "array", "items": _GROUND_CITATION}}}
        return strict_response_format("claim_ground_repair", schema)

    @property
    def identity(self) -> str:
        """``<model>|claimverify:<sha12>``: the model, and :attr:`definition`'s first twelve characters."""
        return f"{self.model_name}|claimverify:{self.definition[:12]}"

    @property
    def definition(self) -> str:
        """A digest of everything but the model that shapes a verdict: the prompts (with the host's guidance), the
        reply schemas, the failures a repair is told, the sentence-break rule, the page budget and selection rule, the
        batch rule, the re-ask rule, the repair budget, ``grounded_in``, the quote cap, the attempts and the verbatim fold's identity.
        What a criterion's fingerprint reads, since which model checked the claims is recorded on each call."""
        return canonical_digest(
            {
                "truth_system": self._system("truth"),
                "ground_system": self._system("ground"),
                "truth_repair_system": self._repair_system("truth"),
                "reask_system": self._reask_system(),
                "ground_repair_system": self._repair_system("ground"),
                "templates": [
                    _USER_TEMPLATE,
                    _FACT_TEMPLATE,
                    _FACT_LINE,
                    _REASK_TEMPLATE,
                    _REASK_REFS,
                    _PASSAGE_TEMPLATE,
                    _NO_EVIDENCE,
                    _PRECEDENCE_HEADING,
                    _PRECEDENCE_LINE,
                    _PRECEDENCE_DOCUMENT,
                    _CLAIM_TEMPLATE,
                    _RESTATED_TEMPLATE,
                    _AGAIN_TEMPLATE,
                    _REPAIR_TEMPLATE,
                ],
                "schemas": [
                    self.truth_response_format,
                    self.ground_response_format,
                    self.truth_repair_response_format,
                    self.ground_repair_response_format,
                    self._reask_response_format(),
                ],
                "pages": self.pages.model_dump(mode="json"),
                "faults": dict(_FAULTS),
                "sentence_break": _sentence_break_definition(),
                "selection": [_SELECTION_RULE, _ANCHOR_MIN_CHARS, _WORD.pattern],
                "batches": _BATCH_RULE,
                "reask": _REASK_RULE,
                "repairs": self.repairs.model_dump(mode="json"),
                "grounded_in": None if self.grounded_in is None else sorted(set(self.grounded_in)),
                "quote_chars": CLAIM_QUOTE_CHARS,
                "attempts": _ATTEMPTS,
                "fold": VERBATIM_FOLD.identity,
            }
        )

    def price_ceiling(
        self,
        segments: Sequence[Segment],
        *,
        key: ReferenceKey | None,
        documents: Sequence[EvidenceDocument] | None,
    ) -> float | None:
        """The most verifying the claims ``segments`` state can cost on this verifier's client, before they are listed.

        Each step on its own: every batch the segments can give (one per unit, one for the segments outside any unit),
        each asked in full up to its attempts, at the step's evidence bound and the echo allowance for the claims of
        that batch's segments (:func:`~threetears.evals.kernel.claims.extraction.echo_chars`); and the most items the
        repair budget lets the step ask again, each at the evidence bound and the allowance for one claim, its answer
        and its failure. Wherever the truth step can be shown evidence (a key fact or a case document), its one re-ask
        too, asked in full up to its attempts, at the truth step's evidence bound, since a doubted claim is shown key
        facts and a conflict the key facts and case passages it cited, and at twice the cell's echo allowance, for its
        claims and the refs beneath each; its failing items are among the truth step's repairs. The evidence bound is
        the key's facts as shown, and, for the passages, every character the page budget lets the step read of its
        documents with each passage's heading. The ground step is priced only where it is asked: ``grounded_in``
        named, and a document of those provenances given.

        :param segments: The cell's segments.
        :param key: The case's reference key, or None.
        :param documents: The cell's evidence documents, or None when undeclared.
        :return: The ceiling in dollars, or None when the client cannot price a call.
        """
        cell = tuple(segments)
        given = tuple(documents or ())
        facts = [] if key is None else [(entity.name, fact) for entity in key.fact_entities for fact in entity.facts]
        case = [document for document in given if document.provenance == "case"]
        steps: dict[_Step, str] = {
            "truth": self._evidence_bound(
                [
                    _FACT_TEMPLATE.format(ref=f"K{at}", fact=_fact_line(entity, fact))
                    for at, (entity, fact) in enumerate(facts, start=1)
                ],
                case,
                document_precedence(given),
            )
        }
        grounding = [document for document in given if self.grounded_in and document.provenance in self.grounded_in]
        if grounding:
            steps["ground"] = self._evidence_bound([], grounding)
        units: dict[int | None, list[str]] = {}
        for segment in cell:
            units.setdefault(segment.unit, []).append(segment.text)
        longest = max((segment.text for segment in cell), key=len, default="")
        echo = filler(echo_chars())
        total = 0.0
        for step, evidence in steps.items():
            for texts in units.values():
                batch = call_ceiling(
                    self.client,
                    system=self._system(step),
                    user=_AGAIN_TEMPLATE.format(
                        user=_USER_TEMPLATE.format(evidence=evidence, claims=filler(echo_chars(*texts))),
                        failure=echo,
                    ),
                    response_format=self.truth_response_format if step == "truth" else self.ground_response_format,
                    attempts=_ATTEMPTS,
                )
                if batch is None:
                    return None
                total += batch
            repair = call_ceiling(
                self.client,
                system=self._repair_system(step),
                user=_REPAIR_TEMPLATE.format(
                    evidence=evidence, claim=filler(echo_chars(longest)), answer=echo, failure=echo
                ),
                response_format=(
                    self.truth_repair_response_format if step == "truth" else self.ground_repair_response_format
                ),
            )
            if repair is None:
                return None
            total += repair * self.repairs.max_items_per_cell
        if facts or case:
            # The re-ask: one call per cell, asked in full up to its attempts, shown at most the truth step's evidence
            # (a doubted claim's key facts, a conflict's cited facts and passages), its claims and the refs beneath
            # each within twice the cell's echo allowance. Its failing items are repaired within the truth step's
            # items, priced above at the whole evidence.
            reask = call_ceiling(
                self.client,
                system=self._reask_system(),
                user=_AGAIN_TEMPLATE.format(
                    user=_REASK_TEMPLATE.format(
                        evidence=steps["truth"],
                        claims=filler(2 * echo_chars(*(segment.text for segment in cell))),
                    ),
                    failure=echo,
                ),
                response_format=self._reask_response_format(),
                attempts=_ATTEMPTS,
            )
            if reask is None:
                return None
            total += reask
        return total

    def _evidence_bound(
        self,
        facts: Sequence[str],
        documents: Sequence[EvidenceDocument],
        precedence: Mapping[str, frozenset[str]] | None = None,
    ) -> str:
        """Text at least as long, in UTF-8 bytes, as any evidence one step can be shown: ``facts`` as rendered, for
        each document every character the page budget lets the step read, with the heading of each passage those
        characters could be cut into, and, where a document overrides another (``precedence``), the precedence block
        naming every passage of each."""
        budget = self.pages
        passages = 0
        heading = 0
        named: dict[str, int] = {}
        for document in documents:
            readable = min(len(document.text), budget.per_document_chars)
            # Tiles are window_chars long but for a document's last; a step takes whole tiles within its share.
            count = min(-(-len(document.text) // budget.window_chars), readable // budget.window_chars + 1)
            title = json.dumps(document.title, ensure_ascii=False)
            one = _PASSAGE_TEMPLATE.format(
                ref=f"S{_WIDEST_REF}", title=title, start=len(document.text), end=len(document.text), text=""
            )
            passages += min(len(document.text.encode("utf-8")), 4 * readable)
            heading += count * (len(one.encode("utf-8")) + 2)
            refs = ", ".join([f"S{_WIDEST_REF}"] * count)
            named[document.id] = len(_PRECEDENCE_DOCUMENT.format(title=title, refs=refs).encode("utf-8")) + 2
        passages = min(passages, 4 * budget.total_chars)
        block = 0
        for overriding, overridden in (precedence or {}).items():
            if overriding in named:
                block += len(_PRECEDENCE_LINE.encode("utf-8")) + named[overriding] + 1
                block += sum(named.get(other, 0) for other in overridden)
        if block:
            block += len(_PRECEDENCE_HEADING.encode("utf-8")) + 2
        return "\n\n".join([*facts, "x" * (passages + heading + block)])

    def _with_guidance(self, prompt: str) -> str:
        return prompt + _GUIDANCE_HEADING + self.guidance.strip() if self.guidance.strip() else prompt

    def _system(self, step: _Step) -> str:
        return self._with_guidance(_TRUTH_SYSTEM if step == "truth" else _GROUND_SYSTEM)

    def _repair_system(self, step: _Step) -> str:
        return self._with_guidance(_TRUTH_REPAIR_SYSTEM if step == "truth" else _GROUND_REPAIR_SYSTEM)

    def _reask_system(self) -> str:
        return self._with_guidance(_REASK_SYSTEM)

    @staticmethod
    def _reask_response_format() -> dict[str, Any]:
        """The strict directive the re-ask sends: ``{"claims": [{"claim", "citations"}]}``, with no suspicion."""
        item = {
            "type": "object",
            "properties": {"claim": {"type": "integer"}, "citations": {"type": "array", "items": _TRUTH_CITATION}},
        }
        return strict_response_format(
            "claim_reask", {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        )

    async def verify(
        self,
        claims: Sequence[ExtractedClaim],
        *,
        key: ReferenceKey | None,
        documents: Sequence[EvidenceDocument] | None,
        ledger: RoleUsageLedger,
        truth_done: Callable[[Verification], None] | None = None,
    ) -> Verification:
        """Check every world claim in ``claims`` against the case's evidence, and against the candidate's own
        documents where ``grounded_in`` names them.

        No world claim makes no call. What the client itself raises in a truth call (a provider failure, a recording
        that holds no answer) is not caught: it is the caller's to classify.

        :param claims: The cell's claims, as the extractor listed them: each ``restates`` names an earlier claim of the
            same scope that restates nothing.
        :param key: The case's reference key, whose facts are numbered ``K1..``; None when the case has none.
        :param documents: The cell's evidence documents, in order: ``()`` when the candidate was given none, None when
            undeclared, which under ``grounded_in`` asks no ground step and loses it (``documents_undeclared``).
        :param ledger: The cell's ``judge`` ledger.
        :param truth_done: Told, once the truth step has finished and before the ground step starts, what truth
            found, with no claim grounded: so a caller whose deadline strikes in the ground step, whose length grows
            with what the arm fetched, keeps truth. Not told when no ground step is asked.
        :return: The evidence shown, the documents truncated, the calls made, one finding per claim, and the ground
            step's loss when it was lost.
        :raises ValueError: ``ledger`` is not the ``judge`` role's; a ``restates`` names no earlier original of its
            scope; or a document's override names no case document given, or overrides run in a cycle.
        :raises ClaimGradingFailed: A truth reply was cut short, a truth batch's reply did not parse in its attempts, or
            a claim still failed its truth checks within the repair budget. The same in the ground step never raises:
            it is :attr:`Verification.ground_loss`, as is a ground call the provider failed or whose completion breaks
            the protocol.
        :raises ApparatusError: The rig could not serve a call (a replay's miss), in either step.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"a claim verifier's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        # Nothing says what the candidate was given, so no ground step can be asked: groundedness alone is lost, and
        # truth, which reads case evidence alone, is checked as for any cell.
        undeclared = self.grounded_in is not None and documents is None
        cell = tuple(claims)
        for position, claim in enumerate(cell):
            original = claim.restates
            if original is not None and not (
                original < position and cell[original].scope == claim.scope and cell[original].restates is None
            ):
                raise ValueError(
                    f"claim {position} restates claim {original}, which is no earlier original of its scope"
                )
        batches = _batches(cell)
        if not batches:
            return Verification((), (), (), tuple(None for _ in cell), *_undeclared(undeclared))
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        world = [claim for claim in cell if claim.scope == "world"]
        facts = [] if key is None else [(entity.name, fact) for entity in key.fact_entities for fact in entity.facts]
        names = [] if key is None else [name for entity in key.fact_entities for name in (entity.name, *entity.aliases)]
        called: dict[str, set[tuple[str, ...]]] = {}
        for entity in () if key is None else key.fact_entities:
            called.setdefault(entity.name, set()).update(_words(name) for name in (entity.name, *entity.aliases))
        anchors = anchors_of([*names, *(claim.claim for claim in world), *(claim.excerpt for claim in world)])
        given = tuple(documents or ())
        truth_passages, truth_cut = select_passages(
            [document for document in given if document.provenance == "case"], anchors, self.pages
        )
        grounding = (
            [document for document in given if document.provenance in self.grounded_in]
            if self.grounded_in is not None
            else []
        )
        ground_passages, ground_cut = select_passages(grounding, anchors, self.pages) if grounding else ((), ())

        numbered: dict[tuple[str, int, int], str] = {}

        def number(passages: Sequence[_Passage]) -> tuple[tuple[str, _Passage], ...]:
            for passage in passages:
                numbered.setdefault((passage.document.id, passage.start, passage.end), f"S{len(numbered) + 1}")
            return tuple((numbered[(p.document.id, p.start, p.end)], p) for p in passages)

        shown: dict[_Step, _Shown] = {
            "truth": _Shown(
                tuple((f"K{at}", entity, fact) for at, (entity, fact) in enumerate(facts, start=1)),
                number(truth_passages),
                document_precedence(given),
            )
        }
        if ground_passages:
            shown["ground"] = _Shown((), number(ground_passages))
        calls: list[ClaimCall] = []
        checked: dict[tuple[_Step, int], _Checked] = {}
        asked = await self._run_step("truth", batches, checked, cell, shown, calls, ledger)
        await self._reask(asked, checked, cell, shown["truth"], called, calls, ledger)
        lost: ClaimGradingFailed | None = None
        if "ground" in shown:
            if truth_done is not None:
                truth_done(
                    Verification(
                        evidence=_index({"truth": shown["truth"]}),
                        documents_truncated=tuple(document.id for document in given if document.id in truth_cut),
                        calls=tuple(calls),
                        findings=_findings(cell, checked, grounded=False),
                        outranks=shown["truth"].outranks(),
                    )
                )
            try:
                await self._run_step("ground", batches, checked, cell, shown, calls, ledger)
            except ClaimGradingFailed as failed:
                # Recorded, never swallowed: the loss rides on the verification for the groundedness reading alone.
                lost = failed
            except ApparatusError:
                # A replay's miss is no grading outcome: it fails the run, naming the call, as every judge call's does.
                raise
            except CompletionProtocolDefect as defect:
                # Recorded, never swallowed: a ground call reads the candidate's own documents, which differ by arm,
                # so how it fails must never cost the cell its truth reading. Named as the host's defect, never as a
                # provider's failure, and logged at error: every arm's ground step meets the same broken client.
                log.error("a claim verifier's ground call's completion breaks the protocol: %s", defect)
                lost = ClaimGradingFailed(
                    f"a ground call's completion breaks the protocol: {defect}", loss="protocol_defect"
                )
            except Exception as failure:  # prawduct:ok-broad-except — provider boundary; a ground call's failure is groundedness's alone
                # NOSILENT: the loss rides on the verification, naming the failure, for the groundedness reading alone.
                contain(failure, log=log, what="a claim verifier's ground call failed", level="error")
                lost = ClaimGradingFailed(
                    f"a ground call failed: {type(failure).__name__}: {failure}", loss="provider_error"
                )
        if undeclared:
            return Verification(
                evidence=_index(shown),
                documents_truncated=tuple(document.id for document in given if document.id in truth_cut),
                calls=tuple(calls),
                findings=_findings(cell, checked, grounded=False),
                ground_loss="documents_undeclared",
                ground_failure=_UNDECLARED,
                outranks=shown["truth"].outranks(),
            )
        return Verification(
            evidence=_index(shown),
            documents_truncated=tuple(document.id for document in given if document.id in {*truth_cut, *ground_cut}),
            calls=tuple(calls),
            findings=_findings(cell, checked, grounded=self.grounded_in is not None and lost is None),
            ground_loss=None if lost is None else lost.loss,
            ground_failure=None if lost is None else str(lost),
            outranks=shown["truth"].outranks(),
        )

    async def _run_step(
        self,
        step: _Step,
        batches: Sequence[tuple[int, ...]],
        checked: dict[tuple[_Step, int], _Checked],
        cell: tuple[ExtractedClaim, ...],
        shown: Mapping[_Step, _Shown],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> int:
        """Ask every batch under ``step``, check each answer, and repair what fails within the step's own budget.

        :return: How many of the step's ``max_items_per_cell`` the repairs spent.
        :raises ClaimGradingFailed: A reply was cut short, a batch could not be used, or a claim still failed.
        """
        for batch in batches:
            answers, call = await self._ask_batch(step, batch, cell, shown[step], calls, ledger)
            for position, answer in answers.items():
                checked[(step, position)] = self._check(answer, call, shown[step])
        return await self._repair(step, checked, cell, calls, ledger)

    async def _reask(
        self,
        asked: int,
        checked: dict[tuple[_Step, int], _Checked],
        cell: tuple[ExtractedClaim, ...],
        shown: _Shown,
        called: Mapping[str, set[tuple[str, ...]]],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> None:
        """Ask the cell's doubted claims and its conflicts again, all in one call, each listed with the refs of its own
        evidence and checked against those alone; repair its failing items within what the truth step's repairs left;
        and read each answer by its kind's rule.

        A doubted claim's standing answer cites nothing and is suspect, and it names a key entity
        (:func:`_names_subject`, over its proposition and its restatements'): it is listed with its own entities'
        facts, and an answer citing evidence replaces the batch's, while one citing nothing leaves the batch's
        standing. A conflict's citations that stand after precedence (:func:`_in_conflict`) both support and
        contradict it: it is listed with exactly what it cited, key facts and case passages, and its answer replaces
        the batch's whatever it cites, while one citing nothing keeps the batch's citations, so the verdict table
        reads it unverifiable with the conflict and never the batch's supported.

        :param asked: How many of the truth step's ``max_items_per_cell`` its repairs spent.
        :param called: Each key entity's name, and the folded words of its name and of each alias.
        :raises ClaimGradingFailed: The reply was cut short or could not be used in its attempts, or a failing item
            could not be repaired within the truth step's repair budget.
        """
        outranks = shown.outranks()
        views: dict[tuple[_Step, int], _Shown] = {}
        conflicts: set[tuple[_Step, int]] = set()
        for item, found in sorted(checked.items()):
            if item[0] != "truth":
                continue
            if _in_conflict(found.citations, outranks):
                cited = {citation.ref for citation in found.citations}
                views[item] = _Shown(
                    tuple(entry for entry in shown.facts if entry[0] in cited),
                    tuple(entry for entry in shown.passages if entry[0] in cited),
                    shown.precedence,
                )
                conflicts.add(item)
                continue
            if found.citations or not found.answer.suspect:
                continue
            said = [_words(claim.claim) for at, claim in enumerate(cell) if item[1] in (at, claim.restates)]
            if facts := tuple(
                entry for entry in shown.facts if any(_names_subject(words, called[entry[1]]) for words in said)
            ):
                views[item] = _Shown(facts, ())
        if not views:
            return
        listed = {ref for view in views.values() for ref in view.refs}
        claims = "\n\n".join(
            _rendered_claim(position, cell)
            + "\n"
            + _REASK_REFS.format(refs=", ".join([ref for ref, _, _ in view.facts] + [ref for ref, _ in view.passages]))
            for (_, position), view in views.items()
        )
        evidence = _Shown(
            tuple(entry for entry in shown.facts if entry[0] in listed),
            tuple(entry for entry in shown.passages if entry[0] in listed),
            shown.precedence,
        )
        answers, call = await self._ask(
            role="truth",
            system=self._reask_system(),
            first=_REASK_TEMPLATE.format(evidence=evidence.numbered(), claims=claims),
            response_format=self._reask_response_format(),
            read=lambda content: _read_reask(content, tuple(position for _, position in views)),
            what=f"re-ask for claims {[position for _, position in views]}",
            calls=calls,
            ledger=ledger,
        )
        reasked = {item: self._check(answers[item[1]], call, view) for item, view in views.items()}
        await self._repair("truth", reasked, cell, calls, ledger, asked=asked)
        for item, found in reasked.items():
            if found.citations:
                checked[item] = found
            elif item in conflicts:
                # A conflict's re-ask that cites nothing never falls back to the batch's answer: the batch's citations
                # stay on record, both sides, and the call that cited nothing decided it, unverifiable.
                before = checked[item]
                checked[item] = _Checked(
                    answer=before.answer, call=found.call, citations=before.citations, fault=None, shown=before.shown
                )
            # A doubted claim's re-ask that cites nothing changes nothing: the batch's answer stands, and the claim
            # goes on to the web.

    async def _complete(
        self,
        *,
        role: Literal["truth", "ground", "repair"],
        system: str,
        user: str,
        response_format: dict[str, Any],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[str, int]:
        """One billed and recorded call's finished reply, and its place among the calls.

        :raises ClaimGradingFailed: The reply was cut short.
        """
        completion = await self.client.generate(system=system, user=user, response_format=response_format)
        ledger.add_llm_result(completion)
        calls.append(claim_call(role, completion))
        cut_short = describe_incomplete_completion(completion)
        if cut_short is not None:
            raise ClaimGradingFailed(f"the claim verifier's {role} reply was cut short: {cut_short}", loss="cut_short")
        return completion.content, len(calls) - 1

    async def _ask_batch(
        self,
        step: _Step,
        batch: tuple[int, ...],
        cell: tuple[ExtractedClaim, ...],
        shown: _Shown,
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[dict[int, _Answer], int]:
        """One batch's answers, by claim position, and the call that gave them, asked again in full, naming why,
        while a finished reply will not parse or does not answer exactly the claims asked.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies could be used.
        """
        return await self._ask(
            role=step,
            system=self._system(step),
            first=_USER_TEMPLATE.format(evidence=shown.numbered(), claims=_rendered_claims(batch, cell)),
            response_format=self.truth_response_format if step == "truth" else self.ground_response_format,
            read=lambda content: _read_batch(step, content, batch),
            what=f"{step} reply for claims {list(batch)}",
            calls=calls,
            ledger=ledger,
        )

    async def _ask(
        self,
        *,
        role: Literal["truth", "ground"],
        system: str,
        first: str,
        response_format: dict[str, Any],
        read: Callable[[str], dict[int, _Answer]],
        what: str,
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[dict[int, _Answer], int]:
        """A call's answers, by claim position, and its place among the calls, asked again in full, naming why, while
        a finished reply will not be ``read``.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies could be used.
        """
        user = first
        refused: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content, call = await self._complete(
                role=role, system=system, user=user, response_format=response_format, calls=calls, ledger=ledger
            )
            try:
                return read(content), call
            except ValueError as unusable:
                refused = unusable
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(unusable))
        raise ClaimGradingFailed(
            f"the claim verifier's {what} could not be used in {_ATTEMPTS} attempts", loss="unreadable"
        ) from refused

    @staticmethod
    def _check(answer: _Answer, call: int, shown: _Shown) -> _Checked:
        """``answer``'s citations, checked: each that passes, resolved to stored offsets, and every failure named."""
        faults: list[str] = []
        citations: list[ClaimCitation] = []
        refs = [cited.ref.strip() for cited in answer.citations]
        if twice := sorted({ref for ref in refs if refs.count(ref) > 1}):
            faults.append(_FAULTS["twice"].format(refs=", ".join(twice)))
        if answer.suspect and answer.citations:
            faults.append(_FAULTS["suspect"])
        for ref, cited in zip(refs, answer.citations, strict=True):
            if ref not in shown.refs:
                faults.append(_FAULTS["unknown_ref"].format(ref=ref))
                continue
            # A key fact's citation is held to the quote a passage's is: its support reads supported before anything
            # else, so it must name, inside the fact, the words that state the value it is cited for.
            citation, fault = _quoted_citation(ref, cited, *shown.quotable(ref))
            if fault is not None:
                faults.append(fault)
            elif citation is not None:
                citations.append(citation)
        return _Checked(
            answer=answer, call=call, citations=tuple(citations), fault="; ".join(faults) or None, shown=shown
        )

    async def _repair(
        self,
        step: _Step,
        checked: dict[tuple[_Step, int], _Checked],
        cell: tuple[ExtractedClaim, ...],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
        *,
        asked: int = 0,
    ) -> int:
        """Ask every claim failing under ``step`` again alone, with its failure named, shown the evidence its answer
        was checked against, within that step's own repair budget; keep each answer that passes.

        :param asked: How many of the step's ``max_items_per_cell`` were spent before.
        :return: How many are spent after.
        :raises ClaimGradingFailed: The next round would ask more items than the step's budget has left, a claim still
            fails after the rounds, or a repair reply was cut short.
        """
        failing = sorted(item for item, found in checked.items() if item[0] == step and found.fault is not None)
        for _round in range(self.repairs.rounds):
            if not failing:
                return asked
            if asked + len(failing) > self.repairs.max_items_per_cell:
                raise ClaimGradingFailed(
                    f"the claim verifier would ask {len(failing)} {step} claim(s) again with {asked} of the step's "
                    f"{self.repairs.max_items_per_cell} repairs spent: {_named(failing, checked)}",
                    loss="repair_budget_spent",
                )
            for item in failing:
                position = item[1]
                asked += 1
                before = checked[item]
                content, call = await self._complete(
                    role="repair",
                    system=self._repair_system(step),
                    user=_REPAIR_TEMPLATE.format(
                        evidence=before.shown.numbered(),
                        claim=_rendered_claim(position, cell),
                        answer=json.dumps(_answer_json(step, before.answer), ensure_ascii=False),
                        failure=before.fault,
                    ),
                    response_format=(
                        self.truth_repair_response_format if step == "truth" else self.ground_repair_response_format
                    ),
                    calls=calls,
                    ledger=ledger,
                )
                try:
                    answer = _read_repair(step, content)
                except ValueError as refused:
                    checked[item] = _Checked(
                        answer=before.answer,
                        call=before.call,
                        citations=before.citations,
                        fault=_FAULTS["correction"].format(failure=before.fault, why=_unreadable(refused)),
                        shown=before.shown,
                    )
                    continue
                checked[item] = self._check(answer, call, before.shown)
            failing = [item for item in failing if checked[item].fault is not None]
        if failing:
            raise ClaimGradingFailed(
                f"the claim verifier could not repair {len(failing)} claim(s) in {self.repairs.rounds} round(s): "
                f"{_named(failing, checked)}",
                loss="repair_budget_spent",
            )
        return asked


def _named(failing: Sequence[tuple[_Step, int]], checked: Mapping[tuple[_Step, int], _Checked]) -> str:
    return "; ".join(f"{step} claim {position}: {checked[(step, position)].fault}" for step, position in failing)


def _place(claim: ExtractedClaim) -> str:
    return "outside any unit" if claim.unit is None else f"in unit {claim.unit}"


def _rendered_claim(position: int, cell: tuple[ExtractedClaim, ...]) -> str:
    """One asked claim, with every restatement of it shown beside it."""
    claim = cell[position]
    lines = [_CLAIM_TEMPLATE.format(position=position, place=_place(claim), claim=claim.claim, excerpt=claim.excerpt)]
    lines += [
        _RESTATED_TEMPLATE.format(position=other, excerpt=again.excerpt)
        for other, again in enumerate(cell)
        if again.restates == position
    ]
    return "\n".join(lines)


def _rendered_claims(batch: tuple[int, ...], cell: tuple[ExtractedClaim, ...]) -> str:
    return "\n\n".join(_rendered_claim(position, cell) for position in batch)


def _answer_json(step: _Step, answer: _Answer) -> dict[str, Any]:
    """An answer as its step's reply shape gives it, to show a repair what it said."""
    if step == "truth":
        return answer.model_dump(mode="json")
    return {"citations": [cited.model_dump(mode="json", exclude={"stance"}) for cited in answer.citations]}


def _read_batch(step: _Step, content: str, batch: tuple[int, ...]) -> dict[int, _Answer]:
    """One batch reply's answers, by claim position.

    :raises ValueError: The reply is not its step's shape, or does not answer each claim asked exactly once.
    """
    payload = extract_json(content)
    if step == "truth":
        answers = [
            (item.claim, _Answer(citations=item.citations, suspect=item.suspect))
            for item in _TruthBatch.model_validate(payload).claims
        ]
    else:
        answers = [
            (item.claim, _Answer(citations=[cited.as_cited() for cited in item.citations], suspect=False))
            for item in _GroundBatch.model_validate(payload).claims
        ]
    _answers_exactly(answers, batch)
    return dict(answers)


def _read_reask(content: str, asked: tuple[int, ...]) -> dict[int, _Answer]:
    """The re-ask reply's answers, by claim position: each claim's citations, and no suspicion.

    :raises ValueError: The reply is not the re-ask's shape, or does not answer each claim asked exactly once.
    """
    answers = [
        (item.claim, _Answer(citations=item.citations, suspect=False))
        for item in _ReaskBatch.model_validate(extract_json(content)).claims
    ]
    _answers_exactly(answers, asked)
    return dict(answers)


def _answers_exactly(answers: Sequence[tuple[int, _Answer]], asked: tuple[int, ...]) -> None:
    """Refuse a reply that does not answer each claim ``asked`` exactly once.

    :raises ValueError: It leaves one out, answers one not asked, or answers one twice.
    """
    named = [position for position, _ in answers]
    if sorted(named) != sorted(asked):
        missing = sorted(set(asked) - set(named))
        extra = sorted({position for position in named if position not in asked or named.count(position) > 1})
        problems = [
            *([_FAULTS["missing"].format(missing=missing)] if missing else []),
            *([_FAULTS["extra"].format(extra=extra)] if extra else []),
        ]
        raise ValueError(_FAULTS["answers"].format(asked=list(asked), problems="; ".join(problems)))


def _read_repair(step: _Step, content: str) -> _Answer:
    """One repair reply's answer.

    :raises ValueError: It is not its step's shape.
    """
    payload = extract_json(content)
    if step == "truth":
        repaired = _TruthRepaired.model_validate(payload)
        return _Answer(citations=repaired.citations, suspect=repaired.suspect)
    ground = _GroundRepaired.model_validate(payload)
    return _Answer(citations=[cited.as_cited() for cited in ground.citations], suspect=False)


def _words(text: str) -> tuple[str, ...]:
    """``text``'s words under the verbatim fold, in order."""
    return tuple(_WORD.findall(VERBATIM_FOLD.fold(text)[0]))


def _names_subject(claimed: tuple[str, ...], names: Iterable[tuple[str, ...]]) -> bool:
    """Whether a key entity names a claim's subject: one of its ``names`` (its name's and each alias's folded words)
    stands in the claim's folded words ``claimed`` as whole words, in order. A reading of names, never of reference
    (the re-ask rule, :data:`_REASK_RULE`)."""
    return any(
        name and any(claimed[at : at + len(name)] == name for at in range(len(claimed) - len(name) + 1))
        for name in names
    )


def _fact_line(entity: str, fact: ReferenceFact) -> str:
    return _FACT_LINE.format(entity=entity, claim=fact.claim)


def _quoted_citation(ref: str, cited: _Cited, shown: str, kind: QuotedKind) -> tuple[ClaimCitation | None, str | None]:
    """A citation checked: its quote a stretch of the text shown under ``ref`` (a passage's, or a key fact's line),
    its subject and value inside that quote and in one sentence or row of it.

    :return: The stored citation, or the failure named.
    """
    checked, fault = check_quote(
        ref, quote=cited.quote, subject=cited.subject, value=cited.value, shown=shown, kind=kind
    )
    if checked is None:
        return None, fault
    return (
        ClaimCitation(
            ref=ref, stance=cited.stance, quote=checked.quote, subject_span=checked.subject, value_span=checked.value
        ),
        None,
    )


@dataclass(frozen=True)
class CheckedQuote:
    """A quote that passed its checks: the shown text's own slice, and where its subject and value stand in it.

    :ivar quote: The slice of the shown text the quote matched, never the model's string.
    :ivar subject: Where the quote names the claim's subject.
    :ivar value: Where it states the value at issue.
    """

    quote: str
    subject: TextSpan
    value: TextSpan


def check_quote(
    ref: str,
    *,
    quote: str | None,
    subject: str | None,
    value: str | None,
    shown: str,
    kind: QuotedKind = "passage",
) -> tuple[CheckedQuote | None, str | None]:
    """Check one quote of a text a call was shown: one unbroken stretch of it, at most
    :data:`~threetears.evals.schema.models.CLAIM_QUOTE_CHARS` characters, with its subject and value inside it and no
    sentence or row break between them. The rule every quote a claim is decided by follows: a passage's, a key fact's
    (its line as the call was shown it after its ref, ``A key fact about <entity>: <fact>``) or a web window's.

    :param ref: How the call cited the text (``S1``, ``K1``, ``W2``), for the failure named.
    :param quote: The quote, as the model wrote it.
    :param subject: The words of it the model says name the claim's subject.
    :param value: The words of it the model says state the value.
    :param shown: The text the call was shown under ``ref``, and nothing more: a passage or window's text, never more
        of its document, or a key fact's line.
    :param kind: What ``shown`` is, as a failure names it: ``passage`` (a web window is named one too) or
        ``key fact``.
    :return: The checked quote, or the failure named, as a repair tells it.
    """
    if quote is None or not quote.strip():
        return None, _FAULTS["no_quote"].format(ref=ref, kind=kind)
    span = verbatim_span(quote, shown)
    if span is None:
        return None, _FAULTS["not_shown"].format(ref=ref, kind=kind)
    held = shown[span[0] : span[1]]
    if len(held) > CLAIM_QUOTE_CHARS:
        return None, _FAULTS["too_long"].format(ref=ref, length=len(held), cap=CLAIM_QUOTE_CHARS)
    spans: dict[str, TextSpan] = {}
    for what, words in (("subject", subject), ("value", value)):
        if words is None or not words.strip():
            return None, _FAULTS["no_span"].format(ref=ref, what=what)
        found = verbatim_span(words, held)
        if found is None:
            return None, _FAULTS["span_outside"].format(ref=ref, what=what)
        spans[what] = TextSpan(start=found[0], end=found[1])
    first, second = sorted((spans["subject"], spans["value"]), key=lambda found: found.start)
    # The quote is the shown text's own, not a model's: where its sentences break is structure, read by position.
    if _breaks_between(held, first.end, second.start):
        return None, _FAULTS["apart"].format(ref=ref)
    return CheckedQuote(quote=held, subject=spans["subject"], value=spans["value"]), None


def _breaks_between(text: str, start: int, end: int) -> bool:
    """Whether a sentence or row break starts in ``text[start:end]``. Each is read in the whole of ``text``, since
    what follows a full stop, and whether it marks an abbreviation, may stand in the subject or value span on either
    side ("Vol. 2", with the value ``2``)."""
    for found in _SENTENCE_END.finditer(text, start):
        if found.start() >= end:
            return False
        if found.group() in "\n\r" or not _ends_no_sentence(text, found.start(), found.end()):
            return True
    return False


def _ends_no_sentence(text: str, at: int, after: int) -> bool:
    """Whether the mark at ``text[at]``, its closing quotes and brackets running to ``after``, ends no sentence.

    A ``!`` ends none when spaces and then ``(`` or a lower-case letter follow ("Wax Trax! (Chicago)", "Wax Trax! in
    the early 1980s"); a ``?`` ends none when spaces and then a lower-case letter follow. A full stop ends none when it
    marks an abbreviation: it follows initials or a dotted acronym ("J. R. R. Tolkien", "the U.S. capital"), or a
    capitalised token of at most :data:`_ABBREVIATION_LETTERS` letters before a capitalised word or a digit ("Dr.
    John", "St. Louis", "Vol. 2", "No. 5"), read past any closing quote or bracket ("(Dr.) John"). Case is read by
    :meth:`str.isupper` and :meth:`str.islower`, so any cased script takes the rule; the shapes themselves are those of
    English and the languages that abbreviate as it does. It accepts a real sentence end of the same shapes too: one
    that ends on a single letter, an acronym or a short capitalised word ("in May. The pamphlet ..."), an exclamation
    or a question before a lower-case word ("Stop! he said"), or an exclamation before a bracketed sentence ("Wow!
    (Her brother moved to York.)")."""
    following = _NEXT_WORD.match(text, after)
    if text[at] != ".":
        if following is None:
            return False
        word = following.group(1)
        return word.islower() or (text[at] == "!" and word == _INSIDE_AFTER_BANG)
    token = _TOKEN_BEFORE.search(text, 0, at)
    if token is None:
        return False
    word = token.group()
    if _INITIALS.fullmatch(word):
        return True
    if not (word.isalpha() and len(word) <= _ABBREVIATION_LETTERS and word[0].isupper() and word[1:].islower()):
        return False
    return following is not None and (following.group(1).isupper() or following.group(1).isdigit())


def _sentence_break_definition() -> list[Any]:
    """The sentence-break rule as an identity reads it: its revision in words, and the patterns and length it uses."""
    return [
        _SENTENCE_RULE,
        _SENTENCE_END.pattern,
        _TOKEN_BEFORE.pattern,
        _INITIALS.pattern,
        _NEXT_WORD.pattern,
        _ABBREVIATION_LETTERS,
        _INSIDE_AFTER_BANG,
    ]


def quote_check_definition() -> dict[str, Any]:
    """Everything :func:`check_quote` holds a quote to, for an identity that reads it: the failures it names, the
    sentence-break rule, the quote cap and the verbatim fold's identity."""
    return {
        "faults": dict(_FAULTS),
        "sentence_break": _sentence_break_definition(),
        "quote_chars": CLAIM_QUOTE_CHARS,
        "fold": VERBATIM_FOLD.identity,
    }


def _index(shown: Mapping[_Step, _Shown]) -> tuple[EvidenceRef, ...]:
    """Every piece of evidence shown, once each: the key facts, then the passages in the order they were numbered."""
    refs: dict[str, EvidenceRef] = {}
    for step in _STEPS:
        if step not in shown:
            continue
        for ref, entity, fact in shown[step].facts:
            refs.setdefault(ref, EvidenceRef(ref=ref, provenance="case", entity=entity, fact=fact))
        for ref, passage in shown[step].passages:
            refs.setdefault(
                ref,
                EvidenceRef(
                    ref=ref,
                    provenance=passage.document.provenance,
                    document=passage.document.id,
                    span=TextSpan(start=passage.start, end=passage.end),
                ),
            )
    return tuple(sorted(refs.values(), key=lambda entry: (entry.ref[0] != "K", int(entry.ref[1:]))))


def _findings(
    cell: tuple[ExtractedClaim, ...], checked: Mapping[tuple[_Step, int], _Checked], *, grounded: bool
) -> tuple[ClaimFinding | None, ...]:
    """One finding per claim: a world original's own, a restatement its original's, a session claim None."""
    findings: list[ClaimFinding | None] = []
    for position, claim in enumerate(cell):
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        if claim.scope != "world":
            findings.append(None)
            continue
        original = position if claim.restates is None else claim.restates
        truth = checked[("truth", original)]
        ground = checked.get(("ground", original)) if grounded else None
        ground_citations = () if ground is None else ground.citations
        findings.append(
            ClaimFinding(
                truth_citations=truth.citations,
                suspect=truth.answer.suspect,
                truth_call=truth.call,
                ground_citations=ground_citations,
                grounded=bool(ground_citations) if grounded else None,
            )
        )
    return tuple(findings)


__all__: list[str] = []
