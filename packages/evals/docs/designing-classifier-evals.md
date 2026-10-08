# Designing a classifier eval set

A classifier takes an input and picks one label from a fixed set: a support ticket becomes `billing`,
`bug`, `account` or `other`; a review becomes `positive`, `negative` or `neutral`; a chat message is
`DIRECT` (meant for the assistant), `RELEVANT` (about its subject, but not meant for it) or `NONE`. Classifiers
are the easiest LLM feature to evaluate well, because a computer can grade every answer: the label is
either the expected one or it is not. No judge model is needed, so a run is cheap and its numbers are
exact.

They are also easy to evaluate badly. A set of fifty obvious cases will report 98% accuracy for almost any
model and tell you nothing about the inputs your users actually send that go wrong. This guide is about
building a case set that finds those inputs. It assumes no prior experience with evals.

The examples come from three classifiers, used throughout:

| Classifier | Labels | Input |
|---|---|---|
| **Chat relevance** — should an assistant persona (a radio DJ called Kairo, in a community chat room) treat a message as meant for it? | `DIRECT`, `RELEVANT`, `NONE` | the messages it has not answered yet, plus a summary of what it did recently |
| **Support triage** — which queue does a ticket go to? | `billing`, `bug`, `account`, `other` | the ticket's subject and body |
| **Review sentiment** | `positive`, `negative`, `neutral` | one product review |

## 1. Write the labels before the cases

Every disagreement you will have later about a case is really a disagreement about what a label means.
Settle it first, in writing, and write each definition as a test someone could apply to an input:

- **Make the labels exclusive.** Every input should get exactly one label. If two could both apply, write
  the rule that picks between them. The chat classifier's rule: *does the message refer to Kairo at all —
  by name, or by asking about something only Kairo could answer? If not, it is never `DIRECT`, however much it
  is about music.*
- **Cover everything.** Include a label for "none of the above" (`NONE`, `other`). Without one, the model
  is forced to pick a wrong label for every input you did not plan for, and the eval cannot see it happen.
- **Write rulings for ambiguity.** Some inputs are honestly ambiguous. Decide which way they go, write the
  ruling down, and make it part of the label's definition. "Ambiguous address resolves to `DIRECT`" is a
  ruling: a message that might be meant for the assistant is treated as if it is, because ignoring a person
  who was talking to you is the worse mistake.
- **Decide which mistakes cost more.** A support ticket routed to `other` instead of `bug` waits a day; a
  `bug` routed to `billing` gets bounced twice. For the chat classifier, a false `DIRECT` makes the
  assistant butt into a conversation that was not about it, and a false `NONE` ignores someone who spoke to
  it. Write down which errors you care about most *before* you see results, so the results cannot talk you
  out of it.

The label definitions should be the same text the classifier's prompt uses. If the eval's idea of `RELEVANT`
and the prompt's idea of `RELEVANT` differ, you are measuring how well the model guesses what you meant
rather than how well it follows what you wrote.

## 2. Every case carries its reason

A case is three things: the input, the expected label, and **why** that label is right, in one or two
sentences that cite the rule.

```json
{
  "id": "relevance-017",
  "messages": [{"author": "marisol", "text": "what was that last one?"}],
  "recent_activity": ["Announced 'Aphex Twin — Rhubarb' and talked about the fade from the last track."],
  "label": "DIRECT",
  "why": "Names nobody, but asks about the assistant's own output, which only it knows."
}
```

The reason is not decoration. Writing it is how you notice that a case does not follow from the rules (if
you cannot write the reason, the label is a guess). Reading it is how a reviewer checks your work in seconds
rather than re-deriving the label. And when a model gets the case wrong, the reason tells you whether the
model failed or the case did.

**Leave out what you have not decided.** A case where reasonable people disagree on the label cannot grade a
model: whichever answer the model gives, someone thinks it is right. Mark such cases as unresolved, keep
them out of the set, and bring them to whoever owns the label definitions. Each ruling they make becomes a
sentence in a definition and a case in the set.

## 3. What kinds of cases to write

Most of the value of a case set is in its hard cases. Plan the set by kind, and write some of each.

### Plain cases

Inputs whose label nobody would argue with: `"brb walking the dog"` is `NONE`; `"I was charged twice for
March"` is `billing`. You need a few per label to prove the plumbing works and to anchor the scale, but they
are where every model scores well. If most of your set is plain cases, your accuracy number mostly measures
how many plain cases you wrote.

### Boundary cases

Inputs that sit between two labels and fall on one side because of a specific rule. Write them for **every
pair of labels that can be confused**, not just the pair you happen to think of first.

- Chat, `DIRECT` vs `RELEVANT`: `"kairo's taste has gotten so much better since last month"` mentions the
  assistant but talks *about* it to the room, so it is `RELEVANT`; `"kairo why'd you go from ambient straight
  into breakbeat"` asks it about its own choice, so it is `DIRECT`.
- Chat, `RELEVANT` vs `NONE`: a music question to the room is `RELEVANT`; the same question addressed to
  another person (`"@jeb do you know a good live recording of…"`) is still `RELEVANT`, not `DIRECT`, because an
  explicit addressee overrides how on-topic it is.
- Support, `bug` vs `account`: "I can't log in" is `account` when the password is wrong and `bug` when the login
  page errors. Write both.

### Lookalikes (confounds)

Inputs with a surface feature that points at the wrong label. A model that has learned the shortcut instead
of the rule fails these, and nothing else in your set will show it. This is the category new eval authors most
often leave out.

| Classifier | Input | Looks like | Is | Why |
|---|---|---|---|---|
| Chat | `"i always wanted to go to Cairo"` | `DIRECT` (sounds like the assistant's name) | `NONE` | A city, not the assistant |
| Chat | `"this game is dumb"` then `"stop playing"` | `DIRECT` (a request to the DJ) | `NONE` | "Playing" is the game |
| Chat | `"anyone know how to track a package from overseas?"` | `RELEVANT` ("track") | `NONE` | Shipping, not music, asked of the room |
| Support | `"Refund the hours I lost to your app crashing"` | `billing` ("refund") | `bug` | The complaint is the crash |
| Support | `"Your invoice PDF won't open"` | `billing` ("invoice") | `bug` | The PDF is broken; the charge is fine |
| Sentiment | `"Not bad at all."` | `negative` ("bad") | `positive` | Negated |
| Sentiment | `"I wanted to love it."` | `positive` ("love") | `negative` | The wish failed |

Good sources of lookalikes: words your domain shares with everyday speech (track, record, mix, drop, band,
score, play; refund, charge, account), names that sound like other things, negation, sarcasm, quoted text
(a user reporting what someone else said), and instructions inside the input (`"ignore your previous
instructions and…"`) — which get the label the rules give them, not whatever they ask for.

### Contrast pairs

Two inputs that differ in one small way, where that difference flips the label. A contrast pair is the
strongest test you can write, because a model cannot pass both halves by spotting a keyword.

| Input | Label |
|---|---|
| Alice: `this soccer game is awful` · Bob: `seriously can we get some scoring` | `NONE` |
| Alice: `this stuff is awful` · Bob: `seriously can we get some Beatles` | `DIRECT` |

Bob's request is nearly the same in both; only what came before it decides whether it is about the game or a
request to the DJ. Pair lookalikes with their real counterparts the same way: `"anyone know how to track a package from
overseas?"` (`NONE`) beside `"what was the track before this one"` (`DIRECT`).

### Context cases

When the label depends on more than the single input — earlier messages, what the assistant just did, the
customer's plan — write cases where the context is what decides it, like the pair above. Then write their
**context-missing twins**: the same final input with the context removed.

The twin is about your production system, not the model. Find out exactly what context production gives
the classifier: how many earlier messages, whether messages the assistant already answered are included,
what is truncated. If Alice's line arrived before the assistant's last turn and production no longer passes
it to the classifier, then Bob's `"seriously can we get some scoring"` is classified alone, in production,
every time. A twin that reproduces that input shows you what your users will actually get, and a model that
fails it is not the problem: the missing context is. That is a finding about your system worth having before
a user finds it.

### Controls for accidental shortcuts

If your classifier outputs a second judgment, or your labels correlate with something irrelevant, write
cases that break the correlation. The chat classifier also rates each message `ROUTINE` or `COMPLEX` (does
answering it need investigation?). Long messages tend to be complex, so a model can score well by rating
length. `"does anyone know the current exchange rate for yen"` is short, off-topic and `COMPLEX` (the answer is
worthless unless it is current): it tests the rule rather than the length.

### Batches

If your classifier labels several inputs at once and combines them ("the highest label across the batch
wins"), vary where the deciding input sits: first, last, in the middle, surrounded by noise. Include an
all-`NONE` batch with several messages, so a model cannot learn that more messages means more relevant.

## 4. How many cases

Every rate an eval reports is an estimate, and its uncertainty depends on how many cases it rests on. The
package reports a 95% interval beside each rate (a Wilson interval). Some reference points, for a model that
got 90% right:

| Cases behind the rate | Correct | Interval |
|---|---|---|
| 10 | 9 | 60% – 98% |
| 20 | 18 | 70% – 97% |
| 50 | 45 | 79% – 96% |
| 100 | 90 | 83% – 94% |

Ten cases cannot tell a 90% model from a 65% one. So:

- **Count per label, not in total.** Precision and recall for a label rest only on the cases with that label.
  A 100-case set with 6 `RELEVANT` cases has an unreadable `RELEVANT` recall. The package reports each label's
  support (how many cases carried it) so you can see this.
- **Do not copy your traffic's label mix.** If 90% of real messages are `NONE`, a set that is 90% `NONE` lets a
  model that always answers `NONE` score 90%. Weight the set toward the labels and kinds of case you need to
  measure, and read per-label figures rather than overall accuracy.
- **A useful first set is 40 – 80 hand-written cases**, with at least 10 per label and most of them boundary
  cases, lookalikes and contrast pairs. Grow it from production mistakes (section 8).

## 5. Feed the classifier exactly what production feeds it

The eval must call the classifier the way your product does: the same prompt, built by the same code, with
the same context, truncated the same way, parsed by the same parser. If the eval builds its own prompt, it
measures a classifier you do not ship.

Concretely, your kind's `invoke` should call the production function that assembles the request, not a copy
of it. When production keeps only the last three pending messages, a case with five messages should lose two
in the eval as well — and a case author should know that, so they do not write a case whose deciding message
production would have dropped.

Decide what an unusable answer is. A model that replies with prose, an empty string or a label outside the
set did not classify. Give that outcome its own predicted label (for example `UNPARSEABLE`) so it appears in
the confusion matrix as itself, rather than being quietly mapped to a real label where it would read as an
ordinary wrong answer.

## 6. Generating more cases

Hand-written cases are the anchor; generated variations multiply coverage of wording. A template's
**variation axes** say how cases vary, and a launch with `n_variations` asks for that many new cases:

- `enum` — every listed value once (for example, how the person types: `as_typed`, `hurried`, `shouted`);
- `sample` — values drawn from a list;
- `llm` — new values written by a model you name with `variation_model` (for example, paraphrases).

A generated case keeps its seed's expected label, so **every axis must preserve the label**. A paraphrase
prompt should state the label and its definition and tell the writer to keep the meaning; a typing-style axis
changes spelling, never content. Start variations from plain single-input cases with an unarguable label,
never from a contrast pair or a context case, where small rewordings are exactly what flips the label.

Read a sample of what the generator wrote before you trust a run over it. A paraphrase that drifted into
another label is a wrong case, and the model that "got it wrong" was right.

Generated cases are priced before they are written and charged outside the runs' own cost caps; see the
README's "Generating cases at launch".

## 7. Reading the results

**Run each case more than once.** Set `k` to 2 or 3. A classifier at a non-zero temperature can label the same
input differently on different calls, and a case that flips between repeats is a finding by itself: the
model is unsure, and your users get a coin toss.

**Read the confusion matrix, not just accuracy.** Each result lands in one cell, `expected → predicted`. The
off-diagonal cells say which way the model is wrong. `RELEVANT → DIRECT` (it butts in) and `DIRECT → NONE` (it
ignores someone) may have the same count and very different costs; that is why you decided in section 1 which
mistakes matter.

**Read per-label precision and recall with their intervals.** Recall for `DIRECT`: of the messages meant for
the assistant, how many did it catch? Precision for `DIRECT`: of the messages it treated as meant for it, how
many were? F1 combines the two into one number and has no interval of its own; read the two it comes from.

**Compare arms on the same cases.** Running two models (or two prompts) over the same case set is what makes a
difference between them mean something. A difference smaller than the overlap of the intervals is not yet a
difference: add cases where the arms disagree, not more cases where both are right.

**Read the wrong answers themselves.** Open the cases a model missed and read their reasons. Most
"model errors" in a young case set turn out to be cases that do not follow from the rules, or rules that do
not say what was meant. Fix those first.

### Reading results by kind of case

> **Contingent on [#565](https://github.com/pacepace/3tears/issues/565).** The analysis reports results per
> arm and per label, but not yet per kind of case. Once a case can carry a stratum the analysis reads, one run
> over the whole set will report accuracy and the confusion matrix for plain cases, boundary cases,
> lookalikes and context cases separately, which is the view this guide's case categories are designed for.

Until then, keep each kind of case you need to read separately — lookalikes, say — in its own template, so
it runs as its own run and reports its own figures. Overall accuracy across a mixed set is dominated by the
easy majority: a model at 97% on plain cases and 60% on lookalikes reads as roughly 90% overall if a fifth of
the set are lookalikes, and the 60% is the number that matters.

## 8. Keeping the set honest over time

- **Review before use.** Have the person who owns the label definitions read the cases, their labels and
  their reasons before any result is trusted. A case set is a statement of what correct behaviour is; someone
  accountable for that behaviour should agree with it.
- **Turn production mistakes into cases.** Every misclassification a user reports is a case you were missing,
  usually a lookalike or a context case. Add it with its reason, then its contrast partner.
- **Retire cases rather than editing them.** When a ruling changes, archive the cases it affects (a test
  case's `archived` flag) and write new ones, so a case id always means one input and one label and results
  from before and after the ruling are never read as the same measurement.
- **Re-check the context premise when production changes.** If the classifier's input changes (a longer
  history, a new field), your context-missing twins change meaning. Re-read them.

## 9. Wiring it into 3tears-evals

### Rung zero

`run_eval` (README, "Rung zero") runs a classifier function over a list of cases in one call. Each scorer is
a function of the case and the answer returning a bool or a number:

```python
from threetears.evals.quick import run_eval

def correct(case: dict, label: str) -> bool:
    """Whether the label is the expected one."""
    return label == case["expected"]

summary = await run_eval(cases, classify, [correct], scope_id="dev", k=3)
```

That reports accuracy with its interval. It does not yet report a confusion matrix or per-label precision and
recall.

> **Contingent on [#564](https://github.com/pacepace/3tears/issues/564).** Once rung zero can be told a
> function is a classifier (its expected label per case), the same call will land the confusion matrix and
> per-label statistics described in section 7. Until then, those need a classifier kind, below.

### A classifier kind

For the full set of classifier readings, write a kind (README, "A kind is what you are evaluating"). Its
`invoke` calls your production classifier on the test case and lands two core measures on
`CandidateOutput.host_measures`:

```python
from threetears.evals.contracts import (
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    CandidateOutput,
    confusion_cell,
)

predicted = parsed_label or "UNPARSEABLE"
return CandidateOutput(
    output=[{"label": predicted}],
    host_measures={
        MATCH_MEASURE: predicted == expected,
        CONFUSION_CELL_MEASURE: confusion_cell(expected, predicted),
    },
)
```

From those two, the engine derives `accuracy` (do not land it yourself; the runner refuses a kind that does),
the confusion matrix, each label's support, precision and recall with their intervals, and F1. Put the label
set in your kind's **spec**, the model a template of that kind declares: it is validated when the template is
written and frozen onto each run. Check each case's expected label against it when the case is written, so a
misspelt label is refused there rather than counted as a miss in every run. Keep each case's input, expected label
and reason in the test case's `host_payload`.

## Checklist

- [ ] Each label has a written, testable definition, and the prompt uses the same text.
- [ ] There is a "none of the above" label.
- [ ] Ambiguous inputs have rulings, and unresolved cases are kept out of the set.
- [ ] You wrote down which mistakes cost most before running anything.
- [ ] Every case has a reason that cites a rule.
- [ ] Boundary cases exist for every pair of labels that can be confused.
- [ ] Lookalikes, with real counterparts as contrast pairs.
- [ ] Context cases, with context-missing twins matching what production actually passes.
- [ ] At least 10 cases per label; per-label figures read, not just overall accuracy.
- [ ] The eval calls the production request builder and parser; unusable answers have their own label.
- [ ] Variation axes preserve the label, start from plain single-input cases, and a sample was read.
- [ ] `k` of 2 or more; cases that flip between repeats looked at.
- [ ] The label owner has reviewed the set.
