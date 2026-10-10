# Designing a classifier eval set

**For** anyone building their first eval of a classifier, or any feature whose answer code can grade.
**Answers:** which cases to write, how many, how to feed them to the classifier, and how to read the results.
It assumes no prior experience with evals. To learn the package itself first, do the [tutorial](tutorial.md).

In short: build the set mostly from hard cases (boundaries between labels, lookalikes, contrast pairs and
context), give every case a written reason, put at least ten cases behind each label, and run them through
the same code production uses.

A classifier takes an input and picks one label from a fixed set. Classifiers are among the easier LLM
features to evaluate, because code can grade every answer: the label is either the expected one or it is
not. No judge model is needed, so runs are cheap and grading is exact. They are also easy to evaluate badly.
A set of obvious cases reports high accuracy for almost any model and says nothing about the inputs your
users send that go wrong.

The examples come from these classifiers:

| Classifier | Labels | Input |
|---|---|---|
| **Chat relevance**: should a chat assistant (a bot called Pip, in a gardening club's group chat) treat a message as meant for it? | `DIRECT`, `RELEVANT`, `NONE` | the messages it has not answered yet, plus a summary of what it did recently |
| **Support triage**: which queue does a ticket go to? | `billing`, `bug`, `account`, `other` | the ticket's subject and body |
| **Review sentiment** | `positive`, `negative`, `neutral` | one product review |

## Contents

1. [Write the labels before the cases](#1-write-the-labels-before-the-cases)
2. [Every case carries its reason](#2-every-case-carries-its-reason)
3. [What kinds of cases to write](#3-what-kinds-of-cases-to-write)
4. [How many cases](#4-how-many-cases)
5. [Feed the classifier exactly what production feeds it](#5-feed-the-classifier-exactly-what-production-feeds-it)
6. [Generating more cases](#6-generating-more-cases)
7. [Reading the results](#7-reading-the-results)
8. [Maintaining the set](#8-maintaining-the-set)
9. [Wiring it into 3tears-evals](#9-wiring-it-into-3tears-evals)
10. [Checklist](#checklist)

## 1. Write the labels before the cases

Most disagreements about a case are really disagreements about what a label means. Settle the meaning
first, in writing, and write each definition as a test someone could apply to an input:

- **Make the labels exclusive.** Every input should get exactly one label. If two could both apply, write
  the rule that picks between them. The chat classifier's rule: *does the message refer to Pip at all, by
  name or by asking about something only Pip could answer? If not, it is never `DIRECT`, however much it is
  about gardening.*
- **Cover everything.** Include a label for "none of the above" (`NONE`, `other`). Without one, the model
  has to pick a wrong label for every input you did not plan for, and the eval cannot see it happen.
- **Write rulings for ambiguity.** Some inputs are ambiguous. Decide which way they go, write the ruling
  down, and make it part of the label's definition. "Ambiguous address resolves to `DIRECT`" is a ruling: a
  message that might be meant for the assistant is treated as if it is, because ignoring someone who was
  talking to you is the worse mistake.
- **Decide which mistakes cost more.** For the chat classifier, a false `DIRECT` makes the assistant butt into
  a conversation that was not about it, and a false `NONE` ignores someone who spoke to it. Write down which
  errors matter most *before* you see results, so the results cannot talk you out of it.

Use the same definitions in the classifier's prompt. If the eval's idea of `RELEVANT` and the prompt's idea
of `RELEVANT` differ, you are measuring how well the model guesses what you meant rather than how well it
follows what you wrote.

## 2. Every case carries its reason

A case is three things: the input, the expected label, and **why** that label is right, in one or two
sentences that cite the rule.

```json
{
  "id": "relevance-017",
  "messages": [{"author": "marisol", "text": "what was that you said about the roses?"}],
  "recent_activity": ["Answered a question about pruning roses in late winter."],
  "label": "DIRECT",
  "why": "Names nobody, but asks about the assistant's own earlier answer, which only it gave."
}
```

If you cannot write the reason, the label is a guess. A reviewer checks a reason in seconds, and when a model
gets the case wrong, the reason tells you whether the model failed or the case did.

**Leave out what you have not decided.** A case where reasonable people disagree on the label cannot grade
a model, because whichever answer the model gives, someone thinks it is right. Mark such cases as
unresolved, keep them out of the set, and bring them to whoever owns the label definitions. Each ruling they
make becomes a sentence in a definition and a case in the set.

## 3. What kinds of cases to write

Most of the value of a case set is in its hard cases. Plan the set by kind, and write some of each.

### Plain cases

Inputs whose label nobody would argue with: `"brb walking the dog"` is `NONE`; `"I was charged twice for
March"` is `billing`. A few per label prove the plumbing works, but every model scores well on them.

### Boundary cases

Inputs that sit between two labels and fall on one side because of a specific rule. Write them for **every
pair of labels that can be confused**, not just the pair you think of first.

- Chat, `DIRECT` vs `RELEVANT`: `"pip's advice has got so much better lately"` mentions the assistant but talks
  *about* it to the room, so it is `RELEVANT`. `"pip why did you say to prune the roses in autumn"` asks it about
  its own answer, so it is `DIRECT`.
- Chat, `RELEVANT` vs `NONE`: `"why do my tomato leaves keep curling"` is about gardening and asked of the room,
  so it is `RELEVANT`. `"anyone else's car failing its inspection"` is asked of the same room and is `NONE`.
- Chat, `RELEVANT` vs `DIRECT`, from the other side: a gardening question addressed to another person (`"@jeb do
  you know a good compost supplier…"`) stays `RELEVANT`, because an explicit addressee overrides how on-topic
  it is.
- Support, `bug` vs `account`: "I can't log in" is `account` when the password is wrong and `bug` when the
  login page errors. Write both.

### Lookalikes (confounds)

Inputs with a surface feature that points at the wrong label. A model that learned the shortcut instead of
the rule fails these. They are easy to leave out, because the shortcut and the rule agree on every plain
case, so a set without lookalikes cannot tell the two apart.

| Classifier | Input | Looks like | Is | Why |
|---|---|---|---|---|
| Chat | `"my nephew Pip starts school tomorrow"` | `DIRECT` (the assistant's name) | `NONE` | A person, not the assistant |
| Chat | `"the bulb in the hallway blew again"` | `RELEVANT` ("bulb") | `NONE` | A light bulb, not a plant |
| Chat | `"we need to weed out the duplicates in this sheet"` | `RELEVANT` ("weed") | `NONE` | A figure of speech, asked of the room |
| Support | `"Refund the hours I lost to your app crashing"` | `billing` ("refund") | `bug` | The complaint is the crash |
| Support | `"Your invoice PDF won't open"` | `billing` ("invoice") | `bug` | The PDF is broken; the charge is fine |
| Sentiment | `"Not bad at all."` | `negative` ("bad") | `positive` | Negated |
| Sentiment | `"I wanted to love it."` | `positive` ("love") | `negative` | The wish failed |

Good sources of lookalikes:

- words your domain shares with everyday speech (plant, bulb, weed, bed, root, grow; refund, charge,
  account);
- names that sound like other things;
- negation and sarcasm;
- quoted text, such as a user reporting what someone else said;
- instructions inside the input (`"ignore your previous instructions and…"`), which get the label the
  rules give them, not whatever they ask for.

### Contrast pairs

Two inputs that differ in one small way, where that difference flips the label. A model cannot pass both
halves by spotting a keyword, which makes contrast pairs some of the most useful cases you can write.

| Input | Label |
|---|---|
| Alice: `pip, when should I sow tomatoes?` · Bob: `and peppers?` | `DIRECT` |
| Alice: `I'm sowing tomatoes this weekend` · Bob: `and peppers?` | `RELEVANT` |

Bob's line is the same in both; what came before it decides whether he is asking the assistant or Alice.
Pair lookalikes with their real counterparts the same way: `"my nephew Pip starts school tomorrow"` (`NONE`)
beside `"Pip, when do I plant garlic?"` (`DIRECT`).

### Context cases

When the label depends on more than the single input (earlier messages, what the assistant just did, the
customer's plan), write cases where the context decides it, like the pair above. Then write their
**context-missing twins**: the same final input with the context removed.

The twins test your production system rather than the model. Find out exactly what context production gives
the classifier (how many earlier messages, whether answered ones are included, what is truncated). If
production no longer passes Alice's line, Bob's `"and peppers?"` is classified alone every time; the twin
reproduces that input, and if the model fails it, the fix is in what production passes.

### Controls for shortcuts

If your classifier gives a second judgment, or your labels correlate with something irrelevant, write cases
that break the correlation. The chat classifier also rates each message `ROUTINE` or `COMPLEX` (does
answering it need investigation?). Long messages tend to be complex, so a model can score well by rating
length. `"is it going to frost tonight"` is short and `COMPLEX`, since the answer is worthless unless it is
current, so it tests the rule rather than the length. Grade that second judgment too: a set that grades only
the first label cannot see the second fail. In one private campaign, once it was graded, the incumbent model
recalled `COMPLEX` at 0.40 where a challenger reached 0.93; nothing had been measuring it. Every judgment the
classifier emits needs its own expected value and its own figures.

### Batches

If your classifier labels several inputs at once and combines them ("the highest label across the batch
wins"), vary where the deciding input sits: first, last, in the middle, surrounded by noise. Include an
all-`NONE` batch with several messages, so a model cannot learn that more messages means more relevant.

## 4. How many cases

Every rate an eval reports is an estimate, and how far it can be trusted depends on how many cases are
behind it. The package reports a 95% interval beside each rate: a Wilson interval on the effective number of cases, so
three runs of each case count for more than one case only as far as the repeats disagree with each other. For a model that got 90% right, one run per case:

| Cases behind the rate | Correct | Interval |
|---|---|---|
| 10 | 9 | 60% – 98% |
| 20 | 18 | 70% – 97% |
| 50 | 45 | 79% – 96% |
| 100 | 90 | 83% – 94% |

Ten cases cannot tell a 90% model from a 65% one, which shapes how you size the set:

- **Count per label, not in total.** Precision and recall for a label rest only on the cases with that
  label. A 100-case set with 6 `RELEVANT` cases has an unreadable `RELEVANT` recall. The package reports each
  label's support — how many observations carried it, and over how many cases when each case ran more than
  once (`recall 1 (4/4 over 2 cases)`) — so you can see this. The cases are what the figure rests on.
- **Do not copy your traffic's label mix.** If 90% of real messages are `NONE`, a set that is 90% `NONE` lets
  a model that always answers `NONE` score 90%. Weight the set toward the labels and kinds of case you need
  to measure, and read per-label figures rather than overall accuracy.
- **If you also need a production-predictive number, keep two tiers and never pool them.** A slice weighted
  like real traffic predicts production accuracy; a diagnostic slice over-sampling the rare labels, the
  boundaries and every contrast pair gives per-label recall. One mixed figure is neither: it predicts no
  traffic and still under-measures the rare labels. Keep the tiers as separate case sets, run and read apart
  (a case's [stratum](concepts.md#stratum) stays free for its kind of case). When the set must shrink, cut the
  proportional tier evenly; never cut half a contrast pair, or the question it asks goes unanswered at any size.
- **Repeats do not add cases.** Three runs of six cases are six pieces of evidence, not eighteen: `k` measures
  how consistent the model is on those inputs, and only more cases widen what the rate covers. Below about six
  cases a label's figure is a smoke test.
- **Start with at least ten hand-written cases per label**, most of them boundary cases, lookalikes and
  contrast pairs, then grow the set from production mistakes ([section 8](#8-maintaining-the-set)). The
  package's examples use one to four cases per label so they run in seconds: they show the mechanics, and by
  this rule their figures are smoke tests.

## 5. Feed the classifier exactly what production feeds it

The eval must call the classifier the way your product does: the same prompt, built by the same code, with
the same context, truncated the same way, parsed by the same parser. If the eval builds its own prompt, it
measures a classifier you do not ship.

Your kind's `invoke` should call the production function that assembles the request, not a copy of it. If
production keeps only the last three pending messages, a case with five messages loses two in the eval as
well. Case authors need to know that, so they do not write a case whose deciding message production would
drop.

Decide what an unusable answer is. A model that replies with prose, an empty string or a label outside the
set did not classify. Give that outcome its own predicted label so it shows in the confusion matrix as itself;
mapped to a real label, it would read as an ordinary wrong answer. Use one label per cause, because each has a
different fix: `TRUNCATED` (the reply hit its output cap: raise the cap or bound the reasoning), `UNPARSEABLE`
(it finished but not in the format: fix the prompt or the parser, or choose another model) and `ERROR` (the
call failed: fix the rig or the provider account). In one private campaign, truncations reported as
unparseable answers pointed an operator at the wrong fix for fourteen days.

## 6. Generating more cases

Generation does not reword your hand-written cases. It writes new cases from a template's **variation
axes**, and a generated case is only a combination of axis values, with no input text and no expected label.
A launch with `n_variations=N`:

1. takes each axis's values: an `enum` axis gives every listed value, a `sample` axis draws up to N of its
   listed values, and an `llm` axis asks the model named by `variation_model` for N new ones;
2. forms every combination of one value per axis, shuffles them and keeps N;
3. stores each as a test case whose `variation_params` map each axis to its value, reusing a stored case
   with the same values.

An `llm` axis's writer sees only that axis's name, its description and the values already stored for it: no
case, no other axis, no label. So it cannot paraphrase a case, and its values are paired at random with every
other axis's values.

Your kind's `invoke` turns the params into the classifier's input, and it must derive the expected label from
them too, since the case carries none. That shapes the axes:

- **Make the label an axis**: an `enum` axis listing your labels, from which `invoke` builds an input with
  that label (from a hand-written base message per label, say).
- **Every other axis must leave the label alone, whatever it is paired with**: how the person types
  (`as_typed`, `hurried`, `shouted`), a name, a time of day. An axis whose value can flip the label, such as a
  model-written topic, yields cases whose expected label is wrong.
- **Keep boundary cases, lookalikes, contrast pairs and context cases hand-written.** A small rewording is
  exactly what flips their label.

Read a sample of generated cases as `invoke` renders them before you trust a run over them. An input that does
not have the label `invoke` gave it is a wrong case, and the model that "got it wrong" was right.

Generated cases are priced before they are written and charged outside the runs' own cost caps; see
[Spend outside any run](cost-and-budgets.md#spend-outside-any-run) in Cost and budgets, and
[Generating cases at launch](adopting-a-host.md#generating-cases-at-launch) for the launcher's side.

## 7. Reading the results

**Run each case more than once.** Keep `k` at 2 or more (the default is 3). A classifier at a non-zero temperature can label the
same input differently on different calls. A case that flips between repeats is a finding by itself: the
model is unsure, and users sending that message get different answers.

**Read the confusion matrix, not just accuracy.** Each result lands in one cell, `expected → predicted`, and
the off-diagonal cells show which way the model is wrong. `RELEVANT → DIRECT` (it butts in) and
`DIRECT → NONE` (it ignores someone) may have the same count and very different costs. This is where the
costs you wrote down in [section 1](#1-write-the-labels-before-the-cases) come in.

**Read per-label precision and recall with their intervals.** Recall for `DIRECT`: of the messages meant for
the assistant, how many did it catch? Precision for `DIRECT`: of the messages it treated as meant for it, how
many were? F1 combines the two into one number and has no interval of its own, so read the two it comes from.

**Compare arms on the same cases.** Running two models (or two prompts) over the same case set is what makes
a difference between them meaningful. When their intervals overlap heavily, the eval has not shown a
difference yet. Add cases where the arms disagree, not more cases where both are right.

**Gate accuracy, latency and cost separately.** A cheaper or faster model is admissible only if it clears each
bar on its own. Folded into one score, a large saving can buy back a real loss of accuracy.

**Measure latency on calls that were not competing.** Accuracy and cost do not depend on how many calls run at
once; latency does. By default a run executes several cells at once and a launch's arms side by side, and the
latency they record is marked read under concurrency and never compared. For a latency gate, launch with
`measure_latency=True` (and declare it on the campaign's design): the cells then run one at a time and the
arms one after another, with nothing beside them.

**Read the wrong answers.** Open the cases a model missed and read their reasons. In a new case set, many
"model errors" turn out to be cases that do not follow from the rules, or rules that do not say what was
meant. Fix those first.

### Reading results by kind of case

Overall accuracy across a mixed set is dominated by the easy majority. A model at 97% on plain cases and 60%
on lookalikes reads as about 90% overall if a fifth of the set are lookalikes, and the 60% is the number to act
on. So give each case the kind it is, and read the results per kind.

Set the kind as the test case's **stratum** when you write the case:

```python
EvalTestCase(
    scope_id=scope_id,
    template_id=template.id,
    stratum="lookalike",
    host_payload={"messages": [...], "label": "NONE", "why": "A city, not the assistant."},
)
```

Use the kinds from [section 3](#3-what-kinds-of-cases-to-write) as the names: `plain`, `boundary`, `lookalike`,
`contrast`, `context`. Keep the stratum out of `host_payload` and `variation_params`. The candidate never sees
the stratum, but it does see the case's input, and a case that says it is a lookalike tells the model what to
look out for.

Run the whole set as one run, in a campaign, and read the campaign's report
(`python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev`). Its **By stratum**
table gives each arm's figures over all cases, then one column per stratum, each with the cases it rests on
([Results by kind of case](reading-reports.md#results-by-kind-of-case-strata)). A stratum with fewer than 10
cases is marked "too few cases to read alone": three lookalikes with one right give a recall interval of about
6% to 79%. Add cases to it before acting on it.

For generated cases, mark the variation axis that names the kind with `stratum=True` and each generated case
takes that axis's value as its stratum. `run_eval` cannot set a stratum yet, so strata need cases stored as test
cases, as a classifier kind's are.

## 8. Maintaining the set

- **Review before use.** Have the person who owns the label definitions read the cases, labels and reasons
  before any result is trusted. A case set states what correct behaviour is, so someone accountable for that
  behaviour should agree with it.
- **Turn production mistakes into cases.** Every misclassification a user reports is a case you were
  missing, usually a lookalike or a context case. Add it with its reason, then its contrast partner.
- **Start a new case set rather than editing cases.** A stored case never changes, and a classifier's cases
  cannot be retired in place (the `archived` flag on a test case applies only to the analysis writer's frozen
  cases). When a ruling changes, write the set anew under a new template. That way a case id always means one
  input and one label, and results from before and after the ruling are never read as the same measurement.
  With `run_eval`, changing a case's expected label already makes its runs a different template.
- **Re-check context twins when production changes.** If the classifier's input changes (a longer history, a
  new field), your context-missing twins test something different. Re-read them.

## 9. Wiring it into 3tears-evals

### The quick path: `run_eval`

`run_eval` (the [tutorial](tutorial.md)'s first step) runs a classifier function over a
list of cases in one call. Pass `expected=`, a function that returns the label a case expects, and `run_eval`
grades the function as a classifier:

```python
from threetears.evals.quick import run_eval

async def classify(case: dict) -> str:
    """Call the production classifier on the case and return the label its parser gives."""
    ...

summary = await run_eval(cases, classify, scope_id="dev", expected=lambda case: case["expected"], k=3)
print(summary.render())
```

Each result records whether the answer was the expected label (`match`, reported as `accuracy`) and its cell
of the confusion matrix (`confusion_cell`), the two measures a classifier kind lands. `summary.confusion` is the
matrix and `summary.labels` gives each label's precision and recall with their intervals, and its F1 (a label
never predicted has no precision, one never expected no recall, and neither has an F1); `render()` prints all
of it. A call to `classify` that raises fails its result: it is counted in
`summary.n_candidate_failed` and, as a miss, under `(unusable answer)` in the matrix, so a model that refuses
the hard cases does not read as more accurate. A provider's rate-limit error counts against the model the same
way, so retry transient errors inside `classify`.

An answer that is not a non-blank string (`None`, a blank string, a number) is counted under its own predicted
label, `UNUSABLE_ANSWER` (printed `(unusable answer)`), and never matches: section 5's unusable label, applied
for you. Any other string is a label exactly as written, so a label outside your set shows as itself, and
`"DIRECT "` differs from `"DIRECT"`. Return what your production parser returns and do not clean it up in the
eval, or the eval stops measuring the parser. To split unusable answers by cause (section 5), return a sentinel
per cause, such as `"(truncated)"`; each counts as its own predicted label.

`run_eval` refuses an `expected=` that gives a case a blank, non-string or `UNUSABLE_ANSWER` label. Scorers
still work beside `expected=`, each reported as its own measure. Two runs over the same cases are runs of one
template, and so comparable, only when they expect the same labels.

### A classifier kind

For the full set of classifier readings, write a kind
(["The kind: what you are evaluating"](adopting-a-host.md#the-kind-what-you-are-evaluating) in Adopting the engine).
Its `invoke` calls your production classifier on the test case and lands two core measures on
`CandidateOutput.host_measures`:

```python
from threetears.evals.kernel import CONFUSION_CELL_MEASURE, MATCH_MEASURE, CandidateOutput, confusion_cell

predicted = parsed_label or "UNPARSEABLE"
return CandidateOutput(
    output=[{"label": predicted}],
    host_measures={
        MATCH_MEASURE: predicted == expected,
        CONFUSION_CELL_MEASURE: confusion_cell(expected, predicted),
    },
)
```

From those two, the engine derives `accuracy`, the confusion matrix, each label's support, precision and
recall with their intervals, and F1. Don't land `accuracy` yourself; the runner refuses a kind that does,
and any other core-named key (`cost_usd`, `score`, a `goal_state:` or `classifier:` name) beside these two.

Put the label set in your kind's **spec**, the model a template of that kind declares. It is validated when
the template is written and frozen onto each run. Check each case's expected label against it when the case
is written, so a misspelt label is refused there instead of counting as a miss in every run. Keep each case's
input, expected label and reason in the test case's `host_payload`.

## Checklist

- [ ] Each label has a written, testable definition, and the prompt uses the same text.
- [ ] There is a "none of the above" label.
- [ ] Ambiguous inputs have rulings, and unresolved cases are kept out of the set.
- [ ] Which mistakes cost most was written down before anything ran.
- [ ] Every case has a reason that cites a rule.
- [ ] Boundary cases exist for every pair of labels that can be confused.
- [ ] Lookalikes exist, each paired with a real counterpart.
- [ ] Context cases exist, with context-missing twins that match what production actually passes.
- [ ] Each label has at least ten cases, and per-label figures are read, not just overall accuracy.
- [ ] Each case declares its kind as its stratum, and each kind you act on has at least ten cases.
- [ ] The eval calls the production request builder and parser, and unusable answers have their own label.
- [ ] Generated cases take their label from a label axis, every other axis leaves it alone, and a sample was read.
- [ ] Every judgment the classifier emits is graded, and unusable answers are split by cause.
- [ ] Accuracy, latency and cost each have their own bar, and latency comes from calls that were not competing.
- [ ] `k` is 2 or more, and cases that flip between repeats were looked at.
- [ ] The label owner has reviewed the set.
