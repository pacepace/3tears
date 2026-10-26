# Story forms

A story form is a way of answering, not a category the eval belongs to. Choose it after investigating, from what the evidence supports. The same campaign design can produce any of them: a control against two candidates can end in a decision, a tradeoff or a diagnosis, depending on what the numbers say. The `story` field names the dominant form so the editor knows what you meant to deliver. It selects nothing.

## Decision

**When:** the evidence supports one call. Usually one option survives the gates and dominates the others, or a declared rule ranks the survivors.

**Headline:** the call as an instruction, with its reason. "Keep the current model. Both candidates cost more per reply and stay in character less often."

**Slots:** recommendation (with the figures that make it), why_not (each rejected option and the one thing that rules it out), watch (the weakest margin, a confound scoped to one arm, a contradicted prior insight), next (the step that could change the call, and which way).

**Lead evidence:** whatever shows the reason for the call. If the call rests on two things (the asked question is answered, and cost decides), each needs its own claim, side by side. One chart alone would hide half the reason.

**Usual failure:** a decision whose deciding axis isn't on the first screen, or a weak axis of the recommendation that appears only in the appendix.

## Tradeoff

**When:** two or more options survive, none dominates, and the campaign declared nothing that ranks the axes, such as an exchange rate or a priority.

**Headline:** the trade, with both sides and where the loss sits. "Switching saves a third of the cost but loses six points of personality, mostly in long conversations."

**Slots:** the_trade, where_it_flips (the price bands where the best option changes), our_lean (labelled yours, with the band it holds for), what_settles_it.

**What it owes:** one axis priced in the other at real volume (`price_tradeoff()`), over the two to four options a decider would actually weigh. Too many variants make the bands too thin to act on. If the loss concentrates in one stratum, a modeled hybrid that avoids it is often the most useful option. Label it modeled, not measured, and state the assumption it rests on. Show any evidence of what users notice. If there is none, say so in Watch. Say plainly that nothing was declared in advance, and name who makes the call.

**Usual failure:** deciding anyway. A lean presented as a recommendation hides the fact that the call is a business choice.

## Diagnosis

**When:** the result can't be read cleanly because two things are tangled. A model changed together with its reasoning budget. The judge is unreliable exactly where the arms differ. A short run touches one arm.

**Headline:** what can't be concluded and why, plus what still holds. "The candidate's drop can't be credited to the model: its reasoning budget changed too. It still costs more."

**Slots:** what_is_tangled, known_despite_it, what_settles_it, next.

**Lead evidence:** the tangle itself, shown so a reader can see why it blocks the reading. For example, the confound's size beside the difference it could explain (`confound_check()`).

**Usual failure:** a hedged decision. If the tangle could reverse the call, the report is a diagnosis, and saying so is more useful than a recommendation with a long Watch.

## Status

**When:** one arm, read against its own thresholds or its own history. There is nothing to choose between.

**Headline:** where it stands against the lines that matter, led by the nearest one. "The router holds every floor; complex news is two points above its line."

**Slots:** usually recommendation (ship, hold, or fix first), watch, next, plus why_not when an alternative was implicitly on the table.

**Usual failure:** no line to read against. A status report needs declared thresholds or a prior run. Without either, it is exploratory.

## Exploratory

**When:** no question was declared, or the campaign measured things without a decision in view.

**Headline:** the most decision-relevant thing learned, stated as a finding, and the question worth asking next.

**Slots:** known_despite_it (what is solid), what_is_tangled where relevant, what_settles_it, next. The next steps matter most here. They are usually declarations: the question, the margin, the control.

**Usual failure:** dressing a pattern found after the fact as a confirmed result. Everything here is a lead. Say so once.

## Mixing forms

Reports often mix. A decision on cost can carry a diagnosis on the judged aspect: "Keep the current model: the candidates cost more. Whether they also hold character worse can't be read until the judge is calibrated." The headline carries the dominant form, meaning the one the reader acts on. The secondary form gets its own claim and its own slot (often Watch or what_is_tangled), and never contradicts the headline unaddressed.

Pick the four decision slots that fit what you are actually saying. A decision with an unresolved secondary question might use recommendation, why_not, what_is_tangled and next.

## When there is no good story

Sometimes the campaign supports no reading at all: nothing separates, the declarations came after the results and could have gone either way, the runs are too short. That is a valid report. Write it as a diagnosis or an exploratory story whose next steps name what to declare or run. A confident story the evidence doesn't hold is the one real failure.
