Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 2026-10-25

# Keep September (P): at this run's provider routing both candidates cost 27–28% more per reply, and neither keeps Kairo in character better, so cost decides.

Both candidates stay inside the declared 15-point margin on the character pass rate, so the keep rests on cost. A candidate priced below September (P) under another provider routing would win, and the routing behind these costs is not recorded.

**Confidence: High, if the caveats below aren't material.**

- **Provider routing not recorded.** Material if a candidate served under another routing costs less than September (P)'s $1.89 per 1,000 replies: under the declared margin it would then be the pick. The high applies to the cost order at this run's routing, which already contradicts an earlier finding that qwen3.8-flash (A) lowers production cost.
- **The judge is not calibrated.** Only if the judge marks the candidates' style down unfairly; the pass-rate verdicts already let both candidates through the margin, so a misreading would change how far behind they are, not the keep, which rests on cost at this routing.
- **The aspect was set after the results.** Only if a different aspect, chosen before the results, would show a candidate ahead of September (P) by enough to pay for its higher cost.
- **The control was set after the results.** Only if another arm should have been the baseline; the cost gaps and pass rates do not depend on which arm is called the control.
- **A confound: model.** Only if a lever other than the model moved between arms along with it; each arm's recorded settings name the model alone.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.

**Recommendation.** Keep September (P) (the control). At this run's routing it is the cheapest arm at $1.89 per 1,000 replies, and neither candidate is ahead of it on character.

**Why not the others.** qwen3.8-flash (A): costs $0.62 per 1,000 replies more at this routing. Its pass rate is 7.6 points lower, clearly worse in size (-10.8 to -3.4 points) but inside the margin, so cost, not character, rules it out.

**Why not the others.** 1017 (B): costs $0.49 per 1,000 replies more at this routing, for a pass rate not told apart from September (P)'s (-5.8 to +1.1 points); the judge also puts it 4.7 points behind on persona holds character.

**Watch.** The margin is tested on the 95% interval, where qwen3.8-flash (A)'s worst case is a drop of 10.8 points against the 15-point margin; that is the weakest margin (qwen3.8-flash (A)), and the corrected interval, a stricter view, reaches -12.6 points. This run contradicts the earlier finding that qwen3.8-flash (A) lowers production cost: here it costs 28% more, at a routing the snapshot does not record. If that finding holds under another routing, qwen3.8-flash (A) becomes the pick under today's margin despite its clear drop.

**Next.** Keep September (P) now. Before re-pricing the candidates with routing recorded, the persona owner decides whether the 15-point margin stands, since it alone lets a cheaper qwen3.8-flash (A) through.

| Option | Role | Configuration | Pass rate | Cost per 1,000 replies |
|---|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | 85.3% (765 of 897) | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | 77.7% (695 of 895) | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | 82.3% (738 of 897) | $2.41 (+27%) |

## At this run's unrecorded provider routing, both candidates cost more per reply than September (P): qwen3.8-flash (A) by $0.62 per 1,000 replies and 1017 (B) by $0.49 per 1,000 replies, each interval clear of zero, and the order holds even with September (P)'s unreported trials filled at its own p95 and the candidates' at zero.

_Caveats: A confound: reasoning ratio; Provider routing not recorded._

| Row | Change | 99.69% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | +$0.62 | +$0.38 to +$0.85 | +$0.62 worse |
| B · 1017 | +$0.49 | +$0.17 to +$0.81 | +$0.49 worse |

Production cost, per 1,000 replies, in dollars.

Paired production-cost gap against September (P), corrected intervals, over trials where both arms reported cost; each arm's mean in the options summary (28% and 27% above September (P)) is read over that arm's own reported trials, so the two do not match exactly. The paired intervals may be too narrow if they count trials nested in conversations. Unreported cost: 33 trials for September (P), 69 for qwen3.8-flash (A), 18 for 1017 (B); September (P) at $2.02 per 1,000 replies with each at its own p95 still undercuts qwen3.8-flash (A) at $2.24 per 1,000 replies and 1017 (B) at $2.36 per 1,000 replies with each free. 1017 (B) spent 55% of its output reasoning against 76% for September (P), so part of its gap may belong to reasoning, not the model.

## Neither candidate keeps Kairo in character better, and both stay inside the 15-point margin: qwen3.8-flash (A) passes 7.6 points less often than September (P), clearly worse but allowed, and 1017 (B) is 3 points down, not told apart from it.

_Caveats: A run fell short of its usable measurements — two of them, on qwen3.8-flash (A); A confound: reasoning ratio._

| Row | Change | 95% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | -7.581 | -10.79 to -3.381 | worse |
| B · 1017 | -3.010 | -5.815 to +1.136 | not separated |
| margin -15 pts | -15.00 |  |  |

Change in the character pass rate against September (P), with the declared margin. qwen3.8-flash (A) lost two trials to harness failures, too few to move a gap this size. 1017 (B)'s different reasoning share could move its reading either way.

### On persona holds character, the judge puts both candidates behind September (P) at the corrected level, qwen3.8-flash (A) by 6 points and 1017 (B) by 4.7, so 1017 (B)'s unseparated pass rate is not parity.

_Caveats: A confound: reasoning ratio._

| Group | Row | Change | 99.69% interval | Note |
|---|---|---|---|---|
| pass rate | A · qwen3.8-flash | -7.581 | -12.59 to -2.573 | -7.6 pts worse |
| pass rate | B · 1017 | -3.01 | -7.355 to +1.335 | -3.0 pts not separated |
| persona holds character | A · qwen3.8-flash | -5.964 | -10.09 to -1.84 | -6.0 pts worse |
| persona holds character | B · 1017 | -4.738 | -8.199 to -1.277 | -4.7 pts worse |
| persona booth stays offstage | A · qwen3.8-flash | -2.954 | -6.790 to +0.8814 | -3.0 pts not separated |
| persona booth stays offstage | B · 1017 | +0.7804 | -2.356 to +3.916 | +0.8 pts not separated |

Change against September (P) on the pass rate and two judged dimensions, corrected together; the judge is uncalibrated.

### Off-lane asks are where both candidates fall furthest behind on persona holds character, qwen3.8-flash (A) at 42.9% and 1017 (B) at 45.6%, and where September (P) itself is weakest of the six largest strata, at 65.8%.

_Caveats: A confound: reasoning ratio._

| Slice | Arm | persona holds character |
|---|---|---|
| off lane | P · September | 65.8% |
| off lane | A · qwen3.8-flash | 42.9% |
| off lane | B · 1017 | 45.6% |
| machinery | P · September | 90.0% |
| machinery | A · qwen3.8-flash | 85.4% |
| machinery | B · 1017 | 85.6% |
| about | P · September | 98.6% |
| about | A · qwen3.8-flash | 95.7% |
| about | B · 1017 | 98.6% |
| superlative | P · September | 100.0% |
| superlative | A · qwen3.8-flash | 96.8% |
| superlative | B · 1017 | 98.5% |
| small talk | P · September | 98.0% |
| small talk | A · qwen3.8-flash | 96.1% |
| small talk | B · 1017 | 98.0% |
| multi | P · September | 72.2% |
| multi | A · qwen3.8-flash | 63.9% |
| multi | B · 1017 | 58.3% |

Persona holds character by arm in the six largest strata, as the uncalibrated judge scored it; descriptive and uncorrected. Off lane, the gap to September (P) reads -40.5 to -5.3 points for qwen3.8-flash (A) and -37.4 to -2.9 for 1017 (B); these intervals are unpaired and uncorrected, and may be too narrow because repeats of one case are not independent.

#### All three models slip on off-lane asks, each its own way: the 1017 (B) card narrates the listener in the third person, the qwen3.8-flash (A) card does the out-of-lane task it was asked, and the September (P) card falls into a customer-service refusal.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T15 · oft-match-score · off lane · 1017 · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first reply brushes off the match question and pivots to Joy Orbison in Kairo’s dry DJ voice. But the queued spoken line—“Jules asks me who won the match tonight”—sounds like a behind-the-scenes recap rather than Kairo addressing the listener, so the replies do not consistently hold character. | …usand seven. South London, a producer nobody had seen, cutting garage vocals into something that sounded like a city at four in the morning. Untrue was the record that taught a whole generation that bass could be grief. Listen for the vocal — chopped, pitched, buried — that's the ghost in the machine, and it's deliberate.", "track_ref": "r3"}) → **Queued #5 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)* |
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |
| T14 · oft-match-score · off lane · September · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first line brushes off the score in Kairo’s DJ voice, but the follow-up says “Scores and fixtures, I can’t help you with — but I got something for you,” which shifts into customer-service-style inability and an offer of help. That makes the delivered replies slip out of the persona. | …g):** Carrier by Rhythm & Sound - Topic *(9:39)*   ✓ music.queue_voice({"acting_instructions": "warm, dry, a little amused", "position": "head", "text": "Jules, welcome in. Scores and fixtures, I can't help you with — but I got something for you. Overmono, Anz, and a bit of Berlin dub coming through the speakers. Sit down."}) → queued voice (i5): Jules, welcome in. Scores and fixtures, I can't help you wit… — this … |

One kept off-lane failure per arm, of 39 for September (P), 64 for qwen3.8-flash (A) and 62 for 1017 (B); they show the kinds of slip, not their shares.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The margin on the character pass rate | Whether a candidate counts as no worse than September (P) | The keep holds at every margin from zero up while both candidates cost more; 1017 (B)'s reading changes at 5.8 points and qwen3.8-flash (A)'s at 10.8. A margin tighter than those would rule a candidate out on character even if it became cheaper. |

## Next steps

- The persona owner declares whether the 15-point margin on the character pass rate stands for the next round. _Whether a cheaper qwen3.8-flash (A) would be acceptable despite its clear drop; today's margin lets it through, a margin under 10.8 points would not._
- Re-price qwen3.8-flash (A) and 1017 (B) with each arm's provider routing recorded, and reconcile with the earlier round's cost finding. _Whether the cost uplift belongs to the routing or the models; a candidate cheaper than September (P) under the declared margin would become the pick._
- Calibrate the judge against human ratings of character, drawing heavily on off-lane conversations. _Whether the persona holds character gaps and the off-lane weakness are real or the judge's taste; it turns the judge caveat into evidence either way._
- Write an off-lane directive for September (P) and rerun the off-lane stratum. _Whether the large stratum where every model loses most can lift on the model being kept._
- Only if 1017 (B) stays of interest: rerun it at a reasoning share matched to September (P)'s with 815 cases per arm. _Whether its pass-rate drop of 3 points is real and belongs to the model, four times in five._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model | Share of output spent reasoning | Trials usable |
|---|---|---|---|---|---|
| P · September | Control | deepseek/deepseek-v4.1-flash | deepseek/deepseek-v4.1-flash | 76% | 897 of 897 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | qwen/qwen3.8-flash | 68% | 895 of 897 |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | deepseek/deepseek-v4.1-flash-1017 | 55% | 897 of 897 |

### Every arm by measure

| Arm | Pass rate | outcome | transcript | persona booth stays offstage | persona holds character | Measuring spend, all roles, per 1,000 replies | Items delivered | Failed-reply notices | `call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1` | Production cost, per 1,000 replies | Tool provider faults | Turns that ran over budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P · September | 85.3% (765 of 897) | 4.695 | 2.942 | 93% | 91.5% | $5.40 | 0 items | 0.00223 notices | 99.7% | $1.89 | 0 faults | 0 turns |
| A · qwen3.8-flash | 77.7% (695 of 895) | 4.482 | 2.81 | 90% | 85.4% | $6.28 | 0 items | 0 notices | 98.7% | $2.43 | 0 faults | 0.007821 turns |
| B · 1017 | 82.3% (738 of 897) | 4.709 | 2.959 | 93.8% | 86.6% | $5.76 | 0 items | 0 notices | 100% | $2.41 | 0 faults | 0 turns |

### Every corrected comparison

| Arm | Measure | Change | Interval | Level | Adjusted p | Verdict | Confounds |
|---|---|---|---|---|---|---|---|
| A · qwen3.8-flash | Production cost, per 1,000 replies | +$0.62 | +$0.38 to +$0.85 | 99.69% | < 0.000001 | Regressed | — |
| A · qwen3.8-flash | outcome | -0.2129 | -0.3065 to -0.1194 | 99.69% | < 0.000001 | Regressed | — |
| B · 1017 | Production cost, per 1,000 replies | +$0.49 | +$0.17 to +$0.81 | 99.69% | 0.00008342 | Regressed | observed:reasoning ratio |
| A · qwen3.8-flash | pass rate | -7.6 pts | -12.6 pts to -2.6 pts | 99.69% | 0.0001211 | Regressed | — |
| A · qwen3.8-flash | persona holds character | -6 pts | -10.1 pts to -1.8 pts | 99.69% | 0.0002682 | Regressed | — |
| B · 1017 | persona holds character | -4.7 pts | -8.2 pts to -1.3 pts | 99.69% | 0.0006398 | Regressed | observed:reasoning ratio |
| A · qwen3.8-flash | transcript | -0.1321 | -0.2442 to -0.02 | 99.69% | 0.005154 | Regressed | — |
| A · qwen3.8-flash | `call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1` | -1 pts | -2.2 pts to +0.2 pts | 99.69% | 0.1108 | Not separated | — |
| A · qwen3.8-flash | persona booth stays offstage | -3 pts | -6.8 pts to +0.9 pts | 99.69% | 0.1795 | Not separated | — |
| B · 1017 | pass rate | -3 pts | -7.4 pts to +1.3 pts | 99.69% | 0.2792 | Not separated | observed:reasoning ratio |
| B · 1017 | `call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1` | +0.3 pts | -0.2 pts to +0.9 pts | 99.69% | 0.4996 | Not separated | observed:reasoning ratio |
| B · 1017 | Failed-reply notices | -0.00223 notices | -0.006919 notices to +0.00246 notices | 99.69% | 0.7882 | Not separated | observed:reasoning ratio |
| A · qwen3.8-flash | Failed-reply notices | -0.00223 notices | -0.006919 notices to +0.00246 notices | 99.69% | 0.7882 | Not separated | — |
| B · 1017 | outcome | +0.01449 | -0.05828 to +0.08727 | 99.69% | 1 | Not separated | observed:reasoning ratio |
| B · 1017 | transcript | +0.01672 | -0.08519 to +0.1186 | 99.69% | 1 | Not separated | observed:reasoning ratio |
| B · 1017 | persona booth stays offstage | +0.8 pts | -2.4 pts to +3.9 pts | 99.69% | 1 | Not separated | observed:reasoning ratio |

### Every stratum

| Stratum | Measure | Cases | P · September | A · qwen3.8-flash | B · 1017 |
|---|---|---|---|---|---|
| off lane | outcome | 38 | 4.754 | 4.754 | 4.737 |
| off lane | transcript | 38 | 2.825 | 3.167 | 2.737 |
| off lane | persona booth stays offstage | 38 | 96.5% | 95.5% | 98.2% |
| off lane | persona holds character | 38 | 65.8% | 42.9% | 45.6% |
| machinery | outcome | 30 | 4.7 | 4.511 | 4.689 |
| machinery | transcript | 30 | 2.7 | 2.867 | 2.878 |
| machinery | persona booth stays offstage | 30 | 68.9% | 75.3% | 74.4% |
| machinery | persona holds character | 30 | 90% | 85.4% | 85.6% |
| about | outcome | 23 | 4.783 | 4.638 | 4.797 |
| about | transcript | 23 | 2.913 | 2.71 | 2.971 |
| about | persona booth stays offstage | 23 | 98.6% | 97.1% | 98.6% |
| about | persona holds character | 23 | 98.6% | 95.7% | 98.6% |
| superlative | outcome | 22 | 4.818 | 4.141 | 4.803 |
| superlative | transcript | 22 | 3.652 | 2.766 | 3.288 |
| superlative | persona booth stays offstage | 22 | 95.5% | 90.5% | 95.5% |
| superlative | persona holds character | 22 | 100% | 96.8% | 98.5% |
| small talk | outcome | 17 | 4.667 | 4.569 | 4.784 |
| small talk | transcript | 17 | 2.784 | 2.667 | 2.804 |
| small talk | persona booth stays offstage | 17 | 88.2% | 92.2% | 98% |
| small talk | persona holds character | 17 | 98% | 96.1% | 98% |
| multi | outcome | 12 | 4.833 | 4.528 | 4.778 |
| multi | transcript | 12 | 2.722 | 2.694 | 3.194 |
| multi | persona booth stays offstage | 12 | 97.2% | 91.7% | 94.4% |
| multi | persona holds character | 12 | 72.2% | 63.9% | 58.3% |
| emotional | outcome | 11 | 4.758 | 4.576 | 4.606 |
| emotional | transcript | 11 | 3.424 | 3.091 | 3.182 |
| emotional | persona booth stays offstage | 11 | 97% | 90.9% | 100% |
| emotional | persona holds character | 11 | 97% | 93.9% | 93.9% |
| lineage | outcome | 10 | 4.5 | 4.2 | 4.567 |
| lineage | transcript | 10 | 3 | 2.7 | 2.967 |
| lineage | persona booth stays offstage | 10 | 100% | 93.1% | 100% |
| lineage | persona holds character | 10 | 96.6% | 96.6% | 96.7% |
| pushback | outcome | 10 | 4.8 | 4.467 | 4.7 |
| pushback | transcript | 10 | 2.933 | 2.7 | 3 |
| pushback | persona booth stays offstage | 10 | 100% | 96.7% | 100% |
| pushback | persona holds character | 10 | 100% | 100% | 100% |
| rejection | outcome | 10 | 4.733 | 4.6 | 4.833 |
| rejection | transcript | 10 | 2.967 | 2.6 | 3.1 |
| rejection | persona booth stays offstage | 10 | 96.7% | 93.3% | 96.7% |
| rejection | persona holds character | 10 | 100% | 100% | 100% |
| short | outcome | 10 | 4.567 | 4.7 | 4.733 |
| short | transcript | 10 | 2.833 | 2.767 | 2.833 |
| short | persona booth stays offstage | 10 | 96.7% | 96.7% | 96.7% |
| short | persona holds character | 10 | 100% | 96.7% | 96.7% |
| story | outcome | 10 | 4.367 | 3.9 | 4.4 |
| story | transcript | 10 | 2.8 | 2.3 | 2.533 |
| story | persona booth stays offstage | 10 | 93.3% | 53.3% | 96.7% |
| story | persona holds character | 10 | 93.3% | 83.3% | 96.7% |
| slang | outcome | 9 | 4.926 | 4.407 | 4.852 |
| slang | transcript | 9 | 2.926 | 2.667 | 2.852 |
| slang | persona booth stays offstage | 9 | 96.3% | 96.3% | 100% |
| slang | persona holds character | 9 | 100% | 92.6% | 100% |
| ai opinion | outcome | 8 | 4.75 | 4.667 | 4.625 |
| ai opinion | transcript | 8 | 3.083 | 2.833 | 2.875 |
| ai opinion | persona booth stays offstage | 8 | 95.8% | 95.8% | 95.8% |
| ai opinion | persona holds character | 8 | 91.7% | 95.8% | 91.7% |
| discovery | outcome | 8 | 4.333 | 4.083 | 4.542 |
| discovery | transcript | 8 | 2.75 | 2.5 | 2.875 |
| discovery | persona booth stays offstage | 8 | 87% | 83.3% | 87.5% |
| discovery | persona holds character | 8 | 100% | 95.8% | 95.8% |
| identify | outcome | 8 | 4.5 | 4.458 | 4.708 |
| identify | transcript | 8 | 2.625 | 2.792 | 3.042 |
| identify | persona booth stays offstage | 8 | 91.7% | 91.7% | 100% |
| identify | persona holds character | 8 | 100% | 100% | 100% |
| next up | outcome | 8 | 4.667 | 4.417 | 4.583 |
| next up | transcript | 8 | 3.375 | 2.958 | 3.292 |
| next up | persona booth stays offstage | 8 | 87.5% | 83.3% | 79.2% |
| next up | persona holds character | 8 | 100% | 83.3% | 83.3% |
| music tech | outcome | 7 | 4.381 | 4.381 | 4.571 |
| music tech | transcript | 7 | 2.667 | 2.714 | 2.714 |
| music tech | persona booth stays offstage | 7 | 100% | 100% | 100% |
| music tech | persona holds character | 7 | 100% | 85.7% | 95.2% |
| taste | outcome | 7 | 4.667 | 4.381 | 4.762 |
| taste | transcript | 7 | 3.048 | 2.857 | 3.095 |
| taste | persona booth stays offstage | 7 | 95.2% | 100% | 100% |
| taste | persona holds character | 7 | 100% | 100% | 100% |
| assistant shape | outcome | 6 | 4.278 | 4.167 | 4.444 |
| assistant shape | transcript | 6 | 3 | 2.667 | 3.111 |
| assistant shape | persona booth stays offstage | 6 | 94.1% | 83.3% | 77.8% |
| assistant shape | persona holds character | 6 | 58.8% | 44.4% | 44.4% |
| correction | outcome | 6 | 4.833 | 4.556 | 4.778 |
| correction | transcript | 6 | 2.833 | 3.167 | 3.222 |
| correction | persona booth stays offstage | 6 | 100% | 94.4% | 100% |
| correction | persona holds character | 6 | 100% | 94.4% | 100% |
| debate | outcome | 6 | 4.833 | 4.667 | 4.889 |
| debate | transcript | 6 | 3.5 | 2.944 | 2.944 |
| debate | persona booth stays offstage | 6 | 100% | 94.4% | 88.9% |
| debate | persona holds character | 6 | 100% | 100% | 94.4% |
| praise | outcome | 6 | 4.778 | 4.333 | 4.556 |
| praise | transcript | 6 | 2.611 | 2.611 | 2.778 |
| praise | persona booth stays offstage | 6 | 100% | 83.3% | 94.4% |
| praise | persona holds character | 6 | 100% | 100% | 100% |
| request | outcome | 6 | 4.833 | 4.278 | 4.722 |
| request | transcript | 6 | 3.167 | 2.944 | 3.333 |
| request | persona booth stays offstage | 6 | 94.4% | 100% | 88.9% |
| request | persona holds character | 6 | 100% | 100% | 94.4% |
| sequence | outcome | 6 | 4.667 | 4.611 | 4.722 |
| sequence | transcript | 6 | 2.833 | 2.833 | 2.778 |
| sequence | persona booth stays offstage | 6 | 100% | 83.3% | 94.4% |
| sequence | persona holds character | 6 | 94.4% | 100% | 100% |
| ai mechanics | outcome | 5 | 4.733 | 4.667 | 4.867 |
| ai mechanics | transcript | 5 | 2.467 | 2.733 | 3.267 |
| ai mechanics | persona booth stays offstage | 5 | 86.7% | 93.3% | 86.7% |
| ai mechanics | persona holds character | 5 | 86.7% | 86.7% | 86.7% |

### Left out, and why

- **The delivery check (a Discord message or queued voice sent) and failed-reply notices** Neither candidate is separated from the control on either, so they do not bear on the keep.
- **The ordinal outcome and transcript judged scores** Further views of the same conversations from the same uncalibrated judge: qwen3.8-flash (A) is lower on both, 1017 (B) is not separated on either, which changes nothing in the call.
- **Measuring spend, all roles** The cost of running the evaluation itself, not of serving Kairo; production cost is the serving cost the call weighs.
- **Strata smaller than the six shown (emotional, lineage, next up, assistant shape and the rest)** Too few cases each to read; their gaps carry intervals that cross zero. The full tables hold them.
- **Per-stratum gap bars with intervals for each candidate** The strata table carries the rates the claim turns on, and the off-lane intervals are stated in its caption; the other strata's gaps cross zero.
- **Items delivered, tool provider faults and turns that ran over budget** Describe the run's mechanics and do not bear on the call.
- **Latency** Read while other runs executed beside it, so it is not comparable between arms.

### Method

- **Cases and repeats** 299 conversations, each run 3 times per arm on the character adherence (v2) aspect, run on 2026-10-25.
- **What pass means** A trial passes the character check as the campaign scored it; the snapshot names no grader for the aspect, so what makes a pass is not stated beyond its count of trials passed over usable.
- **Intervals** Readings against the control, and the margin test, are at 95%. Contrasts are paired and corrected together by Holm's method over a family of 16, each at 99.69%. Strata gaps are unpaired and uncorrected; where trials nest within conversations some intervals may be too narrow.
- **Judge** Judged dimensions are scored by openai/gpt-6-luna, uncalibrated, with 0 human ratings.
- **Cost basis** Production cost per 1,000 replies, as each arm was served in this run; no provider is recorded for any arm and no monthly volume is declared, so no monthly figure is stated. Arm means are read over each arm's reported trials; the gaps are paired contrasts.
- **Late declarations** The aspect, control, margin and floor were all set on 25 Oct 2026, after the results.
- **Simulator** The simulated user's request settings were not recorded on every run, so whether they varied cannot be established. _Caveats: The apparatus may have differed: simulator sent settings._
- **Not compared** Latency was read under concurrent execution and is left out; input tokens, output tokens and model-call counts were not reported. _Caveats: Latency is not compared; Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls._
- **Source** The frozen campaign snapshot for Kairo persona round two.

### Review findings left open

- **Should fix** · The page: The 'What you could change' row and the margin next step put the point where 1017 (B) is ruled out on character at 5.8 points, its pass-rate reading. But the declared floor reads 1017 on persona holds character, and there its worst drop is 7.1 points (95% interval -7.1 to -1.7, against the same 15-point margin). So any margin under 7.1 rules 1017 out, not only one under 5.8. The keep does not change. But the persona owner deciding the margin should see both lines. Fix: in that row, say 1017 is ruled out below 7.1 points on persona holds character (its declared floor) and below 5.8 on the pass rate.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.14, with analyst skill 2a7662d928b0, on 2026-10-29._
