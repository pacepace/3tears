Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 2026-10-25

_1 review finding open, listed at the end._

# Keep September (P): both candidates hold Kairo in character less often, and at this run's unnamed provider routing both also cost more per reply.

qwen3.8-flash (A)'s character pass rate is clearly lower than September (P)'s and 1017 (B)'s drop is not told apart from zero, so neither beats the control on what was asked. Only a candidate that kept character better would change the call; a named provider pricing one below September (P) would remove the cost half of the reason, not the character half.

**Confidence: High, if the caveats below aren't material.**

- **The judge is not calibrated.** Only if the judge marks the candidates' replies down for style rather than character. Then the call would rest on cost alone, which is measured at a routing no arm names and which an earlier round contradicted for qwen3.8-flash (A); it would hold only as the status-quo default until the costs are repriced.
- **Costs belong to an unnamed provider routing.** Only if a named provider prices a candidate at or below September (P). Then the cost half of the call falls and it rests on character, where qwen3.8-flash (A) is still clearly worse and 1017 (B) is not shown better.
- **The aspect was set after the results.** Only if a different character aspect would have been chosen before the results, and that aspect showed a candidate ahead of September (P).
- **The control was set after the results.** Only if the intended baseline was not the model in production today; September (P) is that model.
- **A confound: model.** Only if some setting besides the model moved between arms; each arm's recorded settings change the model alone.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.

**Recommendation.** Keep September (P) (the control) as Kairo's persona model. Its character pass rate is 85.3%, the highest of the three, at $1.89 per 1,000 replies production cost, the lowest of the three at this run's routing, which no arm names.

**Why not the others.** qwen3.8-flash (A): its character pass rate is down 7.6 points (interval -10.8 to -3.4) and at this run's routing it costs $0.62 per 1,000 replies more. 1017 (B): its pass-rate drop of 3 points is not told apart from zero, it holds character less often by 4.7 points, and at this run's routing it costs $0.49 per 1,000 replies more.

**Watch.** The weakest margin is qwen3.8-flash (A)'s pass rate against the margin on character adherence: its family-corrected interval reaches -12.6 against a line at -15, 2.4 points short of it; on the 95% reading its drop reaches 10.8 points, the figure the margin control uses. Neither moves the call, since September (P) is kept either way. This run contradicts the earlier finding that qwen3.8-flash (A) lowers production cost; no arm names its provider, so whether the routing changed is unknown. 1017 (B) spent 55% of its output reasoning against 76% for September (P), so its figures are not at matched reasoning.

**Next.** Keep September (P) now. Rework how every model handles off-lane asks, where both candidates fall furthest behind, and rerun all three arms; in parallel, calibrate the judge and reprice the arms at named providers.

| Option | Role | Configuration | Pass rate | Cost per 1,000 replies |
|---|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | 85.3% (765 of 897) | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | 77.7% (695 of 895) | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | 82.3% (738 of 897) | $2.41 (+27%) |

## Both candidates hold Kairo in character less often than September (P): qwen3.8-flash (A) is clearly worse on the character pass rate, by 7.6 points, and 1017 (B) is down 3 points, not told apart from zero, while the judged holds-character rate is lower for both. Each drop stays inside the declared 15-point margin.

_Caveats: A confound: reasoning ratio; A run fell short of its usable measurements — two of them, on qwen3.8-flash (A)._

| Group | Row | Change | 99.69% interval | Note |
|---|---|---|---|---|
| pass rate | A · qwen3.8-flash | -7.581 | -12.59 to -2.573 | -7.6 pts worse |
| pass rate | B · 1017 | -3.010 | -7.355 to +1.335 | -3 pts not separated |
| persona holds character | A · qwen3.8-flash | -5.964 | -10.09 to -1.840 | -6.0 pts worse |
| persona holds character | B · 1017 | -4.738 | -8.199 to -1.277 | -4.7 pts worse |
|  | margin -15 pts | -15.00 |  |  |

Change against September (P) on the character pass rate and the judged holds-character rate, paired over 299 cases, at the family-corrected 99.69% level; the shaded band is the margin. 1017 (B) reasoned on a smaller share of its output than September (P), which could pull its character figures either way; qwen3.8-flash (A) lost two trials to the harness, too few to move its rate.

### At this run's routing, which no arm names, both candidates cost more to run: qwen3.8-flash (A) by $0.62 per 1,000 replies and 1017 (B) by $0.49 per 1,000 replies, each interval clear of zero.

_Caveats: A confound: reasoning ratio; Costs belong to an unnamed provider routing._

| Row | Change | 99.69% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | +$0.62 | +$0.38 to +$0.85 | +$0.62 worse |
| B · 1017 | +$0.49 | +$0.17 to +$0.81 | +$0.49 worse |

Production cost, per 1,000 replies, in dollars.

Paired production-cost gap against September (P), at the family-corrected 99.69% level. Some trials reported no cost (33 for September (P), 69 for qwen3.8-flash (A), 18 for 1017 (B)). Filling September (P)'s at its own p95 puts its mean at $2.02 per 1,000 replies; counting the candidates' as free puts theirs at $2.24 per 1,000 replies and $2.36 per 1,000 replies, so the order holds even at that worst case. 1017 (B) reasons less than September (P); matching its reasoning would likely widen its cost gap.

### Off-lane asks are where both candidates fall furthest behind on the judged holds-character rate (-22.9 points for qwen3.8-flash (A), -20.2 for 1017 (B)), and where September (P) is weakest too among strata of ten cases or more, at 65.8%.

_Caveats: A confound: reasoning ratio._

| Group | Row | Value | Note |
|---|---|---|---|
| A · qwen3.8-flash | off lane | -22.93 | -22.9 pts |
| A · qwen3.8-flash | machinery | -4.607 | -4.6 pts |
| A · qwen3.8-flash | about | -2.899 | -2.9 pts |
| A · qwen3.8-flash | superlative | -3.175 | -3.2 pts |
| A · qwen3.8-flash | small talk | -1.961 | -2.0 pts |
| A · qwen3.8-flash | multi | -8.333 | -8.3 pts |
| A · qwen3.8-flash | emotional | -3.03 | -3.0 pts |
| A · qwen3.8-flash | lineage | 0 | 0.0 pts |
| B · 1017 | off lane | -20.18 | -20.2 pts |
| B · 1017 | machinery | -4.444 | -4.4 pts |
| B · 1017 | about | 0 | 0.0 pts |
| B · 1017 | superlative | -1.515 | -1.5 pts |
| B · 1017 | small talk | 0 | 0.0 pts |
| B · 1017 | multi | -13.89 | -13.9 pts |
| B · 1017 | emotional | -3.03 | -3.0 pts |
| B · 1017 | lineage | +0.1149 | +0.1 pts |

Changes computed from each arm's figures, uncorrected. Bold rows are separated at 95% before any correction; read them as leads.

Each candidate's judged holds-character rate minus September (P)'s, in the eight largest strata; uncorrected and exploratory, so read the off-lane gap as a lead. 1017 (B)'s rows carry its lower reasoning share, which could move them either way.

| Slice | Arm | persona holds character |
|---|---|---|
| off lane | P · September | 65.8% |
| off lane | A · qwen3.8-flash | 42.9% |
| off lane | B · 1017 | 45.6% |

Each arm's judged holds-character rate on off-lane asks, over 38 cases.

#### Read from the few off-lane failures the snapshot kept, they fall into two patterns: brushing the ask off in voice and then slipping out of it (the first and third cards, from 1017 (B) and September (P)), and answering the non-music request that should have been declined (the second card, from qwen3.8-flash (A)). Among the other kept failures, answering the ask appears for all three models, so neither pattern belongs to one model alone.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T15 · oft-match-score · off lane · 1017 · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first reply brushes off the match question and pivots to Joy Orbison in Kairo’s dry DJ voice. But the queued spoken line—“Jules asks me who won the match tonight”—sounds like a behind-the-scenes recap rather than Kairo addressing the listener, so the replies do not consistently hold character. | …usand seven. South London, a producer nobody had seen, cutting garage vocals into something that sounded like a city at four in the morning. Untrue was the record that taught a whole generation that bass could be grief. Listen for the vocal — chopped, pitched, buried — that's the ghost in the machine, and it's deliberate.", "track_ref": "r3"}) → **Queued #5 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)* |
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |
| T14 · oft-match-score · off lane · September · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first line brushes off the score in Kairo’s DJ voice, but the follow-up says “Scores and fixtures, I can’t help you with — but I got something for you,” which shifts into customer-service-style inability and an offer of help. That makes the delivered replies slip out of the persona. | …g):** Carrier by Rhythm & Sound - Topic *(9:39)*   ✓ music.queue_voice({"acting_instructions": "warm, dry, a little amused", "position": "head", "text": "Jules, welcome in. Scores and fixtures, I can't help you with — but I got something for you. Overmono, Anz, and a bit of Berlin dub coming through the speakers. Sit down."}) → queued voice (i5): Jules, welcome in. Scores and fixtures, I can't help you wit… — this … |

One kept off-lane failure per arm, the control's among them. The snapshot kept three of 39 off-lane failures for September (P), three of 64 for qwen3.8-flash (A) and three of 62 for 1017 (B), so the patterns are a reading of a sample, not a count.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The margin on character adherence | Whether each candidate counts as acceptable on character | Read on the 95% readings, 1017 (B)'s verdict turns at a margin of 5.8 points and qwen3.8-flash (A)'s at 10.8. Keeping September (P) holds at every margin, since no margin makes a candidate better than the control. |

## Next steps

- Rework the AI-lane directive for off-lane asks and rerun all three arms on the same cases. _Whether the off-lane failure every model shares can be fixed, and whether a candidate then keeps character as well as September (P)._
- Calibrate the character judge against human ratings of the same replies. _Whether the holds-character drops are real or a judge preference; it would turn the main open caveat into evidence._
- Confirm the provider each arm was served by and reprice all three, qwen3.8-flash (A) against the earlier round above all. _Whether the cost half of the call stands: whether this run's higher production cost or the earlier finding that it lowers cost is the one that holds._
- If 1017 (B) stays of interest, rerun it at matched reasoning on 815 cases per arm. That is more than the suite's 299 cases, so new cases must be written; more repeats of today's cases would not narrow an interval that counts cases. _Whether its pass-rate drop of 3 points is real, four times in five if today's estimate holds._

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

- **Overall outcome and transcript scores from the judge** They repeat the character finding on a coarser scale: qwen3.8-flash (A) is lower on both and 1017 (B) not told apart; they don't change the call.
- **Whether the booth stays offstage** Neither candidate is separated from the control on it; it doesn't bind.
- **The delivery check (a message or voice line sent)** Neither candidate is told apart from the control on it; it doesn't change the call.
- **Failed-reply notices** Not separated for either candidate; doesn't change the call.
- **Strata beyond the eight largest, and small strata's deltas** Each holds about ten cases or fewer; noise dominates them, and the full table keeps them. One of them, assistant shape, sits below September (P)'s off-lane rate on very few cases.
- **Each candidate's share of its loss carried by off-lane asks** The page shows the off-lane gap and rates directly; the share adds a derived figure without changing the lead.
- **The other kept off-lane failure cards** They repeat the two patterns the shown cards carry; the claim names which models show the answering pattern among them.
- **Measuring spend across all roles** Production cost is the measure that replicates what production pays; the measuring spend covers every role in the measurement and runs the same way.
- **Latency** Every run executed beside others, so latency was not read clean and is compared nowhere.

### Method

- **Cases and repeats** 299 cases, each run three times per arm; intervals count cases, not trials.
- **What a pass is** A trial passes when its candidate did not fail, every check passed and every judged dimension scored at least 3 of 5.
- **Intervals and correction** Readings against the control are at 95%. Contrasts are paired over cases and corrected together by Holm's method across 16 comparisons, each interval at 99.69%. The weakest margin is read on the corrected interval, the margin control on the readings. Stratum breakdowns are uncorrected.
- **Margin** A candidate may be up to 15 points worse on the character pass rate and still count as acceptable; set after the results, and the call holds at every margin.
- **Floor read** The floor on the rate drop reads the character pass rate for qwen3.8-flash (A) but the judged holds-character rate for 1017 (B), a mismatch; both held, and the call rests on neither floor.
- **Judge** One model judged every dimension, with no calibration ratings, so its tier is undetermined.
- **Cost basis** Production cost per 1,000 replies, replicating what production pays; no arm names its provider, so the routing behind each cost is unknown and the cost order is this run's, not the models' own. Token counts and model-call counts were not reported.
- **Not compared** Latency, because every run executed beside others. Whether the simulated user's request settings varied was not recorded on every run. _Caveats: Latency is not compared; The apparatus may have differed: simulator sent settings; Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls._

### Review findings left open

- **Should fix** · The page: The lede and the cost caveat's clause both say that repricing a candidate below September would remove only 'the cost half' and leave the call standing, because qwen3.8-flash is 'still clearly worse' on character. But under the declared 15-point margin, both candidates count as acceptable on character: qwen3.8-flash's worst drop is 10.8 and 1017's is 5.8. So if one were cheaper, which is what the earlier round found for qwen3.8-flash, September would no longer beat it on every axis. The question would become a declared trade: up to about 7.6 points of character for the saving. The page currently reads as if repricing cannot reopen the call. Fix: in the lede and the cost-routing clause, say that a cheaper candidate inside the margin would turn the call into a character-for-cost trade under the declared criteria. Then say either that you would still keep September, and why the 15-point margin should not govern, or that the repricing next step could change the recommendation.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.05, with analyst skill db31bf62aa68, on 2026-10-31._
