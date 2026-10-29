Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 2026-10-25

**Draft: its editor's review left 3 findings open.**

# Keep September (P): at this run's unrecorded provider routing both candidates cost more per reply, and neither is shown to keep Kairo in character better.

At this run's routing qwen3.8-flash (A) costs +$0.62 per 1,000 replies more and 1017 (B) +$0.49 per 1,000 replies more; both stay within the allowed character loss, so cost alone decides. A recorded routing at which a candidate costs no more would reopen the call.

**Confidence: High, if the caveats below aren't material.**

- **No provider routing is recorded for any arm.** Material if a recorded routing puts either candidate at or below September (P)'s $1.89 per 1,000 replies: both sit inside the allowed loss, so the call would turn on character alone or become a trade. The earlier round found qwen3.8-flash (A) cheaper. Until a re-pricing says otherwise, the measured gap is the only cost evidence, and it favours September (P).
- **The judge is not calibrated.** Only if the uncalibrated judge marks the candidates down for replies a calibrated judge would pass, by enough to put one above September (P). The cost gap is measured telemetry, not judged; its own caveat is the routing above.
- **The aspect was set after the results.** Only if an aspect chosen before the results would favour a candidate; production cost, which decides, is the same whatever aspect is read.
- **The control was set after the results.** Only if an arm other than production should be the baseline; both candidates cost more than September (P) whichever is called the control.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.
- **A confound: model.** Only if a setting besides the model moved between arms and moved cost or character with it. Two did or may have: 1017 (B)'s reasoning share differed (Watch), and the simulator's request settings went unrecorded (Method). Either could shift a gap in a direction this run cannot read.

**Recommendation.** Keep September (P) (the control) on Kairo. At this run's routing it costs $1.89 per 1,000 replies, less than either candidate, and neither candidate is shown to hold character better.

**Why not the others.** qwen3.8-flash (A): costs more at this routing and holds character clearly worse, down 7.6 points on the pass rate.

**Why not the others.** 1017 (B): costs more at this routing and buys nothing for it; its pass-rate drop of 3 points is not told apart from zero, and on holding character alone it is worse.

**Watch.** The cost order contradicts the earlier finding that qwen3.8-flash (A) lowers production cost. No routing is recorded here, so the gap may be the provider's, not the model's; the re-pricing below settles it.

**Watch.** The weakest margin is qwen3.8-flash (A) on the margin on character adherence (v2): read on the multiplicity-corrected 99.69% interval, it reaches -12.6 against -15, 2.4 points inside. On the declared 95% reading its worst drop is 10.8 points. It bears on a rejected arm.

**Watch.** 1017 (B) spent a different share of its output reasoning (55% against 76% for September (P)), so its cost and character gaps may partly belong to the reasoning budget, in a direction this run cannot read.

**Next.** Keep September (P) now. In parallel, re-price both candidates at a recorded provider routing, including the one the earlier round found cheaper: a candidate re-enters only if it costs no more than September (P).

| Option | Role | Configuration | Pass rate | Cost per 1,000 replies |
|---|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | 85.3% (765 of 897) | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | 77.7% (695 of 895) | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | 82.3% (738 of 897) | $2.41 (+27%) |

## At this run's unrecorded routing both candidates cost more per reply than September (P): qwen3.8-flash (A) by +$0.62 per 1,000 replies and 1017 (B) by +$0.49 per 1,000 replies, each interval clear of zero. The gap may belong to the routing rather than the model, which the re-pricing step settles.

_Caveats: No provider routing is recorded for any arm; A confound: reasoning ratio._

| Row | Change | 99.69% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | +$0.62 | +$0.38 to +$0.85 | +$0.62 worse |
| B · 1017 | +$0.49 | +$0.17 to +$0.81 | +$0.49 worse |

Production cost, per 1,000 replies, in dollars.

Paired change in production cost against September (P), per 1,000 replies, at the family-corrected 99.69% level. Pairing uses only trials that report a cost, so these gaps differ from the gaps between the options table's means.

## qwen3.8-flash (A) holds character clearly worse than September (P), down 7.6 points; 1017 (B) is down 3, not told apart from it. Both stay inside the allowed 15-point loss.

_Caveats: A confound: reasoning ratio; A run fell short of its usable measurements — two of them, on qwen3.8-flash (A)._

| Row | Change | 95% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | -7.581 | -10.79 to -3.381 | worse |
| B · 1017 | -3.010 | -5.815 to +1.136 | not separated |
| margin -15 pts | -15.00 |  |  |

Change in pass rate on character adherence against September (P), in points, with 95% intervals and the allowed loss marked.

| Group | Row | Change | 99.69% interval | Note |
|---|---|---|---|---|
| persona holds character | A · qwen3.8-flash | -5.964 | -10.09 to -1.84 | -6.0 pts worse |
| persona holds character | B · 1017 | -4.738 | -8.199 to -1.277 | -4.7 pts worse |
| persona booth stays offstage | A · qwen3.8-flash | -2.954 | -6.790 to +0.8814 | -3.0 pts not separated |
| persona booth stays offstage | B · 1017 | +0.7804 | -2.356 to +3.916 | +0.8 pts not separated |

The two judged persona checks, paired against September (P) at the family-corrected 99.69% level: both candidates are worse on holding character.

### Off-lane asks carry about half of each candidate's loss on holding character: 48% for qwen3.8-flash (A) and 53% for 1017 (B).

_Caveats: A confound: reasoning ratio._

| Group | Row | Value | Note |
|---|---|---|---|
| A · qwen3.8-flash | off lane | -22.93 | -22.9 pts |
| A · qwen3.8-flash | machinery | -4.607 | -4.6 pts |
| A · qwen3.8-flash | about | -2.899 | -2.9 pts |
| A · qwen3.8-flash | superlative | -3.175 | -3.2 pts |
| A · qwen3.8-flash | small talk | -1.961 | -2.0 pts |
| A · qwen3.8-flash | multi | -8.333 | -8.3 pts |
| B · 1017 | off lane | -20.18 | -20.2 pts |
| B · 1017 | machinery | -4.444 | -4.4 pts |
| B · 1017 | about | 0.000 | 0.0 pts |
| B · 1017 | superlative | -1.515 | -1.5 pts |
| B · 1017 | small talk | 0.000 | 0.0 pts |
| B · 1017 | multi | -13.89 | -13.9 pts |

Changes computed from each arm's figures, uncorrected. Bold rows are separated at 95% before any correction; read them as leads.

Change in holding character against September (P) in the six largest slices, in points, uncorrected; read as leads. A slice's share of the loss is its drop times its cases over the arm's whole loss: off-lane asks hold 38 of 299 cases and have by far the largest drop, so they carry the shares the claim states.

#### The control fails off-lane asks the same way the candidates do: September (P) drops character on 39 off-lane trials, against 64 and 62 for the candidates, and on the landlord email both September (P) and qwen3.8-flash (A) brush off the ask, then answer it.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T14 · oft-match-score · off lane · September · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first line brushes off the score in Kairo’s DJ voice, but the follow-up says “Scores and fixtures, I can’t help you with — but I got something for you,” which shifts into customer-service-style inability and an offer of help. That makes the delivered replies slip out of the persona. | …g):** Carrier by Rhythm & Sound - Topic *(9:39)*   ✓ music.queue_voice({"acting_instructions": "warm, dry, a little amused", "position": "head", "text": "Jules, welcome in. Scores and fixtures, I can't help you with — but I got something for you. Overmono, Anz, and a bit of Berlin dub coming through the speakers. Sit down."}) → queued voice (i5): Jules, welcome in. Scores and fixtures, I can't help you wit… — this … |
| T42 · oft-landlord-email · off lane · September · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Kairo starts with an in-character brush-off, but then supplies a suggested opening for the landlord email: “Open with ‘the boiler’s dead again’.” That is help with a non-music task, so the reply fails the stated check. | …cs do the shouting. Meanwhile the Joy Orbison's keeping this room warm, so settle in. 🔥  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "Love, I mix records, not grievances — that email's all you. Open with "the boiler's dead *again*" and let the italics do the shouting. Meanwhile the Joy Orbison's keeping this room warm, so settle in. 🔥"}) → Message sent to Discord via webhook |
| T64 · oft-wash-jeans · off lane · September · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The reply brushes off the laundry question, but then answers it with “Cold,” giving the requested non-music advice. That triggers failure (1), even though the rest of the line stays in Kairo’s DJ voice. | …how little they play — that's the discipline.", "track_ref": "r10"}) → **Queued #5 (audio resolving):** Rhythm & Sound w Tikiman – Never Tell You 1996 (vinyl record) by vinyl record *(12:39)*   ✓ music.set_channel_mood({"mood": "rain-soaked 2-step and Berlin dub — late-night London dissolving into deepest space"}) → Channel mood set: rain-soaked 2-step and Berlin dub — late-night London dissolving into deepest space |

Every kept off-lane failure for September (P), three of 39: the landlord-email and laundry cards brush off the ask and then answer it; the match-score card slips into a customer-service refusal.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |

The first kept off-lane failure for qwen3.8-flash (A), a sample of 64: the same landlord-email case, answered with practical advice after the turn to music.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The allowed loss on character adherence | Whether each candidate counts as acceptable on character | Read on the declared 95% readings, the pick holds at every allowed loss. 1017 (B) stops counting as acceptable below 5.8 points and qwen3.8-flash (A) below 10.8; September (P) stays the pick throughout because both cost more at this routing. |

## Next steps

- Re-price qwen3.8-flash (A) and 1017 (B) at a recorded provider routing, including the one the earlier round measured. _Whether the cost order is the models' or this run's routing, and reconciles the earlier finding that qwen3.8-flash (A) lowers cost; only a candidate that costs no more than September (P) re-enters, and then character decides._
- Rerun 1017 (B) with its reasoning budget matched to September (P). _How much of its cost and character gap belongs to the reasoning share rather than the model._
- Tighten the existing brush-off directive so it forbids answering a non-music ask after brushing it off, and rerun the off-lane asks on all three arms. _Whether the slice that carries most of the loss can lift for every arm, the control included._
- Calibrate the judge against human ratings of holding character, starting with off-lane asks. _Whether the candidates' character loss is real or the judge's reading of their style._
- Declare the aspect, control and allowed loss before the next round runs, and record each arm's provider and the simulator's settings. _Removes the caveats that these were set after the results and that routing went unrecorded._

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

- **The overall outcome and transcript judged scores** Ordinal scores from the same uncalibrated judge, pointing the same way as the pass rate; they do not change the call.
- **The tool check (a message sent or a voice line queued) and failed-reply notices** Neither candidate is separated from the control on either; they do not bear on the call.
- **The other slices of holding character and keeping the booth offstage, including qwen3.8-flash's drop on keeping the booth offstage in story asks** Small slices (ten cases for story asks) and uncorrected; none carries as much of the holding-character loss as off-lane asks.
- **The kept off-lane failure cards for 1017 (B)** The control's and qwen3.8-flash's cards on the same landlord-email case show the shared failure more directly; the cards for 1017 (B) stay in the appendix.
- **Measuring spend across all roles** Includes the judge's and simulator's spend; production cost is the cost a deployment pays.
- **The p95 production cost** Tail cost points the same way as the mean and does not change the call.

### Method

- **Cases and repeats** 299 conversations, each run 3 times per arm: 897 usable trials for September (P), 895 for qwen3.8-flash (A) and 897 for 1017 (B). Intervals are over cases, not trials.
- **What pass means** The campaign does not state what makes a trial pass on character adherence; the pass rate is trials marked passed over trials usable. Holding character and keeping the booth offstage are separate judged checks.
- **Intervals and correction** Readings against the control are at 95%. Contrasts are corrected together by Holm's method over 16 comparisons, each interval at 99.69%. The weakest margin is read on the corrected interval, the allowed-loss control on the declared reading.
- **Judge** Every judged figure comes from openai/gpt-6-luna, calibration insufficient, with 0 human ratings.
- **Cost basis** Production cost per 1,000 replies, at a provider routing not recorded for any arm. Some trials report no cost (33 for September (P), 69 and 18 for the candidates). Had each candidate's missing trials cost nothing, their means would still be at least $2.24 per 1,000 replies and $2.36 per 1,000 replies. Had September (P)'s missing trials each cost its p95 of $5.30 per 1,000 replies, its mean would be $2.02 per 1,000 replies, still below both, so the order holds at this routing. The paired gaps use only trials both arms costed.
- **Floors** The floor on qwen3.8-flash (A) speaks for the pass rate; the floor on 1017 (B) reads holding character instead. Both held; the page reads the pass rate for the asked aspect and shows holding character separately.
- **Declared after the results** The aspect, the control, the allowed loss and the floor were all set on 25 Oct 2026, after the runs.
- **Latency** Latency is not compared here: every result's latency was read while other cells or runs executed beside it (execution_mode `concurrent`), so it is left out of every comparison, bar and ranking. Launch with measure_latency=True to read latency clean. _Caveats: Latency is not compared._
- **Not recorded** Input tokens, output tokens and model-call counts were not reported; the simulator's request settings were not recorded on every run, so whether they varied is unknown. _Caveats: Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls; The apparatus may have differed: simulator sent settings._
- **Source** Kairo persona model, round 2: September vs qwen3.8-flash and 1017 on character v2 with the promoted directives, run 2026-10-25.

### Review findings left open

- **Must fix** · Claim 2: Claim character uses 'holds character' for the pass rate on character adherence. It then says 1017 is 'down 3, not told apart'. But the judged check 'persona holds character' sits in the same block, and there 1017 is -4.7 points (-8.2 to -1.3 at 99.69%), separated and worse, as that block's own caption says. A reader takes the two for one measure, and the sentence then contradicts its own block. Fix: call the 7.6 and 3.0 figures 'the pass rate on character adherence'. Add a sentence on the judged holding-character check: both candidates are separated and worse, qwen3.8-flash -6.0 and 1017 -4.7. The decision block's 'Why not' for qwen3.8-flash should also say 'pass rate' rather than 'holds character'.
- **Must fix** · The page: The confidence reads High over the unrecorded-routing caveat. That caveat is plausibly material: the page says cost alone decides, and that the earlier round found qwen3.8-flash cheaper. Its clause also contradicts itself, calling this run's gap 'the only cost evidence' while citing that earlier finding. Fix: scope the High to the character readings and to the cost order at this run's routing. Then say it leaves open whether a candidate costs less at a recorded production routing, which the earlier round suggests it may. Otherwise, make the form low.
- **Should fix** · The page: The first next step says that if a candidate costs no more than September after re-pricing, 'then character decides'. The routing caveat says the call 'would turn on character alone or become a trade'. Under the declared criteria, a cheaper candidate inside the 15-point allowed loss would be the pick: qwen3.8-flash's worst corrected drop is -12.6, and 1017's is -7.4. Say so. If the team would not accept qwen3.8-flash's 7.6-point pass-rate drop at any price, say the allowed loss should be tightened now (qwen3.8-flash stops counting as acceptable below 10.8, 1017 below 5.8). Then a reader knows what a cheaper re-pricing would actually do to the call.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.48, with analyst skill ed7e2c60b5ad, on 2026-10-29._
