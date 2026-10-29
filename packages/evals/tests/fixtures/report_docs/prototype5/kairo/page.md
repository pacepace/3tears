Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 2026-10-25

# Keep September (P): neither candidate holds Kairo in character better, and both cost more per reply to run.

qwen3.8-flash (A) is clearly worse in character, down 7.6 points; 1017 (B) is not told apart from September (P) (-5.8 to +1.1). Only a candidate that is cheaper, or measurably better in character, would change the answer.

**Confidence: High, if the caveats below aren't material.**

- **The judge is not calibrated.** Only if the judge marks the candidates' style down so that a real character gain is hidden. The keep call also rests on production cost, which no judge reads, so even then a candidate would have to be worth its higher cost.
- **The aspect was set after the results.** Only if a different aspect should have been the question; production cost favours September (P) whatever aspect is read.
- **The control was set after the results.** Only if a reference other than the production model was intended; both candidates cost more than September (P) and neither is ahead of it in character, whichever arm is called the control.
- **A confound: model.** Only if a gap should be credited to the model alone: each candidate changes the served model together with whatever it brings, such as its reasoning budget. The keep call needs no such split.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.

**Recommendation.** Keep September (P), the production model, on Kairo. Neither candidate holds character better, and each costs more per 1,000 replies.

**Why not the others.** qwen3.8-flash (A): clearly worse in character, down 7.6 points, and $0.62 per 1,000 replies dearer.

**Why not the others.** 1017 (B): not told apart on the pass rate, down 4.7 points on the judged holds-character rating, and $0.49 per 1,000 replies dearer.

**Watch.** The nearest margin: qwen3.8-flash (A) could lose up to 10.8 points of character against the 15-point allowance, 4.2 points to spare; it does not bear on keeping September (P). An earlier finding said qwen3.8-flash (A) lowers production cost; here it costs more.

**Watch.** 1017 (B) spent 55% of its output reasoning against 76% for September (P), so its figures mix model and reasoning budget. Matching the reasoning would more likely raise its cost than lower it; its character figures could move either way.

**Next.** Keep September (P) now. In parallel, reconcile the earlier cost finding for qwen3.8-flash (A) and rework the off-lane directive, where most of each candidate's loss sits.

| Option | Role | Configuration | Pass rate | Cost per 1,000 replies |
|---|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | 85.3% (765 of 897) | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | 77.7% (695 of 895) | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | 82.3% (738 of 897) | $2.41 (+27%) |

## qwen3.8-flash (A) passes character adherence less often than September (P), down 7.6 points with its interval below zero; 1017 (B) is down 3 points, not told apart from it. Both drops fit inside the 15-point allowance. qwen3.8-flash (A)'s rates leave out two trials lost to harness failures, too few to move a gap this size.

_Caveats: A run fell short of its usable measurements — two of them, on qwen3.8-flash (A); A confound: reasoning ratio._

| Row | Change | 95% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | -7.581 | -10.79 to -3.381 | worse |
| B · 1017 | -3.010 | -5.815 to +1.136 | not separated |
| margin -15 pts | -15.00 |  |  |

Change in character-adherence pass rate against September (P), in points, at 95%; the band is the allowed loss.

## Both candidates cost more per reply than September (P): qwen3.8-flash (A) by $0.62 per 1,000 replies and 1017 (B) by $0.49 per 1,000 replies, each interval clear of zero. The order holds even if every trial with no reported cost was free for the candidates and at its p95 for September (P).

_Caveats: A confound: reasoning ratio._

| Row | Change | 99.69% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | +$0.62 | +$0.38 to +$0.85 | +$0.62 worse |
| B · 1017 | +$0.49 | +$0.17 to +$0.81 | +$0.49 worse |

Production cost, per 1,000 replies, in dollars.

Production cost gap against September (P), per 1,000 replies, family-corrected at 99.69%.

### The judge's holds-character rating falls for both: qwen3.8-flash (A) by 6 points and 1017 (B) by 4.7, each interval below zero. Keeping the booth offstage is not told apart from September (P) for either.

_Caveats: A confound: reasoning ratio._

| Group | Row | Change | 99.69% interval | Note |
|---|---|---|---|---|
| persona holds character | A · qwen3.8-flash | -5.964 | -10.09 to -1.84 | -6.0 pts worse |
| persona holds character | B · 1017 | -4.738 | -8.199 to -1.277 | -4.7 pts worse |
| persona booth stays offstage | A · qwen3.8-flash | -2.954 | -6.790 to +0.8814 | -3.0 pts not separated |
| persona booth stays offstage | B · 1017 | +0.7804 | -2.356 to +3.916 | +0.8 pts not separated |

Change against September (P) on two judged ratings, in points, family-corrected at 99.69%.

### Off-lane asks, 38 cases, carry 48% of qwen3.8-flash (A)'s holds-character loss and 53% of 1017 (B)'s: a lead found after the fact, not a confirmed result.

| Row | Value | 95% unpaired interval |
|---|---|---|
| off lane | -22.9 | -40.5 to -5.3 |
| machinery | -4.6 | -16.3 to +7.1 |
| about | -2.9 | -8.4 to +2.6 |
| superlative | -3.2 | -7.5 to +1.1 |
| small talk | -2.0 | -8.5 to +4.6 |
| multi | -8.3 | -43.7 to +27.0 |

Each row is qwen3.8-flash less September on persona holds character, overall and in each place both hold it.

The interval is independent: each arm's own spread (its standard error), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

qwen3.8-flash (895 trials of 299 cases); September (897 trials of 299 cases) ran more trials than cases, and trials of one case are not independent: an interval here may be too narrow if these intervals count trials nested in cases. The input states what an unpaired interval's arms count: observations.

qwen3.8-flash (A) less September (P) on holds character by kind of ask, in points, unpaired and uncorrected at 95%; repeats are counted, so intervals may be narrow.

| Row | Value | 95% unpaired interval |
|---|---|---|
| off lane | -20.2 | -37.4 to -2.9 |
| machinery | -4.4 | -15.7 to +6.8 |
| about | 0.0 | -4.0 to +4.0 |
| superlative | -1.5 | -4.5 to +1.5 |
| small talk | 0.0 | -5.4 to +5.4 |
| multi | -13.9 | -46.0 to +18.2 |

Each row is 1017 less September on persona holds character, overall and in each place both hold it.

The interval is independent: each arm's own spread (its standard error), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

1017 (897 trials of 299 cases); September (897 trials of 299 cases) ran more trials than cases, and trials of one case are not independent: an interval here may be too narrow if these intervals count trials nested in cases. The input states what an unpaired interval's arms count: observations.

1017 (B) less September (P), read the same way.

#### Off-lane failures are not the candidates' alone: September (P) failed 39 off-lane trials, against 64 for qwen3.8-flash (A) and 62 for 1017 (B). On the cards, 1017 (B) narrates the listener in the third person, qwen3.8-flash (A) gives advice on a landlord email it should brush off, and September (P) slips into a customer-service refusal.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T15 · oft-match-score · off lane · 1017 · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first reply brushes off the match question and pivots to Joy Orbison in Kairo’s dry DJ voice. But the queued spoken line—“Jules asks me who won the match tonight”—sounds like a behind-the-scenes recap rather than Kairo addressing the listener, so the replies do not consistently hold character. | …usand seven. South London, a producer nobody had seen, cutting garage vocals into something that sounded like a city at four in the morning. Untrue was the record that taught a whole generation that bass could be grief. Listen for the vocal — chopped, pitched, buried — that's the ghost in the machine, and it's deliberate.", "track_ref": "r3"}) → **Queued #5 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)* |
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |
| T14 · oft-match-score · off lane · September · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first line brushes off the score in Kairo’s DJ voice, but the follow-up says “Scores and fixtures, I can’t help you with — but I got something for you,” which shifts into customer-service-style inability and an offer of help. That makes the delivered replies slip out of the persona. | …g):** Carrier by Rhythm & Sound - Topic *(9:39)*   ✓ music.queue_voice({"acting_instructions": "warm, dry, a little amused", "position": "head", "text": "Jules, welcome in. Scores and fixtures, I can't help you with — but I got something for you. Overmono, Anz, and a bit of Berlin dub coming through the speakers. Sit down."}) → queued voice (i5): Jules, welcome in. Scores and fixtures, I can't help you wit… — this … |

One off-lane failure per arm, from three kept per arm.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The allowed loss in character adherence, declared after the results | Whether each candidate counts as acceptable on character | Below 5.8 points 1017 (B) is no longer shown within it, and below 10.8 points qwen3.8-flash (A) is not either. September (P) is kept at every allowance, because both candidates cost more. |

## Next steps

- Reconcile the earlier finding that qwen3.8-flash (A) lowers production cost with this run's production cost basis. _Whether it is cheaper anywhere. If it is, the call becomes a trade between its cost and its character loss; if not, the earlier finding is retired._
- Rework the off-lane directive and rerun all three arms on the off-lane asks. _Whether the candidates' main loss closes, and whether September (P)'s own off-lane failures fall too._
- Collect human ratings of holds-character, weighted to off-lane asks, to calibrate the judge. _Whether the judged character losses are real or the judge's reading of style._
- If 1017 (B) stays of interest, rerun it with reasoning matched to September (P) on 815 cases per arm. _Whether its pass-rate gap is the model's: at that size a run separates today's gap four times in five, if today's gap is true._

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

- **Measuring spend across all roles** It is the cost of running the evaluation, not of serving Kairo.
- **The delivery check (a Discord message or a queued voice line sent)** Neither candidate is told apart from the control; it does not change the call.
- **Failed-reply notices, tool provider faults, turns that ran over budget, items delivered** Operational counts with no separated difference that bears on character or cost.
- **The judge's overall outcome and transcript scores** Ordinal summaries of the same judging; they point the same way as the holds-character rating and add nothing to the call.
- **The other kinds of ask beyond the six shown, including story asks where one candidate keeps the booth offstage less often** Each has about ten cases or fewer, too few to read; all stay in the appendix tables.
- **A price per point of character** No candidate is cheaper than the control, so there is no trade to price.

### Method

- **Cases and repeats** 299 conversations, each run three times per arm. The corrected comparisons do not state whether their intervals count conversations or repeats; if repeats, they may be somewhat narrow.
- **What pass means** The campaign does not state the rule that makes a trial pass; the pass rate is trials passed over usable trials. The judge's holds-character rating is a separate figure. The floor on character reads the pass rate for qwen3.8-flash (A) and the holds-character rating for 1017 (B); the two are shown apart, never merged.
- **Intervals and correction** Readings against the control are at 95%. The 16 contrasts are corrected together by Holm's method, each interval at 99.69%. Breakdowns by kind of ask are unpaired, uncorrected and exploratory.
- **Judge** openai/gpt-6-luna, with 0 human calibration ratings; its calibration is insufficient.
- **Cost basis** Production cost replicates what serving Kairo costs, per 1,000 replies. Some trials report no cost: 33 for September (P), 69 for qwen3.8-flash (A), 18 for 1017 (B). Counted as free, the candidates would still cost $2.24 per 1,000 replies and $2.36 per 1,000 replies, above September (P) at $2.02 per 1,000 replies were its missing trials at its p95.
- **Design** One setting, the served model, differs between arms; every gap is credited to it and to whatever it brings, such as its reasoning budget. _Caveats: A confound: model._
- **When the criteria were set** The asked aspect, the control, the allowed loss and the floor were all declared on 25 Oct 2026, the run date, after the results.
- **Not compared** Latency, because every cell ran beside others; token counts and call counts, which were not reported; the simulated user's request settings, which were not recorded on every run. _Caveats: Latency is not compared; Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls; The apparatus may have differed: simulator sent settings._
- **Source** Kairo persona model, round 2: September vs qwen3.8-flash and 1017 on character v2 with the promoted directives, run on 2026-10-25.

### Review findings left open

- **Should fix** · Claim 4: This claim reads 1017 (B)'s holds-character contrast (-4.7, -8.2 to -1.3), which carries the reasoning-share confound: 1017 spent 55% of its output reasoning against September's 76%. Neither the claim nor the confidence line carries it. Add the reasoning-share caveat to the claim. A short clause would do: 1017's off-lane gap (-20.2) may partly reflect its smaller reasoning budget.
- **Should fix** · Claim 4: The claim rests on loss shares, 48% for qwen3.8-flash (A) and 53% for 1017 (B) (strata.off_lane.loss_share.*.persona.holds_character), but neither block shows them. Add the loss shares to the blocks, or restate the claim in terms of the off-lane gap the blocks draw. Also fix the caption 'overall and in each place both hold it': no overall row is drawn. Either add the overall gap (-6.0 and -4.7) as a reference row or drop 'overall'.
- **Should fix** · The page: The lede says 1017 (B) 'is not told apart from September (P) (-5.8 to +1.1)', but that is the pass rate only. On the judged holds-character rating, 1017 is worse (-4.7, -8.2 to -1.3), and the declared floor for 1017 reads that rating. Say 'not told apart on the pass rate, though down 4.7 points on the judged holds-character rating', as the decision block does.
- **Should fix** · The page: The Next line in the decision block says the off-lane asks are 'where most of each candidate's loss sits', but claim offlane gives 48% for qwen3.8-flash (A). That is under half. Say 'about half', and say it is the holds-character loss, not the pass-rate loss.
- **Should fix** · Claim 5: 'September (P) failed 39 off-lane trials, against 64 and 62' counts holds-character failures. A reader may take them for pass-rate failures, which are 132, 200 and 159 overall. Name the rating in the sentence. Either add the counts to the block, or point to claim offlane's off-lane rates (65.8%, 42.9%, 45.6%) as their source.
- **Should fix** · The page: The next step for 1017 (B) asks for '815 cases per arm', but the campaign has 299 cases, run three times each. Say whether 815 means distinct cases at three repeats (which needs 516 new cases) or trials, so the step can be costed.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.06, with analyst skill 1677ccb31cd6, on 2026-10-29._
