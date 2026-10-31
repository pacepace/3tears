Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 2026-10-25

# Keep September (P): both candidates cost about a quarter more per reply (27–28%) and neither keeps Kairo in character better.

qwen3.8-flash (A) passes the character check 7.6 points less often than September (P), clearly worse, and 1017 (B) is not separated from it. Only a candidate that cost no more than September (P) could change the call; cost is measured, so the late and uncalibrated character judge cannot reverse it.

**Confidence: High, if the caveats below aren't material.**

- **The aspect was set after the results.** Only if a different character check, declared in advance, would have favoured a candidate; cost, the deciding finding, is measured and does not depend on the check.
- **The control was set after the results.** Only if September (P) is not the model in production; every cost uplift and character change is read against it.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.
- **The judge is not calibrated.** Only if the judge reads September (P)'s replies as in character more often than equally good candidate replies; even then no candidate is shown better, and cost, the deciding finding, is measured.
- **A confound: model.** Only if a setting other than the served model also moved between arms and made the candidates cost more; cost, the deciding finding, would then belong partly to that setting.

**Recommendation.** Keep September (P), the control, for Kairo. It costs $1.89 per thousand replies, the least of the three, and no candidate holds character better.

**Why not the others.** qwen3.8-flash (A): it costs 28% more and passes the character check less often, a drop whose interval excludes zero.

**Why not the others.** 1017 (B): it costs 27% more for a pass rate not separated from the control's (-5.8 to +1.1 points), and the judge marks it out of character more often.

**Watch.** September (P) itself holds character in only 65.8% of off-lane replies, its weakest place. The prior finding that qwen3.8-flash (A) lowers production cost is contradicted here: it costs 28% more. Its worst plausible drop, -10.8 points, is the figure nearest the margin.

**Next.** Find out why qwen3.8-flash (A) now costs more than the control before any further round on it; a candidate cheaper than September (P) is the only thing that would reopen the call.

| Option | Role | Configuration | Cost per 1,000 replies |
|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | $2.41 (+27%) |

## Both candidates cost more per reply than September (P), and each gap is clear of zero.

| Row | Value | 95% interval | Note |
|---|---|---|---|
| P · September | $1.892 | $1.755 to $2.028 | $1.890 |
| A · qwen3.8-flash | $2.427 | $2.260 to $2.594 | $2.430 |
| B · 1017 | $2.406 | $2.206 to $2.605 | $2.410 |

Production cost per thousand replies, with intervals; the candidates cost 27–28% more than the control.

## qwen3.8-flash (A) passes the character check 7.6 points less often than September (P), clearly worse though within the 15-point margin; 1017 (B) is not separated from the control.

_Caveats: A run fell short of its usable measurements; A run fell short of its usable measurements._

| Row | Change | 95% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | -7.581 | -10.79 to -3.381 | worse |
| B · 1017 | -3.010 | -5.815 to +1.136 | not separated |
| margin -15 pts | -15.00 |  |  |

Change in pass rate against the control, in points, at 95%; the shaded band is the declared margin. Two of qwen3.8-flash (A)'s trials were lost to the harness, so its rate counts slightly fewer trials.

### On the judge's holds-character verdict, both candidates are worse than September (P) after correction; 1017 (B) by 4.7 points.

_Caveats: A confound: reasoning ratio._

| Group | Row | Change | 99.69% interval | Note |
|---|---|---|---|---|
| pass rate | A · qwen3.8-flash | -7.581 | -12.59 to -2.573 | -7.6 pts worse |
| pass rate | B · 1017 | -3.01 | -7.355 to +1.335 | -3.0 pts not separated |
| persona holds character | A · qwen3.8-flash | -5.964 | -10.09 to -1.84 | -6.0 pts worse |
| persona holds character | B · 1017 | -4.738 | -8.199 to -1.277 | -4.7 pts worse |
| persona booth stays offstage | A · qwen3.8-flash | -2.954 | -6.790 to +0.8814 | -3.0 pts not separated |
| persona booth stays offstage | B · 1017 | +0.7804 | -2.356 to +3.916 | +0.8 pts not separated |

Corrected contrasts against the control at 99.69%. 1017 (B) spent 55% of its output reasoning against the control's 76%, so its loss may belong partly to its reasoning budget.

### Off-lane asks, where the user wants non-music help, carry about half of each candidate's holds-character loss (53% for 1017 (B)).

| Group | Row | Value | Note |
|---|---|---|---|
| A · qwen3.8-flash | off lane | -22.93 | -22.9 pts |
| A · qwen3.8-flash | machinery | -4.607 | -4.6 pts |
| A · qwen3.8-flash | about | -2.899 | -2.9 pts |
| A · qwen3.8-flash | superlative | -3.175 | -3.2 pts |
| A · qwen3.8-flash | small talk | -1.961 | -2.0 pts |
| B · 1017 | off lane | -20.18 | -20.2 pts |
| B · 1017 | machinery | -4.444 | -4.4 pts |
| B · 1017 | about | 0.000 | 0.0 pts |
| B · 1017 | superlative | -1.515 | -1.5 pts |
| B · 1017 | small talk | 0.000 | 0.0 pts |

Changes computed from each arm's figures, uncorrected. Bold rows are separated at 95% before any correction; read them as leads.

Holds-character rate minus the control's in the five largest strata, in points; uncorrected, so read as leads.

#### Most kept off-lane failures show one pattern: the reply brushes off the non-music ask, then answers it anyway.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T15 · oft-match-score · off lane · 1017 · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first reply brushes off the match question and pivots to Joy Orbison in Kairo’s dry DJ voice. But the queued spoken line—“Jules asks me who won the match tonight”—sounds like a behind-the-scenes recap rather than Kairo addressing the listener, so the replies do not consistently hold character. | …usand seven. South London, a producer nobody had seen, cutting garage vocals into something that sounded like a city at four in the morning. Untrue was the record that taught a whole generation that bass could be grief. Listen for the vocal — chopped, pitched, buried — that's the ghost in the machine, and it's deliberate.", "track_ref": "r3"}) → **Queued #5 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)* |
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |
| T67 · oft-wash-jeans · off lane · 1017 · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The reply says “Cold” in direct response to the laundry question, supplying the requested non-music fact despite the brush-off. That alone fails the persona check. | …ason half the records I play sound the way they do.", "track_ref": "r3"}) → **Queued #4 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)*   ✓ music.set_channel_mood({"mood": "UK bass lineage at 9pm — garage warmth drifting into dub-techno haze and Burial's rain-soaked small hours"}) → Channel mood set: UK bass lineage at 9pm — garage warmth drifting into dub-techno haze and Burial's rain-soaked small hours |
| T70 · oft-wash-jeans · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for non-music advice about washing jeans, and Kairo supplied a temperature and instructions: “cold wash, 30 max, inside out.” That gives the requested off-lane advice, so it fails the stated check. | ## Jules what temp do i wash jeans at  ## Persona cold wash, 30 max, inside out — hot water's how you get crease fades you never asked for 🧊  that sorted, you want me to stay in this dub haze or push it harder tonight?  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "cold wash, 30 max, inside out — hot water's how you get crease fades you never asked for 🧊  that sorted, you wa… |

Four kept cards from the stratum where the loss concentrates, out of 64 off-lane failures for qwen3.8-flash (A) and 62 for 1017 (B).

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The margin a candidate may fall behind the control on the character check | Whether each candidate counts as no worse than the control | The call holds at every margin. 1017 (B) counts as no worse above 5.8 points and qwen3.8-flash (A) above 10.8; neither changes the call, because both cost more. |

## Next steps

- Reconcile qwen3.8-flash (A)'s cost with the earlier round that found it cheaper: compare the pricing and output length behind each. _Whether it is cheaper than the control anywhere; if so, it becomes a priced tradeoff rather than a rejected option._
- Re-run 1017 (B) with its reasoning budget set to match the control's share of output spent reasoning. _Whether its holds-character loss belongs to the model or to its budget, and what it costs at that budget._
- Tighten the directive to brush off non-music asks without answering them, and re-run the off-lane cases on all three models. _Whether the largest share of the loss is a prompt issue every model shares, the control included._
- Calibrate the judge against human ratings, starting with off-lane replies. _Whether the judged character losses are real or a judge preference._
- Declare the character check, the control and the margin before the next round runs. _Removes the late-declaration caveats from the next comparison._

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
| A · qwen3.8-flash | Production cost | +$0.62 | +$0.38 to +$0.85 | 99.69% | < 0.000001 | Regressed | — |
| A · qwen3.8-flash | outcome | -0.2129 | -0.3065 to -0.1194 | 99.69% | < 0.000001 | Regressed | — |
| B · 1017 | Production cost | +$0.49 | +$0.17 to +$0.81 | 99.69% | 0.00008342 | Regressed | observed:reasoning ratio |
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

- **The judge's ordinal outcome and transcript scores** They point the same way as the pass rate on an uncalibrated scale, and add nothing to the call.
- **The send-a-reply-or-queue-a-voice tool check** Not separated for either candidate; it does not bear on the call.
- **Failed-reply notices, tool provider faults, turns over budget and items delivered** At or near zero for every arm; they do not separate the options.
- **Measuring spend across all roles** Describes what the test cost to run, not what production would pay.
- **Strata of about a dozen cases or fewer, such as story and multi** Too small to read; they stay in the full tables.
- **Latency** Read while other runs executed beside it, so it cannot be compared.

### Method

- **Cases and repeats** 299 conversations, each run 3 times per model; intervals count conversations, not repeats.
- **What pass means** The campaign does not state what makes a trial pass on the character check; the pass rate counts trials marked passed over usable trials.
- **Intervals** Readings against the control are at 95%. Contrasts are corrected together by Holm's method across 16 comparisons, each interval at 99.69%. Stratum breakdowns are uncorrected.
- **Judge** Character verdicts come from openai/gpt-6-luna, with 0 human calibration ratings.
- **Cost basis** Cost replicates what production would pay per reply, shown per thousand replies. No monthly volume was declared, so no monthly figure is given.
- **Simulator settings** Whether the simulated user's request settings were the same on every run was not recorded. _Caveats: The apparatus may have differed: simulator sent settings._
- **Not compared** Latency, input and output tokens and model call counts were not compared. _Caveats: Latency is not compared; Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls._

### Review findings left open

- **Should fix** · Claim 1: The snapshot attaches the reasoning-ratio confound to 1017's production-cost contrast (+$0.49 per thousand replies, +$0.17 to +$0.81). Neither the cost claim nor the 1017 'why not' line carries it, and the lede and confidence line say cost is measured and cannot be reversed. Attach the confound to 1017's cost and say its likely direction. 1017 spends 55% of its output reasoning against the control's 76%, so a matched budget would likely raise its cost, not lower it. Point to the matched-budget next step.
- **Should fix** · Claim 1: The block shows each arm's cost level, but the claim is that each gap is clear of zero. Show the corrected paired gaps instead: +$0.62 (+$0.38 to +$0.85) for qwen3.8-flash and +$0.49 (+$0.17 to +$0.81) for 1017. Reconcile the 27–28% in the headline, which comes from unpaired means, with the paired gap of about 33% for qwen3.8-flash. Also state cost coverage. Cost was read on 864 of 897 trials for September, 828 of 897 (285 of 299 cases) for qwen3.8-flash and 879 of 897 for 1017. Add that even if every missing reading cost zero, the order would not change.
- **Should fix** · Claim 2: The caption writes 'Two of qwen3.8-flash (A)'s trials' in words. Print the count from its key, the lost count, 2 (or usable 895 of 897). The caveat line also repeats 'A run fell short of its usable measurements' twice. Say plainly that two qwen3.8-flash runs each lost one trial to the harness.
- **Should fix** · Claim 5: All four cards are candidate failures. Yet September fails off-lane in the same way: T42 offers a landlord-email opener after a brush-off, and T64 answers 'Cold'. That shared failure is the basis of the third next step. Swap in one control card. Also note that T15 is not the brush-off-then-answer pattern: the judge faulted a behind-the-scenes spoken line. Say 'three of four' rather than letting the card imply the pattern.
- **Should fix** · Claim 4: The claim gives 53% for 1017 only. Qwen3.8-flash's off-lane share is 48%; state both. The caption says 'bold rows are separated at 95%', but the block shows no intervals or bold marking. Either add the stratum intervals or drop that sentence.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.60, with analyst skill b70be6bcfa68, on 2026-10-28._
