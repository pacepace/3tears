Change · control vs candidates · Kairo's character across persona models · September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats · Run 25 Oct 2026

# Keep September (P), the control: as served in this run, qwen3.8-flash (A) and 1017 (B) each cost more per reply, and neither holds Kairo in character any better.

As served in this run, with no provider recorded, production cost rises by $0.62 per 1,000 replies and $0.49 per 1,000 replies, both clear of zero after correction. Only a candidate cheaper than the control on production routing would reopen the question.

**Costs are as served in this run: no arm records the provider that served it, and an earlier finding says qwen3.8-flash costs less than the control.**

**Confidence: High, if the caveats below aren't material.**

- **No arm records the provider that served it.** Only if, on the routing production would use, a candidate cost less than the control's $1.89 per 1,000 replies. Until a re-pricing shows that, keeping the control holds; and even then the call would become a trade, not a switch, because after correction qwen3.8-flash (A) passes 8 points fewer trials and 1017 (B) holds character 5 points less often.
- **The judge is not calibrated.** Only if a candidate also proved cheaper on production routing: then whether its character loss is real would decide the call. In this run the cost order, not the judge, decides it; a judge that misread a candidate's style could shrink or erase its character loss, but would not make it cheaper.
- **The aspect was set after the results.** Only if a different aspect should have been asked before the run. The cost order that decides the call holds whatever aspect is read.
- **The control was set after the results.** Only if another arm should have been the control. September (P) is the least costly of the three arms as served either way.
- **A confound: model.** Only if something besides the served model moved between arms. Two things did or may have: 1017 (B)'s reasoning share, carried beside its claims, and the provider, unrecorded for every arm and carried above.
- (checked) **The margin was set after the results.** The late margin was checked: the call holds at every margin.
- (checked) **The floor was set after the results.** The late floor was checked: the call holds at every margin the floor could take.

**Recommendation.** Keep September (P) on all traffic. As served in this run it is the least costly arm, at $1.89 per 1,000 replies, and neither candidate shows a gain in character to set against its higher cost.

**Why not the others.** qwen3.8-flash (A) costs $0.62 per 1,000 replies more as served and passes fewer trials, down 8 points (-13 to -2 after correction). 1017 (B) costs $0.49 per 1,000 replies more as served and holds character less often, down 5 points after correction, though part of that may come from how much it reasoned; on pass rate the evidence cannot tell it and the control apart.

**Watch.** The weakest margin is qwen3.8-flash (A)'s on the margin on character adherence (v2): at the far end of the 95% interval, adjusted for 16 comparisons, its pass-rate drop reaches 13 points, 2 points inside the line. It bears on a rejected option, not on the call. A prior insight held that qwen3.8-flash lowers production costs; here, as served, it costs more. No arm names its provider, so whether that is the model or this run's routing is unknown: it is the risk the call carries.

**Next.** Price both candidates on the provider routing production would use. If neither falls below the control's cost, the call stands; if one does, its character loss becomes a trade to price.

| Option | Role | Configuration | Pass rate | Cost per 1,000 replies |
|---|---|---|---|---|
| P · September | Control (recommended) | deepseek/deepseek-v4.1-flash | 85% (765 of 897) | $1.89 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | 78% (695 of 895) | $2.43 (+28%) |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | 82% (738 of 897) | $2.41 (+27%) |

## As served in this run, both candidates cost more per reply than the control, and the paired intervals exclude zero for each: $0.62 per 1,000 replies more for qwen3.8-flash (A), $0.49 per 1,000 replies more for 1017 (B). 1017 (B) spent a smaller share of its output reasoning (55% against the control's 76%), so its comparison is not at equal reasoning; matching the control's share would more likely widen its cost gap than close it.

_Caveats: A confound: reasoning ratio._

| Row | Change | 95% interval (adjusted for 16 comparisons) | Note |
|---|---|---|---|
| A · qwen3.8-flash | +$0.62 | +$0.38 to +$0.85 | +$0.62 worse |
| B · 1017 | +$0.49 | +$0.17 to +$0.81 | +$0.49 worse |

Production cost, per 1,000 replies, in dollars.

Production cost per 1,000 replies, each candidate minus September (P), paired by case over the cases where both arms recorded a cost (284 and 293), intervals adjusted for the family's comparisons; these gaps differ from the gaps between the arms' means in the options table. No arm records the provider behind its cost.

### On the uncorrected readings against the control, qwen3.8-flash (A)'s pass rate is down 8 points, clearly worse, and the evidence cannot tell 1017 (B) and the control apart (-6 to +1 points); both drops stay within the 15-point margin.

_Caveats: A run fell short of its usable measurements — two of them, on qwen3.8-flash (A)._

| Row | Change | 95% interval | Note |
|---|---|---|---|
| A · qwen3.8-flash | -8 pts | -11 to -3 pts | worse |
| B · 1017 | -3 pts | -6 to +1 pts | cannot tell apart |
| margin -15 pts; shaded: no worse than the margin | -15 pts |  |  |

Change in pass rate against September (P): each candidate's own reading, uncorrected, with the margin drawn. qwen3.8-flash (A) lost two trials to harness failures, too few to move a gap this size.

### After correction, both candidates hold character less often than the control: qwen3.8-flash (A) by 6 points and 1017 (B) by 5 points. Part of 1017 (B)'s loss may come from how much it reasoned rather than from the model.

_Caveats: A confound: reasoning ratio._

| Group | Row | Change | 95% interval (adjusted for 16 comparisons) | Note |
|---|---|---|---|---|
| Pass rate · September 85% | A · qwen3.8-flash | -8 pts | -13 to -2 pts | -8 pts worse |
| Pass rate · September 85% | B · 1017 | -3 pts | -7 to +3 pts | -3 pts cannot tell apart |
| Persona holds character · September 91% | A · qwen3.8-flash | -6 pts | -12 to -2 pts | -6 pts worse |
| Persona holds character · September 91% | B · 1017 | -5 pts | -9 to -1 pts | -5 pts worse |
| Persona booth stays offstage · September 93% | A · qwen3.8-flash | -3 pts | -8 to +2 pts | -3 pts cannot tell apart |
| Persona booth stays offstage · September 93% | B · 1017 | +1 pts | -3 to +5 pts | +1 pts cannot tell apart |

Each candidate minus September (P), paired by case, intervals adjusted for the family's comparisons; pass rate here is the corrected contrast, wider than the reading above. On keeping the booth offstage the evidence cannot tell either candidate and the control apart.

#### Holding character falls furthest on off-lane asks: qwen3.8-flash (A) is 23 points below the control there and 1017 (B) 20. It is a lead, uncorrected, and part of 1017 (B)'s drop may come from how much it reasoned.

_Caveats: A confound: reasoning ratio._

| Place | First arm | Second arm | Difference | Interval | Level | Read as |
|---|---|---|---|---|---|---|
| off lane | 43% | 66% | -23 | -41 to -5 | 95% | independent: each arm's own spread (its standard error), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |

qwen3.8-flash (895 trials of 299 cases); September (897 trials of 299 cases) ran more trials than cases, and trials of one case are not independent: an interval here may be too narrow if these intervals count trials nested in cases. The input states what an unpaired interval's arms count: observations.

qwen3.8-flash (A) minus September (P) on off-lane asks, an unpaired interval that may be too narrow because repeats of one case are not independent. Off-lane carries 48% of this candidate's loss on holding character.

| Place | First arm | Second arm | Difference | Interval | Level | Read as |
|---|---|---|---|---|---|---|
| off lane | 46% | 66% | -20 | -37 to -3 | 95% | independent: each arm's own spread (its standard error), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |

1017 (897 trials of 299 cases); September (897 trials of 299 cases) ran more trials than cases, and trials of one case are not independent: an interval here may be too narrow if these intervals count trials nested in cases. The input states what an unpaired interval's arms count: observations.

1017 (B) minus September (P) on off-lane asks, read the same way. Off-lane carries 53% of this candidate's loss on holding character.

#### The control fails off-lane asks too: of the off-lane trials, three per case over 38 cases, 42 failed for any cause on September (P), against 68 on qwen3.8-flash (A) and 63 on 1017 (B). Off-lane is a weakness of the persona, not only of the candidates.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| T15 · oft-match-score · off lane · 1017 · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first reply brushes off the match question and pivots to Joy Orbison in Kairo’s dry DJ voice. But the queued spoken line—“Jules asks me who won the match tonight”—sounds like a behind-the-scenes recap rather than Kairo addressing the listener, so the replies do not consistently hold character. | …usand seven. South London, a producer nobody had seen, cutting garage vocals into something that sounded like a city at four in the morning. Untrue was the record that taught a whole generation that bass could be grief. Listen for the vocal — chopped, pitched, buried — that's the ghost in the machine, and it's deliberate.", "track_ref": "r3"}) → **Queued #5 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)* |
| T44 · oft-landlord-email · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for help with a non-music task, and Kairo gave practical advice to put the breakage date in writing, keep replies, and said the boiler was the landlord’s job. The musical turn to King Tubby does not undo that out-of-lane answer. | …t man's room.", "track_ref": "r1"}) → **Queued #4 (audio resolving):** Zion Gate Dub by King Tubby - Topic *(3:13)*   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "instead of a boiler, you get steam from the right kind — going back to the source for you. king tubby, zion gate dub. the whole tonight's uk bass line runs through this man's mixing desk 🇯🇲"}) → Message sent to Discord via webhook |
| T14 · oft-match-score · off lane · September · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The first line brushes off the score in Kairo’s DJ voice, but the follow-up says “Scores and fixtures, I can’t help you with — but I got something for you,” which shifts into customer-service-style inability and an offer of help. That makes the delivered replies slip out of the persona. | …g):** Carrier by Rhythm & Sound - Topic *(9:39)*   ✓ music.queue_voice({"acting_instructions": "warm, dry, a little amused", "position": "head", "text": "Jules, welcome in. Scores and fixtures, I can't help you with — but I got something for you. Overmono, Anz, and a bit of Berlin dub coming through the speakers. Sit down."}) → queued voice (i5): Jules, welcome in. Scores and fixtures, I can't help you wit… — this … |
| T67 · oft-wash-jeans · off lane · 1017 · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The reply says “Cold” in direct response to the laundry question, supplying the requested non-music fact despite the brush-off. That alone fails the persona check. | …ason half the records I play sound the way they do.", "track_ref": "r3"}) → **Queued #4 (audio resolving):** Burial, Archangel by Hyperdub *(3:59)*   ✓ music.set_channel_mood({"mood": "UK bass lineage at 9pm — garage warmth drifting into dub-techno haze and Burial's rain-soaked small hours"}) → Channel mood set: UK bass lineage at 9pm — garage warmth drifting into dub-techno haze and Burial's rain-soaked small hours |
| T70 · oft-wash-jeans · off lane · qwen3.8-flash · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Jules asked for non-music advice about washing jeans, and Kairo supplied a temperature and instructions: “cold wash, 30 max, inside out.” That gives the requested off-lane advice, so it fails the stated check. | ## Jules what temp do i wash jeans at  ## Persona cold wash, 30 max, inside out — hot water's how you get crease fades you never asked for 🧊  that sorted, you want me to stay in this dub haze or push it harder tonight?  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "cold wash, 30 max, inside out — hot water's how you get crease fades you never asked for 🧊  that sorted, you wa… |
| T42 · oft-landlord-email · off lane · September · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Kairo starts with an in-character brush-off, but then supplies a suggested opening for the landlord email: “Open with ‘the boiler’s dead again’.” That is help with a non-music task, so the reply fails the stated check. | …cs do the shouting. Meanwhile the Joy Orbison's keeping this room warm, so settle in. 🔥  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "Love, I mix records, not grievances — that email's all you. Open with "the boiler's dead *again*" and let the italics do the shouting. Meanwhile the Joy Orbison's keeping this room warm, so settle in. 🔥"}) → Message sent to Discord via webhook |
| T77 · oft-which-phone · off lane · 1017 · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | Kairo opens with an in-character brush-off, but then gives purchase advice: “Buy whatever plays the bass cleanest.” The persona’s directive is to brush off non-music asks without answering them, and the brush-off does not excuse advice that follows. | ## Jules which phone should i buy, iphone or android  ## Persona Jules, love — do I look like a phone salesman? Only thing I care about a phone for is whether it holds a crate of dub. Buy whatever plays the bass cleanest and come talk to me about music.  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "Jules, love — do I look like a phone salesman? Only thing I care about a pho… |
| T79 · oft-which-phone · off lane · qwen3.8-flash · judged · repeat 2 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The reply brushes off “phone wars” but then answers the non-music ask by recommending Android and giving phone-feature comparisons (“LDAC/aptX,” file management, and iPhone Bluetooth). A brush-off does not excuse supplying the requested recommendation, so this fails criterion (1). | ## Jules which phone should i buy, iphone or android  ## Persona phone wars? not my lane, love. but for *listening* — Android. LDAC/aptX codecs, sideload your FLACs, actual file management. iPhone's Bluetooth is stuck on AAC and always will be. unless the ecosystem's the whole point, then do you.  Tool actions:   ✓ discord.send_message({"channel_id": 1503186183688032286, "content": "phone wars? not my lane, love. b… |
| T64 · oft-wash-jeans · off lane · September · judged · repeat 1 | Input: The world as it stood before the persona's first turn:  ## music {   "catalog": [     {       "artist": "Two Shell",       "tags": "uk electronic",       "title": "home"     },     {       "artist": "Bakey",       "tags": "uk garage",       "title": "Pinball"     }   ],   "history": [     {       "artist": "Loraine James",       "duration_ms": 165000,       "minutes_ago": 9.4,       "title": "2003",       "uri": "ht… | The reply brushes off the laundry question, but then answers it with “Cold,” giving the requested non-music advice. That triggers failure (1), even though the rest of the line stays in Kairo’s DJ voice. | …how little they play — that's the discipline.", "track_ref": "r10"}) → **Queued #5 (audio resolving):** Rhythm & Sound w Tikiman – Never Tell You 1996 (vinyl record) by vinyl record *(12:39)*   ✓ music.set_channel_mood({"mood": "rain-soaked 2-step and Berlin dub — late-night London dissolving into deepest space"}) → Channel mood set: rain-soaked 2-step and Berlin dub — late-night London dissolving into deepest space |

Off-lane trials failed for any cause, of three per case over 38 cases: 42 on September (P), 68 on qwen3.8-flash (A), 63 on 1017 (B). The cards are every kept one, three per arm, each a nought on holding character, so none shows a trial a check failed. The two match-score cards slip out of the DJ voice in a spoken line; most other cards answer the off-lane ask after brushing it off, one answers it outright, and the laundry cards on every arm, the control's included, give the wash temperature.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The margin on character adherence | Whether each candidate counts as no worse than the control on pass rate | The call to keep September (P) holds at every margin, because it rests on cost. On the uncorrected readings, 1017 (B) stops reading as within the margin below 6 points and qwen3.8-flash (A) below 11; on the corrected contrast Watch cites, qwen3.8-flash (A)'s drop reaches 13. |

## Next steps

- Price qwen3.8-flash (A) and 1017 (B) on the provider routing production would use, and reconcile with the earlier finding that qwen3.8-flash lowers cost. _Whether either candidate can cost less than the control; only then is its character loss a trade worth pricing._
- Calibrate the judge against human ratings on holding character, weighted toward off-lane asks. _Whether the candidates' character losses, and the control's off-lane failures, are real or the judge's reading._
- Rerun 1017 (B) with its reasoning share matched to the control's. _Whether its cost and character gaps belong to the model or to how much it reasoned._
- Revise the off-lane directive so a brush-off carries no answer, and rerun the off-lane asks on all arms. _Whether the stratum that carries the most loss for every arm, the control included, can lift._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model | Share of output spent reasoning | Trials usable |
|---|---|---|---|---|---|
| P · September | Control | deepseek/deepseek-v4.1-flash | deepseek/deepseek-v4.1-flash | 76% | 897 of 897 |
| A · qwen3.8-flash | Candidate | qwen/qwen3.8-flash | qwen/qwen3.8-flash | 68% | 895 of 897 |
| B · 1017 | Candidate | deepseek/deepseek-v4.1-flash-1017 | deepseek/deepseek-v4.1-flash-1017 | 55% | 897 of 897 |

### Every arm by measure

| Measure | P · September | A · qwen3.8-flash | B · 1017 |
|---|---|---|---|
| Pass rate | 85% (765 of 897) | 78% (695 of 895) | 82% (738 of 897) |
| Outcome (1–5) | 4.69 | 4.48 | 4.71 |
| Transcript (1–5) | 2.94 | 2.81 | 2.96 |
| Persona booth stays offstage | 93% | 90% | 94% |
| Persona holds character | 91% | 85% | 87% |
| Measuring spend, all roles, per 1,000 replies | $5.40 | $6.28 | $5.76 |
| Items delivered | 0 items | 0 items | 0 items |
| Failed-reply notices | 0.002 notices | 0 notices | 0 notices |
| Send message or queue voice check | 99.7% | 98.7% | 100.0% |
| Production cost, per 1,000 replies | $1.89 | $2.43 | $2.41 |
| Tool provider faults | 0 faults | 0 faults | 0 faults |
| Turns that ran over budget | 0 turns | 0.008 turns | 0 turns |

### Every corrected comparison

| Arm | Measure | Change | Interval | Level | Adjusted p | Verdict | Confounds |
|---|---|---|---|---|---|---|---|
| A · qwen3.8-flash | Production cost, per 1,000 replies | +$0.62 | +$0.38 to +$0.85 | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | — |
| A · qwen3.8-flash | Outcome (1–5) | -0.21 | -0.31 to -0.12 | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | — |
| B · 1017 | Production cost, per 1,000 replies | +$0.49 | +$0.17 to +$0.81 | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | observed:reasoning ratio |
| A · qwen3.8-flash | Pass rate | -8 pts | -13 to -2 pts | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | — |
| A · qwen3.8-flash | Persona holds character | -6 pts | -12 to -2 pts | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | — |
| B · 1017 | Persona holds character | -5 pts | -9 to -1 pts | 95% (adjusted for 16 comparisons) | < 0.001 | Regressed | observed:reasoning ratio |
| A · qwen3.8-flash | Transcript (1–5) | -0.13 | -0.24 to -0.02 | 95% (adjusted for 16 comparisons) | 0.005 | Regressed | — |
| A · qwen3.8-flash | Send message or queue voice check | -1.0 pts | -3.6 to +1.6 pts | 95% (adjusted for 16 comparisons) | 0.11 | Cannot tell apart | — |
| A · qwen3.8-flash | Persona booth stays offstage | -3 pts | -8 to +2 pts | 95% (adjusted for 16 comparisons) | 0.18 | Cannot tell apart | — |
| B · 1017 | Pass rate | -3 pts | -7 to +3 pts | 95% (adjusted for 16 comparisons) | 0.28 | Cannot tell apart | observed:reasoning ratio |
| B · 1017 | Send message or queue voice check | +0.3 pts | -2.1 to +2.8 pts | 95% (adjusted for 16 comparisons) | 0.50 | Cannot tell apart | observed:reasoning ratio |
| B · 1017 | Failed-reply notices | -0.002 notices | -0.007 to +0.002 notices | 95% (adjusted for 16 comparisons) | 0.79 | Cannot tell apart | observed:reasoning ratio |
| A · qwen3.8-flash | Failed-reply notices | -0.002 notices | -0.007 to +0.002 notices | 95% (adjusted for 16 comparisons) | 0.79 | Cannot tell apart | — |
| B · 1017 | Outcome (1–5) | +0.01 | -0.06 to +0.09 | 95% (adjusted for 16 comparisons) | 1.00 | Cannot tell apart | observed:reasoning ratio |
| B · 1017 | Transcript (1–5) | +0.02 | -0.09 to +0.12 | 95% (adjusted for 16 comparisons) | 1.00 | Cannot tell apart | observed:reasoning ratio |
| B · 1017 | Persona booth stays offstage | +1 pts | -3 to +5 pts | 95% (adjusted for 16 comparisons) | 1.00 | Cannot tell apart | observed:reasoning ratio |

### Every stratum

| Stratum | Measure | Cases | P · September | A · qwen3.8-flash | B · 1017 |
|---|---|---|---|---|---|
| off lane | Outcome (1–5) | 38 | 4.75 | 4.75 | 4.74 |
| off lane | Transcript (1–5) | 38 | 2.82 | 3.17 | 2.74 |
| off lane | Persona booth stays offstage | 38 | 96% | 96% | 98% |
| off lane | Persona holds character | 38 | 66% | 43% | 46% |
| machinery | Outcome (1–5) | 30 | 4.70 | 4.51 | 4.69 |
| machinery | Transcript (1–5) | 30 | 2.70 | 2.87 | 2.88 |
| machinery | Persona booth stays offstage | 30 | 69% | 75% | 74% |
| machinery | Persona holds character | 30 | 90% | 85% | 86% |
| about | Outcome (1–5) | 23 | 4.78 | 4.64 | 4.80 |
| about | Transcript (1–5) | 23 | 2.91 | 2.71 | 2.97 |
| about | Persona booth stays offstage | 23 | 99% | 97% | 99% |
| about | Persona holds character | 23 | 99% | 96% | 99% |
| superlative | Outcome (1–5) | 22 | 4.82 | 4.14 | 4.80 |
| superlative | Transcript (1–5) | 22 | 3.65 | 2.77 | 3.29 |
| superlative | Persona booth stays offstage | 22 | 95% | 90% | 95% |
| superlative | Persona holds character | 22 | 100% | 97% | 98% |
| small talk | Outcome (1–5) | 17 | 4.67 | 4.57 | 4.78 |
| small talk | Transcript (1–5) | 17 | 2.78 | 2.67 | 2.80 |
| small talk | Persona booth stays offstage | 17 | 88% | 92% | 98% |
| small talk | Persona holds character | 17 | 98% | 96% | 98% |
| multi | Outcome (1–5) | 12 | 4.83 | 4.53 | 4.78 |
| multi | Transcript (1–5) | 12 | 2.72 | 2.69 | 3.19 |
| multi | Persona booth stays offstage | 12 | 97% | 92% | 94% |
| multi | Persona holds character | 12 | 72% | 64% | 58% |
| emotional | Outcome (1–5) | 11 | 4.76 | 4.58 | 4.61 |
| emotional | Transcript (1–5) | 11 | 3.42 | 3.09 | 3.18 |
| emotional | Persona booth stays offstage | 11 | 97% | 91% | 100% |
| emotional | Persona holds character | 11 | 97% | 94% | 94% |
| lineage | Outcome (1–5) | 10 | 4.50 | 4.20 | 4.57 |
| lineage | Transcript (1–5) | 10 | 3.00 | 2.70 | 2.97 |
| lineage | Persona booth stays offstage | 10 | 100% | 93% | 100% |
| lineage | Persona holds character | 10 | 97% | 97% | 97% |
| pushback | Outcome (1–5) | 10 | 4.80 | 4.47 | 4.70 |
| pushback | Transcript (1–5) | 10 | 2.93 | 2.70 | 3.00 |
| pushback | Persona booth stays offstage | 10 | 100% | 97% | 100% |
| pushback | Persona holds character | 10 | 100% | 100% | 100% |
| rejection | Outcome (1–5) | 10 | 4.73 | 4.60 | 4.83 |
| rejection | Transcript (1–5) | 10 | 2.97 | 2.60 | 3.10 |
| rejection | Persona booth stays offstage | 10 | 97% | 93% | 97% |
| rejection | Persona holds character | 10 | 100% | 100% | 100% |
| short | Outcome (1–5) | 10 | 4.57 | 4.70 | 4.73 |
| short | Transcript (1–5) | 10 | 2.83 | 2.77 | 2.83 |
| short | Persona booth stays offstage | 10 | 97% | 97% | 97% |
| short | Persona holds character | 10 | 100% | 97% | 97% |
| story | Outcome (1–5) | 10 | 4.37 | 3.90 | 4.40 |
| story | Transcript (1–5) | 10 | 2.80 | 2.30 | 2.53 |
| story | Persona booth stays offstage | 10 | 93% | 53% | 97% |
| story | Persona holds character | 10 | 93% | 83% | 97% |
| slang | Outcome (1–5) | 9 | 4.93 | 4.41 | 4.85 |
| slang | Transcript (1–5) | 9 | 2.93 | 2.67 | 2.85 |
| slang | Persona booth stays offstage | 9 | 96% | 96% | 100% |
| slang | Persona holds character | 9 | 100% | 93% | 100% |
| ai opinion | Outcome (1–5) | 8 | 4.75 | 4.67 | 4.62 |
| ai opinion | Transcript (1–5) | 8 | 3.08 | 2.83 | 2.88 |
| ai opinion | Persona booth stays offstage | 8 | 96% | 96% | 96% |
| ai opinion | Persona holds character | 8 | 92% | 96% | 92% |
| discovery | Outcome (1–5) | 8 | 4.33 | 4.08 | 4.54 |
| discovery | Transcript (1–5) | 8 | 2.75 | 2.50 | 2.88 |
| discovery | Persona booth stays offstage | 8 | 87% | 83% | 88% |
| discovery | Persona holds character | 8 | 100% | 96% | 96% |
| identify | Outcome (1–5) | 8 | 4.50 | 4.46 | 4.71 |
| identify | Transcript (1–5) | 8 | 2.62 | 2.79 | 3.04 |
| identify | Persona booth stays offstage | 8 | 92% | 92% | 100% |
| identify | Persona holds character | 8 | 100% | 100% | 100% |
| next up | Outcome (1–5) | 8 | 4.67 | 4.42 | 4.58 |
| next up | Transcript (1–5) | 8 | 3.38 | 2.96 | 3.29 |
| next up | Persona booth stays offstage | 8 | 88% | 83% | 79% |
| next up | Persona holds character | 8 | 100% | 83% | 83% |
| music tech | Outcome (1–5) | 7 | 4.38 | 4.38 | 4.57 |
| music tech | Transcript (1–5) | 7 | 2.67 | 2.71 | 2.71 |
| music tech | Persona booth stays offstage | 7 | 100% | 100% | 100% |
| music tech | Persona holds character | 7 | 100% | 86% | 95% |
| taste | Outcome (1–5) | 7 | 4.67 | 4.38 | 4.76 |
| taste | Transcript (1–5) | 7 | 3.05 | 2.86 | 3.10 |
| taste | Persona booth stays offstage | 7 | 95% | 100% | 100% |
| taste | Persona holds character | 7 | 100% | 100% | 100% |
| assistant shape | Outcome (1–5) | 6 | 4.28 | 4.17 | 4.44 |
| assistant shape | Transcript (1–5) | 6 | 3.00 | 2.67 | 3.11 |
| assistant shape | Persona booth stays offstage | 6 | 94% | 83% | 78% |
| assistant shape | Persona holds character | 6 | 59% | 44% | 44% |
| correction | Outcome (1–5) | 6 | 4.83 | 4.56 | 4.78 |
| correction | Transcript (1–5) | 6 | 2.83 | 3.17 | 3.22 |
| correction | Persona booth stays offstage | 6 | 100% | 94% | 100% |
| correction | Persona holds character | 6 | 100% | 94% | 100% |
| debate | Outcome (1–5) | 6 | 4.83 | 4.67 | 4.89 |
| debate | Transcript (1–5) | 6 | 3.50 | 2.94 | 2.94 |
| debate | Persona booth stays offstage | 6 | 100% | 94% | 89% |
| debate | Persona holds character | 6 | 100% | 100% | 94% |
| praise | Outcome (1–5) | 6 | 4.78 | 4.33 | 4.56 |
| praise | Transcript (1–5) | 6 | 2.61 | 2.61 | 2.78 |
| praise | Persona booth stays offstage | 6 | 100% | 83% | 94% |
| praise | Persona holds character | 6 | 100% | 100% | 100% |
| request | Outcome (1–5) | 6 | 4.83 | 4.28 | 4.72 |
| request | Transcript (1–5) | 6 | 3.17 | 2.94 | 3.33 |
| request | Persona booth stays offstage | 6 | 94% | 100% | 89% |
| request | Persona holds character | 6 | 100% | 100% | 94% |
| sequence | Outcome (1–5) | 6 | 4.67 | 4.61 | 4.72 |
| sequence | Transcript (1–5) | 6 | 2.83 | 2.83 | 2.78 |
| sequence | Persona booth stays offstage | 6 | 100% | 83% | 94% |
| sequence | Persona holds character | 6 | 94% | 100% | 100% |
| ai mechanics | Outcome (1–5) | 5 | 4.73 | 4.67 | 4.87 |
| ai mechanics | Transcript (1–5) | 5 | 2.47 | 2.73 | 3.27 |
| ai mechanics | Persona booth stays offstage | 5 | 87% | 93% | 87% |
| ai mechanics | Persona holds character | 5 | 87% | 87% | 87% |

### Left out, and why

- **The judge's overall outcome and transcript scores** They point the same way as the pass rate (qwen3.8-flash (A) lower, 1017 (B) cannot be told apart) and do not change the call.
- **The send-message-or-queue-voice check and failed-reply notices** The evidence cannot tell either candidate and the control apart on them, and neither bears on cost.
- **Measuring spend across all roles** It counts the evaluation's own spend, judge and simulator included, not what production would pay.
- **Strata other than off-lane** No other stratum's drop is large enough to lead; most hold ten cases or fewer. They stay in the full tables.
- **Items delivered, tool provider faults and turns that ran over budget** Telemetry that describes the run, not the persona; none bears on the call.
- **Cost uplift as a percentage, in the claims** The options table prints it, read off unpaired means; the claims cite the paired dollar gaps instead, and Method reconciles the two.

### Method

- **Cases and repeats** 299 cases, each run three times per arm; intervals count cases, not trials, except the off-lane comparisons, which may be too narrow.
- **What a pass means** A trial passes when its candidate did not fail, every goal check passed and every judged dimension scored at least 3 of 5.
- **Intervals** Readings against the control are each candidate's own, uncorrected, at 95%. Contrasts are paired by case and corrected together by Holm's method, each at 99.69%, so the whole family's intervals hold together; they read wider than the readings.
- **Judge** Every judged dimension was scored by openai/gpt-6-luna, with calibration insufficient and zero human ratings. _Caveats: The judge is not calibrated._
- **Cost basis** Production cost per 1,000 replies, as served in this run; no arm names the provider that served it. The cost gaps the page leads with are paired by case, over the 284 and 293 cases where both arms recorded a cost, so they differ from the gaps between the arms' means ($2.43 per 1,000 replies and $2.41 per 1,000 replies against $1.89 per 1,000 replies); the options table's percentages are read off those unpaired means. Trials with no cost: 33 for the control, 69 for qwen3.8-flash (A), eighteen for 1017 (B). Had each such candidate trial cost nothing, the candidates would still cost $2.24 per 1,000 replies and $2.36 per 1,000 replies, above the control's $2.02 per 1,000 replies had its own cost its p95; the order holds.
- **Not compared** Latency was read while other runs executed beside it and is left out. Input tokens, output tokens and model calls were not reported. _Caveats: Latency is not compared; Recorded but not reported: input tokens; Recorded but not reported: output tokens; Recorded but not reported: llm calls._
- **Apparatus** The simulated user's request settings were not recorded on every run, so whether they varied cannot be established. _Caveats: The apparatus may have differed: simulator sent settings._
- **Declarations** The asked aspect, the control, the margin and the floor were all set on 25 Oct 2026, after the results.

### Checks

- <a id="g-check-send-message-or-queue-voice-check"></a>**Send message or queue voice check**: The goal check `call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1`, which code runs on every trial.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.20, with analyst skill 365899f8f20b, on 2026-11-11._
