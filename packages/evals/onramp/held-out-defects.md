# Held-out apps: the planted defects

The record of what is wrong with each held-out app, for whoever reads a run on them. Agents never see this file: a
trial gets a copy of `apps/<app>/` only, and the twin is laid over it at grading.

These apps were written blind to the package's docs, and nobody has read a transcript of them while writing a docs
change. They are registered in `held-out.jsonl`, not `apps.jsonl`, so no default run covers them. `held-out.jsonl` is
to be read only by a run whose committed plan names the apps it runs, through the harness's committed-plan selection;
until that selection exists, no harness command runs a held-out app. They stop being held out once any run's
transcripts on them have been read while writing a docs change. From then on they are development-set apps, and a
comparison on them is development-set evidence like any other.

Each share below is printed by `python packages/evals/onramp/held_out_shares.py` (standard library only), which runs
every case in the app's shipped data through the app and through the app with its twin laid over it.
`packages/evals/tests/test_onramp_held_out.py` holds every app in `held-out.jsonl`, on the pool, to three promises:
the twin agrees with every label (for the apps whose labels grade, the app never reads better than the twin), the
defect reaches more than a fifth of the cases and fewer than all, and the app's own `tests/` pass under
`python -m unittest discover tests`, run from the app's directory as a trial would, on the app and on its twin.

Graded apps, whose outputs need a graded judgement rather than an exact match: `weatherboard` (a tolerance),
`faqrank` (a ranking against graded relevance), `digest` (the share of key facts a summary carries), `shipeta` (a
tolerance), `standings` (the share of a night's teams in their published place).

## parcelrate: pricing, a wrong boundary

- **Defect.** `rates.quote` picks a parcel's band with `weight < up_to_kg`. The card says a band covers weights up to
  *and including* its limit, so a parcel at exactly 1, 2, 5 or 10 kg is charged the next band up.
- **Where.** `apps/parcelrate/rates.py`, `quote`, the band loop.
- **Share.** 11 of 40 invoice lines (28%) quote a price other than the courier's charge (`data/courier_invoice.jsonl`,
  `charged_pence`).
- **Twin diff.**
  ```diff
  -        if weight < Decimal(band["up_to_kg"]):
  +        if weight <= Decimal(band["up_to_kg"]):
  ```

## weatherboard: unit conversion, rounding of negative numbers (graded, with a tolerance)

- **Defect.** `board.nearest_degree` rounds with `int(celsius + 0.5)`, which truncates toward zero: every reading at
  or below -0.5 °C shows one degree too warm (-2.7 °C shows as -2, -0.6 °C as 0). Positive readings are right.
- **Where.** `apps/weatherboard/board.py`, `nearest_degree`.
- **Share.** 23 of 56 dual-logged readings (41%) show more than half a degree from the logger's Celsius
  (`data/dual_logged.jsonl`, `temp_c`). The README states the half-degree tolerance.
- **Twin diff.**
  ```diff
  -    return int(celsius + 0.5)
  +    return int(math.copysign(math.floor(abs(celsius) + 0.5), celsius))
  ```

## docnames: text normalisation, a dropped separator rule

- **Defect.** The docstring says apostrophes are dropped so a word keeps together (`Chef's notes` is `chefs-notes`),
  but `naming.slug` treats an apostrophe like any other separator, so `Chef's notes` becomes `chef-s-notes`. A
  trailing apostrophe (`Suppliers' price review`) comes out right either way. Accented letters are handled correctly
  in both the app and the twin: this is not the development app glossator's accent defect.
- **Where.** `apps/docnames/naming.py`, `slug`, the line that splits the title into words.
- **Share.** 13 of 42 archive documents (31%) get a stored name other than the archive's
  (`data/archive_index.jsonl`, `archive_name`).
- **Twin diff.**
  ```diff
  -    words = [word for word in _SEPARATORS.split(folded.lower()) if word]
  +    words = [word for word in _SEPARATORS.split(folded.lower().replace("'", "")) if word]
  ```

## previews: notification text, an off-by-one

- **Defect.** `previews.preview` cuts a long message to `limit` characters and then adds the one-character ellipsis,
  so every cut preview is one character over the channel's limit (unless the cut lands on a space or punctuation that
  the trailing strip removes anyway).
- **Where.** `apps/previews/previews.py`, `preview`, its last line.
- **Share.** 15 of 46 signed-off messages (33%) get a preview other than the approved one
  (`data/signed_off.jsonl`, `approved`); 19 of the 46 are long enough to be cut.
- **Twin diff.**
  ```diff
  -    return text[:limit].rstrip(_TRAILING) + ELLIPSIS
  +    return text[: limit - len(ELLIPSIS)].rstrip(_TRAILING) + ELLIPSIS
  ```

## tickets: a state machine, a dropped case

- **Defect.** The transition table has no entry for a customer replying on a `solved` ticket, so the ticket stays
  `solved` and its `reopens` count never rises, where the documented rule reopens it.
- **Where.** `apps/tickets/desk.py`, `TRANSITIONS`.
- **Share.** 20 of 44 exported tickets (45%) end off the old desk's status or reopen count (`data/export.jsonl`,
  `old_desk`); 15 of them (34%) are off on the status alone.
- **Twin diff.**
  ```diff
       ("pending", "customer_reply"): "open",
  +    ("solved", "customer_reply"): "open",
  ```

## faqrank: ranking, query and index tokenised differently (graded)

- **Defect.** `Index.search` splits the query on whitespace only, while articles are split into runs of letters and
  digits. A query word with punctuation on it (`refund?`, `double-charged`, `permissions/roles`) matches nothing, so
  that word stops counting and the ranking falls back on the query's other words, or returns nothing.
- **Where.** `apps/faqrank/search.py`, `Index.search`, its first line.
- **Share.** 13 of 44 logged queries (30%) rank with a lower NDCG@5 than the twin against the support team's graded
  judgments (`data/judgments.jsonl`); mean NDCG@5 is 0.726 on the app and 0.958 on the twin, and 11 queries lose the
  best article from first place. No query reads better on the app.
- **Data edit.** One query was reworded after the first draft of the data: `verification code never arrives, what
  now?` became `verification code never arrives`. In the first form the shipped app read slightly better than the
  twin (NDCG@5 1.000 against 0.964): the twin read `arrives,` as `arrive` and matched it in the password-reset
  article, which then ranked second, while the app's `arrives,` matched nothing. So the twin did not read at least as
  well as the app on every query, which the pool test requires of a graded app. The reworded query, with no
  punctuation, reads the same on both. The defect's reach was 13 of 44 before the edit and is 13 of 44 after it.
- **Twin diff.**
  ```diff
  -        query_terms = [stem(word) for word in query.lower().split() if word not in STOPWORDS]
  +        query_terms = terms(query)
  ```

## digest: a rule-based summariser, a sentence pattern that is too loose (graded)

- **Defect.** `digest.sentences` ends a sentence at every `.`, `!` or `?` followed by a space, so abbreviations
  (`approx.`, `e.g.`, `i.e.`, `vs.`, `Dec.`) split a sentence in two. The half without the cue words loses its score,
  and the fact it carried drops out of the digest; a fragment can also take one of the three places.
- **Where.** `apps/digest/digest.py`, `_SENTENCE_END`.
- **Share.** 10 of 24 reports (42%) get a digest carrying fewer of the head of operations' key facts than the twin's
  (`data/key_facts.jsonl`); the mean share of facts carried is 0.569 on the app and 0.722 on the twin, and no report
  reads better on the app. The twin does not carry every fact either: three sentences cannot always hold them.
- **Twin diff.**
  ```diff
  -_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
  +_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
  ```

## statements: parsing, a wrong default

- **Defect.** `DAY_FIRST`, the default for reading a slashed date, is `False`, so `04/03/2026` is read as 3 April
  where a UK statement means 4 March. A date whose day is above 12 cannot be read month first and falls back to the
  right reading, and ISO dates are unaffected, so only early-month dates in slashed statements go wrong, silently.
- **Where.** `apps/statements/importer.py`, `DAY_FIRST` (the default of `parse_date`, `read_statement` and
  `read_statement_file`).
- **Share.** 20 of 74 ledger transactions (27%) import with a date other than the hand-kept ledger's
  (`data/ledger.jsonl`); 4 of the 8 statements have at least one. The ledger is in date order across the accounts, as
  a hand-kept ledger is, so the affected rows (each month's early days) recur through it: 6 of its first 20 rows.
- **Twin diff.**
  ```diff
  -DAY_FIRST = False
  +DAY_FIRST = True
  ```

## contactmerge: de-duplication, state kept across calls under the wrong key

- **Defect.** `contacts.keys` remembers each row's match keys in a module-level cache, so a merge asks for them once,
  but it files them under the row's name. The second row with a name already seen, in the same call or any later one,
  gets the first row's email and phone. Two different people who share a name (the README: a shared name alone never
  makes two rows one person) are then joined, and a row named only by a first name can take an earlier namesake's
  keys. Rows with names not seen before are right.
- **Where.** `apps/contactmerge/contacts.py`, `keys`, its first line.
- **Share.** 12 of 44 reviewed pairs (27%) are decided otherwise than Priya's review (`data/reviewed_pairs.jsonl`,
  `same`): the 11 namesake pairs, and one pair whose first-name-only row met an earlier row of that name; 8 of its
  first 20 and 2 of its first 5. The share is measured on a fresh import, pairs in file order.
- **Twin diff.**
  ```diff
  -    cache_key = contact.name
  +    cache_key = contact
  ```

## slotfinder: meeting slots, float accumulation

- **Defect.** `slots.free_slots` steps through the UTC day by adding `step / 60` hours to a float and truncates each
  start to whole minutes. With a step of 30 or 15 minutes the sum is exact; with 20 or 10 minutes (a third or a sixth
  of an hour) it drifts just under the true time, so many starts come out a minute early (`01:39` for `01:40`), and
  the free check is made at that minute. Requests on the half hour or the quarter hour are right. Daylight saving and
  the time zones are handled correctly in both the app and the twin.
- **Where.** `apps/slotfinder/slots.py`, `free_slots`, the line that makes each start.
- **Share.** 13 of 46 logged requests (28%) get free slots other than the ones offered (`data/requests.jsonl`,
  `offered`); 5 of its first 20 and 1 of its first 5. The requests are a plain random draw (each team's step drawn
  from 30, 30, 15, 20, 10, 10 minutes), with no selection on the outcome. 23 of them use a 20 or 10-minute step; the
  other 10 of those come out the same on both, since none of their free windows meets a drifted start.
- **Twin diff.**
  ```diff
  -        start = midnight + timedelta(minutes=int(hour * 60))
  +        start = midnight + timedelta(minutes=round(hour * 60))
  ```

## shipeta: an estimate, the wrong unit (graded, with a tolerance)

- **Defect.** `eta.transit_hours` adds a lane's `customs_days` to a count of hours, so a lane abroad waits one hour
  at customs where the lane table means one day (two for `BHM` to `AMS`). Domestic lanes have no customs wait and are
  right.
- **Where.** `apps/shipeta/eta.py`, `transit_hours`, the customs line.
- **Share.** 15 of 52 delivered shipments (29%) are delivered more than 3 hours from the app's estimate
  (`data/delivered.jsonl`, `delivered_at`); 5 of its first 20. Every shipment is within 3 hours of the twin's
  estimate; the mean miss is 9.4 hours on the app and 0.9 on the twin. The README states the 3-hour tolerance.
- **Twin diff.**
  ```diff
  -    hours += lane.customs_days
  +    hours += lane.customs_days * 24
  ```

## standings: ranking with ties, a wrong tie-break (graded)

- **Defect.** `standings.table` sorts on `(points, handed_in)` with `reverse=True`, which puts more points first but
  also puts the later answer sheet first among teams level on points. The rule is that the earlier sheet ranks higher.
  Teams level on both still share a place, so only nights with a points tie broken by the sheet's time go wrong.
- **Where.** `apps/standings/standings.py`, `table`, its first line.
- **Share.** 9 of 32 quiz nights (28%) have fewer teams in Dee's published place than the twin
  (`data/nights.jsonl`, `published`); 7 of its first 20. The mean share of a night's teams in their published place
  is 0.946 on the app and 1.000 on the twin, and no night reads better on the app.
- **Twin diff.**
  ```diff
  -    ordered = sorted(entries, key=lambda e: (e.points, _seconds(e.handed_in)), reverse=True)
  +    ordered = sorted(entries, key=lambda e: (-e.points, _seconds(e.handed_in)))
  ```

## mdtable: text width, unicode normalisation

- **Defect.** `mdtable.display_width` counts code points, so a name that arrives decomposed (`e` and a combining
  U+0301 for `é`, as text copied from some apps does) counts one character more per accent than a reader sees, and
  its column is padded short or too wide. The docstring says an accented letter counts once either way. Text that
  arrives precomposed is right.
- **Where.** `apps/mdtable/mdtable.py`, `display_width`.
- **Share.** 12 of 42 handbook tables (29%) are written otherwise than the signed-off Markdown
  (`data/tables.jsonl`, `expected`); 5 of its first 20. In the data file every combining mark (and every zero-width
  format character) is written as a JSON escape (`Montre\u0301al`), so it stays visible to whoever copies a case by
  hand; `json.loads` gives the same strings as before.
- **Twin diff.**
  ```diff
  -    return len(text)
  +    return len(unicodedata.normalize("NFC", text))
  ```

## gradecurve: a curve, blank scores counted as zero

- **Defect.** `curve.section_mean` counts a blank score (a student who did not sit the exam) as 0 and divides by the
  whole section, where the rule leaves such students out of the mean. A section with a blank gets a lower mean, a
  bigger lift, and letters a band too high for many of its students. The absent student still gets `I`.
- **Where.** `apps/gradecurve/curve.py`, `section_mean`.
- **Share.** 40 of 114 students (35%) get a letter other than the registrar's (`data/registrar.jsonl`, `letter`); 7
  of its first 20. Five of the eight sections have a blank; B, D and F have none and are right.
- **Twin diff.**
  ```diff
  -    return sum(s or 0.0 for s in scores) / len(scores)
  +    return sum(s for s in scores if s is not None) / sum(s is not None for s in scores)
  ```

## plurals: message templates, a formatting step skipped on one path

- **Defect.** A plain field writes its number through `messages.number` (thousands commas: `1,204`), but the `#` in a
  plural form is replaced with `str(value)`, so a plural field over 999 reads `1204 files` where the docstring says
  `1,204 files`. Numbers below 1,000, and every plain field, are right.
- **Where.** `apps/plurals/messages.py`, `render`, the last line of `field`.
- **Share.** 11 of 46 review-sheet strings (24%) render otherwise than the approved text (`data/strings.jsonl`,
  `approved`); 6 of its first 20 and 3 of its first 5.
- **Twin diff.**
  ```diff
  -        return choose(values[name], forms).replace("#", str(values[name]))
  +        return choose(values[name], forms).replace("#", number(values[name]))
  ```

## proration: billing, integer division

- **Defect.** `proration.share` floors with `//`, where each invoice line is to be rounded to the nearest penny, half
  a penny up. A line whose exact amount has a fraction of a half penny or more comes out one penny short.
- **Where.** `apps/proration/proration.py`, `share`.
- **Share.** 15 of 46 plan changes (33%) get invoice lines other than the provider's (`data/invoices.jsonl`,
  `credit_pence` and `charge_pence`); 7 of its first 20 and 3 of its first 5. The file lists the invoices newest
  first, as the provider's export does, and the README calls it a sample. The invoices were chosen from a larger
  draw to hold the share near a third: on uniformly drawn changes nearly three in four have a line with such a
  fraction.
- **Twin diff.**
  ```diff
  -    return pence * days // of_days
  +    return (2 * pence * days + of_days) // (2 * of_days)
  ```
