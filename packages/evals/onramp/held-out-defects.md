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
the twin agrees with every label (for the two apps whose labels grade, the app never reads better than the twin), the
defect reaches more than a fifth of the cases and fewer than all, and the app's own `tests/` pass under
`python -m unittest discover tests`, run from the app's directory as a trial would, on the app and on its twin.

Graded apps, whose outputs need a graded judgement rather than an exact match: `weatherboard` (a tolerance),
`faqrank` (a ranking against graded relevance), `digest` (the share of key facts a summary carries), `passcheck`
(a score read as ordered bands, where showing a password stronger than it is is the error that matters), `moodscore`
(a score within a tolerance of a panel's mean rating).

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

## reorder: stock control, integer division where a fraction was meant

- **Defect.** `stockroom.daily_demand` divides units sold by the window's days with `//`, so daily demand loses its
  fraction (30 sold in 28 days is 1 a day, 27 is 0). Every reorder point and order built on it comes out low; a line
  whose sales divide exactly by its window, or that sold nothing, is unaffected.
- **Where.** `apps/reorder/stockroom.py`, `daily_demand`, its last line.
- **Share.** 34 of 44 stock lines (77%) get a reorder point or an order other than the buyer's spreadsheet's
  (`data/stock_review.jsonl`, `sheet_reorder_point` and `sheet_order`); 34 are off on the reorder point and 17 on the
  order.
- **Twin diff.**
  ```diff
  -    return sold // days
  +    return sold / days
  ```

## standups: calendar recurrence, the clock change

- **Defect.** `recur._stepped` steps a series forward in UTC and converts each meeting back to London time, so a
  series that crosses a clock change keeps its UTC time and moves an hour on the office clock (a 09:30 stand-up
  becomes 10:30 after the clocks go forward). The module docstring and the README say a meeting keeps its wall-clock
  time. A series that stays inside one season is right.
- **Where.** `apps/standups/recur.py`, `_stepped`, the `yield` line.
- **Share.** 20 of 32 recurring series (62%) send invites other than the old calendar's (`data/series.jsonl`,
  `old_calendar`); 170 of the 492 meetings are an hour out.
- **Twin diff.**
  ```diff
  -        yield (first.astimezone(timezone.utc) + step * n).astimezone(LONDON)
  +        yield first + step * n
  ```

## postal: address normalisation, a greedy pattern

- **Defect.** `address._FLAT` reads a flat's number with a greedy `(.+),`, so it runs to the last comma before the
  postcode rather than the first: `Flat 2, 14 Mill Ln, Leeds` gives the line `FLAT 2, 14 MILL LN`, with the street
  folded into the flat line and its street type not spelt out. Addresses with no flat or unit are right.
- **Where.** `apps/postal/address.py`, `_FLAT`.
- **Share.** 19 of 44 order addresses (43%) get label lines other than the courier's (`data/order_addresses.jsonl`,
  `courier_label`): every address with a flat or unit, since each has a street and town after it.
- **Twin diff.**
  ```diff
  -_FLAT = re.compile(r"^\s*(flat|apartment|apt|unit)\.?\s+(.+),\s*", re.IGNORECASE)
  +_FLAT = re.compile(r"^\s*(flat|apartment|apt|unit)\.?\s+(.+?),\s*", re.IGNORECASE)
  ```

## passcheck: password strength, case-sensitive matching (graded)

- **Defect.** `strength.rate` looks for the common words in the password as typed, while the list is in lower case,
  so a common word with a capital in it (`Summer2026!`, `LONDON2012games`) escapes the 25-point penalty and the
  password shows a band stronger. The docstring and the README say the match ignores case.
- **Where.** `apps/passcheck/strength.py`, `rate`, the line that sets `found`.
- **Share.** 20 of 44 sample passwords (45%) show a band other than the security team's (`data/reviewed.jsonl`,
  `team_band`), all of them stronger than the team rated; 2 that the team rated weak get past the form.
- **Twin diff.**
  ```diff
  -    found = sorted(word for word in _COMMON_WORDS if word in password)
  +    found = sorted(word for word in _COMMON_WORDS if word in password.lower())
  ```

## moodscore: rule-based sentiment, a negation window not closed (graded, with a tolerance)

- **Defect.** After a negator turns round a sentiment word, `sentiment.raw_score` counts the window down by one
  instead of closing it, so a second sentiment word close behind is turned round too: `no complaints great food`
  reads `great` as negative. The docstring says a negator turns round one word only. A review whose negator has no
  second sentiment word in reach, or a clause break before it, is unaffected.
- **Where.** `apps/moodscore/sentiment.py`, `raw_score`, the line after `weight = -NEGATED * weight`.
- **Share.** 15 of 49 rated reviews (31%) score more than 0.25 from the panel's mean rating
  (`data/rated_reviews.jsonl`, `panel`), the tolerance the README states; the mean distance from the panel is 0.261 on
  the app and 0.060 on the twin.
- **Twin diff.**
  ```diff
  -            negation -= 1
  +            negation = 0
  ```

## logtally: log aggregation, an empty size field mishandled

- **Defect.** The log-line pattern takes the size only as digits, so a line logged with `-` for its size (a `304`, a
  `204`, a `HEAD`) does not match: it is counted as unparsed and its hit is lost. The docstring says such a line is a
  hit of 0 bytes. Pages that had no bodiless answer that day are right.
- **Where.** `apps/logtally/tally.py`, `_LINE`, the status-and-size line.
- **Share.** 18 of 55 day-and-page figures (33%) have hits other than the hosting dashboard's (`data/dashboard.jsonl`);
  the bytes stay right, since a bodiless answer sends none.
- **Twin diff.**
  ```diff
  -    r"(?P<status>\d{3}) (?P<size>\d+) "
  +    r"(?P<status>\d{3}) (?P<size>\d+|-) "
  ```

## shelfprice: unit prices, a decimal comma read as a thousands separator

- **Defect.** `unitprice._number` drops a comma from a pack size, so a continental supplier's `0,75 l` is read as
  75 litres and `1,5 kg` as 15 kg, and the unit price comes out far too low. The docstring and the README say a comma
  is a decimal mark. Sizes written with a point are right.
- **Where.** `apps/shelfprice/unitprice.py`, `_number`.
- **Share.** 11 of 40 audited lines (28%) get a unit-price label other than the one the shop manager signed off
  (`data/shelf_audit.jsonl`, `audited_label`): every size written with a comma.
- **Twin diff.**
  ```diff
  -    return Decimal(text.replace(",", ""))
  +    return Decimal(text.replace(",", "."))
  ```

## loyalty: points accrual, an inverted comparison on one branch

- **Defect.** The bronze branch of `points._tier_bonus` gives the 25 bonus points when the eligible spend is at most
  £60, where the rule is at least £60: bronze baskets under £60 get the bonus and those over it do not. The gold and
  silver branches are right, and so is a bronze basket of exactly £60.
- **Where.** `apps/loyalty/points.py`, `_tier_bonus`, the bronze branch.
- **Share.** 15 of 42 October baskets (36%) are awarded points other than the old till's (`data/till_export.jsonl`,
  `points_awarded`): every bronze basket.
- **Twin diff.**
  ```diff
  -        return BRONZE_BONUS if spend <= BRONZE_BONUS_FROM else 0
  +        return BRONZE_BONUS if spend >= BRONZE_BONUS_FROM else 0
  ```
