# Broadcast claims, labelled twice: a real-data fixture for claim-read calibration

License: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Attribution: Discodon / Kairo (Radio Discodon).

The lines are public broadcast lines that Kairo, Radio Discodon's AI DJ, said during production eval runs, and the
labels and answer keys are those runs' eval results, labelled by hand. A listener named in a line ("Jules") is the eval
harness's scripted persona, not a person.

| File | What it is |
|---|---|
| `claimset/claims.jsonl` | 954 factual claims Kairo made, one per line: the claim, the excerpt it was said in, the case, run and result it came from, and the first labeller's label (`supported`, `contradicted` or `unverifiable`) with the source that settled it |
| `claimset/second_labels.jsonl` | A second labeller's label on 180 of them, drawn by the first label: every claim it called contradicted (74), every one it could not settle (46), and 60 of the 834 it supported |
| `claimset/specific.jsonl` | Whether each of the 180 is a specific claim: a name, number, date, place or credit a listener could catch out |
| `keys/` | Answer-key facts derived from the labelled claims, per case (`keys.json`, `keys_barker.json`), facts about entities outside each key (`context_facts*.jsonl`), one pointer per claim to the fact it became (`pointers*.jsonl`), and the keys' own notes (`keys/README.md`). In-sample: every fact was derived from the claims it would grade |

`packages/evals/tests/test_claim_read_corpus.py` reads the first label as the instrument and the second as people,
through the claim read's weighting.
