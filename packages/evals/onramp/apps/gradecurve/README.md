# gradecurve

The end-of-term grade curve for STAT 101. The final exam is set and marked per section, so each section is curved on
its own: `curve_file(path)` reads a section's marks sheet (`student,score`, with a blank score for a student who
didn't sit the paper) and returns every student's curved score and letter. The rules are in the module docstring of
`curve.py`: the section's mean over the students who sat the exam is lifted to 72 if it is below that, nobody goes
over 100, letters are A 90+, B 80+, C 70+, D 60+, F below, and a student with no score gets I (incomplete).

`data/marks/` holds this autumn's eight marks sheets, sections A to H. Last term the course office ran the curve in a
spreadsheet and sent the letters to the registrar; `data/registrar.jsonl` is what the registrar holds for this term,
worked the same way, one line per student in section order. The script should give every student the letter the
registrar has.

Tests: `python -m unittest discover tests` from this directory.
