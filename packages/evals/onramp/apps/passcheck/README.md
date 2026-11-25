# passcheck

The strength meter beside the password box on our sign-up form (`strength.py`). `rate(password, username)` gives a
`Rating`: a score from 0 to 100, the band the form shows (`weak`, `fair`, `good` or `strong`), and the reasons points
were taken off. The form refuses a weak password. `band_of(score)` gives the band for a score.

How a password is scored is in the module docstring: points for length and for mixing kinds of character, and points
off for a common word (from `data/common.txt`, matched whatever its capitals), for the username, for runs like `abc`
or `123`, and for repeated characters.

Before the meter went live the security team took a sample of passwords of the kind people really pick, and rated
each one by hand against the same rules; `data/reviewed.jsonl` is that sample, with the band the team gave each
(`team_band`). The meter should show the team's band. The bands are ordered: showing `fair` where the team said `good`
is a near miss, but showing a password one or two bands stronger than the team rated it is the error that matters,
since it tells someone a weak password is fine, and a weak one shown as fair or better gets past the form.

Tests: `python -m unittest discover tests` from this directory.
