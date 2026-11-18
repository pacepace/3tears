# spamscore

A spam score for the shared support inbox. `score(message, contacts)` gives a message a score from 0 (plainly fine)
to 10 (plainly spam), and `to_review` sends anything scoring 5 or more to the review folder instead of the agents'
queue. `load_contacts("data/contacts.txt")` reads our contacts; `score_file(path, contacts)` scores a file of
messages, one JSON message per line (`id`, `sender`, `subject`, `body`, and optionally `reply_to` and
`attachments`).

The signals and their weights are in the module docstring of `score.py`. The one that matters most is a link from a
stranger (+3): a message whose body carries a link, when the sender is not in our contacts, or when its reply-to
address is on a different domain from the sender's. A message with no link never gets those 3 points, whoever sent
it and wherever replies go: plenty of real customers write from one address and ask for replies at another.

The score is a judgement call, so it is checked against people. `data/rated.jsonl` is a sample of last month's inbox
that three of the support moderators each rated from 0 to 10 for how spammy it is; `panel` is the mean of their
three ratings. The score is good enough when it lands within 1.5 of the panel's mean for every message; a message
outside that is one the score gets wrong.

Tests: `python -m unittest discover tests` from this directory.
