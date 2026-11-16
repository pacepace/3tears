"""The on-ramp's agent test: can an agent with no prior knowledge add an eval to an unfamiliar app?

The test is itself an eval on the package (:mod:`.campaign`). Its cases are three small apps written for it and
named nowhere in the docs (``apps/``), each with a defect planted on a known slice of its inputs and a fixed twin
(``twins/``). Its arms are docs builds: an installed wheel and the ``llms.txt`` inside it (:class:`.sandbox.DocsBuild`).
Its subject is an agent handed only that install, the app's directory and one sentence (:mod:`.agent`). Success is
checked by code, by re-running what the agent wrote against the app, its fixed twin and a do-nothing runner
(:mod:`.checks`). Everything a run did is its stored output: time, turns, tokens, cost, every command and error, every
page read, and where it got stuck.

``scripts/onramp-agent.sh`` is the one command. Nothing here ships in the wheel or imports as ``threetears.evals``;
it reaches the package through its public roots, as any adopter does.
"""
