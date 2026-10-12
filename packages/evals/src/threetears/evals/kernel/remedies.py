"""What a refused caller can do next — the remedy every refusal of a record that is not there names.

A refusal that only says what went wrong leaves its reader to guess the next step, and an agent guesses by retrying
the same call. Each constant here is one remedy, named by what it leads to, so every refusal of the same kind of
record ends the same way and the remedies can be checked in one place: ``tests/test_refusal_remedies.py`` holds that
every :class:`~threetears.evals.kernel.errors.NotFoundError` and
:class:`~threetears.evals.kernel.errors.ConflictError` the engine raises names one, and that every action a remedy
names is an action the engine's catalogue offers — so no remedy points at a call nobody can make.

A leaf module, like :mod:`threetears.evals.kernel.errors`: every layer that refuses reads it, and it reads nothing.
Each remedy names the operation by the name its action and its :mod:`threetears.evals.ops` function share, so a
Python caller and an agent read the same step.
"""

from __future__ import annotations

#: How a caller finds a template's id.
FIND_TEMPLATE = "templates_list names this scope's templates"

#: How a caller finds a run's id.
FIND_RUN = "runs_list names this scope's runs"

#: What a caller does about a stored run this build's model refuses: see each one, and why, so it can be dropped or a
#: build that reads it deployed.
FIND_UNREADABLE_RUNS = "runs_unreadable names each stored run of this scope this build cannot read, with why"

#: How a caller finds a campaign's id.
FIND_CAMPAIGN = "campaigns_list names this scope's campaigns"

#: How a caller finds an analysis's id.
FIND_ANALYSIS = "analyses_list names a campaign's analyses"

#: How a caller finds an insight's id.
FIND_INSIGHT = "insights_list names this scope's insights"

#: How a caller finds a result's id.
FIND_RESULT = "results_list names a run's results"

#: How a caller finds a stored case's id.
FIND_TEST_CASE = "results_list names the cases a run ran, by their test case ids"

#: How a caller finds a case set's name and version.
FIND_CASE_SET = "case_sets_list names this scope's case sets and their versions"

#: How a caller finds a finding of an analysis.
FIND_FINDING = "report_read states the campaign's analysis with its findings numbered from 0"

#: How a caller finds a job's id.
FIND_JOB = (
    "a job's id is the one the call that started it returned (run:<id>, sweep:<id>, launch:<id>, ...); "
    "runs_list names this scope's runs, each of which job_poll reads as run:<id>"
)

#: How a caller finds a judge config's id. No action authors them — they are the host's catalogue — but one lists them.
FIND_JUDGE_CONFIG = "judge_configs_list names this scope's judge configs, each with the dimension it asks for"

#: How a host finds a rubric dim's id. No action authors or lists them: they are the host's catalogue.
FIND_RUBRIC_DIM = "the host's catalogue lists this scope's rubric dims (run.list_rubric_dims)"

#: What a template's author does when its world seed states world state a goal check's control cannot be built over.
SEED_REMEDY = "seed only dimensions this host's world declares as seedable, at values its registry accepts"

#: What a caller does when a run's judging cannot be reproduced from what it recorded — evidence, attribution, a
#: config set, request settings, a temperature or a template the run no longer matches. The run's own judging cannot
#: be sent again, so the reachable step is a run whose judging is recorded in full.
RELAUNCH_TO_REJUDGE = (
    "this run's judging cannot be reproduced as it was made; launch its template again with run_launch for a judged "
    "run that records it in full, and re-judge or repeat that one"
)

#: What a caller does when a document it sent does not validate against its schema: the validator's own words name
#: the field and the rule, so the remedy is to correct what they name.
SCHEMA_REMEDY = "correct the field the validation error names, and send it again"

#: What a caller does when a writer model the host does not allow was asked for an analysis.
WRITER_REMEDY = "name as model a writer this host allows, or ask the host's operator to allow this one"

#: What a caller does when a record changed under its write.
REREAD_AND_RETRY = "re-read it and make the change again"

__all__ = [
    "FIND_ANALYSIS",
    "FIND_CAMPAIGN",
    "FIND_CASE_SET",
    "FIND_FINDING",
    "FIND_INSIGHT",
    "FIND_JOB",
    "FIND_JUDGE_CONFIG",
    "FIND_RESULT",
    "FIND_RUBRIC_DIM",
    "FIND_RUN",
    "FIND_TEMPLATE",
    "FIND_TEST_CASE",
    "FIND_UNREADABLE_RUNS",
    "RELAUNCH_TO_REJUDGE",
    "REREAD_AND_RETRY",
    "SCHEMA_REMEDY",
    "SEED_REMEDY",
    "WRITER_REMEDY",
]
