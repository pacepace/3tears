"""core's integration tests: the process-services checks are collected only where the services are processes.

``test_process_services.py`` checks the services :mod:`threetears.core.testing.services` starts as child
processes under ``THREETEARS_TEST_SERVICES=process`` (the testament pool's integration suite). Anywhere else --
CI's integration job, a laptop's testcontainers run -- there is nothing for it to check, and collecting it would
add a skip per test to every such run. So it is left out of collection there, and the pool, which sets the
variable, collects it, where a missing service fails rather than skips (``THREETEARS_REQUIRE_SERVICES=1``).

A path named on the command line is collected whatever this says, as pytest collects every path it is given.
"""

from __future__ import annotations

import os

collect_ignore = [] if os.environ.get("THREETEARS_TEST_SERVICES") == "process" else ["test_process_services.py"]
