"""Serve the booking eval's runs to the team: ``python serve.py`` is an MCP server on stdio (``[fastmcp]`` extra), and
``python -m threetears.evals ls --host serve:host --scope booking`` is the command line (``report``, ``gate`` too).
"""

from agent import booking
from threetears.evals.actions import Caller, eval_catalogue, read_only_tools
from threetears.evals.kernel import default_job_timeout
from threetears.evals.ops import OpsHost
from threetears.evals.quick import as_host
from threetears.evals.run import LaunchHost, LaunchSettings


def host():
    """The store every run of agent.py fills; hand as_host any DocumentStore instead (Postgres) to share it."""
    return as_host([booking], store=".evals/evals.sqlite")


if __name__ == "__main__":  # REST waits on a later phase; this serves MCP (read-only: list, read, gate) and the CLI
    from fastmcp import FastMCP

    from threetears.evals.transports.fastmcp import mount_fastmcp

    limits = LaunchSettings(
        max_launch_arms=1,
        max_admitted_runs=1,
        judge_concurrency=1,
        enforcement_enabled=False,
        max_cost_usd=1,
        max_metered_calls=None,
        max_out_of_run_cost_usd=1,
    )
    launch = LaunchHost(
        eval_host=host(),
        kinds={},
        settings=lambda: limits,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda run: {},
    )  # nothing is launched over MCP here
    server = FastMCP("booking evals")
    mount_fastmcp(
        server,
        eval_catalogue(),
        host=OpsHost(launch=launch),
        caller=lambda: Caller(scope_id="booking", identity="team"),
        tools=read_only_tools(),
    )
    server.run()
