"""Serve the booking eval's runs to the team, from this directory: ``python serve.py`` is an MCP server on stdio
(``[fastmcp]`` extra), and ``python -m threetears.evals ls --host serve:host --scope booking`` is the command line
(``report`` too; ``gate`` needs a comparison, and this eval has one arm). Both read the store
``python -m threetears.evals agent.py`` fills when run here.
"""

from agent import booking
from threetears.evals.quick import as_host


def host():
    """The store every run of agent.py fills; hand as_host any DocumentStore instead (Postgres) to share it."""
    return as_host([booking], store=".evals/evals.sqlite")


if __name__ == "__main__":  # MCP, read-only (list, read, gate); REST waits on a later phase
    from threetears.evals.transports.fastmcp import mount_fastmcp

    mount_fastmcp(host(), scope="booking").run()
