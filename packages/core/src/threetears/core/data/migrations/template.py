"""
blessed template for writing a new migration module.

a package's ``migrations/<nnn>_<description>.py`` file must define a
single ``async def`` taking a DataStore. the package's
``migrations/__init__.py`` imports every such callable and wires it to
a :class:`~threetears.core.data.migrations.registry.PackageMigrations`
instance. the template below is the canonical shape — copy, rename,
adjust the body.

example module::

    # my_package/migrations/001_create_widgets.py
    from threetears.core.data.store import DataStore
    from threetears.observe import get_logger

    log = get_logger(__name__)


    async def create_widgets(store: DataStore) -> None:
        '''
        create the widgets table in the current schema.

        :param store: DataStore bound to target schema via search_path
        :ptype store: DataStore
        '''
        log.info("creating widgets table")
        await store.execute(
            \"\"\"
            CREATE TABLE IF NOT EXISTS widgets (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                date_created TIMESTAMP NOT NULL,
                date_updated TIMESTAMP NOT NULL
            )
            \"\"\"
        )

and the package's ``migrations/__init__.py``::

    # my_package/migrations/__init__.py
    from threetears.core.data.migrations import MigrationScope, PackageMigrations
    from my_package.migrations.001_create_widgets import create_widgets


    def register(runner):
        '''register my_package migrations with the runner.'''
        pkg = PackageMigrations(
            name='my_package',
            scope=MigrationScope.AGENT,
        )
        pkg.version(1)(create_widgets)
        runner.register(pkg)

rules the template enforces by convention:

- every migration is idempotent (``CREATE TABLE IF NOT EXISTS``,
  ``ADD COLUMN IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS``).
- statements are unqualified — the L3 layer sets search_path so
  hard-coding a schema name breaks cross-schema reuse.
- version numbers are sequential integers starting at 1.
- the file name format ``<nnn>_<description>.py`` matches the order of
  version registration so authors can see at a glance what's pending.
"""

from __future__ import annotations


MIGRATION_FILE_TEMPLATE = '''"""
{short_description}.

{long_description}
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

log = get_logger(__name__)


async def {callable_name}(store: DataStore) -> None:
    """
    {docstring_summary}.

    :param store: DataStore bound to target schema via search_path
    :ptype store: DataStore
    """
    log.info("{log_message}")
    await store.execute("""-- replace with DDL""")
'''


def render_migration_template(
    short_description: str,
    long_description: str,
    callable_name: str,
    docstring_summary: str,
    log_message: str,
) -> str:
    """
    render the blessed migration module template as a string.

    used by documentation and by any generator that wants to emit a
    fresh migration stub. consumers substitute their own field values
    and write the result to
    ``<package>/migrations/<nnn>_<description>.py``.

    :param short_description: single-line module docstring title
    :ptype short_description: str
    :param long_description: multi-line module docstring body
    :ptype long_description: str
    :param callable_name: python identifier for the migration callable
    :ptype callable_name: str
    :param docstring_summary: one-line summary for the callable docstring
    :ptype docstring_summary: str
    :param log_message: structured log message emitted at apply time
    :ptype log_message: str
    :return: rendered module source code
    :rtype: str
    """
    result = MIGRATION_FILE_TEMPLATE.format(
        short_description=short_description,
        long_description=long_description,
        callable_name=callable_name,
        docstring_summary=docstring_summary,
        log_message=log_message,
    )
    return result
