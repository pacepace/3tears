"""Encrypted, GFS-rotated database backups to any ObjectStore, with restore verification.

Beyond the single-database :class:`BackupEngine`: :class:`ClusterBackup` takes whole-cluster
backup SETS (every database enumerated at backup time, plus roles/grants globals) identified by a
durable :class:`BackupManifest`; :class:`SelectiveRestore` brings back individual rows, uuid7 id
ranges, or date ranges from a restored scratch database with a plan/apply dry-run split; and
:class:`DriftComparator` proves a restored backup mostly matches the live database, cutting both
sides off at the backup moment so post-backup growth is expected rather than failure.
"""

from threetears.backup.cluster import ClusterBackup, ManifestNotFoundError, replace_database
from threetears.backup.compare import (
    ComparisonReport,
    DriftComparator,
    TableComparison,
    uuid7_upper_bound,
)
from threetears.backup.config import BackupConfig
from threetears.backup.drivers import (
    DbDumpDriver,
    PostgresDriver,
    YugabyteDriver,
    detect_driver,
    driver_by_name,
    driver_for_version,
)
from threetears.backup.engine import BackupEngine, DeleteNotAllowedError
from threetears.backup.manifest import BackupManifest, DatabaseDump, TableCount, manifest_key
from threetears.backup.process import BackupToolError
from threetears.backup.retention import (
    BackupRecord,
    GfsRetention,
    RetentionDecision,
)
from threetears.backup.selective import (
    PlannedRow,
    RowSelection,
    SelectionTooLargeError,
    SelectiveRestore,
    SelectiveRestorePlan,
)
from threetears.backup.verify import (
    RestoreVerifier,
    VerificationResult,
    count_tables,
    make_subprocess_hook,
    make_temp_db_provisioner,
)

__all__ = [
    "BackupConfig",
    "BackupEngine",
    "BackupManifest",
    "BackupRecord",
    "BackupToolError",
    "ClusterBackup",
    "ComparisonReport",
    "DatabaseDump",
    "DbDumpDriver",
    "DeleteNotAllowedError",
    "DriftComparator",
    "GfsRetention",
    "ManifestNotFoundError",
    "PlannedRow",
    "PostgresDriver",
    "RestoreVerifier",
    "RetentionDecision",
    "RowSelection",
    "SelectionTooLargeError",
    "SelectiveRestore",
    "SelectiveRestorePlan",
    "TableComparison",
    "TableCount",
    "VerificationResult",
    "YugabyteDriver",
    "count_tables",
    "detect_driver",
    "driver_by_name",
    "driver_for_version",
    "make_subprocess_hook",
    "make_temp_db_provisioner",
    "manifest_key",
    "replace_database",
    "uuid7_upper_bound",
]
