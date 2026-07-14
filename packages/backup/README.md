# 3tears-backup

Encrypted, grandfather-father-son (GFS) rotated **database backups** to any
`ObjectStore`, with **restore verification** — built on 3tears primitives so it
drops into any 3tears app.

- **Storage-agnostic.** The engine takes an injected `ObjectStore`
  (`3tears-object-store` — S3 is the first driver, filesystem the second) and
  streams the dump through `EncryptedObjectStore`, so a multi-GB dump is
  client-side AES-256-GCM encrypted and never sits whole in memory.
- **Postgres *and* Yugabyte.** A pluggable `DbDumpDriver` wraps the right dump
  tool; the engine autodetects the target from `version()`.
- **GFS retention.** Keep N daily / weekly / monthly backups; prune the rest.
- **Restore-verified.** Backups are proven by restoring into a throwaway
  temporary database, with an optional hook to spin a stack against it.

Configuration is an injected, frozen `BackupConfig` — construct it however you
like (a `from_env` factory reads `THREETEARS_BACKUP_*` with defaults; most apps
will build it from control-plane settings instead).
