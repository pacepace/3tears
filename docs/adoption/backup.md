# 3tears-backup

`threetears.backup` -- encrypted, GFS-rotated database backups to any
`ObjectStore`, with restore verification.

## Problem

Database backups that are never test-restored are a hope, not a guarantee.
A backup script that dumps to disk unencrypted, or holds a multi-GB dump
whole in memory, doesn't survive contact with a real production database or
a security review.

## What it does

- Encrypted, grandfather-father-son (GFS) rotated backups -- keep N daily /
  weekly / monthly, prune the rest.
- Storage-agnostic: takes an injected `ObjectStore` -- the protocol comes
  from `3tears-media-contracts`; `3tears-object-store` is the typical
  implementation -- and streams the dump through `EncryptedObjectStore`
  (also from `object-store`), so a multi-GB dump is client-side AES-256-GCM
  encrypted and never sits whole in memory.
- Encryption key handling: the host supplies a passphrase; the actual AES
  key is derived per object via scrypt, not used directly. There is no
  built-in key-rotation or secret-manager integration -- the host owns
  passphrase storage and rotation.
- Postgres and Yugabyte support via a pluggable `DbDumpDriver`, autodetected
  from `version()`.
- Restore verification: backups are proven by actually restoring into a
  throwaway temporary database, with an optional hook to spin a stack
  against it.

## Design philosophy

Built on 3tears primitives so it drops into any 3tears app rather than
being a standalone tool with its own infrastructure assumptions. "Restore
verified" is treated as a first-class requirement, not an afterthought --
a backup that has never been restored is not considered a verified backup.

## When to adopt

Any 3tears app running its own PostgreSQL or Yugabyte database that needs
backups it can actually trust, not just backups that exist.

## Composes with

- [`media-contracts`](media-contracts.md) -- the `ObjectStore` protocol
  this package is built against.
- [`object-store`](object-store.md) -- the typical `ObjectStore` and
  `EncryptedObjectStore` implementation backups are written through.
- [`observe`](observe.md) -- logging throughout the backup/restore path.

## Install

```bash
pip install 3tears-backup
```
