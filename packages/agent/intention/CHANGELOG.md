# Changelog

All notable changes to `3tears-agent-intention` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package version moves in **lockstep** with the rest of the
3tears monorepo (every package tracks the framework git tag; see
`README.md` "Versioning policy").

## [0.15.0]

### Added

- Initial release of the standing-wants corpus (Presence/aliveness program).
- `intentions` table (migration `v001`): partition on `agent_id`, composite PK
  `(agent_id, intention_id)`, CAS on `date_updated`; a fresh PG enum
  `intention_status` (`open` / `asked` / `granted` / `dropped`); pgvector `embedding`
  for log-time dedup; the `salience` / `last_decayed_at` decay substrate reused from
  `agent/memory`; a `last_surfaced_at` cooldown anchor; soft-ref `source_memory_id` /
  `source_conversation_id` provenance columns.
- `IntentionsCollection` (three-tier CRUD, CAS fence, `user_id`-required user-facing
  reads), `IntentionEntity`, `intentions_table` factory, and the `IntentionStatus`
  value set.
