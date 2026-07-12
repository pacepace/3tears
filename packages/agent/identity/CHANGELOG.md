# Changelog

All notable changes to `3tears-agent-identity` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package version moves in **lockstep** with the rest of the
3tears monorepo (every package tracks the framework git tag).

## [0.15.0]

### Added

- Initial release of the versioned identity-block store (self-evolution).
- `identity_versions` table (migration `v001`): partition on `agent_id`, composite PK
  `(agent_id, version_id)`, CAS on `date_updated`; two fresh PG enums
  `identity_block_key` (`personality` / `reinforcement` / `anti_sycophant` /
  `self_improvement` / `presence`) and `identity_version_status` (`proposed` /
  `active` / `superseded` / `rejected`); a linear parent-pointer version chain
  (`parent_version_id`); immutable snapshot columns (`content` / `rationale` /
  `content_hash` / `block_key` / `proposer_agent_id`) + the mutable lifecycle
  (`status` / `consenter_user_id`); a partial UNIQUE index enforcing exactly one
  `active` version per `(agent, customer, user, block)`, a block-history btree, and a
  partial pending-queue btree.
- `IdentityVersionsCollection` (three-tier CRUD) + `identity_versions_table` factory +
  the read paths `resolve_active` / `find_versions_for_block` / `find_pending`.
- The block-key / status / consent-tier value types (`IDENTITY_BLOCK_TIERS` encodes
  the tier-from-day-one consent model).
