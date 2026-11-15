# Database migration path and recovery

## Current revision graph

The migration chain is linear:

```text
0c4a1b2d3e5f (base tables)
  -> e9fa8685c983 (reddit_posts.source_id, nullable + FK/index)
  -> 966e480ef159 (NLP tables; source_id becomes NOT NULL)
  -> 17cb1444081b (detected_languages)
  -> 8f1300a2a8f0 (preprocessing columns; head)
```

`0c4a1b2d3e5f` creates `data_sources`, `reddit_posts`, and `reddit_comments` in the schema that precedes the already-existing `e9fa8685c983` migration. The old revision ID and its upgrade/downgrade operations are preserved. Its graph parent now points at the new baseline so a fresh install runs the base-table DDL first.

Databases already stamped at `e9fa8685c983`, `966e480ef159`, `17cb1444081b`, or `8f1300a2a8f0` upgrade from that recorded revision; Alembic does not replay the baseline. The new baseline itself is also a valid revision state. There is no earlier supported revision in the repository history.

On online execution, the baseline checks for any of its three base tables before creating anything. If one or more exist while Alembic is trying to apply the baseline, it stops before making schema changes. It does not stamp a revision or infer that existing tables are compatible. Offline SQL rendering cannot inspect a live database; generated SQL is not a substitute for checking the target schema.

## Fresh PostgreSQL installation

Use a newly provisioned, empty PostgreSQL database and the project's normal protected configuration. Review the target database identity before running migrations, then run:

```text
python -m alembic upgrade head
```

Run this from `backend/`. Do not use an application startup `create_all()` call as a migration substitute. This project change was checked by rendering PostgreSQL-dialect offline SQL; an actual PostgreSQL upgrade has not yet been verified.

## Existing database with a valid Alembic revision

1. Take and verify a database backup using the approved operations process.
2. Confirm `alembic current` reports exactly one known revision from the supported list above, and confirm the live schema matches that revision before proceeding.
3. Review the migration SQL and target identity, then run `python -m alembic upgrade head` from `backend/`.
4. Verify the resulting revision and schema using the team's normal database change procedure.

Do not run this against an unknown database or assume that the revision marker proves the schema has not drifted.

## Existing tables but no valid Alembic revision

Stop before attempting `upgrade head`. Do not stamp `head`, delete `alembic_version`, drop tables, or retry with a different migration command. The baseline deliberately refuses to adopt any of its existing base tables without history.

Recovery requires a database owner/operator to:

1. Take and verify a backup; record the exact database identity and whether `alembic_version` exists and what rows it contains.
2. Capture a schema-only snapshot through the approved credential-safe procedure (for PostgreSQL, for example `pg_dump --schema-only --no-owner`), including columns, types, nullability, primary/foreign keys, indexes, unique constraints, and server defaults.
3. Compare the snapshot against each candidate revision's complete schema—not just table names—and inspect any data or schema drift. Use the migration files and model metadata as evidence; do not assume the current ORM model describes the historical schema.
4. If, and only if, the schema is proven to match a specific supported revision exactly, have the database owner review and approve a one-time, explicit `alembic stamp <verified-revision>` recovery. The application and baseline migration never do this automatically. Record the evidence and approval, then back up and upgrade under the normal change process.
5. If the schema is partial, drifted, or cannot be tied to a known revision, create and review a dedicated data-preserving reconciliation migration. Do not stamp or drop/recreate tables to bypass the discrepancy.

No recovery stamp or configured-database operation was performed while implementing this migration path.
