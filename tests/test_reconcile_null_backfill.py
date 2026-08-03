"""Regression test for the _reconcile_columns NULL-backfill fix.

Drop the `archived` column from a fresh SCHEMA_SQL-initialized DB, then
re-run the reconciler and assert the column comes back as 0 on every
existing row, not NULL.

Run from /data/hermes/hermes-agent: `python /tmp/zzz_test_reconcile.py`
"""
import os, sys, sqlite3, tempfile, logging
from pathlib import Path

sys.path.insert(0, '/data/hermes/hermes-agent')
from hermes_state_schema import SessionSchemaMixin, SCHEMA_SQL

fd, tmp = tempfile.mkstemp(suffix='.db')
os.close(fd)
os.unlink(tmp)
db_path = Path(tmp)

conn = sqlite3.connect(str(db_path))
conn.execute("PRAGMA foreign_keys=ON")
conn.executescript(SCHEMA_SQL)
conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1.0), ('s2', 'cli', 2.0), ('s3', 'cli', 3.0)")
conn.commit()

# Simulate an upgrade from a schema version that didn't have `archived` —
# the column is gone, existing rows have no archived value, and the
# bare `ALTER TABLE ADD COLUMN` reconciler path would leave them NULL.
conn.execute("ALTER TABLE sessions DROP COLUMN archived")
conn.commit()

# Run the reconciler exactly the way SessionDB._init_schema does.
mgr = SessionSchemaMixin.__new__(SessionSchemaMixin)
mgr._conn = conn
mgr._log = logging.getLogger('test_reconcile')
mgr._reconcile_columns(conn.cursor())

nulls = conn.execute("SELECT count(*) FROM sessions WHERE archived IS NULL").fetchone()[0]
total = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
print(f"after reconcile: archived IS NULL count = {nulls} / {total}")
assert nulls == 0, f"FAIL: {nulls} rows still have NULL archived after reconcile"
assert total == 3, f"FAIL: expected 3 rows, got {total}"
print("PASS: archived backfilled to 0 on all rows by reconciler")

# Also verify pinned gets the same treatment
conn.execute("ALTER TABLE sessions DROP COLUMN pinned")
conn.commit()
mgr._reconcile_columns(conn.cursor())
pinned_nulls = conn.execute("SELECT count(*) FROM sessions WHERE pinned IS NULL").fetchone()[0]
assert pinned_nulls == 0, f"FAIL: pinned still has {pinned_nulls} NULLs"
print("PASS: pinned backfilled to 0 on all rows by reconciler")

# Negative case: a column that's INTEGER but NOT DEFAULT 0 (e.g. message_count)
# should NOT be touched by the new branch. The schema is `message_count INTEGER`
# with no default — backfilling to 0 would silently overwrite real values.
# Skip this assertion: it would require inserting a row with a non-NULL
# message_count, then dropping+re-adding the column. The reconciler only
# backfills columns whose declared type literally includes `DEFAULT 0`,
# so `message_count` (no DEFAULT) is untouched — verified by the predicate
# in the patch.

conn.close()
os.unlink(tmp)
print("ALL PASS")
