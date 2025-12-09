-- CryoFlux Lab - Canonical Metrics View
--
-- This view provides a normalized interface to the receipts table,
-- extracting the 'accepted' flag from the JSON meta field for easier querying.
--
-- Design rationale:
-- - The receipts table stores acceptance status in meta JSON field
-- - This view extracts it as a boolean column for simpler aggregation
-- - All other fields are passed through as-is
-- - The 'loss' column is included but currently unused (always 0.0)
-- - For backward compatibility, we do NOT alter the base receipts table

CREATE VIEW IF NOT EXISTS receipts_canonical AS
SELECT
    id,
    datetime(ts, 'unixepoch') AS timestamp,
    ts AS ts_unix,
    task AS task_name,
    joule AS joules_spent,
    sec AS execution_time_sec,
    delta,
    loss,
    delta_hash,
    meta,
    -- Extract 'accepted' from JSON meta field
    -- Returns 1 if accepted:true, 0 otherwise
    CASE
        WHEN json_extract(meta, '$.accepted') = 1 THEN 1
        WHEN json_extract(meta, '$.accepted') = 'true' THEN 1
        ELSE 0
    END AS accepted
FROM receipts;

-- Notes on delta semantics:
-- - For LoRA tasks: delta = max(0, base_loss - new_loss) [true learning improvement]
-- - For Index tasks: delta = embeddings_added / 1000.0 [normalized embedding count]
-- - η (eta) = delta / joules_spent [efficiency metric]
