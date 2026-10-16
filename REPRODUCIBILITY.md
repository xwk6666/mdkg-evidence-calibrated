# Software/data reproduction contract

Use scripts/reproduce_public.py --data-root PATH --target core or --target all.
The 19 method files and two original replay modules are unmodified snapshots.
The public adapter changes only input locations, integrity-manifest handling and
the Human300 join key (existing annotation_id instead of withheld sample_id).
Numerical functions, class definitions, scores, thresholds and row order are not
changed. The original scripts/reproduce.py is retained for source inspection; its
legacy monolithic-core command is not the split-release entrypoint.

Both payload manifests must pass. No synthetic split manifest is made. The absence
of original IDs means public replay cannot independently establish original test
membership or check dataset-identity leakage; prior split audit is reported
metadata. Historical/prospective relation matching uses precomputed flags, not
re-evaluation of undistributed endpoint strings.

core recomputes Human300 three-model Strict/Broad, historical200 candidates and
admission, fixed250 verification, and prospective reused200 relation counts.
all additionally checks the distributed table/figure numeric sources and replays
archived aggregates. Summary replay is not an independently reconstructed query,
bootstrap, graph image or database. See PAPER_EVIDENCE_MAP.md for target-level limits.

No raw datasets, weights, fitted binaries or API responses are supplied. Full
inference would require lawful originals and verified historical model/environment
provenance; this package does not promise it. No model downloads occur in replay.
