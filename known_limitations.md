# Known limitations

Frozen-output reproduction, not full end-to-end reproduction.
The current source snapshot does not establish byte-identical historical API
prompts/responses or an immutable DeepSeek alias. The prompt files are explicitly
current templates, not reconstructed proof of a historical request.

Upstream-linkable sample/platform IDs, dataset row indices and original split
lists have been withheld. Existing annotation_id/blind_id/item_id values are
retained where they are local opaque identifiers; no replacement source IDs or
synthetic split membership are created. Final labels and retained score fields,
row order and counts remain unchanged. Public data cannot independently verify
the original-source split membership or rerun endpoint matching.

The 300 final labels were independently assigned by two different human annotators,
without AI participation: 254 original agreements and 46 manual adjudications,
reported unweighted Cohen's kappa 0.7501. Individual A/B labels are not included,
so kappa and annotation-process independence are not independently reconstructed
from this archive. A separate completed 46-row arbitration original was not
located; no substitute original is fabricated. ITG0137, ITG0219 and ITG0230 retain
the documented original-text display uncertainty; labels are unchanged, source
texts are not redistributed, and no corrected interpretation is asserted.

Human300 uses six-family/frozen-score stratification of previously used original
test inputs, not a new development-unseen external test population. Prospective
seven-family folds reuse historical200 inputs. Constructed-pair discrimination,
human event support and text relation support are different targets.

Fig.6: recall-first relation point estimates can be recomputed from frozen match
flags. Schema-LLM baseline/entity summaries are archived replay. Selected
seed20260912 intervals are not distributed or recomputed.
Fig.9: graph summary numbers only; no viewer image, full Neo4j snapshot or layout.
Table9: 250 fixed candidates, 74 event-relation queries and 94 candidate-bearing
test events are separate populations. Query/coverage macro metrics are archived
summary replay without raw answer membership. Full-corpus 126 events/378 slots
are different again. Coverage/provenance do not establish factual correctness.

The original historical environment is incompletely pinned; no checkpoint or
serialized calibrator is supplied. Some historical workflow producers are outside
the selected source snapshot. Do not claim install-and-run historical inference.
Review is pending review, not human-confirmed truth. Queue-count reduction is not
measured human time saved. No new bootstrap or experimental results are generated.
