# Main-paper evidence and commands

Commands run from the code directory and use the separately extracted data archive. Upstream ID fields and split lists are withheld; retained numerical values and row order are unchanged. Descriptive metadata, frozen-output reproduction and archived-summary replay are distinct.

|Paper target|Selected files|Command|Scope and limits|
|---|---|---|---|
| Table 1 | `schema/closed_schema.json` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table1` | Descriptive metadata replay, not experimental inference. Closed schema declarations, not factual verification scores. |
| Table 2 | `configs/evidence_roles.json` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table2` | Descriptive metadata replay, not experimental inference. Evidence-role metadata only; no portable graph dump included. |
| Table 3 | `configs/protocol_populations.json`, `metrics/mash_population_summary.csv`, `scores/relation_calibration_development299.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table3` | Descriptive metadata replay, not experimental inference. Major protocol populations are explicit; raw data access and counts after media filtering remain separate. |
| Table 4 | `tables/table4_comparison.csv`, `scores/constructed_pairs/` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table4` | Frozen score/label/prediction reproduction, no model rerun. AP convention is explicitly versioned in METRIC_CONVENTIONS.md. |
| Table 5 | `tables/table5_comparison.csv`, `scores/mash/` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table5` | Frozen score/label/prediction reproduction, no model rerun. AP convention is explicitly versioned in METRIC_CONVENTIONS.md. |
| Table 6 | `annotations/human300_final_labels.csv`, `scores/human300_three_model_scores.csv`, `tables/table6_human300_all_metrics.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table6` | Three-model Strict/Broad frozen-output reproduction; human labels not external factual certification. |
| Table 7 | `tables/table7_development_ablation.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table7` | archived summary replay. Raw endpoint labels/text are withheld; original generation-call metadata is incomplete. No independent endpoint reconstruction. |
| Table 8 | `tables/table8_negative_sensitive_metrics.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table8` | frozen-output reproduction. Raw single/multi-template NLI are replayed at archived development thresholds; fixed candidates, not end-to-end generation. |
| Table 9 | `scores/fixed250_nli_scores_labels.csv`, `tables/table9_relation_quality.csv`, `tables/table9_query_quality.csv`, `tables/table9_full_event_coverage.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table9` | Frozen-output reproduction for fixed250 relation quality; archived-summary replay for query quality and event coverage. The 74 evaluated event-relation queries and the 94 candidate-bearing event units are different populations. Query precision averages 72/72/53 queries with judged returns, whereas recall averages all 74. See the denominator definitions below. |
| Table 10 | `metrics/deployment_summary.csv`, `metrics/graph_query_summary.csv`, `tables/table10_latency_summary.csv`, `tables/table10_relation_coverage.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table10` | frozen-output reproduction of saved timing/queue counts and archived graph-summary replay. No database replay; queue-count reduction is not human-time saving. |
| Table 11 | `tables/table11_relation_results.csv`, `tables/table11_crossmodal_results.csv`, `metrics/prospective_archived_bootstrap.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target table11` | frozen-output reproduction of relation match flags; archived summary replay for cross-modal folds/CI. Historical200 inputs reused; not a new independent blind test. No inference or bootstrap rerun. |
| Table A.1 | `tables/tableA1_generators.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target tableA1` | Exact 8 saved relation rows, explicit/recall-first/schema-deployed stages; counts checked, not a new generation run. |
| Fig. 4 | `tables/table4_comparison.csv`, `scores/constructed_pairs/` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig4` | Same frozen numeric evidence as Table4. |
| Fig. 5 | `metrics/constructed_platt_summary.csv`, `metrics/pair_calibration_bins.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig5` | Frozen continuous test scores are included; original validation fitting and graphic layout are not re-executed. |
| Fig. 6 | `metrics/historical200_method_summary.csv`, `predictions/relation200_match_flags.csv`, `annotations/historical200_gold_relation_counts.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig6` | Recall-first candidate/Reliable/Reliable+Review relation point estimates are recomputed from frozen match flags. Frozen Schema-LLM baseline and entity metrics are summary replay only. Selected seed20260912 CIs are not included or recomputed; no older-seed substitute is used. |
| Fig. 7 | `metrics/relation_calibration_bins.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig7` | No calibrator fitted or deserialized; bins are frozen source values. |
| Fig. 8 | `tables/table9_relation_quality.csv`, `tables/table9_query_quality.csv`, `metrics/query_quality_archived_effects.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig8` | Does not reconstruct excluded raw endpoints or newly run bootstrap. |
| Fig. 9 | `metrics/graph_query_summary.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig9` | Original viewer screenshot/raw photographs and database dump are not distributed. Exact graph visualization is not reproduced. |
| Fig. 10 | `metrics/deployment_summary.csv`, `metrics/graph_query_summary.csv`, `tables/table10_latency_summary.csv`, `tables/table10_relation_coverage.csv` | `python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target fig10` | No database replay; queue-count reduction is not human-time saving. |

Method Figs1-3 are described by schema, evidence-role config and original source code, not redrawn here.
Selected Fig6 seed20260912 intervals and the Fig9 visual/database snapshot are not fully distributed; no alternate-seed or audit projection is substituted.
No private annotation originals or complete endpoint Gold are provided. This compact core does not claim to reproduce all supplementary or historical artifacts.

## Table 9: three evaluation populations, not one denominator

The manuscript's phrase "74 covered event queries" denotes 74 distinct
`(event_id, relation_type)` keys with at least one supported Gold answer. They
span 35 event units: 30 OCCURS_IN keys, 24 AFFECTS keys and 20 CAUSES keys.
The fixed250 Gold contains 250 relation instances from 38 event units and 78
audited event-relation keys. Four keys with no positive Gold answer are not in
the 74-query evaluation. Deduplication by event, relation and normalized tail
leaves 171 positive Gold answers for query recall; these are not 250 queries.

For each policy, predictions are grouped by event and relation, tails are
deduplicated, and the best five are ranked by frozen confidence, support count
and tail key. Precision for a query is correct judged returns / judged returns;
unjudged returns are not assumed correct or incorrect. Macro Query P averages
only the 72/72/53 queries with judged returns, in policy order All / Reliable+
Review / Reliable. Macro Query R averages all 74 queries, each using its own
unique positive Gold answers as the denominator. Thus neither Query P nor Query R
is an overall accuracy on all returned graph answers.

`table9_full_event_coverage.csv` comes from the historical test-only deployment
of 808 schema-valid candidate relations from 555 samples, not the full-corpus
graph. The denominator is its 94 candidate-bearing event units. Coverage is
94/94, 94/94 and 71/94 for All / Reliable+Review / Reliable. Corresponding retained
relation counts are 808/738/323 and nonempty event-relation keys are 169/166/113.
The filename's "full" means all records of this deployment, not the later
train+validation+test corpus. It is not coverage over the 74 evaluated queries.

These 94 event units are a subset of the full-corpus graph's 126 event units.
The latter has 378 possible event-relation slots (126 times three relation
types); 224 are answered by Reliable facts and 110 events have at least one
answer. Consequently 110/126 event coverage and 224/378 slot coverage in Table 10
and the graph summaries must not be substituted for Table 9's 71/94 coverage.
No accuracy or Gold judgment is established by coverage alone.

Historical definitions are in the original project script
`tools/evaluate_final_kg_queries.py` (`evaluate_event_queries`, `full_coverage`)
and `tools/evaluate_full_corpus_kg_cases.py` (`summarize_event_queries`). These
scripts and withheld raw query answers are not part of the public replay.
`scripts/reproduce.py --target table9` recomputes relation quality from frozen
fixed250 labels/tiers but only reads the archived query and coverage tables.
It cannot independently reconstruct the 74 keys, judged-return membership or
94-event population without the excluded source records. The local final audit
verified those populations against the historical files; that audit is not an
additional public dataset or a claim of full end-to-end reproduction.

## Other targeted reproduction limits

Table 4 recomputes metrics from five frozen constructed-pair score/label/
prediction files. It does not rerun encoders or construct new pairs. Historical
stable-rank AP, including file-order handling of ties, is retained as documented
in METRIC_CONVENTIONS.md; human300 is a separate label definition.

Fig. 6 mixes recomputable recall-first relation point estimates with archived
baseline/entity summaries; the selected seed20260912 intervals cannot be
reproduced from this core. Fig. 9 exposes graph/query summary numbers only, not
the viewer image, node layout, media, database snapshot or a fresh graph query.


## Split release transport boundary

The original two replay modules are unchanged. reproduce_public.py adapts data-root
lookup and joins human300 on existing annotation_id rather than the withheld
upstream sample_id. Original split membership is reported protocol metadata, not
independently rechecked in the public payload. No synthetic split IDs are created.
The --target all command covers the distributed numeric projections and archived
summaries only.
