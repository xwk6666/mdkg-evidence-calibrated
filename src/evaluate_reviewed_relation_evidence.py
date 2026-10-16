#!/usr/bin/env python
"""Evaluate KG relation evidence scores against human-reviewed relations.

The script keeps relation-level evaluation separate from image-text pair
classification. Scores are joined by stable sample and triple identifiers, and
uncertainty is estimated by resampling samples rather than individual triples.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = ROOT / "outputs" / "llm_review_300" / "relation_items_annotator1.csv"
DEFAULT_KG_CSV = ROOT / "outputs" / "kg_consistency_crisis_bert" / "test_relation_consistency.csv"
DEFAULT_SLOT_CSV = ROOT / "outputs" / "event_prototype_confidence" / "relation_slot_prototype_scores.csv"
DEFAULT_FT_CLIP_CSV = ROOT / "outputs" / "clip_image_text_consistency" / "finetuned_clip_fixed_semantic_hard_test_pairs.csv"
DEFAULT_SEMANTIC_CLIP_CSV = ROOT / "outputs" / "clip_image_text_consistency" / "clip_semantic_top5_seed2026_fixed_fixed_semantic_hard_test_pairs.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reviewed_relation_evidence"
DEFAULT_COVERAGES = (20.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().replace("\\", "/").split())


def relation_key(row: Mapping[str, object], reviewed: bool = False) -> Tuple[str, str, str, str]:
    prefix = "pred_" if reviewed else ""
    return (
        normalize_text(row.get("sample_id")),
        normalize_text(row.get(f"{prefix}head")),
        normalize_text(row.get(f"{prefix}relation")),
        normalize_text(row.get(f"{prefix}tail")),
    )


def relation_index(rows: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, str, str, str], Mapping[str, object]]:
    index: Dict[Tuple[str, str, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = relation_key(row)
        if key in index:
            raise ValueError(f"Duplicate relation key: {key}")
        index[key] = row
    return index


def optional_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    return float(text) if text else None


def positive_pair_score_index(
    rows: Iterable[Mapping[str, object]],
    score_column: str = "score",
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for row in rows:
        if int(float(str(row.get("label", 0)))) != 1 or str(row.get("kind", "")) != "positive":
            continue
        sample_id = normalize_text(row.get("image_sample_id"))
        if not sample_id:
            continue
        score = float(str(row[score_column]))
        if sample_id in scores and scores[sample_id] != score:
            raise ValueError(f"Conflicting positive-pair scores for sample: {sample_id}")
        scores[sample_id] = score
    return scores


def build_joined_rows(
    review_rows: Sequence[Mapping[str, object]],
    kg_rows: Sequence[Mapping[str, object]],
    slot_rows: Sequence[Mapping[str, object]],
    sample_scores: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, object]]:
    kg_index = relation_index(kg_rows)
    slot_index = relation_index(slot_rows)
    joined: List[Dict[str, object]] = []
    missing_kg: List[Tuple[str, str, str, str]] = []
    missing_slot: List[Tuple[str, str, str, str]] = []

    for review in review_rows:
        key = relation_key(review, reviewed=True)
        kg = kg_index.get(key)
        slot = slot_index.get(key)
        if kg is None:
            missing_kg.append(key)
            continue
        if slot is None:
            missing_slot.append(key)
            continue

        sample_id = key[0]
        row: Dict[str, object] = {
            "item_id": str(review.get("item_id", "")),
            "sample_id": sample_id,
            "event_id": str(review.get("event_id", "")),
            "event_family": str(review.get("event_id", "")).split("__", 1)[0],
            "relation_type": str(review.get("pred_relation", "")).strip().upper(),
            "head": str(review.get("pred_head", "")),
            "tail": str(review.get("pred_tail", "")),
            "label": int(float(str(review.get("correct_1_or_0", 0)))),
            "kg_consistency_score": float(str(kg.get("kg_consistency_score", 0))),
            "event_support_score": float(str(kg.get("event_support_score", 0))),
            "alignment_score": optional_float(kg.get("alignment_score")),
            "relation_slot_score": float(str(slot.get("relation_slot_score", 0))),
        }
        for score_name, score_index in sample_scores.items():
            row[score_name] = score_index.get(sample_id)
        joined.append(row)

    if missing_kg or missing_slot:
        raise ValueError(
            f"Unmatched reviewed relations: kg={len(missing_kg)}, slot={len(missing_slot)}"
        )
    return joined


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one positive and one negative relation.")
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    total_positive = sum(labels)
    if not total_positive:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_positive


def eligible_rows(rows: Sequence[Mapping[str, object]], score_field: str) -> List[Mapping[str, object]]:
    return [row for row in rows if row.get(score_field) is not None]


def quality_at_coverage(
    rows: Sequence[Mapping[str, object]],
    score_field: str,
    coverage_pct: float,
) -> Dict[str, object]:
    if coverage_pct <= 0 or coverage_pct > 100:
        raise ValueError("coverage_pct must be in (0, 100].")
    cohort = eligible_rows(rows, score_field)
    if not cohort:
        raise ValueError(f"No rows contain score field: {score_field}")
    ranked = sorted(cohort, key=lambda row: float(row[score_field]), reverse=True)
    retained_count = max(1, math.ceil(len(ranked) * coverage_pct / 100.0))
    retained = ranked[:retained_count]
    retained_correct = sum(int(row["label"]) for row in retained)
    total_correct = sum(int(row["label"]) for row in cohort)
    precision = retained_correct / retained_count
    recall = retained_correct / total_correct if total_correct else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "coverage_pct": float(coverage_pct),
        "eligible_relations": len(cohort),
        "retained_relations": retained_count,
        "retained_correct": retained_correct,
        "retained_errors": retained_count - retained_correct,
        "score_cutoff": float(retained[-1][score_field]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_relation_rate": 1.0 - precision,
    }


def score_summary(rows: Sequence[Mapping[str, object]], score_field: str) -> Dict[str, object]:
    cohort = eligible_rows(rows, score_field)
    scores = [float(row[score_field]) for row in cohort]
    labels = [int(row["label"]) for row in cohort]
    return {
        "score_field": score_field,
        "eligible_relations": len(cohort),
        "eligible_samples": len({str(row["sample_id"]) for row in cohort}),
        "correct_relations": sum(labels),
        "incorrect_relations": len(labels) - sum(labels),
        "unfiltered_precision": sum(labels) / len(labels),
        "auc": binary_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_group_bootstrap_auc(
    rows: Sequence[Mapping[str, object]],
    better_field: str,
    reference_field: str,
    repetitions: int,
    seed: int,
) -> Dict[str, object]:
    cohort = [
        row
        for row in rows
        if row.get(better_field) is not None and row.get(reference_field) is not None
    ]
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in cohort:
        grouped[str(row["sample_id"])].append(row)
    group_ids = sorted(grouped)
    if len(group_ids) < 2:
        raise ValueError("Grouped bootstrap requires at least two samples.")

    def delta(sampled_rows: Sequence[Mapping[str, object]]) -> float:
        labels = [int(row["label"]) for row in sampled_rows]
        better = [float(row[better_field]) for row in sampled_rows]
        reference = [float(row[reference_field]) for row in sampled_rows]
        return binary_auc(better, labels) - binary_auc(reference, labels)

    observed = delta(cohort)
    rng = random.Random(seed)
    bootstrapped: List[float] = []
    attempts = 0
    maximum_attempts = max(repetitions * 20, repetitions + 100)
    while len(bootstrapped) < repetitions and attempts < maximum_attempts:
        attempts += 1
        sampled_ids = [rng.choice(group_ids) for _ in group_ids]
        sampled_rows = [row for group_id in sampled_ids for row in grouped[group_id]]
        labels = {int(row["label"]) for row in sampled_rows}
        if labels != {0, 1}:
            continue
        bootstrapped.append(delta(sampled_rows))
    if len(bootstrapped) < repetitions:
        raise RuntimeError("Could not obtain enough valid grouped bootstrap samples.")

    probability_nonpositive = (sum(value <= 0 for value in bootstrapped) + 1) / (repetitions + 1)
    probability_nonnegative = (sum(value >= 0 for value in bootstrapped) + 1) / (repetitions + 1)
    return {
        "better_field": better_field,
        "reference_field": reference_field,
        "relations": len(cohort),
        "samples": len(group_ids),
        "repetitions": repetitions,
        "observed_delta": observed,
        "ci95_low": percentile(bootstrapped, 0.025),
        "ci95_high": percentile(bootstrapped, 0.975),
        "two_sided_p": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit KG relation evidence scores using 299 human-reviewed relations."
    )
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--kg-csv", type=Path, default=DEFAULT_KG_CSV)
    parser.add_argument("--slot-csv", type=Path, default=DEFAULT_SLOT_CSV)
    parser.add_argument("--ft-clip-csv", type=Path, default=DEFAULT_FT_CLIP_CSV)
    parser.add_argument("--semantic-clip-csv", type=Path, default=DEFAULT_SEMANTIC_CLIP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coverage", nargs="+", type=float, default=list(DEFAULT_COVERAGES))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    review_rows = read_csv(args.review_csv)
    kg_rows = read_csv(args.kg_csv)
    slot_rows = read_csv(args.slot_csv)
    sample_scores = {
        "ft_clip_score": positive_pair_score_index(read_csv(args.ft_clip_csv)),
        "semantic_hn_clip_score": positive_pair_score_index(read_csv(args.semantic_clip_csv)),
    }
    joined = build_joined_rows(review_rows, kg_rows, slot_rows, sample_scores)

    methods = [
        ("KG consistency", "kg_consistency_score", "all_reviewed"),
        ("Event-slot prototype", "relation_slot_score", "all_reviewed"),
        ("KG consistency", "kg_consistency_score", "clip_common"),
        ("Event-slot prototype", "relation_slot_score", "clip_common"),
        ("Fine-tuned CLIP", "ft_clip_score", "clip_common"),
        ("Semantic-HN CLIP", "semantic_hn_clip_score", "clip_common"),
    ]
    clip_common = [
        row
        for row in joined
        if row.get("ft_clip_score") is not None and row.get("semantic_hn_clip_score") is not None
    ]
    cohorts = {"all_reviewed": joined, "clip_common": clip_common}

    summary_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    for method_name, score_field, cohort_name in methods:
        cohort = cohorts[cohort_name]
        summary_rows.append(
            {
                "cohort": cohort_name,
                "method": method_name,
                **score_summary(cohort, score_field),
            }
        )
        for coverage in args.coverage:
            coverage_rows.append(
                {
                    "cohort": cohort_name,
                    "method": method_name,
                    "score_field": score_field,
                    **quality_at_coverage(cohort, score_field, coverage),
                }
            )

    bootstrap_results = [
        paired_group_bootstrap_auc(
            joined,
            better_field="relation_slot_score",
            reference_field="kg_consistency_score",
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
        paired_group_bootstrap_auc(
            clip_common,
            better_field="semantic_hn_clip_score",
            reference_field="ft_clip_score",
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 1,
        ),
    ]

    write_csv(args.output_dir / "reviewed_relation_joined_scores.csv", joined)
    write_csv(args.output_dir / "reviewed_relation_score_summary.csv", summary_rows)
    write_csv(args.output_dir / "reviewed_relation_quality_coverage.csv", coverage_rows)
    write_csv(args.output_dir / "reviewed_relation_bootstrap.csv", bootstrap_results)

    labels = [int(row["label"]) for row in joined]
    summary = {
        "reviewed_relations": len(joined),
        "reviewed_samples": len({str(row["sample_id"]) for row in joined}),
        "correct_relations": sum(labels),
        "incorrect_relations": len(labels) - sum(labels),
        "unfiltered_precision": sum(labels) / len(labels),
        "clip_common_relations": len(clip_common),
        "clip_common_samples": len({str(row["sample_id"]) for row in clip_common}),
        "score_summaries": summary_rows,
        "bootstrap": bootstrap_results,
    }
    (args.output_dir / "reviewed_relation_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Reviewed relations: {len(joined)} from {summary['reviewed_samples']} samples; "
        f"unfiltered precision={summary['unfiltered_precision']:.4f}"
    )
    for row in summary_rows:
        print(
            f"{row['cohort']} | {row['method']}: "
            f"n={row['eligible_relations']} AUC={row['auc']:.4f} AP={row['average_precision']:.4f}"
        )
    for result in bootstrap_results:
        print(
            f"Bootstrap {result['better_field']} - {result['reference_field']}: "
            f"delta={result['observed_delta']:.4f}, "
            f"95% CI [{result['ci95_low']:.4f}, {result['ci95_high']:.4f}], "
            f"p={result['two_sided_p']:.4f}"
        )
    print(f"Saved relation-level audit to: {args.output_dir}")


if __name__ == "__main__":
    main()
