#!/usr/bin/env python
"""Evaluate a locked L2 relation-confidence fusion on independent Gold data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METHOD_SPECS = {
    "fixed_q": None,
    "qm_only": ("q_m", "q_m_missing"),
    "qe_only": ("q_e",),
    "nli_only": ("q_nli",),
    "nli_qm": ("q_nli", "q_m", "q_m_missing"),
    "nli_qe": ("q_nli", "q_e"),
    "full": ("q_nli", "q_m", "q_e", "q_m_missing"),
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_BASELINE = (
    ROOT / "outputs" / "reviewed_relation_evidence" / "reviewed_relation_joined_scores.csv"
)
DEFAULT_DEV_NLI = (
    ROOT / "outputs" / "relation_nli_verifier_ensemble" / "reviewed_relation_nli_scores.csv"
)
DEFAULT_GOLD = (
    ROOT
    / "outputs"
    / "relation_challenge_250"
    / "gold"
    / "relation_challenge_250_gold.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "relation_challenge_250" / "challenge_master_manifest.csv"
)
DEFAULT_GOLD_NLI = (
    ROOT
    / "outputs"
    / "relation_challenge_250"
    / "model_scores"
    / "nli_ensemble"
    / "relation_challenge_nli_scores.csv"
)
DEFAULT_KG = (
    ROOT / "outputs" / "kg_consistency_crisis_bert" / "test_relation_consistency.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "final_relation_confidence_fusion"
METHOD_LABELS = {
    "fixed_q": "Fixed-Q",
    "qm_only": "Qm-only",
    "qe_only": "Qe-only",
    "nli_only": "NLI-only",
    "nli_qm": "NLI+Qm",
    "nli_qe": "NLI+Qe",
    "full": "Full L2 fusion",
}


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def unique_index(
    rows: Sequence[Mapping[str, object]], key_field: str, source_name: str
) -> Dict[str, Mapping[str, object]]:
    index: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        key = str(row.get(key_field, "")).strip()
        if not key:
            raise ValueError(f"Blank {key_field} in {source_name}")
        if key in index:
            raise ValueError(f"Duplicate {key_field} {key} in {source_name}")
        index[key] = row
    return index


def finite_score(value: object, field: str, item_id: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing {field} for {item_id}")
    score = float(text)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite {field} for {item_id}: {text}")
    return score


def clipped_score(value: object, field: str, item_id: str) -> float:
    return min(1.0, max(0.0, finite_score(value, field, item_id)))


def optional_clipped_score(value: object, field: str, item_id: str) -> Tuple[float, int]:
    if not str(value or "").strip():
        return 0.0, 1
    return clipped_score(value, field, item_id), 0


def relation_key(row: Mapping[str, object]) -> Tuple[str, str, str, str]:
    return (
        normalize_text(row.get("sample_id")),
        normalize_text(row.get("head")),
        normalize_text(row.get("relation") or row.get("relation_type")),
        normalize_text(row.get("tail")),
    )


def relation_index(
    rows: Sequence[Mapping[str, object]], source_name: str
) -> Dict[Tuple[str, str, str, str], Mapping[str, object]]:
    index: Dict[Tuple[str, str, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = relation_key(row)
        if not all(key):
            raise ValueError(f"Blank relation key in {source_name}: {key}")
        if key in index:
            raise ValueError(f"Duplicate relation key in {source_name}: {key}")
        index[key] = row
    return index


def _require_same_ids(
    left: Mapping[str, object], right: Mapping[str, object], context: str
) -> None:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"{context} files do not contain the same IDs: "
            f"missing_left={missing_left[:5]}, missing_right={missing_right[:5]}"
        )


def _base_output_row(
    row: Mapping[str, object], item_id: str, label: int
) -> Dict[str, object]:
    return {
        "item_id": item_id,
        "sample_id": str(row.get("sample_id", "")).strip(),
        "event_id": str(row.get("event_id", "")).strip(),
        "event_family": str(row.get("event_family", "")).strip(),
        "relation_type": str(row.get("relation_type") or row.get("relation") or "")
        .strip()
        .upper(),
        "head": str(row.get("head", "")).strip(),
        "tail": str(row.get("tail", "")).strip(),
        "label": int(label),
    }


def load_development_rows(
    baseline_csv: Path | str,
    nli_csv: Path | str,
    expected_count: Optional[int] = 299,
) -> List[Dict[str, object]]:
    baseline = unique_index(read_csv(baseline_csv), "item_id", "development baseline")
    nli = unique_index(read_csv(nli_csv), "item_id", "development NLI")
    _require_same_ids(baseline, nli, "Development score")
    if expected_count is not None and len(baseline) != expected_count:
        raise ValueError(
            f"Expected {expected_count} development relations; found {len(baseline)}"
        )

    output: List[Dict[str, object]] = []
    for item_id, base in baseline.items():
        nli_row = nli[item_id]
        label = int(float(str(base.get("label", 0))))
        nli_label = int(float(str(nli_row.get("label", label))))
        if label != nli_label:
            raise ValueError(f"Development label mismatch for {item_id}")
        q_m, q_m_missing = optional_clipped_score(
            base.get("alignment_score"), "alignment_score", item_id
        )
        row = _base_output_row(base, item_id, label)
        row.update(
            {
                "q_m": q_m,
                "q_m_missing": q_m_missing,
                "q_e": clipped_score(
                    base.get("event_support_score"), "event_support_score", item_id
                ),
                "q_nli": clipped_score(
                    nli_row.get("nli_entailment_score"),
                    "nli_entailment_score",
                    item_id,
                ),
                "fixed_q": clipped_score(
                    base.get("kg_consistency_score"),
                    "kg_consistency_score",
                    item_id,
                ),
            }
        )
        output.append(row)
    return output


def load_gold_rows(
    gold_csv: Path | str,
    manifest_csv: Path | str,
    nli_csv: Path | str,
    kg_csv: Path | str,
    expected_count: Optional[int] = 250,
) -> List[Dict[str, object]]:
    gold_rows = read_csv(gold_csv)
    gold = unique_index(gold_rows, "blind_id", "Gold labels")
    manifest = unique_index(read_csv(manifest_csv), "blind_id", "challenge manifest")
    nli = unique_index(read_csv(nli_csv), "item_id", "Gold NLI")
    _require_same_ids(gold, manifest, "Gold manifest")
    _require_same_ids(gold, nli, "Gold NLI")
    if expected_count is not None and len(gold) != expected_count:
        raise ValueError(f"Expected {expected_count} Gold relations; found {len(gold)}")

    kg = relation_index(read_csv(kg_csv), "original KG evidence")
    output: List[Dict[str, object]] = []
    missing: List[str] = []
    for gold_row in gold_rows:
        item_id = str(gold_row.get("blind_id", "")).strip()
        key = relation_key(gold_row)
        evidence = kg.get(key)
        if evidence is None:
            missing.append(item_id)
            continue
        manifest_row = manifest[item_id]
        nli_row = nli[item_id]
        label = int(float(str(gold_row.get("gold_supported", 0))))
        q_m, q_m_missing = optional_clipped_score(
            evidence.get("alignment_score"), "alignment_score", item_id
        )
        base_for_identity = dict(gold_row)
        base_for_identity["relation_type"] = gold_row.get("relation", "")
        row = _base_output_row(base_for_identity, item_id, label)
        row.update(
            {
                "canonical_error_type": str(
                    gold_row.get("canonical_error_type", "none") or "none"
                ).strip(),
                "q_m": q_m,
                "q_m_missing": q_m_missing,
                "q_e": clipped_score(
                    evidence.get("event_support_score", manifest_row.get("event_support_score")),
                    "event_support_score",
                    item_id,
                ),
                "q_nli": clipped_score(
                    nli_row.get("nli_entailment_score"),
                    "nli_entailment_score",
                    item_id,
                ),
                "fixed_q": clipped_score(
                    evidence.get("kg_consistency_score", manifest_row.get("kg_consistency_score")),
                    "kg_consistency_score",
                    item_id,
                ),
            }
        )
        output.append(row)
    if missing:
        raise ValueError(
            f"Gold relations have missing original KG evidence ({len(missing)}): "
            + ", ".join(missing[:8])
        )
    return output


def validate_no_overlap(
    development_rows: Sequence[Mapping[str, object]],
    gold_rows: Sequence[Mapping[str, object]],
) -> None:
    development_ids = {str(row.get("item_id", "")).strip() for row in development_rows}
    gold_ids = {str(row.get("item_id", "")).strip() for row in gold_rows}
    item_overlap = development_ids & gold_ids
    if item_overlap:
        raise ValueError(f"item_id overlap between development and Gold: {sorted(item_overlap)[:5]}")

    development_samples = {
        normalize_text(row.get("sample_id")) for row in development_rows
    }
    gold_samples = {normalize_text(row.get("sample_id")) for row in gold_rows}
    sample_overlap = development_samples & gold_samples
    if sample_overlap:
        raise ValueError(
            f"sample_id overlap between development and Gold: {sorted(sample_overlap)[:5]}"
        )


def build_feature_matrix(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[field]) for field in fields] for row in rows], dtype=float
    )
    if matrix.ndim != 2 or matrix.shape != (len(rows), len(fields)):
        raise ValueError("Feature matrix has an unexpected shape")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return matrix


def fit_methods(
    development_rows: Sequence[Mapping[str, object]],
) -> Dict[str, Optional[Pipeline]]:
    labels = np.asarray([int(row["label"]) for row in development_rows], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Development labels must contain both classes")
    models: Dict[str, Optional[Pipeline]] = {"fixed_q": None}
    for method, fields in METHOD_SPECS.items():
        if fields is None:
            continue
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="liblinear",
                        class_weight=None,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(build_feature_matrix(development_rows, fields), labels)
        models[method] = pipeline
    return models


def predict_methods(
    models: Mapping[str, Optional[Pipeline]],
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for method, fields in METHOD_SPECS.items():
        if fields is None:
            scores = np.asarray([float(row["fixed_q"]) for row in rows], dtype=float)
        else:
            model = models.get(method)
            if model is None:
                raise ValueError(f"Missing fitted model for {method}")
            scores = model.predict_proba(build_feature_matrix(rows, fields))[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite predictions for {method}")
        output[method] = scores
    return output


def classification_metrics(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> Dict[str, object]:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("Scores and labels must have the same non-zero length")
    predictions = (score_array >= float(threshold)).astype(int)
    tp = int(np.sum((predictions == 1) & (label_array == 1)))
    fp = int(np.sum((predictions == 1) & (label_array == 0)))
    tn = int(np.sum((predictions == 0) & (label_array == 0)))
    fn = int(np.sum((predictions == 0) & (label_array == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "retained": int(tp + fp),
        "accuracy": (tp + tn) / len(label_array),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _threshold_candidates(scores: Sequence[float]) -> List[float]:
    values = sorted({float(score) for score in scores})
    if not values:
        raise ValueError("Cannot select a threshold from empty scores")
    return values


def select_binary_threshold(
    scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, object]:
    rows = [
        classification_metrics(scores, labels, threshold)
        for threshold in _threshold_candidates(scores)
    ]
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
            row["precision"],
            row["threshold"],
        ),
    )


def select_precision_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    target_precision: float,
    min_retained: int = 10,
    max_threshold: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    if min_retained < 1:
        raise ValueError("min_retained must be positive")
    eligible = []
    for threshold in _threshold_candidates(scores):
        if max_threshold is not None and threshold > max_threshold:
            continue
        row = classification_metrics(scores, labels, threshold)
        if row["retained"] >= min_retained and row["precision"] >= target_precision:
            eligible.append(row)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (row["retained"], row["precision"], row["threshold"]),
    )


def freeze_thresholds(
    development_scores: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    min_retained: int = 10,
) -> Dict[str, Dict[str, Optional[Dict[str, object]]]]:
    frozen: Dict[str, Dict[str, Optional[Dict[str, object]]]] = {}
    for method in METHOD_SPECS:
        scores = development_scores[method]
        reliable = select_precision_threshold(
            scores, labels, target_precision=0.95, min_retained=min_retained
        )
        review = select_precision_threshold(
            scores,
            labels,
            target_precision=0.90,
            min_retained=min_retained,
            max_threshold=reliable["threshold"] if reliable is not None else None,
        )
        frozen[method] = {
            "binary": select_binary_threshold(scores, labels),
            "reliable": reliable,
            "review": review,
        }
    return frozen


def ranking_metrics(
    scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, float]:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("Scores and labels must have the same non-zero length")
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("Ranking metrics require both classes")
    return {
        "auc": float(roc_auc_score(label_array, score_array)),
        "average_precision": float(
            average_precision_score(label_array, score_array)
        ),
    }


def brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("Scores and labels must have the same non-zero length")
    return float(np.mean((score_array - label_array) ** 2))


def equal_frequency_ece(
    scores: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("Scores and labels must have the same non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    order = np.argsort(score_array, kind="mergesort")
    partitions = np.array_split(order, min(bins, len(order)))
    error = 0.0
    for indices in partitions:
        if len(indices) == 0:
            continue
        error += (len(indices) / len(order)) * abs(
            float(np.mean(score_array[indices]))
            - float(np.mean(label_array[indices]))
        )
    return float(error)


def quality_at_coverage(
    scores: Sequence[float], labels: Sequence[int], coverage_pct: float
) -> Dict[str, object]:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if len(score_array) == 0 or len(score_array) != len(label_array):
        raise ValueError("Scores and labels must have the same non-zero length")
    if not 0.0 < coverage_pct <= 100.0:
        raise ValueError("coverage_pct must be in (0, 100]")
    retained = max(1, int(math.ceil(len(score_array) * coverage_pct / 100.0)))
    selected = np.argsort(-score_array, kind="mergesort")[:retained]
    selected_labels = label_array[selected]
    tp = int(np.sum(selected_labels == 1))
    fp = retained - tp
    total_positive = int(np.sum(label_array == 1))
    precision = tp / retained
    recall = tp / total_positive if total_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "coverage_pct": float(coverage_pct),
        "retained": retained,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": fp / retained,
        "true_positive": tp,
        "false_positive": fp,
    }


def assign_tiers(
    scores: Sequence[float],
    reliable_threshold: Optional[float],
    review_threshold: Optional[float],
) -> np.ndarray:
    score_array = np.asarray(scores, dtype=float)
    tiers = np.full(len(score_array), "reject", dtype=object)
    if review_threshold is not None:
        tiers[score_array >= float(review_threshold)] = "review"
    if reliable_threshold is not None:
        tiers[score_array >= float(reliable_threshold)] = "reliable"
    return tiers


def group_bootstrap_indices(
    groups: Sequence[object], rng: np.random.Generator
) -> np.ndarray:
    group_array = np.asarray(groups, dtype=object)
    if len(group_array) == 0:
        raise ValueError("Cannot bootstrap empty groups")
    unique_groups = list(dict.fromkeys(group_array.tolist()))
    members = {
        group: np.flatnonzero(group_array == group) for group in unique_groups
    }
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    return np.concatenate([members[group] for group in sampled]).astype(int)


def _bootstrap_metric_bundle(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    ranking = ranking_metrics(scores, labels)
    classification = classification_metrics(scores, labels, threshold)
    coverage = quality_at_coverage(scores, labels, 50.0)
    return {
        "auc": ranking["auc"],
        "average_precision": ranking["average_precision"],
        "f1": float(classification["f1"]),
        "balanced_accuracy": float(classification["balanced_accuracy"]),
        "brier": brier_score(scores, labels),
        "precision_at_50": float(coverage["precision"]),
    }


def grouped_bootstrap_distributions(
    rows: Sequence[Mapping[str, object]],
    scores_by_method: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    binary_thresholds: Mapping[str, float],
    repetitions: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    label_array = np.asarray(labels, dtype=int)
    groups = np.asarray([str(row.get("sample_id", "")) for row in rows], dtype=object)
    if len(label_array) != len(rows) or any(not group for group in groups):
        raise ValueError("Rows, labels, and nonblank sample groups must align")
    score_arrays = {
        method: np.asarray(scores, dtype=float)
        for method, scores in scores_by_method.items()
    }
    for method, scores in score_arrays.items():
        if len(scores) != len(rows):
            raise ValueError(f"Score length mismatch for {method}")
        if method not in binary_thresholds:
            raise ValueError(f"Missing binary threshold for {method}")

    metric_names = (
        "auc",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "brier",
        "precision_at_50",
    )
    distributions = {
        method: {metric: [] for metric in metric_names} for method in score_arrays
    }
    rng = np.random.default_rng(seed)
    effective = 0
    for _ in range(repetitions):
        indices = group_bootstrap_indices(groups, rng)
        sampled_labels = label_array[indices]
        if set(sampled_labels.tolist()) != {0, 1}:
            continue
        effective += 1
        for method, scores in score_arrays.items():
            bundle = _bootstrap_metric_bundle(
                scores[indices], sampled_labels, float(binary_thresholds[method])
            )
            for metric, value in bundle.items():
                distributions[method][metric].append(float(value))
    if effective == 0:
        raise ValueError("No valid two-class bootstrap repetitions")
    return {
        "requested_repetitions": int(repetitions),
        "effective_repetitions": int(effective),
        "seed": int(seed),
        "methods": distributions,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("Cannot compute an interval from empty values")
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def bootstrap_confidence_interval_rows(
    bootstrap: Mapping[str, object]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, metrics in bootstrap["methods"].items():
        for metric, values in metrics.items():
            lower, upper = percentile_interval(values)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "effective_repetitions": bootstrap["effective_repetitions"],
                }
            )
    return rows


def paired_bootstrap_delta_rows(
    bootstrap: Mapping[str, object],
    comparisons: Sequence[Tuple[str, str]] = (
        ("full", "nli_only"),
        ("full", "fixed_q"),
    ),
) -> List[Dict[str, object]]:
    methods = bootstrap["methods"]
    rows: List[Dict[str, object]] = []
    for metric in methods[comparisons[0][0]]:
        metric_rows = []
        for primary, baseline in comparisons:
            primary_values = np.asarray(methods[primary][metric], dtype=float)
            baseline_values = np.asarray(methods[baseline][metric], dtype=float)
            if len(primary_values) != len(baseline_values):
                raise ValueError("Paired bootstrap distributions are misaligned")
            delta = primary_values - baseline_values
            lower, upper = percentile_interval(delta)
            non_positive = (np.sum(delta <= 0.0) + 1) / (len(delta) + 1)
            non_negative = (np.sum(delta >= 0.0) + 1) / (len(delta) + 1)
            p_value = min(1.0, 2.0 * min(non_positive, non_negative))
            metric_rows.append(
                {
                    "comparison": f"{primary}-{baseline}",
                    "metric": metric,
                    "mean_delta": float(np.mean(delta)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": p_value,
                    "effective_repetitions": len(delta),
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, corrected in zip(metric_rows, adjusted):
            row["holm_p_value"] = corrected
            rows.append(row)
    return rows


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _tier_threshold(record: Optional[Mapping[str, object]]) -> Optional[float]:
    return float(record["threshold"]) if record is not None else None


def evaluate_method_record(
    method: str,
    scores: Sequence[float],
    labels: Sequence[int],
    thresholds: Mapping[str, Optional[Mapping[str, object]]],
) -> Dict[str, object]:
    binary_threshold = float(thresholds["binary"]["threshold"])
    ranking = ranking_metrics(scores, labels)
    classification = classification_metrics(scores, labels, binary_threshold)
    reliable_threshold = _tier_threshold(thresholds.get("reliable"))
    review_threshold = _tier_threshold(thresholds.get("review"))
    tiers = assign_tiers(scores, reliable_threshold, review_threshold)
    label_array = np.asarray(labels, dtype=int)
    record: Dict[str, object] = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "relations": len(label_array),
        "positive_relations": int(np.sum(label_array == 1)),
        "negative_relations": int(np.sum(label_array == 0)),
        **ranking,
        **classification,
        "brier": brier_score(scores, labels),
        "ece": equal_frequency_ece(scores, labels, bins=10),
        "reliable_threshold": reliable_threshold,
        "review_threshold": review_threshold,
    }
    for tier in ("reliable", "review", "reject"):
        mask = tiers == tier
        count = int(np.sum(mask))
        record[f"{tier}_count"] = count
        record[f"{tier}_precision"] = (
            float(np.mean(label_array[mask])) if count else None
        )
    return record


def scored_rows(
    rows: Sequence[Mapping[str, object]],
    scores_by_method: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, Mapping[str, Optional[Mapping[str, object]]]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    tier_arrays = {}
    for method in METHOD_SPECS:
        tier_arrays[method] = assign_tiers(
            scores_by_method[method],
            _tier_threshold(thresholds[method].get("reliable")),
            _tier_threshold(thresholds[method].get("review")),
        )
    for index, source in enumerate(rows):
        row = dict(source)
        for method in METHOD_SPECS:
            row[f"{method}_score"] = float(scores_by_method[method][index])
            row[f"{method}_tier"] = str(tier_arrays[method][index])
        output.append(row)
    return output


def relation_type_metric_rows(
    rows: Sequence[Mapping[str, object]],
    scores_by_method: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, Mapping[str, Optional[Mapping[str, object]]]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    relation_types = sorted({str(row["relation_type"]) for row in rows})
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    for relation_type in relation_types:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["relation_type"] == relation_type],
            dtype=int,
        )
        subset_labels = labels[indices]
        positives = int(np.sum(subset_labels == 1))
        negatives = int(np.sum(subset_labels == 0))
        for method in METHOD_SPECS:
            subset_scores = np.asarray(scores_by_method[method], dtype=float)[indices]
            classification = classification_metrics(
                subset_scores,
                subset_labels,
                float(thresholds[method]["binary"]["threshold"]),
            )
            ranking = (
                ranking_metrics(subset_scores, subset_labels)
                if set(subset_labels.tolist()) == {0, 1}
                else {"auc": None, "average_precision": None}
            )
            output.append(
                {
                    "relation_type": relation_type,
                    "method": method,
                    "relations": len(indices),
                    "positives": positives,
                    "negatives": negatives,
                    "stable_subgroup": positives >= 10 and negatives >= 10,
                    **ranking,
                    **classification,
                }
            )
    return output


def paper_table(summary_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Final relation-confidence fusion on independent Gold",
        "",
        "| Method | AUC | AP | Precision | Recall | F1 | Balanced Acc. | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method_label} | {auc:.4f} | {average_precision:.4f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} | "
            "{balanced_accuracy:.4f} | {brier:.4f} | {ece:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _find_delta(
    delta_rows: Sequence[Mapping[str, object]], comparison: str, metric: str
) -> Mapping[str, object]:
    matches = [
        row
        for row in delta_rows
        if row["comparison"] == comparison and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one delta row for {comparison}/{metric}")
    return matches[0]


def render_chinese_report(
    summary_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    development_count: int,
    gold_count: int,
) -> Tuple[str, bool]:
    full = next(row for row in summary_rows if row["method"] == "full")
    nli = next(row for row in summary_rows if row["method"] == "nli_only")
    auc_delta = _find_delta(delta_rows, "full-nli_only", "auc")
    p50_delta = _find_delta(delta_rows, "full-nli_only", "precision_at_50")
    supported = bool(
        float(auc_delta["ci_lower"]) > 0.0
        or float(p50_delta["ci_lower"]) > 0.0
    )
    if supported:
        conclusion = "完整融合至少在一个锁定主指标上取得了置信区间大于0的增益，可以支持多源证据融合提升关系质量控制的主张。"
    else:
        conclusion = "完整融合相对NLI-only没有获得置信区间明确大于0的增益，论文应把NLI作为主要写入分数，图文与事件证据保留为溯源和辅助复核信息。"
    lines = [
        "# 最终关系置信度融合实验",
        "",
        "## 简短中文总结",
        "",
        f"- **怎么做：** 使用旧{development_count}条关系学习L2逻辑回归系数和写入阈值，模型冻结后在独立{gold_count}条Gold关系上评价。",
        "- **原理：** 本体约束先执行硬门控；逻辑回归综合多模板NLI、图文一致性、事件重复支持和图像缺失状态，L2正则用于限制小样本过拟合。",
        "- **能够实现什么：** 为每条Schema合法关系输出统一置信度，并根据开发集冻结阈值分成reliable、review和reject。",
        f"- **效果怎么样：** Full模型AUC={full['auc']:.4f}、F1={full['f1']:.4f}；NLI-only AUC={nli['auc']:.4f}、F1={nli['f1']:.4f}。{conclusion}",
        "",
        "## 统计判定",
        "",
        f"- Full-NLI-only AUC差值：{auc_delta['mean_delta']:.4f}，95% CI [{auc_delta['ci_lower']:.4f}, {auc_delta['ci_upper']:.4f}]，Holm校正p={auc_delta['holm_p_value']:.4f}。",
        f"- Full-NLI-only 50%覆盖率Precision差值：{p50_delta['mean_delta']:.4f}，95% CI [{p50_delta['ci_lower']:.4f}, {p50_delta['ci_upper']:.4f}]，Holm校正p={p50_delta['holm_p_value']:.4f}。",
        "",
        conclusion,
    ]
    return "\n".join(lines) + "\n", supported


def run_experiment(
    *,
    dev_baseline_csv: Path | str,
    dev_nli_csv: Path | str,
    gold_csv: Path | str,
    manifest_csv: Path | str,
    gold_nli_csv: Path | str,
    kg_csv: Path | str,
    output_dir: Path | str,
    expected_development: Optional[int] = 299,
    expected_gold: Optional[int] = 250,
    bootstrap_repetitions: int = 5000,
    min_retained: int = 10,
) -> Dict[str, object]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {
        "dev_baseline": Path(dev_baseline_csv),
        "dev_nli": Path(dev_nli_csv),
        "gold": Path(gold_csv),
        "manifest": Path(manifest_csv),
        "gold_nli": Path(gold_nli_csv),
        "kg": Path(kg_csv),
    }
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}

    development = load_development_rows(
        inputs["dev_baseline"], inputs["dev_nli"], expected_development
    )
    gold = load_gold_rows(
        inputs["gold"],
        inputs["manifest"],
        inputs["gold_nli"],
        inputs["kg"],
        expected_gold,
    )
    validate_no_overlap(development, gold)
    models = fit_methods(development)
    development_scores = predict_methods(models, development)
    development_labels = np.asarray([row["label"] for row in development], dtype=int)
    thresholds = freeze_thresholds(
        development_scores, development_labels, min_retained=min_retained
    )
    gold_scores = predict_methods(models, gold)
    gold_labels = np.asarray([row["label"] for row in gold], dtype=int)
    summary_rows = [
        evaluate_method_record(method, gold_scores[method], gold_labels, thresholds[method])
        for method in METHOD_SPECS
    ]
    coverage_rows = []
    for method in METHOD_SPECS:
        for coverage in (20.0, 50.0, 100.0):
            coverage_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **quality_at_coverage(gold_scores[method], gold_labels, coverage),
                }
            )
    binary_thresholds = {
        method: float(thresholds[method]["binary"]["threshold"])
        for method in METHOD_SPECS
    }
    bootstrap = grouped_bootstrap_distributions(
        gold,
        gold_scores,
        gold_labels,
        binary_thresholds,
        repetitions=bootstrap_repetitions,
        seed=42,
    )
    interval_rows = bootstrap_confidence_interval_rows(bootstrap)
    delta_rows = paired_bootstrap_delta_rows(bootstrap)
    relation_rows = relation_type_metric_rows(gold, gold_scores, thresholds)
    report, fusion_supported = render_chinese_report(
        summary_rows, delta_rows, len(development), len(gold)
    )
    summary = {
        "development_relations": len(development),
        "development_positive": int(np.sum(development_labels == 1)),
        "development_negative": int(np.sum(development_labels == 0)),
        "gold_relations": len(gold),
        "gold_positive": int(np.sum(gold_labels == 1)),
        "gold_negative": int(np.sum(gold_labels == 0)),
        "methods": summary_rows,
        "fusion_improvement_supported": fusion_supported,
        "bootstrap_requested": bootstrap["requested_repetitions"],
        "bootstrap_effective": bootstrap["effective_repetitions"],
    }
    lock = {
        "analysis_name": "final_relation_confidence_fusion",
        "input_paths": {name: str(path.resolve()) for name, path in inputs.items()},
        "input_sha256": input_hashes,
        "development_relations": len(development),
        "gold_relations": len(gold),
        "feature_definitions": METHOD_SPECS,
        "model": {
            "type": "L2 logistic regression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "class_weight": None,
            "max_iter": 2000,
            "random_state": 42,
            "continuous_scaling": "development StandardScaler",
        },
        "threshold_policy": {
            "binary": "maximum development balanced accuracy",
            "reliable_precision_target": 0.95,
            "review_precision_target": 0.90,
            "min_retained": min_retained,
        },
        "thresholds": thresholds,
        "bootstrap": {
            "unit": "sample_id",
            "requested_repetitions": bootstrap_repetitions,
            "effective_repetitions": bootstrap["effective_repetitions"],
            "seed": 42,
        },
        "locked_comparisons": ["full-nli_only", "full-fixed_q"],
        "script_sha256": sha256_file(Path(__file__)),
    }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        write_json(staging / "analysis_lock.json", lock)
        joblib.dump(
            {
                "models": models,
                "thresholds": thresholds,
                "method_specs": METHOD_SPECS,
                "input_sha256": input_hashes,
            },
            staging / "final_confidence_fusion.joblib",
        )
        write_csv(
            staging / "development_scores.csv",
            scored_rows(development, development_scores, thresholds),
        )
        write_csv(
            staging / "gold_scores.csv",
            scored_rows(gold, gold_scores, thresholds),
        )
        write_csv(staging / "ablation_summary.csv", summary_rows)
        write_csv(staging / "quality_coverage.csv", coverage_rows)
        write_csv(staging / "bootstrap_confidence_intervals.csv", interval_rows)
        write_csv(staging / "paired_bootstrap_deltas.csv", delta_rows)
        write_csv(staging / "relation_type_metrics.csv", relation_rows)
        write_json(staging / "final_relation_confidence_summary.json", summary)
        (staging / "final_relation_confidence_report.md").write_text(
            report, encoding="utf-8"
        )
        (staging / "paper_table.md").write_text(
            paper_table(summary_rows), encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-baseline-csv", type=Path, default=DEFAULT_DEV_BASELINE)
    parser.add_argument("--dev-nli-csv", type=Path, default=DEFAULT_DEV_NLI)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold-nli-csv", type=Path, default=DEFAULT_GOLD_NLI)
    parser.add_argument("--kg-csv", type=Path, default=DEFAULT_KG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--min-retained", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_experiment(
        dev_baseline_csv=args.dev_baseline_csv,
        dev_nli_csv=args.dev_nli_csv,
        gold_csv=args.gold_csv,
        manifest_csv=args.manifest_csv,
        gold_nli_csv=args.gold_nli_csv,
        kg_csv=args.kg_csv,
        output_dir=args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        min_retained=args.min_retained,
    )
    print(f"Development relations: {summary['development_relations']}")
    print(f"Independent Gold relations: {summary['gold_relations']}")
    print(
        "Fusion improvement supported: "
        f"{summary['fusion_improvement_supported']}"
    )
    print(f"Saved outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
