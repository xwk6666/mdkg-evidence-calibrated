#!/usr/bin/env python
"""Deploy the frozen multi-template NLI confidence policy to all test relations."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np

from score_reviewed_relations_nli import (
    DEFAULT_MODEL_NAME,
    build_relation_hypothesis,
    score_records,
    str_to_bool,
)


RelationKey = Tuple[str, str, str, str]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_JSON = (
    ROOT
    / "outputs"
    / "kg_consistency_crisis_bert"
    / "test_consistency_filtered_samples.json"
)
DEFAULT_EVIDENCE_CSV = (
    ROOT / "outputs" / "kg_consistency_crisis_bert" / "test_relation_consistency.csv"
)
DEFAULT_POLICY = (
    ROOT
    / "outputs"
    / "final_relation_confidence_fusion"
    / "final_confidence_fusion.joblib"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_relation_nli_deployment"


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().replace("\\", "/").split())


def relation_key(
    sample_id: object,
    head: object,
    relation: object,
    tail: object,
) -> RelationKey:
    return (
        normalize_text(sample_id),
        normalize_text(head),
        normalize_text(relation),
        normalize_text(tail),
    )


def stable_item_id(key: RelationKey) -> str:
    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:16]
    return f"REL-{digest}"


def _unique_relation_index(
    rows: Sequence[Mapping[str, object]], source_name: str
) -> Dict[RelationKey, Mapping[str, object]]:
    index: Dict[RelationKey, Mapping[str, object]] = {}
    for row in rows:
        key = relation_key(
            row.get("sample_id"),
            row.get("head"),
            row.get("relation") or row.get("relation_type"),
            row.get("tail"),
        )
        if not all(key):
            raise ValueError(f"Blank relation key in {source_name}: {key}")
        if key in index:
            raise ValueError(f"Duplicate relation key in {source_name}: {key}")
        index[key] = row
    return index


def _flatten_payload(payload: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sample_index, sample in enumerate(payload.get("samples", []) or []):
        sample_id = str(sample.get("sample_id") or sample.get("filename") or "").replace(
            "\\", "/"
        )
        for relation_index, relation in enumerate(sample.get("relations", []) or []):
            rows.append(
                {
                    "sample_id": sample_id,
                    "event_id": str(sample.get("event_id", "")),
                    "filename": str(sample.get("filename", "")),
                    "text": str(sample.get("text", "")),
                    "head": str(relation.get("head", "")).strip(),
                    "relation": str(relation.get("relation", "")).strip().upper(),
                    "tail": str(relation.get("tail", "")).strip(),
                    "_sample_index": sample_index,
                    "_relation_index": relation_index,
                }
            )
    return rows


def _float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


def prepare_deployment_records(
    payload: Mapping[str, object],
    evidence_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Join all JSON relations to evidence rows without silently dropping relations."""

    payload_rows = _flatten_payload(payload)
    payload_index = _unique_relation_index(payload_rows, "filtered sample JSON")
    evidence_index = _unique_relation_index(evidence_rows, "relation evidence CSV")
    if set(payload_index) != set(evidence_index):
        missing_evidence = sorted(set(payload_index) - set(evidence_index))
        missing_payload = sorted(set(evidence_index) - set(payload_index))
        raise ValueError(
            "JSON/CSV relation sets do not match: "
            f"missing_evidence={missing_evidence[:3]}, missing_payload={missing_payload[:3]}"
        )

    records: List[Dict[str, object]] = []
    for row in payload_rows:
        key = relation_key(row["sample_id"], row["head"], row["relation"], row["tail"])
        evidence = evidence_index[key]
        relation = str(row["relation"]).upper()
        record = {
            "item_id": stable_item_id(key),
            "sample_id": str(row["sample_id"]),
            "event_id": str(row["event_id"]),
            "event_family": str(row["event_id"]).split("__", 1)[0],
            "filename": str(row["filename"]),
            "text": str(row["text"]),
            "head": str(row["head"]),
            "relation_type": relation,
            "tail": str(row["tail"]),
            "hypothesis": build_relation_hypothesis(
                str(row["head"]), relation, str(row["tail"])
            ),
            "relation_slot_score": 1.0,
            "schema_valid": normalize_text(evidence.get("schema_valid"))
            in {"true", "1", "yes"},
            "invalid_reason": str(evidence.get("invalid_reason", "")),
            "raw_kg_consistency_score": _float_or_none(
                evidence.get("raw_kg_consistency_score")
                or evidence.get("kg_consistency_score")
            ),
            "_relation_key": key,
            "_sample_index": int(row["_sample_index"]),
            "_relation_index": int(row["_relation_index"]),
            "_evidence": dict(evidence),
        }
        records.append(record)
    return records


def assign_quality_tier(
    score: float,
    schema_valid: bool,
    thresholds: Mapping[str, float],
) -> str:
    if not schema_valid:
        return "reject"
    reliable = float(thresholds["reliable"])
    review = float(thresholds["review"])
    if reliable <= review:
        raise ValueError("Reliable threshold must be greater than review threshold.")
    if score >= reliable:
        return "reliable"
    if score >= review:
        return "review"
    return "reject"


def load_frozen_nli_policy(path: Path | str):
    policy_path = Path(path)
    bundle = joblib.load(policy_path)
    if not isinstance(bundle, Mapping):
        raise ValueError("Frozen policy must be a mapping.")
    models = bundle.get("models", {})
    model = models.get("nli_only") if isinstance(models, Mapping) else None
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Frozen policy is missing the nli_only probability model.")

    specs = bundle.get("method_specs", {})
    features = list(specs.get("nli_only", ())) if isinstance(specs, Mapping) else []
    if features != ["q_nli"]:
        raise ValueError(f"nli_only must use exactly q_nli; received {features}.")

    all_thresholds = bundle.get("thresholds", {})
    nli_thresholds = (
        all_thresholds.get("nli_only", {}) if isinstance(all_thresholds, Mapping) else {}
    )
    output_thresholds: Dict[str, float] = {}
    for tier in ("reliable", "review"):
        threshold_info = nli_thresholds.get(tier)
        if not isinstance(threshold_info, Mapping) or threshold_info.get("threshold") is None:
            raise ValueError(f"Frozen nli_only policy is missing the {tier} threshold.")
        output_thresholds[tier] = float(threshold_info["threshold"])
    if output_thresholds["reliable"] <= output_thresholds["review"]:
        raise ValueError("Frozen reliable threshold must exceed the review threshold.")

    return model, output_thresholds, {
        "method": "nli_only",
        "features": features,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


def apply_frozen_nli_policy(
    records: Sequence[Mapping[str, object]],
    model,
    thresholds: Mapping[str, float],
) -> List[Dict[str, object]]:
    if not records:
        return []
    raw_scores = []
    for record in records:
        item_id = str(record.get("item_id", ""))
        score = float(record.get("nli_entailment_score", "nan"))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid NLI score for {item_id}: {score}")
        raw_scores.append(score)

    calibrated = model.predict_proba(np.asarray(raw_scores, dtype=float).reshape(-1, 1))[:, 1]
    output: List[Dict[str, object]] = []
    for record, confidence in zip(records, calibrated):
        row = dict(record)
        probability = float(confidence)
        row["calibrated_confidence"] = probability
        row["calibrated_uncertainty"] = 1.0 - probability
        row["calibrated_quality_label"] = assign_quality_tier(
            probability,
            bool(row.get("schema_valid")),
            thresholds,
        )
        row["confidence_model"] = "ontology_gate+nli_l2_calibration"
        output.append(row)
    return output


def build_reliable_payload(
    payload: Mapping[str, object],
    scored_records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    reliable: Dict[RelationKey, Mapping[str, object]] = {}
    for record in scored_records:
        if str(record.get("calibrated_quality_label", "")) != "reliable":
            continue
        key = relation_key(
            record.get("sample_id"),
            record.get("head"),
            record.get("relation_type") or record.get("relation"),
            record.get("tail"),
        )
        if key in reliable:
            raise ValueError(f"Duplicate reliable relation key: {key}")
        reliable[key] = record

    output = copy.deepcopy(dict(payload))
    output_samples = output.get("samples", []) or []
    for sample in output_samples:
        sample_id = str(sample.get("sample_id") or sample.get("filename") or "").replace(
            "\\", "/"
        )
        retained = []
        for relation in sample.get("relations", []) or []:
            key = relation_key(
                sample_id,
                relation.get("head"),
                relation.get("relation"),
                relation.get("tail"),
            )
            score = reliable.get(key)
            if score is None:
                continue
            enriched = dict(relation)
            enriched.update(
                {
                    "calibrated_confidence": float(score["calibrated_confidence"]),
                    "calibrated_quality_label": "reliable",
                    "confidence_model": str(score.get("confidence_model", "")),
                    "nli_entailment_score": float(score["nli_entailment_score"]),
                }
            )
            retained.append(enriched)
        sample["relations"] = retained

    output["confidence_policy"] = {
        "automatic_write_tier": "reliable",
        "review_and_reject_excluded_from_semantic_relations": True,
        "confidence_model": "ontology_gate+nli_l2_calibration",
    }
    return output


def _public_row(record: Mapping[str, object]) -> Dict[str, object]:
    evidence = record.get("_evidence", {})
    row = dict(evidence) if isinstance(evidence, Mapping) else {}
    for key, value in record.items():
        if key.startswith("_") or key == "relation_slot_score":
            continue
        row[key] = value
    row["relation"] = str(record.get("relation_type") or record.get("relation") or "")
    return row


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = [
        "item_id",
        "sample_id",
        "event_id",
        "filename",
        "text",
        "head",
        "relation",
        "tail",
        "schema_valid",
        "invalid_reason",
        "nli_entailment_score",
        "nli_contradiction_probability",
        "nli_neutral_probability",
        "nli_support_margin",
        "calibrated_confidence",
        "calibrated_uncertainty",
        "calibrated_quality_label",
        "confidence_model",
        "raw_kg_consistency_score",
        "kg_consistency_score",
        "kg_quality_label",
    ]
    available = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def publish_deployment_artifacts(
    output_dir: Path | str,
    payload: Mapping[str, object],
    scored_records: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    split_name: str = "test",
) -> None:
    split_name = validate_split_name(split_name)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        public_rows = [_public_row(record) for record in scored_records]
        fields = _fieldnames(public_rows)
        review_rows = [
            row
            for row in public_rows
            if row.get("calibrated_quality_label") == "review"
        ]
        reject_rows = [
            row
            for row in public_rows
            if row.get("calibrated_quality_label") == "reject"
        ]
        _write_csv(temporary / "all_relation_confidence.csv", public_rows, fields)
        _write_csv(temporary / "review_queue.csv", review_rows, fields)
        _write_csv(temporary / "reject_audit.csv", reject_rows, fields)

        reliable_payload = build_reliable_payload(payload, scored_records)
        (temporary / f"{split_name}_entities.json").write_text(
            json.dumps(reliable_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = dict(metadata)
        summary.update(
            {
                "split": split_name,
                "relations_scored": len(scored_records),
                "samples_scored": len(
                    {str(record.get("sample_id", "")) for record in scored_records}
                ),
                "quality_tier_counts": dict(
                    Counter(
                        str(record.get("calibrated_quality_label", ""))
                        for record in scored_records
                    )
                ),
                "relation_type_counts": dict(
                    Counter(str(record.get("relation_type", "")) for record in scored_records)
                ),
                "automatic_write_relations": sum(
                    record.get("calibrated_quality_label") == "reliable"
                    for record in scored_records
                ),
                "manual_review_relations": len(review_rows),
                "rejected_relations": len(reject_rows),
            }
        )
        (temporary / "deployment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Expected a JSON object with a samples list: {path}")
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError(f"Invalid split name: {split_name!r}")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score all schema-valid test relations with frozen multi-template NLI "
            "and export reliable/review/reject deployment artifacts."
        )
    )
    parser.add_argument("--samples-json", type=Path, default=DEFAULT_SAMPLES_JSON)
    parser.add_argument("--evidence-csv", type=Path, default=DEFAULT_EVIDENCE_CSV)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="")
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    parser.add_argument("--local-files-only", type=str_to_bool, default=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use 0 for all relations; positive values are for smoke tests only.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-name", default="test")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new --output-dir to preserve the existing result."
        )
    for path in (args.samples_json, args.evidence_csv, args.policy):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.max_length <= 0 or args.limit < 0:
        raise ValueError("batch-size/max-length must be positive and limit cannot be negative.")
    split_name = validate_split_name(args.split_name)

    payload = _read_json(args.samples_json)
    evidence_rows = _read_csv(args.evidence_csv)
    all_records = prepare_deployment_records(payload, evidence_rows)
    records = all_records[: args.limit] if args.limit > 0 else all_records
    if not records:
        raise ValueError("No relations are available for scoring.")
    calibration_model, thresholds, policy_metadata = load_frozen_nli_policy(args.policy)

    print(f"Relations validated: {len(all_records)}")
    print(f"Relations selected for scoring: {len(records)}")
    nli_runtime = score_records(
        records=records,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
        fp16=args.fp16,
        local_files_only=args.local_files_only,
        template_mode="ensemble",
    )
    scored_records = apply_frozen_nli_policy(records, calibration_model, thresholds)
    metadata = {
        "method": "ontology_gate + multi_template_nli + frozen_l2_calibration",
        "gold_labels_used_for_training_or_tuning": False,
        "source_relations": len(all_records),
        "limited_run": args.limit > 0,
        "thresholds": dict(thresholds),
        "policy": policy_metadata,
        "nli_runtime": nli_runtime,
        "input_paths": {
            "samples_json": str(args.samples_json.resolve()),
            "evidence_csv": str(args.evidence_csv.resolve()),
        },
        "input_sha256": {
            "samples_json": _sha256(args.samples_json),
            "evidence_csv": _sha256(args.evidence_csv),
        },
    }
    publish_deployment_artifacts(
        args.output_dir,
        payload,
        scored_records,
        metadata,
        split_name=split_name,
    )

    tiers = Counter(
        str(record["calibrated_quality_label"]) for record in scored_records
    )
    print("\n中文总结")
    print(f"怎么做：完成 {len(scored_records)} 条关系的多模板 NLI 与冻结校准。")
    print("原理：本体硬门控后，仅用独立 Gold 已验证的 NLI 主分数进行三级分流。")
    print(
        "能实现什么：reliable 自动入图，review 进入人工队列，reject 仅保留审计。"
    )
    print(
        "效果怎么样："
        f"reliable={tiers.get('reliable', 0)}, review={tiers.get('review', 0)}, "
        f"reject={tiers.get('reject', 0)}。"
    )
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
