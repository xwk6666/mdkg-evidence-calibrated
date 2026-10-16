#!/usr/bin/env python
"""Score human-reviewed disaster KG relations with a zero-shot NLI model."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from evaluate_reviewed_relation_evidence import (
    DEFAULT_COVERAGES,
    DEFAULT_REVIEW_CSV,
    DEFAULT_SLOT_CSV,
    paired_group_bootstrap_auc,
    quality_at_coverage,
    read_csv,
    relation_index,
    relation_key,
    score_summary,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "relation_nli_verifier"
_DLL_DIRECTORY_HANDLES = []


def find_conda_nvrtc_directory(prefix: Path) -> Optional[Path]:
    for directory in (prefix / "bin", prefix / "Library" / "bin", prefix):
        if directory.is_dir() and any(directory.glob("nvrtc-builtins*.dll")):
            return directory
    return None


def configure_windows_cuda_runtime() -> Optional[Path]:
    if os.name != "nt":
        return None
    dll_directory = find_conda_nvrtc_directory(Path(sys.prefix))
    if dll_directory is None:
        return None

    current_paths = os.environ.get("PATH", "").split(os.pathsep)
    normalized = {os.path.normcase(path) for path in current_paths if path}
    if os.path.normcase(str(dll_directory)) not in normalized:
        os.environ["PATH"] = str(dll_directory) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_directory)))
    return dll_directory


def str_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_relation_hypothesis(head: str, relation: str, tail: str) -> str:
    relation = relation.strip().upper()
    if relation == "OCCURS_IN":
        return f"{head} occurred in {tail}."
    if relation == "AFFECTS":
        return f"{head} affected or damaged {tail}."
    if relation == "CAUSES":
        return f"{head} caused {tail}."
    raise ValueError(f"Unsupported disaster relation: {relation}")


def build_relation_hypotheses(head: str, relation: str, tail: str) -> List[str]:
    relation = relation.strip().upper()
    if relation == "OCCURS_IN":
        return [
            f"{head} occurred in {tail}.",
            f"{head} took place in {tail}.",
            f"{tail} was the location of {head}.",
        ]
    if relation == "AFFECTS":
        return [
            f"{head} affected {tail}.",
            f"{head} damaged {tail}.",
            f"{tail} was affected by {head}.",
            f"{tail} was damaged by {head}.",
        ]
    if relation == "CAUSES":
        return [
            f"{head} caused {tail}.",
            f"{head} resulted in {tail}.",
            f"{tail} resulted from {head}.",
        ]
    raise ValueError(f"Unsupported disaster relation: {relation}")


def select_best_hypothesis_result(
    hypotheses: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
) -> Dict[str, object]:
    if not hypotheses or len(hypotheses) != len(probabilities):
        raise ValueError("Hypotheses and probabilities must have the same non-zero length.")
    best_index = max(
        range(len(hypotheses)),
        key=lambda index: float(probabilities[index]["entailment"]),
    )
    selected = probabilities[best_index]
    return {
        "selected_hypothesis": hypotheses[best_index],
        "template_count": len(hypotheses),
        "contradiction": float(selected["contradiction"]),
        "entailment": float(selected["entailment"]),
        "neutral": float(selected["neutral"]),
    }


def resolve_nli_label_indices(id2label: Mapping[object, object]) -> Dict[str, int]:
    indices = {
        str(label).strip().lower(): int(index)
        for index, label in id2label.items()
    }
    required = {"contradiction", "entailment", "neutral"}
    if set(indices) != required:
        raise ValueError(
            "The NLI model must expose contradiction, entailment and neutral labels; "
            f"received {indices}."
        )
    return indices


def geometric_evidence_fusion(nli_score: float, slot_score: float) -> float:
    if not 0.0 <= nli_score <= 1.0 or not 0.0 <= slot_score <= 1.0:
        raise ValueError("Evidence scores must be in [0, 1].")
    return math.sqrt(nli_score * slot_score)


def prepare_records(
    review_rows: Sequence[Mapping[str, object]],
    slot_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    slots = relation_index(slot_rows)
    records: List[Dict[str, object]] = []
    missing = []
    for review in review_rows:
        key = relation_key(review, reviewed=True)
        slot = slots.get(key)
        if slot is None:
            missing.append(key)
            continue
        head = str(review.get("pred_head", "")).strip()
        relation = str(review.get("pred_relation", "")).strip().upper()
        tail = str(review.get("pred_tail", "")).strip()
        records.append(
            {
                "item_id": str(review.get("item_id", "")),
                "sample_id": key[0],
                "event_id": str(review.get("event_id", "")),
                "event_family": str(review.get("event_id", "")).split("__", 1)[0],
                "text": str(review.get("text", "")),
                "head": head,
                "relation_type": relation,
                "tail": tail,
                "hypothesis": build_relation_hypothesis(head, relation, tail),
                "label": int(float(str(review.get("correct_1_or_0", 0)))),
                "relation_slot_score": float(str(slot.get("relation_slot_score", 0))),
            }
        )
    if missing:
        raise ValueError(f"Missing event-slot scores for {len(missing)} reviewed relations.")
    return records


def attach_reference_scores(
    records: List[Dict[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    output_field: str,
    score_field: str = "nli_entailment_score",
) -> None:
    reference = {
        str(row.get("item_id", "")): float(str(row[score_field]))
        for row in reference_rows
    }
    missing = []
    for record in records:
        item_id = str(record.get("item_id", ""))
        if item_id not in reference:
            missing.append(item_id)
            continue
        record[output_field] = reference[item_id]
    if missing:
        raise ValueError(f"Reference NLI scores are missing {len(missing)} item IDs.")


def score_records(
    records: List[Dict[str, object]],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
    fp16: bool,
    local_files_only: bool,
    template_mode: str,
) -> Dict[str, object]:
    cuda_dll_directory = configure_windows_cuda_runtime()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).to(device)
    if fp16 and device.type == "cuda":
        model.half()
    model.eval()
    label_indices = resolve_nli_label_indices(model.config.id2label)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        premises: List[str] = []
        hypotheses: List[str] = []
        record_hypotheses: List[List[str]] = []
        spans = []
        for record in batch:
            current_hypotheses = (
                [str(record["hypothesis"])]
                if template_mode == "single"
                else build_relation_hypotheses(
                    str(record["head"]),
                    str(record["relation_type"]),
                    str(record["tail"]),
                )
            )
            span_start = len(hypotheses)
            hypotheses.extend(current_hypotheses)
            premises.extend([str(record["text"])] * len(current_hypotheses))
            spans.append((span_start, len(hypotheses)))
            record_hypotheses.append(current_hypotheses)
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu()
        for record, current_hypotheses, (span_start, span_end) in zip(
            batch,
            record_hypotheses,
            spans,
        ):
            template_probabilities = []
            for probability in probabilities[span_start:span_end]:
                template_probabilities.append(
                    {
                        "contradiction": float(
                            probability[label_indices["contradiction"]].item()
                        ),
                        "entailment": float(probability[label_indices["entailment"]].item()),
                        "neutral": float(probability[label_indices["neutral"]].item()),
                    }
                )
            selected = select_best_hypothesis_result(
                current_hypotheses,
                template_probabilities,
            )
            contradiction = float(selected["contradiction"])
            entailment = float(selected["entailment"])
            neutral = float(selected["neutral"])
            record["nli_template_mode"] = template_mode
            record["nli_template_count"] = int(selected["template_count"])
            record["nli_selected_hypothesis"] = str(selected["selected_hypothesis"])
            record["nli_template_scores_json"] = json.dumps(
                [
                    {"hypothesis": hypothesis, **scores}
                    for hypothesis, scores in zip(current_hypotheses, template_probabilities)
                ],
                ensure_ascii=False,
            )
            record["nli_contradiction_probability"] = contradiction
            record["nli_entailment_score"] = entailment
            record["nli_neutral_probability"] = neutral
            record["nli_support_margin"] = entailment - max(contradiction, neutral)
            record["nli_slot_geomean"] = geometric_evidence_fusion(
                entailment,
                float(record["relation_slot_score"]),
            )
        completed = min(start + batch_size, len(records))
        if completed == len(records) or completed % max(32, batch_size) == 0:
            print(f"Scored {completed}/{len(records)} reviewed relations")

    return {
        "device": str(device),
        "model_name": model_name,
        "fp16": bool(fp16 and device.type == "cuda"),
        "label_indices": label_indices,
        "cuda_dll_directory": str(cuda_dll_directory) if cuda_dll_directory else None,
        "template_mode": template_mode,
    }


def summarize_by_relation_type(
    records: Sequence[Mapping[str, object]],
    methods: Sequence[tuple],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    relation_types = sorted({str(record["relation_type"]) for record in records})
    for relation_type in relation_types:
        subset = [
            record for record in records if str(record["relation_type"]) == relation_type
        ]
        for method, score_field in methods:
            summary = score_summary(subset, score_field)
            half_coverage = quality_at_coverage(subset, score_field, 50.0)
            rows.append(
                {
                    "relation_type": relation_type,
                    "method": method,
                    "score_field": score_field,
                    "relations": summary["eligible_relations"],
                    "correct_relations": summary["correct_relations"],
                    "incorrect_relations": summary["incorrect_relations"],
                    "auc": summary["auc"],
                    "average_precision": summary["average_precision"],
                    "precision_at_50pct": half_coverage["precision"],
                    "errors_at_50pct": half_coverage["retained_errors"],
                }
            )
    return rows


def evaluate_scores(
    records: Sequence[Mapping[str, object]],
    coverages: Sequence[float],
    bootstrap_repetitions: int,
    seed: int,
    template_mode: str,
) -> Dict[str, object]:
    nli_method_name = "Zero-shot NLI" if template_mode == "single" else "Multi-template zero-shot NLI"
    methods = [
        ("Event-slot prototype", "relation_slot_score"),
        (nli_method_name, "nli_entailment_score"),
        ("NLI + event-slot geometric fusion", "nli_slot_geomean"),
    ]
    has_reference = bool(records) and all(
        record.get("single_template_nli_score") is not None for record in records
    )
    if has_reference:
        methods.insert(1, ("Single-template zero-shot NLI", "single_template_nli_score"))
    summaries = []
    coverage_rows = []
    for method, score_field in methods:
        summaries.append({"method": method, **score_summary(records, score_field)})
        for coverage in coverages:
            coverage_rows.append(
                {
                    "method": method,
                    "score_field": score_field,
                    **quality_at_coverage(records, score_field, coverage),
                }
            )
    bootstrap = [
        paired_group_bootstrap_auc(
            records,
            better_field="nli_entailment_score",
            reference_field="relation_slot_score",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        paired_group_bootstrap_auc(
            records,
            better_field="nli_slot_geomean",
            reference_field="relation_slot_score",
            repetitions=bootstrap_repetitions,
            seed=seed + 1,
        ),
        paired_group_bootstrap_auc(
            records,
            better_field="nli_slot_geomean",
            reference_field="nli_entailment_score",
            repetitions=bootstrap_repetitions,
            seed=seed + 2,
        ),
    ]
    if has_reference:
        bootstrap.append(
            paired_group_bootstrap_auc(
                records,
                better_field="nli_entailment_score",
                reference_field="single_template_nli_score",
                repetitions=bootstrap_repetitions,
                seed=seed + 3,
            )
        )
    relation_type_rows = summarize_by_relation_type(records, methods)
    return {
        "score_summaries": summaries,
        "coverage_rows": coverage_rows,
        "bootstrap": bootstrap,
        "relation_type_rows": relation_type_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a zero-shot NLI verifier to 299 reviewed disaster KG relations."
    )
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--slot-csv", type=Path, default=DEFAULT_SLOT_CSV)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="")
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    parser.add_argument("--local-files-only", type=str_to_bool, default=False)
    parser.add_argument("--template-mode", choices=["single", "ensemble"], default="single")
    parser.add_argument("--reference-score-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--coverage", nargs="+", type=float, default=list(DEFAULT_COVERAGES))
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = prepare_records(read_csv(args.review_csv), read_csv(args.slot_csv))
    if args.limit > 0:
        records = records[: args.limit]
    if args.reference_score_csv:
        attach_reference_scores(
            records,
            read_csv(args.reference_score_csv),
            output_field="single_template_nli_score",
        )

    runtime = score_records(
        records=records,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
        fp16=args.fp16,
        local_files_only=args.local_files_only,
        template_mode=args.template_mode,
    )
    write_csv(args.output_dir / "reviewed_relation_nli_scores.csv", records)

    labels = {int(record["label"]) for record in records}
    if labels != {0, 1}:
        print("Skipping aggregate evaluation because the selected records do not contain both labels.")
        return

    evaluation = evaluate_scores(
        records,
        coverages=args.coverage,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        template_mode=args.template_mode,
    )
    write_csv(args.output_dir / "relation_nli_score_summary.csv", evaluation["score_summaries"])
    write_csv(args.output_dir / "relation_nli_quality_coverage.csv", evaluation["coverage_rows"])
    write_csv(args.output_dir / "relation_nli_bootstrap.csv", evaluation["bootstrap"])
    write_csv(
        args.output_dir / "relation_nli_by_relation_type.csv",
        evaluation["relation_type_rows"],
    )

    summary = {
        "reviewed_relations": len(records),
        "reviewed_samples": len({str(record["sample_id"]) for record in records}),
        "correct_relations": sum(int(record["label"]) for record in records),
        "incorrect_relations": sum(1 - int(record["label"]) for record in records),
        "human_labels_used_for_training": False,
        "template_mode": args.template_mode,
        "fusion": "sqrt(nli_entailment_score * relation_slot_score)",
        "runtime": runtime,
        **evaluation,
    }
    (args.output_dir / "relation_nli_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for row in evaluation["score_summaries"]:
        print(
            f"{row['method']}: AUC={row['auc']:.4f}, "
            f"AP={row['average_precision']:.4f}, n={row['eligible_relations']}"
        )
    for result in evaluation["bootstrap"]:
        print(
            f"{result['better_field']} - {result['reference_field']}: "
            f"delta={result['observed_delta']:.4f}, "
            f"95% CI [{result['ci95_low']:.4f}, {result['ci95_high']:.4f}], "
            f"p={result['two_sided_p']:.4f}"
        )
    print(f"Saved zero-shot relation verifier outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
