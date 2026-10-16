import argparse
import math
import csv
import json
import re
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, Mapping


OUTPUT_FIELDS = [
    "sample_id",
    "event_id",
    "filename",
    "text",
    "alignment_score",
    "raw_alignment_score",
    "calibrated_probability",
    "decision_threshold",
    "evidence_label",
    "confidence",
    "i2t_rank",
    "t2i_rank",
    "model_name",
    "calibration_model",
    "score_type",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRUCTURED_JSON = REPO_ROOT / "outputs" / "final_relation_nli_deployment" / "test_entities.json"
DEFAULT_IMG_DIR = REPO_ROOT / "data" / "CrisisMMD_v2.0"
DEFAULT_MODEL_NAME = "models/clip_dce_semantic_top5_best"
DEFAULT_CALIBRATION_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "clip_calibration"
    / "semantic_top5_clip_fixed_hard_calibrated_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "final_clip_alignment_scores"


def apply_platt_calibration(
    raw_score: float,
    weight: float,
    bias: float,
    score_mean: float,
    score_std: float,
) -> float:
    if float(score_std) <= 0:
        raise ValueError("score_std must be positive")

    standardized = (float(raw_score) - float(score_mean)) / float(score_std)
    logit = float(weight) * standardized + float(bias)
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def evidence_label(probability: float, threshold: float) -> str:
    return "consistent" if float(probability) >= float(threshold) else "inconsistent"


def parse_calibration_payload(payload: Mapping[str, object]) -> Dict[str, float]:
    calibrator = payload.get("calibrator")
    validation = payload.get("validation")
    if not isinstance(calibrator, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("Calibration summary must contain calibrator and validation objects")
    if calibrator.get("type") != "platt_logistic":
        raise ValueError("Only platt_logistic calibration is supported")

    required = ("weight", "bias", "score_mean", "score_std")
    missing = [name for name in required if calibrator.get(name) is None]
    if missing:
        raise ValueError(f"Calibration summary is missing fields: {', '.join(missing)}")

    threshold = validation.get("probability_threshold")
    if threshold is None:
        threshold = validation.get("threshold")
    if threshold is None:
        raise ValueError("Calibration summary is missing probability_threshold")

    config = {
        "type": "platt_logistic",
        "weight": float(calibrator["weight"]),
        "bias": float(calibrator["bias"]),
        "score_mean": float(calibrator["score_mean"]),
        "score_std": float(calibrator["score_std"]),
        "threshold": float(threshold),
    }
    if config["score_std"] <= 0:
        raise ValueError("score_std must be positive")
    if not 0.0 <= config["threshold"] <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    return config


def normalize_sample_id(sample: Mapping[str, object]) -> str:
    sample_id = str(sample.get("sample_id") or sample.get("filename") or "")
    sample_id = sample_id.replace("\\", "/")
    lower = sample_id.lower()
    for suffix in (".jpeg", ".jpg", ".png"):
        if lower.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def build_output_row(
    sample: Mapping[str, object],
    raw_score: float,
    calibrated_probability: float,
    threshold: float,
    i2t_rank: int,
    t2i_rank: int,
    model_name: str,
    calibration_name: str,
) -> Dict[str, object]:
    probability = float(calibrated_probability)
    return {
        "sample_id": normalize_sample_id(sample),
        "event_id": str(sample.get("event_id") or ""),
        "filename": str(sample.get("filename") or ""),
        "text": str(sample.get("text") or ""),
        "alignment_score": probability,
        "raw_alignment_score": float(raw_score),
        "calibrated_probability": probability,
        "decision_threshold": float(threshold),
        "evidence_label": evidence_label(probability, threshold),
        "confidence": evidence_label(probability, threshold),
        "i2t_rank": int(i2t_rank),
        "t2i_rank": int(t2i_rank),
        "model_name": str(model_name),
        "calibration_model": str(calibration_name),
        "score_type": "platt_calibrated_probability",
    }


def compute_pair_ranks(image_features, text_features):
    import torch
    import torch.nn.functional as F

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must have shape [N, D]")
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have the same shape")

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    similarities = image_features @ text_features.T
    ground_truth = torch.arange(similarities.size(0)).unsqueeze(1)
    i2t_order = torch.argsort(similarities, dim=1, descending=True)
    t2i_order = torch.argsort(similarities.T, dim=1, descending=True)
    i2t_ranks = torch.argmax((i2t_order == ground_truth).to(torch.int64), dim=1) + 1
    t2i_ranks = torch.argmax((t2i_order == ground_truth).to(torch.int64), dim=1) + 1
    return i2t_ranks.tolist(), t2i_ranks.tolist()


def compute_pair_ranks_blockwise(image_features, text_features, block_size: int = 256):
    """Compute exact one-based ranks without materializing an N x N matrix."""
    import torch.nn.functional as F

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must have shape [N, D]")
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have the same shape")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    pair_scores = (image_features * text_features).sum(dim=1)
    i2t_ranks = []
    t2i_ranks = []
    sample_count = image_features.size(0)

    for start in range(0, sample_count, block_size):
        end = min(start + block_size, sample_count)
        i2t_scores = image_features[start:end] @ text_features.T
        i2t_ranks.extend(
            (i2t_scores > pair_scores[start:end, None]).sum(dim=1).add(1).tolist()
        )

        t2i_scores = text_features[start:end] @ image_features.T
        t2i_ranks.extend(
            (t2i_scores > pair_scores[start:end, None]).sum(dim=1).add(1).tolist()
        )

    return [int(rank) for rank in i2t_ranks], [int(rank) for rank in t2i_ranks]


def cosine_diagonal_scores(image_features, text_features):
    import torch.nn.functional as F

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must have shape [N, D]")
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have the same shape")
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    return (image_features * text_features).sum(dim=1).tolist()


def prepare_missing_rows(missing_images):
    rows = []
    for item in missing_images:
        rows.append(
            {
                "sample_id": normalize_sample_id(item),
                "filename": str(item.get("filename") or ""),
                "reason": "missing_image",
            }
        )
    return rows


def build_scored_rows(
    samples,
    raw_scores,
    i2t_ranks,
    t2i_ranks,
    calibration,
    model_name: str,
    calibration_name: str,
):
    lengths = {len(samples), len(raw_scores), len(i2t_ranks), len(t2i_ranks)}
    if len(lengths) != 1:
        raise ValueError("samples, scores, and ranks must have the same length")

    rows = []
    for sample, raw_score, i2t_rank, t2i_rank in zip(
        samples,
        raw_scores,
        i2t_ranks,
        t2i_ranks,
    ):
        probability = apply_platt_calibration(
            raw_score=raw_score,
            weight=calibration["weight"],
            bias=calibration["bias"],
            score_mean=calibration["score_mean"],
            score_std=calibration["score_std"],
        )
        rows.append(
            build_output_row(
                sample=sample,
                raw_score=raw_score,
                calibrated_probability=probability,
                threshold=calibration["threshold"],
                i2t_rank=i2t_rank,
                t2i_rank=t2i_rank,
                model_name=model_name,
                calibration_name=calibration_name,
            )
        )
    return rows


def build_summary(
    rows,
    source_samples: int,
    missing_images: int,
    model_name: str,
    calibration_path: str,
    calibration,
    structured_json: str,
    elapsed_seconds: float,
    split_name: str = "test",
):
    raw_scores = [float(row["raw_alignment_score"]) for row in rows]
    probabilities = [float(row["calibrated_probability"]) for row in rows]

    def describe(values, prefix):
        if not values:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
        return {
            f"{prefix}_mean": fmean(values),
            f"{prefix}_std": pstdev(values),
            f"{prefix}_min": min(values),
            f"{prefix}_max": max(values),
        }

    summary = {
        "task": "sample_level_clip_evidence_export",
        "split": split_name,
        "structured_json": str(structured_json),
        "model_name": str(model_name),
        "calibration_summary": str(calibration_path),
        "calibration_type": "platt_logistic",
        "decision_threshold": float(calibration["threshold"]),
        "source_samples": int(source_samples),
        "scored_samples": len(rows),
        "missing_images": int(missing_images),
        "consistent_pairs": sum(row["evidence_label"] == "consistent" for row in rows),
        "inconsistent_pairs": sum(row["evidence_label"] == "inconsistent" for row in rows),
        "elapsed_seconds": float(elapsed_seconds),
    }
    summary.update(describe(raw_scores, "raw_score"))
    summary.update(describe(probabilities, "calibrated_probability"))
    return summary


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError(f"Invalid split name: {split_name!r}")
    return normalized


def write_outputs(rows, missing_rows, summary, output_dir: Path, split_name: str = "test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = validate_split_name(split_name)
    paths = {
        "csv": output_dir / f"{split_name}_alignment_scores.csv",
        "json": output_dir / f"{split_name}_alignment_scores.json",
        "missing": output_dir / f"{split_name}_skipped_missing_images.csv",
        "summary": output_dir / f"{split_name}_alignment_summary.json",
    }

    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump({"samples": list(rows)}, handle, ensure_ascii=False, indent=2)

    with paths["missing"].open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["sample_id", "filename", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export sample-level Sem-HN CLIP evidence with a validation-frozen "
            "Platt calibrator."
        )
    )
    parser.add_argument("--structured-json", default=str(DEFAULT_STRUCTURED_JSON))
    parser.add_argument("--img-dir", default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--calibration-summary", default=str(DEFAULT_CALIBRATION_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--rank-block-size",
        type=int,
        default=256,
        help="Rows per exact-rank block; avoids an N x N memory spike on the train split.",
    )
    return parser.parse_args(argv)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv=None):
    import torch

    from evaluate_clip_image_text_consistency import (
        ClipStructuredDataset,
        encode_dataset_features,
        load_clip_model_and_processor,
        resolve_clip_model_name,
    )

    args = parse_args(argv)
    started = time.perf_counter()
    structured_json = resolve_path(args.structured_json)
    img_dir = resolve_path(args.img_dir)
    calibration_path = resolve_path(args.calibration_summary)
    output_dir = resolve_path(args.output_dir)
    model_path = resolve_clip_model_name(args.model_name, REPO_ROOT)

    if not structured_json.exists():
        raise FileNotFoundError(f"Structured JSON not found: {structured_json}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration summary not found: {calibration_path}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.rank_block_size <= 0:
        raise ValueError("rank_block_size must be positive")
    split_name = validate_split_name(args.split_name)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    with structured_json.open("r", encoding="utf-8") as handle:
        source_payload = json.load(handle)
    source_samples = source_payload.get("samples")
    if not isinstance(source_samples, list):
        raise ValueError("Structured JSON must contain a samples list")

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = parse_calibration_payload(json.load(handle))

    print(f"Using device: {device.type.upper()}")
    print(f"CLIP model: {model_path}")
    print(f"Frozen calibration: {calibration_path}")
    print(f"1. Loading and auditing {split_name} samples...")
    dataset = ClipStructuredDataset(structured_json, img_dir, skip_missing_images=True)
    missing_rows = prepare_missing_rows(dataset.missing_images)
    if len(dataset) + len(missing_rows) != len(source_samples):
        raise RuntimeError("Scored and missing-image sample counts do not cover the source payload")
    if not dataset.data:
        raise RuntimeError("No samples with available images were found")
    print(
        f"Source samples: {len(source_samples)} | Valid images: {len(dataset)} | "
        f"Missing images: {len(missing_rows)}"
    )

    print("2. Loading frozen Sem-HN CLIP checkpoint...")
    model, processor = load_clip_model_and_processor(model_path, device)

    print("3. Independently encoding images and texts...")
    image_features, text_features = encode_dataset_features(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    raw_scores = cosine_diagonal_scores(image_features, text_features)

    print("4. Computing retrieval ranks and frozen calibrated evidence...")
    i2t_ranks, t2i_ranks = compute_pair_ranks_blockwise(
        image_features,
        text_features,
        block_size=args.rank_block_size,
    )
    rows = build_scored_rows(
        samples=dataset.data,
        raw_scores=raw_scores,
        i2t_ranks=i2t_ranks,
        t2i_ranks=t2i_ranks,
        calibration=calibration,
        model_name=args.model_name,
        calibration_name=calibration_path.name,
    )
    elapsed = time.perf_counter() - started
    summary = build_summary(
        rows=rows,
        source_samples=len(source_samples),
        missing_images=len(missing_rows),
        model_name=args.model_name,
        calibration_path=str(calibration_path),
        calibration=calibration,
        structured_json=str(structured_json),
        elapsed_seconds=elapsed,
        split_name=split_name,
    )

    print("5. Writing evidence and audit artifacts...")
    paths = write_outputs(
        rows, missing_rows, summary, output_dir, split_name=split_name
    )
    print(
        "Chinese summary: "
        f"scored={summary['scored_samples']}, missing={summary['missing_images']}, "
        f"consistent={summary['consistent_pairs']}, inconsistent={summary['inconsistent_pairs']}"
    )
    print(f"Saved scores: {paths['csv']}")
    print(f"Saved summary: {paths['summary']}")


if __name__ == "__main__":
    main()
