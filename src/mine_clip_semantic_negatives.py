import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from clip_semantic_negatives import (
    infer_disaster_type,
    infer_event_family,
    mine_topk_from_features,
    write_semantic_negative_cache,
)
from train_clip_finetune import (
    CrisisMMDClipDataset,
    collate_clip_batch,
    unwrap_clip_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSON = (
    PROJECT_ROOT / "data" / "CrisisMMD_v2.0" / "structured" / "train_entities.json"
)
DEFAULT_IMG_DIR = PROJECT_ROOT / "data" / "CrisisMMD_v2.0"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clip_finetuned_crisismmd_best"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "clip_semantic_negatives"
    / "train_top5_semantic_negatives.jsonl"
)


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_model_name(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.is_absolute():
        return str(candidate)
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return str(project_candidate)
    models_candidate = PROJECT_ROOT / "models" / candidate
    if models_candidate.exists():
        return str(models_candidate)
    return model_name


@torch.no_grad()
def encode_dataset_features(
    model,
    processor,
    dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch,
    )
    image_features = []
    text_features = []

    model.eval()
    for batch_index, batch in enumerate(loader):
        image_inputs = processor(
            images=batch["images"],
            return_tensors="pt",
        )
        text_inputs = processor(
            text=batch["texts"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        pixel_values = image_inputs["pixel_values"].to(device)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        batch_image_features = model.get_image_features(pixel_values=pixel_values)
        batch_text_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        batch_image_features = unwrap_clip_features(
            batch_image_features,
            model.visual_projection,
        )
        batch_text_features = unwrap_clip_features(
            batch_text_features,
            model.text_projection,
        )
        image_features.append(F.normalize(batch_image_features, p=2, dim=-1).cpu())
        text_features.append(F.normalize(batch_text_features, p=2, dim=-1).cpu())

        processed = min((batch_index + 1) * batch_size, len(dataset))
        if batch_index == 0 or processed == len(dataset) or processed % (batch_size * 20) == 0:
            print(f"Encoded {processed}/{len(dataset)} samples")

    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def build_mining_summary(
    samples: Sequence[Dict[str, object]],
    records: Sequence[Dict[str, object]],
    model_name: str,
    top_k: int,
    elapsed_seconds: float,
) -> Dict[str, object]:
    candidate_counts = [len(record["negative_sample_ids"]) for record in records]
    top1_scores = [
        float(record["negative_scores"][0])
        for record in records
        if record["negative_scores"]
    ]
    type_counts = Counter(infer_disaster_type(sample) for sample in samples)
    family_counts = Counter(
        infer_event_family(str(sample.get("event_id", ""))) for sample in samples
    )
    with_candidates = sum(count > 0 for count in candidate_counts)

    return {
        "task": "clip_semantic_hard_negative_mining",
        "model_name": model_name,
        "top_k": top_k,
        "num_samples": len(samples),
        "num_samples_with_candidates": with_candidates,
        "candidate_coverage": with_candidates / len(samples) if samples else 0.0,
        "mean_candidates_per_sample": (
            float(np.mean(candidate_counts)) if candidate_counts else 0.0
        ),
        "top1_score_mean": float(np.mean(top1_scores)) if top1_scores else None,
        "top1_score_std": float(np.std(top1_scores)) if top1_scores else None,
        "top1_score_min": float(np.min(top1_scores)) if top1_scores else None,
        "top1_score_max": float(np.max(top1_scores)) if top1_scores else None,
        "disaster_type_counts": dict(sorted(type_counts.items())),
        "event_family_counts": dict(sorted(family_counts.items())),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine same-disaster, different-event-family Top-K CLIP negatives "
            "from the training split."
        )
    )
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--img-dir", default=str(DEFAULT_IMG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("--batch-size and --chunk-size must be at least 1.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_name = resolve_model_name(args.model_name)
    train_json = resolve_project_path(args.train_json)
    img_dir = resolve_project_path(args.img_dir)
    output_path = resolve_project_path(args.output)
    summary_path = (
        resolve_project_path(args.summary_output)
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )

    print(f"Using device: {device.type.upper()}")
    print(f"Mining model: {model_name}")
    print("1. Loading training samples...")
    dataset = CrisisMMDClipDataset(str(train_json), str(img_dir))
    if args.max_samples is not None:
        dataset.data = dataset.data[: args.max_samples]
    if not dataset.data:
        raise ValueError("No training samples with readable images were found.")
    print(f"Mining samples: {len(dataset)}")

    print("2. Loading CLIP...")
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    start = time.perf_counter()
    print("3. Independently encoding image and text features...")
    image_features, text_features = encode_dataset_features(
        model,
        processor,
        dataset,
        batch_size=args.batch_size,
        device=device,
    )

    print("4. Mining semantic Top-K candidates...")
    records = mine_topk_from_features(
        dataset.data,
        image_features,
        text_features,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        device=device,
    )
    elapsed_seconds = time.perf_counter() - start

    write_semantic_negative_cache(records, output_path)
    summary = build_mining_summary(
        dataset.data,
        records,
        model_name=model_name,
        top_k=args.top_k,
        elapsed_seconds=elapsed_seconds,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Candidate coverage: {summary['candidate_coverage']:.2%}")
    print(f"Mean candidates/sample: {summary['mean_candidates_per_sample']:.2f}")
    print(f"Saved cache: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Elapsed: {summary['elapsed_minutes']:.2f} min")


if __name__ == "__main__":
    main()
