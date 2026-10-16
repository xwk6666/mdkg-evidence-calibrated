import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from fixed_semantic_hard_testset import (
    load_fixed_pair_manifest,
    load_fixed_pairs,
)
from evaluate_image_text_consistency import (
    DEFAULT_IMG_DIR,
    DEFAULT_TEST_JSON,
    DEFAULT_VAL_JSON,
    PairExample,
    build_pair_indices,
    choose_best_threshold,
    compute_binary_metrics,
    compute_metrics_by_kind,
    count_pair_kinds,
    print_grouped_metrics,
    print_metrics,
    resolve_repo_path,
    str_to_bool,
)


DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_DIR = "outputs/clip_image_text_consistency"


class ClipStructuredDataset(Dataset):
    """Lightweight CrisisMMD dataset for CLIP pair-consistency scoring."""

    def __init__(self, structured_json_path: Path, img_dir: Path, skip_missing_images: bool = True):
        self.structured_json_path = str(structured_json_path)
        self.img_dir = Path(img_dir)
        self.missing_images: List[Dict[str, str]] = []

        with structured_json_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)["samples"]
        self.data = self._filter_and_resolve_images(raw_data, skip_missing_images=skip_missing_images)

        event_ids = sorted({item.get("event_id", "unknown_event") for item in self.data})
        self.event_to_id = {event_id: idx for idx, event_id in enumerate(event_ids)}

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, filename: str) -> Optional[Path]:
        image_path = self.img_dir / filename
        if image_path.exists():
            return image_path

        suffixes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        stem = image_path.with_suffix("")
        for suffix in suffixes:
            candidate = stem.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return None

    def _filter_and_resolve_images(self, samples: List[Dict[str, object]], skip_missing_images: bool) -> List[Dict[str, object]]:
        kept = []
        for item in samples:
            filename = str(item.get("filename", ""))
            resolved_path = self._resolve_image_path(filename)
            if resolved_path is None:
                self.missing_images.append(
                    {
                        "filename": filename,
                        "sample_id": str(item.get("sample_id") or filename),
                    }
                )
                if skip_missing_images:
                    continue
            new_item = dict(item)
            if resolved_path is not None:
                new_item["_resolved_image_path"] = str(resolved_path)
            kept.append(new_item)

        if self.missing_images and skip_missing_images:
            print(
                f"Skipped {len(self.missing_images)} samples with missing images "
                f"from {self.structured_json_path}. Kept {len(kept)}/{len(samples)} samples."
            )
        return kept

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.data[index]
        image_path = Path(item.get("_resolved_image_path") or (self.img_dir / str(item["filename"])))
        return {
            "image": Image.open(image_path).convert("RGB"),
            "text": str(item.get("text", "")),
            "filename": str(item.get("filename", "")),
            "event_id": str(item.get("event_id", "unknown_event")),
            "sample_id": str(item.get("sample_id", "")),
        }


class ClipPairDataset(Dataset):
    def __init__(self, base_dataset: ClipStructuredDataset, pairs: Sequence[PairExample]):
        self.base_dataset = base_dataset
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        pair = self.pairs[index]
        image_item = self.base_dataset[pair.image_index]
        text_item = self.base_dataset.data[pair.text_index]
        return {
            "image": image_item["image"],
            "text": str(text_item.get("text", "")),
            "label": int(pair.label),
            "kind": pair.kind,
            "pair": pair,
        }


def collate_clip_pairs(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "images": [item["image"] for item in batch],
        "texts": [item["text"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "kinds": [item["kind"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }


def collate_clip_samples(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "images": [item["image"] for item in batch],
        "texts": [item["text"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "event_ids": [item["event_id"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }


def dataset_event_labels(dataset: ClipStructuredDataset) -> List[int]:
    labels = []
    for item in dataset.data:
        event_id = item.get("event_id", "unknown_event")
        labels.append(int(dataset.event_to_id.get(event_id, 0)))
    return labels


def load_pair_manifest_for_dataset(
    manifest: Dict[str, object],
    dataset: ClipStructuredDataset,
) -> List[PairExample]:
    return load_fixed_pairs(manifest, dataset.data)


def cosine_scores_from_features(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    if image_features.shape != text_features.shape:
        raise ValueError(
            "image_features and text_features must have the same shape, "
            f"got {tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    if image_features.dim() != 2:
        raise ValueError(f"features must have shape [batch, dim], got {tuple(image_features.shape)}.")
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    return (image_features * text_features).sum(dim=-1)


def scores_from_feature_cache(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pairs: Sequence[PairExample],
) -> List[float]:
    scores: List[float] = []
    for pair in pairs:
        score = cosine_scores_from_features(
            image_features[pair.image_index : pair.image_index + 1],
            text_features[pair.text_index : pair.text_index + 1],
        )
        scores.append(float(score.item()))
    return scores


def resolve_clip_model_name(model_name: str, repo_root: Path) -> str:
    path = Path(model_name)
    if path.is_absolute():
        return str(path)
    if model_name.startswith("models/") or model_name.startswith("models\\"):
        return str(repo_root / path)
    candidate = repo_root / "models" / model_name
    if candidate.exists():
        return str(candidate)
    return model_name


def find_cached_clip_components(
    model_name: str,
    cache_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    if "/" not in model_name:
        raise FileNotFoundError(
            f"Cannot derive a Hugging Face cache directory from: {model_name}"
        )
    cache_root = Path(cache_root or HF_HUB_CACHE)
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"No local Hugging Face cache was found for {model_name}."
        )

    config_dir = None
    weights_dir = None
    for snapshot in sorted(snapshots_dir.iterdir()):
        if not snapshot.is_dir():
            continue
        if (
            config_dir is None
            and (snapshot / "config.json").exists()
            and (snapshot / "preprocessor_config.json").exists()
        ):
            config_dir = snapshot
        if weights_dir is None and (snapshot / "model.safetensors").exists():
            weights_dir = snapshot

    if config_dir is None or weights_dir is None:
        raise FileNotFoundError(
            f"Local cache for {model_name} does not contain both configuration "
            "files and model.safetensors."
        )
    return config_dir, weights_dir


def load_clip_model_and_processor(
    model_name_or_path: str,
    device: torch.device,
) -> Tuple[CLIPModel, CLIPProcessor]:
    local_path = Path(model_name_or_path)
    if local_path.exists():
        model = CLIPModel.from_pretrained(
            str(local_path),
            use_safetensors=True,
        ).to(device)
        processor = CLIPProcessor.from_pretrained(str(local_path))
        return model, processor

    try:
        config_dir, weights_dir = find_cached_clip_components(model_name_or_path)
    except FileNotFoundError:
        model = CLIPModel.from_pretrained(
            model_name_or_path,
            use_safetensors=True,
        ).to(device)
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        return model, processor

    print(
        f"Using split local Hugging Face cache: config={config_dir.name}, "
        f"weights={weights_dir.name}"
    )
    config = CLIPConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        str(weights_dir),
        config=config,
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        str(config_dir),
        local_files_only=True,
    )
    return model, processor


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_pair_scores_csv(
    path: Path,
    dataset: ClipStructuredDataset,
    pairs: Sequence[PairExample],
    scores: Sequence[float],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "image_index",
                "text_index",
                "label",
                "prediction",
                "kind",
                "score",
                "threshold",
                "image_event_id",
                "text_event_id",
                "image_sample_id",
                "text_sample_id",
                "image_filename",
                "text_filename",
            ],
        )
        writer.writeheader()
        for pair_id, (pair, score) in enumerate(zip(pairs, scores)):
            image_item = dataset.data[pair.image_index]
            text_item = dataset.data[pair.text_index]
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "image_index": pair.image_index,
                    "text_index": pair.text_index,
                    "label": pair.label,
                    "prediction": int(score >= threshold),
                    "kind": pair.kind,
                    "score": f"{score:.8f}",
                    "threshold": f"{threshold:.8f}",
                    "image_event_id": image_item.get("event_id", ""),
                    "text_event_id": text_item.get("event_id", ""),
                    "image_sample_id": image_item.get("sample_id", ""),
                    "text_sample_id": text_item.get("sample_id", ""),
                    "image_filename": image_item.get("filename", ""),
                    "text_filename": text_item.get("filename", ""),
                }
            )


def make_loader(dataset: ClipStructuredDataset, pairs: Sequence[PairExample], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ClipPairDataset(dataset, pairs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_clip_pairs,
    )


def make_sample_loader(dataset: ClipStructuredDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_clip_samples,
    )


@torch.no_grad()
def encode_dataset_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset: ClipStructuredDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = make_sample_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    image_feature_batches = []
    text_feature_batches = []

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = processor(
            text=batch["texts"],
            images=batch["images"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        image_feature_batches.append(F.normalize(image_features, p=2, dim=-1).detach().cpu())
        text_feature_batches.append(F.normalize(text_features, p=2, dim=-1).detach().cpu())

        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
            print(f"Encoded {min(batch_idx * batch_size, len(dataset))}/{len(dataset)} samples")

    return torch.cat(image_feature_batches, dim=0), torch.cat(text_feature_batches, dim=0)


@torch.no_grad()
def score_clip_pairs(
    model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[float], List[int], List[PairExample]]:
    model.eval()
    scores: List[float] = []
    labels: List[int] = []
    pairs: List[PairExample] = []

    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs = processor(
            text=batch["texts"],
            images=batch["images"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        batch_scores = cosine_scores_from_features(image_features, text_features).detach().cpu().tolist()

        scores.extend(float(score) for score in batch_scores)
        labels.extend(int(label) for label in batch["labels"].cpu().tolist())
        pairs.extend(batch["pairs"])

        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(dataloader):
            print(f"Scored {min(batch_idx * dataloader.batch_size, len(dataloader.dataset))}/{len(dataloader.dataset)} pairs")

    return scores, labels, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP on the same image-text consistency pairs used by KG experiments.")
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--negative-mode", choices=["random", "event_hard", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--save-pair-csv", type=str_to_bool, default=True)
    parser.add_argument("--save-val-pair-csv", type=str_to_bool, default=True)
    parser.add_argument("--val-pairs-json", default=None)
    parser.add_argument("--test-pairs-json", default=None)
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    run_start = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_or_path = resolve_clip_model_name(args.model_name, repo_root)
    val_json = resolve_repo_path(args.val_json, repo_root)
    test_json = resolve_repo_path(args.test_json, repo_root)
    img_dir = resolve_repo_path(args.img_dir, repo_root)
    output_dir = resolve_repo_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device.type.upper()}")
    print(f"CLIP model: {model_name_or_path}")

    print("1. Loading validation and test datasets...")
    val_dataset = ClipStructuredDataset(val_json, img_dir)
    test_dataset = ClipStructuredDataset(test_json, img_dir)

    if bool(args.val_pairs_json) != bool(args.test_pairs_json):
        raise ValueError(
            "--val-pairs-json and --test-pairs-json must be provided together."
        )

    max_val_samples = args.max_val_samples if args.max_val_samples is not None else args.max_samples
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else args.max_samples
    val_pairs_path = None
    test_pairs_path = None
    val_manifest = None
    test_manifest = None
    if args.val_pairs_json:
        val_pairs_path = resolve_repo_path(args.val_pairs_json, repo_root)
        test_pairs_path = resolve_repo_path(args.test_pairs_json, repo_root)
        val_manifest = load_fixed_pair_manifest(val_pairs_path)
        test_manifest = load_fixed_pair_manifest(test_pairs_path)
        val_pairs = load_pair_manifest_for_dataset(val_manifest, val_dataset)
        test_pairs = load_pair_manifest_for_dataset(test_manifest, test_dataset)
        pair_mode = "fixed_semantic_hard"
    else:
        val_pairs = build_pair_indices(
            dataset_event_labels(val_dataset),
            negative_mode=args.negative_mode,
            seed=args.seed,
            max_samples=max_val_samples,
        )
        test_pairs = build_pair_indices(
            dataset_event_labels(test_dataset),
            negative_mode=args.negative_mode,
            seed=args.seed + 1000,
            max_samples=max_test_samples,
        )
        pair_mode = args.negative_mode
    print(f"Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Val pairs: {len(val_pairs)} {count_pair_kinds(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)} {count_pair_kinds(test_pairs)}")

    print("2. Loading CLIP model...")
    model, processor = load_clip_model_and_processor(model_name_or_path, device)

    print("3. Encoding validation features...")
    val_image_features, val_text_features = encode_dataset_features(
        model=model,
        processor=processor,
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print("4. Scoring validation pairs...")
    val_scores = scores_from_feature_cache(val_image_features, val_text_features, val_pairs)
    val_labels = [pair.label for pair in val_pairs]
    scored_val_pairs = list(val_pairs)
    if args.threshold is None:
        threshold, val_metrics = choose_best_threshold(val_scores, val_labels)
        print(f"Selected threshold on validation: {threshold:.6f}")
    else:
        threshold = args.threshold
        val_metrics = compute_binary_metrics(val_scores, val_labels, threshold)
        print(f"Using fixed threshold: {threshold:.6f}")
    val_by_kind = compute_metrics_by_kind(val_scores, scored_val_pairs, threshold)

    print("5. Encoding test features...")
    test_image_features, test_text_features = encode_dataset_features(
        model=model,
        processor=processor,
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print("6. Scoring test pairs...")
    test_scores = scores_from_feature_cache(test_image_features, test_text_features, test_pairs)
    test_labels = [pair.label for pair in test_pairs]
    scored_test_pairs = list(test_pairs)
    test_metrics = compute_binary_metrics(test_scores, test_labels, threshold)
    test_by_kind = compute_metrics_by_kind(test_scores, scored_test_pairs, threshold)

    print()
    print_metrics("Validation metrics", val_metrics)
    print()
    print_grouped_metrics("Validation metrics by negative type", val_by_kind)
    print()
    print_metrics("Test metrics", test_metrics)
    print()
    print_grouped_metrics("Test metrics by negative type", test_by_kind)

    model_tag = Path(str(model_name_or_path)).name if Path(str(model_name_or_path)).exists() else str(args.model_name).replace("/", "_")
    save_prefix = args.save_prefix or f"{model_tag}_clip_consistency"
    summary_path = output_dir / f"{save_prefix}_{pair_mode}_summary.json"
    val_csv_path = output_dir / f"{save_prefix}_{pair_mode}_val_pairs.csv"
    pair_csv_path = output_dir / f"{save_prefix}_{pair_mode}_test_pairs.csv"

    summary = {
        "task": "clip_image_text_evidence_consistency",
        "model_name": args.model_name,
        "model_name_or_path": model_name_or_path,
        "val_json": str(val_json),
        "test_json": str(test_json),
        "img_dir": str(img_dir),
        "negative_mode": args.negative_mode,
        "pair_mode": pair_mode,
        "val_pairs_json": str(val_pairs_path) if val_pairs_path else None,
        "test_pairs_json": str(test_pairs_path) if test_pairs_path else None,
        "pair_mining_model": (
            test_manifest.get("mining_model") if test_manifest else None
        ),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_val_samples": max_val_samples,
        "max_test_samples": max_test_samples,
        "threshold": threshold,
        "val_pair_counts": count_pair_kinds(val_pairs),
        "test_pair_counts": count_pair_kinds(test_pairs),
        "validation": val_metrics,
        "validation_by_negative_type": val_by_kind,
        "test": test_metrics,
        "test_by_negative_type": test_by_kind,
        "timing": {
            "total_seconds": time.perf_counter() - run_start,
        },
    }
    save_json(summary, summary_path)
    if args.save_pair_csv and args.save_val_pair_csv:
        save_pair_scores_csv(val_csv_path, val_dataset, scored_val_pairs, val_scores, threshold)
    if args.save_pair_csv:
        save_pair_scores_csv(pair_csv_path, test_dataset, scored_test_pairs, test_scores, threshold)

    print(f"Saved summary: {summary_path}")
    if args.save_pair_csv and args.save_val_pair_csv:
        print(f"Saved validation pair scores: {val_csv_path}")
    if args.save_pair_csv:
        print(f"Saved test pair scores: {pair_csv_path}")


if __name__ == "__main__":
    main()
