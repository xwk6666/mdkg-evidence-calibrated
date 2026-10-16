import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from clip_semantic_negatives import (
    infer_disaster_type,
    infer_event_family,
    load_semantic_negative_cache,
    sample_identifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSON = PROJECT_ROOT / "data" / "CrisisMMD_v2.0" / "structured" / "train_entities.json"
VAL_JSON = PROJECT_ROOT / "data" / "CrisisMMD_v2.0" / "structured" / "val_entities.json"
IMG_DIR = PROJECT_ROOT / "data" / "CrisisMMD_v2.0"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs" / "clip_finetuned_crisismmd"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAVE_DIR_NAME = "clip_finetuned_crisismmd"
BEST_DIR_NAME = "clip_finetuned_crisismmd_best"

BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-6
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
MAX_GRAD_NORM = 1.0
USE_AMP = True
SEED = 42
NUM_WORKERS = 0
RECALL_KS = (1, 5, 10)


def build_event_negative_indices(samples: Sequence[Dict[str, object]], seed: int = SEED) -> List[int]:
    """Choose one static negative text index from a different event for each sample."""

    if len(samples) < 2:
        raise ValueError("At least two samples are required to build event hard negatives.")

    all_indices = list(range(len(samples)))
    rng = random.Random(seed)
    negative_indices: List[int] = []
    for index, item in enumerate(samples):
        event_id = str(item.get("event_id", ""))
        candidates = [
            candidate
            for candidate in all_indices
            if candidate != index and str(samples[candidate].get("event_id", "")) != event_id
        ]
        if not candidates:
            raise ValueError("Unable to build event hard negatives: all samples share the same event_id.")
        negative_indices.append(rng.choice(candidates))
    return negative_indices


class CrisisMMDClipDataset(Dataset):
    def __init__(
        self,
        structured_json_path: str,
        img_dir: str,
        include_event_negative: bool = False,
        hard_negative_seed: int = SEED,
        semantic_negative_cache_path: Optional[str] = None,
        semantic_negative_top_k: int = 5,
    ):
        self.structured_json_path = str(structured_json_path)
        with open(structured_json_path, "r", encoding="utf-8") as handle:
            raw_data = json.load(handle)["samples"]
        self.img_dir = Path(img_dir)
        self.missing_images: List[Dict[str, str]] = []
        self.data = self._filter_and_resolve_images(raw_data)
        self.hard_negative_seed = hard_negative_seed
        self.current_epoch = 0
        self.event_negative_indices: Optional[List[int]] = None
        self.semantic_negative_indices: Optional[List[List[int]]] = None
        if semantic_negative_cache_path:
            self.semantic_negative_indices = self._load_semantic_negative_indices(
                semantic_negative_cache_path,
                semantic_negative_top_k,
            )
        elif include_event_negative:
            self.event_negative_indices = build_event_negative_indices(self.data, seed=hard_negative_seed)

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

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

    def _filter_and_resolve_images(self, samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
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
                continue
            new_item = dict(item)
            new_item["_resolved_image_path"] = str(resolved_path)
            kept.append(new_item)

        if self.missing_images:
            print(
                f"Skipped {len(self.missing_images)} samples with missing images "
                f"from {self.structured_json_path}. "
                f"Kept {len(kept)}/{len(samples)} samples."
            )
        return kept

    def _load_semantic_negative_indices(
        self,
        cache_path: str,
        top_k: int,
    ) -> List[List[int]]:
        if top_k < 1:
            raise ValueError("semantic_negative_top_k must be at least 1.")

        cache = load_semantic_negative_cache(Path(cache_path))
        id_to_index = {
            sample_identifier(sample): index for index, sample in enumerate(self.data)
        }
        candidate_indices: List[List[int]] = []
        for anchor_index, sample in enumerate(self.data):
            anchor_id = sample_identifier(sample)
            anchor_type = infer_disaster_type(sample)
            anchor_family = infer_event_family(str(sample.get("event_id", "")))
            valid_candidates = []
            record = cache.get(anchor_id, {})
            for candidate_id in record.get("negative_sample_ids", []) or []:
                candidate_index = id_to_index.get(str(candidate_id))
                if candidate_index is None or candidate_index == anchor_index:
                    continue
                candidate = self.data[candidate_index]
                if infer_disaster_type(candidate) != anchor_type:
                    continue
                candidate_family = infer_event_family(
                    str(candidate.get("event_id", ""))
                )
                if candidate_family == anchor_family:
                    continue
                valid_candidates.append(candidate_index)
                if len(valid_candidates) >= top_k:
                    break
            candidate_indices.append(valid_candidates)

        covered = sum(bool(candidates) for candidates in candidate_indices)
        print(
            f"Loaded semantic negative cache: {covered}/{len(self.data)} "
            f"samples have valid Top-{top_k} candidates."
        )
        return candidate_indices

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.data[idx]
        image = Image.open(item["_resolved_image_path"]).convert("RGB")
        output = {
            "image": image,
            "text": item["text"],
            "filename": item["filename"],
            "event_id": item.get("event_id", ""),
        }
        if self.semantic_negative_indices is not None:
            candidates = self.semantic_negative_indices[idx]
            if candidates:
                rng = random.Random(
                    self.hard_negative_seed
                    + idx * 1000003
                    + self.current_epoch * 9176
                )
                negative_index = candidates[rng.randrange(len(candidates))]
                negative_item = self.data[negative_index]
                output["negative_text"] = negative_item["text"]
                output["negative_event_id"] = negative_item.get("event_id", "")
                output["negative_sample_id"] = sample_identifier(negative_item)
                output["negative_is_valid"] = True
            else:
                output["negative_text"] = item["text"]
                output["negative_event_id"] = item.get("event_id", "")
                output["negative_sample_id"] = sample_identifier(item)
                output["negative_is_valid"] = False
        elif self.event_negative_indices is not None:
            negative_item = self.data[self.event_negative_indices[idx]]
            output["negative_text"] = negative_item["text"]
            output["negative_event_id"] = negative_item.get("event_id", "")
            output["negative_sample_id"] = sample_identifier(negative_item)
            output["negative_is_valid"] = True
        return output


def collate_clip_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    output = {
        "images": [item["image"] for item in batch],
        "texts": [item["text"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "event_ids": [item.get("event_id", "") for item in batch],
    }
    if "negative_text" in batch[0]:
        output["negative_texts"] = [item["negative_text"] for item in batch]
        output["negative_event_ids"] = [item.get("negative_event_id", "") for item in batch]
        output["negative_sample_ids"] = [item.get("negative_sample_id", "") for item in batch]
        output["negative_valid_mask"] = torch.tensor(
            [bool(item.get("negative_is_valid", True)) for item in batch],
            dtype=torch.bool,
        )
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def unwrap_clip_features(output, projection):
    if torch.is_tensor(output):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        pooled = output.pooler_output
        if pooled.shape[-1] == projection.out_features:
            return pooled
        return projection(pooled)
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)}")


def compute_hard_negative_margin_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    negative_text_features: torch.Tensor,
    margin: float = 0.2,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    negative_text_features = F.normalize(negative_text_features, p=2, dim=-1)
    positive_scores = (image_features * text_features).sum(dim=-1)
    negative_scores = (image_features * negative_text_features).sum(dim=-1)
    losses = F.relu(negative_scores - positive_scores + margin)
    if valid_mask is None:
        return losses.mean()
    valid_mask = valid_mask.to(device=losses.device, dtype=torch.bool)
    if valid_mask.numel() != losses.numel():
        raise ValueError("valid_mask must contain one value per training sample.")
    if not bool(valid_mask.any()):
        return losses.sum() * 0.0
    return losses[valid_mask].mean()


def compute_clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    negative_text_features: Optional[torch.Tensor] = None,
    negative_valid_mask: Optional[torch.Tensor] = None,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.2,
) -> torch.Tensor:
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    logits = torch.matmul(image_features, text_features.T) / TEMPERATURE
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
    base_loss = 0.5 * (loss_i2t + loss_t2i)
    if negative_text_features is None or hard_negative_weight <= 0:
        return base_loss
    hard_negative_loss = compute_hard_negative_margin_loss(
        image_features=image_features,
        text_features=text_features,
        negative_text_features=negative_text_features,
        margin=hard_negative_margin,
        valid_mask=negative_valid_mask,
    )
    return base_loss + hard_negative_weight * hard_negative_loss


def compute_retrieval_metrics(sim_matrix: torch.Tensor, ks=RECALL_KS) -> Dict[str, float]:
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    gt = torch.arange(sim_matrix.size(0), device=sim_matrix.device).unsqueeze(1)
    ranks = torch.argmax((sorted_indices == gt).to(torch.int64), dim=1) + 1

    metrics: Dict[str, float] = {}
    recalls = []
    for k in ks:
        if k <= sim_matrix.size(0):
            value = (ranks <= k).float().mean().item() * 100.0
            metrics[f"R@{k}"] = value
            recalls.append(value)
    metrics["MeanRecall"] = sum(recalls) / len(recalls) if recalls else 0.0
    metrics["MedR"] = float(ranks.median().item())
    metrics["MeanRank"] = float(ranks.float().mean().item())
    return metrics


def encode_batch(model, processor, batch, device):
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
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    image_features = unwrap_clip_features(image_features, model.visual_projection)
    text_features = unwrap_clip_features(text_features, model.text_projection)
    return image_features, text_features


def encode_texts(model, processor, texts: Sequence[str], device):
    inputs = processor(
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    return unwrap_clip_features(text_features, model.text_projection)


@torch.no_grad()
def evaluate_retrieval(model, processor, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_img_feats = []
    all_txt_feats = []

    for batch in dataloader:
        image_features, text_features = encode_batch(model, processor, batch, device)
        all_img_feats.append(image_features.cpu())
        all_txt_feats.append(text_features.cpu())

    all_img_feats = F.normalize(torch.cat(all_img_feats, dim=0), p=2, dim=-1)
    all_txt_feats = F.normalize(torch.cat(all_txt_feats, dim=0), p=2, dim=-1)
    sim_matrix = torch.matmul(all_img_feats, all_txt_feats.T)

    i2t = compute_retrieval_metrics(sim_matrix, ks=RECALL_KS)
    t2i = compute_retrieval_metrics(sim_matrix.T, ks=RECALL_KS)
    return {
        "val_i2t_R@1": i2t["R@1"],
        "val_i2t_R@5": i2t["R@5"],
        "val_i2t_R@10": i2t["R@10"],
        "val_t2i_R@1": t2i["R@1"],
        "val_t2i_R@5": t2i["R@5"],
        "val_t2i_R@10": t2i["R@10"],
        "val_mean_recall_6": (
            i2t["R@1"] + i2t["R@5"] + i2t["R@10"] + t2i["R@1"] + t2i["R@5"] + t2i["R@10"]
        ) / 6.0,
    }


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP as a CrisisMMD comparison baseline.")
    parser.add_argument("--model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--train-json", default=str(TRAIN_JSON))
    parser.add_argument("--val-json", default=str(VAL_JSON))
    parser.add_argument("--img-dir", default=str(IMG_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save-dir-name", default=SAVE_DIR_NAME)
    parser.add_argument("--best-dir-name", default=BEST_DIR_NAME)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--hard-negative-seed", type=int, default=SEED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--semantic-negative-cache", default=None)
    parser.add_argument("--semantic-negative-top-k", type=int, default=5)
    parser.add_argument("--log-dir-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    print("1. Loading train/val datasets...")
    use_hard_negative = args.hard_negative_weight > 0
    train_dataset = CrisisMMDClipDataset(
        args.train_json,
        args.img_dir,
        include_event_negative=use_hard_negative and not args.semantic_negative_cache,
        hard_negative_seed=args.hard_negative_seed,
        semantic_negative_cache_path=(
            args.semantic_negative_cache if use_hard_negative else None
        ),
        semantic_negative_top_k=args.semantic_negative_top_k,
    )
    val_dataset = CrisisMMDClipDataset(args.val_json, args.img_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_clip_batch,
        generator=make_torch_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_clip_batch,
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    print(f"2. Loading CLIP model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    run_log_dir = PROJECT_ROOT / "logs" / (
        args.log_dir_name or args.save_dir_name
    )
    run_log_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = MODEL_DIR / args.save_dir_name
    best_dir = MODEL_DIR / args.best_dir_name

    config = {
        "model_name": args.model_name,
        "train_json": args.train_json,
        "val_json": args.val_json,
        "img_dir": args.img_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature": TEMPERATURE,
        "weight_decay": WEIGHT_DECAY,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_margin": args.hard_negative_margin,
        "hard_negative_seed": args.hard_negative_seed,
        "seed": args.seed,
        "semantic_negative_cache": args.semantic_negative_cache,
        "semantic_negative_top_k": args.semantic_negative_top_k,
    }
    save_json(config, run_log_dir / "config.json")

    best_metric = -1.0
    best_epoch = 0
    history = []

    print("3. Fine-tuning CLIP...")
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                image_features, text_features = encode_batch(model, processor, batch, device)
                negative_text_features = None
                negative_valid_mask = None
                if use_hard_negative:
                    negative_text_features = encode_texts(model, processor, batch["negative_texts"], device)
                    negative_valid_mask = batch["negative_valid_mask"].to(device)
                loss = compute_clip_loss(
                    image_features,
                    text_features,
                    negative_text_features=negative_text_features,
                    negative_valid_mask=negative_valid_mask,
                    hard_negative_weight=args.hard_negative_weight,
                    hard_negative_margin=args.hard_negative_margin,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            if (batch_idx + 1) % 50 == 0:
                avg = loss_sum / (batch_idx + 1)
                print(f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {avg:.4f}")

        scheduler.step()

        val_metrics = evaluate_retrieval(model, processor, val_loader, device)
        avg_loss = loss_sum / max(len(train_loader), 1)
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "epoch_seconds": epoch_seconds,
            "lr": scheduler.get_last_lr()[0],
            **val_metrics,
        }
        history.append(record)
        save_json(history, run_log_dir / "val_metric_history.json")

        metric = val_metrics["val_i2t_R@1"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
            print(f"Saved new best CLIP checkpoint at epoch {best_epoch}: {best_metric:.2f}")

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] done | Loss={avg_loss:.4f} | "
            f"Val I2T R@1={val_metrics['val_i2t_R@1']:.2f} | "
            f"Val T2I R@1={val_metrics['val_t2i_R@1']:.2f} | "
            f"Time={epoch_seconds:.1f}s"
        )

    model.save_pretrained(save_dir, safe_serialization=True)
    processor.save_pretrained(save_dir)

    total_seconds = time.perf_counter() - run_start
    save_json(
        {
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60.0,
            "best_epoch": best_epoch,
            "best_metric_name": "val_i2t_R@1",
            "best_metric_value": best_metric,
            "best_checkpoint": str(best_dir),
            "last_checkpoint": str(save_dir),
        },
        run_log_dir / "timing_summary.json",
    )
    print(f"Training finished. Best epoch={best_epoch}, best val_i2t_R@1={best_metric:.2f}")
    print(f"Best checkpoint: {best_dir}")


if __name__ == "__main__":
    main()
