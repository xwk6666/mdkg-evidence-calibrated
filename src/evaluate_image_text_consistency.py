import argparse
import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CHECKPOINT = "attention_entity_event_finetune_model_best.pth"
DEFAULT_VAL_JSON = "data/CrisisMMD_v2.0/structured/val_entities.json"
DEFAULT_TEST_JSON = "data/CrisisMMD_v2.0/structured/test_entities.json"
DEFAULT_IMG_DIR = "data/CrisisMMD_v2.0"
DEFAULT_OUTPUT_DIR = "outputs/image_text_consistency"


@dataclass(frozen=True)
class PairExample:
    image_index: int
    text_index: int
    label: int
    kind: str


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")

    num_positive = sum(1 for label in labels if label == 1)
    num_negative = len(labels) - num_positive
    if num_positive == 0 or num_negative == 0:
        return 0.0

    # Rank-based AUC is equivalent to pairwise comparison, but avoids O(P*N) loops.
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1

        first_rank = index + 1
        last_rank = next_index
        average_rank = (first_rank + last_rank) / 2.0
        tied_positive = sum(1 for _, label in pairs[index:next_index] if label == 1)
        positive_rank_sum += tied_positive * average_rank
        index = next_index

    auc = (positive_rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)
    return float(auc)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    total_positive = sum(1 for label in labels if label == 1)
    if total_positive == 0:
        return 0.0

    hit_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            hit_positive += 1
            precision_sum += hit_positive / rank
    return precision_sum / total_positive


def metrics_from_confusion(
    threshold: float,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    num_pairs: int,
    num_positive: int,
) -> Dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, num_pairs)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "num_pairs": int(num_pairs),
        "num_positive": int(num_positive),
        "num_negative": int(num_pairs - num_positive),
    }


def compute_threshold_metrics(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("scores must not be empty.")

    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)

    return metrics_from_confusion(
        threshold=threshold,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        num_pairs=len(labels),
        num_positive=sum(labels),
    )


def compute_binary_metrics(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    metrics = compute_threshold_metrics(scores, labels, threshold)
    metrics["auc"] = binary_auc(scores, labels)
    metrics["average_precision"] = average_precision(scores, labels)
    return metrics


def candidate_thresholds(scores: Sequence[float]) -> List[float]:
    unique_scores = sorted({float(score) for score in scores})
    if not unique_scores:
        raise ValueError("scores must not be empty.")
    thresholds = list(unique_scores)
    for left, right in zip(unique_scores, unique_scores[1:]):
        thresholds.append((left + right) / 2.0)
    thresholds.append(unique_scores[0] - 1e-6)
    thresholds.append(unique_scores[-1] + 1e-6)
    return sorted(set(thresholds))


def choose_best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, Dict[str, float]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("scores must not be empty.")

    best_threshold: Optional[float] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, float, float, float]] = None

    def consider(threshold: float, tp: int, tn: int, fp: int, fn: int) -> None:
        nonlocal best_threshold, best_metrics, best_key
        metrics = metrics_from_confusion(
            threshold=threshold,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            num_pairs=len(labels),
            num_positive=num_positive,
        )
        key = (metrics["f1"], metrics["accuracy"], metrics["precision"], metrics["recall"])
        if best_metrics is None or best_key is None:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
            return
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key

    grouped_scores: Dict[float, List[int]] = {}
    for score, label in zip(scores, labels):
        grouped_scores.setdefault(float(score), [0, 0])[int(label)] += 1

    unique_scores = sorted(grouped_scores)
    num_positive = int(sum(labels))
    num_negative = int(len(labels) - num_positive)

    tp = num_positive
    fp = num_negative
    tn = 0
    fn = 0

    consider(unique_scores[0] - 1e-6, tp, tn, fp, fn)
    consider(unique_scores[0], tp, tn, fp, fn)

    previous_score = unique_scores[0]
    for score in unique_scores[1:]:
        previous_negative, previous_positive = grouped_scores[previous_score]
        tp -= previous_positive
        fn += previous_positive
        fp -= previous_negative
        tn += previous_negative

        consider((previous_score + score) / 2.0, tp, tn, fp, fn)
        consider(score, tp, tn, fp, fn)
        previous_score = score

    last_negative, last_positive = grouped_scores[previous_score]
    tp -= last_positive
    fn += last_positive
    fp -= last_negative
    tn += last_negative
    consider(unique_scores[-1] + 1e-6, tp, tn, fp, fn)

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Failed to select a threshold.")
    return float(best_threshold), compute_binary_metrics(scores, labels, best_threshold)


def choose_random_negative(anchor: int, indices: Sequence[int], rng: random.Random) -> int:
    if len(indices) < 2:
        raise ValueError("At least two samples are required to create negative pairs.")
    candidate = anchor
    while candidate == anchor:
        candidate = rng.choice(indices)
    return candidate


def choose_event_hard_negative(
    anchor: int,
    indices: Sequence[int],
    event_labels: Sequence[int],
    rng: random.Random,
) -> Optional[int]:
    anchor_event = event_labels[anchor]
    candidates = [idx for idx in indices if idx != anchor and event_labels[idx] != anchor_event]
    if not candidates:
        return None
    return rng.choice(candidates)


def build_pair_indices(
    event_labels: Sequence[int],
    negative_mode: str = "both",
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[PairExample]:
    if negative_mode not in {"random", "event_hard", "both"}:
        raise ValueError("negative_mode must be one of: random, event_hard, both.")

    n_samples = len(event_labels)
    if n_samples < 2:
        raise ValueError("At least two samples are required.")

    rng = random.Random(seed)
    all_indices = list(range(n_samples))
    anchors = list(all_indices)
    if max_samples is not None and max_samples > 0 and max_samples < n_samples:
        rng.shuffle(anchors)
        anchors = sorted(anchors[:max_samples])

    pairs: List[PairExample] = []
    for anchor in anchors:
        pairs.append(PairExample(anchor, anchor, 1, "positive"))

        if negative_mode in {"random", "both"}:
            negative = choose_random_negative(anchor, all_indices, rng)
            pairs.append(PairExample(anchor, negative, 0, "random_negative"))

        if negative_mode in {"event_hard", "both"}:
            hard_negative = choose_event_hard_negative(anchor, all_indices, event_labels, rng)
            if hard_negative is not None:
                pairs.append(PairExample(anchor, hard_negative, 0, "event_hard_negative"))

    return pairs


def count_pair_kinds(pairs: Iterable[PairExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pair in pairs:
        counts[pair.kind] = counts.get(pair.kind, 0) + 1
    return counts


def limit_pairs_by_anchor(
    pairs: Sequence[PairExample],
    max_anchors: Optional[int],
) -> List[PairExample]:
    """Keep complete pair groups for the first N image anchors."""

    if max_anchors is None or max_anchors <= 0:
        return list(pairs)

    selected_anchors: List[int] = []
    selected_anchor_set = set()
    for pair in pairs:
        if pair.image_index in selected_anchor_set:
            continue
        selected_anchors.append(pair.image_index)
        selected_anchor_set.add(pair.image_index)
        if len(selected_anchors) >= max_anchors:
            break
    return [pair for pair in pairs if pair.image_index in selected_anchor_set]


def resolve_scoring_mode(requested_mode: str, use_attention: bool) -> str:
    """Choose a leakage-safe scoring strategy for the model architecture."""

    if requested_mode not in {"auto", "feature_cache", "pairwise"}:
        raise ValueError("scoring_mode must be one of: auto, feature_cache, pairwise.")
    if requested_mode == "auto":
        return "pairwise" if use_attention else "feature_cache"
    if requested_mode == "feature_cache" and use_attention:
        raise ValueError(
            "feature_cache is invalid for a text-conditioned attention model; "
            "score every candidate image-text pair with pairwise mode instead."
        )
    return requested_mode


def sanitize_filename_component(value: str) -> str:
    """Return a Windows-safe filename component for experiment artifacts."""

    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", str(value))
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
    if not cleaned:
        return "result"
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if cleaned.upper() in reserved_names:
        return f"_{cleaned}"
    return cleaned


def scores_from_feature_cache(
    image_features,
    text_features,
    pairs: Sequence[PairExample],
) -> List[float]:
    """Score fixed pairs from independently encoded image and text features."""

    import torch.nn.functional as F

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must both have shape [N, D].")
    if image_features.size(1) != text_features.size(1):
        raise ValueError("image_features and text_features must have the same feature dimension.")

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    scores: List[float] = []
    for pair in pairs:
        score = (
            image_features[pair.image_index]
            * text_features[pair.text_index]
        ).sum()
        scores.append(float(score.item()))
    return scores


def load_fixed_pair_manifest_for_dataset(
    manifest_path: Path,
    dataset,
    max_anchors: Optional[int] = None,
):
    """Load a fixed pair manifest and map stable IDs to this dataset."""

    from fixed_semantic_hard_testset import (
        load_fixed_pair_manifest,
        load_fixed_pairs,
    )

    manifest = load_fixed_pair_manifest(manifest_path)
    pairs = load_fixed_pairs(manifest, dataset.data)
    return manifest, limit_pairs_by_anchor(pairs, max_anchors=max_anchors)


def compute_metrics_by_kind(
    scores: Sequence[float],
    pairs: Sequence[PairExample],
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    if len(scores) != len(pairs):
        raise ValueError("scores and pairs must have the same length.")

    grouped: Dict[str, Dict[str, float]] = {}
    negative_kinds = sorted({pair.kind for pair in pairs if pair.label == 0})
    for negative_kind in negative_kinds:
        selected_scores: List[float] = []
        selected_labels: List[int] = []
        for score, pair in zip(scores, pairs):
            if pair.kind == "positive" or pair.kind == negative_kind:
                selected_scores.append(score)
                selected_labels.append(pair.label)
        if selected_scores:
            grouped[f"{negative_kind}_vs_positive"] = compute_binary_metrics(
                selected_scores,
                selected_labels,
                threshold=threshold,
            )
    return grouped


def load_model(checkpoint_path: Path, text_encoder_name: str, use_attention: bool, use_entity: bool, use_hard_negative: bool, device):
    import torch

    from model import AblationDisasterModel

    model = AblationDisasterModel(
        use_attention=use_attention,
        use_hard_negative=use_hard_negative,
        use_entity_branch=use_entity,
        text_encoder_name=text_encoder_name,
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_dataset(structured_json_path: Path, img_dir: Path, tokenizer_name: str):
    from structured_dataset import StructuredCrisisDataset

    return StructuredCrisisDataset(
        structured_json_path=str(structured_json_path),
        img_dir=str(img_dir),
        tokenizer_name=tokenizer_name,
    )


def dataset_event_labels(dataset) -> List[int]:
    labels: List[int] = []
    for item in dataset.data:
        event_id = item.get("event_id", "unknown_event")
        labels.append(int(dataset.event_to_id.get(event_id, 0)))
    return labels


def text_tensors_for_index(dataset, index: int):
    cached = dataset._text_cache[index]
    return cached["input_ids"].clone(), cached["attention_mask"].clone()


def score_pairs(model, dataset, pairs: Sequence[PairExample], device, batch_size: int) -> List[float]:
    import torch
    import torch.nn.functional as F

    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            images = []
            input_ids = []
            attention_masks = []

            for pair in batch_pairs:
                image_item = dataset[pair.image_index]
                text_ids, text_mask = text_tensors_for_index(dataset, pair.text_index)
                images.append(image_item["image"])
                input_ids.append(text_ids)
                attention_masks.append(text_mask)

            image_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
            input_id_tensor = torch.stack(input_ids, dim=0).to(device, non_blocking=True)
            attention_mask_tensor = torch.stack(attention_masks, dim=0).to(device, non_blocking=True)

            image_feat, text_feat = model(image_tensor, input_id_tensor, attention_mask_tensor)
            image_feat = F.normalize(image_feat, p=2, dim=-1)
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            batch_scores = (image_feat * text_feat).sum(dim=-1).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)

            finished = min(start + batch_size, len(pairs))
            if finished == len(pairs) or finished % (batch_size * 20) == 0:
                print(f"Scored {finished}/{len(pairs)} pairs")

    return scores


def encode_dataset_features(model, dataset, device, batch_size: int):
    """Independently encode every image and text for a pure dual encoder."""

    import torch
    import torch.nn.functional as F

    if getattr(model, "use_attention", False):
        raise ValueError(
            "Independent feature caching is invalid for text-conditioned attention models."
        )

    model.eval()
    image_feature_batches = []
    text_feature_batches = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            items = [dataset[index] for index in range(start, stop)]
            images = torch.stack([item["image"] for item in items], dim=0).to(
                device,
                non_blocking=True,
            )
            input_ids = torch.stack(
                [item["input_ids"] for item in items],
                dim=0,
            ).to(device, non_blocking=True)
            attention_masks = torch.stack(
                [item["attention_mask"] for item in items],
                dim=0,
            ).to(device, non_blocking=True)

            image_features = model.encode_image(images)
            text_features = model.encode_text(input_ids, attention_masks)
            image_feature_batches.append(
                F.normalize(image_features, p=2, dim=-1).detach().cpu()
            )
            text_feature_batches.append(
                F.normalize(text_features, p=2, dim=-1).detach().cpu()
            )

            if stop == len(dataset) or stop % (batch_size * 20) == 0:
                print(f"Encoded {stop}/{len(dataset)} samples")

    return (
        torch.cat(image_feature_batches, dim=0),
        torch.cat(text_feature_batches, dim=0),
    )


def score_dataset_pairs(
    model,
    dataset,
    pairs: Sequence[PairExample],
    device,
    batch_size: int,
    scoring_mode: str,
) -> List[float]:
    if scoring_mode == "pairwise":
        return score_pairs(
            model,
            dataset,
            pairs,
            device=device,
            batch_size=batch_size,
        )
    if scoring_mode == "feature_cache":
        image_features, text_features = encode_dataset_features(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        return scores_from_feature_cache(image_features, text_features, pairs)
    raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")


def save_pair_scores_csv(path: Path, dataset, pairs: Sequence[PairExample], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-text evidence consistency with positive and mismatched disaster pairs."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint filename under models/ or an absolute path.")
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--text-encoder-name", default="bert-base-uncased")
    parser.add_argument("--attention", type=str_to_bool, default=True)
    parser.add_argument("--entity", type=str_to_bool, default=True)
    parser.add_argument("--hard-negative", type=str_to_bool, default=False)
    parser.add_argument("--negative-mode", choices=["random", "event_hard", "both"], default="both")
    parser.add_argument(
        "--scoring-mode",
        choices=["auto", "feature_cache", "pairwise"],
        default="auto",
        help=(
            "auto uses independent feature caching for a pure dual encoder and "
            "candidate-wise scoring for text-conditioned attention."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap applied to val and test anchors.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed threshold; otherwise selected on validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--save-pair-csv", type=str_to_bool, default=True)
    parser.add_argument("--save-val-pair-csv", type=str_to_bool, default=True)
    parser.add_argument("--val-pairs-json", default=None)
    parser.add_argument("--test-pairs-json", default=None)
    return parser.parse_args()


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    print("-" * len(title))
    print(f"threshold         : {metrics['threshold']:.6f}")
    print(f"accuracy          : {metrics['accuracy']:.4f}")
    print(f"precision         : {metrics['precision']:.4f}")
    print(f"recall            : {metrics['recall']:.4f}")
    print(f"f1                : {metrics['f1']:.4f}")
    print(f"auc               : {metrics['auc']:.4f}")
    print(f"average_precision : {metrics['average_precision']:.4f}")
    print(f"confusion         : TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")


def print_grouped_metrics(title: str, grouped_metrics: Dict[str, Dict[str, float]]) -> None:
    if not grouped_metrics:
        return
    print(title)
    print("-" * len(title))
    for name, metrics in grouped_metrics.items():
        print(
            f"{name}: Acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
            f"(TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']})"
        )


def main() -> None:
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    run_start = time.perf_counter()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = repo_root / "models" / checkpoint

    val_json = resolve_repo_path(args.val_json, repo_root)
    test_json = resolve_repo_path(args.test_json, repo_root)
    img_dir = resolve_repo_path(args.img_dir, repo_root)
    output_dir = resolve_repo_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_save_prefix = args.save_prefix or checkpoint.stem
    save_prefix = sanitize_filename_component(requested_save_prefix)
    if save_prefix != requested_save_prefix:
        print(
            "Warning: sanitized invalid characters in --save-prefix: "
            f"{requested_save_prefix!r} -> {save_prefix!r}"
        )

    if bool(args.val_pairs_json) != bool(args.test_pairs_json):
        raise ValueError(
            "--val-pairs-json and --test-pairs-json must be provided together."
        )

    scoring_mode = resolve_scoring_mode(args.scoring_mode, args.attention)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Scoring mode: {scoring_mode}")

    print("1. Loading validation and test datasets...")
    val_dataset = load_dataset(val_json, img_dir, args.text_encoder_name)
    test_dataset = load_dataset(test_json, img_dir, args.text_encoder_name)
    print(f"Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    max_val_samples = args.max_val_samples if args.max_val_samples is not None else args.max_samples
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else args.max_samples

    val_pairs_path = None
    test_pairs_path = None
    val_manifest = None
    test_manifest = None
    if args.val_pairs_json:
        val_pairs_path = resolve_repo_path(args.val_pairs_json, repo_root)
        test_pairs_path = resolve_repo_path(args.test_pairs_json, repo_root)
        val_manifest, val_pairs = load_fixed_pair_manifest_for_dataset(
            val_pairs_path,
            val_dataset,
            max_anchors=max_val_samples,
        )
        test_manifest, test_pairs = load_fixed_pair_manifest_for_dataset(
            test_pairs_path,
            test_dataset,
            max_anchors=max_test_samples,
        )
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
    print(f"Val pairs: {len(val_pairs)} {count_pair_kinds(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)} {count_pair_kinds(test_pairs)}")

    print("2. Loading model...")
    model = load_model(
        checkpoint_path=checkpoint,
        text_encoder_name=args.text_encoder_name,
        use_attention=args.attention,
        use_entity=args.entity,
        use_hard_negative=args.hard_negative,
        device=device,
    )

    print("3. Scoring validation pairs...")
    val_scores = score_dataset_pairs(
        model,
        val_dataset,
        val_pairs,
        device=device,
        batch_size=args.batch_size,
        scoring_mode=scoring_mode,
    )
    val_labels = [pair.label for pair in val_pairs]
    if args.threshold is None:
        threshold, val_metrics = choose_best_threshold(val_scores, val_labels)
        print(f"Selected threshold on validation: {threshold:.6f}")
    else:
        threshold = args.threshold
        val_metrics = compute_binary_metrics(val_scores, val_labels, threshold=threshold)
        print(f"Using fixed threshold: {threshold:.6f}")
    val_by_kind = compute_metrics_by_kind(val_scores, val_pairs, threshold=threshold)

    print("4. Scoring test pairs...")
    test_scores = score_dataset_pairs(
        model,
        test_dataset,
        test_pairs,
        device=device,
        batch_size=args.batch_size,
        scoring_mode=scoring_mode,
    )
    test_labels = [pair.label for pair in test_pairs]
    test_metrics = compute_binary_metrics(
        test_scores,
        test_labels,
        threshold=threshold,
    )
    test_by_kind = compute_metrics_by_kind(test_scores, test_pairs, threshold=threshold)

    print()
    print_metrics("Validation metrics", val_metrics)
    print()
    print_grouped_metrics("Validation metrics by negative type", val_by_kind)
    print()
    print_metrics("Test metrics", test_metrics)
    print()
    print_grouped_metrics("Test metrics by negative type", test_by_kind)

    summary_path = output_dir / f"{save_prefix}_{pair_mode}_summary.json"
    val_csv_path = output_dir / f"{save_prefix}_{pair_mode}_val_pairs.csv"
    pair_csv_path = output_dir / f"{save_prefix}_{pair_mode}_test_pairs.csv"

    summary = {
        "task": "image_text_evidence_consistency",
        "checkpoint": str(checkpoint),
        "val_json": str(val_json),
        "test_json": str(test_json),
        "img_dir": str(img_dir),
        "text_encoder_name": args.text_encoder_name,
        "use_attention": args.attention,
        "use_entity": args.entity,
        "use_hard_negative": args.hard_negative,
        "negative_mode": args.negative_mode,
        "pair_mode": pair_mode,
        "scoring_mode": scoring_mode,
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
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")

    if args.save_pair_csv and args.save_val_pair_csv:
        save_pair_scores_csv(
            val_csv_path,
            val_dataset,
            val_pairs,
            val_scores,
            threshold,
        )
        print(f"Saved validation pair scores: {val_csv_path}")
    if args.save_pair_csv:
        save_pair_scores_csv(pair_csv_path, test_dataset, test_pairs, test_scores, threshold)
        print(f"Saved test pair scores: {pair_csv_path}")


if __name__ == "__main__":
    main()
