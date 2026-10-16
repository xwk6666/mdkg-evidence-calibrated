import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from evaluate_image_text_consistency import choose_best_threshold, compute_binary_metrics, resolve_repo_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL_CSV = PROJECT_ROOT / "outputs" / "clip_image_text_consistency" / "finetuned_clip_consistency_both_val_pairs.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "outputs" / "clip_image_text_consistency" / "finetuned_clip_consistency_both_test_pairs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clip_calibration"


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def standardize_scores(
    scores: Sequence[float],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[List[float], float, float]:
    if not scores:
        raise ValueError("scores must not be empty.")
    if mean is None:
        mean = float(sum(scores) / len(scores))
    if std is None:
        denominator = max(len(scores) - 1, 1)
        variance = sum((float(score) - mean) ** 2 for score in scores) / denominator
        std = math.sqrt(variance)
    if std <= 1e-12:
        std = 1.0
    return [(float(score) - mean) / std for score in scores], mean, std


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length.")
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    return float(sum((float(prob) - int(label)) ** 2 for prob, label in zip(probabilities, labels)) / len(labels))


def binary_log_loss(probabilities: Sequence[float], labels: Sequence[int], eps: float = 1e-12) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length.")
    total = 0.0
    for prob, label in zip(probabilities, labels):
        clipped = min(max(float(prob), eps), 1.0 - eps)
        total += int(label) * math.log(clipped) + (1 - int(label)) * math.log(1.0 - clipped)
    return float(-total / len(labels)) if labels else 0.0


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], num_bins: int = 10) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if not probabilities:
        raise ValueError("probabilities must not be empty.")

    total = len(probabilities)
    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            indices = [idx for idx, prob in enumerate(probabilities) if lower <= float(prob) <= upper]
        else:
            indices = [idx for idx, prob in enumerate(probabilities) if lower <= float(prob) < upper]
        if not indices:
            continue
        confidence = sum(float(probabilities[idx]) for idx in indices) / len(indices)
        accuracy = sum(int(labels[idx]) for idx in indices) / len(indices)
        ece += (len(indices) / total) * abs(confidence - accuracy)
    return float(ece)


def read_pair_scores(path: Path) -> Tuple[List[Dict[str, str]], List[float], List[int]]:
    rows: List[Dict[str, str]] = []
    scores: List[float] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"score", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(dict(row))
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
    return rows, scores, labels


def fit_logistic_calibrator(scores: Sequence[float], labels: Sequence[int], max_iter: int = 200) -> Tuple[float, float, float, float]:
    standardized_scores, mean, std = standardize_scores(scores)
    x = torch.tensor(standardized_scores, dtype=torch.float32).view(-1, 1)
    y = torch.tensor([float(label) for label in labels], dtype=torch.float32).view(-1, 1)
    weight = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(weight.item()), float(bias.item()), mean, std


def apply_logistic_calibrator(scores: Sequence[float], weight: float, bias: float, mean: float, std: float) -> List[float]:
    standardized_scores, _, _ = standardize_scores(scores, mean=mean, std=std)
    return [sigmoid(weight * score + bias) for score in standardized_scores]


def summarize_probabilities(probabilities: Sequence[float], labels: Sequence[int], num_bins: int) -> Dict[str, float]:
    threshold, threshold_metrics = choose_best_threshold(probabilities, labels)
    metrics = compute_binary_metrics(probabilities, labels, threshold)
    metrics.update(
        {
            "probability_threshold": threshold,
            "brier_score": brier_score(probabilities, labels),
            "log_loss": binary_log_loss(probabilities, labels),
            "ece": expected_calibration_error(probabilities, labels, num_bins=num_bins),
        }
    )
    metrics["threshold_metrics_selected_on_same_split"] = threshold_metrics
    return metrics


def save_calibrated_csv(path: Path, rows: Sequence[Dict[str, str]], probabilities: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["calibrated_probability"] if rows else ["calibrated_probability"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            output = dict(row)
            output["calibrated_probability"] = f"{probability:.8f}"
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate CLIP image-text consistency scores into KG confidence probabilities.")
    parser.add_argument("--val-pairs-csv", default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--test-pairs-csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-prefix", default="clip_consistency_calibrated")
    parser.add_argument("--num-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = PROJECT_ROOT
    val_csv = resolve_repo_path(args.val_pairs_csv, repo_root)
    test_csv = resolve_repo_path(args.test_pairs_csv, repo_root)
    output_dir = resolve_repo_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows, val_scores, val_labels = read_pair_scores(val_csv)
    test_rows, test_scores, test_labels = read_pair_scores(test_csv)
    weight, bias, mean, std = fit_logistic_calibrator(val_scores, val_labels)
    val_probabilities = apply_logistic_calibrator(val_scores, weight, bias, mean, std)
    test_probabilities = apply_logistic_calibrator(test_scores, weight, bias, mean, std)

    val_summary = summarize_probabilities(val_probabilities, val_labels, num_bins=args.num_bins)
    # Test uses the probability threshold selected on validation to avoid test-set tuning.
    test_threshold = float(val_summary["probability_threshold"])
    test_summary = compute_binary_metrics(test_probabilities, test_labels, threshold=test_threshold)
    test_summary.update(
        {
            "probability_threshold": test_threshold,
            "brier_score": brier_score(test_probabilities, test_labels),
            "log_loss": binary_log_loss(test_probabilities, test_labels),
            "ece": expected_calibration_error(test_probabilities, test_labels, num_bins=args.num_bins),
        }
    )

    summary = {
        "task": "clip_consistency_probability_calibration",
        "val_pairs_csv": str(val_csv),
        "test_pairs_csv": str(test_csv),
        "calibrator": {
            "type": "platt_logistic",
            "weight": weight,
            "bias": bias,
            "score_mean": mean,
            "score_std": std,
        },
        "num_bins": args.num_bins,
        "validation": val_summary,
        "test": test_summary,
    }
    summary_path = output_dir / f"{args.save_prefix}_summary.json"
    val_out = output_dir / f"{args.save_prefix}_val_pairs.csv"
    test_out = output_dir / f"{args.save_prefix}_test_pairs.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_calibrated_csv(val_out, val_rows, val_probabilities)
    save_calibrated_csv(test_out, test_rows, test_probabilities)

    print(f"Saved calibration summary: {summary_path}")
    print(f"Saved calibrated validation pairs: {val_out}")
    print(f"Saved calibrated test pairs: {test_out}")
    print(f"Test F1={test_summary['f1']:.4f} AUC={test_summary['auc']:.4f} Brier={test_summary['brier_score']:.4f} ECE={test_summary['ece']:.4f}")


if __name__ == "__main__":
    main()
