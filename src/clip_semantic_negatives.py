import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DISASTER_TYPE_HINTS = (
    ("hurricane", "hurricane"),
    ("cyclone", "hurricane"),
    ("typhoon", "hurricane"),
    ("earthquake", "earthquake"),
    ("flood", "flood"),
    ("wildfire", "wildfire"),
    ("wildfires", "wildfire"),
)


def sample_identifier(sample: Mapping[str, object]) -> str:
    return str(sample.get("sample_id") or sample.get("filename") or "")


def infer_event_family(event_id: str) -> str:
    return str(event_id).split("__", 1)[0].strip().lower()


def infer_disaster_type(sample: Mapping[str, object]) -> str:
    for entity in sample.get("entities", []) or []:
        if not isinstance(entity, Mapping):
            continue
        if str(entity.get("type", "")).strip().lower() != "disastertype":
            continue
        name = str(entity.get("name", "")).strip().lower()
        if name:
            return name

    event_family = infer_event_family(str(sample.get("event_id", "")))
    for token, disaster_type in DISASTER_TYPE_HINTS:
        if token in event_family:
            return disaster_type
    return event_family


def _validate_similarity_matrix(samples: Sequence[Mapping[str, object]], similarity) -> np.ndarray:
    matrix = np.asarray(similarity, dtype=np.float32)
    expected_shape = (len(samples), len(samples))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Similarity matrix shape must be {expected_shape}, got {matrix.shape}."
        )
    return matrix


def select_topk_semantic_negatives(
    samples: Sequence[Mapping[str, object]],
    similarity,
    top_k: int,
) -> List[Dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    matrix = _validate_similarity_matrix(samples, similarity)
    disaster_types = [infer_disaster_type(sample) for sample in samples]
    event_families = [
        infer_event_family(str(sample.get("event_id", ""))) for sample in samples
    ]
    records: List[Dict[str, object]] = []

    for anchor_index, sample in enumerate(samples):
        candidates = [
            candidate_index
            for candidate_index in range(len(samples))
            if candidate_index != anchor_index
            and disaster_types[candidate_index] == disaster_types[anchor_index]
            and event_families[candidate_index] != event_families[anchor_index]
        ]
        candidates.sort(
            key=lambda candidate_index: (
                -float(matrix[anchor_index, candidate_index]),
                candidate_index,
            )
        )
        selected = candidates[:top_k]
        records.append(
            {
                "sample_id": sample_identifier(sample),
                "event_id": str(sample.get("event_id", "")),
                "event_family": event_families[anchor_index],
                "disaster_type": disaster_types[anchor_index],
                "negative_sample_ids": [
                    sample_identifier(samples[candidate_index])
                    for candidate_index in selected
                ],
                "negative_scores": [
                    float(matrix[anchor_index, candidate_index])
                    for candidate_index in selected
                ],
            }
        )
    return records


@torch.no_grad()
def mine_topk_from_features(
    samples: Sequence[Mapping[str, object]],
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Image and text features must be two-dimensional.")
    if image_features.shape != text_features.shape:
        raise ValueError(
            "Image and text feature matrices must have the same shape, got "
            f"{tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    if image_features.shape[0] != len(samples):
        raise ValueError(
            f"Feature count {image_features.shape[0]} does not match "
            f"sample count {len(samples)}."
        )

    image_features = F.normalize(image_features.float(), p=2, dim=-1).cpu()
    text_features = F.normalize(text_features.float(), p=2, dim=-1).cpu()
    disaster_types = [infer_disaster_type(sample) for sample in samples]
    event_families = [
        infer_event_family(str(sample.get("event_id", ""))) for sample in samples
    ]

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, disaster_type in enumerate(disaster_types):
        grouped_indices[disaster_type].append(index)

    records: List[Dict[str, object]] = [
        {
            "sample_id": sample_identifier(sample),
            "event_id": str(sample.get("event_id", "")),
            "event_family": event_families[index],
            "disaster_type": disaster_types[index],
            "negative_sample_ids": [],
            "negative_scores": [],
        }
        for index, sample in enumerate(samples)
    ]

    for group_indices in grouped_indices.values():
        if len(group_indices) < 2:
            continue

        group_text_features = text_features[group_indices].to(device)
        family_to_id: Dict[str, int] = {}
        family_ids = []
        for global_index in group_indices:
            family = event_families[global_index]
            if family not in family_to_id:
                family_to_id[family] = len(family_to_id)
            family_ids.append(family_to_id[family])
        group_family_ids = torch.tensor(family_ids, dtype=torch.long, device=device)
        available_k = min(top_k, len(group_indices))

        for start in range(0, len(group_indices), chunk_size):
            chunk_global_indices = group_indices[start : start + chunk_size]
            chunk_image_features = image_features[chunk_global_indices].to(device)
            similarities = torch.matmul(chunk_image_features, group_text_features.T)
            anchor_family_ids = group_family_ids[start : start + len(chunk_global_indices)]
            invalid_mask = anchor_family_ids[:, None] == group_family_ids[None, :]
            similarities.masked_fill_(invalid_mask, float("-inf"))

            values, positions = torch.topk(
                similarities,
                k=available_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            values = values.cpu()
            positions = positions.cpu()

            for row_index, anchor_global_index in enumerate(chunk_global_indices):
                selected_ids = []
                selected_scores = []
                for score, group_position in zip(
                    values[row_index].tolist(),
                    positions[row_index].tolist(),
                ):
                    if not np.isfinite(score):
                        continue
                    candidate_global_index = group_indices[group_position]
                    selected_ids.append(
                        sample_identifier(samples[candidate_global_index])
                    )
                    selected_scores.append(float(score))
                records[anchor_global_index]["negative_sample_ids"] = selected_ids
                records[anchor_global_index]["negative_scores"] = selected_scores

    return records


def write_semantic_negative_cache(
    records: Iterable[Mapping[str, object]],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def load_semantic_negative_cache(path: Path) -> Dict[str, Dict[str, object]]:
    cache: Dict[str, Dict[str, object]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in cache line {line_number}.")
            if sample_id in cache:
                raise ValueError(f"Duplicate sample_id in semantic cache: {sample_id}")
            negative_ids = record.get("negative_sample_ids")
            negative_scores = record.get("negative_scores")
            if not isinstance(negative_ids, list) or not isinstance(negative_scores, list):
                raise ValueError(
                    f"Invalid candidate lists in semantic cache line {line_number}."
                )
            if len(negative_ids) != len(negative_scores):
                raise ValueError(
                    f"Candidate ID/score length mismatch in cache line {line_number}."
                )
            cache[sample_id] = record
    return cache
