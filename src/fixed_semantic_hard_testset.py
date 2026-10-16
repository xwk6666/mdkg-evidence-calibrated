import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from clip_semantic_negatives import sample_identifier
from evaluate_image_text_consistency import PairExample


def _sample_id_to_index(
    samples: Sequence[Mapping[str, object]],
) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for index, sample in enumerate(samples):
        sample_id = sample_identifier(sample)
        if not sample_id:
            raise ValueError(f"Dataset sample at index {index} has no stable sample_id.")
        if sample_id in mapping:
            raise ValueError(f"Duplicate sample_id in dataset: {sample_id}")
        mapping[sample_id] = index
    return mapping


def build_fixed_pair_manifest(
    samples: Sequence[Mapping[str, object]],
    mined_records: Sequence[Mapping[str, object]],
    split: str,
    mining_model: str,
    include_random_negative: bool = True,
    seed: int = 42,
) -> Dict[str, object]:
    id_to_index = _sample_id_to_index(samples)
    records_by_id = {
        str(record.get("sample_id", "")): record for record in mined_records
    }
    rng = random.Random(seed)
    all_sample_ids = list(id_to_index)
    pairs: List[Dict[str, object]] = []
    eligible_anchors = 0

    for anchor_id in all_sample_ids:
        record = records_by_id.get(anchor_id)
        if not record:
            continue
        negative_ids = [
            str(candidate_id)
            for candidate_id in record.get("negative_sample_ids", []) or []
            if str(candidate_id) in id_to_index and str(candidate_id) != anchor_id
        ]
        if not negative_ids:
            continue

        negative_scores = list(record.get("negative_scores", []) or [])
        semantic_id = negative_ids[0]
        semantic_score = (
            float(negative_scores[0]) if negative_scores else None
        )
        eligible_anchors += 1
        pairs.append(
            {
                "image_sample_id": anchor_id,
                "text_sample_id": anchor_id,
                "label": 1,
                "kind": "positive",
                "mining_score": None,
            }
        )

        if include_random_negative:
            random_candidates = [
                candidate_id
                for candidate_id in all_sample_ids
                if candidate_id != anchor_id and candidate_id != semantic_id
            ]
            if not random_candidates:
                random_candidates = [
                    candidate_id
                    for candidate_id in all_sample_ids
                    if candidate_id != anchor_id
                ]
            if random_candidates:
                pairs.append(
                    {
                        "image_sample_id": anchor_id,
                        "text_sample_id": rng.choice(random_candidates),
                        "label": 0,
                        "kind": "random_negative",
                        "mining_score": None,
                    }
                )

        pairs.append(
            {
                "image_sample_id": anchor_id,
                "text_sample_id": semantic_id,
                "label": 0,
                "kind": "semantic_hard_negative",
                "mining_score": semantic_score,
            }
        )

    pair_counts = Counter(pair["kind"] for pair in pairs)
    return {
        "task": "fixed_semantic_hard_testset",
        "version": 1,
        "split": split,
        "mining_model": mining_model,
        "selection_rule": (
            "top1 cosine-similar text from the same disaster type and a "
            "different event family"
        ),
        "seed": seed,
        "include_random_negative": include_random_negative,
        "num_dataset_samples": len(samples),
        "num_eligible_anchors": eligible_anchors,
        "anchor_coverage": (
            eligible_anchors / len(samples) if samples else 0.0
        ),
        "pair_counts": dict(sorted(pair_counts.items())),
        "pairs": pairs,
    }


def load_fixed_pairs(
    manifest: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
) -> List[PairExample]:
    id_to_index = _sample_id_to_index(samples)
    pairs: List[PairExample] = []
    for pair_number, record in enumerate(manifest.get("pairs", []) or []):
        image_id = str(record.get("image_sample_id", ""))
        text_id = str(record.get("text_sample_id", ""))
        if image_id not in id_to_index:
            raise ValueError(
                f"Fixed pair {pair_number} references missing image sample_id: {image_id}"
            )
        if text_id not in id_to_index:
            raise ValueError(
                f"Fixed pair {pair_number} references missing text sample_id: {text_id}"
            )
        label = int(record.get("label", -1))
        if label not in {0, 1}:
            raise ValueError(
                f"Fixed pair {pair_number} has invalid label: {label}"
            )
        kind = str(record.get("kind", "")).strip()
        if not kind:
            raise ValueError(f"Fixed pair {pair_number} has no kind.")
        pairs.append(
            PairExample(
                image_index=id_to_index[image_id],
                text_index=id_to_index[text_id],
                label=label,
                kind=kind,
            )
        )
    if not pairs:
        raise ValueError("Fixed pair manifest contains no usable pairs.")
    return pairs


def save_fixed_pair_manifest(
    manifest: Mapping[str, object],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2)


def load_fixed_pair_manifest(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Fixed pair manifest must be a JSON object.")
    return manifest
