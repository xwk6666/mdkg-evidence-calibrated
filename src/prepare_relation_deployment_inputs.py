#!/usr/bin/env python
"""Apply the deterministic disaster schema gate before frozen NLI deployment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from disaster_kg_schema import clean_entities, clean_relations


ROOT = Path(__file__).resolve().parents[1]


def normalize_sample_id(sample: Mapping[str, object]) -> str:
    return str(sample.get("sample_id") or sample.get("filename") or "").replace(
        "\\", "/"
    )


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError(f"Invalid split name: {split_name!r}")
    return normalized


def prepare_payload(
    payload: Mapping[str, object], split_name: str
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    split_name = validate_split_name(split_name)
    source_samples = payload.get("samples")
    if not isinstance(source_samples, list):
        raise ValueError("Input JSON must contain a samples list")

    prepared_samples: List[Dict[str, object]] = []
    evidence_rows: List[Dict[str, object]] = []
    counts = Counter()

    for sample in source_samples:
        if not isinstance(sample, Mapping):
            counts["invalid_samples"] += 1
            continue
        raw_entities = list(sample.get("entities") or [])
        raw_relations = list(sample.get("relations") or [])
        entities = clean_entities(raw_entities)
        relations = clean_relations(raw_relations, entities)
        sample_id = normalize_sample_id(sample)
        if not sample_id:
            counts["invalid_samples"] += 1
            continue

        converted = dict(sample)
        converted["sample_id"] = sample_id
        converted["entities"] = entities
        converted["relations"] = relations
        prepared_samples.append(converted)

        counts["samples"] += 1
        counts["raw_entities"] += len(raw_entities)
        counts["schema_valid_entities"] += len(entities)
        counts["raw_relations"] += len(raw_relations)
        counts["schema_valid_relations"] += len(relations)
        counts["samples_with_relations"] += int(bool(relations))
        counts["failed_llm_samples"] += int(
            bool(sample.get("llm_status")) and sample.get("llm_status") != "ok"
        )

        for relation in relations:
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "event_id": str(sample.get("event_id") or ""),
                    "filename": str(sample.get("filename") or ""),
                    "text": str(sample.get("text") or ""),
                    "head": relation["head"],
                    "relation": relation["relation"],
                    "tail": relation["tail"],
                    "schema_valid": True,
                    "invalid_reason": "",
                    "raw_kg_consistency_score": "",
                    "kg_consistency_score": "",
                    "kg_quality_label": "schema_valid_pending_nli",
                }
            )

    summary = {
        "split": split_name,
        "samples": counts["samples"],
        "invalid_samples": counts["invalid_samples"],
        "failed_llm_samples": counts["failed_llm_samples"],
        "raw_entities": counts["raw_entities"],
        "schema_valid_entities": counts["schema_valid_entities"],
        "dropped_entities": counts["raw_entities"] - counts["schema_valid_entities"],
        "raw_relations": counts["raw_relations"],
        "schema_valid_relations": counts["schema_valid_relations"],
        "dropped_relations": counts["raw_relations"]
        - counts["schema_valid_relations"],
        "samples_with_relations": counts["samples_with_relations"],
        "relation_types": dict(Counter(row["relation"] for row in evidence_rows)),
    }
    prepared_payload = {
        "schema_gate": "disaster_kg_schema.clean_entities+clean_relations",
        "split": split_name,
        "samples": prepared_samples,
        "statistics": summary,
    }
    return prepared_payload, evidence_rows, summary


def write_prepared_artifacts(
    output_dir: Path,
    split_name: str,
    payload: Mapping[str, object],
    evidence_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> Dict[str, Path]:
    split_name = validate_split_name(split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples": output_dir / f"{split_name}_consistency_filtered_samples.json",
        "relations": output_dir / f"{split_name}_relation_consistency.csv",
        "summary": output_dir / f"{split_name}_preparation_summary.json",
    }
    paths["samples"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "sample_id",
        "event_id",
        "filename",
        "text",
        "head",
        "relation",
        "tail",
        "schema_valid",
        "invalid_reason",
        "raw_kg_consistency_score",
        "kg_consistency_score",
        "kg_quality_label",
    ]
    with paths["relations"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evidence_rows)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare one LLM split for frozen ontology-gated NLI deployment."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.input_json.is_file():
        raise FileNotFoundError(args.input_json)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    prepared, evidence, summary = prepare_payload(payload, args.split_name)
    paths = write_prepared_artifacts(
        args.output_dir, args.split_name, prepared, evidence, summary
    )
    print(
        "Chinese summary: "
        f"split={summary['split']}, samples={summary['samples']}, "
        f"valid_relations={summary['schema_valid_relations']}, "
        f"dropped_relations={summary['dropped_relations']}"
    )
    print(f"Saved samples: {paths['samples']}")
    print(f"Saved relations: {paths['relations']}")


if __name__ == "__main__":
    main()
