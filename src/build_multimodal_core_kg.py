import argparse
import csv
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from disaster_kg_schema import (
    clean_entities,
    clean_relations,
    normalize_entity_name,
    normalize_entity_type,
)

try:
    from neo4j import GraphDatabase
except ImportError:  # Dry-run mode should still work without the Neo4j driver.
    GraphDatabase = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "CrisisMMD_v2.0" / "structured_llm_core"
SPLIT_NAMES = ("train", "val", "test")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""

ENTITY_LABELS = {
    "Disaster": "Disaster",
    "Location": "Location",
    "Infrastructure": "Infrastructure",
    "Casualty": "Casualty",
}

RELATION_TYPES = {"OCCURS_IN", "AFFECTS", "CAUSES"}


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_float(row: Dict[str, str], field: str):
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float in {field}: {value!r}") from exc


def optional_int(row: Dict[str, str], field: str):
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer in {field}: {value!r}") from exc


def load_alignment_scores(path: str) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}

    score_path = Path(path)
    if not score_path.exists():
        raise FileNotFoundError(f"Alignment score file not found: {score_path}")

    scores = {}
    with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).replace("\\", "/")
            if not sample_id:
                continue
            if sample_id in scores:
                raise ValueError(f"Duplicate alignment sample_id: {sample_id}")

            alignment_score = optional_float(row, "alignment_score")
            alignment_probability = optional_float(row, "calibrated_probability")
            if alignment_probability is None:
                alignment_probability = alignment_score
            if alignment_score is None:
                alignment_score = alignment_probability

            score_type = row.get("score_type") or "legacy_score"
            if (
                score_type == "platt_calibrated_probability"
                and alignment_probability is not None
                and not 0.0 <= alignment_probability <= 1.0
            ):
                raise ValueError(
                    f"Calibrated probability must be between 0 and 1 for {sample_id}"
                )
            scores[sample_id] = {
                "alignment_score": alignment_score,
                "alignment_probability": alignment_probability,
                "raw_alignment_score": optional_float(row, "raw_alignment_score"),
                "decision_threshold": optional_float(row, "decision_threshold"),
                "alignment_confidence": (
                    row.get("evidence_label") or row.get("confidence") or None
                ),
                "alignment_model": (
                    row.get("model_name") or row.get("model") or "legacy_unspecified"
                ),
                "calibration_model": row.get("calibration_model") or None,
                "score_type": score_type,
                "i2t_rank": optional_int(row, "i2t_rank"),
                "t2i_rank": optional_int(row, "t2i_rank"),
            }
    return scores


def relation_consistency_key(sample_id: str, head: str, relation_type: str, tail: str) -> str:
    head_key = str(head or "").strip().lower()
    tail_key = str(tail or "").strip().lower()
    relation_key = str(relation_type or "").strip().upper()
    return f"{sample_id}::{head_key}::{relation_key}::{tail_key}"


def load_relation_consistency(path: str) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}

    consistency_path = Path(path)
    if not consistency_path.exists():
        raise FileNotFoundError(f"Relation consistency file not found: {consistency_path}")

    consistency = {}
    with consistency_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).replace("\\", "/")
            if not sample_id:
                continue
            key = relation_consistency_key(sample_id, row.get("head"), row.get("relation"), row.get("tail"))
            calibrated_score = row.get("calibrated_confidence")
            calibrated_label = row.get("calibrated_quality_label")
            consistency[key] = {
                "kg_consistency_score": (
                    float(calibrated_score)
                    if calibrated_score not in (None, "")
                    else (
                        float(row["kg_consistency_score"])
                        if row.get("kg_consistency_score")
                        else None
                    )
                ),
                "kg_quality_label": (
                    calibrated_label or row.get("kg_quality_label") or None
                ),
                "raw_kg_consistency_score": (
                    float(row["raw_kg_consistency_score"])
                    if row.get("raw_kg_consistency_score")
                    else (
                        float(row["kg_consistency_score"])
                        if row.get("kg_consistency_score")
                        else None
                    )
                ),
                "calibrated_confidence": (
                    float(calibrated_score)
                    if calibrated_score not in (None, "")
                    else None
                ),
                "calibrated_uncertainty": (
                    float(row["calibrated_uncertainty"])
                    if row.get("calibrated_uncertainty")
                    else None
                ),
                "confidence_model": row.get("confidence_model") or None,
                "prototype_similarity": (
                    float(row["prototype_similarity"])
                    if row.get("prototype_similarity")
                    else None
                ),
                "prototype_margin": (
                    float(row["prototype_margin"])
                    if row.get("prototype_margin")
                    else None
                ),
                "event_prototype_score": (
                    float(row["event_prototype_score"])
                    if row.get("event_prototype_score")
                    else None
                ),
                "slot_prototype_similarity": (
                    float(row["slot_prototype_similarity"])
                    if row.get("slot_prototype_similarity")
                    else None
                ),
                "slot_prototype_margin": (
                    float(row["slot_prototype_margin"])
                    if row.get("slot_prototype_margin")
                    else None
                ),
                "relation_slot_score": (
                    float(row["relation_slot_score"])
                    if row.get("relation_slot_score")
                    else None
                ),
                "event_support_count": int(row["event_support_count"]) if row.get("event_support_count") else None,
                "event_support_score": float(row["event_support_score"]) if row.get("event_support_score") else None,
                "schema_valid": str(row.get("schema_valid", "")).lower() == "true",
                "invalid_reason": row.get("invalid_reason") or None,
            }
    return consistency


def normalize_sample_id(sample: Dict[str, object]) -> str:
    raw_id = sample.get("sample_id") or sample.get("filename")
    if not raw_id:
        raise ValueError("sample is missing both sample_id and filename")

    sample_id = str(raw_id).replace("\\", "/")
    if sample_id.lower().endswith((".jpg", ".jpeg", ".png")):
        sample_id = sample_id.rsplit(".", 1)[0]
    return sample_id


def normalize_post_id(sample: Dict[str, object], sample_id: str) -> str:
    post_id = sample.get("post_id")
    if post_id:
        return str(post_id).replace("\\", "/")

    image_id = sample_id.rsplit("/", 1)[-1]
    return image_id.rsplit("_", 1)[0]


def normalize_event_name(event_id: str) -> str:
    event_name = event_id.split("__", 1)[0]
    return event_name.replace("_", " ").title()


def scoped_id(graph_scope: str, source_id: str) -> str:
    scope = str(graph_scope or "").strip()
    return f"{scope}::{source_id}" if scope else source_id


def _canonical_key_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def canonical_entity_key(entity: Dict[str, object], graph_scope: str = "") -> str:
    entity_type = str(entity.get("type") or "")
    canonical_name = normalize_entity_name(entity.get("name"), entity_type)
    source_key = f"{entity_type}::{_canonical_key_text(canonical_name)}"
    return scoped_id(graph_scope, source_key)


def entity_key(entity: Dict[str, object]) -> str:
    return canonical_entity_key(entity)


def extract_observation_date(event_id: str, image_path: str = ""):
    candidates = [str(event_id or ""), str(image_path or "").replace("\\", "/")]
    patterns = (r"__(\d{1,2})_(\d{1,2})_(\d{4})$", r"/(\d{1,2})_(\d{1,2})_(\d{4})/")
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if not match:
                continue
            day, month, year = (int(value) for value in match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                continue
    return None


def safe_entities(sample: Dict[str, object]) -> List[Dict[str, str]]:
    raw_entities = list(sample.get("entities", []) or [])
    cleaned = clean_entities(raw_entities)
    aliases_by_key = {}
    for entity in raw_entities:
        entity_type = normalize_entity_type(entity.get("type"))
        if not entity_type:
            continue
        canonical_name = normalize_entity_name(entity.get("name"), entity_type)
        if not canonical_name:
            continue
        key = (entity_type, _canonical_key_text(canonical_name))
        aliases_by_key.setdefault(key, [])
        surface = str(entity.get("name") or "").strip()
        if surface and surface not in aliases_by_key[key]:
            aliases_by_key[key].append(surface)

    entities = []
    for entity in cleaned:
        key = (entity["type"], _canonical_key_text(entity["name"]))
        aliases = aliases_by_key.get(key, [])
        if entity["name"] not in aliases:
            aliases.append(entity["name"])
        entities.append({**entity, "aliases": aliases})
    return entities


def safe_relations(
    sample: Dict[str, object],
    entities: Iterable[Dict[str, str]],
) -> List[Tuple[Dict[str, str], str, Dict[str, str]]]:
    entities = list(entities)
    by_name = {_canonical_key_text(entity["name"]): entity for entity in entities}
    relations = []
    seen = set()

    for relation in clean_relations(sample.get("relations", []) or [], entities):
        relation_type = relation["relation"]
        head = by_name.get(_canonical_key_text(relation["head"]))
        tail = by_name.get(_canonical_key_text(relation["tail"]))
        if not head or not tail:
            continue

        key = (entity_key(head), relation_type, entity_key(tail))
        if key in seen:
            continue
        seen.add(key)
        relations.append((head, relation_type, tail))

    return relations


class MultimodalCoreKGBuilder:
    def __init__(self, uri: str, user: str, password: str, graph_scope: str = ""):
        if GraphDatabase is None:
            raise RuntimeError("Missing Python package: neo4j. Install it with: pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph_scope = str(graph_scope or "").strip()

    def close(self) -> None:
        self.driver.close()

    def clear_database(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self) -> None:
        specs = [
            ("disaster_event_id", "DisasterEvent", "id"),
            ("disaster_sample_id", "DisasterSample", "id"),
            ("social_post_id", "SocialPost", "id"),
            ("disaster_text_id", "DisasterText", "id"),
            ("disaster_image_id", "DisasterImage", "id"),
            ("kg_entity_key", "KGEntity", "key"),
            ("observation_date_key", "ObservationDate", "key"),
        ]
        with self.driver.session() as session:
            for name, label, property_name in specs:
                self._create_unique_constraint(session, name, label, property_name)

    @staticmethod
    def _create_unique_constraint(session, name: str, label: str, property_name: str) -> None:
        modern_query = (
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        )
        legacy_query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{property_name} IS UNIQUE"
        try:
            session.run(modern_query).consume()
            return
        except Exception:
            pass

        try:
            session.run(legacy_query).consume()
        except Exception as exc:
            message = str(exc)
            if "already exists" in message.lower() or "equivalent" in message.lower():
                return
            raise

    def import_sample(
        self,
        sample: Dict[str, object],
        split: str,
        alignment_score: Dict[str, object] = None,
        relation_consistency: Dict[str, Dict[str, object]] = None,
    ) -> Dict[str, int]:
        source_sample_id = normalize_sample_id(sample)
        source_post_id = normalize_post_id(sample, source_sample_id)
        source_event_id = str(sample.get("event_id") or "unknown_event")
        image_path = str(sample.get("filename") or "")
        text = str(sample.get("text") or "")
        event_name = normalize_event_name(source_event_id)
        source_image_id = (
            image_path.replace("\\", "/") if image_path else source_sample_id
        )
        sample_id = scoped_id(self.graph_scope, source_sample_id)
        post_id = scoped_id(self.graph_scope, source_post_id)
        event_id = scoped_id(self.graph_scope, source_event_id)
        image_id = scoped_id(self.graph_scope, source_image_id)
        observation_date = extract_observation_date(source_event_id, image_path)

        entities = safe_entities(sample)
        relations = safe_relations(sample, entities)
        alignment_score = alignment_score or {}
        relation_consistency = relation_consistency or {}

        relation_records = []
        for head, relation_type, tail in relations:
            consistency_key = relation_consistency_key(
                source_sample_id, head["name"], relation_type, tail["name"]
            )
            relation_records.append(
                (
                    head,
                    relation_type,
                    tail,
                    relation_consistency.get(consistency_key, {}),
                )
            )

        def write_sample_bundle(tx) -> None:
            self._merge_multimodal_sample(
                tx,
                split,
                sample_id,
                post_id,
                event_id,
                event_name,
                image_id,
                image_path,
                text,
                alignment_score.get("alignment_score"),
                alignment_score.get("alignment_confidence"),
                alignment_score.get("i2t_rank"),
                alignment_score.get("t2i_rank"),
                alignment_score.get("raw_alignment_score"),
                alignment_score.get("alignment_probability"),
                alignment_score.get("decision_threshold"),
                alignment_score.get("alignment_model"),
                alignment_score.get("calibration_model"),
                alignment_score.get("score_type"),
                self.graph_scope,
                source_sample_id,
                source_post_id,
                source_event_id,
                source_image_id,
                observation_date,
                scoped_id(self.graph_scope, observation_date)
                if observation_date
                else None,
            )
            for entity in entities:
                self._merge_entity_mention(
                    tx,
                    sample_id,
                    entity,
                    self.graph_scope,
                )
            for head, relation_type, tail, consistency in relation_records:
                self._merge_semantic_relation(
                    tx,
                    source_sample_id,
                    head,
                    relation_type,
                    tail,
                    consistency,
                    self.graph_scope,
                    split,
                )

        with self.driver.session() as session:
            session.execute_write(write_sample_bundle)

        return {
            "samples": 1,
            "entities": len(entities),
            "relations": len(relations),
            "empty_entities": int(not entities),
            "empty_relations": int(not relations),
        }

    @staticmethod
    def _merge_multimodal_sample(
        tx,
        split: str,
        sample_id: str,
        post_id: str,
        event_id: str,
        event_name: str,
        image_id: str,
        image_path: str,
        text: str,
        alignment_score: float,
        alignment_confidence: str,
        i2t_rank: int,
        t2i_rank: int,
        raw_alignment_score: float = None,
        alignment_probability: float = None,
        decision_threshold: float = None,
        alignment_model: str = "legacy_unspecified",
        calibration_model: str = None,
        score_type: str = None,
        graph_scope: str = "",
        source_sample_id: str = None,
        source_post_id: str = None,
        source_event_id: str = None,
        source_image_id: str = None,
        observation_date: str = None,
        observation_date_key: str = None,
    ) -> None:
        tx.run(
            """
            MERGE (event:DisasterEvent {id: $event_id})
            SET event.name = $event_name,
                event.source_id = $source_event_id,
                event.graph_scope = $graph_scope,
                event.observation_date = $observation_date

            MERGE (post:SocialPost {id: $post_id})
            SET post.event_id = $event_id,
                post.source_id = $source_post_id,
                post.graph_scope = $graph_scope,
                post.observation_date = $observation_date

            MERGE (sample:DisasterSample {id: $sample_id})
            SET sample.split = $split,
                sample.event_id = $event_id,
                sample.post_id = $post_id,
                sample.source_id = $source_sample_id,
                sample.graph_scope = $graph_scope,
                sample.observation_date = $observation_date

            MERGE (text:DisasterText {id: $sample_id})
            SET text.content = $text,
                text.split = $split,
                text.source_id = $source_sample_id,
                text.graph_scope = $graph_scope

            MERGE (image:DisasterImage {id: $image_id})
            SET image.path = $image_path,
                image.split = $split,
                image.source_id = $source_image_id,
                image.graph_scope = $graph_scope

            MERGE (event)-[:HAS_SAMPLE]->(sample)
            MERGE (post)-[:HAS_SAMPLE]->(sample)
            MERGE (sample)-[:HAS_TEXT]->(text)
            MERGE (sample)-[:HAS_IMAGE]->(image)
            FOREACH (_ IN CASE WHEN $observation_date_key IS NULL THEN [] ELSE [1] END |
                MERGE (observation:ObservationDate {key: $observation_date_key})
                SET observation.value = $observation_date,
                    observation.graph_scope = $graph_scope
                MERGE (event)-[:OBSERVED_AT]->(observation)
                MERGE (post)-[:OBSERVED_AT]->(observation)
                MERGE (sample)-[:OBSERVED_AT]->(observation)
            )
            FOREACH (_ IN CASE WHEN $alignment_probability IS NULL THEN [] ELSE [1] END |
                MERGE (text)-[txt_img:ALIGNED_WITH]->(image)
                SET txt_img.score = $alignment_probability,
                    txt_img.raw_score = $raw_alignment_score,
                    txt_img.calibrated_probability = $alignment_probability,
                    txt_img.threshold = $decision_threshold,
                    txt_img.confidence = $alignment_confidence,
                    txt_img.i2t_rank = $i2t_rank,
                    txt_img.t2i_rank = $t2i_rank,
                    txt_img.model = $alignment_model,
                    txt_img.calibration_model = $calibration_model,
                    txt_img.score_type = $score_type,
                    txt_img.graph_scope = $graph_scope

                MERGE (image)-[img_txt:ALIGNED_WITH]->(text)
                SET img_txt.score = $alignment_probability,
                    img_txt.raw_score = $raw_alignment_score,
                    img_txt.calibrated_probability = $alignment_probability,
                    img_txt.threshold = $decision_threshold,
                    img_txt.confidence = $alignment_confidence,
                    img_txt.i2t_rank = $i2t_rank,
                    img_txt.t2i_rank = $t2i_rank,
                    img_txt.model = $alignment_model,
                    img_txt.calibration_model = $calibration_model,
                    img_txt.score_type = $score_type,
                    img_txt.graph_scope = $graph_scope
            )
            """,
            split=split,
            sample_id=sample_id,
            post_id=post_id,
            event_id=event_id,
            event_name=event_name,
            image_id=image_id,
            image_path=image_path,
            text=text,
            alignment_score=alignment_score,
            raw_alignment_score=raw_alignment_score,
            alignment_probability=alignment_probability,
            decision_threshold=decision_threshold,
            alignment_confidence=alignment_confidence,
            i2t_rank=i2t_rank,
            t2i_rank=t2i_rank,
            alignment_model=alignment_model,
            calibration_model=calibration_model,
            score_type=score_type,
            graph_scope=graph_scope,
            source_sample_id=source_sample_id or sample_id,
            source_post_id=source_post_id or post_id,
            source_event_id=source_event_id or event_id,
            source_image_id=source_image_id or image_id,
            observation_date=observation_date,
            observation_date_key=observation_date_key,
        )

    @staticmethod
    def _merge_entity_mention(
        tx,
        sample_id: str,
        entity: Dict[str, object],
        graph_scope: str = "",
    ) -> None:
        label = ENTITY_LABELS[entity["type"]]
        tx.run(
            f"""
            MERGE (entity:KGEntity:{label} {{key: $entity_key}})
            SET entity.name = $name,
                entity.type = $type,
                entity.graph_scope = $graph_scope,
                entity.aliases = reduce(
                    accumulated = coalesce(entity.aliases, []),
                    alias IN $aliases |
                    CASE
                        WHEN alias IN accumulated THEN accumulated
                        ELSE accumulated + alias
                    END
                )

            WITH entity
            MATCH (sample:DisasterSample {{id: $sample_id}})
            MATCH (text:DisasterText {{id: $sample_id}})
            MERGE (sample)-[:MENTIONS]->(entity)
            MERGE (text)-[:MENTIONS]->(entity)
            """,
            sample_id=sample_id,
            entity_key=canonical_entity_key(entity, graph_scope),
            name=entity["name"],
            type=entity["type"],
            aliases=list(entity.get("aliases") or [entity["name"]]),
            graph_scope=graph_scope,
        )

    @staticmethod
    def _merge_semantic_relation(
        tx,
        sample_id: str,
        head: Dict[str, str],
        relation_type: str,
        tail: Dict[str, str],
        consistency: Dict[str, object],
        graph_scope: str = "",
        split: str = "",
    ) -> None:
        tx.run(
            f"""
            MATCH (head:KGEntity {{key: $head_key}})
            MATCH (tail:KGEntity {{key: $tail_key}})
            MERGE (head)-[rel:{relation_type}]->(tail)
            WITH rel,
                 coalesce(rel.sample_ids, []) AS old_ids,
                 coalesce(rel.kg_consistency_scores, []) AS old_scores,
                 coalesce(rel.kg_quality_labels, []) AS old_labels,
                 coalesce(rel.event_prototype_scores, []) AS old_prototype_scores
            WITH rel,
                 CASE WHEN $sample_id IN old_ids THEN old_ids ELSE old_ids + $sample_id END AS new_ids,
                 CASE
                    WHEN $sample_id IN old_ids OR $kg_consistency_score IS NULL THEN old_scores
                    ELSE old_scores + $kg_consistency_score
                 END AS new_scores,
                 CASE
                    WHEN $sample_id IN old_ids OR $kg_quality_label IS NULL THEN old_labels
                    ELSE old_labels + $kg_quality_label
                 END AS new_labels,
                 CASE
                    WHEN $sample_id IN old_ids OR $event_prototype_score IS NULL
                    THEN old_prototype_scores
                    ELSE old_prototype_scores + $event_prototype_score
                 END AS new_prototype_scores
            SET rel.sample_ids = new_ids,
                rel.evidence_count = size(new_ids),
                rel.kg_consistency_scores = new_scores,
                rel.kg_quality_labels = new_labels,
                rel.event_prototype_scores = new_prototype_scores,
                rel.kg_consistency_score = CASE
                    WHEN size(new_scores) = 0 THEN NULL
                    ELSE reduce(total = 0.0, score IN new_scores | total + score) / size(new_scores)
                END,
                rel.max_kg_consistency_score = CASE
                    WHEN size(new_scores) = 0 THEN NULL
                    ELSE reduce(max_score = 0.0, score IN new_scores |
                        CASE WHEN score > max_score THEN score ELSE max_score END)
                END,
                rel.confidence = CASE
                    WHEN size(new_scores) = 0 THEN NULL
                    ELSE reduce(max_score = 0.0, score IN new_scores |
                        CASE WHEN score > max_score THEN score ELSE max_score END)
                END,
                rel.uncertainty = CASE
                    WHEN size(new_scores) = 0 THEN NULL
                    ELSE 1.0 - reduce(max_score = 0.0, score IN new_scores |
                        CASE WHEN score > max_score THEN score ELSE max_score END)
                END,
                rel.quality_label = CASE
                    WHEN size([label IN new_labels WHERE label = 'reliable']) > 0 THEN 'reliable'
                    WHEN size([label IN new_labels WHERE label = 'review']) > 0 THEN 'review'
                    ELSE NULL
                END,
                rel.reliable_evidence_count = size([label IN new_labels WHERE label = 'reliable']),
                rel.review_evidence_count = size([label IN new_labels WHERE label = 'review']),
                rel.raw_kg_consistency_score = $raw_kg_consistency_score,
                rel.calibrated_confidence = $calibrated_confidence,
                rel.calibrated_uncertainty = $calibrated_uncertainty,
                rel.confidence_model = $confidence_model,
                rel.prototype_similarity = $prototype_similarity,
                rel.prototype_margin = $prototype_margin,
                rel.slot_prototype_similarity = $slot_prototype_similarity,
                rel.slot_prototype_margin = $slot_prototype_margin,
                rel.relation_slot_score = $relation_slot_score,
                rel.event_prototype_score = CASE
                    WHEN size(new_prototype_scores) = 0 THEN NULL
                    ELSE reduce(total = 0.0, score IN new_prototype_scores |
                        total + score) / size(new_prototype_scores)
                END,
                rel.event_support_count = $event_support_count,
                rel.event_support_score = $event_support_score,
                rel.schema_valid = $schema_valid,
                rel.invalid_reason = $invalid_reason,
                rel.graph_scope = $graph_scope,
                rel.splits = CASE
                    WHEN $split = '' OR $split IN coalesce(rel.splits, [])
                    THEN coalesce(rel.splits, [])
                    ELSE coalesce(rel.splits, []) + $split
                END
            """,
            sample_id=sample_id,
            head_key=canonical_entity_key(head, graph_scope),
            tail_key=canonical_entity_key(tail, graph_scope),
            kg_consistency_score=consistency.get("kg_consistency_score"),
            kg_quality_label=consistency.get("kg_quality_label"),
            raw_kg_consistency_score=consistency.get("raw_kg_consistency_score"),
            calibrated_confidence=consistency.get("calibrated_confidence"),
            calibrated_uncertainty=consistency.get("calibrated_uncertainty"),
            confidence_model=consistency.get("confidence_model"),
            prototype_similarity=consistency.get("prototype_similarity"),
            prototype_margin=consistency.get("prototype_margin"),
            event_prototype_score=consistency.get("event_prototype_score"),
            slot_prototype_similarity=consistency.get("slot_prototype_similarity"),
            slot_prototype_margin=consistency.get("slot_prototype_margin"),
            relation_slot_score=consistency.get("relation_slot_score"),
            event_support_count=consistency.get("event_support_count"),
            event_support_score=consistency.get("event_support_score"),
            schema_valid=consistency.get("schema_valid"),
            invalid_reason=consistency.get("invalid_reason"),
            graph_scope=graph_scope,
            split=split,
        )


def should_skip_sample(sample: Dict[str, object], include_failed: bool) -> bool:
    status = sample.get("llm_status")
    return bool(status and status != "ok" and not include_failed)


def import_split(
    builder: MultimodalCoreKGBuilder,
    input_path: Path,
    split: str,
    limit: int,
    include_failed: bool,
    dry_run: bool,
    alignment_scores: Dict[str, Dict[str, object]],
    relation_consistency: Dict[str, Dict[str, object]],
) -> Counter:
    payload = load_json(input_path)
    samples = payload.get("samples", [])
    counts = Counter()

    for sample in samples:
        if should_skip_sample(sample, include_failed):
            counts["skipped_failed"] += 1
            continue
        if limit > 0 and counts["samples"] >= limit:
            break

        entities = safe_entities(sample)
        relations = safe_relations(sample, entities)
        sample_id = normalize_sample_id(sample)
        if sample_id in alignment_scores:
            counts["alignment_scored"] += 1
        else:
            counts["alignment_missing"] += 1

        for head, relation_type, tail in relations:
            consistency_key = relation_consistency_key(
                sample_id,
                head["name"],
                relation_type,
                tail["name"],
            )
            if consistency_key in relation_consistency:
                counts["relation_confidence_matched"] += 1
            else:
                counts["relation_confidence_missing"] += 1
        counts.update(
            {
                "samples": 1,
                "entities": len(entities),
                "relations": len(relations),
                "empty_entities": int(not entities),
                "empty_relations": int(not relations),
            }
        )

        if not dry_run:
            builder.import_sample(
                sample,
                split,
                alignment_scores.get(sample_id),
                relation_consistency,
            )

    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Build a multimodal core disaster KG in Neo4j.")
    parser.add_argument("--split", choices=[*SPLIT_NAMES, "all"], default="test")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Use 0 to import the full split.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI") or NEO4J_URI)
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER") or NEO4J_USER)
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD") or NEO4J_PASSWORD)
    parser.add_argument("--clear", action="store_true", help="Dangerous: delete all existing Neo4j nodes first.")
    parser.add_argument("--include-failed", action="store_true", help="Also import samples whose llm_status is not ok.")
    parser.add_argument("--alignment-scores", default="", help="Optional CSV from compute_alignment_scores.py.")
    parser.add_argument("--relation-consistency", default="", help="Optional CSV from evaluate_kg_consistency.py.")
    parser.add_argument("--dry-run", action="store_true", help="Only count samples/entities/relations; do not write Neo4j.")
    parser.add_argument(
        "--graph-scope",
        default="",
        help=(
            "Optional namespace for an isolated graph copy, e.g. full_corpus_v1. "
            "Source IDs remain available as node properties."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.neo4j_password:
        raise RuntimeError("Missing Neo4j password. Pass --neo4j-password or set $env:NEO4J_PASSWORD.")

    input_dir = Path(args.input_dir)
    splits = SPLIT_NAMES if args.split == "all" else (args.split,)
    alignment_scores = load_alignment_scores(args.alignment_scores)
    if alignment_scores:
        print(f"Loaded alignment scores: {len(alignment_scores)}")
    relation_consistency = load_relation_consistency(args.relation_consistency)
    if relation_consistency:
        print(f"Loaded relation consistency rows: {len(relation_consistency)}")

    builder = None
    if not args.dry_run:
        builder = MultimodalCoreKGBuilder(
            args.neo4j_uri,
            args.neo4j_user,
            args.neo4j_password,
            graph_scope=args.graph_scope,
        )
        if args.clear:
            builder.clear_database()
        builder.create_constraints()

    try:
        total = Counter()
        for split in splits:
            input_path = input_dir / f"{split}_entities.json"
            if not input_path.exists():
                print(f"Skip missing split file: {input_path}")
                continue

            counts = import_split(
                builder=builder,
                input_path=input_path,
                split=split,
                limit=args.limit,
                include_failed=args.include_failed,
                dry_run=args.dry_run,
                alignment_scores=alignment_scores,
                relation_consistency=relation_consistency,
            )
            total.update(counts)
            print(
                f"{split}: samples={counts['samples']} | entities={counts['entities']} | "
                f"relations={counts['relations']} | empty_entities={counts['empty_entities']} | "
                f"empty_relations={counts['empty_relations']} | skipped_failed={counts['skipped_failed']} | "
                f"alignment_scored={counts['alignment_scored']} | "
                f"alignment_missing={counts['alignment_missing']} | "
                f"relation_confidence_matched={counts['relation_confidence_matched']} | "
                f"relation_confidence_missing={counts['relation_confidence_missing']}"
            )

        action = "Checked" if args.dry_run else "Imported"
        if args.graph_scope:
            print(f"Graph scope: {args.graph_scope}")
        print(
            f"{action} total: samples={total['samples']} | entities={total['entities']} | "
            f"relations={total['relations']} | empty_entities={total['empty_entities']} | "
            f"empty_relations={total['empty_relations']} | skipped_failed={total['skipped_failed']} | "
            f"alignment_scored={total['alignment_scored']} | "
            f"alignment_missing={total['alignment_missing']} | "
            f"relation_confidence_matched={total['relation_confidence_matched']} | "
            f"relation_confidence_missing={total['relation_confidence_missing']}"
        )
    finally:
        if builder is not None:
            builder.close()


if __name__ == "__main__":
    main()
