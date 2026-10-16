import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple


CORE_ENTITY_TYPES = ("Disaster", "Location", "Infrastructure", "Casualty")
CORE_RELATION_TYPES = ("OCCURS_IN", "AFFECTS", "CAUSES")
EXTRACTION_MODES = ("conservative", "candidate")

RELATION_SIGNATURES = {
    "OCCURS_IN": ("Disaster", "Location"),
    "AFFECTS": ("Disaster", "Infrastructure"),
    "CAUSES": ("Disaster", "Casualty"),
}

ENTITY_TYPE_ALIASES = {
    "DisasterType": "Disaster",
    "Disaster_Type": "Disaster",
    "Place": "Location",
    "Geo": "Location",
    "Facility": "Infrastructure",
    "Facilities": "Infrastructure",
    "Building": "Infrastructure",
    "Buildings": "Infrastructure",
    "House": "Infrastructure",
    "Houses": "Infrastructure",
    "Home": "Infrastructure",
    "Homes": "Infrastructure",
    "Road": "Infrastructure",
    "Roads": "Infrastructure",
    "Bridge": "Infrastructure",
    "Bridges": "Infrastructure",
    "School": "Infrastructure",
    "Schools": "Infrastructure",
    "Hospital": "Infrastructure",
    "Hospitals": "Infrastructure",
    "Business": "Infrastructure",
    "Businesses": "Infrastructure",
    "Church": "Infrastructure",
    "Churches": "Infrastructure",
    "Seawall": "Infrastructure",
    "Seawalls": "Infrastructure",
    "Block": "Infrastructure",
    "Blocks": "Infrastructure",
    "Pool": "Infrastructure",
    "Pools": "Infrastructure",
    "Boat": "Infrastructure",
    "Boats": "Infrastructure",
    "Car": "Infrastructure",
    "Cars": "Infrastructure",
    "Archive": "Infrastructure",
    "Archives": "Infrastructure",
    "Record": "Infrastructure",
    "Records": "Infrastructure",
    "Utility": "Infrastructure",
    "Utilities": "Infrastructure",
    "Terminal": "Infrastructure",
    "Terminals": "Infrastructure",
    "Crop": "Infrastructure",
    "Crops": "Infrastructure",
    "Beach": "Infrastructure",
    "Beaches": "Infrastructure",
    "Damage": "Infrastructure",
    "Victim": "Casualty",
    "Victims": "Casualty",
    "Fatality": "Casualty",
    "Fatalities": "Casualty",
    "Death": "Casualty",
    "Deaths": "Casualty",
    "Injury": "Casualty",
    "Injured": "Casualty",
    "Missing": "Casualty",
    "Rescued": "Casualty",
    "Rescue": "Casualty",
    "Evacuee": "Casualty",
    "Evacuees": "Casualty",
    "Patient": "Casualty",
    "Patients": "Casualty",
    "Survivor": "Casualty",
    "Survivors": "Casualty",
}

RELATION_ALIASES = {
    "LOCATED_IN": "OCCURS_IN",
    "OCCURS_AT": "OCCURS_IN",
    "HAPPENS_IN": "OCCURS_IN",
    "HIT": "OCCURS_IN",
    "HITS": "OCCURS_IN",
    "STRUCK": "OCCURS_IN",
    "STRIKES": "OCCURS_IN",
    "RAVAGED": "OCCURS_IN",
    "CAUSES_DAMAGE": "AFFECTS",
    "CAUSED_DAMAGE": "AFFECTS",
    "DAMAGES": "AFFECTS",
    "DAMAGED": "AFFECTS",
    "AFFECTED": "AFFECTS",
    "IMPACTS": "AFFECTS",
    "IMPACTED": "AFFECTS",
    "DESTROYS": "AFFECTS",
    "DESTROYED": "AFFECTS",
    "COLLAPSES": "AFFECTS",
    "COLLAPSED": "AFFECTS",
    "CLOSES": "AFFECTS",
    "CLOSED": "AFFECTS",
    "EVACUATES": "AFFECTS",
    "EVACUATED": "AFFECTS",
    "BURNS": "AFFECTS",
    "BURNED": "AFFECTS",
    "LOST": "AFFECTS",
    "WASHED_OUT": "AFFECTS",
    "ERODES": "AFFECTS",
    "ERODED": "AFFECTS",
    "CAUSES_CASUALTY": "CAUSES",
    "CAUSED_CASUALTY": "CAUSES",
    "KILLS": "CAUSES",
    "KILLED": "CAUSES",
    "INJURES": "CAUSES",
    "INJURED": "CAUSES",
    "CLAIMS": "CAUSES",
    "CLAIMED": "CAUSES",
    "RESCUES": "CAUSES",
    "RESCUED": "CAUSES",
    "EVACUATES_PEOPLE": "CAUSES",
    "LEAVES_DEAD": "CAUSES",
    "RESULTS_IN": "CAUSES",
}

RELATION_BY_TAIL_TYPE = {
    "Location": "OCCURS_IN",
    "Infrastructure": "AFFECTS",
    "Casualty": "CAUSES",
}


DISASTER_NAME_ALIASES = {
    "hurricaneharvey": "Hurricane Harvey",
    "hurricane harvey": "Hurricane Harvey",
    "harvey": "Hurricane Harvey",
    "hurricaneirma": "Hurricane Irma",
    "hurricane irma": "Hurricane Irma",
    "irma": "Hurricane Irma",
    "hurricanemaria": "Hurricane Maria",
    "hurricane maria": "Hurricane Maria",
    "maria": "Hurricane Maria",
    "mexicoearthquake": "Mexico Earthquake",
    "mexico earthquake": "Mexico Earthquake",
    "californiawildfires": "California Wildfires",
    "california wildfires": "California Wildfires",
}

DISASTER_TYPE_ALIASES = {
    "wildfires": "Wildfire",
    "wildfire": "Wildfire",
    "floods": "Flood",
    "flood": "Flood",
    "earthquake": "Earthquake",
    "hurricane": "Hurricane",
    "cyclone": "Cyclone",
}


_POWER_SERVICE_IMPACT_PATTERN = re.compile(
    r"\b(?:restore(?:d|s|ing)?|without|lost|loses?|cuts?|cutting|lack(?:ing)?|"
    r"disrupt(?:ed|s|ion)|no)\s+(?:the\s+)?power\b|"
    r"\bpower\s+(?:restor(?:ed|ation)|outage|outages|loss|cut|cuts|disruption)\b",
    re.IGNORECASE,
)

_EXPLICIT_DISASTER_HEAD_PATTERN = re.compile(
    r"\b(?:earthquake|quake|hurricane|cyclone|typhoon|tornado|"
    r"(?:wild|bush|forest\s+)?fire|flood|tsunami|landslide|mudslide|"
    r"avalanche|volcano|volcanic\s+eruption|eruption|drought|blizzard|storm)\b",
    re.IGNORECASE,
)


def _has_explicit_disaster_head(entities: Iterable[Dict[str, str]]) -> bool:
    return any(
        entity["type"] == "Disaster"
        and _EXPLICIT_DISASTER_HEAD_PATTERN.search(entity["name"])
        for entity in entities
    )


def split_hashtag_camel_case(text: str) -> str:
    text = text.replace("#", "")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return text


def normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = split_hashtag_camel_case(str(name).strip())
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip(" .,;:!?\"'")
    return text


def normalize_entity_name(name: object, entity_type: str) -> str:
    text = normalize_name(name)
    key = text.lower()
    if entity_type == "Disaster":
        return DISASTER_NAME_ALIASES.get(key) or DISASTER_TYPE_ALIASES.get(key) or text
    return text


def normalize_entity_type(entity_type: object) -> str:
    normalized = normalize_name(entity_type)
    normalized = ENTITY_TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in CORE_ENTITY_TYPES else ""


def normalize_relation_type(relation_type: object) -> str:
    if relation_type is None:
        return ""
    normalized = str(relation_type).strip()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").upper()
    normalized = RELATION_ALIASES.get(normalized, normalized)
    return normalized if normalized in CORE_RELATION_TYPES else ""


def clean_entities(entities: Iterable[Dict[str, object]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    seen = set()

    for entity in entities or []:
        entity_type = normalize_entity_type(entity.get("type"))
        name = normalize_entity_name(entity.get("name"), entity_type)
        if not name or not entity_type:
            continue

        key = (name.lower(), entity_type)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"name": name, "type": entity_type})

    return cleaned


def _entity_type_map(entities: Iterable[Dict[str, str]]) -> Dict[str, str]:
    return {entity["name"].lower(): entity["type"] for entity in entities}


def _entity_name_map(entities: Iterable[Dict[str, str]]) -> Dict[str, str]:
    mapping = {}
    for entity in entities:
        name = entity["name"]
        entity_type = entity["type"]
        mapping[name.lower()] = name
        normalized_name = normalize_entity_name(name, entity_type)
        mapping[normalized_name.lower()] = name
    return mapping


def clean_relations(
    relations: Iterable[Dict[str, object]],
    entities: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    seen = set()
    entities = list(entities or [])
    type_map = _entity_type_map(entities)
    name_map = _entity_name_map(entities)

    for relation in relations or []:
        relation_type = normalize_relation_type(relation.get("relation"))
        raw_head = normalize_entity_name(relation.get("head"), "Disaster")
        raw_tail = normalize_name(relation.get("tail"))
        head = name_map.get(raw_head.lower(), raw_head)
        tail = name_map.get(raw_tail.lower(), raw_tail)
        if not head or not tail or not relation_type:
            continue

        expected_head_type, expected_tail_type = RELATION_SIGNATURES[relation_type]
        if type_map.get(head.lower()) != expected_head_type:
            continue
        if type_map.get(tail.lower()) != expected_tail_type:
            continue

        key = (head.lower(), relation_type, tail.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"head": head, "relation": relation_type, "tail": tail})

    return cleaned


def build_relations_from_entities(entities: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    entities = list(entities or [])
    disaster_heads = [entity["name"] for entity in entities if entity["type"] == "Disaster"]
    if not disaster_heads:
        return []

    head = disaster_heads[0]
    relations = []
    for entity in entities:
        relation_type = RELATION_BY_TAIL_TYPE.get(entity["type"])
        if not relation_type:
            continue
        relations.append({"head": head, "relation": relation_type, "tail": entity["name"]})
    return clean_relations(relations, entities)


def build_candidate_relations_from_entities(
    entities: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build every type-valid disaster relation for recall-first candidate verification."""
    entities = list(entities or [])
    disaster_heads = [entity["name"] for entity in entities if entity["type"] == "Disaster"]
    relations = []
    for head in disaster_heads:
        for entity in entities:
            relation_type = RELATION_BY_TAIL_TYPE.get(entity["type"])
            if relation_type:
                relations.append(
                    {"head": head, "relation": relation_type, "tail": entity["name"]}
                )
    return clean_relations(relations, entities)


def validate_extraction(
    extraction: Dict[str, object],
    infer_missing_relations: bool = False,
    extraction_mode: str = "conservative",
    source_text: str = "",
) -> Dict[str, List[Dict[str, str]]]:
    if extraction_mode not in EXTRACTION_MODES:
        raise ValueError(
            f"Unknown extraction mode {extraction_mode!r}; choose one of {EXTRACTION_MODES}."
        )
    entities = clean_entities(extraction.get("entities", []))
    if (
        extraction_mode == "candidate"
        and _POWER_SERVICE_IMPACT_PATTERN.search(str(source_text or ""))
        and _has_explicit_disaster_head(entities)
        and not any(
            entity["type"] == "Infrastructure" and entity["name"].casefold() == "power"
            for entity in entities
        )
    ):
        entities.append({"name": "power", "type": "Infrastructure"})
    relations = clean_relations(extraction.get("relations", []), entities)
    if extraction_mode == "candidate":
        generated = build_candidate_relations_from_entities(entities)
        relations = clean_relations([*relations, *generated], entities)
    elif infer_missing_relations and not relations:
        relations = build_relations_from_entities(entities)
    return {"entities": entities, "relations": relations}


def summarize_samples(samples: Iterable[Dict[str, object]]) -> Dict[str, object]:
    entity_counter = Counter()
    relation_counter = Counter()
    no_entity = 0
    no_relation = 0

    for sample in samples:
        entities = sample.get("entities", [])
        relations = sample.get("relations", [])
        if not entities:
            no_entity += 1
        if not relations:
            no_relation += 1
        for entity in entities:
            entity_counter[entity.get("type", "Unknown")] += 1
        for relation in relations:
            relation_counter[relation.get("relation", "UNKNOWN")] += 1

    return {
        "entity_types": dict(entity_counter),
        "relation_types": dict(relation_counter),
        "samples_without_entities": no_entity,
        "samples_without_relations": no_relation,
    }


def _build_candidate_llm_prompt(text: str) -> str:
    schema = {
        "allowed_entity_types": list(CORE_ENTITY_TYPES),
        "allowed_relation_types": list(CORE_RELATION_TYPES),
        "valid_relation_signatures": [
            {"relation": relation, "head_type": head, "tail_type": tail}
            for relation, (head, tail) in RELATION_SIGNATURES.items()
        ],
    }
    return f"""
You are a recall-first candidate generator for a disaster knowledge graph.
Your output is not accepted directly as fact: a downstream verifier will audit every candidate.
Prioritize recall while remaining grounded in the given text.

Extract every explicit or text-normalizable entity that belongs to the closed schema:
- Disaster: named disaster events or clear disaster types.
- Location: countries, regions, cities, counties, islands, coasts, and other place mentions.
- Infrastructure: homes, roads, bridges, schools, hospitals, utilities, power/electricity networks, crops, vehicles, and other affected physical assets.
- Casualty: deaths, injured, missing, trapped, homeless, displaced, evacuated, rescued people, victims, survivors, patients, or quantified affected people.

Candidate relation signatures:
- OCCURS_IN: Disaster -> Location
- AFFECTS: Disaster -> Infrastructure
- CAUSES: Disaster -> Casualty

Recall-first rules:
- Extract all supported entity spans even when the sentence discusses relief, donations, recovery, warnings, news, or attribution.
- Recover an explicit place embedded in a named event when linguistically clear, for example Mexico Earthquake -> Mexico and California Wildfires -> California.
- Normalize clear hashtags and short event mentions, for example #HurricaneHarvey or Harvey in hurricane context -> Hurricane Harvey.
- Preserve casualty phrases with their informative quantity or state, for example "death toll tops 300", "thousands homeless", "10 injured", or "survivors".
- Preserve concrete affected assets rather than replacing them with generic words.
- Emit AFFECTS only when the text states damage, disruption, closure, loss, or recovery from a prior disruption of a concrete physical asset. Mere co-occurrence is not sufficient.
- Animals, people, organizations, responders, debris, floodwater, weather phenomena, and generic relief resources are not Infrastructure. Keep a named organization only when the text explicitly denotes its physical facility or network.
- Do not emit AFFECTS from warnings, prevention advice, requests for help, donations, evacuation support, or explicit non-impact statements such as "not affected". Entities may still be extracted when valid; only the unsupported relation is omitted.
- Uncertain but text-grounded candidates are allowed because the downstream verifier decides acceptance.
- Use only the given text. Never infer from file names, dataset labels, image paths, metadata, or outside knowledge.
- Do not output Time, Organization, Resource, Number, or entity types outside the closed schema.
- Output strict JSON only. Do not use markdown.

Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Text:
{text}

Output format:
{{
  "entities": [
    {{"name": "entity name", "type": "Disaster"}}
  ],
  "relations": [
    {{"head": "disaster entity", "relation": "OCCURS_IN", "tail": "location entity"}}
  ]
}}
""".strip()


def build_llm_prompt(text: str, mode: str = "conservative") -> str:
    if mode not in EXTRACTION_MODES:
        raise ValueError(f"Unknown extraction mode {mode!r}; choose one of {EXTRACTION_MODES}.")
    if mode == "candidate":
        return _build_candidate_llm_prompt(text)

    schema = {
        "allowed_entity_types": list(CORE_ENTITY_TYPES),
        "allowed_relation_types": list(CORE_RELATION_TYPES),
        "valid_relation_signatures": [
            {"relation": relation, "head_type": head, "tail_type": tail}
            for relation, (head, tail) in RELATION_SIGNATURES.items()
        ],
    }
    return f"""
You are an information extraction system for disaster social media text.
Extract only disaster-related entities and relations from the given text.

Allowed entity types:
- Disaster: disaster type or disaster event, such as flood, earthquake, hurricane, wildfire.
- Location: affected or mentioned place, such as city, country, region, road area.
- Infrastructure: affected object or facility, such as road, bridge, house, building, school, hospital, business, church, seawall, power line.
- Casualty: casualty, death, fatality, victim, survivor, trapped/missing/injured information.

Allowed relation types:
- OCCURS_IN: Disaster -> Location
- AFFECTS: Disaster -> Infrastructure
- CAUSES: Disaster -> Casualty

Rules:
- Do not invent entities or relations that are not supported by the text.
- Do not output Time, Organization, Resource, Number, or other entity types.
- If no valid entity or relation exists, return empty arrays.
- Judge only from the given text. Do not use file names, dataset event names, image paths, or outside knowledge.
- Extract Disaster names from hashtags when clear, e.g., #HurricaneMaria -> Hurricane Maria, #MexicoEarthquake -> Mexico Earthquake.
- Extract a Disaster entity when the text mentions a disaster name in relief, recovery, victim, evacuee, rescue, response, or damage context, e.g., Hurricane Harvey relief, Harvey victims, Irma evacuees, Maria storm.
- Normalize short disaster mentions when disaster context is clear: Harvey -> Hurricane Harvey, Irma -> Hurricane Irma, Maria -> Hurricane Maria.
- Do not treat personal names as disasters: Steve Harvey, PJ Harvey, or ambiguous "Harvey Rescue" should not become Hurricane Harvey unless the text clearly mentions hurricane/storm/disaster context.
- For OCCURS_IN, only use places where the disaster really occurs or causes impact.
- Do not create OCCURS_IN for prayer/support/fundraising/relief/recovery/claims-processing texts, concerts, comparison maps, hypothetical examples, hashtags alone, or "transposed over" descriptions.
- Do not create OCCURS_IN for shelters, evacuee/current residence, news source locations, distance references such as "miles from", or preparation/warning/forecast/path/nears/watch texts unless actual impact is stated.
- Do not create AFFECTS for "spared by", "not damaged", "no damage", or pure warning/risk phrases such as Fire Danger or Tornado Warning.
- Create OCCURS_IN when the text explicitly states that a disaster hit, struck, ravaged, damaged, destroyed, affected, occurred in, made landfall in, or is named with a real affected place such as Mexico Earthquake or California Wildfires.
- Treat resident groups as Location evidence when the place is explicit, e.g., Houstonians -> Houston, Tampa Bay residents -> Tampa Bay.
- Create AFFECTS for concrete facilities or objects affected by the disaster, including homes, houses, roads, bridges, schools, hospitals, businesses, churches, buildings, apartments, blocks, seawalls, pools, boats, cars, archives, beaches, power lines, refineries, LNG terminals, crops, and utilities.
- Trigger AFFECTS when the object is described as damaged, destroyed, burned, lost, collapsed, washed out, closed, evacuated, strained, eroded, or otherwise directly impacted by the disaster.
- Create CAUSES for deaths, killed people, injured people, victims, fatalities, missing/trapped people, evacuees, rescued people, patients, survivors, and other clearly disaster-related casualty information.
- For "rescued", "saved", "evacuated", or "survivors", create a Casualty entity only when people affected by the disaster are clearly described.
- Normalize hashtag/camel-case disaster names when possible, e.g., HurricaneHarvey -> Hurricane Harvey, MexicoEarthquake -> Mexico Earthquake.
- Treat power outage, relief, support, fundraising, and news as context unless the text clearly describes disaster damage, location, infrastructure impact, or casualties.
- If the text contains both a disaster and a location but only expresses aid, prayer, fundraising, distance, forecast, or evacuation destination, output the Disaster entity if appropriate but do not output OCCURS_IN.
- Output strict JSON only. Do not use markdown.

Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Text:
{text}

Output format:
{{
  "entities": [
    {{"name": "entity name", "type": "Disaster"}}
  ],
  "relations": [
    {{"head": "disaster entity", "relation": "OCCURS_IN", "tail": "location entity"}}
  ]
}}
""".strip()
