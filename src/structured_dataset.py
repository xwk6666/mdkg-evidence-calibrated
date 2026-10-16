import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


ENTITY_TYPE_TO_ID = {
    "PAD": 0,
    "Unknown": 1,
    "Disaster": 2,
    "DisasterType": 3,
    "Location": 4,
    "Infrastructure": 5,
    "Casualty": 6,
    "Organization": 7,
    "Time": 8,
}


class StructuredCrisisDataset(Dataset):
    """Dataset that returns image/text pairs together with lightweight entity supervision."""

    def __init__(
        self,
        structured_json_path: str,
        img_dir: str,
        tokenizer_name: str = "bert-base-uncased",
        text_max_length: int = 77,
        entity_max_length: int = 16,
        max_entities: int = 8,
        skip_missing_images: bool = True,
    ):
        self.img_dir = Path(img_dir)
        self.structured_json_path = str(structured_json_path)
        self.missing_images: List[Dict[str, str]] = []
        with open(structured_json_path, "r", encoding="utf-8") as handle:
            raw_data = json.load(handle)["samples"]

        self.data = self._filter_and_resolve_images(raw_data, skip_missing_images=skip_missing_images)

        event_ids = sorted({item.get("event_id", "unknown_event") for item in self.data})
        self.event_to_id = {event_id: idx for idx, event_id in enumerate(event_ids)}

        self.text_max_length = text_max_length
        self.entity_max_length = entity_max_length
        self.max_entities = max_entities

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._text_cache: List[Dict[str, torch.Tensor]] = []
        self._entity_cache: List[Dict[str, torch.Tensor]] = []
        self._build_token_cache()

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

    def _filter_and_resolve_images(self, samples: List[Dict[str, object]], skip_missing_images: bool):
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
        elif self.missing_images:
            print(
                f"Warning: found {len(self.missing_images)} missing images in "
                f"{self.structured_json_path}; skip_missing_images=False, so loading may fail later."
            )
        return kept

    def _build_token_cache(self) -> None:
        texts = [item["text"] for item in self.data]
        encoded_texts = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )

        for idx, item in enumerate(self.data):
            self._text_cache.append(
                {
                    "input_ids": encoded_texts["input_ids"][idx].clone(),
                    "attention_mask": encoded_texts["attention_mask"][idx].clone(),
                }
            )
            self._entity_cache.append(self._encode_entities(item.get("entities", [])))

    def _encode_entities(self, entities: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        entity_texts: List[str] = []
        entity_type_ids: List[int] = []

        for entity in entities[: self.max_entities]:
            entity_texts.append(entity.get("name", ""))
            entity_type = entity.get("type", "Unknown")
            entity_type_ids.append(ENTITY_TYPE_TO_ID.get(entity_type, ENTITY_TYPE_TO_ID["Unknown"]))

        num_entities = len(entity_texts)
        if num_entities == 0:
            entity_texts = [""]
            entity_type_ids = [ENTITY_TYPE_TO_ID["PAD"]]
            num_entities = 0

        encoded = self.tokenizer(
            entity_texts,
            padding="max_length",
            truncation=True,
            max_length=self.entity_max_length,
            return_tensors="pt",
        )

        pad_count = self.max_entities - encoded["input_ids"].size(0)
        if pad_count > 0:
            pad_ids = torch.zeros((pad_count, self.entity_max_length), dtype=torch.long)
            encoded["input_ids"] = torch.cat([encoded["input_ids"], pad_ids], dim=0)
            encoded["attention_mask"] = torch.cat([encoded["attention_mask"], pad_ids], dim=0)
            entity_type_ids.extend([ENTITY_TYPE_TO_ID["PAD"]] * pad_count)

        entity_mask = torch.zeros(self.max_entities, dtype=torch.float32)
        if num_entities > 0:
            entity_mask[: min(num_entities, self.max_entities)] = 1.0

        return {
            "entity_input_ids": encoded["input_ids"][: self.max_entities],
            "entity_attention_mask": encoded["attention_mask"][: self.max_entities],
            "entity_type_ids": torch.tensor(entity_type_ids[: self.max_entities], dtype=torch.long),
            "entity_mask": entity_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        img_path = Path(item.get("_resolved_image_path") or (self.img_dir / item["filename"]))
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        text_encoded = self._text_cache[idx]
        entity_features = self._entity_cache[idx]

        return {
            "image": image_tensor,
            "input_ids": text_encoded["input_ids"].clone(),
            "attention_mask": text_encoded["attention_mask"].clone(),
            "entity_input_ids": entity_features["entity_input_ids"].clone(),
            "entity_attention_mask": entity_features["entity_attention_mask"].clone(),
            "entity_type_ids": entity_features["entity_type_ids"].clone(),
            "entity_mask": entity_features["entity_mask"].clone(),
            "event_label": torch.tensor(self.event_to_id.get(item.get("event_id", "unknown_event"), 0), dtype=torch.long),
        }
