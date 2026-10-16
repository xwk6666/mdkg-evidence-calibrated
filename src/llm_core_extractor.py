import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from disaster_kg_schema import EXTRACTION_MODES, build_llm_prompt, validate_extraction


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "CrisisMMD_v2.0" / "structured"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "CrisisMMD_v2.0" / "structured_llm_core"
SPLIT_NAMES = ("train", "val", "test")

# ==================== LLM Config ====================
# Keep secrets out of this tracked script. If you do not want to use
# PowerShell environment variables, copy llm_local_config.py.example to
# llm_local_config.py and fill your private API settings there.


# # 方案 A: 使用 DeepSeek (推荐)
API_KEY = ""
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"


def load_local_llm_config() -> Dict[str, str]:
    try:
        import llm_local_config  # type: ignore
    except ImportError:
        return {}

    return {
        "API_KEY": getattr(llm_local_config, "API_KEY", ""),
        "BASE_URL": getattr(llm_local_config, "BASE_URL", ""),
        "MODEL_NAME": getattr(llm_local_config, "MODEL_NAME", ""),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def strip_markdown_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    if text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


class CoreDisasterLLMExtractor:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        extraction_mode: str = "conservative",
    ):
        if extraction_mode not in EXTRACTION_MODES:
            raise ValueError(
                f"Unknown extraction mode {extraction_mode!r}; choose one of {EXTRACTION_MODES}."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.extraction_mode = extraction_mode

    def extract(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        prompt = build_llm_prompt(text, mode=self.extraction_mode)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You extract disaster KG entities and relations. Output strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = strip_markdown_json(response.choices[0].message.content or "")
        parsed = json.loads(raw)
        return validate_extraction(
            parsed,
            infer_missing_relations=False,
            extraction_mode=self.extraction_mode,
            source_text=text,
        )


def get_done_ids(samples: List[Dict[str, object]]) -> set:
    done = set()
    for sample in samples:
        if sample.get("llm_status") != "ok":
            continue
        sample_id = sample.get("sample_id") or sample.get("filename")
        if sample_id:
            done.add(sample_id)
    return done


def is_fatal_api_error(error_message: str) -> bool:
    fatal_markers = (
        "Arrearage",
        "overdue-payment",
        "Access denied",
        "invalid_api_key",
        "Unauthorized",
        "insufficient_quota",
        "permission",
    )
    return any(marker.lower() in error_message.lower() for marker in fatal_markers)


def convert_split(
    extractor: CoreDisasterLLMExtractor,
    input_path: Path,
    output_path: Path,
    limit: int,
    resume: bool,
    sleep_seconds: float,
) -> None:
    source = load_json(input_path)
    source_samples = source["samples"]

    output_samples: List[Dict[str, object]] = []
    if resume and output_path.exists():
        existing_samples = load_json(output_path).get("samples", [])
        output_samples = [sample for sample in existing_samples if sample.get("llm_status") == "ok"]
        dropped = len(existing_samples) - len(output_samples)
        print(f"Resuming from {output_path}, existing ok samples: {len(output_samples)}, retryable failures: {dropped}")

    done_ids = get_done_ids(output_samples)
    processed_now = 0

    for index, sample in enumerate(source_samples):
        sample_id = sample.get("sample_id") or sample.get("filename")
        if sample_id in done_ids:
            continue
        if limit > 0 and processed_now >= limit:
            break

        text = sample.get("text", "")
        try:
            extraction = extractor.extract(text)
            status = "ok"
        except Exception as exc:
            extraction = {"entities": [], "relations": []}
            status = f"failed: {exc}"

        converted = {
            "sample_id": sample.get("sample_id"),
            "post_id": sample.get("post_id"),
            "event_id": sample.get("event_id"),
            "filename": sample.get("filename"),
            "text": text,
            "entities": extraction["entities"],
            "relations": extraction["relations"],
            "llm_status": status,
        }
        output_samples.append(converted)
        processed_now += 1

        if status != "ok" and is_fatal_api_error(status):
            save_json({"samples": output_samples}, output_path)
            raise RuntimeError(
                "Fatal LLM API error detected. Please fix the API key/account/base_url first, "
                f"then rerun with --resume. Latest error: {status}"
            )

        if processed_now % 10 == 0:
            save_json({"samples": output_samples}, output_path)
            print(f"Processed {processed_now} new samples; latest status: {status}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    save_json({"samples": output_samples}, output_path)
    print(f"Saved LLM core extraction to: {output_path}")
    print(f"Total output samples: {len(output_samples)} | New samples: {processed_now}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract core disaster KG entities/relations with an LLM.")
    parser.add_argument("--split", choices=[*SPLIT_NAMES, "all"], default="test")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=20, help="Use 0 to process the full split.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="DISASTER_LLM_API_KEY")
    parser.add_argument("--base-url-env", default="DISASTER_LLM_BASE_URL")
    parser.add_argument("--model-env", default="DISASTER_LLM_MODEL")
    parser.add_argument(
        "--extraction-mode",
        choices=EXTRACTION_MODES,
        default="conservative",
        help="Use candidate for recall-first candidate generation before downstream verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_config = load_local_llm_config()
    api_key = os.getenv(args.api_key_env) or local_config.get("API_KEY") or API_KEY
    base_url = os.getenv(args.base_url_env) or local_config.get("BASE_URL") or BASE_URL
    model_name = os.getenv(args.model_env) or local_config.get("MODEL_NAME") or MODEL_NAME

    if not api_key:
        raise RuntimeError(
            "Missing API key. Fill src/llm_local_config.py, "
            f"or set ${args.api_key_env} before running."
        )

    extractor = CoreDisasterLLMExtractor(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        extraction_mode=args.extraction_mode,
    )
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    splits = SPLIT_NAMES if args.split == "all" else (args.split,)

    for split in splits:
        print(f"=== Extracting split: {split} ===")
        convert_split(
            extractor=extractor,
            input_path=input_dir / f"{split}_entities.json",
            output_path=output_dir / f"{split}_entities.json",
            limit=args.limit,
            resume=args.resume,
            sleep_seconds=args.sleep,
        )


if __name__ == "__main__":
    main()
