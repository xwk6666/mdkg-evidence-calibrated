"""Local frozen replay across separate software/data packages; no models/network."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import reproduce as frozen


def verify_manifest(root):
    path = root / "manifests/files_sha256.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    indexed = frozen.unique_index(rows, "path")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and ".git" not in p.relative_to(root).parts}
    expected = {r["path"] for r in rows}
    frozen.require(actual == expected | {"manifests/files_sha256.tsv"}, "Release file inventory differs from manifest")
    for row in rows:
        file = (root / row["path"]).resolve()
        frozen.require(root in file.parents and not file.is_symlink(), "Unsafe manifest path")
        content = file.read_bytes()
        frozen.require(hashlib.sha256(content).hexdigest() == row["sha256"] and len(content) == int(row["bytes"]),
                       "Release checksum mismatch: " + row["path"])
    return {"payload_files": len(actual), "hashes_verified": len(indexed), "git_metadata_excluded": True}


def human300_without_source_ids():
    gold = frozen.read("annotations/human300_final_labels.csv")
    scores = frozen.read("scores/human300_three_model_scores.csv")
    g = frozen.unique_index(gold, "annotation_id")
    s = frozen.unique_index(scores, "annotation_id")
    frozen.require(set(g) == set(s) and len(g) == 300, "Human300 annotation IDs mismatch")
    frozen.require(Counter(r["final_label_0_1_2_U"] for r in gold) == {"0": 62, "1": 92, "2": 146}, "Human300 labels")
    protocol = json.loads((frozen.ROOT / "configs/human_annotation_protocol.json").read_text(encoding="utf-8"))
    frozen.require(protocol["pre_adjudication_agreement"] == 254 and protocol["manually_adjudicated_disagreements"] == 46,
                   "Author-confirmed process metadata mismatch")
    archive = frozen.read("tables/table6_human300_all_metrics.csv")
    out = []
    for model, prefix in [("Semantic-HN", "semantic_hn"), ("FT-CLIP", "ft_clip"), ("Random-HN", "random_hn")]:
        for task, positive in [("strict_support", {"2"}), ("broad_relatedness", {"1", "2"})]:
            labels, probabilities, predictions = [], [], []
            for key, row in s.items():
                frozen.require(row["final_label_0_1_2_U"] == g[key]["final_label_0_1_2_U"], "Frozen Gold mismatch")
                labels.append(int(row["final_label_0_1_2_U"] in positive))
                probabilities.append(float(row[prefix + "_probability"]))
                predictions.append(int(row[prefix + "_prediction"]))
                frozen.require(predictions[-1] == int(probabilities[-1] >= float(row[prefix + "_threshold"])),
                               "Frozen threshold/prediction mismatch")
            result = {"model": model, "task": task, "threshold": float(scores[0][prefix + "_threshold"]),
                      **frozen.binary(labels, probabilities, predictions), **frozen.calibration(labels, probabilities)}
            matches = [r for r in archive if r["model"] == model and r["task"] == task]
            frozen.require(len(matches) == 1, "Human300 archived row missing")
            for key, field in [("auroc", "auc_probability"), ("ap", "ap_probability"), ("f1", "f1"),
                               ("specificity", "specificity"), ("brier_score", "brier_score"), ("ece", "ece")]:
                frozen.require(abs(result[key] - float(matches[0][field])) < 1e-12, "Frozen Human300 metric mismatch")
            out.append(result)
    frozen.require(round(out[0]["auroc"], 4) == .5842, "Strict Semantic-HN AUROC mismatch")
    return {"mode": "frozen-output reproduction using original annotation_id; source sample IDs withheld",
            "n": 300, "label_counts": {"0": 62, "1": 92, "2": 146}, "metrics": out,
            "reported_pre_adjudication_agreement": 254, "reported_manually_adjudicated": 46,
            "process_counts_source": "author-confirmed metadata; no private A/B labels included",
            "original_test_membership_independently_verified_by_public_package": False,
            "cohens_kappa_independently_recomputed": False}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", required=True)
    args, remaining = parser.parse_known_args()
    data = Path(args.data_root).resolve()
    code = Path(__file__).resolve().parents[1]
    frozen.ROOT = data
    frozen.check_hashes = lambda: {"software": verify_manifest(code), "data": verify_manifest(data),
                                 "source_id_and_split_membership_publicly_verifiable": False}
    frozen.human300 = human300_without_source_ids
    sys.argv = [sys.argv[0], *remaining]
    frozen.main()


if __name__ == "__main__":
    main()
