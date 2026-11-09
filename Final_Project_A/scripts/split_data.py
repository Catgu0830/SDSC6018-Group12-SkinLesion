"""Freeze a deterministic, diagnosis-stratified lesion split without fitting transforms.

Usage: python3 Final_Project_A/scripts/split_data.py
Reproduce: add --output-dir /private/tmp/ham10000_split_repro --no-report
Dependencies: Python standard library only. Existing versions are never overwritten.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SPLITS = ["train", "val", "test"]
SEED = 42
ALGORITHM = "HAM10000-lesion-split-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def apportion(weights, target):
    """Hamilton apportionment with exact integer arithmetic and lexical tie breaks."""
    total = sum(weights.values())
    if total <= 0 or not 0 <= target <= total:
        raise ValueError("Invalid allocation target or class counts")
    quota = {label: count * target // total for label, count in weights.items()}
    order = sorted(weights, key=lambda label: (-(weights[label] * target % total), label))
    for label in order[:target - sum(quota.values())]:
        quota[label] += 1
    return quota


def assign_lesions(groups):
    strata = defaultdict(list)
    for lesion_id, rows in groups.items():
        labels = {row["dx"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels within {lesion_id}")
        strata[next(iter(labels))].append(lesion_id)
    if set(strata) != set(LABELS):
        raise ValueError("Expected exactly the seven HAM10000 diagnostic labels")
    n = len(groups)
    targets = {"train": (n * 70 + 50) // 100, "val": (n * 15 + 50) // 100}
    targets["test"] = n - targets["train"] - targets["val"]
    counts = {label: len(strata[label]) for label in LABELS}
    train = apportion(counts, targets["train"])
    holdout = {label: counts[label] - train[label] for label in LABELS}
    val = apportion(holdout, targets["val"])
    quotas = {label: {"train": train[label], "val": val[label], "test": holdout[label] - val[label]} for label in LABELS}
    if any(quotas[label][split] < 1 for label in LABELS for split in SPLITS):
        raise ValueError("Insufficient lesions to include every class in every split")
    assignments = {}
    for label in LABELS:
        ordered = sorted(strata[label], key=lambda lesion: (
            hashlib.sha256(f"{ALGORITHM}|{SEED}|{label}|{lesion}".encode()).hexdigest(), lesion))
        cut1, cut2 = train[label], train[label] + val[label]
        for i, lesion in enumerate(ordered):
            assignments[lesion] = "train" if i < cut1 else "val" if i < cut2 else "test"
    return assignments, quotas, targets


def run(output, make_report=True):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing split version is protected: {output}")
    raw_path = ROOT / "data/raw/HAM10000_metadata.csv"
    official = json.loads((ROOT / "data/raw/harvard_dataset_record_v4.json").read_text())
    source = next(item["dataFile"] for item in official["data"]["latestVersion"]["files"] if item["dataFile"]["id"] == 4338392)
    if hashlib.md5(raw_path.read_bytes()).hexdigest() != source["checksum"]["value"]:
        raise ValueError("Original metadata checksum mismatch")
    audit = json.loads((ROOT / "docs/image_audit.json").read_text())
    if not audit["checks"] or not all(audit["checks"].values()):
        raise ValueError("Image audit must pass before splitting")
    fields, metadata = read_csv(raw_path)
    _, image_rows = read_csv(ROOT / "data/image_manifest.csv")
    images = {row["image_id"]: row for row in image_rows}
    if len(images) != len(image_rows) or len({row["image_id"] for row in metadata}) != len(metadata):
        raise ValueError("Duplicate image IDs in inputs")
    if set(images) != {row["image_id"] for row in metadata}:
        raise ValueError("Image manifest does not match metadata")
    groups = defaultdict(list)
    for row in metadata:
        if not row["lesion_id"] or not row["image_id"]:
            raise ValueError("Missing image or lesion ID")
        image = images[row["image_id"]]
        if (row["lesion_id"], row["dx"]) != (image["lesion_id"], image["dx"]):
            raise ValueError(f"Image manifest has conflicting metadata: {row['image_id']}")
        path = (ROOT / image["relative_path"]).resolve()
        if not path.is_relative_to((ROOT / "data/raw/images").resolve()) or not path.is_file():
            raise ValueError(f"Invalid or missing image path: {row['image_id']}")
        groups[row["lesion_id"]].append(row)
    assignments, quotas, targets = assign_lesions(groups)
    label_mapping = {label: i for i, label in enumerate(LABELS)}
    combined = []
    for row in metadata:
        image = images[row["image_id"]]
        combined.append({**row, "label": label_mapping[row["dx"]],
            "split": assignments[row["lesion_id"]], "relative_path": image["relative_path"],
            "image_sha256": image["sha256"]})
    partitions = {split: [row for row in combined if row["split"] == split] for split in SPLITS}
    lesion_sets = {split: {row["lesion_id"] for row in rows} for split, rows in partitions.items()}
    image_sets = {split: {row["image_id"] for row in rows} for split, rows in partitions.items()}
    hashes = defaultdict(set)
    for row in combined:
        hashes[row["image_sha256"]].add(row["split"])
    checks = {
        "every_original_image_assigned_once": len(combined) == len(set.union(*image_sets.values())) == len(metadata),
        "all_original_fields_preserved": all({key: row[key] for key in fields} == original for row, original in zip(combined, metadata)),
        "all_lesions_assigned_once": len(set.union(*lesion_sets.values())) == len(groups),
        "all_splits_contain_seven_classes": all({row["dx"] for row in rows} == set(LABELS) for rows in partitions.values()),
        "no_exact_byte_duplicates_cross_splits": all(len(splits) == 1 for splits in hashes.values()),
        "lesion_counts_match_targets": all(len(lesion_sets[split]) == targets[split] for split in SPLITS),
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        checks[f"no_{left}_{right}_lesion_overlap"] = not lesion_sets[left] & lesion_sets[right]
        checks[f"no_{left}_{right}_image_overlap"] = not image_sets[left] & image_sets[right]
    if not all(checks.values()):
        raise ValueError(f"Split integrity failure: {checks}")
    config = {
        "algorithm": ALGORITHM, "seed": SEED, "group_field": "lesion_id", "stratify_field": "dx",
        "team_alignment_status": "A first proposal; ratio, seed and label mapping not yet confirmed with B",
        "target_ratios_by_lesion": {"train": 0.70, "val": 0.15, "test": 0.15},
        "global_target_lesions": targets, "per_class_lesion_quotas": quotas,
        "class_order": LABELS, "label_mapping": label_mapping,
        "allocation": "Hamilton train quotas, then Hamilton validation quotas from held-out lesions; lexical tie breaks",
        "ordering": "Sort each class by SHA256(algorithm|seed|dx|lesion_id), then lesion_id; assign train, val, test sequentially",
        "relative_path_root": "Final_Project_A",
        "original_samples_retained": True, "preprocessing_fitted": False,
        "input_sha256": {"HAM10000_metadata.csv": digest(raw_path),
                         "image_manifest.csv": digest(ROOT / "data/image_manifest.csv"),
                         "split_data.py": digest(Path(__file__))},
    }
    output.mkdir(parents=True, exist_ok=True)
    combined_fields = fields + ["label", "split", "relative_path", "image_sha256"]
    write_csv(output / "combined.csv", combined_fields, combined)
    for split in SPLITS:
        write_csv(output / f"{split}.csv", combined_fields, partitions[split])
    lesions = [{"lesion_id": lesion_id, "dx": groups[lesion_id][0]["dx"],
                "label": label_mapping[groups[lesion_id][0]["dx"]], "split": assignments[lesion_id],
                "n_images": len(groups[lesion_id])} for lesion_id in sorted(groups)]
    write_csv(output / "lesions.csv", ["lesion_id", "dx", "label", "split", "n_images"], lesions)
    write_json(output / "label_mapping.json", label_mapping)
    write_json(output / "split_config.json", config)
    write_json(output / "checksums.json", {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    summary = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "checks": checks,
        "image_counts": {split: len(rows) for split, rows in partitions.items()},
        "lesion_counts": {split: len(lesion_sets[split]) for split in SPLITS},
        "class_image_counts": {split: dict(Counter(row["dx"] for row in rows)) for split, rows in partitions.items()},
        "class_lesion_counts": {split: dict(Counter(row["dx"] for row in lesions if row["split"] == split)) for split in SPLITS}}
    if make_report:
        docs = ROOT / "docs"
        (docs / "split_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        lines = ["# HAM10000 lesion-level split results", "", f"Generated: {summary['generated_at']}", "",
            "The split groups images by `lesion_id` and stratifies lesions by `dx`. Version `v1_seed42` uses seed 42 and target lesion proportions of 70% / 15% / 15%. Original image records and metadata fields are retained. No image IDs, lesion IDs or exact-byte image hashes overlap between subsets.", "",
            "The ratio, seed and label mapping are A-stage implementation choices pending coordination with the training member. They are not course-mandated settings. No model results were used to choose the seed.", "",
            "| Subset | Lesions | Lesion proportion | Images | Image proportion |", "| --- | ---: | ---: | ---: | ---: |"]
        for split in SPLITS:
            lines.append(f"| {split} | {len(lesion_sets[split]):,} | {len(lesion_sets[split])/len(groups):.2%} | {len(partitions[split]):,} | {len(partitions[split])/len(metadata):.2%} |")
        lines += ["", "## Counts by diagnosis", "", "| Diagnosis | Train images / lesions | Validation images / lesions | Test images / lesions |", "| --- | ---: | ---: | ---: |"]
        for label in LABELS:
            lines.append("| " + label + " | " + " | ".join(f"{summary['class_image_counts'][split][label]} / {summary['class_lesion_counts'][split][label]}" for split in SPLITS) + " |")
        lines += ["", "## Integrity checks", ""]
        lines += [f"- {'PASS' if ok else 'FAIL'}: `{name}`." for name, ok in checks.items()]
        lines += ["", "## Using this split", "",
            "Read `train.csv`, `val.csv` and `test.csv` from `data/splits/v1_seed42/`. Use the same version for every model and ablation. Resolve `relative_path` against the A-stage directory. Join predictions by `image_id`, never by row position.", "",
            "Candidate predictors are the image, age, sex and localization. `dx` and `label` are targets. Image/lesion identifiers, split membership, `dx_type`, `dataset` and image hashes are audit fields, not predictors.", "",
            "The label mapping follows lexical diagnosis order. `split_config.json` records the allocation algorithm and source hashes; `checksums.json` identifies the frozen files. Matching seeds alone does not guarantee matching samples.", "",
            "## Reproduction and limitations", "",
            "The saved [validation record](split_validation.json) documents independent field, coverage and overlap checks, deterministic reproduction, input-order invariance, diagnosis-conflict rejection and overwrite protection. See the [split method](04_data_split_method.md) for the exact algorithm.", "",
            "Image proportions differ slightly from the lesion targets because lesions have different numbers of images. Known exact-byte duplicate pairs are retained within their lesion. Lesion-level separation does not establish patient-level separation or rule out unidentified near-duplicates.", "",
            "Raw missingness is preserved in these split files. The separate [preprocessing stage](07_preprocessing_guide.md) fits its parameters on training records only. Validation supports model selection; test outcomes must remain outside tuning and model selection.", ""]
        (docs / "05_data_split_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/splits/v1_seed42")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), not args.no_report)
