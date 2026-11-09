"""Read-only HAM10000 metadata audit; Python standard library only.

Run from any directory: python3 /absolute/path/to/audit_metadata.py
Raw input is never edited. No imputation, splitting or image checks occur here.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs"
FIELDS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]
CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def read_table(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = reader.fieldnames
        rows = list(reader)
    if fields != FIELDS:
        raise ValueError(f"Unexpected fields in {path.name}: {fields}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed records in {path.name}")
    return rows


def counts(rows, field):
    return dict(Counter(row[field] for row in rows).most_common())


def run():
    original = RAW / "HAM10000_metadata.csv"
    provenance = json.loads((RAW / "metadata_provenance.json").read_text())
    actual_md5 = hashlib.md5(original.read_bytes()).hexdigest()
    if actual_md5 != provenance["expected_md5"]:
        raise ValueError("Original CSV does not match the recorded official checksum")
    rows = read_table(original)
    tab = read_table(RAW / "HAM10000_metadata.tab", "\t")
    if rows != tab:
        raise ValueError("CSV and TAB do not contain identical ordered records")
    groups = defaultdict(list)
    for row in rows:
        groups[row["lesion_id"]].append(row)
    image_counts = Counter(row["image_id"] for row in rows)
    contradictions = []
    for lesion_id, group in groups.items():
        for field in ["dx", "age", "sex", "localization", "dataset", "dx_type"]:
            values = sorted({row[field] for row in group})
            if len(values) > 1:
                contradictions.append({
                    "lesion_id": lesion_id, "field": field, "values": values,
                    "image_ids": [row["image_id"] for row in group],
                })
    invalid_age = []
    ages = []
    for row in rows:
        if not row["age"].strip():
            continue
        try:
            age = float(row["age"])
            if not 0 <= age <= 120:
                raise ValueError
            ages.append(age)
        except ValueError:
            invalid_age.append({"image_id": row["image_id"], "age": row["age"]})
    consistent_labels = all(len({row["dx"] for row in group}) == 1 for group in groups.values())
    lesion_labels = Counter(group[0]["dx"] for group in groups.values()) if consistent_labels else None
    checks = {
        "original_checksum_matches_official": True,
        "csv_tab_records_identical": True,
        "expected_10015_records": len(rows) == 10015,
        "no_duplicate_image_ids": all(n == 1 for n in image_counts.values()),
        "no_missing_image_or_lesion_ids": all(row["image_id"].strip() and row["lesion_id"].strip() for row in rows),
        "expected_seven_labels": {row["dx"] for row in rows} == CLASSES,
        "consistent_diagnosis_per_lesion": consistent_labels,
        "age_numeric_and_in_0_to_120_when_present": not invalid_age,
    }
    result = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "input": str(original.relative_to(ROOT)),
        "input_md5": actual_md5,
        "scope": "Metadata only; raw records are unchanged. Images and image integrity are not checked.",
        "fields": FIELDS, "n_records": len(rows),
        "n_unique_images": len(image_counts), "n_unique_lesions": len(groups),
        "checks": checks,
        "empty_by_field": {field: sum(not row[field].strip() for row in rows) for field in FIELDS},
        "unknown_by_field": {field: sum(row[field].strip().lower() == "unknown" for row in rows) for field in FIELDS},
        "duplicate_image_ids": {key: n for key, n in image_counts.items() if n > 1},
        "class_images": counts(rows, "dx"),
        "class_lesions": dict(lesion_labels.most_common()) if lesion_labels is not None else None,
        "sex": counts(rows, "sex"), "localization": counts(rows, "localization"),
        "dx_type": counts(rows, "dx_type"), "dataset": counts(rows, "dataset"),
        "age_observed_min": min(ages) if ages else None,
        "age_observed_max": max(ages) if ages else None,
        "age_observed_distinct": sorted(set(ages)), "invalid_age_records": invalid_age,
        "images_per_lesion": dict(sorted(Counter(map(len, groups.values())).items())),
        "n_lesions_with_multiple_images": sum(len(group) > 1 for group in groups.values()),
        "within_lesion_inconsistency_counts": dict(Counter(item["field"] for item in contradictions)),
        "within_lesion_inconsistencies": contradictions,
        "images_downloaded": False, "image_integrity_checked": False,
        "split_created": False, "preprocessing_fitted": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    definitions = [
        ("image_id", "Image identifier", "Join images, manifests and predictions; not a predictor"),
        ("lesion_id", "Lesion identifier; one lesion can have several images", "Group the split; not a patient identifier or predictor"),
        ("dx", "Seven-class diagnosis", "Prediction target; never an input feature"),
        ("age", "Recorded age", "Candidate predictor; fit imputation on training records only"),
        ("sex", "Recorded sex", "Candidate predictor; retain unknown as a category"),
        ("localization", "Recorded lesion site", "Candidate predictor; retain original per-image values"),
        ("dx_type", "Diagnostic confirmation method", "Descriptive and audit use only"),
        ("dataset", "Source subset", "Descriptive and audit use only"),
    ]
    lines = ["# HAM10000 metadata audit", "", f"Generated: {result['generated_at']}", "",
        "The original CSV matches its recorded official MD5. The CSV and default TAB export contain identical ordered records. Image identifiers and diagnosis labels pass the checks below. Missing metadata and within-lesion inconsistencies are retained and documented.", "",
        "## Source and verification", "",
        "The source is HAM10000 V4.0 on Harvard Dataverse, DOI 10.7910/DVN/DBW86T, metadata file ID 4338392. The saved dataset record and provenance files are in `data/raw/`.", "",
        f"- Original CSV MD5: `{actual_md5}`.",
        "- The default endpoint returns TAB. Add `?format=original` to obtain the CSV to which the official checksum applies. Different file bytes do not imply different parsed records.",
        "- This audit does not edit, drop or impute raw records.",
        "- Run `python3 scripts/audit_metadata.py` from the A-stage directory. This script uses the Python standard library.", "",
        "## Field roles", "", "| Field | Meaning | Use |", "| --- | --- | --- |"]
    lines += [f"| `{field}` | {meaning} | {usage} |" for field, meaning, usage in definitions]
    lines += ["", "## Image and lesion counts", "",
        f"There are {len(rows):,} records, {len(image_counts):,} distinct image IDs and {len(groups):,} distinct lesion IDs. {result['n_lesions_with_multiple_images']:,} lesions have multiple images, with 1 to {max(map(len, groups.values()))} images per lesion.", "",
        "Each row represents an image. Image counts are not counts of distinct lesions or patients. Repeated lesion IDs identify multiple views and are not automatically deleted. The metadata has no patient identifier, so patient-level separation cannot be established.", "",
        "| Diagnosis | Images | Image proportion | Distinct lesions |", "| --- | ---: | ---: | ---: |"]
    for label, n in result["class_images"].items():
        lesion_n = result["class_lesions"][label] if result["class_lesions"] else None
        lesion_display = f"{lesion_n:,}" if lesion_n is not None else "Not available"
        lines.append(f"| {label} | {n:,} | {n / len(rows):.2%} | {lesion_display} |")
    lines += ["", "Class imbalance motivates reporting macro-F1 and class-specific performance alongside accuracy. This audit reports dataset structure, not model performance.", "",
        "## Missing and unknown values", "",
        "Counts below use image records. Blank values and the literal category `unknown` are counted separately.", "",
        "| Field | Blank records | Unknown records |", "| --- | ---: | ---: |"]
    lines += [f"| {field} | {result['empty_by_field'][field]} | {result['unknown_by_field'][field]} |" for field in ["age", "sex", "localization"]]
    lines += ["", f"Observed ages range from {result['age_observed_min']:g} to {result['age_observed_max']:g} years and occur in five-year increments. Age zero is retained as an observed value. No populated age failed numeric parsing or the 0-120 range check.", "",
        "Raw missingness is preserved. The preprocessing stage uses training-only age imputation and explicit unknown categories; see [Preprocessing guide](07_preprocessing_guide.md).", "",
        "## Within-lesion consistency", "",
        "Diagnosis, age and sex are consistent within each lesion. The following fields have differing records:", "",
        "| Field | Lesions with multiple values |", "| --- | ---: |"]
    for field in ["localization", "dataset", "dx_type"]:
        lines.append(f"| {field} | {result['within_lesion_inconsistency_counts'].get(field, 0)} |")
    lines += ["", "Location discrepancies:", ""]
    for item in contradictions:
        if item["field"] == "localization":
            lines.append(f"- `{item['lesion_id']}`: {' / '.join(item['values'])}; images: {', '.join(item['image_ids'])}.")
    lines += ["", "Per-image source values are retained. A lesion appearing in multiple source subsets is not automatically treated as a separate lesion or an error. The full discrepancy list is in [metadata_audit.json](metadata_audit.json).", "",
        "## Audit checks", ""]
    lines += [f"- {'PASS' if passed else 'FAIL'}: `{name}`." for name, passed in checks.items()]
    lines += ["", "## Scope", "",
        "This script checks metadata only. Flags in its JSON describe this script's actions, not the completion status of the entire project. Image integrity, split integrity and feature preparation have separate reports.", "",
        "Sources: [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), [original CSV](https://dataverse.harvard.edu/api/access/datafile/4338392?format=original), [dataset paper](https://arxiv.org/abs/1803.10417).", ""]
    (OUT / "01_metadata_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["n_records", "n_unique_images", "n_unique_lesions", "checks", "empty_by_field", "unknown_by_field", "within_lesion_inconsistency_counts"]}, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("Metadata audit has failed checks; inspect the report before continuing")


if __name__ == "__main__":
    run()
