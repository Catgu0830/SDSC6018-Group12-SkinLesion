"""Independently verify EDA counts, artifact formats and deterministic output.

Usage: python3 Final_Project_A/scripts/validate_eda.py --deps-dir /path/to/plot/deps
Counting uses csv/Counter/statistics rather than the plotting program's pandas aggregates.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/eda/v1_seed42"
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(deps_dir=None):
    summary = json.loads((OUT / "eda_summary.json").read_text())
    manifest = json.loads((OUT / "eda_manifest.json").read_text())
    assert all(digest(ROOT / path) == expected for path, expected in manifest["inputs_sha256"].items())
    assert all(digest(OUT / path) == expected for path, expected in manifest["outputs_sha256"].items())
    assert digest(ROOT / "scripts/eda.py") == manifest["script_sha256"]
    parts = {split: read_csv(ROOT / f"data/splits/v1_seed42/{split}.csv") for split in ["train", "val", "test"]}
    train = parts["train"]
    groups = defaultdict(list)
    for row in train:
        groups[row["lesion_id"]].append(row)
    assert len(train) == summary["train_image_n"] == 7050
    assert len(groups) == summary["train_lesion_n"] == 5229
    classes = Counter(row["dx"] for row in train)
    lesions = Counter(group[0]["dx"] for group in groups.values())
    for row in read_csv(OUT / "tables/train_class_counts.csv"):
        label = row["dx"]
        assert int(row["image_n"]) == classes[label] == summary["class_counts"][label]["image_n"]
        assert int(row["lesion_n"]) == lesions[label] == summary["class_counts"][label]["lesion_n"]
        assert abs(float(row["image_pct"]) - classes[label]/len(train)*100) < 1e-8
        assert abs(float(row["lesion_pct"]) - lesions[label]/len(groups)*100) < 1e-8
    known = [float(row["age"]) for row in train if row["age"] != ""]
    assert len(known) == summary["age_original_observed_n"] == 7010
    assert statistics.median(known) == summary["age_original_observed_median"] == 50
    q1, _, q3 = statistics.quantiles(known, n=4, method="inclusive")
    assert (q1, q3) == (summary["age_original_observed_q1"], summary["age_original_observed_q3"])
    age_counts = {float(row["age_years"]): int(row["image_n"]) for row in read_csv(OUT / "tables/train_age_counts.csv")}
    assert age_counts == dict(Counter(known))
    for row in read_csv(OUT / "tables/train_age_by_dx.csv"):
        selected = [item for item in train if item["dx"] == row["dx"]]
        ages = [float(item["age"]) for item in selected if item["age"]]
        assert (len(selected), len(ages), len(selected)-len(ages)) == (int(row["image_n"]), int(row["observed_age_n"]), int(row["missing_age_n"]))
        assert abs(statistics.mean(ages) - float(row["mean"])) < 1e-8
        assert statistics.median(ages) == float(row["median"])
    for field, table in [("sex", "train_sex_counts"), ("localization", "train_localization_counts"),
                         ("dataset", "train_source_counts"), ("dx_type", "train_diagnosis_confirmation")]:
        actual = {row[field]: int(row["image_n"]) for row in read_csv(OUT / f"tables/{table}.csv")}
        assert actual == dict(Counter(row[field] for row in train))
        assert sum(actual.values()) == len(train)
    missing_ids = {
        "age_missing": {row["image_id"] for row in train if row["age"] == ""},
        "sex_unknown": {row["image_id"] for row in train if row["sex"] in ["", "unknown"]},
        "localization_unknown": {row["image_id"] for row in train if row["localization"] in ["", "unknown"]},
    }
    assert {field: len(ids) for field, ids in missing_ids.items()} == summary["missing_field_counts"]
    assert len(set.union(*missing_ids.values())) == summary["any_missing_or_unknown_image_n"] == 183
    for row in read_csv(OUT / "tables/train_missingness.csv"):
        assert int(row["image_n"]) == len(missing_ids[row["field"]])
        assert int(row["denominator"]) == len(train)
        assert abs(float(row["image_pct"]) - len(missing_ids[row["field"]])/len(train)*100) < 1e-8
    for row in read_csv(OUT / "tables/train_missingness_by_dx.csv"):
        ids = {item["image_id"] for item in train if item["dx"] == row["dx"]}
        count = len(ids & missing_ids[row["field"]])
        assert int(row["denominator"]) == len(ids) and int(row["missing_or_unknown_n"]) == count
        assert abs(float(row["image_pct"]) - count/len(ids)*100) < 1e-8
    patterns = Counter("+".join(field for field, ids in missing_ids.items() if row["image_id"] in ids) or "complete" for row in train)
    assert patterns == {row["pattern"]: int(row["image_n"]) for row in read_csv(OUT / "tables/train_missingness_patterns.csv")}
    assert sum(int(row["image_n"]) for row in read_csv(OUT / "tables/train_missingness_patterns.csv")) == len(train)
    multi = Counter(map(len, groups.values()))
    assert multi == {int(row["images_per_lesion"]): int(row["lesion_n"]) for row in read_csv(OUT / "tables/train_images_per_lesion.csv")}
    assert sum(count * size for size, count in multi.items()) == len(train)
    assert sum(count for size, count in multi.items() if size > 1) == summary["lesions_with_multiple_images"]
    for row in read_csv(OUT / "tables/split_class_composition.csv"):
        selected = [item for item in parts[row["split"]] if item["dx"] == row["dx"]]
        assert len(selected) == int(row["image_n"])
        assert len({item["lesion_id"] for item in selected}) == int(row["lesion_n"])
    for row in read_csv(OUT / "tables/split_overview.csv"):
        selected = parts[row["split"]]
        assert len(selected) == int(row["image_n"])
        assert len({item["lesion_id"] for item in selected}) == int(row["lesion_n"])
    assert len(list((OUT / "figures").glob("*.png"))) == len(summary["figures"]) == 6
    for name in summary["figures"]:
        with Image.open(OUT / f"figures/{name}.png") as image:
            assert image.format == "PNG" and min(image.size) > 1200
            assert min(image.info["dpi"]) > 299
            image.verify()
        assert ET.parse(OUT / f"figures/{name}.svg").getroot().tag.endswith("svg")
    with TemporaryDirectory(prefix="ham10000_eda_repro_", dir="/private/tmp") as temp:
        command = [sys.executable, str(ROOT / "scripts/eda.py"), "--output-dir", temp, "--no-index"]
        if deps_dir:
            command += ["--deps-dir", str(deps_dir)]
        subprocess.run(command, capture_output=True, text=True, check=True)
        # Renderer serialization may vary in SVGs. Require exact statistics/PNG
        # reproduction; validate SVG structure and canvas dimensions separately.
        reproducible = {name: expected for name, expected in manifest["outputs_sha256"].items() if not name.endswith(".svg")}
        mismatch = [name for name, expected in reproducible.items() if digest(Path(temp)/name) != expected]
        assert not mismatch, f"Non-reproducible outputs: {mismatch}"
        for name in summary["figures"]:
            original_svg = ET.parse(OUT / f"figures/{name}.svg").getroot()
            reproduced_svg = ET.parse(Path(temp) / f"figures/{name}.svg").getroot()
            assert original_svg.attrib == reproduced_svg.attrib
        again = subprocess.run(command, capture_output=True, text=True)
        assert again.returncode != 0 and "protected" in again.stderr
    assert all(digest(ROOT / path) == expected for path, expected in manifest["inputs_sha256"].items())
    result = {
        "independent_csv_counter_and_statistics_checks": "passed",
        "all_13_statistics_tables_reconciled": "passed",
        "raw_age_not_imputed_age_used": "passed",
        "missing_field_overlap_and_denominators_verified": "passed",
        "lesion_multiplicity_weighted_total_verified": "passed",
        "six_300dpi_pngs_and_six_svg_files_valid": "passed",
        "all_22_statistics_report_requirements_and_png_files_byte_identical_on_reproduction": "passed",
        "reproduced_svg_structure_and_canvas_dimensions_valid": "passed",
        "svg_byte_identity_required": False,
        "existing_version_overwrite_rejected": "passed", "all_inputs_unchanged": "passed",
        "heldout_scope": "split sizes and class counts only",
        "model_training_performed": False,
    }
    (ROOT / "docs/eda_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deps-dir", type=Path)
    run(parser.parse_args().deps_dir)
