"""Freeze train-fit metadata features; image transforms remain lazy.

Usage: python3 Final_Project_A/scripts/preprocess_data.py
Reproduce with --output-dir /private/tmp/ham10000_processed_repro --no-report
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont, ImageOps

from preprocessing import default_image_config, fit_metadata, prepare_pil_image, transform_metadata


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/v1_seed42"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_json(path, content):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(content, ensure_ascii=False, indent=2) + "\n")


def make_preview(train_rows, image_config):
    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    canvas = Image.new("RGB", (840, 2040), "#f5f5f3")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((15, 10), "HAM10000 preprocessing: training examples only", fill="black", font=title_font)
    for i, heading in enumerate(["Original (full view)", "Fixed resize and crop", "Train augmentation, epoch 0"]):
        draw.text((i * 280 + 12, 46), heading, fill="black", font=font)
    for i, label in enumerate(labels):
        row = next(row for row in train_rows if row["dx"] == label)
        path = ROOT / row["relative_path"]
        y = 82 + i * 278
        with Image.open(path) as original:
            thumb = ImageOps.contain(original.convert("RGB"), (256, 224))
            canvas.paste(thumb, (12 + (256 - thumb.width) // 2, y + (224 - thumb.height) // 2))
        for column, split in [(1, "val"), (2, "train")]:
            image = prepare_pil_image(path, split, image_config, image_id=row["image_id"], epoch=0, seed=42)
            canvas.paste(image, (column * 280 + 28, y))
        draw.text((15, y + 230), f"{label} | {row['image_id']} | lesion: {row['lesion_id']}", fill="black", font=font)
    canvas.save(ROOT / "figures/preprocessing_train_preview.png")


def run(output, report=True):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing processed version is protected: {output}")
    checksums = json.loads((SPLITS / "checksums.json").read_text())
    if not all(digest(SPLITS / name) == expected for name, expected in checksums.items()):
        raise ValueError("Frozen split checksum mismatch")
    fields, train = read_rows(SPLITS / "train.csv")
    metadata_config = fit_metadata(train)  # val/test have not been read for fitting.
    config = {
        "schema_version": "ham10000-preprocessing-v1", "split_version": "v1_seed42",
        "team_alignment_status": "A first proposal; confirm image weights and recipe with B",
        "metadata": metadata_config, "image": default_image_config(),
        "source_sha256": {"train.csv": digest(SPLITS / "train.csv"),
                          "val.csv": digest(SPLITS / "val.csv"), "test.csv": digest(SPLITS / "test.csv"),
                          "split_checksums.json": digest(SPLITS / "checksums.json"),
                          "preprocessing.py": digest(ROOT / "scripts/preprocessing.py"),
                          "preprocess_data.py": digest(Path(__file__))},
    }
    schema = {"feature_order": metadata_config["feature_order"], "n_features": len(metadata_config["feature_order"]),
              "feature_modes": metadata_config["feature_modes"],
              "audit_only_added_columns": ["age_missing", "age_imputed"],
              "retained_original_columns": fields,
              "label_mapping": json.loads((SPLITS / "label_mapping.json").read_text()),
              "model_input_policy": "Use only explicitly listed features; do not use all numeric CSV columns",
              "relative_path_root": "Final_Project_A"}
    transformed = {}
    for split in ["train", "val", "test"]:
        split_fields, rows = (fields, train) if split == "train" else read_rows(SPLITS / f"{split}.csv")
        if split_fields != fields or any(row["split"] != split for row in rows):
            raise ValueError(f"Wrong schema or split membership in {split}")
        transformed[split] = [{**row, **transform_metadata(row, metadata_config)} for row in rows]
        if any({field: processed[field] for field in fields} != original for original, processed in zip(rows, transformed[split])):
            raise ValueError("Original fields changed during preprocessing")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in transformed.items():
        with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + ["age_missing", "age_imputed"] + metadata_config["feature_order"])
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "preprocessing_config.json", config)
    write_json(output / "feature_schema.json", schema)
    write_json(output / "checksums.json", {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    summary = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "fit_split": "train", "fit_n_rows": len(train), "n_features": schema["n_features"],
               "age_parameters": metadata_config["age"],
               "split_rows": {split: len(rows) for split, rows in transformed.items()},
               "age_missing": {split: sum(row["age_missing"] for row in rows) for split, rows in transformed.items()},
               "feature_mode_dimensions": {mode: len(columns) for mode, columns in metadata_config["feature_modes"].items()},
               "images_processed_lazily": True, "model_training_performed": False}
    if report:
        make_preview(train, config["image"])
        (ROOT / "docs/preprocessing_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        age = metadata_config["age"]

        lines = ["# HAM10000 preprocessing guide", "", f"Generated: {summary['generated_at']}", "",
            "Age imputation, standardization and categorical vocabularies are fitted on the frozen training image records only. The three processed CSVs retain all original fields and append explicitly ordered model features. Image transforms are applied lazily; raw JPEGs are unchanged.", "",
            "## Training age parameters", "", "| Parameter | Value |", "| --- | ---: |",
            f"| Training image records | {len(train)} |", f"| Observed / missing training ages | {age['observed_n']} / {age['missing_n']} |",
            f"| Imputation median | {age['median']} |", f"| Mean after imputation | {age['mean_after_imputation']:.12f} |",
            f"| Population standard deviation after imputation | {age['population_std_after_imputation']:.12f} |", "",
            "`age_z = (age_imputed - train_mean) / train_std`. Parameters use image records, so a multi-image lesion contributes more than one row. Validation/test ages are not used to estimate these parameters.", "",
            "## Outputs and feature dimensions", "", "| Subset | Records | Missing original ages |", "| --- | ---: | ---: |"]
        lines += [f"| {split} | {len(rows)} | {summary['age_missing'][split]} |" for split, rows in transformed.items()]
        lines += ["", f"The full metadata vector has {schema['n_features']} features in this order:", "",
            ", ".join(f"`{name}`" for name in schema["feature_order"]), "",
            "Feature modes: `all_metadata=19`, `age_only=1`, `localization_only=15`, `image_only=0`. Use `feature_modes` in the saved schema to select columns. Original age, missingness/imputation audit columns, labels and identifiers are not additional model features. Blank or unseen categorical values map to the reserved `unknown` category without refitting.", "",
            "## Image transforms", "",
            "The current recipe targets `ResNet50_Weights.IMAGENET1K_V2`: resize the shorter side to 232 with bilinear interpolation, centre-crop to 224 x 224, scale RGB values to [0,1], then normalize with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Output is float32 CHW with shape (3, 224, 224). Training additionally uses horizontal flips and right-angle rotations; validation/test transforms are deterministic.", "",
            "Reference: [TorchVision ResNet-50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html). The weight version and recipe still require agreement with the training member.", "",
            "The implementation uses Pillow and NumPy. Pixel equivalence with TorchVision has not been tested. Centre cropping can remove lesion boundaries; the [training preview](../figures/preprocessing_train_preview.png) illustrates the transform but does not establish preservation of all lesion content.", "",
            "## Training interface", "",
            "Use the processed CSVs, `preprocessing_config.json` and `feature_schema.json` under `data/processed/v1_seed42/`. Do not pass all numeric columns to the model or normalize already-normalized outputs a second time.", "",
            "`scripts/preprocessing.py` exposes `sample_arrays(row, config, data_root=..., mode=..., epoch=..., seed=...)`. It returns the image array, metadata array, label and identifiers. A training Dataset can wrap this function. Update the epoch for training augmentation; synchronize it with persistent data workers if used.", "",
            "For subgroup analysis, retain original `age`, `sex`, `localization` and `age_missing`, and join predictions on `image_id`. Imputed age is not an observed age for subgroup definitions.", "",
            "## Verification and scope", "",
            "The [preprocessing validation record](preprocessing_validation.json) covers independently calculated age parameters, one-hot vectors, preservation of source fields, training-only fitting, unseen-category fallback, six-file deterministic reproduction, overwrite protection, RGB normalization and all 10,015 image inputs. See the [method](06_preprocessing_method.md) for design details.", "",
            "Source hashes and output checksums identify this version. No model training or model-performance result is included in this stage.", ""]
        (ROOT / "docs/07_preprocessing_guide.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/v1_seed42")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), not args.no_report)
