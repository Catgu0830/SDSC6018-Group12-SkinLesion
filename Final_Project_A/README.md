# HAM10000 data preparation and exploratory analysis

This directory contains Group 12's A-stage data audit, lesion-level split, preprocessing and exploratory data analysis. Team members and the research question are listed in the [repository README](../README.md).

The dataset contains 10,015 images from 7,470 distinct lesion IDs and seven diagnoses. All original records are retained. The frozen `v1_seed42` split has 7,050 training, 1,474 validation and 1,491 test images. Split settings and the image recipe await coordination with the training member.

## Documentation

| Topic | Document |
| --- | --- |
| Source, fields and metadata quality | [Metadata audit](docs/01_metadata_audit.md) |
| Image verification | [Workflow](docs/02_image_validation_workflow.md), [results](docs/03_image_validation_results.md) |
| Grouping, stratification and reproducibility | [Split method](docs/04_data_split_method.md), [results](docs/05_data_split_results.md) |
| Training-only fitting and model inputs | [Preprocessing method](docs/06_preprocessing_method.md), [usage](docs/07_preprocessing_guide.md) |
| Exploratory analysis | [EDA method](docs/08_eda_method.md), [results](docs/09_eda_results.md), [report](output/eda/v1_seed42/eda_report.md) |
| Work performed and assistance | [Contribution record](docs/contributions.md) |

English documents are the maintained reference. A [Chinese collaboration guide](docs/zh-CN/data_preparation_guide.md) explains the complete handover for teammates. It is supplementary and can be removed, together with this paragraph, before the final English submission.

## Files and input roles

```text
data/raw/                  Metadata and source records; local JPEGs/ZIPs are ignored
data/image_manifest.csv    Image IDs, lesion IDs, paths, dimensions and SHA-256
data/splits/v1_seed42/      Fixed lists, labels, configuration and checksums
data/processed/v1_seed42/   Feature tables, fitted parameters and schema
scripts/                   Audit, split, preprocessing, EDA and validation programs
docs/                      Methods, results and validation records
figures/                   Original-image examples and transform preview
output/eda/v1_seed42/       Six figure groups, 13 tables, report and provenance
```

Predictors are the image and selected age/sex/site features. `dx`/`label` are targets; IDs, split membership, source, confirmation method and hashes are audit fields. Select features with `feature_schema.json`; do not use all numeric columns as predictors. Predictions must retain `image_id` for comparison and subgroup analysis.

## Raw data setup

Source: [HAM10000 on Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T), saved dataset version 4.0. Consult `data/raw/harvard_dataset_record_v4.json` for the source record and usage terms.

Raw image archives and JPEGs are not committed. For archive validation, place `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` in `data/raw/`. The image audit verifies and extracts them to `data/raw/images/`. To consume the frozen split, place all 10,015 original JPEGs directly in that image directory; paths must be `data/raw/images/<image_id>.jpg` relative to this directory. Verify bytes against the image manifest.

The repository's existing `dataverse_files/` layout is separate from this package. Images there alone do not satisfy the relative paths used here.

## Running the programs

Use Python 3.9 or later. Metadata auditing and splitting use the standard library; image checking needs Pillow; preprocessing needs NumPy and Pillow; the validation programs also use pandas. EDA uses pandas, NumPy and Matplotlib. Recorded versions are in [requirements_eda.txt](output/eda/v1_seed42/requirements_eda.txt).

Change into `Final_Project_A/` in a terminal. Audit programs regenerate their own reports. Frozen data/output directories are protected against overwriting; use new empty directories to reproduce them:

```bash
python3 scripts/audit_metadata.py
python3 scripts/audit_images.py
python3 scripts/split_data.py --output-dir /tmp/ham_split_check --no-report
python3 scripts/preprocess_data.py --output-dir /tmp/ham_processed_check --no-report
python3 scripts/validate_preprocessing.py
python3 scripts/eda.py --output-dir /tmp/ham_eda_check --no-index
python3 scripts/validate_eda.py
```

Choose unused output paths for repeated runs. `--deps-dir` supports isolated plotting dependencies. Split reproduction uses the checked-in image manifest; preprocessing and EDA use the checked-in frozen split. Regenerate linked provenance records when a producing script changes.

## Status and limitations

Data preparation and EDA are complete for this version. Parameter fitting uses training image records only; validation/test metadata does not determine imputation or vocabularies. Numeric checks and deterministic reproduction are recorded in `docs/`. All 10,015 image inputs have been checked. SVG structure/canvas dimensions are checked, but identical SVG bytes are not required.

This package contains no model-performance results. Lesion separation does not establish patient independence, and exact-byte checks do not exclude visual near-duplicates. Centre cropping can lose lesion boundaries. Pillow/NumPy transforms have not been tested for pixel equivalence with TorchVision. The training member must confirm the split version, label order, backbone weights and transforms before model comparisons.
