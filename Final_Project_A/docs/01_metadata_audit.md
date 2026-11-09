# HAM10000 metadata audit

Generated: 2026-11-09T18:50:33.517400+08:00

The original CSV matches its recorded official MD5. The CSV and default TAB export contain identical ordered records. Image identifiers and diagnosis labels pass the checks below. Missing metadata and within-lesion inconsistencies are retained and documented.

## Source and verification

The source is HAM10000 V4.0 on Harvard Dataverse, DOI 10.7910/DVN/DBW86T, metadata file ID 4338392. The saved dataset record and provenance files are in `data/raw/`.

- Original CSV MD5: `8f85fb1aa29d80a2797247e434deb79d`.
- The default endpoint returns TAB. Add `?format=original` to obtain the CSV to which the official checksum applies. Different file bytes do not imply different parsed records.
- This audit does not edit, drop or impute raw records.
- Run `python3 scripts/audit_metadata.py` from the A-stage directory. This script uses the Python standard library.

## Field roles

| Field | Meaning | Use |
| --- | --- | --- |
| `image_id` | Image identifier | Join images, manifests and predictions; not a predictor |
| `lesion_id` | Lesion identifier; one lesion can have several images | Group the split; not a patient identifier or predictor |
| `dx` | Seven-class diagnosis | Prediction target; never an input feature |
| `age` | Recorded age | Candidate predictor; fit imputation on training records only |
| `sex` | Recorded sex | Candidate predictor; retain unknown as a category |
| `localization` | Recorded lesion site | Candidate predictor; retain original per-image values |
| `dx_type` | Diagnostic confirmation method | Descriptive and audit use only |
| `dataset` | Source subset | Descriptive and audit use only |

## Image and lesion counts

There are 10,015 records, 10,015 distinct image IDs and 7,470 distinct lesion IDs. 1,956 lesions have multiple images, with 1 to 6 images per lesion.

Each row represents an image. Image counts are not counts of distinct lesions or patients. Repeated lesion IDs identify multiple views and are not automatically deleted. The metadata has no patient identifier, so patient-level separation cannot be established.

| Diagnosis | Images | Image proportion | Distinct lesions |
| --- | ---: | ---: | ---: |
| nv | 6,705 | 66.95% | 5,403 |
| mel | 1,113 | 11.11% | 614 |
| bkl | 1,099 | 10.97% | 727 |
| bcc | 514 | 5.13% | 327 |
| akiec | 327 | 3.27% | 228 |
| vasc | 142 | 1.42% | 98 |
| df | 115 | 1.15% | 73 |

Class imbalance motivates reporting macro-F1 and class-specific performance alongside accuracy. This audit reports dataset structure, not model performance.

## Missing and unknown values

Counts below use image records. Blank values and the literal category `unknown` are counted separately.

| Field | Blank records | Unknown records |
| --- | ---: | ---: |
| age | 57 | 0 |
| sex | 0 | 57 |
| localization | 0 | 234 |

Observed ages range from 0 to 85 years and occur in five-year increments. Age zero is retained as an observed value. No populated age failed numeric parsing or the 0-120 range check.

Raw missingness is preserved. The preprocessing stage uses training-only age imputation and explicit unknown categories; see [Preprocessing guide](07_preprocessing_guide.md).

## Within-lesion consistency

Diagnosis, age and sex are consistent within each lesion. The following fields have differing records:

| Field | Lesions with multiple values |
| --- | ---: |
| localization | 2 |
| dataset | 9 |
| dx_type | 0 |

Location discrepancies:

- `HAM_0000871`: chest / trunk; images: ISIC_0025964, ISIC_0030623, ISIC_0026506.
- `HAM_0001726`: back / trunk; images: ISIC_0029281, ISIC_0024367, ISIC_0024700.

Per-image source values are retained. A lesion appearing in multiple source subsets is not automatically treated as a separate lesion or an error. The full discrepancy list is in [metadata_audit.json](metadata_audit.json).

## Audit checks

- PASS: `original_checksum_matches_official`.
- PASS: `csv_tab_records_identical`.
- PASS: `expected_10015_records`.
- PASS: `no_duplicate_image_ids`.
- PASS: `no_missing_image_or_lesion_ids`.
- PASS: `expected_seven_labels`.
- PASS: `consistent_diagnosis_per_lesion`.
- PASS: `age_numeric_and_in_0_to_120_when_present`.

## Scope

This script checks metadata only. Flags in its JSON describe this script's actions, not the completion status of the entire project. Image integrity, split integrity and feature preparation have separate reports.

Sources: [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), [original CSV](https://dataverse.harvard.edu/api/access/datafile/4338392?format=original), [dataset paper](https://arxiv.org/abs/1803.10417).
