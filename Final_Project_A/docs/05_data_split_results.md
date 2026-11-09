# HAM10000 lesion-level split results

Generated: 2026-11-09T18:50:47.146300+08:00

The split groups images by `lesion_id` and stratifies lesions by `dx`. Version `v1_seed42` uses seed 42 and target lesion proportions of 70% / 15% / 15%. Original image records and metadata fields are retained. No image IDs, lesion IDs or exact-byte image hashes overlap between subsets.

The ratio, seed and label mapping are A-stage implementation choices pending coordination with the training member. They are not course-mandated settings. No model results were used to choose the seed.

| Subset | Lesions | Lesion proportion | Images | Image proportion |
| --- | ---: | ---: | ---: | ---: |
| train | 5,229 | 70.00% | 7,050 | 70.39% |
| val | 1,121 | 15.01% | 1,474 | 14.72% |
| test | 1,120 | 14.99% | 1,491 | 14.89% |

## Counts by diagnosis

| Diagnosis | Train images / lesions | Validation images / lesions | Test images / lesions |
| --- | ---: | ---: | ---: |
| akiec | 230 / 160 | 51 / 34 | 46 / 34 |
| bcc | 365 / 229 | 79 / 49 | 70 / 49 |
| bkl | 762 / 509 | 178 / 109 | 159 / 109 |
| df | 86 / 51 | 14 / 11 | 15 / 11 |
| mel | 788 / 430 | 156 / 92 | 169 / 92 |
| nv | 4724 / 3782 | 968 / 811 | 1013 / 810 |
| vasc | 95 / 68 | 28 / 15 | 19 / 15 |

## Integrity checks

- PASS: `every_original_image_assigned_once`.
- PASS: `all_original_fields_preserved`.
- PASS: `all_lesions_assigned_once`.
- PASS: `all_splits_contain_seven_classes`.
- PASS: `no_exact_byte_duplicates_cross_splits`.
- PASS: `lesion_counts_match_targets`.
- PASS: `no_train_val_lesion_overlap`.
- PASS: `no_train_val_image_overlap`.
- PASS: `no_train_test_lesion_overlap`.
- PASS: `no_train_test_image_overlap`.
- PASS: `no_val_test_lesion_overlap`.
- PASS: `no_val_test_image_overlap`.

## Using this split

Read `train.csv`, `val.csv` and `test.csv` from `data/splits/v1_seed42/`. Use the same version for every model and ablation. Resolve `relative_path` against the A-stage directory. Join predictions by `image_id`, never by row position.

Candidate predictors are the image, age, sex and localization. `dx` and `label` are targets. Image/lesion identifiers, split membership, `dx_type`, `dataset` and image hashes are audit fields, not predictors.

The label mapping follows lexical diagnosis order. `split_config.json` records the allocation algorithm and source hashes; `checksums.json` identifies the frozen files. Matching seeds alone does not guarantee matching samples.

## Reproduction and limitations

The saved [validation record](split_validation.json) documents independent field, coverage and overlap checks, deterministic reproduction, input-order invariance, diagnosis-conflict rejection and overwrite protection. See the [split method](04_data_split_method.md) for the exact algorithm.

Image proportions differ slightly from the lesion targets because lesions have different numbers of images. Known exact-byte duplicate pairs are retained within their lesion. Lesion-level separation does not establish patient-level separation or rule out unidentified near-duplicates.

Raw missingness is preserved in these split files. The separate [preprocessing stage](07_preprocessing_guide.md) fits its parameters on training records only. Validation supports model selection; test outcomes must remain outside tuning and model selection.
