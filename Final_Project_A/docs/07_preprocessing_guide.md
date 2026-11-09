# HAM10000 preprocessing guide

Generated: 2026-11-09T18:50:58.760645+08:00

Age imputation, standardization and categorical vocabularies are fitted on the frozen training image records only. The three processed CSVs retain all original fields and append explicitly ordered model features. Image transforms are applied lazily; raw JPEGs are unchanged.

## Training age parameters

| Parameter | Value |
| --- | ---: |
| Training image records | 7050 |
| Observed / missing training ages | 7010 / 40 |
| Imputation median | 50.0 |
| Mean after imputation | 51.754609929078 |
| Population standard deviation after imputation | 16.914031588127 |

`age_z = (age_imputed - train_mean) / train_std`. Parameters use image records, so a multi-image lesion contributes more than one row. Validation/test ages are not used to estimate these parameters.

## Outputs and feature dimensions

| Subset | Records | Missing original ages |
| --- | ---: | ---: |
| train | 7050 | 40 |
| val | 1474 | 10 |
| test | 1491 | 7 |

The full metadata vector has 19 features in this order:

`age_z`, `sex__female`, `sex__male`, `sex__unknown`, `localization__abdomen`, `localization__acral`, `localization__back`, `localization__chest`, `localization__ear`, `localization__face`, `localization__foot`, `localization__genital`, `localization__hand`, `localization__lower_extremity`, `localization__neck`, `localization__scalp`, `localization__trunk`, `localization__unknown`, `localization__upper_extremity`

Feature modes: `all_metadata=19`, `age_only=1`, `localization_only=15`, `image_only=0`. Use `feature_modes` in the saved schema to select columns. Original age, missingness/imputation audit columns, labels and identifiers are not additional model features. Blank or unseen categorical values map to the reserved `unknown` category without refitting.

## Image transforms

The current recipe targets `ResNet50_Weights.IMAGENET1K_V2`: resize the shorter side to 232 with bilinear interpolation, centre-crop to 224 x 224, scale RGB values to [0,1], then normalize with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Output is float32 CHW with shape (3, 224, 224). Training additionally uses horizontal flips and right-angle rotations; validation/test transforms are deterministic.

Reference: [TorchVision ResNet-50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html). The weight version and recipe still require agreement with the training member.

The implementation uses Pillow and NumPy. Pixel equivalence with TorchVision has not been tested. Centre cropping can remove lesion boundaries; the [training preview](../figures/preprocessing_train_preview.png) illustrates the transform but does not establish preservation of all lesion content.

## Training interface

Use the processed CSVs, `preprocessing_config.json` and `feature_schema.json` under `data/processed/v1_seed42/`. Do not pass all numeric columns to the model or normalize already-normalized outputs a second time.

`scripts/preprocessing.py` exposes `sample_arrays(row, config, data_root=..., mode=..., epoch=..., seed=...)`. It returns the image array, metadata array, label and identifiers. A training Dataset can wrap this function. Update the epoch for training augmentation; synchronize it with persistent data workers if used.

For subgroup analysis, retain original `age`, `sex`, `localization` and `age_missing`, and join predictions on `image_id`. Imputed age is not an observed age for subgroup definitions.

## Verification and scope

The [preprocessing validation record](preprocessing_validation.json) covers independently calculated age parameters, one-hot vectors, preservation of source fields, training-only fitting, unseen-category fallback, six-file deterministic reproduction, overwrite protection, RGB normalization and all 10,015 image inputs. See the [method](06_preprocessing_method.md) for design details.

Source hashes and output checksums identify this version. No model training or model-performance result is included in this stage.
