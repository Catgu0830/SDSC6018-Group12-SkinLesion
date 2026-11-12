# B stage — modelling: image-only baseline and metadata fusion

This directory holds the modelling half of Group 12's project: two models trained
on the **frozen** A-stage split, the two planned ablations, and the per-sample
test predictions that the C-stage trade-off analysis consumes.

The research question, the dataset and the evaluation strategy are described in
the [repository README](../README.md); the split, features and EDA come from
[`Final_Project_A`](../Final_Project_A/README.md). This package adds no data
preparation of its own — it reads A's checksummed tables so that the models never
disagree with the frozen split.

## What this stage produces

| Output | Where | Why it matters |
| --- | --- | --- |
| Image-only ResNet-50 baseline | `outputs/baseline_image/` | The reference every fusion result is compared against |
| Full-metadata two-stream fusion | `outputs/fusion_all_metadata/` | The model under test |
| Ablations: age only, localization only | `outputs/fusion_age_only/`, `outputs/fusion_localization_only/` | Separates *which* metadata carries the signal |
| Per-sample predictions (`predictions_test.csv`) | each run directory | Enables the Benefit / Neutral / Harm partition in C |
| Global metrics + confusion matrices | each run directory | Required by the report |
| `outputs/summary.csv` | `outputs/` | One table comparing all four experiments |

## Inputs (read-only, from the A stage)

Everything below is produced by A and must not be edited:

```text
Final_Project_A/data/splits/v1_seed42/          frozen train/val/test lists
Final_Project_A/data/processed/v1_seed42/       19 metadata features + feature_schema.json
Final_Project_A/data/image_manifest.csv         image inventory (bytes + SHA-256)
Final_Project_A/data/raw/images/<image_id>.jpg  raw JPEGs, git-ignored
```

`relative_path` in A's tables is relative to `Final_Project_A/`, so images must sit
at `Final_Project_A/data/raw/images/`. They are not in the repository; run the
first command below once per machine.

## Quick start

```bash
# 0) environment: torch + torchvision (CUDA build), pandas, scikit-learn, Pillow, matplotlib
python -m pip install -r requirements.txt

# 1) one-time: unzip part 2 and hard-link all 10,015 JPEGs into A's expected path
python scripts/prepare_images.py            # add --verify-sample 1000 for a wider checksum audit

# 2) one experiment
python scripts/train.py --config baseline_image

# 3) all four experiments, then the comparison table
python scripts/run_all.py

# 4) re-evaluate a finished run without retraining
python scripts/evaluate.py --run fusion_all_metadata --split test
```

Paths are resolved from the module location, so the commands work from any
directory as long as `scripts/` is on the path (running them as `python
scripts/train.py` from this folder is the intended usage).

## Layout

```text
Final_Project_B/
├── README.md
├── requirements.txt
├── configs/                     one JSON per experiment, sharing configs/common.json
│   ├── common.json              image recipe, batch size, optimiser, seed
│   ├── baseline_image.json      ablation 1
│   ├── fusion_all_metadata.json ablation 2
│   ├── fusion_age_only.json     ablation 3
│   └── fusion_localization_only.json  ablation 4
├── scripts/                     all Python lives here
│   ├── config.py                config loading, path resolution, input checks
│   ├── data.py                  dataset, transforms, loaders
│   ├── models.py                ResNet-50 baseline and two-stream fusion
│   ├── metrics.py               accuracy / macro-F1 / OvR-AUC / confusion matrix
│   ├── engine.py                training loop, AMP, early stopping, checkpoints
│   ├── prepare_images.py        unzip + hard-link images, checksum sampling
│   ├── train.py                 train one experiment, evaluate, export predictions
│   ├── evaluate.py              re-run evaluation from a checkpoint
│   └── run_all.py               all four experiments + summary table
└── outputs/                     every artifact (git-ignored)
```

## Design decisions (and why)

**Identical training conditions.** Every experiment uses the same split, the same
ResNet-50 initialisation (`IMAGENET1K_V2`), the same optimiser, batch size, epoch
budget and augmentation. The only thing that varies is the metadata stream — so a
difference in accuracy is attributable to the metadata, not to tuning drift. This
is the precondition for the paired per-sample comparison in stage C.

**Frozen features.** The 19 metadata columns are taken from A's processed tables
(fitted on training records only), not recomputed here. Re-deriving them would
risk a silent divergence from A's checksums and would duplicate leakage control
that A already verified.

**Augmentation.** Horizontal flip (p=0.5) and a rotation drawn from
{0°, 90°, 180°, 270°}, applied after the centre crop. Quarter turns are lossless
on a square crop and dermatoscopic images carry no canonical orientation. A's
reference implementation selects the transform by a hash of (seed, epoch, image
id); this stage draws it at random, which is statistically equivalent for
training. Validation and test images use no augmentation at all.

**Class imbalance.** Plain cross-entropy, matching the proposal, with balanced
accuracy handled at report time by macro-F1 and one-vs-rest AUC. `train.loss` in
`configs/common.json` can switch to inverse-frequency or square-root-inverse
class weights (`"none"`, `"inverse"`, `"sqrt_inverse"`); if that switch is used,
it must be applied to **all four** experiments or the comparison is void.

**Model architecture.** The fusion model concatenates the 2048-d pooled visual
features with a 128-d metadata embedding (two linear + BatchNorm + ReLU + dropout
blocks), then applies one 256-d fully connected layer with dropout before the
7-class output. This is the "mid-level feature concatenation" described in the
proposal.

**Reproducibility.** One seed (`train.seed`) seeds Python, NumPy and Torch; the
resolved configuration is written to `config.json` in each run directory;
checkpoints store the config and feature-column list, so `evaluate.py` can rebuild
the exact model without reading the YAML of the day. CUDA kernel selection stays
non-deterministic (`cudnn.benchmark`), so bit-identical reruns are not claimed.

## Output contract for stage C

`outputs/<run>/predictions_test.csv` (and `predictions_val.csv`) has one row per
image:

| column | meaning |
| --- | --- |
| `image_id`, `lesion_id` | join keys — C must join on `image_id`, never on row order |
| `dx`, `label`, `y_true` | ground truth (label = frozen 0–6 index) |
| `y_pred` | argmax prediction |
| `prob_akiec` … `prob_vasc` | softmax probabilities in the frozen class order |
| `correct` | 1 if `y_pred == y_true` |

Joining the baseline and fusion tables on `image_id` yields exactly the
Benefit (baseline wrong, fusion right) / Harm (baseline right, fusion wrong) /
Neutral partition.

## Status

- [x] Environment verified: CUDA available, torch 2.7.1+cu118 on an RTX 3060
- [x] Image materialisation (`prepare_images.py`), 10,015/10,015 linked, checksum sample verified
- [x] Baseline, fusion model, metrics, training loop, prediction export
- [x] Baseline trained and evaluated (3 seeds + ensemble)
- [x] Fusion model trained (all metadata, age only, localization only), 3 seeds each
- [x] Age-only and localization-only ablations complete
- [x] Seed robustness: mean +- std and paired deltas vs baseline
- [x] Seed ensemble (3 members) with val/test predictions
- [x] `audit_runs.py` passes: every artefact present, metrics recomputed from the
      exported predictions match, one identical split across all 16 runs, configs
      differ only in the metadata stream
- [x] `results.md` written with the headline tables and the caveats

Key numbers (test, lesion-level split `v1_seed42`):

| model | Acc | macro-F1 | AUC |
| --- | --- | --- | --- |
| image-only baseline (3-seed ensemble) | 0.8303 | 0.6608 | 0.9494 |
| full-metadata fusion (3-seed ensemble) | **0.8417** | **0.7007** | **0.9596** |

Paired delta of the fusion over the baseline, averaged across seeds:
macro-F1 **+0.0383 +- 0.0103**, AUC **+0.0158 +- 0.0057**, accuracy +0.0058 +- 0.0051
(not significant). See `results.md` for the per-class breakdown, the class-weight
sensitivity analysis and the leakage controls.

## Points still to settle with the team

1. A's split config still says the ratio (70/15/15), seed (42) and label order are
   "not yet confirmed with B". They must be confirmed **before** the first
   headline run, because changing them invalidates every prediction table.
2. Centre cropping to 224 px from a 600×450 source discards the left/right border;
   lesions near the frame edge can be cut. A `Resize(224×224)` squash is the
   alternative — a decision to take jointly, since A reports the recipe it fitted.
3. Class weighting (see above) is currently off; if the rare classes (`df`, `vasc`)
   prove unlearnable, decide whether to weight all four experiments or none.
