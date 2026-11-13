# How to run this code

## Important: the copy inside a delivery package cannot run on its own

The scripts resolve the A-stage inputs through paths **relative to the repository
root**. Running a loose copy of `scripts/` elsewhere fails immediately with:

```
input check failed:
  A-stage split tables missing: ...\Final_Project_A\data\splits\v1_seed42
  A-stage processed tables missing: ...\Final_Project_A\data\processed\v1_seed42
  ...
```

That is intentional: an explicit failure beats silently computing something wrong.

## Correct usage

Restore `scripts/` and `configs/` into the repository layout:

```
6018_Team_Project/
├── Final_Project_A/          # stage A inputs (audits, frozen split, features)
└── Final_Project_B/
    ├── scripts/              # from this folder
    ├── configs/              # from this folder
    ├── requirements.txt
    └── outputs/              # created on the first run (git-ignored)
```

Then, from `Final_Project_B/`:

```powershell
python -m pip install -r requirements.txt

python scripts/prepare_images.py                 # one-off: hard-link the 10,015 JPEGs
python scripts/run_all.py                        # the four headline experiments + summary.csv
python scripts/run_seeds.py                      # 4 experiments x 3 seeds, mean +- std, paired deltas
python scripts/ensemble_seeds.py --all-splits    # three-seed probability ensemble
python scripts/enrich_predictions.py             # merge the subgroup fields into the predictions
python scripts/audit_runs.py                     # completeness audit (expect ALL CHECKS PASSED)
python scripts/dump_metrics.py                   # every metric into one markdown report
```

Binary task and leakage controls (**not** our main results, only used to explain
the published 96% figures):

```powershell
python scripts/binary_report.py --from-multiclass fusion_all_metadata --tta
python scripts/make_control_split.py --protocol image_level
python scripts/train.py --config control_image_level
python scripts/analyze_leakage.py --run control_image_level `
    --train-split data/control_splits/image_level_seed42/train.csv
```

Assemble the hand-over package (writes outside the repository on purpose):

```powershell
python scripts/build_delivery.py
```

## Script responsibilities

| Script | Purpose | Main path |
|---|---|---|
| `config.py` | config loading, path resolution, input checks | library |
| `data.py` | dataset, image transforms, data loaders | library |
| `models.py` | ResNet-50 baseline, two-stream fusion, sequential CNN | library |
| `metrics.py` | accuracy / macro-F1 / OvR-AUC / confusion matrix / TTA | library |
| `engine.py` | training loop, AMP, early stopping, checkpoints, prediction export | library |
| `prepare_images.py` | unzip + hard-link the JPEGs + sampled checksum verification | yes |
| `train.py` | train one experiment, evaluate, export predictions | yes |
| `evaluate.py` | re-run evaluation from a checkpoint | yes |
| `run_all.py` | the four headline experiments + comparison table | yes |
| `run_seeds.py` | multiple seeds, mean +- std, paired deltas | yes |
| `ensemble_seeds.py` | seed-probability ensemble | yes |
| `enrich_predictions.py` | merge subgroup variables into the prediction tables | yes |
| `benchmark.py` | throughput/VRAM benchmark used to pick the batch size | tool |
| `audit_runs.py` | pre-delivery completeness and consistency audit | tool |
| `dump_metrics.py` | dump every metric into one markdown report | tool |
| `build_delivery.py` | assemble the hand-over package | tool |
| `binary_report.py` | binary (malignant/benign) evaluation and threshold selection | control |
| `make_control_split.py` | build the image-level / oversampled control splits | control |
| `analyze_leakage.py` | exposed-versus-clean subset analysis | control |

## Environment

See `requirements.txt`. Verified with Python 3.12, torch 2.7.1+cu118,
torchvision 0.22.1+cu118, numpy 2.3.5, pandas 3.1.1, scikit-learn 1.9.1,
Pillow 13.0.0, matplotlib 3.11.2. A single RTX 3060 Laptop (6 GB) is enough:
batch 64 with AMP uses about 3.4 GB and takes roughly 30 s per epoch.
