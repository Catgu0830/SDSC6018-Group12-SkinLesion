# Fusion of Dermoscopic Images and Patient Metadata for Skin Lesion Classification

### A Sample-Level Benefit–Cost Trade-off Analysis

**SDSC 6018 Health Data Analysis — Final Project (Semester A, 2026/27)**

| | |
|---|---|
| **Group** | 12 |
| **Gu Junyang** | 60238140 |
| **Wang Mingda** | 61120607 |
| **AN Qixuan** | 60907001 |
| **SONG Yutong** | 60863048 |

---

## 1. Research question

Skin lesion screening is critical for early melanoma diagnosis. Clinicians combine dermoscopic
visuals with patient metadata (age, sex, lesion location), and prior multimodal work fuses the two
for 7-class classification on HAM10000. Such studies report **global** metrics — accuracy, macro-F1,
one-vs-rest AUC — even when they use leakage-free, lesion-level splits. Global metrics, however,
average over patients: an overall gain can hide subgroups that improve and subgroups that get worse.

> **Under a lesion-level train/validation/test split on HAM10000, is the performance change caused
> by metadata fusion evenly distributed across test samples? Do some subgroups benefit while others
> degrade?**

**Hypothesis.** Metadata-driven performance shifts are non-uniform. Even when global metrics
improve, gains and harms cluster within specific disease, age, sex or lesion-location subgroups, so
global-only evaluation is insufficient for medical-AI assessment.

**What is new here.** The contribution is not a new metric or a new architecture. It is a
*stratified evaluation protocol*: every held-out test sample is assigned to one of three disjoint
sets based on the paired behaviour of the two models, and those sets are then characterised.

| Subset | Image-only model | Metadata-fusion model |
|---|---|---|
| **Benefit** | wrong | correct |
| **Harm** | correct | wrong |
| **Neutral** | both correct / both wrong | both correct / both wrong |

## 2. Data

Public [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (Tschandl et al., *Scientific Data*, 2018) —
no IRB required.

- **10,015** dermoscopic images of **7,470** distinct lesions, **7** diagnostic classes
- **Modality 1:** dermoscopic image
- **Modality 2:** tabular metadata — `age`, `sex`, `localization`, `lesion_id`, `dx`

Class distribution is strongly imbalanced and drives most of the modelling choices below:

| `dx` | nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|---|
| images | 6705 | 1113 | 1099 | 514 | 327 | 142 | 115 |

Known gaps: `age` missing for 57 images, `sex = unknown` for 57, `localization = unknown` for 234.
Images per lesion: mean 1.34, max 6 — which is exactly why the split must be **grouped by
`lesion_id`**: splitting by image leaks the same lesion across train and test and inflates results.

### Data setup (files are not committed)

The dataset is deliberately excluded from this repository (GitHub's 100 MB per-file limit, and the
data is public anyway). Reproduce the layout below before running anything:

```text
dataverse_files/
├── HAM10000_metadata                       # tracked — 10,015 rows
├── HAM10000_metadata.xlsx                  # not tracked — one-column export of the above
├── HAM10000_images_part_1/                 # 5,000 .jpg   (download + unzip)
├── HAM10000_images_part_2/                 # 5,015 .jpg   (download + unzip)
├── HAM10000_segmentations_lesion_tschandl/ # optional: lesion masks
├── ISIC2018_Task3_Test_GroundTruth.csv     # tracked — optional external test set
└── ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv
```

Get the images from either source, then unzip both parts into one folder:

- Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>
- Dataverse: <https://doi.org/10.7910/DVN/DBW86T>

After unzipping, `dataverse_files/HAM10000_images_part_1/` + `part_2/` together must contain
**10,015** files whose stems match `HAM10000_metadata.image_id`.

## 3. Method

Supervised 7-class classification. Two models are trained on **identical** train/validation
partitions and evaluated on the same held-out test partition.

1. **Baseline (image-only).** ImageNet-pretrained ResNet-50 with the classification head replaced,
   fine-tuned with cross-entropy loss.
2. **Multimodal (two-stream).** ResNet-50 visual branch; metadata projected to the matching
   dimension by a small MLP; mid-level feature concatenation, followed by fully-connected and
   dropout layers, trained end-to-end.

**Ablations.** (1) image only, (2) image + all metadata, (3) image + age only,
(4) image + localization only.

**Preprocessing.** Image resize and normalization, random flip/rotation augmentation; age
standardised; sex and localization one-hot encoded; **stratified lesion-level** train/val/test
split, so all images of one lesion stay in one partition.

## 4. Evaluation

**Global.** Accuracy, macro-F1, one-vs-rest AUC, confusion matrix.

**Sample-level (core analysis).** For every test image, compare baseline vs. multimodal
correctness to build the disjoint Benefit / Neutral / Harm sets; then compare `dx`, age band, sex
and localization distributions across those sets, and compute the metrics above within each
demographic subgroup. Because Benefit and Harm can be small, subgroup claims are reported with
bootstrap confidence intervals and interpreted as correlational patterns, not causal effects.

## 5. Repository layout

```text
├── README.md
├── dataverse_files/            # HAM10000 metadata + small reference tables (images/archives not committed)
├── 小组计划书/                  # submitted proposal and course hand-outs
├── Final_Project_A/            # stage A: audits, frozen lesion-level split, preprocessing, EDA
│   ├── data/splits/v1_seed42/      # frozen train/val/test lists (the shared contract)
│   ├── data/processed/v1_seed42/   # 19 metadata features fitted on training records only
│   ├── scripts/                    # audit / split / preprocess / EDA / validation
│   └── docs/                       # method, results and validation records
└── Final_Project_B/            # stage B: modelling (this contribution)
    ├── configs/                    # one JSON per experiment, sharing common.json
    ├── scripts/                    # data, models, training loop, evaluation, robustness, audit
    ├── outputs/                    # run artefacts (git-ignored; regenerate with the commands below)
    └── results.md                  # every headline number with its caveats
```

Run artefacts (checkpoints, per-sample predictions, per-run figures) are deliberately
**not committed**: they are large and fully regenerable. Only the small summary tables
(`outputs/deliverables_headline.csv`, `seeds_*.csv`, `ensemble_test.csv`,
`metrics_report.md`) and the ensemble confusion matrices are tracked so a reader can see
the results without training anything.

## 6. Status

- [x] Research question, prior work, dataset, method, timeline (proposal submitted)
- [x] Dataset downloaded and metadata verified (10,015 images / 7,470 lesions)
- [x] Lesion-level stratified split (stage A, `v1_seed42`, zero lesion/image overlap)
- [x] Image-only baseline trained (3 seeds + ensemble)
- [x] Multimodal fusion model trained (3 seeds + ensemble)
- [x] Ablations: age only, localization only (3 seeds each)
- [x] Seed-robustness analysis (mean +- std, paired deltas vs baseline)
- [ ] Benefit / Neutral / Harm partition and subgroup analysis (stage C)
- [ ] Final report, presentation, individual reflections

## 7. References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
2. Arshad, M., et al. (2021). Skin lesion classification using deep learning with integration of clinical metadata. *IEEE Access*.
3. Velaga, S., et al. (2026). Interpretable multimodal fusion for skin lesion classification. *Frontiers in Medicine*.
4. Li, Y., et al. (2026). Multimodal skin lesion classification with Swin Transformer and clinical metadata fusion. arXiv preprint.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
6. A data-driven framework for identifying patient subgroups on which an AI/machine learning model may underperform. (2024). *npj Digital Medicine*, 7(1), 334.

## 8. Acknowledgement

HAM10000 is used under its public Dataverse licence for academic coursework only. All images are
de-identified and publicly released; no ethics approval is required for this analysis.
