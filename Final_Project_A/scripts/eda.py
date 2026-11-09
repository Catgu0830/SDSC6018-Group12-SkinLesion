"""Create descriptive EDA figures and report from frozen HAM10000 splits.

Usage: python3 Final_Project_A/scripts/eda.py
If dependencies are isolated, add --deps-dir /private/tmp/codex_6018_eda_plotdeps
Dependencies: pandas, NumPy, Matplotlib. No fitting, sample changes or training.
"""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits/v1_seed42"
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SPLITS = ["train", "val", "test"]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, deps_dir=None, write_index=True):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing EDA version is protected: {output}")
    if deps_dir:
        sys.path.insert(0, str(deps_dir.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex_6018_mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex_6018_fontcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True, "savefig.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "ham10000-eda-v1"})
    frozen = json.loads((SPLIT_DIR / "checksums.json").read_text())
    if not all(sha256(SPLIT_DIR / name) == value for name, value in frozen.items()):
        raise ValueError("Frozen split checksum mismatch")
    input_files = [SPLIT_DIR / name for name in ["train.csv", "val.csv", "test.csv", "combined.csv", "checksums.json"]]
    input_files += [ROOT / "data/raw/HAM10000_metadata.csv", ROOT / "data/processed/v1_seed42/preprocessing_config.json"]
    inputs_before = {str(path.relative_to(ROOT)): sha256(path) for path in input_files}
    frames = {split: pd.read_csv(SPLIT_DIR / f"{split}.csv", dtype=str, keep_default_na=False) for split in SPLITS}
    train = frames["train"]
    raw = pd.read_csv(ROOT / "data/raw/HAM10000_metadata.csv", dtype=str, keep_default_na=False)
    merged = pd.concat(frames.values(), ignore_index=True)
    if not merged.image_id.is_unique or set(merged.image_id) != set(raw.image_id):
        raise ValueError("Incomplete or duplicated sample coverage")
    if not merged.set_index("image_id").loc[raw.image_id, raw.columns.drop("image_id")].reset_index().equals(raw):
        # Compare by column name, since reset_index places image_id first.
        check = merged.set_index("image_id").loc[raw.image_id].reset_index()[raw.columns]
        if not check.equals(raw):
            raise ValueError("Original metadata differs from split records")
    for split, frame in frames.items():
        if set(frame["split"]) != {split} or set(frame.dx) != set(LABELS):
            raise ValueError(f"Wrong split membership or class set: {split}")
    if merged.groupby("lesion_id")["split"].nunique().max() != 1 or merged.groupby("lesion_id").dx.nunique().max() != 1:
        raise ValueError("Lesion leakage or conflicting labels")
    n, n_lesions = len(train), train.lesion_id.nunique()
    ages = pd.to_numeric(train.age, errors="coerce")
    masks = pd.DataFrame({"age_missing": ages.isna(), "sex_unknown": train.sex.eq("unknown") | train.sex.eq(""),
                          "localization_unknown": train.localization.eq("unknown") | train.localization.eq("")})
    class_counts = pd.DataFrame({"dx": LABELS})
    class_counts["image_n"] = [int(train.dx.eq(label).sum()) for label in LABELS]
    class_counts["lesion_n"] = [int(train.loc[train.dx.eq(label), "lesion_id"].nunique()) for label in LABELS]
    class_counts["image_pct"] = class_counts.image_n / n * 100
    class_counts["lesion_pct"] = class_counts.lesion_n / n_lesions * 100
    age_counts = ages.dropna().value_counts().sort_index().rename_axis("age_years").reset_index(name="image_n")
    age_groups = []
    for label in LABELS:
        group = ages[train.dx.eq(label)]
        known = group.dropna()
        age_groups.append({"dx": label, "image_n": len(group), "observed_age_n": len(known),
            "missing_age_n": int(group.isna().sum()), "mean": known.mean(), "median": known.median(),
            "q1": known.quantile(.25), "q3": known.quantile(.75), "min": known.min(), "max": known.max()})
    age_by_dx = pd.DataFrame(age_groups)

    def frequencies(field):
        result = train[field].value_counts().rename_axis(field).reset_index(name="image_n")
        result["image_pct"] = result.image_n / n * 100
        return result

    sex, location, source, confirmation = [frequencies(field) for field in ["sex", "localization", "dataset", "dx_type"]]
    missing = pd.DataFrame({"field": list(masks.columns), "image_n": masks.sum().to_numpy(dtype=int)})
    missing["denominator"] = n
    missing["image_pct"] = missing.image_n / n * 100
    missing_dx = []
    for label in LABELS:
        selected = train.dx.eq(label)
        for field in masks:
            count = int(masks.loc[selected, field].sum())
            denominator = int(selected.sum())
            missing_dx.append({"dx": label, "field": field, "missing_or_unknown_n": count,
                               "denominator": denominator, "image_pct": count / denominator * 100})
    missing_by_dx = pd.DataFrame(missing_dx)
    pattern_labels = masks.apply(lambda row: "+".join(field for field in masks if row[field]) or "complete", axis=1)
    patterns = pattern_labels.value_counts().rename_axis("pattern").reset_index(name="image_n")
    per_lesion = train.groupby("lesion_id").size()
    multiplicity = per_lesion.value_counts().sort_index().rename_axis("images_per_lesion").reset_index(name="lesion_n")
    multiplicity["image_n"] = multiplicity.images_per_lesion * multiplicity.lesion_n
    overview, composition = [], []
    for split, frame in frames.items():
        lesion_n = frame.lesion_id.nunique()
        overview.append({"split": split, "image_n": len(frame), "lesion_n": lesion_n,
                         "image_pct_all": len(frame) / len(merged) * 100,
                         "lesion_pct_all": lesion_n / merged.lesion_id.nunique() * 100})
        for label in LABELS:
            selected = frame.loc[frame.dx.eq(label)]
            composition.append({"split": split, "dx": label, "image_n": len(selected),
                "image_pct": len(selected) / len(frame) * 100, "lesion_n": selected.lesion_id.nunique(),
                "lesion_pct": selected.lesion_id.nunique() / lesion_n * 100})
    overview, composition = pd.DataFrame(overview), pd.DataFrame(composition)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir, table_dir = output / "figures", output / "tables"
    figure_dir.mkdir(); table_dir.mkdir()
    tables = {"train_class_counts": class_counts, "train_age_counts": age_counts,
        "train_age_by_dx": age_by_dx, "train_sex_counts": sex, "train_localization_counts": location,
        "train_missingness": missing, "train_missingness_by_dx": missing_by_dx,
        "train_missingness_patterns": patterns, "train_images_per_lesion": multiplicity,
        "train_source_counts": source, "train_diagnosis_confirmation": confirmation,
        "split_overview": overview, "split_class_composition": composition}
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, float_format="%.8f")
    title_suffix = f"Training images n={n:,}; distinct lesions n={n_lesions:,}"
    figure_names = []

    def save(fig, name, footer):
        fig.supxlabel(footer, fontsize=8, color="#444444")
        fig.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        figure_names.append(name)
        print(f"Saved {name} (PNG and SVG)", flush=True)

    def horizontal(ax, names, values, denominator, title, color=BLUE):
        positions = np.arange(len(names))
        colors = [ORANGE if str(name) == "unknown" else color for name in names]
        bars = ax.barh(positions, values, color=colors, height=.65)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.48)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * .018, bar.get_y() + bar.get_height()/2,
                    f"{value:,} ({value/denominator:.1%})", va="center", fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Training class distribution", fontsize=16)
    horizontal(axes[0], LABELS, class_counts.image_n.tolist(), n, f"Image records (n={n:,})")
    horizontal(axes[1], LABELS, class_counts.lesion_n.tolist(), n_lesions, f"Distinct lesions (n={n_lesions:,})", GREEN)
    axes[0].set_xlabel("Images"); axes[1].set_xlabel("Lesions")
    save(fig, "01_class_distribution", "Counts use different units; multiple images from the same lesion are not independent patients.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Recorded age in the training set", fontsize=16)
    axes[0].bar(age_counts.age_years, age_counts.image_n, width=4, color=BLUE)
    axes[0].set(xlabel="Recorded age (years; 5-year values)", ylabel="Images",
                title=f"Observed n={int(ages.notna().sum()):,}; missing n={int(ages.isna().sum())}")
    axes[0].set_xticks(np.arange(0, 90, 10)); axes[0].grid(axis="y", alpha=.2); axes[0].set_axisbelow(True)
    boxes = axes[1].boxplot([ages[train.dx.eq(label)].dropna() for label in LABELS], tick_labels=LABELS,
        patch_artist=True, showfliers=True, medianprops={"color": "black"},
        flierprops={"marker": ".", "markersize": 3, "alpha": .4})
    for box in boxes["boxes"]:
        box.set_facecolor("#b9d9e9")
    axes[1].set(xlabel="Recorded diagnosis", ylabel="Recorded age (years)", title="Observed ages by diagnosis (image records)")
    axes[1].grid(axis="y", alpha=.2); axes[1].set_axisbelow(True)
    save(fig, "02_age_distribution", "Original non-missing ages only; no imputed ages. Boxes show median/IQR; whiskers use 1.5 IQR. Descriptive, not causal.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7), gridspec_kw={"width_ratios": [1, 1.7]})
    fig.suptitle("Recorded sex and lesion location", fontsize=16)
    sex_order = ["female", "male", "unknown"]
    horizontal(axes[0], sex_order, [int(train.sex.eq(value).sum()) for value in sex_order], n, "Sex (image records)")
    horizontal(axes[1], location.localization.tolist(), location.image_n.tolist(), n, "Location (image records)")
    axes[0].set_xlabel("Images"); axes[1].set_xlabel("Images")
    save(fig, "03_sex_and_location", title_suffix + ". Orange marks unknown; category counts do not establish fairness or model performance.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.55]})
    fig.suptitle("Missing or unknown metadata in training", fontsize=16)
    horizontal(axes[0], ["Age: blank", "Sex: unknown", "Location: unknown"], missing.image_n.tolist(), n, f"Field counts / {n:,} images", ORANGE)
    axes[0].set_xlabel("Images (fields may overlap)")
    heat = missing_by_dx.pivot(index="dx", columns="field", values="image_pct").reindex(index=LABELS, columns=masks.columns)
    heat_counts = missing_by_dx.pivot(index="dx", columns="field", values="missing_or_unknown_n").reindex(index=LABELS, columns=masks.columns)
    values = heat.to_numpy(dtype=float)
    im = axes[1].imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=max(float(values.max()), 1))
    axes[1].set_xticks(range(3), ["Age blank", "Sex unknown", "Location unknown"])
    axes[1].set_yticks(range(7), [f"{label} (n={int(train.dx.eq(label).sum())})" for label in LABELS])
    axes[1].set_title("Within-diagnosis image-record rates")
    for i in range(7):
        for j in range(3):
            axes[1].text(j, i, f"{int(heat_counts.iloc[i,j])}\n{values[i,j]:.2f}%", ha="center", va="center",
                         fontsize=9, color="white" if values[i,j] > values.max() * .6 else "#111111")
    fig.colorbar(im, ax=axes[1], label="Records within diagnosis (%)", shrink=.8)
    any_missing = int(masks.any(axis=1).sum())
    overlap = int((masks.sum(axis=1) > 1).sum())
    save(fig, "04_missingness", f"At least one missing/unknown field: {any_missing}/{n} images; more than one: {overlap}. Field counts must not be added as unique images.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.2, 1]})
    fig.suptitle("Training sample multiplicity and provenance", fontsize=16)
    bars = axes[0].bar(multiplicity.images_per_lesion, multiplicity.lesion_n, color=BLUE)
    axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(xlabel="Images per lesion", ylabel="Lesions", title=f"Lesions n={n_lesions:,}")
    axes[0].set_xticks(multiplicity.images_per_lesion); axes[0].set_ylim(0, multiplicity.lesion_n.max()*1.17)
    horizontal(axes[1], source.dataset.tolist(), source.image_n.tolist(), n, "Source (image records)")
    horizontal(axes[2], confirmation.dx_type.tolist(), confirmation.image_n.tolist(), n, "Confirmation (image records)", GREEN)
    axes[1].set_xlabel("Images"); axes[2].set_xlabel("Images")
    save(fig, "05_multiplicity_and_provenance", "A lesion can have multiple images and occur in more than one source subset. Source/confirmation are audit fields, not model inputs.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle("Basic class-composition check across fixed splits", fontsize=16)
    positions, width = np.arange(7), .24
    for ax, metric, title in [(axes[0], "image_pct", "Image-record proportions"), (axes[1], "lesion_pct", "Distinct-lesion proportions")]:
        for i, (split, color) in enumerate(zip(SPLITS, [BLUE, ORANGE, GREEN])):
            values = composition.loc[composition.split.eq(split)].set_index("dx").loc[LABELS, metric]
            ax.bar(positions + (i-1)*width, values, width, label=split, color=color)
        ax.set_xticks(positions, LABELS); ax.set_ylim(0, 85)
        ax.set(xlabel="Recorded diagnosis", ylabel="Within-split records (%)", title=title)
        ax.legend(loc="upper left", frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    save(fig, "06_split_class_composition", "Stratification targets lesions, not images. Held-out use is limited to this basic count check; no seed/model selection was performed.")

    class_map = class_counts.set_index("dx").to_dict("index")
    summary = {
        "analysis_scope": "Training EDA; held-out sets used only for basic sample/class counts. No model results or fairness conclusions.",
        "train_image_n": n, "train_lesion_n": int(n_lesions), "class_counts": class_map,
        "class_imbalance_image_max_min_ratio": float(class_counts.image_n.max()/class_counts.image_n.min()),
        "train_majority_class": class_counts.loc[class_counts.image_n.idxmax(), "dx"],
        "train_majority_image_fraction": float(class_counts.image_n.max()/n),
        "age_original_observed_n": int(ages.notna().sum()), "age_original_missing_n": int(ages.isna().sum()),
        "age_original_observed_mean": float(ages.mean()), "age_original_observed_median": float(ages.median()),
        "age_original_observed_q1": float(ages.quantile(.25)), "age_original_observed_q3": float(ages.quantile(.75)),
        "age_original_observed_min": float(ages.min()), "age_original_observed_max": float(ages.max()),
        "sex_counts": {key: int(value) for key, value in train.sex.value_counts().items()},
        "location_counts": {key: int(value) for key, value in train.localization.value_counts().items()},
        "missing_field_counts": {field: int(masks[field].sum()) for field in masks},
        "any_missing_or_unknown_image_n": any_missing, "multiple_missing_or_unknown_image_n": overlap,
        "lesions_with_multiple_images": int(per_lesion.gt(1).sum()),
        "source_counts": {key: int(value) for key, value in train.dataset.value_counts().items()},
        "confirmation_counts": {key: int(value) for key, value in train.dx_type.value_counts().items()},
        "split_overview": overview.to_dict("records"), "figures": figure_names,
        "model_training_performed": False,
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    nv, df = class_map["nv"], class_map["df"]
    lines = ["# HAM10000 exploratory data analysis", "",
        "This descriptive analysis uses original fields from the frozen split. Training records are the primary EDA population. Validation and test inspection is restricted to basic sizes and class composition. No samples, split assignments or fitted parameters are changed, and no models are trained.", "",
        "## Findings", "",
        f"The training subset contains {n:,} images from {n_lesions:,} distinct lesions. Class imbalance is substantial: nv accounts for {nv['image_n']:,} images ({nv['image_pct']:.2f}%), whereas df contains {df['image_n']} images ({df['image_pct']:.2f}%) from {df['lesion_n']} lesions. The largest-to-smallest image-count ratio is {summary['class_imbalance_image_max_min_ratio']:.2f}. This motivates reporting macro-F1, class-specific metrics and a confusion matrix alongside accuracy.", "",
        f"Of the {n_lesions:,} training lesions, {summary['lesions_with_multiple_images']:,} have more than one image. Image-level metadata summaries therefore repeat some lesion information and do not represent independent patient observations. No patient identifier is available.", "",
        f"Age summaries use {summary['age_original_observed_n']:,} observed values, excluding {summary['age_original_missing_n']} missing values. The median is {summary['age_original_observed_median']:g} years (IQR {summary['age_original_observed_q1']:g}-{summary['age_original_observed_q3']:g}), with range {summary['age_original_observed_min']:g}-{summary['age_original_observed_max']:g} and mean {summary['age_original_observed_mean']:.4f}. Ages occur in five-year increments and age zero is retained. The observed-age mean differs from the preprocessing mean, which includes imputed records.", "",
        f"There are {int(masks.age_missing.sum())} missing ages, {int(masks.sex_unknown.sum())} unknown sex values and {int(masks.localization_unknown.sum())} unknown sites. The fields overlap: {any_missing} distinct images have at least one missing/unknown field and {overlap} have more than one. The field counts must not be added as a count of affected images.", "",
        f"Recorded sex counts are male={int(train.sex.eq('male').sum()):,}, female={int(train.sex.eq('female').sum()):,} and unknown={int(train.sex.eq('unknown').sum())}. The most frequent site, {location.iloc[0]['localization']}, has {int(location.iloc[0].image_n):,} images; the least frequent, {location.iloc[-1]['localization']}, has {int(location.iloc[-1].image_n)}. Small subgroups require cautious interpretation of later performance estimates.", "",
        f"The training data includes four source subsets and four confirmation methods. Histopathology (`histo`) is recorded for {int(train.dx_type.eq('histo').sum()):,} images ({train.dx_type.eq('histo').mean():.2%}). This image-record proportion is not a lesion-level pathology-confirmation rate. Source and confirmation method are audit fields, not predictors.", "",
        "All subsets contain seven diagnoses. Stratification targets lesion proportions, so image proportions can differ from 70% / 15% / 15%. The composition figure is a basic integrity check; the seed was not selected from model or held-out outcomes.", "",
        "## Figures", ""]
    explanations = [
        ("01_class_distribution", "Class distribution", "Image and distinct-lesion counts use different denominators. Minority-class evidence remains limited."),
        ("02_age_distribution", "Observed age", "Both panels exclude missing ages. Imputed ages are not treated as observations. Diagnosis-associated age patterns alone do not establish predictive benefit."),
        ("03_sex_and_location", "Recorded sex and site", "Orange denotes unknown values. These counts describe coverage; they do not measure fairness or model performance."),
        ("04_missingness", "Missing and unknown metadata", "The left panel uses all training images as its denominator. Each heatmap row uses images of that diagnosis. Counts can overlap across fields."),
        ("05_multiplicity_and_provenance", "Multiplicity and provenance", "The multiplicity panel counts lesions; the other panels count image records. A lesion can occur in more than one source subset."),
        ("06_split_class_composition", "Class composition across subsets", "Within-subset proportions are shown separately for images and lesions. This is not a held-out subgroup-effect analysis or model comparison."),
    ]
    for name, heading, explanation in explanations:
        lines += [f"### {heading}", "", explanation, "", f"![{heading}](figures/{name}.png)", "",
                  f"[SVG version](figures/{name}.svg). Underlying counts are available in `tables/`.", ""]
    lines += ["## Interpretation limits", "",
        "These are descriptive dataset results. They do not establish metadata-fusion gains, Benefit/Harm/Neutral counts, diagnostic performance, subgroup fairness or clinical benefit. Patient identifiers and relevant attributes such as skin tone are unavailable. Multiple images per lesion, minority classes, unknown metadata, source variation and confirmation methods should be addressed in the final discussion.", "",
        "## Reproduction", "",
        "Run `python3 scripts/eda.py --output-dir <new-empty-directory>` from the A-stage directory. Dependencies are recorded in `requirements_eda.txt`; `--deps-dir` supports an isolated dependency directory. Existing non-empty outputs are protected. Figures are 300 dpi PNG and text-editable SVG. Tables are CSV, complete statistics are in `eda_summary.json`, and hashes are in `eda_manifest.json`.", "",
        "The English narrative is generated by the script. Regenerate it together with its manifest when changing the report template. Final report integration must reflect the group's actual experiments.", ""]
    (output / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    packages = ["numpy", "pandas", "Pillow", "matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing", "packaging", "python-dateutil", "six"]
    versions = {name: package_metadata.version(name) for name in packages}
    (output / "requirements_eda.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    inputs_after = {str(path.relative_to(ROOT)): sha256(path) for path in input_files}
    if inputs_after != inputs_before:
        raise ValueError("EDA changed an input file")
    manifest = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": sha256(Path(__file__)), "inputs_sha256": inputs_before, "inputs_unchanged": True,
        "python_version": sys.version.split()[0], "packages": versions,
        "figure_dpi": 300, "figures_visually_reviewed": False,
        "outputs_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "eda_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if write_index:

        index = ["# EDA results", "",
            "Six descriptive figure groups and 13 statistical tables are generated from the frozen data. The source files remain unchanged. The analysis does not include model training or fairness results.", "",
            f"[EDA report](../{output.relative_to(ROOT).as_posix()}/eda_report.md)", "",
            f"Training data: {n:,} images and {n_lesions:,} distinct lesions. nv accounts for {nv['image_pct']:.2f}% of images; df has {df['image_n']} images from {df['lesion_n']} lesions. At least one missing/unknown metadata field affects {any_missing} images. Field counts overlap.", "",
            "See [numeric validation](eda_validation.json) and [visual review](eda_visual_review.json) for the recorded checks. PNG/statistics/report reproduction is checked byte-for-byte; SVG structure and canvas dimensions are checked without requiring identical serialization.", ""]
        (ROOT / "docs/09_eda_results.md").write_text("\n".join(index), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_figures": len(figure_names), "n_tables": len(tables),
        "train_images": n, "train_lesions": int(n_lesions), "any_missing_or_unknown": any_missing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/eda/v1_seed42")
    parser.add_argument("--deps-dir", type=Path)
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.deps_dir, not args.no_index)
