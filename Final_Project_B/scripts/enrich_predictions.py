"""Merge the subgroup variables into the exported prediction tables.

Why this exists: the raw prediction export only carries `image_id`, `lesion_id`,
`dx` and the class probabilities, because stage C needs the demographic and
clinical grouping fields (age, sex, lesion location, ...) and asking them to join
a second file by hand is exactly where mistakes happen.

Nothing is imputed. Missing values are preserved and additionally exposed as an
`age_missing` flag and an `unknown` level in the two age bands, so a subgroup
analysis can decide what to do with them instead of silently inheriting a choice.

    python scripts/enrich_predictions.py
    python scripts/enrich_predictions.py --out outputs/enriched_predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
import run_all

#: (run directory, tag used in the delivery file names)
SOURCES = [(f"{e}_ens3", "ens3") for e in run_all.EXPERIMENTS] + [
    (e, "s42") for e in run_all.EXPERIMENTS
]

META_COLUMNS = (
    "image_id", "age", "age_missing", "sex", "localization", "dx_type", "dataset",
)

#: coarse banding with enough samples per cell for a subgroup comparison
AGE_BINS = (0, 40, 60, 200)
AGE_LABELS = ("<40", "40-59", "60+")

#: finer banding for descriptive plots only
AGE_BINS_FINE = (0, 40, 50, 60, 70, 80, 200)
AGE_LABELS_FINE = ("<40", "40-49", "50-59", "60-69", "70-79", "80+")

MALIGNANT = {"akiec", "bcc", "mel"}


def load_metadata() -> pd.DataFrame:
    """All 10,015 processed rows with the grouping variables."""
    frames = [pd.read_csv(config.split_csv(name)) for name in ("train", "val", "test")]
    frame = pd.concat(frames, ignore_index=True)

    # how many images this lesion contributes in the whole dataset; useful for
    # deciding whether the subgroup test needs lesion-level clustering
    frame["lesion_image_count"] = frame.groupby("lesion_id")["image_id"].transform("size")
    frame["is_malignant"] = frame["dx"].isin(MALIGNANT).astype(int)
    frame["age_missing"] = frame["age_missing"].astype(int)
    frame["age_band"] = pd.cut(
        frame["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(object).where(frame["age"].notna(), "unknown")
    frame["age_band_fine"] = pd.cut(
        frame["age"], bins=AGE_BINS_FINE, labels=AGE_LABELS_FINE, right=False
    ).astype(object).where(frame["age"].notna(), "unknown")

    derived = ["age_band", "age_band_fine", "lesion_image_count", "is_malignant"]
    keep = [c for c in META_COLUMNS if c in frame.columns] + derived
    return frame[keep].copy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="output directory (default: Final_Project_B/predictions)")
    args = parser.parse_args(argv)

    # Tracked location on purpose: these tables are the shipped hand-off to stage C,
    # so they belong in the repository rather than in the git-ignored outputs/.
    out_dir = Path(args.out) if args.out else config.B_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    print(f"metadata rows: {len(meta)}  columns: {list(meta.columns)}")
    print(f"age missing: {int(meta.age_missing.sum())}  "
          f"sex unknown: {int((meta.sex == 'unknown').sum())}  "
          f"localization unknown: {int((meta.localization == 'unknown').sum())}")

    # one tidy table keyed by image_id, for anyone who prefers a separate join
    for split in ("test", "val"):
        source = pd.read_csv(config.split_csv(split))
        table = meta[meta.image_id.isin(source.image_id)].copy()
        table.to_csv(out_dir / f"subgroups_{split}.csv", index=False)
        print(f"  subgroups_{split}.csv: {len(table)} rows")

    written = 0
    for run, tag in SOURCES:
        for split in ("test", "val"):
            path = config.OUTPUTS_DIR / run / f"predictions_{split}.csv"
            if not path.exists():
                print(f"  [skip] missing {path}")
                continue
            frame = pd.read_csv(path)
            missing = set(frame.image_id) - set(meta.image_id)
            if missing:
                raise SystemExit(f"{path}: {len(missing)} image_id not in the metadata")
            merged = frame.merge(meta, on="image_id", how="left", validate="many_to_one")
            if len(merged) != len(frame):
                raise SystemExit(f"{path}: row count changed during the merge")

            # keep the original column order, append the new fields at the end
            added = [c for c in meta.columns if c != "image_id"]
            ordered = list(frame.columns) + added
            merged[ordered].to_csv(out_dir / f"{tag}__{run}__{split}.csv", index=False)
            written += 1

    print(f"\nwritten {written} enriched prediction tables to {out_dir}")
    sample = pd.read_csv(out_dir / f"ens3__{run_all.EXPERIMENTS[0]}_ens3__test.csv")
    print(f"columns ({len(sample.columns)}): {list(sample.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
