"""Build a deliberately leaky control split, to compare against the published literature.

Why this exists
---------------
A large part of the HAM10000 literature reports 95-99% accuracy. Many of those
papers state that they avoided leakage, but split **images** at random. HAM10000
has 10,015 images from only 7,470 lesions, and 26% of lesions contribute two or
more near-identical photographs, so an image-level split puts the same lesion in
train and test. This script reproduces that protocol so that the effect can be
measured on our own pipeline instead of argued about.

Protocols
---------
``image_level``
    Stratified random split **by image**. This is the most common protocol in the
    papers we are comparing against. Metrics stay identical to our main runs; only
    the partition changes.
``image_level_oversampled``
    RandomOverSampler-style class balancing applied **before** splitting, which
    additionally puts exact duplicate copies of the same image in different
    partitions. Some published pipelines do this to "fix" class imbalance.

Nothing here is used by a headline result. Output goes to
``Final_Project_B/data/control_splits/<protocol>_seed<seed>/`` and every run that
consumes it must be labelled as a *control*, not as our result.

    python scripts/make_control_split.py --protocol image_level --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

CONTROL_ROOT = config.B_DIR / "data" / "control_splits"


def load_all_rows() -> pd.DataFrame:
    """All 10,015 processed rows with A's fitted features (concatenated subsets)."""
    frames = [pd.read_csv(config.split_csv(name)) for name in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)
    if len(df) != len(df["image_id"].unique()):
        raise SystemExit("duplicate image_id while concatenating the frozen subsets")
    return df


def stratified_image_split(df: pd.DataFrame, ratios: dict, rng: np.random.RandomState) -> dict:
    """Split rows at random, stratified by label. No lesion grouping by design."""
    parts = {name: [] for name in ratios}
    for label in sorted(df["label"].unique()):
        index = np.array(df.index[df["label"] == label], dtype=np.int64, copy=True)
        rng.shuffle(index)
        n = len(index)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        parts["train"].append(index[:n_train])
        parts["val"].append(index[n_train:n_train + n_val])
        parts["test"].append(index[n_train + n_val:])
    return {
        name: df.loc[np.sort(np.concatenate(chunks))].reset_index(drop=True)
        for name, chunks in parts.items()
    }


def oversample_before_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Balance every class to the majority count by sampling with replacement."""
    rng = np.random.RandomState(seed)
    target = int(df["label"].value_counts().max())
    chunks = []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        if len(subset) >= target:
            chunks.append(subset)
            continue
        extra = subset.sample(
            n=target - len(subset), replace=True, random_state=rng.randint(0, 2**31 - 1)
        )
        chunks.append(pd.concat([subset, extra], ignore_index=True))
    return pd.concat(chunks, ignore_index=True)


def describe(split: dict, protocol: str, seed: int) -> dict:
    train_lesions = set(split["train"]["lesion_id"])
    test = split["test"]
    exposed = test["lesion_id"].isin(train_lesions)
    return {
        "protocol": protocol,
        "seed": seed,
        "rows": {name: int(len(frame)) for name, frame in split.items()},
        "unique_images": {
            name: int(frame["image_id"].nunique()) for name, frame in split.items()
        },
        "lesions": {
            name: int(frame["lesion_id"].nunique()) for name, frame in split.items()
        },
        "test_images_with_lesion_in_train": int(exposed.sum()),
        "test_images_total": int(len(test)),
        "test_fraction_exposed": round(float(exposed.mean()), 4),
        "note": (
            "Control split. Images of the same lesion may cross partitions; "
            "never report these numbers as our main result."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=("image_level", "image_level_oversampled"),
                        default="image_level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="output directory (default: data/control_splits/<protocol>_seed<seed>)")
    args = parser.parse_args(argv)

    df = load_all_rows()
    rng = np.random.RandomState(args.seed)
    print(f"loaded {len(df)} rows, {df['lesion_id'].nunique()} lesions, "
          f"{df['image_id'].nunique()} unique images")

    if args.protocol == "image_level":
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        working = df
    else:
        working = oversample_before_split(df, args.seed)
        ratios = {"train": 0.75, "val": 0.15, "test": 0.10}
        print(f"oversampled to {len(working)} rows before splitting")

    split = stratified_image_split(working.reset_index(drop=True), ratios, rng)

    out_dir = Path(args.out) if args.out else CONTROL_ROOT / f"{args.protocol}_seed{args.seed}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in split.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  {name}: {len(frame)} rows, {frame.lesion_id.nunique()} lesions, "
              f"{frame.image_id.nunique()} unique images")

    manifest = describe(split, args.protocol, args.seed)
    manifest["ratios"] = ratios
    (out_dir / "control_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nexposure: {manifest['test_images_with_lesion_in_train']} of "
          f"{manifest['test_images_total']} test images "
          f"({manifest['test_fraction_exposed']:.1%}) have a same-lesion image in train")
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
