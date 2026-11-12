"""Explain a control run by separating leaked from clean test images.

For every test image we ask a single question: does another image of the *same
lesion* appear in the training partition? If yes, the image is "exposed" to the
model and its accuracy is not evidence of generalisation.

    python scripts/analyze_leakage.py --run control_image_level \
        --train-split data/control_splits/image_level_seed42/train.csv

Reports overall metrics, then metrics split by exposure, plus the same numbers
for the lesion-level reference run if its predictions exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
import data as data_lib
import metrics as metrics_lib


def load_predictions(run: str) -> pd.DataFrame | None:
    path = config.OUTPUTS_DIR / run / "predictions_test.csv"
    return pd.read_csv(path) if path.exists() else None


def subset_metrics(pred: pd.DataFrame, mask, class_names) -> dict:
    chunk = pred[mask]
    if chunk.empty:
        return {"n": 0}
    probs = chunk[[f"prob_{c}" for c in class_names]].to_numpy()
    summary = metrics_lib.compute_metrics(chunk["y_true"].to_numpy(), probs, class_names)
    return {
        "n": int(len(chunk)),
        "accuracy": round(summary["accuracy"], 4),
        "macro_f1": round(summary["macro_f1"], 4),
        "auc": None if summary["auc_ovr_macro"] != summary["auc_ovr_macro"]
        else round(summary["auc_ovr_macro"], 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="control run directory name under outputs/")
    parser.add_argument("--train-split", required=True,
                        help="train.csv of the control split (path relative to Final_Project_B)")
    parser.add_argument("--reference", default="baseline_image",
                        help="lesion-level reference run for comparison")
    args = parser.parse_args(argv)

    split_path = Path(args.train_split)
    if not split_path.is_absolute():
        split_path = config.B_DIR / split_path
    train = pd.read_csv(split_path)
    seen_lesions = set(train["lesion_id"])

    pred = load_predictions(args.run)
    if pred is None:
        raise SystemExit(f"no predictions_test.csv for run {args.run!r}")

    class_names = list(data_lib.CLASS_NAMES)
    exposed = pred["lesion_id"].isin(seen_lesions)
    everything = pd.Series(True, index=pred.index)

    overall = subset_metrics(pred, everything, class_names)
    leaky = subset_metrics(pred, exposed, class_names)
    clean = subset_metrics(pred, ~exposed, class_names)

    print(f"run: {args.run}  (test images: {len(pred)})")
    print(f"exposed to a same-lesion training image: {int(exposed.sum())} "
          f"({exposed.mean():.1%})")
    print()
    print(f"{'subset':<26}{'n':>6}{'accuracy':>11}{'macro-F1':>11}{'AUC':>9}")
    for name, row in (
        ("all test images", overall),
        ("exposed (leaked)", leaky),
        ("clean (unseen lesion)", clean),
    ):
        print(f"{name:<26}{row['n']:>6}{row.get('accuracy', float('nan')):>11.4f}"
              f"{row.get('macro_f1', float('nan')):>11.4f}"
              f"{(row.get('auc') or float('nan')):>9.4f}")

    ref = load_predictions(args.reference)
    if ref is not None:
        ref_metrics = subset_metrics(ref, pd.Series(True, index=ref.index), class_names)
        print()
        print(f"reference ({args.reference}, lesion-level split): "
              f"n={ref_metrics['n']} acc={ref_metrics['accuracy']:.4f} "
              f"macro-F1={ref_metrics['macro_f1']:.4f} AUC={ref_metrics.get('auc')}")

    payload = {
        "run": args.run,
        "train_split": str(split_path),
        "all": overall,
        "exposed": leaky,
        "clean": clean,
    }
    out = config.OUTPUTS_DIR / args.run / "leakage_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
