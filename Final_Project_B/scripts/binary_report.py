"""Maximise binary (malignant/benign) performance the honest way.

Two sources of a malignancy score are supported:

``--run <binary run>``
    A model trained directly on the binary label; the score is ``p(malignant)``.
``--from-multiclass <7-class run>``
    Reuses an already trained 7-class model and scores malignancy as
    ``p(akiec) + p(bcc) + p(mel)``. No extra training, and usually stronger than a
    dedicated binary head because the 7-class model has to separate the malignant
    classes from each other first.

Anti-overfitting rules enforced here:
*   the decision threshold is chosen on the **validation** split only;
*   the test split is evaluated once, with that frozen threshold, and with 0.5 for
    reference;
*   test-time augmentation is lossless (flip + quarter turns) and is applied
    identically to validation and test.

    python scripts/binary_report.py --from-multiclass fusion_all_metadata --tta
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
import data as data_lib
import engine
import metrics as metrics_lib
import models as models_lib

MALIGNANT_NAMES = ("akiec", "bcc", "mel")


def find_checkpoint(run: str):
    path = config.OUTPUTS_DIR / run / "best.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint for run {run!r}")
    return path


def malignancy_score(state: dict, cfg: dict, bundles: dict, split: str, device, tta: bool):
    """Return (y_true_binary, score, frame) for one split."""
    class_names = list(state["class_names"])
    model = models_lib.build_model(cfg, bundles["n_features"], len(class_names)).to(device)
    model.load_state_dict(state["state_dict"])

    loader = bundles["loaders"][split]
    frame = bundles["frames"][split]
    _y, probs, index = metrics_lib.collect_probabilities(
        model, loader, device, bool(cfg["train"].get("amp", True)), tta=tta
    )
    if len(class_names) == 2:
        score = probs[:, 1]
    else:
        malignant_idx = [class_names.index(name) for name in MALIGNANT_NAMES]
        score = probs[:, malignant_idx].sum(axis=1)

    dx = frame.iloc[index]["dx"].reset_index(drop=True)
    y_binary = dx.isin(MALIGNANT_NAMES).astype(int).to_numpy()
    return y_binary, score, frame.iloc[index].reset_index(drop=True)


def scores_at(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y_pred = (score >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_malignant": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_malignant": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_malignant": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, score)), 4),
        "n": int(len(y_true)),
        "n_malignant": int(y_true.sum()),
    }


def best_threshold(y_true: np.ndarray, score: np.ndarray, target: str) -> tuple[float, float]:
    """Pick the threshold that maximises ``target`` on the validation split."""
    grid = np.round(np.arange(0.05, 0.951, 0.005), 3)
    best_t, best_v = 0.5, -1.0
    for threshold in grid:
        y_pred = (score >= threshold).astype(int)
        value = (
            accuracy_score(y_true, y_pred)
            if target == "accuracy"
            else balanced_accuracy_score(y_true, y_pred)
        )
        if value > best_v:
            best_t, best_v = float(threshold), float(value)
    return best_t, float(best_v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="binary-model run directory")
    parser.add_argument("--from-multiclass", default=None, help="7-class run directory to reuse")
    parser.add_argument("--input", default=None, help="image-level split directory name")
    parser.add_argument("--tta", action="store_true", help="average 5 lossless views")
    parser.add_argument("--target", choices=("accuracy", "balanced_accuracy"), default="accuracy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if bool(args.run) == bool(args.from_multiclass):
        raise SystemExit("pass exactly one of --run / --from-multiclass")

    source = args.run or args.from_multiclass
    checkpoint = find_checkpoint(source)
    device = engine.select_device(args.device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    if args.input:
        cfg.setdefault("data", {})["split_dir"] = f"data/control_splits/{args.input}"

    bundles = data_lib.make_loaders(cfg)
    val_y, val_s, _ = malignancy_score(state, cfg, bundles, "val", device, args.tta)
    threshold, val_best = best_threshold(val_y, val_s, args.target)
    test_y, test_s, test_frame = malignancy_score(state, cfg, bundles, "test", device, args.tta)

    print(f"source: {source}  ({'binary model' if args.run else '7-class model, summed malignant probability'})")
    print(f"test-time augmentation: {'on (5 views)' if args.tta else 'off'}")
    print(f"threshold chosen on VALIDATION for {args.target}: {threshold:.3f} "
          f"({args.target}={val_best:.4f})")
    print()
    report = {
        "source": source,
        "mode": "binary_model" if args.run else "multiclass_malignant_sum",
        "tta": bool(args.tta),
        "target": args.target,
        "val_threshold": threshold,
        "val_at_threshold": scores_at(val_y, val_s, threshold),
        "test_at_threshold": scores_at(test_y, test_s, threshold),
        "test_at_half": scores_at(test_y, test_s, 0.5),
    }

    rows = [
        ("val  @ chosen", report["val_at_threshold"]),
        ("test @ chosen", report["test_at_threshold"]),
        ("test @ 0.50  ", report["test_at_half"]),
    ]
    print(f"{'split':<14}{'thr':>6}{'acc':>9}{'bal_acc':>9}{'prec':>8}{'recall':>8}{'f1':>8}{'auc':>8}")
    for label, payload in rows:
        print(f"{label:<14}{payload['threshold']:>6.3f}{payload['accuracy']:>9.4f}"
              f"{payload['balanced_accuracy']:>9.4f}{payload['precision_malignant']:>8.4f}"
              f"{payload['recall_malignant']:>8.4f}{payload['f1_malignant']:>8.4f}"
              f"{payload['auc']:>8.4f}")

    out_path = (
        config.OUTPUTS_DIR / source / (
            f"binary_report{'_tta' if args.tta else ''}_{args.target}.json"
        )
        if args.out is None
        else config.B_DIR / args.out
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pred = test_frame[["image_id", "lesion_id", "dx"]].copy()
    pred["y_true_binary"] = test_y
    pred["score"] = test_s
    pred["y_pred_binary"] = (test_s >= threshold).astype(int)
    pred.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"\nwritten {out_path.name} and {out_path.with_suffix('.csv').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
