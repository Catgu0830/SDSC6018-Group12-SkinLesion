"""Train one B-stage experiment end to end.

    python scripts/train.py --config fusion_all_metadata.json

Produces, under ``Final_Project_B/outputs/<run name>/``:
    config.json              resolved configuration snapshot
    run.log                  epoch-by-epoch log
    history.csv / history.png
    best.pt                  checkpoint (state dict + config + feature columns)
    metrics_val.json / metrics_test.json
    confusion_val.csv / confusion_val.png / confusion_test.csv / confusion_test.png
    predictions_val.csv / predictions_test.csv

``predictions_*.csv`` is the contract with the C stage: one row per image with
``image_id``, ``y_true``, ``y_pred`` and the seven class probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import config
import data as data_lib
import engine
import metrics as metrics_lib
import models as models_lib


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="experiment config (name in configs/ or a path)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value, e.g. train.epochs=3")
    parser.add_argument("--name", default=None, help="override the run/output directory name")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training and evaluate the existing checkpoint")
    return parser.parse_args(argv)


def evaluate_split(model, loader, frame, class_names, device, amp, out_dir: Path, split: str, logger):
    y_true, probs, _ = metrics_lib.collect_probabilities(model, loader, device, amp)
    summary = metrics_lib.compute_metrics(y_true, probs, class_names)

    engine.save_json(out_dir / f"metrics_{split}.json", summary)
    metrics_lib.metrics_frame(summary, class_names).to_csv(
        out_dir / f"per_class_{split}.csv", index=False
    )
    metrics_lib.save_confusion_png(
        summary["confusion_matrix"], class_names,
        out_dir / f"confusion_{split}.png", title=f"Confusion matrix ({split})",
    )
    engine.save_json(
        out_dir / f"confusion_{split}.csv",
        {"labels": list(class_names), "matrix": summary["confusion_matrix"]},
    )

    predictions = engine.predict_frame(model, loader, frame, class_names, device, amp)
    predictions.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    logger(
        f"[{split}] n={summary['n_samples']} acc={summary['accuracy']:.4f} "
        f"macro-F1={summary['macro_f1']:.4f} AUC={summary['auc_ovr_macro']:.4f}"
    )
    return summary


def run(cfg: dict, device: torch.device) -> dict:
    problems = config.check_inputs()
    if problems:
        raise SystemExit("input check failed:\n  " + "\n  ".join(problems))

    out_dir = config.output_dir(cfg)
    logger = engine.Logger(out_dir / "run.log")
    logger("=" * 78)
    logger(f"run: {config.experiment_name(cfg)}")
    logger(f"config: {cfg['_config_path']}")
    logger(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    logger("=" * 78)

    bundles = data_lib.make_loaders(cfg)
    loaders = bundles["loaders"]
    feature_columns = bundles["feature_columns"]
    class_names = bundles["class_names"]

    logger(f"metadata mode: {cfg['data']['metadata_mode']} ({len(feature_columns)} features)")
    for split, frame in bundles["frames"].items():
        logger(f"  {split:<5} images={len(frame):>5}  lesions={frame['lesion_id'].nunique():>5}")

    engine.set_seed(int(cfg["train"]["seed"]))
    model = models_lib.build_model(cfg, bundles["n_features"], len(class_names)).to(device)
    logger(models_lib.describe_model(model))

    engine.save_json(out_dir / "config.json", cfg)

    checkpoint = out_dir / "best.pt"
    if not (cfg.get("run", {}).get("eval_only") or False):
        engine.fit(
            cfg, model, loaders, bundles["class_counts"], class_names,
            feature_columns, device, out_dir, logger,
        )

    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    logger(f"loaded checkpoint from epoch {state['epoch']}")

    results = {}
    for split in ("val", "test"):
        results[split] = evaluate_split(
            model, loaders[split], bundles["frames"][split],
            class_names, device, bool(cfg["train"].get("amp", True)), out_dir, split, logger,
        )

    engine.save_json(out_dir / "summary.json", {
        "run": config.experiment_name(cfg),
        "metadata_mode": cfg["data"]["metadata_mode"],
        "model": cfg["model"]["kind"],
        "seed": cfg["train"]["seed"],
        "best_epoch": state["epoch"],
        "val": {k: results["val"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")},
        "test": {k: results["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")},
    })
    logger(f"artifacts written to {out_dir}")
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    cfg = config.apply_overrides(cfg, args.overrides)
    if args.name:
        cfg.setdefault("run", {})["name"] = args.name
    if args.eval_only:
        cfg.setdefault("run", {})["eval_only"] = True

    device = engine.select_device(args.device)
    run(cfg, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
