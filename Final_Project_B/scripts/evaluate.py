"""Re-evaluate a saved checkpoint and export its predictions.

    python scripts/evaluate.py --run fusion_all_metadata --split test

Useful for regenerating metrics or prediction tables without retraining, and for
a second opinion on a checkpoint after the analysis code changes.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import config
import data as data_lib
import engine
import models as models_lib
import train


def find_checkpoint(run_or_path: str) -> tuple:
    path = config.OUTPUTS_DIR / run_or_path / "best.pt"
    if path.exists():
        return path
    direct = config.B_DIR / run_or_path
    if direct.exists() and direct.is_file():
        return direct
    raise SystemExit(f"no checkpoint found for {run_or_path!r} (looked at {path})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory name under outputs/, or a best.pt path")
    parser.add_argument("--split", default="val,test", help="comma-separated splits to evaluate")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    checkpoint_path = find_checkpoint(args.run)
    device = engine.select_device(args.device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    splits = [s.strip() for s in args.split.split(",") if s.strip()]

    out_dir = checkpoint_path.parent
    logger = engine.Logger(out_dir / "evaluate.log")
    logger(f"checkpoint: {checkpoint_path} (epoch {state['epoch']})")

    bundles = data_lib.make_loaders(cfg)
    class_names = bundles["class_names"]
    model = models_lib.build_model(cfg, bundles["n_features"], len(class_names)).to(device)
    model.load_state_dict(state["state_dict"])

    for split in splits:
        train.evaluate_split(
            model, bundles["loaders"][split], bundles["frames"][split],
            class_names, device, bool(cfg["train"].get("amp", True)), out_dir, split, logger,
        )

    # Keep summary.json consistent with train.py so run_all.py can aggregate runs
    # that were finished (or repaired) through this entry point.
    payload = {"run": out_dir.name, "metadata_mode": cfg["data"]["metadata_mode"],
               "model": cfg["model"]["kind"], "seed": cfg["train"]["seed"],
               "best_epoch": state["epoch"]}
    for split in ("val", "test"):
        path = out_dir / f"metrics_{split}.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        payload[split] = {
            key: metrics[key]
            for key in ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")
        }
    engine.save_json(out_dir / "summary.json", payload)
    logger(f"summary written to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
