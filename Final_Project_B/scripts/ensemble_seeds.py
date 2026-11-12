"""Average the test-time probabilities of the same experiment across random seeds.

A seed ensemble is the cheapest legitimate accuracy gain available: it uses no
extra data, no test-set information, and no tuning. Each member was trained with a
different weight initialisation and data order on the *same* frozen split, so the
members differ only by run-to-run randomness.

Fair comparison rule: an ensemble of three baselines must be compared against an
ensemble of three fusion models, never against a single run.

    python scripts/ensemble_seeds.py                       # all four experiments
    python scripts/ensemble_seeds.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

import config
import engine
import metrics as metrics_lib
import run_all

PROB_PREFIX = "prob_"


def seed_runs(experiment: str, seeds) -> list[str]:
    """Seed 42 lives in the plain run directory; later seeds get a _s<seed> suffix."""
    names = []
    for seed in seeds:
        names.append(experiment if seed == 42 else f"{experiment}_s{seed}")
    return names


def load_predictions(run: str, split: str) -> pd.DataFrame:
    path = config.OUTPUTS_DIR / run / f"predictions_{split}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def ensemble(experiment: str, seeds, split: str) -> dict:
    runs = seed_runs(experiment, seeds)
    frames = [load_predictions(run, split).sort_values("image_id").reset_index(drop=True)
              for run in runs]

    reference = frames[0]
    for run, frame in zip(runs, frames):
        if set(frame["image_id"]) != set(reference["image_id"]):
            raise SystemExit(f"{run}: image_id set differs from {runs[0]}")

    prob_cols = [c for c in reference.columns if c.startswith(PROB_PREFIX)]
    stacked = np.stack([frame[prob_cols].to_numpy() for frame in frames], axis=0)
    mean_probs = stacked.mean(axis=0)

    y_true = reference["y_true"].to_numpy()
    class_names = [c[len(PROB_PREFIX):] for c in prob_cols]
    summary = metrics_lib.compute_metrics(y_true, mean_probs, class_names)

    out_dir = config.OUTPUTS_DIR / f"{experiment}_ens{len(seeds)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = reference[["image_id", "lesion_id", "dx", "label"]].copy()
    table["y_true"] = y_true
    table["y_pred"] = mean_probs.argmax(axis=1)
    for k, name in enumerate(class_names):
        table[f"prob_{name}"] = mean_probs[:, k]
    table["correct"] = (table["y_true"] == table["y_pred"]).astype(int)
    table.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    payload = {
        "run": f"{experiment}_ens{len(seeds)}",
        "members": runs,
        "split": split,
        "n_samples": int(summary["n_samples"]),
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "auc_ovr_macro": summary["auc_ovr_macro"],
        "per_class": summary["per_class"],
        "confusion_matrix": summary["confusion_matrix"],
    }
    engine.save_json(out_dir / f"metrics_{split}.json", payload)
    metrics_lib.metrics_frame(summary, class_names).to_csv(
        out_dir / f"per_class_{split}.csv", index=False
    )
    metrics_lib.save_confusion_png(
        summary["confusion_matrix"], class_names,
        out_dir / f"confusion_{split}.png", title=f"Confusion matrix ({split}, {len(seeds)}-seed ensemble)",
    )
    return payload


def single_run_metrics(run: str, split: str) -> dict:
    path = config.OUTPUTS_DIR / run / f"metrics_{split}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "accuracy": payload["accuracy"],
        "macro_f1": payload["macro_f1"],
        "auc_ovr_macro": payload["auc_ovr_macro"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=list(run_all.EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--split", default="test")
    parser.add_argument("--all-splits", action="store_true",
                        help="produce val and test ensembles plus a summary.json per experiment")
    args = parser.parse_args(argv)

    splits = ["val", "test"] if args.all_splits else [args.split]
    rows = []
    for experiment in args.experiments:
        payloads = {}
        for split in splits:
            try:
                payloads[split] = ensemble(experiment, args.seeds, split)
            except SystemExit as exc:
                print(f"[skip] {experiment}/{split}: {exc}", flush=True)
        if args.split not in payloads and not args.all_splits:
            continue
        if args.all_splits and len(payloads) == 2:
            source = json.loads(
                (config.OUTPUTS_DIR / seed_runs(experiment, args.seeds)[0] / "summary.json")
                .read_text(encoding="utf-8")
            )
            engine.save_json(config.OUTPUTS_DIR / f"{experiment}_ens{len(args.seeds)}" / "summary.json", {
                "run": f"{experiment}_ens{len(args.seeds)}",
                "metadata_mode": source.get("metadata_mode"),
                "model": source.get("model"),
                "seed": "ensemble",
                "best_epoch": None,
                "members": seed_runs(experiment, args.seeds),
                "val": {k: payloads["val"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")},
                "test": {k: payloads["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")},
            })

        main_payload = payloads.get("test")
        if not main_payload:
            continue
        singles = [single_run_metrics(run, "test") for run in seed_runs(experiment, args.seeds)]
        singles = [s for s in singles if s]
        rows.append({
            "experiment": experiment,
            "n_members": len(args.seeds),
            "ens_accuracy": main_payload["accuracy"],
            "ens_macro_f1": main_payload["macro_f1"],
            "ens_auc": main_payload["auc_ovr_macro"],
            "single_acc_mean": float(np.mean([s["accuracy"] for s in singles])),
            "single_acc_best": float(np.max([s["accuracy"] for s in singles])),
            "single_f1_mean": float(np.mean([s["macro_f1"] for s in singles])),
            "single_f1_best": float(np.max([s["macro_f1"] for s in singles])),
        })

    if not rows:
        raise SystemExit("no ensembles produced")

    table = pd.DataFrame(rows)
    table.to_csv(config.OUTPUTS_DIR / f"ensemble_{args.split}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== seed-ensemble, {args.split} split ===")
    print(table.round(4).to_string(index=False))

    print("\n=== ensemble gain over the mean single run ===")
    gain = table.assign(
        d_acc=(table.ens_accuracy - table.single_acc_mean).round(4),
        d_f1=(table.ens_macro_f1 - table.single_f1_mean).round(4),
    )[["experiment", "d_acc", "d_f1"]]
    print(gain.to_string(index=False))

    print("\n=== paired delta vs baseline ensemble ===")
    if "baseline_image" in set(table["experiment"]):
        base = table[table.experiment == "baseline_image"].iloc[0]
        for _, row in table.iterrows():
            if row["experiment"] == "baseline_image":
                continue
            print(f"{row['experiment']:<28} "
                  f"dAcc={row['ens_accuracy'] - base['ens_accuracy']:+.4f}  "
                  f"dF1={row['ens_macro_f1'] - base['ens_macro_f1']:+.4f}  "
                  f"dAUC={row['ens_auc'] - base['ens_auc']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
