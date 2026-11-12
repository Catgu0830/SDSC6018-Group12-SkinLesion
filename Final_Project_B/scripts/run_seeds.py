"""Run every headline experiment across several random seeds and aggregate.

Why this exists: a single run cannot distinguish a real effect from run-to-run
noise. On a 1,491-image test split the standard error of accuracy is about 0.9
percentage points, so the fusion-vs-baseline gap has to be reported as
mean +- standard deviation over seeds before it can be claimed.

Seed 42 reuses the run that already exists in ``outputs/<name>`` so nothing is
retrained unnecessarily; extra seeds get their own ``<name>_s<seed>`` directory.

    python scripts/run_seeds.py                     # all four experiments, seeds 42/43/44
    python scripts/run_seeds.py --seeds 42 43       # subset
    python scripts/run_seeds.py --experiments baseline_image fusion_all_metadata

Writes ``outputs/seeds_summary.csv`` (per-run rows) and
``outputs/seeds_aggregate.csv`` (mean +- std per experiment, plus the paired delta
against the baseline).
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config
import engine
import train
import run_all

METRICS = ("test_accuracy", "test_macro_f1", "test_auc_ovr_macro",
           "val_accuracy", "val_macro_f1")

#: flat column name -> (split, key inside summary.json)
METRIC_KEYS = {
    "test_accuracy": ("test", "accuracy"),
    "test_macro_f1": ("test", "macro_f1"),
    "test_auc_ovr_macro": ("test", "auc_ovr_macro"),
    "val_accuracy": ("val", "accuracy"),
    "val_macro_f1": ("val", "macro_f1"),
}


def read_summary(run_name: str) -> dict | None:
    path = config.OUTPUTS_DIR / run_name / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(experiment: str, seed: int, device, overrides) -> str:
    """Train (or reuse) one experiment/seed pair; returns the output directory name."""
    if seed == 42 and read_summary(experiment) is not None:
        print(f"[skip] {experiment} already trained for seed 42", flush=True)
        return experiment

    run_name = f"{experiment}_s{seed}"
    cfg = config.load_config(experiment)
    cfg = config.apply_overrides(cfg, overrides)
    cfg.setdefault("run", {})["name"] = run_name
    cfg.setdefault("train", {})["seed"] = seed
    print("\n" + "#" * 78)
    print(f"# {experiment}  seed={seed}  ->  {run_name}")
    print("#" * 78, flush=True)
    train.run(cfg, device)
    return run_name


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per experiment with flat column names, and the paired difference.

    Columns are flattened (``test_macro_f1_mean``) rather than written as a
    two-level header, so the CSV opens cleanly in Excel and can be read back with
    a plain ``pd.read_csv`` - the nested header broke a downstream reader once.
    """
    grouped = rows.groupby("experiment")[list(METRICS)]
    summary = grouped.agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.index.name = "experiment"

    if "baseline_image" in set(rows["experiment"]):
        base = rows[rows["experiment"] == "baseline_image"].set_index("seed")
        deltas = []
        for experiment, chunk in rows.groupby("experiment"):
            if experiment == "baseline_image":
                continue
            chunk = chunk.set_index("seed")
            common = chunk.index.intersection(base.index)
            if len(common) == 0:
                continue
            for metric in ("test_accuracy", "test_macro_f1", "test_auc_ovr_macro"):
                diff = chunk.loc[common, metric] - base.loc[common, metric]
                deltas.append({
                    "experiment": experiment,
                    "metric": metric,
                    "n_seeds": len(common),
                    "delta_mean": round(float(diff.mean()), 4),
                    "delta_std": round(float(diff.std(ddof=1)) if len(common) > 1 else 0.0, 4),
                })
        if deltas:
            delta_frame = pd.DataFrame(deltas)
            delta_frame.to_csv(config.OUTPUTS_DIR / "seeds_deltas.csv", index=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="*", default=list(run_all.EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = engine.select_device(args.device)
    rows = []
    for experiment in args.experiments:
        for seed in args.seeds:
            run_name = run_one(experiment, seed, device, args.overrides)
            payload = read_summary(run_name)
            if payload is None:
                print(f"[warn] no summary for {run_name}", flush=True)
                continue
            row = {"experiment": experiment, "seed": seed, "run": run_name,
                   "best_epoch": payload.get("best_epoch")}
            for metric in METRICS:
                split, key = METRIC_KEYS[metric]
                row[metric] = payload.get(split, {}).get(key, float("nan"))
            rows.append(row)

    if not rows:
        raise SystemExit("no runs completed")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(config.OUTPUTS_DIR / "seeds_summary.csv", index=False)
    print("\n=== per-run ===")
    print(per_run[["experiment", "seed", "test_accuracy", "test_macro_f1",
                   "test_auc_ovr_macro"]].round(4).to_string(index=False))

    summary = aggregate(per_run)
    summary.to_csv(config.OUTPUTS_DIR / "seeds_aggregate.csv")
    print("\n=== mean +- std over seeds ===")
    print(summary.to_string())

    delta_path = config.OUTPUTS_DIR / "seeds_deltas.csv"
    if delta_path.exists():
        print("\n=== paired delta vs baseline ===")
        print(pd.read_csv(delta_path).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
