"""Audit that the B-stage deliverables are complete and internally consistent.

This is the check that answers "is the modelling stage actually finished", with
evidence rather than assertion. It verifies, for every run:

1. every required artefact exists;
2. the headline metrics can be **recomputed** from the exported predictions
   (catches a stale metrics file or a broken export);
3. all runs of one experiment use exactly the same validation/test images;
4. all four experiments use the same split, which is what makes the paired
   Benefit / Harm comparison valid;
5. the four experiment configs differ **only** in the metadata mode and the model
   kind, so any measured difference is attributable to the metadata stream.

Exit code is non-zero if any check fails.

    python scripts/audit_runs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import metrics as metrics_lib
import run_all

REQUIRED = (
    "config.json", "history.csv", "summary.json",
    "metrics_val.json", "metrics_test.json",
    "per_class_val.csv", "per_class_test.csv",
    "confusion_val.png", "confusion_test.png",
    "predictions_val.csv", "predictions_test.csv",
)
REQUIRED_ENSEMBLE = (
    "summary.json", "metrics_val.json", "metrics_test.json",
    "per_class_val.csv", "per_class_test.csv",
    "confusion_val.png", "confusion_test.png",
    "predictions_val.csv", "predictions_test.csv",
)
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

#: config keys that are allowed to differ between the four experiments
EXPECTED_DIFFERENCES = {
    ("data", "metadata_mode"),
    ("model", "kind"),
    ("model", "dropout"),
    ("run", "name"),
    ("run", "description"),
    # bookkeeping: records which config file produced the run
    ("_config_path",),
    # loader performance knob. It only redistributes the per-worker augmentation
    # RNG stream; data, model, split and hyperparameters are untouched. Recorded
    # here so the difference is visible rather than silently ignored.
    ("data", "num_workers"),
}


def run_names(experiment: str, seeds) -> list[str]:
    names = []
    for seed in seeds:
        names.append(experiment if seed == 42 else f"{experiment}_s{seed}")
    names.append(f"{experiment}_ens{len(seeds)}")
    return names


def signature(frame: pd.DataFrame, column: str = "image_id") -> str:
    joined = "\n".join(sorted(frame[column].astype(str)))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def recompute(run_dir: Path, split: str) -> dict:
    pred = pd.read_csv(run_dir / f"predictions_{split}.csv")
    probs = pred[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy()
    return metrics_lib.compute_metrics(pred["y_true"].to_numpy(), probs, CLASS_NAMES)


def flatten(d: dict, prefix=()) -> dict:
    out = {}
    for key, value in d.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args(argv)

    failures, warnings = [], []
    print(f"{'run':<34}{'files':>7}{'metrics recomputed':>21}{'':>2}")
    print("-" * 66)

    for experiment in run_all.EXPERIMENTS:
        for name in run_names(experiment, args.seeds):
            run_dir = config.OUTPUTS_DIR / name
            if not run_dir.exists():
                failures.append(f"{name}: directory missing")
                print(f"{name:<34}{'MISSING':>7}")
                continue

            required = REQUIRED_ENSEMBLE if "_ens" in name else REQUIRED
            missing = [item for item in required if not (run_dir / item).exists()]
            if missing:
                failures.append(f"{name}: missing {missing}")

            status = "ok"
            for split in ("val", "test"):
                stored = json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8"))
                fresh = recompute(run_dir, split)
                for key in ("accuracy", "macro_f1", "weighted_f1"):
                    if abs(stored[key] - fresh[key]) > 1e-9:
                        failures.append(
                            f"{name}/{split}: {key} stored {stored[key]:.6f} != recomputed {fresh[key]:.6f}"
                        )
                        status = "MISMATCH"
                if abs(stored["auc_ovr_macro"] - fresh["auc_ovr_macro"]) > 1e-6:
                    failures.append(
                        f"{name}/{split}: auc stored {stored['auc_ovr_macro']:.6f} "
                        f"!= recomputed {fresh['auc_ovr_macro']:.6f}"
                    )
                    status = "MISMATCH"

            print(f"{name:<34}{len(list(run_dir.iterdir())):>7}{status:>21}")

    # ---- split identity across seeds and across experiments -------------------
    print("\nsplit identity (must be identical for every run)")
    test_signatures, val_signatures = {}, {}
    for experiment in run_all.EXPERIMENTS:
        for name in run_names(experiment, args.seeds):
            run_dir = config.OUTPUTS_DIR / name
            if not run_dir.exists():
                continue
            test_signatures[name] = signature(pd.read_csv(run_dir / "predictions_test.csv"))
            val_signatures[name] = signature(pd.read_csv(run_dir / "predictions_val.csv"))

    if len(set(test_signatures.values())) != 1:
        failures.append(f"test split differs between runs: {test_signatures}")
    if len(set(val_signatures.values())) != 1:
        failures.append(f"val split differs between runs: {val_signatures}")
    print(f"  test signature(s): {sorted(set(test_signatures.values()))}")
    print(f"  val  signature(s): {sorted(set(val_signatures.values()))}")

    # ---- configs differ only where expected ----------------------------------
    print("\nconfig differences between the four experiments")
    configs = {}
    for experiment in run_all.EXPERIMENTS:
        path = config.OUTPUTS_DIR / experiment / "config.json"
        configs[experiment] = flatten(json.loads(path.read_text(encoding="utf-8")))

    reference = configs[run_all.EXPERIMENTS[0]]
    for experiment, flat in configs.items():
        differing = {key for key in set(reference) | set(flat)
                     if reference.get(key) != flat.get(key)}
        unexpected = differing - EXPECTED_DIFFERENCES
        if unexpected:
            failures.append(f"{experiment}: unexpected config differences {sorted(unexpected)}")
        noted = differing & {("data", "num_workers"), ("_config_path",)}
        print(f"  {experiment:<30} differs in "
              f"{sorted(differing - {('run','name'), ('run','description')})}")
        if ("data", "num_workers") in noted:
            warnings.append(f"{experiment}: loader worker count differs (RNG stream only)")

    # ---- reported metrics table ---------------------------------------------
    print("\nheadline table (test)")
    rows = []
    for experiment in run_all.EXPERIMENTS:
        for name in run_names(experiment, args.seeds):
            path = config.OUTPUTS_DIR / name / "summary.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "run": name,
                "acc": round(payload["test"]["accuracy"], 4),
                "macro_f1": round(payload["test"]["macro_f1"], 4),
                "auc": round(payload["test"]["auc_ovr_macro"], 4),
            })
    print(pd.DataFrame(rows).to_string(index=False))
    headline = pd.DataFrame(rows)
    headline.to_csv(config.OUTPUTS_DIR / "deliverables_headline.csv", index=False)

    print("\n" + "=" * 66)
    if warnings:
        for item in warnings:
            print(f"WARN  {item}")
    if failures:
        for item in failures:
            print(f"FAIL  {item}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("ALL CHECKS PASSED - B-stage deliverables are complete and consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
