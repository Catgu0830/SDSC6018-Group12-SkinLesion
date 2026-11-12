"""Run every B-stage experiment in order and build the comparison table.

    python scripts/run_all.py                # baseline + fusion + 2 ablations
    python scripts/run_all.py --only baseline_image

Each experiment is an ordinary ``train.py`` run; this script only sequences them
and collects ``outputs/summary.csv`` so the report can quote one consistent table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
import engine
import train

#: Execution order matters: the baseline is the reference every fusion row is
#: compared against, and the two ablations reuse the baseline settings.
EXPERIMENTS = (
    "baseline_image",
    "fusion_all_metadata",
    "fusion_age_only",
    "fusion_localization_only",
)

SUMMARY_COLUMNS = ("accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro")


def summarise(names) -> pd.DataFrame:
    rows = []
    for name in names:
        path = config.OUTPUTS_DIR / name / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run": payload["run"],
            "metadata_mode": payload["metadata_mode"],
            "model": payload["model"],
            "seed": payload["seed"],
            "best_epoch": payload["best_epoch"],
        }
        for split in ("val", "test"):
            for metric in SUMMARY_COLUMNS:
                row[f"{split}_{metric}"] = payload[split][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="subset of experiment names")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    names = list(args.only) if args.only else list(EXPERIMENTS)
    device = engine.select_device(args.device)

    for name in names:
        cfg = config.load_config(name)
        cfg = config.apply_overrides(cfg, args.overrides)
        cfg.setdefault("run", {})["name"] = name
        print("\n" + "#" * 78)
        print(f"# experiment: {name}")
        print("#" * 78, flush=True)
        train.run(cfg, device)

    table = summarise(names)
    if not table.empty:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(config.OUTPUTS_DIR / "summary.csv", index=False)
        show = table[[
            "run", "test_accuracy", "test_macro_f1", "test_auc_ovr_macro", "test_weighted_f1"
        ]]
        print("\n=== test-set comparison ===")
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
