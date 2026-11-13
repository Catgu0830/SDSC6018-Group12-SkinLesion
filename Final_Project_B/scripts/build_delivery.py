"""Assemble the hand-over (delivery) package for the B stage.

The package is written **outside the git working tree** on purpose: it contains
prediction tables and figures that must not be committed. Re-running this script
rebuilds the package from `outputs/` and the source tree.

    python scripts/build_delivery.py
    python scripts/build_delivery.py --out "D:/somewhere/else"

Layout produced:

    <out>/
    ├── 00_READ_ME_FIRST.md        navigation, headline numbers, red lines
    ├── MANIFEST.md                every file with size and role
    ├── 01_结果报告/                results.md, metrics_report.md, tables/
    ├── 02_给C的预测文件/            HANDOFF_TO_C.md, predictions/
    ├── 03_图表/                    confusion + training-curve PNGs
    ├── 04_代码与配置/              scripts/, configs/, requirements.txt, README.md
    └── 05_核查记录/                audit output, seed tables
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import config
import run_all

EXPERIMENTS = list(run_all.EXPERIMENTS)

#: which prediction tables go to C. Ensembles first (headline), then the
#: single-seed-42 runs for continuity with the original analysis.
PREDICTION_SOURCES = [
    (f"{e}_ens3", "ens3") for e in EXPERIMENTS
] + [
    (e, "s42") for e in EXPERIMENTS
]

TABLE_FILES = (
    "deliverables_headline.csv",
    "seeds_summary.csv",
    "seeds_aggregate.csv",
    "seeds_deltas.csv",
    "ensemble_test.csv",
    "summary.csv",
)


def copy(src: Path, dst_dir: Path, note: list, role: str) -> None:
    if not src.exists():
        note.append(f"MISSING  {src}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    shutil.copy2(src, target)
    note.append(f"OK       {target}")


def make_history_png(run: str, dst_dir: Path) -> Path | None:
    """Draw the training curve from history.csv (train.py does not save one)."""
    import pandas as pd

    import metrics as metrics_lib

    history = config.OUTPUTS_DIR / run / "history.csv"
    if not history.exists():
        return None
    frame = pd.read_csv(history)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / f"history_{run}.png"
    metrics_lib.save_history_png(frame, out)
    return out


def main(argv=None) -> int:
    # A fixed directory name on purpose: keying it on the build date silently
    # created a second package on the next day's rebuild, splitting the delivery.
    default_out = config.REPO_ROOT.parent / "6018_Group12_B阶段交付"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.resolve().is_relative_to(config.REPO_ROOT.resolve()):
        raise SystemExit(
            f"refusing to build the package inside the git working tree: {out}\n"
            "Choose a path outside the repository so it cannot be committed by accident."
        )
    out.mkdir(parents=True, exist_ok=True)

    report_dir = out / "01_结果报告"
    table_dir = report_dir / "tables"
    pred_dir = out / "02_给C的预测文件" / "predictions"
    figure_dir = out / "03_图表"
    code_dir = out / "04_代码与配置"
    audit_dir = out / "05_核查记录"
    for path in (report_dir, table_dir, pred_dir, figure_dir, code_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    # ---- 01 documents ------------------------------------------------------
    copy(config.B_DIR / "results.md", report_dir, log, "headline results")
    copy(config.OUTPUTS_DIR / "metrics_report.md", report_dir, log, "all metrics")
    for name in TABLE_FILES:
        copy(config.OUTPUTS_DIR / name, table_dir, log, "table")

    # ---- 02 predictions ----------------------------------------------------
    # Prefer the enriched tables (grouping variables merged in) so stage C does not
    # have to join anything by hand. Fall back to the raw export with a warning.
    enriched = config.OUTPUTS_DIR / "enriched_predictions"
    for run, tag in PREDICTION_SOURCES:
        for split in ("test", "val"):
            name = f"{tag}__{run}__{split}.csv"
            src = enriched / name
            if not src.exists():
                src = config.OUTPUTS_DIR / run / f"predictions_{split}.csv"
                log.append(f"NOTE     enriched table missing for {name}; copied the raw export")
            if not src.exists():
                log.append(f"MISSING  {src}")
                continue
            target = pred_dir / name
            shutil.copy2(src, target)
            log.append(f"OK       {target}")

    for split in ("test", "val"):
        src = enriched / f"subgroups_{split}.csv"
        if src.exists():
            shutil.copy2(src, pred_dir / src.name)
            log.append(f"OK       {pred_dir / src.name}")

    # ---- 03 figures --------------------------------------------------------
    for experiment in EXPERIMENTS:
        src = config.OUTPUTS_DIR / f"{experiment}_ens3" / "confusion_test.png"
        if src.exists():
            shutil.copy2(src, figure_dir / f"confusion_test_{experiment}_ens3.png")
            log.append(f"OK       {figure_dir / f'confusion_test_{experiment}_ens3.png'}")
        for run in (experiment, f"{experiment}_ens3"):
            made = make_history_png(run, figure_dir)
            if made:
                log.append(f"OK       {made}")

    # ---- 04 code -----------------------------------------------------------
    for name in ("scripts", "configs"):
        source = config.B_DIR / name
        if source.exists():
            shutil.copytree(source, code_dir / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
            log.append(f"OK       {code_dir / name}")
    for name in ("requirements.txt", "README.md", "RUNNING.md"):
        copy(config.B_DIR / name, code_dir, log, "doc")

    # ---- 05 audit ----------------------------------------------------------
    import audit_runs

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = audit_runs.main([])
    (audit_dir / "audit_runs_output.txt").write_text(buffer.getvalue(), encoding="utf-8")
    log.append(f"OK       {audit_dir / 'audit_runs_output.txt'} (exit {exit_code})")
    for name in ("seeds_summary.csv", "seeds_deltas.csv"):
        copy(config.OUTPUTS_DIR / name, audit_dir, log, "table")

    # ---- manifest ----------------------------------------------------------
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.md":
            rows.append((str(path.relative_to(out)).replace("\\", "/"),
                         f"{path.stat().st_size / 1024:.1f} KB"))
    total_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    manifest = ["# 交付包清单（自动生成）", "",
                f"生成日期：{date.today():%Y-%m-%d}　总大小：{total_mb:.1f} MB　文件数：{len(rows)}", "",
                "| 文件 | 大小 |", "|---|---|"]
    manifest += [f"| `{name}` | {size} |" for name, size in rows]
    (out / "MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    # ---- console summary ---------------------------------------------------
    print(f"delivery package: {out}")
    print(f"files: {len(rows)}  total: {total_mb:.1f} MB")
    print(f"audit exit code: {exit_code}")
    for line in log:
        if line.startswith("MISSING"):
            print(line)
    print("\nremember to write 00_READ_ME_FIRST.md and 02_给C的预测文件/HANDOFF_TO_C.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
