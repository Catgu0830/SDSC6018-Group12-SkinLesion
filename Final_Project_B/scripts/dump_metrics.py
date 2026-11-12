"""Dump every metric produced by the B stage into one readable report.

Reads only the artefacts under ``outputs/`` - it never re-runs training - and
writes ``outputs/metrics_report.md`` so that every number quoted in the final
report can be traced back to a file.

    python scripts/dump_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import config
import run_all

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
OUT = config.OUTPUTS_DIR / "metrics_report.md"
LINES: list[str] = []


def emit(text: str = "") -> None:
    LINES.append(text)
    print(text)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seed_runs(experiment: str, seeds=(42, 43, 44)) -> list[str]:
    names = [experiment if s == 42 else f"{experiment}_s{s}" for s in seeds]
    return names + [f"{experiment}_ens{len(seeds)}"]


def table_from_runs(names, split: str) -> pd.DataFrame:
    rows = []
    for name in names:
        payload = read_json(config.OUTPUTS_DIR / name / "summary.json")
        if payload is None:
            continue
        block = payload.get(split, {})
        rows.append({
            "run": name,
            "accuracy": round(block.get("accuracy", float("nan")), 4),
            "macro_f1": round(block.get("macro_f1", float("nan")), 4),
            "weighted_f1": round(block.get("weighted_f1", float("nan")), 4),
            "auc_ovr": round(block.get("auc_ovr_macro", float("nan")), 4),
            "best_epoch": payload.get("best_epoch"),
        })
    return pd.DataFrame(rows)


def md(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(out)


def per_class_table(experiment: str) -> pd.DataFrame:
    merged = None
    for label, run in (("baseline_ens3", f"{experiment}_ens3"),):
        path = config.OUTPUTS_DIR / run / "per_class_test.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.rename(columns={c: f"{c}" for c in frame.columns})
        frame["model"] = label
        merged = frame
    return merged if merged is not None else pd.DataFrame()


def main(argv=None) -> int:
    emit("# B 阶段全部指标汇总（自动生成）")
    emit()
    emit(f"数据目录：`{config.OUTPUTS_DIR}`")
    emit("切分：`Final_Project_A` 冻结的 `v1_seed42`（病灶级 70/15/15）")
    emit()

    # ---------------- 1. main experiments, all runs ----------------
    emit("## 1. 七分类主线：每个运行的全部指标")
    emit()
    for split in ("test", "val"):
        names = []
        for experiment in run_all.EXPERIMENTS:
            names += seed_runs(experiment)
        frame = table_from_runs(names, split)
        emit(f"### 1.{1 if split == 'test' else 2} {split}")
        emit()
        emit(md(frame))
        emit()

    # ---------------- 2. aggregates ----------------
    emit("## 2. 均值 ± 标准差（3 种子，不含集成）")
    emit()
    for path, title in (
        ("seeds_aggregate.csv", "按实验汇总"),
        ("seeds_deltas.csv", "配对差值（同种子下 融合 − 基线）"),
    ):
        frame_path = config.OUTPUTS_DIR / path
        if frame_path.exists():
            emit(f"### 2.{1 if 'aggregate' in path else 2} {title}")
            emit()
            emit(md(pd.read_csv(frame_path).round(4)))
            emit()

    # ---------------- 3. ensembles ----------------
    emit("## 3. 三种子集成（test）")
    emit()
    ens_path = config.OUTPUTS_DIR / "ensemble_test.csv"
    if ens_path.exists():
        emit(md(pd.read_csv(ens_path).round(4)))
        emit()

    # ---------------- 4. per-class ----------------
    emit("## 4. 逐类指标（集成模型，test）")
    emit()
    for experiment in run_all.EXPERIMENTS:
        path = config.OUTPUTS_DIR / f"{experiment}_ens3" / "per_class_test.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).round(3)
        emit(f"### {experiment}")
        emit()
        emit(md(frame))
        emit()

    # ---------------- 5. confusion matrices ----------------
    emit("## 5. 混淆矩阵（集成模型，test）")
    emit()
    for experiment in run_all.EXPERIMENTS:
        payload = read_json(config.OUTPUTS_DIR / f"{experiment}_ens3" / "confusion_test.csv")
        if payload is None:
            payload = read_json(config.OUTPUTS_DIR / f"{experiment}_ens3" / "metrics_test.json")
            if payload:
                payload = {"labels": CLASS_NAMES, "matrix": payload["confusion_matrix"]}
        if not payload:
            continue
        labels = payload["labels"]
        frame = pd.DataFrame(payload["matrix"], index=labels, columns=labels)
        emit(f"### {experiment}（行=真实，列=预测）")
        emit()
        emit(md(frame.reset_index().rename(columns={"index": "true\\pred"})))
        emit()

    # ---------------- 6. class weights ----------------
    emit("## 6. 类别权重敏感性（单种子 42）")
    emit()
    rows = []
    for experiment in run_all.EXPERIMENTS:
        for suffix, label in (("", "pure CE"), ("_w", "sqrt_inverse")):
            payload = read_json(config.OUTPUTS_DIR / f"{experiment}{suffix}" / "summary.json")
            if payload is None:
                continue
            rows.append({
                "experiment": experiment,
                "loss": label,
                "test_acc": round(payload["test"]["accuracy"], 4),
                "test_macro_f1": round(payload["test"]["macro_f1"], 4),
                "test_auc": round(payload["test"]["auc_ovr_macro"], 4),
                "best_epoch": payload["best_epoch"],
            })
    emit(md(pd.DataFrame(rows)))
    emit()

    emit("### 6.1 逐类 F1 对比（加权 − 纯 CE）")
    emit()
    for experiment in ("baseline_image", "fusion_all_metadata"):
        a = pd.read_csv(config.OUTPUTS_DIR / experiment / "per_class_test.csv").set_index("class")
        b = pd.read_csv(config.OUTPUTS_DIR / f"{experiment}_w" / "per_class_test.csv").set_index("class")
        frame = pd.DataFrame({
            "f1_CE": a["f1"].round(3),
            "f1_weighted": b["f1"].round(3),
            "d_f1": (b["f1"] - a["f1"]).round(3),
            "recall_CE": a["recall"].round(3),
            "recall_weighted": b["recall"].round(3),
            "d_recall": (b["recall"] - a["recall"]).round(3),
        })
        emit(f"#### {experiment}")
        emit()
        emit(md(frame.reset_index()))
        emit()

    # ---------------- 7. binary ----------------
    emit("## 7. 二分类（恶性 / 良性）")
    emit()
    rows = []
    for run, label in (
        ("binary_base", "dedicated binary, plain aug"),
        ("binary_strong", "dedicated binary, strong aug"),
        ("baseline_image", "7-class baseline -> sum"),
        ("fusion_all_metadata", "7-class fusion(all) -> sum"),
        ("fusion_localization_only", "7-class fusion(loc) -> sum"),
    ):
        payload = read_json(config.OUTPUTS_DIR / run / "binary_report_tta_accuracy.json")
        if payload is None:
            continue
        val = payload["val_at_threshold"]
        test = payload["test_at_threshold"]
        half = payload["test_at_half"]
        rows.append({
            "source": label,
            "val_acc": round(val["accuracy"], 4),
            "thr": round(payload["val_threshold"], 3),
            "test_acc": round(test["accuracy"], 4),
            "test_bal_acc": round(test["balanced_accuracy"], 4),
            "test_precision": round(test["precision_malignant"], 4),
            "test_recall": round(test["recall_malignant"], 4),
            "test_f1": round(test["f1_malignant"], 4),
            "test_auc": round(test["auc"], 4),
            "test_acc@0.5": round(half["accuracy"], 4),
        })
    emit(md(pd.DataFrame(rows)))
    emit()

    # ---------------- 8. controls ----------------
    emit("## 8. 泄漏对照（不是我们的结果）")
    emit()
    rows = []
    for run, label, task in (
        ("control_image_level", "image-level split", "7-class"),
        ("control_binary_image_level", "image-level split", "binary"),
    ):
        payload = read_json(config.OUTPUTS_DIR / run / "summary.json")
        if payload is None:
            continue
        rows.append({
            "control": label,
            "task": task,
            "test_acc": round(payload["test"]["accuracy"], 4),
            "test_macro_f1": round(payload["test"]["macro_f1"], 4),
            "test_auc": round(payload["test"]["auc_ovr_macro"], 4),
        })
    emit(md(pd.DataFrame(rows)))
    emit()

    leak = read_json(config.OUTPUTS_DIR / "control_image_level" / "leakage_analysis.json")
    if leak:
        emit("### 8.1 暴露 vs 干净（图像级切分）")
        emit()
        frame = pd.DataFrame([leak["all"], leak["exposed"], leak["clean"]])
        frame.insert(0, "subset", ["all", "exposed", "clean"])
        emit(md(frame))
        emit()

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
