"""Evaluation metrics and plots shared by every B-stage experiment.

The headline numbers must stay identical to the proposal's metric list:
accuracy, macro-F1, one-vs-rest AUC and the confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@torch.no_grad()
def collect_probabilities(model, loader, device, amp: bool = True, tta: bool = False):
    """Run a model over a loader and return (y_true, probabilities, row indices).

    ``tta=True`` averages softmax outputs over five lossless views (identity,
    horizontal flip and the three quarter turns). Quarter turns are exact on a
    square crop and dermatoscopic images have no canonical orientation, so this is
    a legitimate inference-time average rather than a label-changing augmentation.
    """
    model.eval()
    all_probs, all_labels, all_index = [], [], []
    use_amp = amp and device.type == "cuda"

    def views(images):
        if not tta:
            yield images
            return
        yield images
        yield torch.flip(images, dims=[-1])
        for k in (1, 2, 3):
            yield torch.rot90(images, k, dims=[-2, -1])

    for images, meta, labels, index in loader:
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        accumulated = None
        for view in views(images):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(view, meta)
            probs = torch.softmax(logits.float(), dim=1)
            accumulated = probs if accumulated is None else accumulated + probs
        probs = accumulated / (5.0 if tta else 1.0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_index.append(index.numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_index),
    )


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names) -> dict:
    """Global metrics plus per-class precision/recall/F1/support."""
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    try:
        if len(class_names) == 2:
            # Binary task: sklearn wants a single score column, not a 2-column matrix.
            auc_macro = float(roc_auc_score(y_true, probs[:, 1], labels=labels))
        else:
            auc_macro = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError:
        # Happens when a class is absent from the evaluated subset.
        auc_macro = float("nan")

    auc_per_class = {}
    for k, name in enumerate(class_names):
        binary = (y_true == k).astype(int)
        if binary.sum() in (0, len(binary)):
            auc_per_class[name] = float("nan")
            continue
        try:
            auc_per_class[name] = float(roc_auc_score(binary, probs[:, k]))
        except ValueError:
            auc_per_class[name] = float("nan")

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "auc_ovr_macro": auc_macro,
        "per_class": {
            name: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
                "auc": auc_per_class[name],
            }
            for k, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def metrics_frame(metrics: dict, class_names) -> pd.DataFrame:
    """Flatten the per-class part of a metrics dict into a tidy table."""
    rows = []
    for name in class_names:
        row = {"class": name}
        row.update(metrics["per_class"][name])
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_png(matrix, class_names, path: Path, title: str = "Confusion matrix") -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_history_png(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["epoch"], history["val_acc"], label="val accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
