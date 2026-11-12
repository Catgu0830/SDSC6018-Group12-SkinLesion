"""Training loop, optimisation setup, checkpointing and logging for the B stage.

Kept deliberately small: one function trains one model end-to-end and returns a
history table plus the best checkpoint, so every ablation run follows exactly the
same code path.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import metrics as metrics_lib


class Logger:
    """Print to stdout and append the same lines to ``run.log``."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no CUDA device found, training on CPU will be very slow", file=sys.stderr)
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: dict):
    """AdamW with a smaller learning rate for the pretrained visual backbone."""
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    backbone_lr = lr * float(train_cfg.get("backbone_lr_mult", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("features.") else head_params).append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_criterion(cfg: dict, class_counts: np.ndarray, device: torch.device) -> nn.Module:
    loss_cfg = cfg["train"].get("loss", {})
    weights = loss_cfg.get("class_weights", "none")
    smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    if weights == "none":
        weight = None
    elif weights == "inverse":
        freq = class_counts.astype(np.float64) / class_counts.sum()
        weight = torch.tensor((1.0 / np.clip(freq, 1e-9, None)) / len(freq), dtype=torch.float32)
        weight = weight / weight.sum() * len(freq)
    elif weights == "sqrt_inverse":
        freq = class_counts.astype(np.float64) / class_counts.sum()
        weight = torch.tensor(1.0 / np.sqrt(np.clip(freq, 1e-9, None)), dtype=torch.float32)
        weight = weight / weight.sum() * len(freq)
    else:
        raise ValueError(f"unknown class_weights setting: {weights!r}")

    if weight is not None:
        weight = weight.to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)


def _run_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    running_loss, seen = 0.0, 0

    for step, (images, meta, labels, _index) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images, meta)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += float(loss.detach()) * len(labels)
        seen += len(labels)

    return running_loss / max(seen, 1)


def fit(
    cfg: dict,
    model: nn.Module,
    loaders: dict,
    class_counts: np.ndarray,
    class_names,
    feature_columns,
    device: torch.device,
    out_dir: Path,
    logger: Logger,
) -> tuple[pd.DataFrame, Path]:
    """Train with AMP, early-stop on validation macro-F1, keep the best checkpoint."""
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("patience", 6))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    max_batches = train_cfg.get("max_batches")

    criterion = build_criterion(cfg, class_counts, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(train_cfg["lr"]) * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_score, best_epoch, stale = -1.0, 0, 0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = _run_epoch(
            model, loaders["train"], criterion, device, amp,
            optimizer=optimizer, scaler=scaler, max_batches=max_batches,
        )
        val_loss = _run_epoch(
            model, loaders["val"], criterion, device, amp, max_batches=max_batches
        )
        y_true, probs, _ = metrics_lib.collect_probabilities(model, loaders["val"], device, amp)
        val_metrics = metrics_lib.compute_metrics(y_true, probs, class_names)

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_auc": val_metrics["auc_ovr_macro"],
            "lr": lr_now,
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        logger(
            f"epoch {epoch:>3}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| val_acc {val_metrics['accuracy']:.4f} "
            f"| val_macroF1 {val_metrics['macro_f1']:.4f} "
            f"| val_AUC {val_metrics['auc_ovr_macro']:.4f} | {row['seconds']:.0f}s"
        )

        scheduler.step()

        if val_metrics["macro_f1"] > best_score + 1e-4:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "class_names": list(class_names),
                    "feature_columns": list(feature_columns),
                },
                best_path,
            )
            logger(f"  -> new best (val macro-F1 {best_score:.4f}), checkpoint saved")
        else:
            stale += 1
            if stale >= patience:
                logger(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out_dir / "history.csv", index=False)
    logger(f"best epoch: {best_epoch}  best val macro-F1: {best_score:.4f}")
    return history_frame, best_path


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def predict_frame(model, loader, frame, class_names, device, amp: bool = True) -> pd.DataFrame:
    """Per-sample predictions joined back to the split rows (the hand-off to C)."""
    y_true, probs, index = metrics_lib.collect_probabilities(model, loader, device, amp)
    out = frame.iloc[index][["image_id", "lesion_id", "dx", "label"]].reset_index(drop=True).copy()
    out["y_true"] = y_true
    out["y_pred"] = probs.argmax(axis=1)
    for k, name in enumerate(class_names):
        out[f"prob_{name}"] = probs[:, k]
    out["correct"] = (out["y_true"] == out["y_pred"]).astype(int)
    return out
