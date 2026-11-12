"""Datasets, image transforms and data loaders for HAM10000 (B stage).

The split tables come from the frozen A-stage package
(``Final_Project_A/data/processed/v1_seed42``). Those tables already contain the
19 metadata features fitted on training records only, so this module never
re-fits or re-derives a feature: doing so could silently diverge from A's
checksummed artefacts.

Image recipe (matches ``Final_Project_A/docs/06_preprocessing_method.md``):
    resize shorter side to 232 px (bilinear) -> centre crop 224 -> /255 ->
    ImageNet normalisation. Training adds a horizontal flip (p=0.5) and a
    rotation drawn from {0, 90, 180, 270} degrees after cropping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

#: Malignant vs benign grouping, used only by the literature-comparison control.
#: Several widely cited HAM10000 papers report accuracy on this binary task while
#: stating a 7-class dataset, which changes the achievable accuracy substantially.
MALIGNANT = {"akiec", "bcc", "mel"}
CLASS_NAMES_BINARY = ("benign", "malignant")


class RandomQuarterTurn:
    """Rotate a PIL image by a random multiple of 90 degrees (lossless)."""

    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = self.angles[int(torch.randint(len(self.angles), (1,)).item())]
        return img if angle == 0 else img.rotate(angle, expand=False)


def build_transform(train: bool, image_size: int = 224, resize_to: int = 232, aug: dict | None = None):
    """Deterministic evaluation transform, or the training transform with augmentation.

    ``aug`` is optional so the default recipe stays byte-for-byte identical to the
    frozen A-stage configuration. It supports:

    ``rrc_scale``      [lo, hi] random-resized-crop scale (replaces resize+centre-crop)
    ``color_jitter``   float intensity for brightness/contrast/saturation (hue = 0.02)
    ``erasing``        float probability of random erasing after normalisation
    ``quarter_turn``   bool, rotate by a random multiple of 90 degrees
    ``flip``           float probability of horizontal flip
    """
    aug = aug or {}
    image_steps = []
    rrc_scale = aug.get("rrc_scale")
    if train and rrc_scale:
        image_steps.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(rrc_scale),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    else:
        image_steps += [
            transforms.Resize(resize_to, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]

    if train:
        image_steps.append(transforms.RandomHorizontalFlip(p=float(aug.get("flip", 0.5))))
        if aug.get("quarter_turn", True):
            image_steps.append(RandomQuarterTurn())
        if aug.get("color_jitter"):
            strength = float(aug["color_jitter"])
            image_steps.append(
                transforms.ColorJitter(
                    brightness=strength, contrast=strength, saturation=strength, hue=0.02
                )
            )

    steps = [
        *image_steps,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if train and aug.get("erasing"):
        steps.append(transforms.RandomErasing(p=float(aug["erasing"]), value="random"))
    return transforms.Compose(steps)


def read_split(which: str, base=None) -> pd.DataFrame:
    """Read one subset table (``train`` / ``val`` / ``test``)."""
    df = pd.read_csv(config.split_csv(which, base))
    required = {"image_id", "lesion_id", "dx", "label", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{which}.csv is missing required columns: {sorted(missing)}")
    return df


def resolve_feature_columns(mode: str) -> list[str]:
    """Feature column list for a metadata mode, as declared by A's schema."""
    schema = config.load_feature_schema()
    modes = schema["feature_modes"]
    if mode not in modes:
        raise KeyError(f"unknown metadata mode {mode!r}; available: {sorted(modes)}")
    cols = list(modes[mode])
    known = set(schema["feature_order"])
    unknown = [c for c in cols if c not in known]
    if unknown:
        raise ValueError(f"schema lists unknown feature columns: {unknown}")
    return cols


class HAM10000Dataset(Dataset):
    """Image (+ optional metadata) -> label, with the row index for traceability."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        transform,
        image_root: Path = config.IMAGE_ROOT,
    ):
        self.frame = frame.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.transform = transform
        self.image_root = Path(image_root)
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)
        self.paths = [self.image_root / p for p in self.frame["relative_path"]]
        if self.feature_columns:
            self.meta = self.frame[self.feature_columns].to_numpy(dtype=np.float32)
        else:
            self.meta = np.zeros((len(self.frame), 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"image not found: {path}\n"
                "Run scripts/prepare_images.py to materialise data/raw/images first."
            ) from exc
        image = self.transform(image)
        meta = torch.from_numpy(self.meta[index].copy())
        return image, meta, torch.tensor(self.labels[index]), torch.tensor(index)


def _worker_init(_worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % (2**32))


def make_loader(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cfg: dict,
    train: bool,
) -> DataLoader:
    data_cfg = cfg["data"]
    dataset = HAM10000Dataset(
        frame,
        feature_columns,
        build_transform(
            train, data_cfg["image_size"], data_cfg["resize_to"], data_cfg.get("aug")
        ),
    )
    workers = int(data_cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=train,
        num_workers=workers,
        pin_memory=bool(data_cfg["pin_memory"]),
        # BatchNorm inside the metadata branch cannot handle a final batch of size 1.
        drop_last=train,
        # Keep the worker pool alive between epochs: re-spawning on Windows costs
        # seconds per epoch and, with three loaders, leaves orphaned processes.
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=_worker_init,
    )


def make_loaders(cfg: dict) -> dict:
    """Build train/val/test loaders plus the metadata needed by the models."""
    feature_mode = cfg["data"]["metadata_mode"]
    feature_columns = resolve_feature_columns(feature_mode)

    frames = {
        name: read_split(name, config.resolve_split_base(cfg))
        for name in ("train", "val", "test")
    }

    label_mode = cfg.get("data", {}).get("label_mode", "multiclass")
    if label_mode == "binary":
        for frame in frames.values():
            frame["label"] = frame["dx"].isin(MALIGNANT).astype(int)
        class_names = CLASS_NAMES_BINARY
    elif label_mode == "multiclass":
        class_names = CLASS_NAMES
    else:
        raise ValueError(f"unknown label_mode: {label_mode!r}")

    label_mapping = config.load_label_mapping()
    for name, frame in frames.items():
        if frame["label"].min() < 0 or frame["label"].max() > len(class_names) - 1:
            raise ValueError(f"{name} contains labels outside the frozen mapping")

    loaders = {
        name: make_loader(frame, feature_columns, cfg, train=(name == "train"))
        for name, frame in frames.items()
    }

    class_counts = (
        frames["train"]["label"].value_counts().reindex(range(len(class_names)), fill_value=0)
    )
    return {
        "loaders": loaders,
        "frames": frames,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "class_counts": class_counts.to_numpy(),
        "class_names": class_names,
        "label_mapping": label_mapping,
    }
