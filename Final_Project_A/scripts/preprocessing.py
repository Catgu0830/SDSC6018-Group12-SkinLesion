"""Frozen train-fit metadata processing and lazy RGB image transforms.

Dependencies: NumPy and Pillow. No model or Torch dependency.
Original metadata, labels, image IDs and lesion IDs are never model features.
"""

import hashlib
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image


def parse_age(value):
    if value is None or str(value).strip() == "":
        return None
    age = float(value)
    if not math.isfinite(age):
        raise ValueError("Age must be finite or blank")
    return age


def category(value):
    return str(value).strip().lower() if value is not None and str(value).strip() else "unknown"


def column_name(field, value):
    return f"{field}__{value.replace(' ', '_')}"


def fit_metadata(train_rows):
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("fit_metadata accepts only non-empty train records")
    ids = [row["image_id"] for row in train_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate fit image IDs")
    observed = [parse_age(row["age"]) for row in train_rows]
    available = [age for age in observed if age is not None]
    if not available:
        raise ValueError("Training data has no observed ages")
    median = statistics.median(available)
    filled = [median if age is None else age for age in observed]
    mean, std = statistics.mean(filled), statistics.pstdev(filled)
    if std == 0:
        raise ValueError("Training age variance is zero")
    vocab = {field: sorted({category(row[field]) for row in train_rows} | {"unknown"})
             for field in ["sex", "localization"]}
    features = ["age_z"] + [column_name(field, value) for field in vocab for value in vocab[field]]
    if len(set(features)) != len(features):
        raise ValueError("Feature column name collision")
    return {
        "fit_split": "train", "fit_n_rows": len(train_rows),
        "fit_image_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "fit_unit": "image record; lesions with multiple images contribute multiple rows",
        "age": {"imputation": "observed training median", "median": median,
                "mean_after_imputation": mean, "population_std_after_imputation": std,
                "observed_n": len(available), "missing_n": len(observed) - len(available),
                "age_missing_is_model_feature": False},
        "vocabulary": vocab, "unknown_category_policy": "blank or unseen maps to reserved unknown",
        "feature_order": features,
        "feature_modes": {"all_metadata": features, "age_only": ["age_z"],
                          "localization_only": [column_name("localization", value) for value in vocab["localization"]],
                          "image_only": []},
    }


def transform_metadata(row, metadata_config):
    age = parse_age(row["age"])
    filled = metadata_config["age"]["median"] if age is None else age
    result = {"age_missing": int(age is None), "age_imputed": filled,
              "age_z": (filled - metadata_config["age"]["mean_after_imputation"]) /
                       metadata_config["age"]["population_std_after_imputation"]}
    for field, vocabulary in metadata_config["vocabulary"].items():
        value = category(row[field])
        if value not in vocabulary:
            value = "unknown"
        for known in vocabulary:
            result[column_name(field, known)] = int(value == known)
    if any(not math.isfinite(float(value)) for value in result.values()):
        raise ValueError(f"Non-finite processed value: {row.get('image_id')}")
    return result


def default_image_config():
    return {
        "suggested_weights": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_status": "A first proposal; confirm with B before training",
        "source_url": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html",
        "resize_shorter_edge": 232, "crop_size": 224, "interpolation": "Pillow BILINEAR",
        "rgb_mean": [0.485, 0.456, 0.406], "rgb_std": [0.229, 0.224, 0.225],
        "normalization_source": "fixed pretrained ImageNet constants; not estimated from HAM10000",
        "output_dtype": "float32", "output_layout": "CHW",
        "train_augmentation": {"horizontal_flip_probability": 0.5, "rotation_degrees": [0, 90, 180, 270],
                               "key": "SHA256(ham10000-image-augmentation-v1|run_seed|epoch|image_id)"},
        "evaluation_augmentation": "none; epoch and seed do not affect val/test",
        "crop_limitations": "Center crop may remove peripheral lesion content; review train previews with B",
    }


def prepare_pil_image(path, split, image_config, *, image_id=None, epoch=0, seed=42):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    path = Path(path)
    with Image.open(path) as original:
        image = original.convert("RGB")
    width, height = image.size
    short = image_config["resize_shorter_edge"]
    if width <= height:
        size = (short, int(short * height / width))
    else:
        size = (int(short * width / height), short)
    image = image.resize(size, Image.Resampling.BILINEAR)
    crop = image_config["crop_size"]
    if min(image.size) < crop:
        raise ValueError("Resize is smaller than center crop")
    left, top = int(round((image.width - crop) / 2)), int(round((image.height - crop) / 2))
    image = image.crop((left, top, left + crop, top + crop))
    if split == "train":
        key = f"ham10000-image-augmentation-v1|{seed}|{epoch}|{image_id or path.stem}"
        bits = hashlib.sha256(key.encode()).digest()
        if bits[0] & 1:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rotation = bits[1] % 4
        if rotation:
            image = image.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                                     3: Image.Transpose.ROTATE_270}[rotation])
    return image


def image_array(path, split, image_config, *, image_id=None, epoch=0, seed=42):
    image = prepare_pil_image(path, split, image_config, image_id=image_id, epoch=epoch, seed=seed)
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - np.asarray(image_config["rgb_mean"], dtype=np.float32)) / np.asarray(image_config["rgb_std"], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)


def sample_arrays(row, config, *, data_root, mode="all_metadata", epoch=0, seed=42):
    """Read a processed CSV row; returns NumPy arrays for B's training Dataset."""
    if mode not in config["metadata"]["feature_modes"]:
        raise ValueError(f"Unknown feature mode: {mode}")
    root = Path(data_root).resolve()
    image_path = (root / row["relative_path"]).resolve()
    if not image_path.is_relative_to((root / "data/raw/images").resolve()):
        raise ValueError("Image path is outside the raw image directory")
    columns = config["metadata"]["feature_modes"][mode]
    return {
        "image": image_array(image_path, row["split"], config["image"], image_id=row["image_id"], epoch=epoch, seed=seed),
        "metadata": np.asarray([float(row[column]) for column in columns], dtype=np.float32),
        "label": int(row["label"]), "image_id": row["image_id"], "lesion_id": row["lesion_id"],
    }
