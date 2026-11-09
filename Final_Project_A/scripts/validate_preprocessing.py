"""Independent numeric, leakage, reproducibility and full-image smoke checks.

Usage: python3 Final_Project_A/scripts/validate_preprocessing.py
Dependencies: NumPy, pandas, Pillow. Does not fit on held-out data or train models.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version

from preprocessing import fit_metadata, image_array, sample_arrays, transform_metadata


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/v1_seed42"
SPLITS = ROOT / "data/splits/v1_seed42"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    config = json.loads((PROCESSED / "preprocessing_config.json").read_text())
    schema = json.loads((PROCESSED / "feature_schema.json").read_text())
    frozen = json.loads((PROCESSED / "checksums.json").read_text())
    assert all(file_sha256(PROCESSED / name) == expected for name, expected in frozen.items())
    for name in ["train.csv", "val.csv", "test.csv"]:
        assert file_sha256(SPLITS / name) == config["source_sha256"][name]
    for name in ["preprocessing.py", "preprocess_data.py"]:
        assert file_sha256(ROOT / "scripts" / name) == config["source_sha256"][name]
    source = {split: pd.read_csv(SPLITS / f"{split}.csv", keep_default_na=False) for split in ["train", "val", "test"]}
    frames = {split: pd.read_csv(PROCESSED / f"{split}.csv", keep_default_na=False) for split in source}
    all_rows = []
    for split, frame in frames.items():
        assert frame[source[split].columns].equals(source[split])
        features = frame[schema["feature_order"]].to_numpy(dtype=float)
        assert features.shape == (len(frame), 19) and np.isfinite(features).all()
        for field in ["sex", "localization"]:
            columns = [column for column in schema["feature_order"] if column.startswith(field + "__")]
            one_hot = frame[columns].to_numpy(dtype=float)
            assert np.isin(one_hot, [0.0, 1.0]).all() and (one_hot.sum(axis=1) == 1).all()
        observed = pd.to_numeric(source[split].age, errors="coerce")
        assert np.array_equal(frame.age_missing.to_numpy(dtype=int), observed.isna().to_numpy(dtype=int))
        independently_filled = observed.fillna(config["metadata"]["age"]["median"]).to_numpy(dtype=float)
        assert np.array_equal(independently_filled, frame.age_imputed.to_numpy(dtype=float))
        expected_z = (independently_filled - config["metadata"]["age"]["mean_after_imputation"]) / config["metadata"]["age"]["population_std_after_imputation"]
        assert np.allclose(expected_z, frame.age_z.to_numpy(dtype=float), atol=1e-12, rtol=0)
        all_rows.extend(frame.to_dict("records"))
    age = pd.to_numeric(source["train"].age, errors="coerce")
    median = float(age.median())
    filled = age.fillna(median).to_numpy(dtype=float)
    assert median == config["metadata"]["age"]["median"] == 50.0
    assert abs(filled.mean() - config["metadata"]["age"]["mean_after_imputation"]) < 1e-12
    assert abs(filled.std(ddof=0) - config["metadata"]["age"]["population_std_after_imputation"]) < 1e-12
    assert abs(frames["train"].age_z.mean()) < 1e-12
    assert abs(frames["train"].age_z.std(ddof=0) - 1) < 1e-12
    assert config["metadata"]["fit_n_rows"] == len(source["train"]) == 7050
    assert all(set(config["metadata"]["vocabulary"][field]) == set(source["train"][field]) | {"unknown"} for field in ["sex", "localization"])
    features = set(schema["feature_order"])
    assert not features & {"label", "dx", "age", "age_imputed", "age_missing", "image_id", "lesion_id", "dx_type", "dataset", "split"}
    before = json.dumps(config, sort_keys=True)
    stress = {**source["val"].iloc[0].to_dict(), "age": "120", "sex": "unseen sex", "localization": "unseen site"}
    transformed = transform_metadata(stress, config["metadata"])
    assert transformed["sex__unknown"] == transformed["localization__unknown"] == 1
    assert transformed["age_z"] == (120 - config["metadata"]["age"]["mean_after_imputation"]) / config["metadata"]["age"]["population_std_after_imputation"]
    assert json.dumps(config, sort_keys=True) == before
    try:
        fit_metadata(source["train"].to_dict("records") + [source["val"].iloc[0].to_dict()])
    except ValueError as error:
        assert "only non-empty train" in str(error)
    else:
        raise AssertionError("Mixed train/validation fitting was not rejected")
    with TemporaryDirectory(prefix="ham10000_preprocess_repro_", dir="/private/tmp") as temp:
        subprocess.run([sys.executable, str(ROOT / "scripts/preprocess_data.py"), "--output-dir", temp, "--no-report"], check=True, capture_output=True, text=True)
        assert all((Path(temp) / path.name).read_bytes() == path.read_bytes() for path in PROCESSED.iterdir() if path.is_file())
        second = subprocess.run([sys.executable, str(ROOT / "scripts/preprocess_data.py"), "--output-dir", temp, "--no-report"], capture_output=True, text=True)
        assert second.returncode != 0 and "protected" in second.stderr
        synthetic = Path(temp) / "constant_rgb.png"
        Image.new("RGB", (600, 450), (255, 0, 128)).save(synthetic)
        array = image_array(synthetic, "test", config["image"])
        expected_pixel = (np.asarray([1.0, 0.0, 128.0 / 255.0], dtype=np.float32) - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        assert array.shape == (3, 224, 224) and np.allclose(array, expected_pixel[:, None, None], atol=1e-6)
    for split in ["val", "test"]:
        for _, row in frames[split].groupby("dx", sort=True).first().iterrows():
            path = ROOT / row.relative_path
            a = image_array(path, split, config["image"], image_id=row.image_id, epoch=0, seed=42)
            b = image_array(path, split, config["image"], image_id=row.image_id, epoch=50, seed=12345)
            assert np.array_equal(a, b)
    changed = 0
    for row in frames["train"].head(20).to_dict("records"):
        path = ROOT / row["relative_path"]
        a = image_array(path, "train", config["image"], image_id=row["image_id"], epoch=0, seed=42)
        b = image_array(path, "train", config["image"], image_id=row["image_id"], epoch=0, seed=42)
        c = image_array(path, "train", config["image"], image_id=row["image_id"], epoch=1, seed=42)
        assert np.array_equal(a, b)
        changed += int(not np.array_equal(a, c))
    assert changed > 0
    first = frames["train"].iloc[0].to_dict()
    for mode, dim in {"all_metadata": 19, "age_only": 1, "localization_only": 15, "image_only": 0}.items():
        assert sample_arrays(first, config, data_root=ROOT, mode=mode)["metadata"].shape == (dim,)

    def check_image(row):
        sample = sample_arrays(row, config, data_root=ROOT)
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == np.float32 and sample["image"].flags.c_contiguous
        assert np.isfinite(sample["image"]).all()
        assert sample["metadata"].shape == (19,) and sample["metadata"].dtype == np.float32
        assert np.isfinite(sample["metadata"]).all()
        assert sample["label"] == int(row["label"])
        assert file_sha256(ROOT / row["relative_path"]) == row["image_sha256"]
        return row["split"]

    counts = {"train": 0, "val": 0, "test": 0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(check_image, row) for row in all_rows]
        for completed, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1
            if completed % 1000 == 0 or completed == len(all_rows):
                print(f"Validated model inputs and unchanged source bytes: {completed}/{len(all_rows)}", flush=True)
    assert counts == {split: len(frame) for split, frame in frames.items()}
    report = {
        "independent_numeric_and_one_hot_checks": "passed",
        "original_fields_and_missing_flags_preserved": "passed",
        "fit_statistics_and_vocabulary_from_train_only": "passed",
        "mixed_split_fit_rejected": "passed", "unseen_category_fallback_without_refitting": "passed",
        "all_6_output_files_byte_identical_on_reproduction": "passed", "existing_version_overwrite_rejected": "passed",
        "rgb_normalization_known_color_test": "passed", "validation_test_seed_epoch_invariance": "passed",
        "training_seed_epoch_reproducibility": "passed", "different_training_epoch_changes_some_augmentations": "passed",
        "feature_mode_dimensions": {"all_metadata": 19, "age_only": 1, "localization_only": 15, "image_only": 0},
        "all_image_inputs_validated": counts, "all_original_image_sha256_unchanged": True,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "Pillow": pillow_version},
        "torchvision_pixel_equivalence_tested": False, "model_training_performed": False,
    }
    (ROOT / "docs/preprocessing_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
