"""Configuration loading and path resolution for the B-stage (modelling) experiments.

All entry points under ``scripts/`` import this module first so that every path is
resolved from one place and no experiment depends on the current working directory.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed locations (derived from this file, never from os.getcwd())
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
B_DIR = SCRIPTS_DIR.parent                      # Final_Project_B/
REPO_ROOT = B_DIR.parent                        # repository root
A_DIR = REPO_ROOT / "Final_Project_A"           # A-stage package (frozen inputs)

SPLIT_VERSION = "v1_seed42"
SPLITS_DIR = A_DIR / "data" / "splits" / SPLIT_VERSION
PROCESSED_DIR = A_DIR / "data" / "processed" / SPLIT_VERSION
CONFIGS_DIR = B_DIR / "configs"
OUTPUTS_DIR = B_DIR / "outputs"

#: A's split tables store paths like ``data/raw/images/ISIC_0027419.jpg`` relative
#: to ``Final_Project_A/``.
IMAGE_ROOT = A_DIR


def load_config(path: str | Path) -> dict:
    """Load an experiment config, resolving its optional ``inherit`` chain.

    ``path`` may be a bare experiment name (``baseline_image``), a file name, an
    absolute path, or a path relative to the working directory.
    """
    path = Path(path)
    if not path.is_absolute():
        candidate = CONFIGS_DIR / path
        if candidate.suffix == "" and candidate.with_suffix(".json").exists():
            candidate = candidate.with_suffix(".json")
        if candidate.exists():
            path = candidate
    # utf-8-sig tolerates a BOM, which some editors add to JSON files on Windows.
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(parent)
        cfg = _deep_merge(base, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply ``--set a.b.c=value`` overrides. Values are parsed as JSON when possible."""
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg


def experiment_name(cfg: dict) -> str:
    """Directory name for a run: ``<run.name>`` or the config file stem."""
    run = cfg.get("run", {})
    return str(run.get("name") or Path(cfg["_config_path"]).stem)


def output_dir(cfg: dict) -> Path:
    out = OUTPUTS_DIR / experiment_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_label_mapping() -> dict[str, int]:
    path = SPLITS_DIR / "label_mapping.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_schema() -> dict:
    path = PROCESSED_DIR / "feature_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def split_csv(which: str, base: Path | None = None) -> Path:
    base = Path(base) if base else PROCESSED_DIR
    path = base / f"{which}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; A-stage processed tables are required for exact feature parity"
        )
    return path


def resolve_split_base(cfg: dict) -> Path:
    """Directory holding ``train/val/test.csv``.

    Defaults to A's frozen processed tables. A config may point ``data.split_dir``
    elsewhere to run a **control** experiment (for example the deliberately leaky
    image-level split used to compare against the published literature).
    """
    raw = cfg.get("data", {}).get("split_dir")
    if not raw:
        return PROCESSED_DIR
    path = Path(raw)
    return path if path.is_absolute() else (B_DIR / path)


def check_inputs() -> list[str]:
    """Return a list of human-readable problems with the frozen inputs."""
    problems = []
    for name, path in (
        ("A-stage split tables", SPLITS_DIR),
        ("A-stage processed tables", PROCESSED_DIR),
        ("A-stage feature schema", PROCESSED_DIR / "feature_schema.json"),
        ("A-stage image manifest", A_DIR / "data" / "image_manifest.csv"),
    ):
        if not path.exists():
            problems.append(f"{name} missing: {path}")
    return problems
