"""Materialise the raw JPEGs where the frozen A-stage tables expect them.

A's split tables store ``relative_path`` as ``data/raw/images/<image_id>.jpg``
relative to ``Final_Project_A/``. That directory is not in the repository (raw
images are git-ignored), so this script links the JPEGs that are already on disk
into it.

Default mode is ``hardlink``: the images live on the same volume as the
repository, so a hard link costs no extra space and no copy time. ``copy`` is
available as a fallback for other filesystems.

    python scripts/prepare_images.py --verify-sample 300

Exit code is non-zero when any expected image is missing or a sampled checksum
does not match the A-stage manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

import config

DEFAULT_TARGET = config.A_DIR / "data" / "raw" / "images"
DEFAULT_SOURCE_ROOT = config.REPO_ROOT / "dataverse_files"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_extracted(zip_path: Path, out_dir: Path, verbose: bool = True) -> int:
    """Extract ``zip_path`` into ``out_dir`` once; returns the number of files there."""
    if not zip_path.exists():
        return 0
    if out_dir.exists() and any(out_dir.iterdir()):
        return sum(1 for _ in out_dir.glob("*.jpg"))
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members:
            archive.extract(member, out_dir)
            count += 1
            if verbose and count % 1000 == 0:
                print(f"  extracted {count}/{len(members)}", flush=True)
    return count


def index_sources(roots) -> dict[str, Path]:
    """Map image stem -> existing JPEG path across all candidate directories."""
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.jpg"):
            index.setdefault(path.stem, path)
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--verify-sample", type=int, default=200,
                        help="number of images to checksum against the manifest (0 disables)")
    parser.add_argument("--no-extract", action="store_true", help="do not unzip the image archives")
    args = parser.parse_args(argv)

    source_root = Path(args.source_root)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    part1 = source_root / "HAM10000_images_part_1"
    part2 = source_root / "HAM10000_images_part_2"
    if not args.no_extract:
        print("checking image archives ...", flush=True)
        ensure_extracted(source_root / "HAM10000_images_part_2.zip", part2)
        ensure_extracted(source_root / "HAM10000_images_part_1.zip", part1)

    index = index_sources([part1, part2])
    print(f"source JPEGs found on disk: {len(index)}", flush=True)

    manifest = pd.read_csv(config.A_DIR / "data" / "image_manifest.csv")
    expected = manifest["image_id"].astype(str).tolist()
    print(f"images expected by A's manifest: {len(expected)}", flush=True)

    missing_sources, created, existing, failed = [], 0, 0, []
    for count, image_id in enumerate(expected, start=1):
        destination = target / f"{image_id}.jpg"
        if destination.exists():
            existing += 1
            continue
        source = index.get(image_id)
        if source is None:
            missing_sources.append(image_id)
            continue
        try:
            if args.mode == "hardlink":
                destination.hardlink_to(source)
            else:
                shutil.copy2(source, destination)
            created += 1
        except OSError as exc:
            failed.append(f"{image_id}: {exc}")
        if count % 2000 == 0:
            print(f"  linked {created} new / {existing} existing ({count}/{len(expected)})", flush=True)

    print(f"created={created} already_present={existing} "
          f"missing_source={len(missing_sources)} failed={len(failed)}", flush=True)
    if missing_sources[:5]:
        print("  first missing:", missing_sources[:5], flush=True)
    if failed[:3]:
        print("  first failures:", failed[:3], flush=True)

    status = 0
    if args.verify_sample and len(expected):
        sample_size = min(args.verify_sample, len(expected))
        sample = random.Random(0).sample(expected, sample_size)
        manifest_index = manifest.set_index("image_id")
        bad = []
        for image_id in sample:
            path = target / f"{image_id}.jpg"
            if not path.exists():
                bad.append(f"{image_id}: absent")
                continue
            actual = sha256_of(path)
            if actual != manifest_index.loc[image_id, "sha256"]:
                bad.append(f"{image_id}: checksum mismatch")
        print(f"checksum sample: {sample_size - len(bad)}/{sample_size} match", flush=True)
        if bad:
            print("  problems:", bad[:5], flush=True)
            status = 1

    present = sum(1 for image_id in expected if (target / f"{image_id}.jpg").exists())
    print(f"images now available at {target}: {present}/{len(expected)}", flush=True)
    if present != len(expected):
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
