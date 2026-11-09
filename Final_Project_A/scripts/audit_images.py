"""Verify official archives, decode all images, and preserve original JPEG bytes.

Usage: python3 Final_Project_A/scripts/audit_images.py
Dependencies: Python 3.9+ and Pillow. No network access or model training.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs"
IMAGES = RAW / "images"
ImageFile.LOAD_TRUNCATED_IMAGES = False


def file_hash(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_examples(rows):
    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    selected = {label: next(row for row in rows if row["dx"] == label) for label in labels}
    canvas = Image.new("RGB", (1120, 600), "#f5f5f3")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((20, 12), "HAM10000 original images: one example per recorded class", fill="black", font=title_font)
    draw.text((20, 43), "Labels are dataset annotations. Thumbnails are for inspection only.", fill="#555555", font=font)
    for i, label in enumerate(labels):
        row = selected[label]
        x, y = (i % 4) * 280 + 10, (i // 4) * 260 + 82
        with Image.open(IMAGES / f"{row['image_id']}.jpg") as original:
            thumb = ImageOps.contain(original.convert("RGB"), (260, 196))
            canvas.paste(thumb, (x + (260 - thumb.width) // 2, y))
        draw.text((x, y + 201), f"{label}  |  {row['image_id']}", fill="black", font=font)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    canvas.save(out / "original_image_examples.png")


def run():
    DOCS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    record_path = RAW / "harvard_dataset_record_v4.json"
    record = json.loads(record_path.read_text())
    official = {entry["dataFile"]["filename"]: entry["dataFile"] for entry in record["data"]["latestVersion"]["files"]}
    with (RAW / "HAM10000_metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata_source = official["HAM10000_metadata.tab"]
    if file_hash(RAW / "HAM10000_metadata.csv", "md5") != metadata_source["checksum"]["value"]:
        raise ValueError("Metadata no longer matches its official checksum")
    expected = {row["image_id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate metadata image IDs")
    provenance = []
    archives = []
    entries = {}
    try:
        for name in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]:
            source = official[name]
            final = RAW / name
            download = final if final.exists() else RAW / (name + ".part")
            if not download.exists():
                raise FileNotFoundError(f"Download missing: {download}")
            if download.stat().st_size != source["filesize"]:
                raise ValueError(f"Incomplete or wrong archive size: {download.name}")
            actual_md5 = file_hash(download, "md5")
            if actual_md5 != source["checksum"]["value"]:
                raise ValueError(f"Official MD5 mismatch: {download.name}")
            if download != final:
                download.rename(final)
            print(f"Verified official size and MD5: {name}", flush=True)
            provenance.append({
                "file": name, "official_file_id": source["id"],
                "download_url": f"https://dataverse.harvard.edu/api/access/datafile/{source['id']}",
                "bytes": final.stat().st_size, "expected_md5": source["checksum"]["value"],
                "actual_md5": actual_md5, "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            })
            archive = zipfile.ZipFile(final)
            archives.append(archive)
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"Unsafe archive member: {info.filename}")
                if member.suffix.lower() not in {".jpg", ".jpeg"}:
                    archive.read(info)  # Validate CRC for any non-image files too.
                    continue
                if member.stem in entries:
                    raise ValueError(f"Repeated image ID in archives: {member.stem}")
                entries[member.stem] = (archive, info)
        missing = sorted(set(expected) - set(entries))
        extra = sorted(set(entries) - set(expected))
        if missing or extra:
            (DOCS / "image_mapping_failure.json").write_text(json.dumps({"missing": missing, "extra": extra}, indent=2))
            raise ValueError(f"Metadata/archive mismatch: {len(missing)} missing, {len(extra)} extra")

        def inspect(image_id):
            archive, info = entries[image_id]
            data = archive.read(info)  # Checks member CRC after full read.
            with Image.open(BytesIO(data)) as probe:
                if probe.format != "JPEG":
                    raise ValueError(f"Unexpected image format: {image_id}: {probe.format}")
                probe.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()  # Force full pixel decoding; truncated files are not allowed.
                width, height = image.size
                mode = image.mode
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {image_id}")
            sha256 = hashlib.sha256(data).hexdigest()
            target = IMAGES / f"{image_id}.jpg"
            if target.exists():
                if file_hash(target, "sha256") != sha256:
                    raise ValueError(f"Existing extracted file differs from official archive: {target.name}")
            else:
                # Exclusive create protects any pre-existing file.
                with target.open("xb") as stream:
                    stream.write(data)
            meta = expected[image_id]
            return {
                "image_id": image_id, "lesion_id": meta["lesion_id"], "dx": meta["dx"],
                "relative_path": target.relative_to(ROOT).as_posix(),
                "archive": Path(archive.filename).name, "width": width, "height": height,
                "mode": mode, "bytes": len(data), "sha256": sha256,
            }

        inspected = {}
        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(inspect, key): key for key in expected}
            for completed, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    inspected[key] = future.result()
                except Exception as error:
                    failures.append({"image_id": key, "error": str(error)})
                if completed % 1000 == 0 or completed == len(expected):
                    print(f"CRC, decode and extraction: {completed}/{len(expected)}; failures={len(failures)}", flush=True)
        manifest = [inspected[row["image_id"]] for row in rows if row["image_id"] in inspected]
        by_hash = defaultdict(list)
        for item in manifest:
            by_hash[item["sha256"]].append(item)
        duplicate_groups = [[{key: item[key] for key in ["image_id", "lesion_id", "dx"]} for item in group]
                            for group in by_hash.values() if len(group) > 1]
        cross_lesion = [group for group in duplicate_groups if len({item["lesion_id"] for item in group}) > 1]
        extracted_ids = {path.stem for path in IMAGES.glob("*.jpg")}
        checks = {
            "official_archive_sizes_and_md5_match": True,
            "archive_image_ids_unique": True,
            "metadata_archive_image_ids_match": not missing and not extra,
            "all_images_pass_crc_and_full_decoding": not failures and len(manifest) == len(rows),
            "extracted_image_ids_match_metadata": extracted_ids == set(expected),
            "no_exact_byte_duplicates_across_lesions": not cross_lesion,
        }
        result = {
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "scope": "Official ZIPs, full CRC reads, full JPEG decoding, original-byte extraction and exact-file duplicate checks; no near-duplicate or patient-identity audit.",
            "n_metadata_records": len(rows), "n_archive_images": len(entries),
            "n_decoded_images": len(manifest), "n_extracted_jpg_files": len(extracted_ids),
            "archive_counts": dict(Counter(item["archive"] for item in manifest)),
            "dimension_counts": dict(Counter(f"{item['width']}x{item['height']}" for item in manifest)),
            "mode_counts": dict(Counter(item["mode"] for item in manifest)),
            "class_counts": dict(Counter(item["dx"] for item in manifest)),
            "n_unique_lesions": len({item["lesion_id"] for item in manifest}),
            "failures": failures, "missing_image_ids": missing, "extra_image_ids": extra,
            "exact_byte_duplicate_groups": duplicate_groups,
            "cross_lesion_exact_byte_duplicate_groups": cross_lesion, "checks": checks,
            "split_created": False, "image_preprocessing_applied": False,
        }
        (DOCS / "image_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (RAW / "images_provenance.json").write_text(json.dumps({
            "doi": "10.7910/DVN/DBW86T", "dataset_version": "4.0",
            "official_record": record_path.name, "archives": provenance,
            "metadata_md5": metadata_source["checksum"]["value"],
            "usage_terms_record": "harvard_dataset_record_v4.json -> data.latestVersion.termsOfUse",
            "original_jpeg_bytes_preserved": True,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if manifest:
            with (ROOT / "data" / "image_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)

        lines = ["# HAM10000 image validation results", "", f"Generated: {result['generated_at']}", "",
            "Both ZIP archives are checked against the saved official V4.0 file sizes and MD5 values. Each JPEG is read with ZIP CRC verification, format validation and full pixel decoding. Extracted files preserve the original bytes; manifest rows follow the metadata order.", "",
            "| Check | Result |", "| --- | ---: |",
            f"| Metadata image records | {len(rows):,} |", f"| Images in archives | {len(entries):,} |",
            f"| Fully decoded and extracted images | {len(manifest):,} |", f"| Missing / extra image IDs | {len(missing)} / {len(extra)} |",
            f"| Decode or extraction failures | {len(failures)} |", f"| Exact-byte duplicate groups | {len(duplicate_groups)} |",
            f"| Cross-lesion exact-byte duplicate groups | {len(cross_lesion)} |", "",
            "## Image dimensions and colour mode", "", "| Dimension or mode | Images |", "| --- | ---: |"]
        lines += [f"| {key} | {value:,} |" for key, value in {**result["dimension_counts"], **result["mode_counts"]}.items()]
        lines += ["", "## Validation checks", ""]
        lines += [f"- {'PASS' if ok else 'FAIL'}: `{name}`." for name, ok in checks.items()]
        lines += ["", "## Exact-byte duplicates", ""]
        for group in duplicate_groups:
            lines.append("- " + "; ".join(f"`{item['image_id']}` ({item['lesion_id']}, {item['dx']})" for item in group) + ".")
        if not duplicate_groups:
            lines.append("No exact-byte duplicate JPEGs were found.")
        elif not cross_lesion:
            lines += ["", "Both duplicate pairs are within a lesion. All source records are retained; lesion-level splitting keeps each pair in one subset. Any later deduplication must use an explicit manifest and be applied consistently across model comparisons."]
        lines += ["", "## Files and scope", "",
            "- `data/raw/images/`: original JPEGs named by image ID; not committed.",
            "- [Image manifest](../data/image_manifest.csv): identifiers, relative paths, archive, dimensions, mode, byte count and SHA-256. Paths are relative to `Final_Project_A/`.",
            "- [Image provenance](../data/raw/images_provenance.json): archive sources and checksums.",
            "- [Machine-readable audit](image_audit.json): full results and duplicate groups.", "",
            "This script does not detect visual near-duplicates, re-encoded copies or common patient identity. It does not resize images, fit preprocessing or train models. Its action flags describe this validation stage only.", "",
            "See [Image validation workflow](02_image_validation_workflow.md) and [Split results](05_data_split_results.md).", ""]
        (DOCS / "03_image_validation_results.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if not all(checks.values()):
            raise SystemExit("Image audit has failed checks; inspect results before splitting")
        make_examples(rows)
    finally:
        for archive in archives:
            archive.close()


if __name__ == "__main__":
    run()
