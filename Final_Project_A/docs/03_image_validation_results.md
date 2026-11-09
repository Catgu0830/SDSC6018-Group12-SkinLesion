# HAM10000 image validation results

Generated: 2026-11-09T18:50:48.746645+08:00

Both ZIP archives are checked against the saved official V4.0 file sizes and MD5 values. Each JPEG is read with ZIP CRC verification, format validation and full pixel decoding. Extracted files preserve the original bytes; manifest rows follow the metadata order.

| Check | Result |
| --- | ---: |
| Metadata image records | 10,015 |
| Images in archives | 10,015 |
| Fully decoded and extracted images | 10,015 |
| Missing / extra image IDs | 0 / 0 |
| Decode or extraction failures | 0 |
| Exact-byte duplicate groups | 2 |
| Cross-lesion exact-byte duplicate groups | 0 |

## Image dimensions and colour mode

| Dimension or mode | Images |
| --- | ---: |
| 600x450 | 10,015 |
| RGB | 10,015 |

## Validation checks

- PASS: `official_archive_sizes_and_md5_match`.
- PASS: `archive_image_ids_unique`.
- PASS: `metadata_archive_image_ids_match`.
- PASS: `all_images_pass_crc_and_full_decoding`.
- PASS: `extracted_image_ids_match_metadata`.
- PASS: `no_exact_byte_duplicates_across_lesions`.

## Exact-byte duplicates

- `ISIC_0029861` (HAM_0002300, nv); `ISIC_0024366` (HAM_0002300, nv).
- `ISIC_0029938` (HAM_0001895, nv); `ISIC_0024777` (HAM_0001895, nv).

Both duplicate pairs are within a lesion. All source records are retained; lesion-level splitting keeps each pair in one subset. Any later deduplication must use an explicit manifest and be applied consistently across model comparisons.

## Files and scope

- `data/raw/images/`: original JPEGs named by image ID; not committed.
- [Image manifest](../data/image_manifest.csv): identifiers, relative paths, archive, dimensions, mode, byte count and SHA-256. Paths are relative to `Final_Project_A/`.
- [Image provenance](../data/raw/images_provenance.json): archive sources and checksums.
- [Machine-readable audit](image_audit.json): full results and duplicate groups.

This script does not detect visual near-duplicates, re-encoded copies or common patient identity. It does not resize images, fit preprocessing or train models. Its action flags describe this validation stage only.

See [Image validation workflow](02_image_validation_workflow.md) and [Split results](05_data_split_results.md).
