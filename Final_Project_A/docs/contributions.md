# A-stage contribution record

This record describes the data-preparation artifacts and verification. It supports a later individual reflection; it does not establish that the student independently authored or has mastered every component.

## Data preparation, 7 November 2026

| Work area | Implementation and artifacts | Recorded verification |
| --- | --- | --- |
| Metadata | Saved the HAM10000 V4.0 record and CSV/TAB exports; implemented field, missingness and consistency checks | Official CSV MD5 matched; exports had identical parsed records; 10,015 images and 7,470 lesion IDs |
| Images | Verified two archives, read CRCs, fully decoded JPEGs, retained bytes and wrote the image manifest | All images decoded; 600 x 450 RGB; two same-lesion exact-byte duplicate pairs retained |
| Splitting | Implemented deterministic diagnosis-stratified lesion allocation, labels and frozen lists | Complete coverage; empty image/lesion/hash intersections; reproduction and overwrite protection |
| Preprocessing | Fitted age parameters/vocabularies on training records; wrote tables, schema and lazy transforms | Independent numeric checks; 19-feature schema; six-file reproduction; all image inputs checked |
| EDA | Generated six figure groups, 13 tables, report and provenance | Independent counts/denominators; statistics/report/PNG reproduction; SVG and visual checks |

Methods, limitations and machine-readable evidence are documented in the method/result pages. Split ratio, seed, label mapping and image recipe await training-member alignment. No models were trained; no prediction-based Benefit/Harm/Neutral or fairness results were produced.

## Documentation revision, 9 November 2026

The handover was reorganized into English methods, results and usage documentation. A Chinese guide supports collaboration. Report templates were revised, linked provenance was regenerated and outputs were revalidated. Dataset membership and feature values were preserved.

## Assistance and individual responsibility

AI assistance was used to draft code and documentation, execute local checks and interpret outputs. Automated checks and assistant visual review are recorded as such; they are not student approval or independent external review. The student's reflection should state the work they actually performed, reviewed, understood and discussed with teammates. Actual coordination and presentation work should be added when it occurs.

The checks establish specified technical properties, not patient independence, clinical validity or optimal model settings. The reflection should follow applicable course tool-use disclosure requirements and accurately describe assisted work.
