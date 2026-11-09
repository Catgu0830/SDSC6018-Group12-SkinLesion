# EDA results

Six descriptive figure groups and 13 statistical tables are generated from the frozen data. The source files remain unchanged. The analysis does not include model training or fairness results.

[EDA report](../output/eda/v1_seed42/eda_report.md)

Training data: 7,050 images and 5,229 distinct lesions. nv accounts for 67.01% of images; df has 86 images from 51 lesions. At least one missing/unknown metadata field affects 183 images. Field counts overlap.

See [numeric validation](eda_validation.json) and [visual review](eda_visual_review.json) for the recorded checks. PNG/statistics/report reproduction is checked byte-for-byte; SVG structure and canvas dimensions are checked without requiring identical serialization.
