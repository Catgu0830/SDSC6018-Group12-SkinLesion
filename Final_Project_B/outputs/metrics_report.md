# B 阶段全部指标汇总（自动生成）

数据目录：`D:\学习经历\港城硕士\硕士学习\Semester A 2026-27\6018 Health Data Analysis\6018_Team_Project\Final_Project_B\outputs`
切分：`Final_Project_A` 冻结的 `v1_seed42`（病灶级 70/15/15）

## 1. 七分类主线：每个运行的全部指标

### 1.1 test

| run | accuracy | macro_f1 | weighted_f1 | auc_ovr | best_epoch |
|---|---|---|---|---|---|
| baseline_image | 0.827 | 0.6344 | 0.8218 | 0.9329 | 18.0 |
| baseline_image_s43 | 0.8216 | 0.634 | 0.8134 | 0.9378 | 11.0 |
| baseline_image_s44 | 0.8229 | 0.655 | 0.8223 | 0.9403 | 7.0 |
| baseline_image_ens3 | 0.8303 | 0.6608 | 0.8249 | 0.9494 | nan |
| fusion_all_metadata | 0.827 | 0.6735 | 0.8236 | 0.9526 | 12.0 |
| fusion_all_metadata_s43 | 0.831 | 0.6822 | 0.8246 | 0.9471 | 14.0 |
| fusion_all_metadata_s44 | 0.831 | 0.6827 | 0.8275 | 0.9586 | 12.0 |
| fusion_all_metadata_ens3 | 0.8417 | 0.7007 | 0.8365 | 0.9596 | nan |
| fusion_age_only | 0.8296 | 0.6567 | 0.8247 | 0.9487 | 18.0 |
| fusion_age_only_s43 | 0.8229 | 0.6514 | 0.816 | 0.9436 | 16.0 |
| fusion_age_only_s44 | 0.8209 | 0.673 | 0.8166 | 0.9443 | 9.0 |
| fusion_age_only_ens3 | 0.8303 | 0.6769 | 0.8234 | 0.9507 | nan |
| fusion_localization_only | 0.8256 | 0.6813 | 0.8228 | 0.9543 | 10.0 |
| fusion_localization_only_s43 | 0.8296 | 0.6751 | 0.8225 | 0.9514 | 14.0 |
| fusion_localization_only_s44 | 0.8203 | 0.6616 | 0.8175 | 0.9511 | 12.0 |
| fusion_localization_only_ens3 | 0.8323 | 0.6895 | 0.8281 | 0.9573 | nan |

### 1.2 val

| run | accuracy | macro_f1 | weighted_f1 | auc_ovr | best_epoch |
|---|---|---|---|---|---|
| baseline_image | 0.8419 | 0.7397 | 0.8369 | 0.9603 | 18.0 |
| baseline_image_s43 | 0.8372 | 0.7129 | 0.8272 | 0.9608 | 11.0 |
| baseline_image_s44 | 0.8372 | 0.6848 | 0.8333 | 0.9616 | 7.0 |
| baseline_image_ens3 | 0.8453 | 0.7199 | 0.8383 | 0.9685 | nan |
| fusion_all_metadata | 0.8467 | 0.732 | 0.8438 | 0.9662 | 12.0 |
| fusion_all_metadata_s43 | 0.844 | 0.7266 | 0.8363 | 0.9641 | 14.0 |
| fusion_all_metadata_s44 | 0.8474 | 0.7228 | 0.8419 | 0.9687 | 12.0 |
| fusion_all_metadata_ens3 | 0.8521 | 0.7363 | 0.8464 | 0.9714 | nan |
| fusion_age_only | 0.8358 | 0.6859 | 0.8287 | 0.9625 | 18.0 |
| fusion_age_only_s43 | 0.8379 | 0.707 | 0.8298 | 0.9601 | 16.0 |
| fusion_age_only_s44 | 0.8331 | 0.6987 | 0.8245 | 0.9639 | 9.0 |
| fusion_age_only_ens3 | 0.8351 | 0.6791 | 0.8259 | 0.9685 | nan |
| fusion_localization_only | 0.8467 | 0.7331 | 0.8426 | 0.9666 | 10.0 |
| fusion_localization_only_s43 | 0.8446 | 0.7013 | 0.8372 | 0.9612 | 14.0 |
| fusion_localization_only_s44 | 0.8446 | 0.7417 | 0.8415 | 0.9655 | 12.0 |
| fusion_localization_only_ens3 | 0.8528 | 0.7323 | 0.8476 | 0.9698 | nan |

## 2. 均值 ± 标准差（3 种子，不含集成）

### 2.1 按实验汇总

| Unnamed: 0 | test_accuracy | test_accuracy.1 | test_macro_f1 | test_macro_f1.1 | test_auc_ovr_macro | test_auc_ovr_macro.1 | val_accuracy | val_accuracy.1 | val_macro_f1 | val_macro_f1.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| nan | mean | std | mean | std | mean | std | mean | std | mean | std |
| experiment | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| baseline_image | 0.8238 | 0.0028 | 0.6411 | 0.012 | 0.937 | 0.0038 | 0.8388 | 0.0027 | 0.7125 | 0.0274 |
| fusion_age_only | 0.8245 | 0.0046 | 0.6604 | 0.0112 | 0.9455 | 0.0027 | 0.8356 | 0.0024 | 0.6972 | 0.0106 |
| fusion_all_metadata | 0.8296 | 0.0023 | 0.6795 | 0.0052 | 0.9528 | 0.0058 | 0.846 | 0.0018 | 0.7272 | 0.0046 |
| fusion_localization_only | 0.8252 | 0.0047 | 0.6726 | 0.0101 | 0.9523 | 0.0018 | 0.8453 | 0.0012 | 0.7254 | 0.0213 |

### 2.2 配对差值（同种子下 融合 − 基线）

| experiment | metric | n_seeds | delta_mean | delta_std |
|---|---|---|---|---|
| fusion_age_only | test_accuracy | 3 | 0.0007 | 0.0024 |
| fusion_age_only | test_macro_f1 | 3 | 0.0192 | 0.0026 |
| fusion_age_only | test_auc_ovr_macro | 3 | 0.0085 | 0.0064 |
| fusion_all_metadata | test_accuracy | 3 | 0.0058 | 0.0051 |
| fusion_all_metadata | test_macro_f1 | 3 | 0.0383 | 0.0103 |
| fusion_all_metadata | test_auc_ovr_macro | 3 | 0.0158 | 0.0057 |
| fusion_localization_only | test_accuracy | 3 | 0.0013 | 0.0058 |
| fusion_localization_only | test_macro_f1 | 3 | 0.0315 | 0.0218 |
| fusion_localization_only | test_auc_ovr_macro | 3 | 0.0153 | 0.0056 |

## 3. 三种子集成（test）

| experiment | n_members | ens_accuracy | ens_macro_f1 | ens_auc | single_acc_mean | single_acc_best | single_f1_mean | single_f1_best |
|---|---|---|---|---|---|---|---|---|
| baseline_image | 3 | 0.8303 | 0.6608 | 0.9494 | 0.8238 | 0.827 | 0.6411 | 0.655 |
| fusion_all_metadata | 3 | 0.8417 | 0.7007 | 0.9596 | 0.8296 | 0.831 | 0.6795 | 0.6827 |
| fusion_age_only | 3 | 0.8303 | 0.6769 | 0.9507 | 0.8245 | 0.8296 | 0.6604 | 0.673 |
| fusion_localization_only | 3 | 0.8323 | 0.6895 | 0.9573 | 0.8252 | 0.8296 | 0.6726 | 0.6813 |

## 4. 逐类指标（集成模型，test）

### baseline_image

| class | precision | recall | f1 | support | auc |
|---|---|---|---|---|---|
| akiec | 0.561 | 0.5 | 0.529 | 46 | 0.972 |
| bcc | 0.629 | 0.8 | 0.704 | 70 | 0.983 |
| bkl | 0.719 | 0.692 | 0.705 | 159 | 0.947 |
| df | 0.625 | 0.333 | 0.435 | 15 | 0.915 |
| mel | 0.638 | 0.491 | 0.555 | 169 | 0.91 |
| nv | 0.902 | 0.933 | 0.917 | 1013 | 0.947 |
| vasc | 0.727 | 0.842 | 0.78 | 19 | 0.972 |

### fusion_all_metadata

| class | precision | recall | f1 | support | auc |
|---|---|---|---|---|---|
| akiec | 0.615 | 0.522 | 0.565 | 46 | 0.979 |
| bcc | 0.683 | 0.8 | 0.737 | 70 | 0.985 |
| bkl | 0.712 | 0.686 | 0.699 | 159 | 0.946 |
| df | 0.667 | 0.4 | 0.5 | 15 | 0.966 |
| mel | 0.672 | 0.533 | 0.594 | 169 | 0.909 |
| nv | 0.903 | 0.942 | 0.922 | 1013 | 0.951 |
| vasc | 0.941 | 0.842 | 0.889 | 19 | 0.983 |

### fusion_age_only

| class | precision | recall | f1 | support | auc |
|---|---|---|---|---|---|
| akiec | 0.595 | 0.543 | 0.568 | 46 | 0.969 |
| bcc | 0.628 | 0.843 | 0.72 | 70 | 0.987 |
| bkl | 0.721 | 0.635 | 0.676 | 159 | 0.945 |
| df | 0.714 | 0.333 | 0.455 | 15 | 0.929 |
| mel | 0.638 | 0.479 | 0.547 | 169 | 0.903 |
| nv | 0.894 | 0.94 | 0.916 | 1013 | 0.949 |
| vasc | 0.938 | 0.789 | 0.857 | 19 | 0.974 |

### fusion_localization_only

| class | precision | recall | f1 | support | auc |
|---|---|---|---|---|---|
| akiec | 0.676 | 0.5 | 0.575 | 46 | 0.978 |
| bcc | 0.704 | 0.814 | 0.755 | 70 | 0.983 |
| bkl | 0.711 | 0.635 | 0.671 | 159 | 0.947 |
| df | 0.583 | 0.467 | 0.519 | 15 | 0.965 |
| mel | 0.606 | 0.556 | 0.58 | 169 | 0.899 |
| nv | 0.9 | 0.932 | 0.916 | 1013 | 0.945 |
| vasc | 0.833 | 0.789 | 0.811 | 19 | 0.985 |

## 5. 混淆矩阵（集成模型，test）

### baseline_image（行=真实，列=预测）

| true\pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 23 | 11 | 6 | 0 | 2 | 4 | 0 |
| bcc | 3 | 56 | 0 | 2 | 1 | 7 | 1 |
| bkl | 6 | 9 | 110 | 0 | 10 | 24 | 0 |
| df | 3 | 1 | 2 | 5 | 0 | 4 | 0 |
| mel | 4 | 3 | 14 | 0 | 83 | 62 | 3 |
| nv | 2 | 8 | 21 | 1 | 34 | 945 | 2 |
| vasc | 0 | 1 | 0 | 0 | 0 | 2 | 16 |

### fusion_all_metadata（行=真实，列=预测）

| true\pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 24 | 10 | 8 | 0 | 0 | 4 | 0 |
| bcc | 4 | 56 | 1 | 1 | 1 | 7 | 0 |
| bkl | 6 | 7 | 109 | 0 | 10 | 27 | 0 |
| df | 3 | 1 | 3 | 6 | 0 | 2 | 0 |
| mel | 1 | 5 | 12 | 0 | 90 | 61 | 0 |
| nv | 1 | 2 | 20 | 2 | 33 | 954 | 1 |
| vasc | 0 | 1 | 0 | 0 | 0 | 2 | 16 |

### fusion_age_only（行=真实，列=预测）

| true\pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 25 | 10 | 5 | 0 | 1 | 5 | 0 |
| bcc | 3 | 59 | 1 | 0 | 2 | 5 | 0 |
| bkl | 5 | 8 | 101 | 0 | 12 | 33 | 0 |
| df | 3 | 2 | 3 | 5 | 0 | 2 | 0 |
| mel | 4 | 7 | 12 | 0 | 81 | 65 | 0 |
| nv | 2 | 7 | 18 | 2 | 31 | 952 | 1 |
| vasc | 0 | 1 | 0 | 0 | 0 | 3 | 15 |

### fusion_localization_only（行=真实，列=预测）

| true\pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 23 | 8 | 8 | 0 | 3 | 4 | 0 |
| bcc | 2 | 57 | 1 | 1 | 1 | 7 | 1 |
| bkl | 5 | 7 | 101 | 0 | 17 | 29 | 0 |
| df | 2 | 1 | 3 | 7 | 0 | 2 | 0 |
| mel | 1 | 2 | 10 | 0 | 94 | 61 | 1 |
| nv | 1 | 4 | 19 | 4 | 40 | 944 | 1 |
| vasc | 0 | 2 | 0 | 0 | 0 | 2 | 15 |

## 6. 类别权重敏感性（单种子 42）

| experiment | loss | test_acc | test_macro_f1 | test_auc | best_epoch |
|---|---|---|---|---|---|
| baseline_image | pure CE | 0.827 | 0.6344 | 0.9329 | 18 |
| baseline_image | sqrt_inverse | 0.8169 | 0.6749 | 0.9446 | 11 |
| fusion_all_metadata | pure CE | 0.827 | 0.6735 | 0.9526 | 12 |
| fusion_all_metadata | sqrt_inverse | 0.8149 | 0.6621 | 0.9567 | 6 |
| fusion_age_only | pure CE | 0.8296 | 0.6567 | 0.9487 | 18 |
| fusion_age_only | sqrt_inverse | 0.8223 | 0.6611 | 0.939 | 19 |
| fusion_localization_only | pure CE | 0.8256 | 0.6813 | 0.9543 | 10 |
| fusion_localization_only | sqrt_inverse | 0.8122 | 0.6681 | 0.951 | 6 |

### 6.1 逐类 F1 对比（加权 − 纯 CE）

#### baseline_image

| class | f1_CE | f1_weighted | d_f1 | recall_CE | recall_weighted | d_recall |
|---|---|---|---|---|---|---|
| akiec | 0.45 | 0.559 | 0.109 | 0.391 | 0.565 | 0.174 |
| bcc | 0.696 | 0.759 | 0.062 | 0.786 | 0.786 | 0.0 |
| bkl | 0.714 | 0.671 | -0.043 | 0.698 | 0.692 | -0.006 |
| df | 0.333 | 0.5 | 0.167 | 0.267 | 0.467 | 0.2 |
| mel | 0.55 | 0.555 | 0.005 | 0.509 | 0.568 | 0.059 |
| nv | 0.918 | 0.908 | -0.01 | 0.931 | 0.895 | -0.036 |
| vasc | 0.78 | 0.773 | -0.008 | 0.842 | 0.895 | 0.053 |

#### fusion_all_metadata

| class | f1_CE | f1_weighted | d_f1 | recall_CE | recall_weighted | d_recall |
|---|---|---|---|---|---|---|
| akiec | 0.558 | 0.615 | 0.057 | 0.522 | 0.696 | 0.174 |
| bcc | 0.726 | 0.718 | -0.008 | 0.757 | 0.8 | 0.043 |
| bkl | 0.698 | 0.648 | -0.049 | 0.711 | 0.66 | -0.05 |
| df | 0.444 | 0.452 | 0.007 | 0.4 | 0.467 | 0.067 |
| mel | 0.54 | 0.551 | 0.011 | 0.503 | 0.497 | -0.006 |
| nv | 0.915 | 0.906 | -0.009 | 0.925 | 0.903 | -0.022 |
| vasc | 0.833 | 0.744 | -0.089 | 0.789 | 0.842 | 0.053 |

## 7. 二分类（恶性 / 良性）

| source | val_acc | thr | test_acc | test_bal_acc | test_precision | test_recall | test_f1 | test_auc | test_acc@0.5 |
|---|---|---|---|---|---|---|---|---|---|
| dedicated binary, plain aug | 0.9003 | 0.55 | 0.882 | 0.7944 | 0.7072 | 0.6526 | 0.6788 | 0.9171 | 0.8813 |
| dedicated binary, strong aug | 0.8874 | 0.59 | 0.8833 | 0.7832 | 0.7284 | 0.6211 | 0.6705 | 0.9193 | 0.886 |
| 7-class baseline -> sum | 0.9077 | 0.58 | 0.8893 | 0.7856 | 0.7586 | 0.6175 | 0.6809 | 0.9189 | 0.89 |
| 7-class fusion(all) -> sum | 0.9138 | 0.55 | 0.8887 | 0.7945 | 0.7409 | 0.6421 | 0.688 | 0.9258 | 0.8873 |
| 7-class fusion(loc) -> sum | 0.9071 | 0.595 | 0.8893 | 0.7882 | 0.7542 | 0.6246 | 0.6833 | 0.9257 | 0.8887 |

## 8. 泄漏对照（不是我们的结果）

| control | task | test_acc | test_macro_f1 | test_auc |
|---|---|---|---|---|
| image-level split | 7-class | 0.8616 | 0.7262 | 0.9714 |
| image-level split | binary | 0.8929 | 0.8278 | 0.9298 |

### 8.1 暴露 vs 干净（图像级切分）

| subset | n | accuracy | macro_f1 | auc |
|---|---|---|---|---|
| all | 1503 | 0.8616 | 0.7262 | 0.9714 |
| exposed | 518 | 0.8205 | 0.7341 | 0.9711 |
| clean | 985 | 0.8832 | 0.707 | 0.9692 |

