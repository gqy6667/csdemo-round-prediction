# M10 Probability Calibration Report

Calibration method selection used grouped out-of-fold validation predictions only.
Selected method: **uncalibrated**.

| Split | Method | Log Loss | Brier | ECE10 | AUC |
|---|---|---:|---:|---:|---:|
| validation_oof | uncalibrated | 0.596038 | 0.207575 | 0.011285 | 0.718596 |
| validation_oof | sigmoid | 0.596250 | 0.207634 | 0.008777 | 0.718221 |
| validation_oof | isotonic | 0.605117 | 0.208231 | 0.012645 | 0.713903 |
| test | uncalibrated | 0.591733 | 0.205294 | 0.023198 | 0.727122 |
| test | sigmoid | 0.591890 | 0.205347 | 0.022049 | 0.727122 |
| test | isotonic | 0.594422 | 0.205676 | 0.010360 | 0.726040 |

## Acceptance

Selected test ECE <= 0.04: True.
Selected test ECE <= 0.03: True.
Test Log Loss change vs raw: +0.000000.
Test Brier change vs raw: +0.000000.
Test ECE change vs raw: +0.000000.
The test set was not used to select the calibration method.
