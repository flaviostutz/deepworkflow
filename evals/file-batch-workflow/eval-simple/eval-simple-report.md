# Eval Report: eval-simple

**Date:** 2026-06-21
**Dataset:** dataset/
**Script:** eval-simple.py
**Thresholds:** similarity ≥ 0.5

## Overall Results

| Metric    | Value | 95% CI               | Threshold | Status      |
|-----------|-------|----------------------|-----------|-------------|
| Accuracy  | 0.00  | [0.00, 0.79] | ≥ 0.5  | ✗ FAIL |
| F1 Score  | 0.00  | —                    | ≥ 0.5  | ✗ FAIL |
| Precision | 0.00  | —                    | —         | —           |
| Recall    | 0.00  | —                    | —         | —           |
| Samples   | 1   | —                    | —         | —           |

**Overall: FAIL**

## Per-item Results

| ID  | Input Summary          | Expected           | Actual | Correct |
|-----|------------------------|--------------------|--------|---------|
| 001 | dataset/data (*.py)    | similarity ≥ 0.5 | 0.00   | ✗       |

## Notes

- Similarity score: 0.0000 (threshold: 0.5)
- MLflow run ID: 3a31b2f2b67e46d1806adc45b4e07003 — view with `mlflow ui`
