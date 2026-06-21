# Eval Report: eval-simple

**Date:** 2026-06-22
**Dataset:** dataset/
**Script:** eval-simple.py
**Thresholds:** similarity ≥ 0.5

## Overall Results

| Metric    | Value | 95% CI               | Threshold | Status      |
|-----------|-------|----------------------|-----------|-------------|
| Accuracy  | 1.00  | [0.21, 1.00] | ≥ 0.5  | ✓ PASS |
| F1 Score  | 1.00  | —                    | ≥ 0.5  | ✓ PASS |
| Precision | 1.00  | —                    | —         | —           |
| Recall    | 1.00  | —                    | —         | —           |
| Samples   | 1   | —                    | —         | —           |

**Overall: PASS**

## Per-item Results

| ID  | Input Summary          | Expected           | Actual | Correct |
|-----|------------------------|--------------------|--------|---------|
| 001 | dataset/data (*.py)    | similarity ≥ 0.5 | 0.50   | ✓       |

## Notes

- Similarity score: 0.5000 (threshold: 0.5)
- MLflow run ID: 08b9e1defcd34d5b9960329c15ceae1d — view with `mlflow ui`
