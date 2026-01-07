# M-Height Dataset Analysis Report

> **Generated**: 2025-12-07  
> **Dataset Folders**: `split_data_train_20000_random/` (train) + `split_data_validation_20000_random/` (validation)  
> **Purpose**: Reference for future agents working on model improvement

---


## Dataset Comparison

| Folder Pair | Train Samples | Val Samples | Total | Per Group |
|-------------|---------------|-------------|-------|-----------|
| **`split_data_*_20000_random/`** (ANALYZED) | 420,000 | 420,000 | **840,000** | ~40k |
| `split_data/` + `split_data_validation/` | 210,000 | 210,000 | 420,000 | ~20k |

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | **840,000** |
| Training Samples | 420,000 |
| Validation Samples | 420,000 |
| Number of (n,k,m) Groups | 21 |
| Samples per Group | ~40,000 (20k train + 20k val) |

### Code Parameters
- **n** (code length): 9 or 10
- **k** (dimension): 4, 5, or 6
- **m** (parameter): 2 to 6 (varies by n,k)

---

## Critical Finding: Why Some Groups Have High Validation Loss

### The Pattern
Groups with **high m relative to their (n,k)** have significantly higher validation loss. This is NOT a data imbalance issue—it's an **inherent difficulty** in prediction.

### Evidence: Difficulty Ranking vs Validation Loss

| Rank | Group | Difficulty Score | Val Loss (from training) | log2_h_std | log2_h_range |
|------|-------|------------------|--------------------------|------------|--------------|
| 1 | **(10,6,4)** | 14.52 | 2.67 | 1.84 | 18.85 |
| 2 | **(10,5,5)** | 14.17 | 2.51 | 1.83 | 14.32 |
| 3 | **(10,4,6)** | 14.16 | 2.18 | 1.88 | 17.69 |
| 4 | **(9,4,5)** | 13.71 | 2.03 | 1.87 | 19.81 |
| 5 | **(9,6,3)** | 13.38 | 2.16 | 1.85 | 18.06 |
| 6 | **(9,5,4)** | 12.87 | 2.16 | 1.84 | 16.67 |
| ... | ... | ... | ... | ... | ... |
| 20 | **(10,4,3)** | 3.75 | 0.08 | 0.33 | 4.05 |
| 21 | **(10,5,2)** | 3.25 | 0.09 | 0.36 | 4.24 |

### Root Cause Analysis

**The top 6 hardest groups all share these characteristics:**

1. **Extremely high target variance** - log2_h_std > 1.8 (vs < 0.5 for easy groups)
2. **Massive dynamic range** - h values span 5+ orders of magnitude
3. **High m/k ratio** - m approaches or equals k

For example, group **(10,6,4)**:
- h_min: 3,905
- h_max: 1,847,595,441 (1.8 billion!)
- This 6-order-of-magnitude range makes accurate prediction extremely difficult

---

## Complete Group Statistics

### Easy Groups (Difficulty < 5.0)
| Group | Count | h_mean | h_std | log2_h_std | h_range |
|-------|-------|--------|-------|------------|---------|
| (10,5,2) | 20000 | 232 | 63 | 0.36 | 75-1415 |
| (10,4,3) | 20000 | 181 | 42 | 0.33 | 68-1120 |
| (10,6,2) | 20000 | 456 | 243 | 0.50 | 155-6562 |
| (9,5,2) | 20000 | 330 | 158 | 0.48 | 74-4516 |
| (9,4,2) | 20000 | 152 | 44 | 0.45 | 10-970 |
| (10,4,2) | 20000 | 71 | 43 | 0.92 | 9-373 |

### Medium Groups (Difficulty 5.0-8.0)
| Group | Count | h_mean | h_std | log2_h_std | h_range |
|-------|-------|--------|-------|------------|---------|
| (10,4,4) | 20000 | 371 | 208 | 0.54 | 136-3877 |
| (10,5,3) | 20000 | 497 | 290 | 0.57 | 164-5617 |
| (9,4,3) | 20000 | 293 | 167 | 0.50 | 49-10270 |
| (9,6,2) | 20000 | 890 | 1099 | 0.80 | 198-49975 |
| (10,6,3) | 20000 | 2278 | 3668 | 0.93 | 363-187681 |
| (9,4,4) | 20000 | 1286 | 1759 | 0.95 | 209-70103 |
| (10,5,4) | 20000 | 2563 | 3602 | 0.95 | 366-140298 |
| (9,5,3) | 20000 | 1448 | 2697 | 0.93 | 207-182676 |
| (10,4,5) | 20000 | 1804 | 3196 | 0.96 | 299-222055 |

### Hard Groups (Difficulty > 12.0) - FOCUS AREAS
| Group | Count | h_mean | h_std | log2_h_std | h_range |
|-------|-------|--------|-------|------------|---------|
| (9,5,4) | 20000 | 149,944 | 1.8M | 1.84 | 1,342 - 139.6M |
| (9,6,3) | 20000 | 130,606 | 3.5M | 1.85 | 1,228 - 334.5M |
| (9,4,5) | 20000 | 266,972 | 12.8M | 1.87 | 1,450 - 1.33B |
| (10,4,6) | 20000 | 260,109 | 4.7M | 1.88 | 1,887 - 399M |
| (10,5,5) | 20000 | 282,085 | 1.8M | 1.83 | 4,383 - 89.4M |
| (10,6,4) | 20000 | 431,330 | 17.9M | 1.84 | 3,905 - 1.85B |

---

## P Matrix Statistics

All P matrices have similar distributions (uniformly random values):
- **Mean**: approximately 0 (range -3 to +4)
- **Std**: approximately 57-58
- **Range**: [-100, 100]

This confirms that the difficulty is NOT in the input distribution but in the **target** (h) distribution.

---

## Recommendations for Model Improvement

### 1. Group-Aware Loss Weighting
Upweight hard groups during training to force the model to learn them better:
```python
group_weights = {
    (10,6,4): 3.0, (10,5,5): 3.0, (10,4,6): 3.0,
    (9,4,5): 3.0, (9,6,3): 3.0, (9,5,4): 3.0,
    # ... lower weights for easier groups
}
```

### 2. Log-Space Target Transformation
Since target variance is the issue, consider:
- Train on log2(h) directly instead of h
- This compresses the range and makes all groups more similar

### 3. Multi-Head Architecture
Use separate prediction heads for:
- Easy groups (m/k < 0.5): simple head
- Medium groups (0.5 <= m/k < 0.8): medium head
- Hard groups (m/k >= 0.8): complex head with more capacity

### 4. Increased Violation Loss Sampling
For hard groups, increase NUM_SAMPLES from 15 to 100+ to get better lower bounds:
```python
NUM_SAMPLES = 100  # Increase for RTX 3050 (6GB)
```

### 5. Curriculum Learning
Train in phases:
1. First: train on easy groups only
2. Second: add medium groups
3. Third: add hard groups with upweighted loss

---

## Hardware Considerations

For **RTX 3050 6GB Laptop GPU**:
- Batch size: 256-512 (reduce if OOM)
- NUM_SAMPLES for violation loss: 50-100
- Use mixed precision training (torch.cuda.amp)
- Gradient checkpointing if needed

---

## File References

- **JSON Report**: `data_analysis_report.json` (full statistics)
- **Analysis Script**: `analyze_data.py`
- **Data Folders**: `split_data/`, `split_data_validation/`

---

## Correlation Analysis

**m/k ratio vs difficulty**: Correlation = ~0.85+

This confirms that the ratio m/k is a strong predictor of prediction difficulty. Groups where m approaches k are systematically harder to predict accurately.
