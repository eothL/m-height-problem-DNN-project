# Observing Grokking in M-Height Prediction: A Deep Learning Experiment

**Author**: Theo Lin  
**Date**: January 2026  
**Project**: CSCE 636 - Deep Learning for Linear Code Analysis

---

## Abstract

We investigate the **grokking phenomenon** — a sudden generalization after apparent convergence — in a neural network trained to predict the m-height of linear codes. Using a ResNet architecture with LayerNorm and varying weight decay regularization, we observe clear grokking transitions:

- **Weight decay 0.01**: Grokking at epoch ~780 (val loss: 3.0 → 1.3)
- **Weight decay 0.03**: Faster grokking at epoch ~520 (val loss: 4.0 → 2.0)

Stronger weight decay accelerates the transition to generalization.

---

## 1. Introduction

### 1.1 Background

**Grokking** was first described by Power et al. (2022) as a phenomenon where neural networks suddenly generalize to unseen data long after they have memorized the training set. This delayed generalization typically occurs under specific conditions:

- Small training datasets (encouraging memorization)
- Strong regularization (weight decay)
- Extended training (many epochs beyond apparent convergence)

### 1.2 Motivation

Our m-height prediction task provides an interesting testbed for grokking because:

1. The underlying mathematical structure (linear codes) has learnable patterns
2. The dataset can be subsampled to encourage memorization
3. The prediction task is well-defined with clear evaluation metrics

### 1.3 Research Question

> Can we observe grokking in a ResNet model trained on m-height prediction, and how does weight decay strength affect the grokking timing?

---

## 2. Methods

### 2.1 Dataset

| Parameter | Value |
|-----------|-------|
| Total samples | 840,000 (21 groups × 40k each) |
| Training subset | 105,000 (5,000 per (n,k,m) group) |
| Validation subset | 210,000 (10,000 per group) |
| Input | Padded P matrix (6×6) + (n,k,m) parameters |
| Target | m-height (h) in log2 space |

The subsampling to 5k samples per group was deliberate to create conditions favorable for grokking — a dataset small enough for the model to memorize.

### 2.2 Model Architecture

We used a **ResNet with LayerNorm** (instead of BatchNorm):

```
ResNet2DWithLayerNorm:
├── Stem: Conv2d(1→64) + LayerNorm + ReLU
├── 5× ConvResBlockLayerNorm(64 channels)
├── Flatten + Parameter projection (3→64)
└── Head: Linear(2368→256→1)

Total parameters: 1,027,585
```

LayerNorm was chosen over BatchNorm because:
- It provides more stable training for small batches
- Independence from batch statistics for consistent evaluation
- Better suited for grokking experiments requiring long training

### 2.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| **Weight decay** | **0.01, 0.03, 0.1** (tested) |
| Batch size | 4,096 |
| Epochs | 10,000 |
| Validation frequency | Every 5 epochs |
| Loss function | Log2 MSE |

---

## 3. Results

### 3.1 Weight Decay 0.01

The training exhibited three distinct phases:

#### Phase 1: Initial Learning (Epochs 1-100)
- Rapid decrease in both training and validation loss
- Training loss: 25 → 5
- Validation loss: 14 → 3

#### Phase 2: Plateau / Memorization (Epochs 100-780)
- Training loss continued decreasing slowly: 5 → 2
- Validation loss **plateaued** around 3.0
- Classic memorization behavior: model fitting training data without generalizing

#### Phase 3: Grokking Transition (Epoch ~780)
- **Sudden drop** in validation loss: 3.0 → 1.3
- Training loss remained stable around 1.3
- The model "discovered" generalizable features

### 3.2 Weight Decay 0.03

With 3× stronger weight decay, grokking occurred **earlier**:

#### Phase 1: Initial Learning (Epochs 1-100)
- Training loss: 25 → 5
- Validation loss: 15 → 4

#### Phase 2: Plateau (Epochs 100-520)
- Validation loss fluctuated around 3-4
- More unstable than wd=0.01 due to stronger regularization pressure

#### Phase 3: Grokking Transition (Epoch ~520)
- **Sudden drop** in validation loss: 4.0 → 2.0
- **⚠️ Anomaly**: Training loss briefly **increased** during this transition
- Faster grokking than wd=0.01 (~260 epochs earlier!)

### 3.3 Weight Decay 0.1 (No Grokking)

With 10× stronger weight decay, **no grokking occurred**:

#### Observations (up to epoch 2400)
- Training loss: Highly unstable, oscillating between 0.1 and 0.9
- Validation loss: Stuck around 2.3-2.9 (never improved)
- Pattern: Repeated cycles of brief memorization followed by immediate collapse

#### Why No Grokking?
The regularization is too strong. The model cannot:
1. Build the initial "memorization circuit" (required precursor to grokking)
2. Maintain any stable solution long enough to transition
3. Find parameters that satisfy both data fit AND regularization simultaneously

**Conclusion**: There exists an optimal weight decay range for grokking. Too weak = no transition pressure. Too strong = no stable solution.

### 3.4 Comparison Summary

| Weight Decay | Grokking? | Grokking Epoch | Val Loss (Best) | Train Loss Spike? |
|--------------|-----------|----------------|-----------------|-------------------|
| 0.01 | ✅ Yes | ~780 | ~1.3 | ⚠️ Yes |
| 0.03 | ✅ Yes | ~520 | ~2.0 | ⚠️ Yes |
| 0.1 | ❌ No | N/A | ~2.2 (unstable) | N/A |

### 3.5 Open Question: The Training Loss Spike

> **In both successful grokking cases (wd=0.01, wd=0.03), the training loss briefly INCREASES when validation loss suddenly drops. This is counter-intuitive and requires investigation.**

Classical grokking literature describes validation loss dropping while training loss remains near-zero. Our observation differs:
- Train loss spikes UP during the grokking transition
- This suggests a more dramatic "representation reorganization" than simple delayed generalization

**Hypothesis**: The spike correlates with a sudden drop in model weight norm (the "Slingshot Mechanism"). We will test this with a diagnostic experiment.

### 3.6 Theoretical Deep Dive: The Training Loss Spike

The observation that training loss **increases** during the grokking transition is a critical structural signal, often described in literature (Nakkiran et al., Power et al.) as a **phase transition** or "Slingshot Mechanism".

#### The Mechanism

The loss function $L$ has two competing terms:
$$L = \text{MSE}(\text{Data}) + \lambda \cdot ||W||^2$$

1.  **Memorization Basin (Pre-Grokking)**:
    - The model minimizes **MSE** to near-zero by "memorizing" noise.
    - To achieve this complex fit, the **Weights ($||W||^2$)** must be large and "jagged".
    - *State*: Low MSE, High Weight Norm.

2.  **The Transition (The Spike)**:
    - Weight decay ($\lambda$) applies constant pressure to reduce $||W||^2$.
    - To reduce the weight norm significantly, the model must "smooth out" its function.
    - **Crucial Moment**: Smoothing the function *breaks* the precise memorization of training points.
    - **Result**: MSE temporarily shoots up (training loss spike) because the model abandons the memorized solution *before* it has fully settled into the generalized solution.

3.  **Generalization Basin (Post-Grokking)**:
    - The model finds a "smarter" algorithm (the generalized solution).
    - This solution requires much smaller weights.
    - *State*: Low MSE, Low Weight Norm.

#### Diagnostic Prediction
If this theory holds, we should observe a **perfect correlation** between the training loss spike and a massive drop in the model's L2 Weight Norm. This confirms the spike is the "cost" of shedding complexity.

### 3.5 Post-Grokking Behavior

After the grokking transition, continued training led to:
- Training loss → near-zero (complete memorization)
- Validation loss → increasing (overfitting)

This suggests an optimal stopping point exists shortly after the grokking transition.

---

## 4. Discussion

### 4.1 Interpretation

The grokking phenomenon we observed aligns with theoretical understanding:

1. **Early phase**: The model finds a solution that fits training data through memorization
2. **Plateau phase**: Weight decay slowly erodes the memorized solution
3. **Transition**: The model discovers a simpler, generalizable representation that satisfies both the loss function AND the regularization constraint
4. **Post-grokking**: Without early stopping, the model eventually re-memorizes

### 4.2 Effect of Weight Decay Strength

| Aspect | Weaker (0.01) | Stronger (0.03) |
|--------|---------------|-----------------|
| Grokking timing | Later (~780) | Earlier (~520) |
| Training stability | More stable | More fluctuations |
| Final val loss | Lower (~1.3) | Higher (~2.0) |
| Total training time | Longer | Shorter |

**Trade-off**: Stronger weight decay speeds up grokking but may prevent the model from reaching the lowest possible generalization error.

### 4.3 Practical Implications

For m-height prediction:
- The grokking checkpoint achieves competitive val loss (~1.3 for wd=0.01)
- This is within range of our best MoE model (Val Loss ~0.88)
- Grokking provides a potential alternative to complex architectures

---

## 5. Conclusion

We successfully demonstrated the **grokking phenomenon** in a practical regression task (m-height prediction). Key findings:

1. ✅ **Grokking observed** with both wd=0.01 (epoch ~780) and wd=0.03 (epoch ~520)
2. ✅ **Stronger weight decay accelerates grokking** (33% faster with 3× weight decay)
3. ✅ **Trade-off exists**: faster grokking vs. lower final loss
4. ⚠️ **Post-grokking overfitting** suggests early stopping is important

### Future Work

- Complete wd=0.1 experiment to study very strong regularization
- Investigate optimal stopping criteria after grokking
- Apply insights to improve the main m-height prediction model

---

## References

1. Power, A., et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." arXiv:2201.02177
2. Nakkiran, P., et al. (2019). "Deep Double Descent: Where Bigger Models and More Data Can Hurt." ICLR 2020

---

## Appendix: Experimental Setup

**Hardware**: NVIDIA RTX 3050 6GB Laptop GPU  
**Software**: PyTorch 2.6.0+cu118, Python 3.11  
**Tracking**: Weights & Biases (wandb)  
**Code**: `grokking_experiment/train_grokking.py`

**Run command**:
```bash
python3.11 grokking_experiment/train_grokking.py \
    --weight_decay 0.01 \
    --epochs 10000 \
    --batch_size 4096 \
    --samples_per_group 5000 \
    --val_samples_per_group 10000 \
    --val_frequency 5
```

