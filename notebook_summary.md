# Lin_Theo_m_height_problem.ipynb – Experiment Log


## Goal & Dataset Creation
- Objective: approximate the m-height returned by the traditional LP-based solver with a deep model that is much faster while keeping the error minimal.
- Data generation: initial attempt with PuLP on HPRC yielded ~10k samples (limited by large `m`). To scale further, a dataset created by a student from the class was adopted (≈420k rows out of 1M). Data are stored as pickle DataFrames and reshaped into balanced `split_data_train_20000_random/` and `split_data_validation_20000_random/` folders.
- Overall workflow: load `.pkl` shards with `PickleFolderDataset`, pad each `P` matrix to `(max_k=6, max_(n-k)=6)`, append `(n, k, m)` metadata, and train with Log2-MSE.

## Shared Infrastructure between each project step
- Created helper modules for data loading, normalization functions, and dataset statistics (cell 6 prints `(n,k,m)` distribution histograms to confirm balance after the 80/20 split).
- Defined the `LogMSELoss` (squared error in log2 space) plus reusable PyTorch training/evaluation loops.

## Project 3 Agenda (from notebook cell 8)
1. Reframe regression as classification.
2. Try Mixture-of-Experts (MoE) with hard and sparse gating.
3. Explore ViT-style transformers.
4. Fine-tune ResNet and MoE variants.

Everything below follows that agenda.

## Attempt 1 – Classification Reformulation
- Analyzed log2(h) distributions per `(n,k,m)` and overall to determine how many classes would be reasonable. Used Freedman–Diaconis heuristics; first test used 100 bins, later increased to 288 bins (cell 17) without any metric gains.
- Wrapped the regression dataset with `ClassifWrapper`, which converts the real targets into class indices via the chosen bin edges while retaining the raw `h` for evaluation.
- Implemented two classifier backbones: a flexible dense network and a `ResNet2DWithParams` classifier.
- Trained both with cross-entropy (50 epochs, Adam), then converted predictions back to scalar heights to report Log2-MSE.
- Result: neither classifier beat the best regression baseline (1.13 Log2-MSE). The follow-up comparison cell explicitly states that the classification framing did not improve performance.

## Attempt 2 – Vision Transformer
- Implemented `ViT2DWithParams`: patches the padded `P` matrix, feeds them into a Transformer encoder, concatenates pooled tokens with `(n,k,m)`, and regresses m-height.
- Training outcome: validation loss plateaued around **1.30**, slower and worse than standard ResNet baselines despite the heavier architecture.

## Attempt 3 – Mixture of Experts (ResNet Experts)
### Hard-Gated MoE
- Experts: six ResNet2D backbones (one per `(n,k)` pair). Gating looked at `(n,k)` and routed each example to exactly one expert.
- Trained for 3000 epochs with AdamW (`lr=3e-4`, `wd=1e-4`).
- Observed strong overfitting: training loss ≈0.19, validation loss ≈**1.29**. Notebook notes suspect different `(n,k,m)` proportions between the random train/val splits.

### Sparse Soft-Gated MoE
- Introduced `MoE_ResNet_SparseGate`: transformer-style gating network that scores experts and selects top-`k` experts per sample (k=1 or 2). Includes learnable temperature/regularizer to encourage sparse gates.
- Training summary (cell 24–25):
  - `top_k=1` → validation Log2-MSE ~**1.19**.
  - `top_k=2` → validation Log2-MSE ~**1.21**.
  - Both significantly improve on hard gating.

### Pre-trained ResNet Experts
- Re-trained standalone ResNets with varied base channels and residual block counts (cell 26) to use them as initialization for the sparse MoE.
- Findings:
  - 32 channels × 3 blocks already competitive.
  - 64 channels × 5 blocks performed best individually, so both checkpoints were recycled for MoE experiments (cell 27).
- Fine-tuned sparse MoE with `base_ch=64`, `num_blocks=5`, `top_k=2`. Best validation Log2-MSE recorded in the notebook: **0.9197** (cell 29). Using the lighter 32-channel experts yielded ~**1.0687**.
- Evaluation cell 30 reloads `best_moe_resnet_65_5_sparsegate_e6_k2_finetuned.pth` and runs validation to confirm the metric.

## Attempt 4 – Alternative Input Normalization
- Ran a parallel experiment that normalized `P` values with `transforms.Normalize(mean=[0], std=[100])`, creating `train_loader_norm`/`val_loader_norm` (cells 110–112). No explicit metric was logged afterward, but it shows that normalization support exists if future models need it.

## Attempt 5 – Baseline Architecture Shootout
- Compared a plain fully connected network (FCN) against a CNN when fed flattened `P` matrices.
- Outcomes (cell 113): FCN achieved ~**1.6** loss while the CNN ballooned to ~**9.5**, leading to the conclusion that spatial convolutions were not the right inductive bias.
- Hypothesis documented (cell 113 markdown): m-height depends more on global relationships within `G=[I|P]` than on local pixel-like patterns, which motivates FCN-style processing.

## Attempt 6 – FCN Hyperparameter Search
- Built `FlexibleDenseNetworkWithParams` and supporting training routines.
- Grid searched 22 configurations varying hidden widths, dropout, and learning rates (cell 116). Best validation Log2-MSE: **1.4056** for `[512,256,128]`, `lr=5e-4`, dropout 0.3.
- Final FCN training (cell 118) saved weights to `best_fcn_pretrained_weights.pth`; validation losses around **1.37** with weight decay, **1.36** without (cell 119).

## Attempt 7 – MoE with FCN Experts
### Hard Gate
- Implemented `MoE_FCN` to route samples to dense experts keyed by `(n,k)` (cell 120–121). Training code mirrors the ResNet version, but no explicit metric was logged in the visible cells.

### Soft Gate & Fine-Tuning
- Added a softmax gating network (`GatingNetwork`) plus small FCN experts (cells 122–125). Key observations:
  - Soft gating supports smoother expert blending and better gradient flow vs. hard gating (cell 125 markdown).
  - Re-used best FCN weights for each expert and fine-tuned the whole MoE (cell 126–128). Validation loss improved from **1.37 → 1.344**.

## Final Evaluation Cells
1. **Project 1 evaluator** (cell 133) using `best_moe_finetuned_final.pth` and the provided evaluation harness. Validation loss stayed around **1.34**, but test loss increased to **1.54** on `split_data_small_test` (cell 134).
2. **Project 2 evaluator** (cell 136) for the standalone ResNet checkpoint `best_resnet2d_no_norm.pth`. Validation Log2-MSE ≈**1.13**, random test ≈**1.19** (cell 137).
3. **Project 3 evaluator** (cell 139) for the final sparse MoE. Reported metrics: validation **0.88**, random test **0.9349** (cell 140), confirming a substantive improvement over prior phases.

## High-Level Lessons Captured in the Notebook
- Dataset hygiene (balanced `(n,k,m)` sampling, padding logic, normalization knobs) was essential before meaningful model comparisons could happen.
- Turning the problem into classification and deploying heavier Transformers both underperformed the simpler regression ResNet, reinforcing that inductive bias matters more than raw model capacity here.
- MoE structures dominate once experts are initialized from strong single-task checkpoints and gating is allowed to be soft/sparse. Fine-tuning experts together with the gate brought validation loss down to ~0.92, and final evaluator runs recorded even lower (~0.88).
- FCNs remain strong baselines for quick experimentation, and their weights served as good starting points for MoE experiments even after the ResNet variants took over.

