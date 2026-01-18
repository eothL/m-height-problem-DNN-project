# m-Height Prediction Project

This repository captures my CSCE 636 project at Texas A&M under the professor Anxiao Jiang on predicting the m-height of linear codes directly from generator matrices. The aim is to replace the slow LP-based solver with a deep model that can deliver a acceptable accuracy in a fraction of the time. I just dumped everything in this repo, it is not clean at all but you can see all the model tried and code, I created a cleaned version.

## Project Overview
- **Data Source** – Began with ~10k samples generated via PuLP/HPRC, then incorporated ~420k rows from a dataset generated. All records are stored as pickled DataFrames and reshaped into balanced shards under `split_data_*` folders.
- **Core Pipeline** – `PickleFolderDataset` pads each `P` matrix to `(6, 6)`, attaches `(n, k, m)` metadata, and feeds tensors into PyTorch models using a Log2-MSE objective. Scripts like `split_pickle_data.py`, `final_best_model_training.py`, and numerous `resnet_*`/`finetune_*` files implement the experiments.
- **Version Control** – `.gitignore` keeps every folder containing "data" (and all checkpoints) out of Git so only code and lightweight docs live in the repo.

## Workload Steps
1. **Data Preparation** – Verified `(n, k, m)` distributions after splitting the master pickle; added optional normalization hooks for future models.
2. **Baseline Regression Models** – Benchmarked FCNs, CNNs, RNNs, and vanilla ResNets on smaller subsets; FCNs and ResNets emerged as the only competitive families.
3. **Alternative Formulations** – Reframed the task as classification (100–288 bins) and implemented a ViT-style encoder, but both underperformed the Log2-MSE regression baseline.
4. **FCN Refinement** – Ran a 22-configuration sweep over hidden widths, dropout, learning rate, and weight decay. Best FCN achieved ~1.36–1.37 Log2-MSE and served as a strong initialization point.
5. **ResNet Enhancements** – Added lower-bound enforcement (`1 + softplus(z)`), custom violation-aware loss, and Optuna tuning to push the pure ResNet model to ~1.13 Log2-MSE.
6. **Mixture-of-Experts (MoE)** – Built both hard- and sparse-gated MoEs using ResNet (and later FCN) experts. Soft/sparse gating with pre-trained experts provided the largest jump in accuracy.
7. **Evaluation & Packaging** – Final models were validated with the provided TAMU evaluator scripts, and best checkpoints were saved under descriptive filenames (`best_moe_resnet_65_5_sparsegate_e6_k2_finetuned.pth`, etc.).

## Final Result
- **Best Architecture** – Sparse soft-gated MoE with six ResNet experts (each 64 channels × 5 residual blocks), gating on `(n, k)` and fine-tuned end-to-end.
- **Metrics** – Validation Log2-MSE ≈ **0.88**, random hold-out test ≈ **0.93**, compared to the original LP baseline (~1.13).
- **Reproduction** – Running `python final_best_model_training.py` after regenerating `split_data_*` shards reproduces the lower-bound + violation-loss ResNet. For the MoE, load the saved weights and reuse the same data folders.
- **Average log2 MSE loss per (n,k,m) group**:  
Validation Loss per (n, k, m) Group:  
            Avg Log2 MSE 
(n, k, m)                 
(9, 4, 2)           0.1456   
(9, 4, 3)           0.1425    
(9, 4, 4)           0.5461    
(9, 4, 5)           2.0272    
(9, 5, 2)           0.1180    
(9, 5, 3)           0.4621     
(9, 5, 4)           2.1613    
(9, 6, 2)           0.1349    
(9, 6, 3)           2.1572    
(10, 4, 2)          0.8037    
(10, 4, 3)          0.0827     
(10, 4, 4)          0.1773   
(10, 4, 5)          0.6114     
(10, 4, 6)          2.1771    
(10, 5, 2)          0.0862     
(10, 5, 3)          0.1978    
(10, 5, 4)          0.6641    
(10, 5, 5)          2.5144      
(10, 6, 2)          0.1197     
(10, 6, 3)          0.5451    
(10, 6, 4)          2.6729

We can see that high value of m in each pairs of (n,k) has higher loss value, it maybe due to the value difference is bigger. 


