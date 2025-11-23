# m-Height Prediction with ResNet Variants

This repository contains the experiments and tooling I used for the CSCE 636 deep learning class project on predicting the m-height of a linear code from its generator matrix. The work started from a ResNet baseline and expanded into several architectural tweaks, mathematically informed losses, and hyper-parameter searches that pushed the validation LogMSE below the original reference implementation.

## Highlights
- End-to-end PyTorch training pipeline (`final_best_model_training.py`) that learns directly from padded generator matrices plus `(n, k, m)` metadata.
- Custom violation-informed loss that samples candidate codewords to ensure predictions respect known upper bounds, optionally combined with a lower-bound enforcing output layer.
- Utilities for splitting giant pickled DataFrames into balanced train/validation shards (`split_pickle_data.py`) and scripts for running ablations on attention blocks, activation functions, and gating strategies.
- Checkpoints for every major variant (SwiGLU, GELU, SE-ResNet, monotonic/violation-loss ResNets, Mixture-of-Experts) stored under descriptive filenames.
- Final best model: **ReLU-ResNet + lower-bound head + violation loss**, fine-tuned with Optuna, reaching a best validation LogMSE of **1.1231** on the validation set.

## Repository Layout
- `*.ipynb` &nbsp;— interactive notebooks for exploratory training, Optuna searches, and result visualizations.
- `final_*_training.py`, `resnet_*_training.py`, `finetune_*` &nbsp;— scriptable training entry points for each architecture variant.
- `split_pickle_data.py`, `create_small_test_set.py` &nbsp;— dataset preparation helpers that turn the large `results_dataframe` pickle into manageable folders of `.pkl` shards.
-  `data/`, `split_data*/`, `test_data/` &nbsp;— ignored dataset drops (see `.gitignore`).
- `best_*/*.pth` &nbsp;— saved checkpoints and Optuna-selected weights.
- `project_summary.md` &nbsp;— deeper write-up of the experimentation timeline and metrics.

## Setup
1. Use Python 3.10+ and create an isolated environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies. The core scripts rely on `torch`, `numpy`, `pandas`, `matplotlib`, `joblib`, `optuna`, and `scikit-learn`:
   ```bash
   pip install torch torchvision torchaudio pandas numpy matplotlib joblib optuna scikit-learn
   ```
3. (Optional) Install `jupyter` if you want to rerun the notebooks.

## Preparing Data
1. Place the master pickle (e.g., `results_dataframe(1)/results_dataframe.pkl`) under the project root.
2. Update the `input_pickle_path` inside `split_pickle_data.py` if your path differs.
3. Run the script to create balanced shards for each `(n, k, m)` combination:
   ```bash
   python split_pickle_data.py
   ```
   - Training shards land in `split_data_train_20000_random/` and validation shards in `split_data_validation_20000_random/`.
   - Adjust `data_per_file` or `combinations` to control shard size and coverage.
4. For quick sanity checks you can generate miniature folders via `create_small_test_set.py` or reuse the provided `split_data_small_test/` directory.

## Training the Best Model
The reproducible pipeline lives in `final_best_model_training.py`.
```bash
python final_best_model_training.py
```
Key details:
- Automatically loads all `.pkl` files under `split_data_train_20000_random/` and `split_data_validation_20000_random/`, pads matrices to `(max_k=6, max_n-k=6)`, and builds PyTorch `DataLoader`s.
- Uses the lower-bound enforced `ResNet2DWithParams` plus the violation-aware loss with Optuna-selected hyper-parameters.
- Saves the top checkpoint as `best_relu_resnet_lowerbound_violationloss_finetuned.pth` (and variants) and plots training vs. validation loss.

To experiment with other variants, run the corresponding script (e.g., `python resnet_swiglu_training.py`) or tweak the `BEST_PARAMS` dictionary to scan different architectures.

## Results 
- Ranking of the strongest models and a full discussion of each phase is available in `project_summary.md`.
- The saved weights under `best_*.pth` can be reloaded with standard `torch.load` calls for evaluation or downstream fine-tuning.

