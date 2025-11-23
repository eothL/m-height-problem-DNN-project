import torch
import os

# --- Configuration ---
# <<<--- MODIFY HERE: Path to your saved model checkpoint file
MODEL_PATH = r'C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\best_moe_resnet_sparsegate_e6_k2_finetuned.pth'

# --- Load State Dict ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

print(f"Loading state_dict from: {MODEL_PATH}")
# Load onto CPU first to avoid potential GPU memory issues if the file is large
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
print("State dictionary loaded.")

# --- Inference ---
inferred_params = {}
max_block_index = -1

try:
    # Infer base_ch from stem conv weight shape [out_channels, in_channels, kH, kW]
    if 'stem.0.weight' in state_dict:
        inferred_params['base_ch'] = state_dict['stem.0.weight'].shape[0]
    elif 'stem.1.weight' in state_dict: # Or from stem BatchNorm weight shape [num_features]
         inferred_params['base_ch'] = state_dict['stem.1.weight'].shape[0]
    elif 'blocks.0.conv1.weight' in state_dict: # Or from first block conv weight [out, in, k, k]
        inferred_params['base_ch'] = state_dict['blocks.0.conv1.weight'].shape[0]
    else:
        print("Warning: Could not infer 'base_ch'. Look for keys like 'stem.0.weight', 'stem.1.weight', or 'blocks.0.conv1.weight'.")

    # Infer n_params from param_proj weight shape [out_features, in_features]
    if 'param_proj.weight' in state_dict:
        inferred_params['n_params'] = state_dict['param_proj.weight'].shape[1]
    else:
        # Check if the old name 'param_proj.linear.weight' was used
        if 'param_proj.linear.weight' in state_dict:
             inferred_params['n_params'] = state_dict['param_proj.linear.weight'].shape[1]
        else:
             print("Warning: Could not infer 'n_params'. Look for keys like 'param_proj.weight'.")


    # Infer num_blocks by finding the highest block index
    for key in state_dict.keys():
        if key.startswith('blocks.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_index = int(parts[1])
                max_block_index = max(max_block_index, block_index)

    if max_block_index != -1:
        inferred_params['num_blocks'] = max_block_index + 1
    else:
         print("Warning: Could not infer 'num_blocks'. No keys starting with 'blocks.[number].' found.")


    # Infer k_max * nk_max product
    if 'head.0.weight' in state_dict and 'base_ch' in inferred_params:
        head_input_dim_total = state_dict['head.0.weight'].shape[1]
        # head input = flat_dim + 64 (from param_proj)
        # flat_dim = base_ch * k_max * nk_max
        flat_dim = head_input_dim_total - 64
        base_ch = inferred_params['base_ch']
        if base_ch > 0 and flat_dim > 0 and flat_dim % base_ch == 0:
            inferred_params['k_max * nk_max'] = flat_dim // base_ch
        else:
             print(f"Warning: Could not infer 'k_max * nk_max'. Calculation invalid (flat_dim={flat_dim}, base_ch={base_ch}).")
    else:
         print("Warning: Could not infer 'k_max * nk_max'. Need 'head.0.weight' and 'base_ch'.")


except Exception as e:
    print(f"An error occurred during inference: {e}")
    print("Please ensure the model path is correct and the file contains a valid state_dict for ResNet2DWithParams.")

# --- Print Results ---
print("\n--- Inferred Hyperparameters ---")
if inferred_params:
    for key, value in inferred_params.items():
        print(f"{key}: {value}")

    # Add notes about defaults and ambiguity
    if 'n_params' not in inferred_params:
         print("n_params: Could not infer (Default is 3)")
    if 'k_max * nk_max' in inferred_params:
        print("Note: Could only infer the product 'k_max * nk_max'. Common values might be k_max=6, nk_max=6 if their product matches.")
    else:
         print("k_max, nk_max: Could not infer (Defaults are 6, 6)")

else:
    print("Could not infer any hyperparameters. Please check the state_dict keys manually:")
    print(list(state_dict.keys()))
