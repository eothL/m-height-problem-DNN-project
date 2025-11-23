# --- Libraries ---
# Standard libraries
import os
import time
import uuid
import itertools
import traceback
import glob
import multiprocessing
import sys


# Data handling
import numpy as np
import pandas as pd
import pickle
import pulp 

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms

# Visualization
import matplotlib.pyplot as plt

# Utility checks
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
print("Default device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Example: UUID for unique filenames
unique_filename = f"data_{uuid.uuid4().hex}.pkl"
print("Example unique filename:", unique_filename)

# Example: Parallel multiprocessing
print("CPU cores available:", multiprocessing.cpu_count())
# --- Define the class "PickleFolderDataset" to load the data from pkl files ---
# --- Define the class "PickleFolderDataset" to load the data from pkl files ---

# --- Normalisation function ---
EPS = 1e-9
def norm_none(t):
    return t                                      # raw P

def norm_row_standard(t):
    mu  = t.mean(dim=1, keepdim=True)
    std = t.std (dim=1, keepdim=True) + EPS
    return (t - mu) / std                         # each row N(0,1)

def norm_col_minmax(t, to=(-1., 1.)):
    lo, hi = to
    col_min = t.min(dim=0, keepdim=True).values
    col_max = t.max(dim=0, keepdim=True).values
    rng = (col_max - col_min).clamp_min(EPS)
    return (t - col_min) / rng * (hi - lo) + lo   # columns -> [-1,1]

NORMALISERS = {
    "none":           norm_none,
    "row_standard":   norm_row_standard,
    "col_minmax":     norm_col_minmax,
}

# --- Data Loader class from Pickle files ---
class PickleFolderDataset(Dataset):
    """
    PyTorch Dataset for loading data from a **list of .pkl file paths**.

    Assumes each .pkl file contains a Pandas DataFrame with columns
    'n', 'k', 'm', 'result', 'P'.
    Handles cases where 'P' might be a 2D array OR a flattened 1D array.
    Reshapes 1D 'P' arrays to 2D using 'n' and 'k' before padding.
    Handles padding P_matrices. 
    Returns data as (params, h, P).
    """
    def __init__(self, file_paths: list, max_k :int = 6, max_nk:int = 6, transform:callable=None, p_normaliser:str = "none", return_col_indices=False): 
        """
        Args:
            file_paths (list[str]): A list of full paths to the .pkl files.
            max_k (int): The height to pad P matrices to.
            max_nk (int): The width to pad P matrices to.
            p_normaliser (str): key in NORMALISERS (str) – 'none', 'row_standard', 'col_minmax', ...
            return_col_indices (bool): if True -> __getitem__ returns an extra tensor [n‑k] with 0…n‑k‑1
            transform (callable): additional callable applied AFTER padding & normalisation
        """
        super().__init__()
        if p_normaliser not in NORMALISERS:
            raise ValueError(f"Invalid p_normaliser: {p_normaliser}. Must be one of: {list(NORMALISERS.keys())}")
        
        self.p_normaliser = NORMALISERS[p_normaliser]
        self.return_col_indices = return_col_indices
        self.max_k,self.max_nk = max_k,max_nk
        self.transform = transform

        if not file_paths:
            raise ValueError("The provided file_paths list is empty.")


        # ------------------------------------------------------------------
        # LOAD DATA  
        # ------------------------------------------------------------------        
        # Store data directly as loaded from pickle (P can be 1D or 2D)
        self.P_raw, self.h_vals, self.n_vals, self.k_vals, self.m_vals = \
            [], [], [], [], []

        print(f"Loading data from {len(file_paths)} specified pickle files (expecting DataFrames)...")
        required = ['n', 'k', 'm', 'result', 'P']

        for fp in file_paths: # Iterate through the provided list
            # print(f"  Loading: {os.path.basename(file_path)}") # Keep print concise
            try:
                with open(fp, "rb") as f:
                    df = pickle.load(f)
                if not all(col in df.columns for col in required):
                    continue

                self.P_raw.extend([np.array(p) for p in df['P']])
                self.h_vals.extend(df['result'].astype(float).tolist())
                self.n_vals.extend(df['n'].astype(int).tolist())
                self.k_vals.extend(df['k'].astype(int).tolist())
                self.m_vals.extend(df['m'].astype(int).tolist())

            except FileNotFoundError:
                    print(f"Error: File not found {fp}. Skipping.")
            except Exception as e:
                print(f"Error loading/processing {fp}: {type(e).__name__} - {e}. Skipping file.")
                # print(traceback.format_exc()) # Uncomment for more detail

        if not self.h_vals:
                raise ValueError("No valid data loaded from any pickle files.")

        if not self.h_vals:
                raise ValueError("No valid data loaded from any pickle files.")

        print(f"Finished initial loading. Total samples found: {len(self.all_h_values)}")

        # --- Convert non-P lists to tensors ---
        self.h_vals  = torch.tensor(self.h_vals , dtype=torch.float32)
        self.n_vals  = torch.tensor(self.n_vals , dtype=torch.float32)
        self.k_vals  = torch.tensor(self.k_vals , dtype=torch.float32)
        self.m_vals  = torch.tensor(self.m_vals , dtype=torch.float32)

    def __len__(self):
        return len(self.h_vals)

    def _pad(self, p2d: torch.Tensor):
        k_act, nk_act = p2d.shape
        pad = (0, self.max_nk - nk_act, 0, self.max_k - k_act)  # (W_left,W_right,H_top,H_bottom)
        return F.pad(p2d, pad, value=0.)

    def __getitem__(self, idx):
        # --- reshape / pad ---
        p_np   = self.P_raw[idx]
        n, k   = int(self.n_vals[idx]), int(self.k_vals[idx])
        target = (k, n - k)
        if p_np.ndim == 1:
            p_np = p_np.reshape(*target)  # raises if shape mismatched
        p_t = torch.tensor(p_np, dtype=torch.float32)
        p_t = self._pad(p_t)

        # --- normalise ---
        p_t = self._normalise(p_t)

        # --- extra transform hook ---
        if self.transform:
            p_t = self.transform(p_t)

        # --- build output ---
        params  = torch.tensor([self.n_vals[idx], self.k_vals[idx], self.m_vals[idx]],
                               dtype=torch.float32)
        h       = self.h_vals[idx].unsqueeze(0)

        if self.return_col_indices:
            col_idx = torch.arange(target[1], dtype=torch.long)      # 0 … n‑k‑1
            return params, h, p_t, col_idx

        return params, h, p_t
    


# --- Data Loading and Preparation ---
# --- Configuration ---
TRAIN_DATA_FOLDERS = ['./split_data_train_20000_random']
VALIDATION_DATA_FOLDERS = ['./split_data_validation_20000_random']
max_k = 6
max_nk = 6
batch_size = 512 
val_split_ratio = 0.2 # 80/20 split
NUM_WORKERS = 0 # Safer default for Windows
PIN_MEMORY = True # Generally good if using GPU

# --- Variables to store loaders (so they are accessible by the next cell) ---
train_loader = None
val_loader = None

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load and Split Data ---
try:
    print("--- Preparing Data ---")
    all_files = []
    for folder in TRAIN_DATA_FOLDERS + VALIDATION_DATA_FOLDERS:
         files = glob.glob(os.path.join(folder, '*.pkl'))
         if not files:
              print(f"Warning: No .pkl files found in folder: {folder}")
         all_files.extend(files)

    if not all_files:
        raise FileNotFoundError("No .pkl files found in any specified train/validation folders.")

    print(f"Found {len(all_files)} total .pkl files.")

    # Instantiate the dataset with the combined list of files
    combined_dataset = PickleFolderDataset(
        file_paths=all_files,
        max_k=max_k,
        max_nk=max_nk
    )

    total_samples = len(combined_dataset)
    print(f"Total samples loaded from combined files: {total_samples}")

    # Perform random split
    val_size = int(total_samples * val_split_ratio)
    train_size = total_samples - val_size

    if train_size <= 0 or val_size <= 0:
         raise ValueError(f"Calculated train ({train_size}) or validation ({val_size}) size is zero or less.")

    print(f"Splitting data: Training={train_size} ({100*(1-val_split_ratio):.1f}%), Validation={val_size} ({100*val_split_ratio:.1f}%)")
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    # Create DataLoaders and store them in global scope for this cell block
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    print("\nFinal DataLoaders created successfully.")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

except FileNotFoundError as e:
    print(f"\nData Loading Error: {e}")
    print("Please ensure the TRAIN_DATA_FOLDERS and VALIDATION_DATA_FOLDERS paths are correct.")
except ValueError as e:
     print(f"\nData Loading/Splitting Error: {e}")
     print("Check data contents, split ratio, or if the dataset ended up empty.")
except NameError as e:
     print(f"\nDefinition Error: {e}. Make sure 'PickleFolderDatasetWithParams' class is defined and executed first.")
except ImportError as e:
     print(f"\nImport Error: {e}. Make sure required libraries (torch, pickle, pandas etc.) are imported.")
except Exception as e:
    print(f"\nAn unexpected error occurred during data loading/splitting: {type(e).__name__} - {e}")
    print("Traceback:")
    print(traceback.format_exc())

# --- Define functions for training and testing the model and data pre processing, plotting---

# --- Define the custom MSE loss function on the log2 scale --- 
class LogMSELoss(nn.Module):
    """
    Calculates the Mean Squared Error on the log2 scale.
    Loss = mean( (log2(y_true) - log2(y_pred))^2 )
    Adds a small epsilon to prevent log2(0).
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Ground truth values.
        """
        # Clamp inputs to be positive to avoid log2(<=0)
        y_pred_safe = torch.clamp(y_pred, min=self.eps)
        y_true_safe = torch.clamp(y_true, min=self.eps) # Clamp true values too for safety

        # Calculate log base 2
        log2_pred = torch.log2(y_pred_safe)
        log2_true = torch.log2(y_true_safe)

        # Calculate the squared difference and mean over the batch
        loss = torch.mean((log2_true - log2_pred) ** 2)
        return loss

def train_epoch_with_params(model, dataloader, criterion, optimizer, device):
    """Trains the model (accepting params, h, P) for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Make sure this unpacking order matches the Dataset's __getitem__ return order
    for i, batch_data in enumerate(dataloader):
        # Unpack data correctly based on Dataset return order (params, h, P)
        params, targets, p_matrices = batch_data
        params, targets, p_matrices = params.to(device), targets.to(device), p_matrices.to(device)

        optimizer.zero_grad()

        # Pass data to the model in the order its forward method expects
        # Assuming models expect (p_matrix, params)
        outputs = model(p_matrices, params)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss correctly using the actual batch size
        running_loss += loss.item() * params.size(0)
        total_samples += params.size(0)

        ## --- Batch-level print ---
        # if (i + 1) % 100 == 0:
        #      print(f'  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')

    # Calculate average loss for the epoch
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss

# --- Keep validate_epoch_with_params as is (it doesn't have batch printing) ---
def validate_epoch_with_params(model, dataloader, criterion, device):
    """Evaluates the model (accepting params, h, P) on the validation set."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        # Make sure this unpacking order matches the Dataset's __getitem__ return order
        for batch_data in dataloader:
            # Unpack data correctly (params, h, P)
            params, targets, p_matrices = batch_data
            params, targets, p_matrices = params.to(device), targets.to(device), p_matrices.to(device)

            # Pass data to the model in the order its forward method expects
            # Assuming models expect (p_matrix, params)
            outputs = model(p_matrices, params)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * params.size(0) # Use actual batch size
            total_samples += params.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss



def load_and_split_data(train_folders, val_folders, max_k, max_nk, batch_size, val_split_ratio=0.2, num_workers=0, pin_memory=True, dataset_class=PickleFolderDataset):
    """
    Loads data from specified folders, combines them, performs a random split,
    and returns training and validation DataLoaders.

    Args:
        train_folders (list[str]): List of paths to folders containing training .pkl files.
        val_folders (list[str]): List of paths to folders containing validation .pkl files.
                                  (Data will be combined before splitting).
        max_k (int): Max height for padding P matrices.
        max_nk (int): Max width for padding P matrices.
        batch_size (int): Batch size for DataLoaders.
        val_split_ratio (float): Proportion of the *combined* data to use for validation (e.g., 0.2 for 20%).
        num_workers (int): Number of worker processes for DataLoader.
        pin_memory (bool): Whether to use pin_memory for DataLoader.
        dataset_class (Dataset): The Dataset class to use (e.g., PickleFolderDataset).

    Returns:
        tuple: (train_loader, val_loader)
    """
    print("--- Preparing Data ---")
    all_files = []
    for folder in train_folders + val_folders: # Combine paths from both lists
         files = glob.glob(os.path.join(folder, '*.pkl'))
         if not files:
              print(f"Warning: No .pkl files found in folder: {folder}")
         all_files.extend(files)

    if not all_files:
        raise FileNotFoundError("No .pkl files found in any of the specified train/validation folders.")

    print(f"Found {len(all_files)} total .pkl files.")

    # Instantiate the dataset with the combined list of files
    # Note: This loads ALL data into memory. May be an issue for extremely large datasets.
    combined_dataset = dataset_class(
        file_paths=all_files,
        max_k=max_k,
        max_nk=max_nk
        # transform can be added here if needed
    )

    total_samples = len(combined_dataset)
    print(f"Total samples loaded from combined files: {total_samples}")

    # Perform random split
    val_size = int(total_samples * val_split_ratio)
    train_size = total_samples - val_size

    if train_size <= 0 or val_size <= 0:
         raise ValueError(f"Calculated train ({train_size}) or validation ({val_size}) size is zero or less. Check data or split ratio.")

    print(f"Splitting data: Training={train_size} ({100*(1-val_split_ratio):.1f}%), Validation={val_size} ({100*val_split_ratio:.1f}%)")
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print("DataLoaders created.")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    return train_loader, val_loader


def run_training(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_name="Model", scheduler=None):
    """
    Runs the training and validation loop for a given model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        model_name (str): Name of the model for printing logs.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.

    Returns:
        tuple: (train_losses, val_losses) lists containing loss per epoch.
    """
    print(f"\n--- Starting Training Loop for {model_name} ---")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time_train = time.time()

    # Ensure train_epoch_with_params and validate_epoch_with_params expect (params, h, P)
    # If they don't, they need to be redefined/updated before this function is called.

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs} ({model_name})")

        # Ensure train/validate functions handle the (params, h, P) order
        train_loss = train_epoch_with_params(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch_with_params(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary ({model_name}): Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            print(f"  {model_name} Val loss improved ({best_val_loss:.4f} -> {val_loss:.4f}).")
            best_val_loss = val_loss
            # Optional: Save model checkpoint here
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ","_")}_final.pth')

        # Step the scheduler, if provided
        if scheduler is not None:
            # ReduceLROnPlateau expects the validation loss, others do not
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    total_time_train = time.time() - start_time_train
    print(f"\n{model_name} Training Finished. Total time: {total_time_train:.2f} seconds")
    return train_losses, val_losses


def plot_losses(results_dict):
    """
    Plots training and validation losses for multiple models.

    Args:
        results_dict (dict): A dictionary where keys are model names and
                             values are tuples of (train_losses, val_losses).
                             Example: {'Dense': ([...], [...]), 'CNN': ([...], [...])}
    """
    num_models = len(results_dict)
    if num_models == 0:
        print("No results to plot.")
        return

    print("\nPlotting losses...")
    # Adjust layout based on number of models
    cols = 2
    rows = (num_models + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 5 * rows))

    i = 1
    for model_name, (train_losses, val_losses) in results_dict.items():
        if not train_losses or not val_losses:
            print(f"Skipping plot for {model_name} due to missing loss data.")
            continue
        plt.subplot(rows, cols, i)
        plt.plot(train_losses, label=f'{model_name} Train Loss')
        plt.plot(val_losses, label=f'{model_name} Val Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log2 MSE Loss') # Assuming LogMSELoss was used
        plt.legend()
        plt.grid(True)
        i += 1

    plt.tight_layout()
    plt.show()


# --- fine tuning FCN ---
# --- configuration ---
fcn_padded_p_matrix_flat_size = max_k * max_nk
fcn_params_size = 3

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for FCN tuning: {device}")

# --- Flexible Dense Network Definition ---
class FlexibleDenseNetworkWithParams(nn.Module):
    """
    Dense Network accepting flattened padded P matrix and n, k, m parameters.
    Hidden layers are defined by a list of dimensions.
    """
    def __init__(self, p_input_size, param_input_size, hidden_dims=[128, 64, 32], output_size=1, dropout_prob=0.2):
        super().__init__()
        self.param_input_size = param_input_size
        combined_input_size = p_input_size + param_input_size

        layers = []
        prev_dim = combined_input_size
        # Input layer check: Ensure first hidden dim connects to combined input
        if not hidden_dims: # Handle case of no hidden layers (direct linear)
             layers.append(nn.Linear(combined_input_size, output_size))
        else:
            # First hidden layer
            layers.append(nn.Linear(combined_input_size, hidden_dims[0]))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dims[0]

            # Subsequent hidden layers
            for i in range(1, len(hidden_dims)):
                h_dim = hidden_dims[i]
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
                prev_dim = h_dim

            # Final output layer connects from the last hidden layer
            layers.append(nn.Linear(prev_dim, output_size))

        self.network = nn.Sequential(*layers)
        print(f"Initialized FlexibleDenseNetworkWithParams:")
        print(f"  Input Size (Flat P + Params): {combined_input_size}")
        print(f"  Hidden Dims: {hidden_dims}")
        print(f"  Output Size: {output_size}")
        # print(self.network) # Optional: print the layer structure

    def forward(self, p_matrix, params):
        batch_size = p_matrix.size(0)
        p_flat = p_matrix.view(batch_size, -1) # Flatten P matrix

        # Ensure params tensor has the correct shape (batch_size, num_params)
        if params.dim() == 1: # If it's a single sample (batch size 1 during inference maybe?)
             params = params.unsqueeze(0)
        if params.size(1) != self.param_input_size:
             raise ValueError(f"Params tensor second dimension ({params.size(1)}) != expected param_input_size ({self.param_input_size})")

        combined_input = torch.cat((p_flat, params), dim=1)
        return self.network(combined_input)

# --- Training Function for One Tuning Trial ---
def run_fcn_tuning_trial(config, train_loader, val_loader, p_input_size, param_input_size, epochs, device):
    """Trains and evaluates one FCN configuration."""
    print(f"\n--- Starting Trial: {config.get('name', config)} ---")
    start_time = time.time()

    # Extract config
    hidden_dims = config['hidden_dims']
    learning_rate = config['lr']
    dropout_prob = config.get('dropout', 0.2) # Use default if not specified
    weight_decay = config.get('weight_decay', 0) # Get weight_decay from config, default to 0

    # Ensure Loss function is defined (should be LogMSELoss)
    if 'LogMSELoss' not in globals():
         raise NameError("LogMSELoss class is not defined.")
    criterion = LogMSELoss()

    # Instantiate model, criterion, optimizer
    model = FlexibleDenseNetworkWithParams(
        p_input_size=p_input_size,
        param_input_size=param_input_size,
        hidden_dims=hidden_dims,
        dropout_prob=dropout_prob
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Check if training functions are defined
    if 'train_epoch_with_params' not in globals() or 'validate_epoch_with_params' not in globals():
         raise NameError("train_epoch_with_params or validate_epoch_with_params function is not defined.")

    print(f"  Training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch_with_params(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch_with_params(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    end_time = time.time()
    print(f"--- Trial Finished: {config.get('name', config)} ---")
    print(f"  Best Val Loss during trial: {best_val_loss:.4f}")
    print(f"  Total Time: {end_time - start_time:.2f}s")

    # Return the minimum validation loss achieved during this trial
    return {'config': config, 'best_val_loss': best_val_loss, 'train_losses': train_losses, 'val_losses': val_losses}

print("Setup Complete: Flexible FCN class and tuning trial function are defined.")

# --- Final Training of the Best FCN Configuration ---
# The BEST result printed at the end of the previous cell 
best_fcn_config = {
    'name': 'FCN_128_64_32_lr5e-4', 
    'hidden_dims': [512, 256, 128],    
    'lr': 0.0005,                  
    'dropout': 0.3,                  
    'weight_decay': 1e-5  # I added L2 regularization as it improves a little bit the generalization of the model when I test with and without it
}

# --- Training Parameters for Final Run ---
epochs = 40 #
model_save_path = 'best_fcn_pretrained_weights.pth' # Filename for saved weights

# --- Prerequisites Check ---
if 'FlexibleDenseNetworkWithParams' not in locals():
    print("ERROR: FlexibleDenseNetworkWithParams class definition not found. Run Cell 1.")
elif 'LogMSELoss' not in locals():
    print("ERROR: LogMSELoss class definition not found.")
elif 'train_epoch_with_params' not in locals() or 'validate_epoch_with_params' not in locals():
     print("ERROR: Training loop epoch functions not found.")
elif 'train_loader' not in locals() or 'val_loader' not in locals():
    print("ERROR: train_loader or val_loader not found.")
else:
    print(f"\n--- Starting Final Training for Selected FCN ---")
    print(f"Using Configuration: {best_fcn_config}")
    print(f"Training for {epochs} epochs.")
    print(f"Best model weights will be saved to: {model_save_path}")

    # --- Instantiate the Best Model ---
    fcn_model = FlexibleDenseNetworkWithParams(
        p_input_size=fcn_padded_p_matrix_flat_size, # Should be defined in previous cell
        param_input_size=fcn_params_size,           # Should be defined in previous cell
        hidden_dims=best_fcn_config['hidden_dims'],
        dropout_prob=best_fcn_config.get('dropout', 0.2) # Use dropout if specified
    ).to(device)

    # --- Criterion and Optimizer ---
    criterion = LogMSELoss()
    optimizer = optim.Adam(fcn_model.parameters(), lr=best_fcn_config['lr'],weight_decay = best_fcn_config['weight_decay'])

    # --- Training Loop (with saving best model) ---
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    print("Starting final training run...")
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Perform one epoch of training and validation
        train_loss = train_epoch_with_params(fcn_model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch_with_params(fcn_model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model's state dictionary
            torch.save(fcn_model.state_dict(), model_save_path)
            print(f"  -> Val loss improved to {best_val_loss:.4f}. Model weights saved to {model_save_path}")

    total_time = time.time() - start_time
    print(f"\nFinal FCN Training Finished.")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Best Validation Loss achieved during training: {best_val_loss:.4f}")
    print(f"  Best model weights saved to: {model_save_path}")

    # --- Optional: Plot Training Curve ---
    if train_losses and val_losses and 'matplotlib' in sys.modules:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label=f'Val Loss (Best: {best_val_loss:.4f})')
            # Mark the best validation point
            best_epoch_idx = np.argmin(val_losses)
            plt.scatter([best_epoch_idx], [best_val_loss], color='red', s=50, zorder=5, label=f'Best Val @ Epoch {best_epoch_idx+1}')
            plt.title(f'Final Training: Best FCN ({best_fcn_config.get("name", "Config")})')
            plt.xlabel('Epoch')
            plt.ylabel('Log2 MSE Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as plot_err:
            print(f"\nWarning: Plotting curve failed - {plot_err}")


print("\nStep A (FCN Pre-training) Complete. Weights are saved in:", model_save_path)