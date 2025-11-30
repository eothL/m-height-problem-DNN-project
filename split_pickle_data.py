import pandas as pd
import os
# import pickle # No longer directly used for loading the main file
import sys
import joblib # Added for loading

# --- Configuration ---
# Use a raw string (r"...") or double backslashes (\) for Windows paths
input_pickle_path = r"C:\\Users\\theo-\\OneDrive\\Documents\\VS Code project\\Deep learning\\Project\\results_dataframe(1)\\results_dataframe.pkl"
# Place the output directories in the current script's directory
output_dir_train = "split_data_train_20000_random"
output_dir_val = "split_data_validation_20000_random"
data_per_file = 20000
random_state_seed = 42 # Seed for reproducible shuffling
combinations = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
    (10, 4, 2), (10, 4, 3), (10, 4, 4), (10, 4, 5), (10, 4, 6),
    (10, 5, 2), (10, 5, 3), (10, 5, 4), (10, 5, 5),
    (10, 6, 2), (10, 6, 3), (10, 6, 4)
]
# --- End Configuration ---

# --- Main Logic ---
# Create output directories if they don't exist relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
full_output_dir_train = os.path.join(script_dir, output_dir_train)
full_output_dir_val = os.path.join(script_dir, output_dir_val)
os.makedirs(full_output_dir_train, exist_ok=True)
os.makedirs(full_output_dir_val, exist_ok=True)
print(f"Training output directory: {full_output_dir_train}")
print(f"Validation output directory: {full_output_dir_val}")

df = None # Initialize df
print(f"Loading data from {input_pickle_path} using joblib...")
try:
    # Load data using joblib
    loaded_data = joblib.load(input_pickle_path)
    print("Data loaded successfully using joblib.")
    print(f"Type of loaded data: {type(loaded_data)}")

    # Check if it's already a DataFrame
    if isinstance(loaded_data, pd.DataFrame):
        print("Loaded data is already a pandas DataFrame.")
        df = loaded_data
    else:
        # Attempt to convert the loaded data to a pandas DataFrame
        print("Attempting to convert loaded data to pandas DataFrame...")
        try:
            df = pd.DataFrame(loaded_data)
            print("Successfully converted loaded data to pandas DataFrame.")
        except Exception as convert_e:
            print(f"Error: Could not convert the data loaded by joblib (type: {type(loaded_data)}) into a pandas DataFrame: {convert_e}")
            print("The script requires a DataFrame structure to filter by 'n', 'k', 'm' columns.")
            print("Please inspect the structure of the data in the input pickle file manually to determine how to convert it.")
            # Example: If it's a list of dicts, pd.DataFrame(loaded_data) might work.
            # If it's a list of lists, you might need pd.DataFrame(loaded_data, columns=['col1', 'col2', ...])
            # If it's a dict of lists, pd.DataFrame(loaded_data) might work.
            sys.exit(1)

    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    # Check if required columns exist
    required_cols = ['n', 'k', 'm']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: DataFrame missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        print("Please ensure the columns 'n', 'k', and 'm' exist or adjust the script.")
        sys.exit(1) # Exit the script

except FileNotFoundError:
    print(f"Error: Input file not found at {input_pickle_path}")
    sys.exit(1)
except ImportError as import_e:
     # Check if it's pandas or joblib
     if 'pandas' in str(import_e):
         print("Error: pandas library not found. Please install it using 'pip install pandas'")
     elif 'joblib' in str(import_e):
         print("Error: joblib library not found. Please install it using 'pip install joblib'")
     else:
         print(f"Error: Required library not found: {import_e}. Please install it.")
     sys.exit(1)
except Exception as e:
    print(f"Error loading file with joblib or processing initial DataFrame: {e}")
    sys.exit(1)

if df is None:
    # This check might be redundant now due to earlier exits, but safe to keep.
    print("Error: DataFrame could not be created or loaded. Exiting.")
    sys.exit(1)

print(f"\nProcessing {len(combinations)} combinations...")

total_saved_train_count = 0
total_saved_val_count = 0

for n_val, k_val, m_val in combinations: # Renamed loop variables slightly
    print(f"\n  Processing combination: n={n_val}, k={k_val}, m={m_val}")

    try:
        # Ensure columns are numeric for comparison (do this once before the loop if possible, but safer here if df structure varies)
        # Add error handling in case conversion fails for specific rows/values
        try:
            # Create temporary columns for comparison if conversion is needed, or convert inplace carefully
            n_col = pd.to_numeric(df['n'], errors='coerce')
            k_col = pd.to_numeric(df['k'], errors='coerce')
            m_col = pd.to_numeric(df['m'], errors='coerce')
        except KeyError as ke:
             print(f"    Error: Missing column during numeric conversion check: {ke}. Skipping combination.")
             continue
        except Exception as num_e:
             print(f"    Warning: Error during numeric conversion: {num_e}. Filtering might be unreliable.")
             # Decide whether to continue with potentially mixed types or skip
             # For now, let's try filtering anyway, pandas might handle it depending on the error
             n_col = df['n']
             k_col = df['k']
             m_col = df['m']


        # Filter the DataFrame using the potentially converted numeric columns
        # Handle potential NaN values from coercion if necessary
        subset_df = df[(n_col == n_val) & (k_col == k_val) & (m_col == m_val) & n_col.notna() & k_col.notna() & m_col.notna()]


        if subset_df.empty:
            print(f"    Warning: No data found for combination n={n_val}, k={k_val}, m={m_val}.")
            continue

        available_rows = len(subset_df)
        print(f"    Total data points available: {available_rows}")

        # --- Shuffle the data for this combination ---
        if available_rows > 1: # Shuffling requires more than 1 row
            print(f"    Shuffling data with random state {random_state_seed}...")
            subset_df = subset_df.sample(frac=1, random_state=random_state_seed).reset_index(drop=True)
        else:
            print("    Only 1 or 0 rows available, skipping shuffle.")

        # --- Define indices for training and validation (applied to shuffled data) ---
        train_start_index = 0
        train_end_index = data_per_file
        val_start_index = data_per_file
        val_end_index = data_per_file * 2

        # --- Process Training Set ---
        train_subset_to_save = None
        if available_rows >= train_start_index:
            actual_train_end_index = min(train_end_index, available_rows)
            train_subset_to_save = subset_df.iloc[train_start_index:actual_train_end_index].copy()
            num_train_saved = len(train_subset_to_save)
            print(f"    Selected {num_train_saved} random points for training.")
            if num_train_saved < data_per_file:
                 print(f"    Warning: Only {num_train_saved} data points available for training set (less than {data_per_file}).")

            # Define output filename and path for training data
            train_output_filename = f"data_n{n_val}_k{k_val}_m{m_val}.pkl"
            train_output_path = os.path.join(full_output_dir_train, train_output_filename)

            # Save the training subset
            train_subset_to_save.to_pickle(train_output_path)
            print(f"    Saved training data to {train_output_path}")
            total_saved_train_count += num_train_saved
        else:
             print(f"    Warning: Not enough data ({available_rows}) to create even a partial training set (needs > {train_start_index}). Skipping training set.")


        # --- Process Validation Set ---
        val_subset_to_save = None
        if available_rows > val_start_index: # Need at least one row *after* the training set ends
            actual_val_end_index = min(val_end_index, available_rows)
            val_subset_to_save = subset_df.iloc[val_start_index:actual_val_end_index].copy()
            num_val_saved = len(val_subset_to_save)

            if num_val_saved > 0:
                print(f"    Selected {num_val_saved} random points for validation (from remaining data).")
                if num_val_saved < data_per_file:
                    print(f"    Warning: Only {num_val_saved} data points available for validation set (less than {data_per_file} in the target range).")

                # Define output filename and path for validation data
                val_output_filename = f"data_n{n_val}_k{k_val}_m{m_val}.pkl" # Same name, different dir
                val_output_path = os.path.join(full_output_dir_val, val_output_filename)

                # Save the validation subset
                val_subset_to_save.to_pickle(val_output_path)
                print(f"    Saved validation data to {val_output_path}")
                total_saved_val_count += num_val_saved
            else:
                 # This case might occur if available_rows is exactly data_per_file, so the slice is empty.
                 print(f"    No data points available in the range {val_start_index} to {actual_val_end_index-1} for validation set.")

        else:
            print(f"    Warning: Not enough data points ({available_rows}) to create a validation set (requires > {val_start_index}). Skipping validation set.")


    except Exception as e:
        print(f"    Error processing or saving combination n={n_val}, k={k_val}, m={m_val}: {e}")

print(f"\nProcessing complete.")
print(f"Total training data points saved across all files: {total_saved_train_count}")
print(f"Total validation data points saved across all files: {total_saved_val_count}") 