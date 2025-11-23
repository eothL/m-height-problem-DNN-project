import pandas as pd
import os
import sys
import joblib 

# --- Configuration ---
# Path to the ORIGINAL large dataset
original_large_pickle_path = r"C:\\Users\\theo-\\OneDrive\\Documents\\VS Code project\\Deep learning\\Project\\results_dataframe(1)\\results_dataframe.pkl"

# Directory where the new small (10-sample) random test files will be saved
output_dir = "split_data_small_test"
# Number of data points to select RANDOMLY for each small file
data_per_file = 10
# Use a fixed random state for reproducible sampling
random_seed = 42

# The specific (n, k, m) combinations to process
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
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Create full path for the output directory relative to the script
full_output_dir = os.path.join(script_dir, output_dir)

print(f"Loading original large dataset from: {original_large_pickle_path}")
print(f"Output directory for small random samples: {full_output_dir}")

# Load the entire original dataset
df_large = None
try:
    loaded_data = joblib.load(original_large_pickle_path)
    print("Original data loaded successfully using joblib.")
    if isinstance(loaded_data, pd.DataFrame):
        df_large = loaded_data
    else:
        try:
            df_large = pd.DataFrame(loaded_data)
            print("Converted loaded data to pandas DataFrame.")
        except Exception as convert_e:
            print(f"Error: Could not convert loaded data to DataFrame: {convert_e}")
            sys.exit(1)

    print(f"Original DataFrame shape: {df_large.shape}")
    # Check for required columns needed for filtering
    required_cols = ['n', 'k', 'm']
    if not all(col in df_large.columns for col in required_cols):
        print(f"Error: Original DataFrame missing one or more required columns: {required_cols}")
        print(f"Available columns: {df_large.columns.tolist()}")
        sys.exit(1)
    # Convert relevant columns to numeric for reliable filtering
    try:
        df_large['n'] = pd.to_numeric(df_large['n'], errors='coerce')
        df_large['k'] = pd.to_numeric(df_large['k'], errors='coerce')
        df_large['m'] = pd.to_numeric(df_large['m'], errors='coerce')
        # Drop rows where conversion failed if necessary, or handle NaNs in filtering
        df_large.dropna(subset=['n', 'k', 'm'], inplace=True) # Drop rows where n, k, or m couldn't be made numeric
        print("Converted n, k, m columns to numeric.")
    except KeyError as ke:
        # Should be caught by the check above, but belt-and-suspenders
        print(f"Error: Missing column during numeric conversion: {ke}")
        sys.exit(1)
    except Exception as num_e:
        print(f"Error during numeric conversion: {num_e}")
        sys.exit(1)

except FileNotFoundError:
    print(f"Error: Original input file not found at {original_large_pickle_path}")
    sys.exit(1)
except ImportError:
     print("Error: Required library (pandas or joblib) not found. Please install them.")
     sys.exit(1)
except Exception as e:
    print(f"Error loading or processing original large file: {e}")
    sys.exit(1)

if df_large is None:
    print("Error: Failed to load and prepare the large DataFrame. Exiting.")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(full_output_dir, exist_ok=True)

print(f"\nProcessing {len(combinations)} combinations to create small random test files...")

total_saved_count = 0
for n_val, k_val, m_val in combinations:
    print(f"  Processing combination: n={n_val}, k={k_val}, m={m_val}")

    # Construct the output filename
    output_filename = f"data_n{n_val}_k{k_val}_m{m_val}.pkl"
    output_path = os.path.join(full_output_dir, output_filename)

    try:
        # Filter the large DataFrame for the current combination
        subset_df = df_large[(df_large['n'] == n_val) & (df_large['k'] == k_val) & (df_large['m'] == m_val)]

        available_rows = len(subset_df)
        print(f"    Found {available_rows} matching rows in the original dataset.")

        if available_rows == 0:
            print("    Warning: No data found for this combination. Skipping file creation.")
            continue
        elif available_rows < data_per_file:
            print(f"    Warning: Only {available_rows} rows available (less than {data_per_file}). Taking all available rows.")
            subset_to_save = subset_df.copy() # No sampling needed, take all
        else:
            # Randomly sample the desired number of data points
            print(f"    Randomly sampling {data_per_file} rows...")
            subset_to_save = subset_df.sample(n=data_per_file, random_state=random_seed).copy()

        # Reset the index for the small subset
        subset_to_save.reset_index(drop=True, inplace=True)

        # Save the small random subset to the output directory
        subset_to_save.to_pickle(output_path)
        print(f"    Saved {len(subset_to_save)} data points to {output_path}")
        total_saved_count += len(subset_to_save)

    except Exception as e:
        print(f"    Error processing combination n={n_val}, k={k_val}, m={m_val}: {e}")

print(f"\nProcessing complete. Total data points saved across all small random test files: {total_saved_count}") 