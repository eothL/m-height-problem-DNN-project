import pickle
import pandas as pd
import numpy as np
import sys

# --- Configuration ---
# <<< 1. PLEASE PROVIDE THE PATH TO YOUR PKL FILE HERE >>>
pickle_file_path = r'C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\results_dataframe(1)\results_dataframe.pkl' # <--- CHANGE THIS
output_csv_path = 'nkm_hm_p_matrix_sampled.csv'
num_samples_per_combination = 10

# Increase field size limit for CSV writing if matrices are large
# Adjust if needed, this allows fields up to ~1MB
try:
    # Set a large field size limit for CSV processing, needed for potentially long flattened matrices
    # Using sys.maxsize might be excessive and platform-dependent, using a large fixed number instead.
    # Adjust this value if you encounter _csv.Error: field larger than field limit
    large_field_limit = 10 * 1024 * 1024 # 10 MB
    import csv
    csv.field_size_limit(large_field_limit)
    print(f"CSV field size limit set to {large_field_limit} bytes")
except ModuleNotFoundError:
    print("Warning: 'csv' module not found, cannot set field size limit.")
except Exception as e:
    print(f"Warning: Could not set CSV field size limit: {e}")


# The 21 target combinations
target_combinations_set = {
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3),
    (10, 4, 2), (10, 4, 3), (10, 4, 4), (10, 4, 5), (10, 4, 6),
    (10, 5, 2), (10, 5, 3), (10, 5, 4), (10, 5, 5),
    (10, 6, 2), (10, 6, 3), (10, 6, 4)
}

# --- Load Data ---
try:
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Successfully loaded data from {pickle_file_path}")

    # Assume data is a pandas DataFrame based on your snippet
    if not isinstance(data, pd.DataFrame):
         # Add conversion logic here if it's a list of dicts/tuples etc.
         raise TypeError("Loaded data is not a pandas DataFrame. Please adapt the script or ensure the PKL contains a DataFrame.")

    df = data
    # Rename 'result' to 'h_m' for clarity, adjust if your column name is different
    if 'result' in df.columns:
        df.rename(columns={'result': 'h_m'}, inplace=True)
        print("Renamed column 'result' to 'h_m'.")

    # Ensure required columns exist
    required_cols = ['n', 'k', 'm', 'h_m', 'P']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

    print(f"DataFrame loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")

    # --- Filter for Target Combinations ---
    # Create a tuple column for easy filtering
    df['nkm_tuple'] = df.apply(lambda row: (row['n'], row['k'], row['m']), axis=1)
    filtered_df = df[df['nkm_tuple'].isin(target_combinations_set)].copy() # Use .copy() to avoid SettingWithCopyWarning

    if filtered_df.empty:
        print("No rows found matching the target (n, k, m) combinations.")
        sys.exit()

    print(f"Found {len(filtered_df)} rows matching the target combinations.")

    # --- Sampling ---
    # <<< 2. CONFIRM SAMPLING: Set sample=True to sample, False to take all matching rows >>>
    should_sample = True # <--- CHANGE TO False IF YOU WANT ALL MATCHING ROWS

    if should_sample:
        print(f"Sampling up to {num_samples_per_combination} rows per combination...")
        # Use random_state for reproducibility
        sampled_df = filtered_df.groupby(['n', 'k', 'm'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_samples_per_combination), random_state=42)
        )
        print(f"Sampled {len(sampled_df)} rows.")
    else:
        print("Taking all matching rows (sampling disabled).")
        sampled_df = filtered_df

    if sampled_df.empty:
        print("No rows selected after filtering (and possibly sampling).")
        sys.exit()


    # --- Process P Matrix and Flatten ---
    processed_rows = []
    max_p_len = 0 # Track max flattened P length for column naming

    print("Processing rows to reshape and flatten P matrices...")
    for index, row in sampled_df.iterrows():
        n, k, m, h_m = row['n'], row['k'], row['m'], row['h_m']
        p_list = row['P']

        # Validate P is a list or array-like
        if not isinstance(p_list, (list, np.ndarray)):
            print(f"Warning: Skipping row index {index} for combination ({n},{k},{m}). 'P' is not a list or numpy array (type: {type(p_list)}).")
            continue

        expected_len = n * (n - k)
        actual_len = len(p_list)

        # Check length for reshaping
        if actual_len != expected_len:
            print(f"Warning: Skipping row index {index} for combination ({n},{k},{m}). 'P' list has length {actual_len}, but expected n*(n-k) = {n}*({n}-{k}) = {expected_len}.")
            continue

        try:
            # Reshape and flatten
            p_matrix = np.array(p_list).reshape((n, n - k))
            p_flat = p_matrix.flatten()
            max_p_len = max(max_p_len, len(p_flat))

            # Create row data: n, k, m, h_m, followed by flattened P elements
            row_data = {'n': n, 'k': k, 'm': m, 'h_m': h_m}
            # Add flattened P elements with standard names p_0, p_1, ...
            for i, p_val in enumerate(p_flat):
                row_data[f'p_{i}'] = p_val
            processed_rows.append(row_data)

        except ValueError as e:
             print(f"Error reshaping P for row index {index} ({n},{k},{m}): {e}. Skipping row.")
        except Exception as e:
             print(f"Unexpected error processing row index {index} ({n},{k},{m}): {e}. Skipping row.")


    # --- Create Final DataFrame and Save ---
    if not processed_rows:
        print("No rows were successfully processed after reshaping P matrix.")
        sys.exit()

    # Create final DataFrame from the list of dictionaries
    final_df = pd.DataFrame(processed_rows)

    # Ensure consistent columns, fill missing p_i columns with NaN if needed (though should not happen with current logic)
    # Generate all expected p_i column names based on the max length found
    p_cols = [f'p_{i}' for i in range(max_p_len)]
    all_cols = ['n', 'k', 'm', 'h_m'] + p_cols
    # Reindex to ensure all columns are present and in order, fill missing with NaN
    final_df = final_df.reindex(columns=all_cols)


    print(f"Created final DataFrame with {len(final_df)} rows and {len(final_df.columns)} columns.")
    print(f"Saving final DataFrame to {output_csv_path}...")
    try:
        final_df.to_csv(output_csv_path, index=False, lineterminator='\n') # Use standard line terminator
        print(f"Successfully saved data to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        print("This might be due to very large fields (flattened matrices). Check the CSV field size limit section at the top of the script.")


except FileNotFoundError:
    print(f"Error: The file {pickle_file_path} was not found. Please check the path.")
except ImportError as e:
    print(f"Error: Missing required library. Please install pandas and numpy (`pip install pandas numpy`). Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Please check the pickle file path, its contents, required columns ('n', 'k', 'm', 'result'/'h_m', 'P'), and the data types.")
