import pandas as pd
import joblib

original_file = r"C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\split_data\data_n9_k4_m2.pkl" # Path to original split
validation_file = r"C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\split_data_validation\data_n9_k4_m2.pkl" # Path to validation split

try:
    print(f"Loading ORIGINAL file: {original_file}")
    df_orig = pd.read_pickle(original_file) # Assuming saved with pandas
    print("Original file HEAD:")
    print(df_orig.head())
    print(f"Original file index sample: {df_orig.index[:5].tolist()}")

    print("-" * 30)

    print(f"Loading VALIDATION file: {validation_file}")
    df_val = pd.read_pickle(validation_file) # Assuming saved with pandas
    print("Validation file HEAD:")
    print(df_val.head())
    print(f"Validation file index sample: {df_val.index[:5].tolist()}")

except Exception as e:
    print(f"Error during comparison: {e}")
