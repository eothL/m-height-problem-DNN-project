import pandas as pd
import joblib

# Define the file path
file_path = r"C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\split_data_validation\data_n10_k4_m6.pkl"
print('start')
# Load the data using joblib
try:
    data = joblib.load(file_path)
except Exception as e:
    print(f"Error loading with joblib: {e}")
    data = None

# Inspect the loaded data
if data is not None:
    print(f"Successfully loaded data of type: {type(data)}")

    # Check if it's a DataFrame and print head
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame shape: {data.shape}")
        print("First 5 rows:")
        print(data.head())

        # Check if 'P' column exists and print the first element
        if 'P' in data.columns:
            print("\nFirst sample from 'P' column:")
            first_p_element = data['P'].iloc[0]
            print(first_p_element)
            # Optional: Try to evaluate if it looks like a list string
            # import ast
            # try:
            #     print("\nEvaluated as list:")
            #     print(ast.literal_eval(first_p_element))
            # except (ValueError, SyntaxError):
            #     print("(Could not evaluate the string as a Python literal)")
        else:
            print("\nColumn 'P' not found in the DataFrame.")

    # Check if it's a Series and print head
    elif isinstance(data, pd.Series):
        print(f"Series length: {len(data)}")
        print("First 5 items:")
        print(data.head())
    # Fallback for other types
    elif isinstance(data, list):
        print(f"Total items: {len(data)}")
        print("First 5 items:")
        try:
            for i in range(min(5, len(data))):
                print(data[i])
        except Exception as e:
            print(f"Could not print list elements: {e}")
    elif isinstance(data, dict):
        # Keep dict check in case data format changes, but make it less prominent
        print(f"Data is a dictionary with {len(data)} keys.")
        print("First 5 keys and value types:")
        try:
            for i, (k, v) in enumerate(data.items()):
                if i >= 5:
                    break
                print(f"{k}: {type(v)}")
        except Exception as e:
            print(f"Could not print dict items: {e}")
    else:
        # Handle other types like NumPy arrays if needed
        print("\nLoaded data is not a DataFrame, Series, list, or dict.")
        # This might be the matrix 'P' if the loaded data itself is the matrix
        try:
             # Attempt to print the first few elements if possible
             print("Data preview (first 5 elements/rows):")
             print(data[:5]) # Common slicing for array-like objects
        except TypeError:
             print("Cannot preview this data type directly via slicing.")
             # Try printing the whole object if slicing fails
             try:
                 print("Attempting to print the full data:")
                 print(data)
             except Exception as e:
                 print(f"Could not print data: {e}")


