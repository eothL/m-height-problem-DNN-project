import numpy as np
import os

# Define the path to your .npz file
# Use an absolute path or a path relative to where you run the script
file_path = r'C:\Users\theo-\OneDrive\Documents\VS Code project\Deep learning\Project\data\m_height_data_n9_k4_m3_20250407_033832_9ff2ef8e.npz'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        # Load the .npz file
        data = np.load(file_path)

        # Print the names of the arrays stored in the file
        print("Arrays stored in the file:", data.files)

        # Print the name and shape of each array
        print("\nDetails of each array:")
        for key in data.files:
            print(f"- Name: '{key}', Shape: {data[key].shape}")

        # You can also access a specific array like this:
        # if 'your_array_name' in data.files:
        #    specific_array = data['your_array_name']
        #    print(f"\nExample: Accessing '{'your_array_name'}':\n{specific_array}")

        # Close the file handle
        data.close()

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


"""
Details of each array:
- Name: 'P_matrices', Shape: (250, 4, 5)
- Name: 'h_values', Shape: (250,)
- Name: 'n_val', Shape: (1,)
- Name: 'k_val', Shape: (1,)
- Name: 'm_val', Shape: (1,)
"""