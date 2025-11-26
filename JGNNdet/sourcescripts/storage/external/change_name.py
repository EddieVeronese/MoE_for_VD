import pandas as pd

# Path to the dataset
file_path = "sourcescripts/storage/external/j_processed_CWE-284_dataset.csv"

# Dictionary with columns to rename
# format: {"old_name": "new_name"}
rinomine = {
    "processed_func": "func_before",
    "vul_func_with_fix": "func_after",
    "target": "vul",
}

# Load the dataset
df = pd.read_csv(file_path)

print("Original columns:", df.columns.tolist())

# Rename the specified columns
df = df.rename(columns=rinomine)

print("Columns after renaming:", df.columns.tolist())

# Overwrites the same file
df.to_csv(file_path, index=False)

print(f"Columns renamed correctly in {file_path}")
