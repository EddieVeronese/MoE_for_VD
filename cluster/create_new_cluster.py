import pandas as pd
import os

# --- Step 1: Extract mapping CWE ID → dominant cluster from the file ---

df_clustered = pd.read_csv("clustered.csv")
df_clustered["CWE-ID"] = df_clustered["CWE-ID"].astype(str)
df_clustered["cluster"] = df_clustered["cluster"].astype(str)

# Create dictionary CWE ID → dominant cluster
obs = pd.crosstab(df_clustered["CWE-ID"], df_clustered["cluster"])
dominant_cluster = obs.idxmax(axis=1)
cwe_to_group = dominant_cluster.to_dict()

# --- Step 2: Function to process each split ---

def process_split(split_name):
    input_file = f"datasets/j_processed_{split_name}.csv"
    output_file = f"new_j_processed_{split_name}.csv"

    # Load and add new_class_CWE
    df = pd.read_csv(input_file)
    df["CWE-ID"] = df["CWE-ID"].astype(str)
    df["new_class_CWE"] = df["CWE-ID"].map(cwe_to_group).fillna("unknown")

    # Save updated full file
    df.to_csv(output_file, index=False)
    print(f"\n Saved: {output_file}")

    # Print count of unique CWE IDs per cluster
    print(f"\n Unique CWE ID count per cluster in {split_name}:")
    cluster_counts = df.groupby("new_class_CWE")["CWE-ID"].nunique()
    print(cluster_counts)

    # Create sub-datasets split by new_class_CWE (only if not unknown)
    clusters = df["new_class_CWE"].unique()
    clusters = [c for c in clusters if c != "unknown"]

    for c in clusters:
        sub_df = df[df["new_class_CWE"] == c]
        sub_df.to_csv(f"j_{split_name}_cluster_{c}.csv", index=False)
        print(f" Saved subset: j_{split_name}_cluster_{c}.csv ({len(sub_df)} rows)")

# --- Step 3: Apply to train / val / test ---

for split in ["train", "val", "test"]:
    process_split(split)
