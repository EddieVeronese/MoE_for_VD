import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
df = pd.read_csv("bigvul_clustered.csv")
df["CWE ID"] = df["CWE ID"].astype(str)
df["cluster"] = df["cluster"].astype(str)

# Build CWE ID vs cluster table
obs = pd.crosstab(df["CWE ID"], df["cluster"])

# Normalize rows (relative frequency of CWEs in clusters)
obs_norm = obs.div(obs.sum(axis=1), axis=0)

# Cluster distribution vectors across clusters (based on diversity)
kmeans = KMeans(n_clusters=8, random_state=42)
cwe_groups = kmeans.fit_predict(obs_norm)

# Map CWE ID → group
group_labels = [f"group_{i}" for i in cwe_groups]
cwe_to_group = dict(zip(obs.index, group_labels))
df["CWE_group"] = df["CWE ID"].map(cwe_to_group)

# Build new contingency table group vs cluster
group_obs = pd.crosstab(df["CWE_group"], df["cluster"])
group_row_totals = group_obs.sum(axis=1).values.reshape(-1, 1)
group_col_totals = group_obs.sum(axis=0).values.reshape(1, -1)
group_total = group_obs.values.sum()
group_exp = (group_row_totals @ group_col_totals) / group_total
group_ratio = (group_obs / group_exp).round(2)

# Display results
print("\n Observed/Expected ratio after merging CWE IDs into 8 groups (for maximum inter-cluster diversity):")
print(group_ratio)

# Highlight strong concentrations
print("\n Strong concentrations after grouping:")
for row in group_ratio.index:
    for col in group_ratio.columns:
        val = group_ratio.loc[row, col]
        if val > 1.5:
            print(f"→ {row} over-represented in cluster {col}: {val}× expected")
        elif val < 0.5:
            print(f"→ {row} under-represented in cluster {col}: {val}× expected")
