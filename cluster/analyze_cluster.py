import pandas as pd
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv("clustered.csv")

# Columns to ignore
excluded = ["processed_func", "cluster", "func_after", "func_before", "Summary", "Known Exploits", "commit_message", "files_changed"]
columns_to_check = [col for col in df.columns if col not in excluded]

print("\n Correlation analysis between clusters and label columns:\n")

# Analysis
for col in columns_to_check:
    try:
        unique_vals = df[col].nunique()
        if df[col].dtype == "object" or unique_vals < 30:
            # CATEGORICAL
            ct = pd.crosstab(df["cluster"], df[col])
            chi2, p, _, _ = chi2_contingency(ct)
            if p < 0.001:
                status = "🟢 very strong"
            elif p < 0.05:
                status = "🟡 moderate"
            else:
                status = "🔴 weak/none"
            print(f"→ {col.ljust(30)} | p = {p:.4f} | correlation: {status} | distinct values: {unique_vals}")
        else:
            # NUMERICAL
            grouped = df.groupby("cluster")[col].mean().std()
            if grouped > 1.0:
                status = "🟢 high variation"
            elif grouped > 0.2:
                status = "🟡 moderate variation"
            else:
                status = "🔴 almost uniform"
            print(f"→ {col.ljust(30)} | std = {grouped:.2f} | distribution: {status} | distinct values: {unique_vals}")
    except Exception as e:
        print(f"{col.ljust(30)} → Error: {e}")
