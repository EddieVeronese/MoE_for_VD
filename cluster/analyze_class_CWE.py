import pandas as pd

# Load dataset
df = pd.read_csv("bigvul_clustered.csv")

# Build contingency table
obs = pd.crosstab(df["class_CWE"], df["cluster"])

# Totals
row_totals = obs.sum(axis=1).values.reshape(-1, 1)  # class_CWE totals
col_totals = obs.sum(axis=0).values.reshape(1, -1)  # cluster totals
total = obs.values.sum()

# Calculate expected distribution if there were no correlation
exp = (row_totals @ col_totals) / total  # row x column / total

# Calculate observed / expected ratio
ratio = obs / exp
ratio = ratio.round(2)

# Display result
print("\n Observed / Expected ratio (ideally ≈ 1):")
print(ratio)

# Highlight where values are much higher (>1.5) or lower (<0.5)
print("\n Strong concentrations:")
for row in ratio.index:
    for col in ratio.columns:
        val = ratio.loc[row, col]
        if val > 1.5:
            print(f"→ {row} is over-represented in cluster {col}: {val}× expected")
        elif val < 0.5:
            print(f"→ {row} is under-represented in cluster {col}: {val}× expected")
