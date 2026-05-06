import pandas as pd

# Load your CSV
df = pd.read_csv("complex_rmsd_results-5-residues.csv")

# Extract molecule ID (recX) from docked_file
df["molecule"] = df["docked_file"].str.extract(r"(rec_\d+)")

# For each molecule, get the row with lowest ligand RMSD
idx = df.groupby("molecule")["ligand_RMSD_A"].idxmin()
best_df = df.loc[idx]

# Count how many are < 2 Å
total = len(best_df)
below_2 = (best_df["ligand_RMSD_A"] < 2).sum()

percentage = (below_2 / total) * 100

print(f"Total molecules: {total}")
print(f"With ligand RMSD < 2 Å: {below_2}")
print(f"Percentage: {percentage:.2f}%")
