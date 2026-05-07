#import pandas as pd

# Load your CSV
#df = pd.read_csv("complex_rmsd_results-3-residues.csv")

# Extract molecule ID (recX) from docked_file
#df["molecule"] = df["docked_file"].str.extract(r"(rec_\d+)")

# For each molecule, get the row with lowest ligand RMSD
#idx = df.groupby("molecule")["pocket_RMSD_A"].idxmin()
#best_df = df.loc[idx]

# Count how many are < 2 Å
#total = len(best_df)
#below_2 = (best_df["pocket_RMSD_A"] < 2).sum()

#percentage = (below_2 / total) * 100

#print(f"Total molecules: {total}")
#print(f"With ligand RMSD < 2 Å: {below_2}")
#print(f"Percentage: {percentage:.2f}%")

import pandas as pd
import re

# Load your CSV (same file as Script 1)
df = pd.read_csv("complex_rmsd_results-3-residues.csv")

# ----------------------------------------------------------
# STEP 0: Extract mol_id EXACTLY as in plotting script
# ----------------------------------------------------------
def extract_molid(filename):
    m = re.search(r"_mol(\d+)\.pdb$", filename)
    return int(m.group(1)) if m else None

df["mol_id"] = df["docked_file"].apply(extract_molid)

# Remove rows where extraction failed
df = df.dropna(subset=["mol_id"])

# ----------------------------------------------------------
# STEP 1: BLUE points = lowest ligand RMSD per mol_id
# (THIS now matches Script 1 exactly)
# ----------------------------------------------------------
idx = df.groupby("mol_id")["ligand_RMSD_A"].idxmin()
blue_df = df.loc[idx]

# ----------------------------------------------------------
# STEP 2: compute % with pocket RMSD < 2 Å
# ----------------------------------------------------------
total_blue = len(blue_df)
below_2 = (blue_df["pocket_RMSD_A"] < 2).sum()

percentage = (below_2 / total_blue) * 100 if total_blue > 0 else 0

# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
print(f"Total blue points (lowest ligand RMSD per mol_id): {total_blue}")
print(f"Blue points with pocket RMSD < 2 Å: {below_2}")
print(f"Percentage: {percentage:.2f}%")