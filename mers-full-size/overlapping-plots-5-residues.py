import re
import csv
import math
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# OVERLAY PLOT: FEgrow (Script 1) vs Submitted (Script 2)
# ==========================================================

# Load both datasets
df_fegrow = pd.read_csv("complex_rmsd_results-5-residues.csv")
df_sub    = pd.read_csv("complex_rmsd_results-submitted-vs-ref-5-residues.csv")

# --- Extract mol_id for FEgrow ---
def extract_molid_fegrow(filename):
    m = re.search(r"_mol(\d+)\.pdb$", filename)
    return int(m.group(1)) if m else None

df_fegrow["mol_id"] = df_fegrow["docked_file"].apply(extract_molid_fegrow)

# --- Extract mol_id for Submitted ---
def extract_molid_sub(filename):
    m = re.search(r"complex-MERS-mol(\d+)\.pdb$", filename)
    return int(m.group(1)) if m else None

df_sub["mol_id"] = df_sub["docked_file"].apply(extract_molid_sub)

# --- Lowest ligand RMSD per mol group ---
lowest_idx_fegrow = df_fegrow.groupby("mol_id")["ligand_RMSD_A"].idxmin()
lowest_idx_sub    = df_sub.groupby("mol_id")["ligand_RMSD_A"].idxmin()

df_fegrow["is_lowest"] = False
df_sub["is_lowest"]    = False

df_fegrow.loc[lowest_idx_fegrow, "is_lowest"] = True
df_sub.loc[lowest_idx_sub, "is_lowest"]       = True

# ----------------------------------------------------------
# Percentage of BLUE points (FEgrow lowest-per-mol) with POCKET RMSD < 2 Å
# ----------------------------------------------------------
blue_df = df_fegrow[df_fegrow["is_lowest"]]

total_blue = len(blue_df)
blue_below_2 = (blue_df["pocket_RMSD_A"] < 2).sum()

blue_percentage = (blue_below_2 / total_blue) * 100 if total_blue > 0 else 0

print(f"\nBLUE points (FEgrow lowest-per-mol): {total_blue}")
print(f"BLUE points with POCKET RMSD < 2 Å: {blue_below_2}")
print(f"BLUE percentage: {blue_percentage:.2f}%")
# ----------------------------------------------------------
# Percentage of lowest-per-mol with ligand RMSD < 2 Å
# ----------------------------------------------------------
lowest_fegrow = df_fegrow[df_fegrow["is_lowest"]]

total = len(lowest_fegrow)
below_2 = (lowest_fegrow["ligand_RMSD_A"] < 2).sum()

percentage = (below_2 / total) * 100

print(f"Total lowest-per-molecule points: {total}")
print(f"With ligand RMSD < 2 Å: {below_2}")
print(f"Percentage: {percentage:.2f}%")

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------
plt.figure(figsize=(7,7))

# FEgrow (gray + blue highlight)
plt.scatter(
    df_fegrow["pocket_RMSD_A"],
    df_fegrow["ligand_RMSD_A"],
    color="lightgray",
    alpha=0.5,
    label="ApoDock receptors"
)

plt.scatter(
    df_fegrow.loc[df_fegrow["is_lowest"], "pocket_RMSD_A"],
    df_fegrow.loc[df_fegrow["is_lowest"], "ligand_RMSD_A"],
    color="blue",
    s=120,
    edgecolor="black",
    label="ApoDock receptors (lowest per mol)"
)

# Submitted (orange + red highlight)
#plt.scatter(
#    df_sub["pocket_RMSD_A"],
#    df_sub["ligand_RMSD_A"],
#    color="orange",
#    alpha=0.4,
#    label="Submitted (all)"
#)

#plt.scatter(
#    df_sub.loc[df_sub["is_lowest"], "pocket_RMSD_A"],
#    df_sub.loc[df_sub["is_lowest"], "ligand_RMSD_A"],
#    color="red",
#    s=120,
#    edgecolor="black",
#    label="Original Submission"
#)

plt.scatter(
    df_sub["pocket_RMSD_A"],
    df_sub["ligand_RMSD_A"],
    color="red",
    s=120,
    edgecolor="black",
    label="Original Submission"
)

# Specific points to highlight
point1 = (4.770911805362309, 2.1209278106689453)
point2 = (0.553152343528901, 1.8176279067993164)

# Highlight point 1 (e.g., green)
plt.scatter(
    point1[1], point1[0],  # (x = pocket, y = ligand)
    color="orange",
    s=150,
    edgecolor="black",
    label="Mol 02 originally"
)

# Highlight point 2 (e.g., purple)
plt.scatter(
    point2[1], point2[0],
    color="yellow",
    s=150,
    edgecolor="black",
    label="Mol 02 after side-chain modelling"
)

plt.xlabel("Pocket RMSD (Å)")
plt.ylabel("Ligand RMSD (Å)")
plt.title("Ligand vs Pocket RMSD for MERS-CoV Mpro")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("overlay_ligand_vs_pocket_rmsd.png", dpi=300)
plt.show()

