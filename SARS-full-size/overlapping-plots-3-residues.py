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
df_fegrow = pd.read_csv("complex_rmsd_results-3-residues.csv")
df_sub    = pd.read_csv("complex_rmsd_results-submitted-vs-ref-3-residues.csv")
                        

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

plt.xlabel("Pocket RMSD (Å)")
plt.ylabel("Ligand RMSD (Å)")
plt.title("Ligand vs Pocket RMSD for SARS-CoV-2 Mpro")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("overlay_ligand_vs_pocket_rmsd.png", dpi=300)
plt.show()