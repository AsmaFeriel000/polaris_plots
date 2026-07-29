#!/usr/bin/env python3

import re
import csv
import math
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PyMOL (headless mode)
# -------------------------
import pymol
pymol.finish_launching(['pymol', '-qc'])
from pymol import cmd


"""Compute ligand RMSD and pocket RMSD between docked MERS-CoV Mpro complexes (FEgrow to ApoDock receptors)
and reference structures using RDKit’s CalcRMS for ligands and PyMOL rms_cur for pocket.
Heavy atoms only. NO superposition for pocket RMSD.
Outputs results to "complex_rmsd_results-5-residues.csv".

"""

output_dir = Path.cwd() / "output"
output_dir.mkdir(parents=True, exist_ok=True)

out_csv = output_dir / "complex_rmsd_results-5-residues.csv"

# ======================
# USER SETTINGS
# ======================
docked_dir = Path("../../03-Building-complexes/output")
reference_dir = Path("../../../released-receptor-structures/group2_MERS-CoV_Mpro/mers_files")

dock_resname = "UNL"
ref_resname  = "UNK"

POCKET_RESIDS = [
    192, 49, 25, 168, 167
]

# ======================


# ------------------------------------------------
# Ligand RMSD (RDKit symmetry aware)
# ------------------------------------------------
def extract_ligand_block(resname, pdb_file):
    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("HETATM") and resname in line:
                lines.append(line)
    return "".join(lines)


def ligand_rmsd(dock_pdb, ref_pdb):

    dock = Chem.MolFromPDBBlock(
        extract_ligand_block(dock_resname, dock_pdb),
        sanitize=False
    )
    dock = Chem.RemoveHs(dock)

    ref = Chem.MolFromPDBBlock(
        extract_ligand_block(ref_resname, ref_pdb),
        sanitize=False
    )
    ref = Chem.RemoveHs(ref)

    return rdMolAlign.CalcRMS(dock, ref)


# ------------------------------------------------
# Pocket RMSD (PyMOL rms_cur, NO superposition)
# ------------------------------------------------
def pocket_rmsd(dock_pdb, ref_pdb):

    cmd.reinitialize()

    cmd.load(str(dock_pdb), "dock")
    cmd.load(str(ref_pdb), "ref")

    resid_string = "+".join(str(r) for r in POCKET_RESIDS)

    dock_sel = f"dock and chain A and resi {resid_string} and sidechain and not hydro"
    ref_sel  = f"ref  and chain A and resi {resid_string} and sidechain and not hydro"

    print(cmd.count_atoms(dock_sel))
    print(cmd.count_atoms(ref_sel))

    # rms_cur does NOT superimpose
    rms = cmd.rms_cur(dock_sel, ref_sel)

    return rms


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def get_molid(filename):
    m = re.search(r"_mol(\d+)\.pdb$", filename.name)
    return m.group(1) if m else None


# ------------------------------------------------
# MAIN
# ------------------------------------------------
results = []

dock_files = sorted(docked_dir.glob("rec_*_mol*.pdb"))
print(f"Found {len(dock_files)} docked complexes\n")

for dock_file in dock_files:

    molid = get_molid(dock_file)
    if not molid:
        continue

    ref_file = reference_dir / f"mers_{molid}.pdb"

    if not ref_file.exists():
        print(f"Missing reference for mol{molid}")
        continue

    try:
        lig = ligand_rmsd(dock_file, ref_file)
        pocket = pocket_rmsd(dock_file, ref_file)

        print(f"{dock_file.name:30s}  ligand={lig:6.3f}  pocket={pocket:6.3f}")

        results.append([dock_file.name, lig, pocket])

    except Exception as e:
        print(f"FAILED {dock_file.name}: {e}")


with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["docked_file", "ligand_RMSD_A", "pocket_RMSD_A"])
    writer.writerows(results)

print(f"\nSaved → {out_csv}")

# -------------------------
# Load CSV
# -------------------------
csv_file = output_dir / "complex_rmsd_results-5-residues.csv"
df = pd.read_csv(csv_file)

# -------------------------
# Extract mol_id from filename
# rec_<x>_mol<y>.pdb
# -------------------------
def extract_molid(filename):
    m = re.search(r"_mol(\d+)\.pdb$", filename)
    return int(m.group(1)) if m else None

df["mol_id"] = df["docked_file"].apply(extract_molid)

# -------------------------
# Find lowest ligand RMSD per mol_id
# -------------------------
# idxmin gives the index of the lowest value per group
lowest_idx = df.groupby("mol_id")["ligand_RMSD_A"].idxmin()

# Create boolean mask
df["is_lowest_in_group"] = False
df.loc[lowest_idx, "is_lowest_in_group"] = True

# -------------------------
# Prepare data
# -------------------------
lig = df["ligand_RMSD_A"]
poc = df["pocket_RMSD_A"]
mask = df["is_lowest_in_group"]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6,6))

# Plot all points first (gray)
plt.scatter(poc, lig, color="gray", alpha=0.6)

# Overlay lowest-per-mol points (red)
plt.scatter(
    poc[mask],
    lig[mask],
    color="red",
    s=120,
    edgecolor="black",
    label="Lowest ligand RMSD per mol"
)

plt.xlabel("Pocket RMSD (Å)")
plt.ylabel("Ligand RMSD (Å)")
plt.title("Ligand vs Pocket RMSD")

plt.legend()
plt.grid(True)
plt.tight_layout()


plt.show()

print(f"Highlighted {mask.sum()} lowest-RMSD structures (one per mol group)")