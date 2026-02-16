#!/usr/bin/env python3

import re
import csv
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolAlign

"""Compute ligand RMSD between docked poses and reference structures.
Assumes docked complexes are in "fegrow_result/complexes_pdbs" and reference structures are in "released_MERS-CoV_Mpro/mers_files".         
Ligand residue names are "UNL" in docked files and "UNK" in reference files.    
Outputs results to "ligand_rmsd_results.csv" with columns: docked_file, ligand_RMSD_A.      
"""

# ======================
# USER SETTINGS
# ======================
docked_dir = Path("fegrow_result/complexes_pdbs")
reference_dir = Path("released_MERS-CoV_Mpro/mers_files")

dock_resname = "UNL"
ref_resname  = "UNK"

out_csv = "all_ligands_rmsd_results.csv"
# ======================


def extract_ligand_block(resname, pdb_file):
    """Return only ligand block as PDB text"""
    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("HETATM") and resname in line:
                lines.append(line)
    return "".join(lines)


def pdb_block_to_mol(block):
    """Create RDKit mol WITHOUT bond guessing"""
    mol = Chem.MolFromPDBBlock(
        block,
        sanitize=False,
        removeHs=False
    )

    print("atoms before:", mol.GetNumAtoms())
    if mol is None:
        return None

    mol = Chem.RemoveHs(mol)  # ← reassignment REQUIRED
    print("atoms after:", mol.GetNumAtoms())  
    return mol


def compute_rmsd(dock_pdb, ref_pdb):

    dock_block = extract_ligand_block(dock_resname, dock_pdb)
    ref_block  = extract_ligand_block(ref_resname,  ref_pdb)

    dock = pdb_block_to_mol(dock_block)
    ref  = pdb_block_to_mol(ref_block)

    if dock is None or ref is None:
        raise ValueError("Failed reading ligand")

    # Check number of atoms before RMSD calculation
    # save number of atoms in a separate file for debugging
    with open("ligand_atom_counts_all_ligands.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([dock_pdb, ref_pdb, dock.GetNumAtoms(), ref.GetNumAtoms()])
    
    # symmetry-correct RMSD WITHOUT needing bonds
    return rdMolAlign.CalcRMS(dock, ref)


def get_molid(filename):
    m = re.search(r"_mol(\d+)\.pdb$", filename.name)
    return m.group(1) if m else None


# ======================
# MAIN
# ======================
results = []

dock_files = sorted(docked_dir.glob("rec_*_mol*.pdb"))
print(f"Found {len(dock_files)} docked files\n")

for dock_file in dock_files:

    molid = get_molid(dock_file)
    if not molid:
        continue

    ref_file = reference_dir / f"mers_{molid}.pdb"

    if not ref_file.exists():
        print(f"Missing reference for mol{molid}")
        continue

    try:
        rmsd = compute_rmsd(dock_file, ref_file)
        print(f"{dock_file.name:35s} {rmsd:7.3f}")
        results.append([dock_file.name, rmsd])

    except Exception as e:
        print(f"FAILED {dock_file.name}: {e}")


with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["docked_file", "ligand_RMSD_A"])
    writer.writerows(results)

print(f"\nSaved → {out_csv}")
