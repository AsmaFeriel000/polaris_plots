import os
import glob
import re
import logging

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

"""
Compute pocket RMSD and ligand RMSD between docked MERS-CoV Mpro complexes
and reference structures using RDKit’s CalcRMS (no alignment).
Heavy atoms only. Logs skipped files with reasons.
"""

# --------------------
# Configuration
# --------------------
query_dir = "fegrow_result/complexes_pdbs"
ref_dir = "released_MERS-CoV_Mpro/mers_files"
LOG_FILE = "rmsd_skipped.log"

REF_LIG = "UNK"  # reference ligand name
REC_LIG = "UNL"  # docked ligand name

POCKET_RESIDS = [
    25, 26, 27, 28, 41, 49, 143, 144, 145, 146,
    148, 166, 167, 168, 169, 175, 190, 191, 192
]

# --------------------
# Setup logging
# --------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# --------------------
# Helper functions
# --------------------
def load_universe(pdb_file):
    """Load a PDB file with MDAnalysis."""
    try:
        return mda.Universe(pdb_file)
    except Exception as e:
        logging.error(f"Failed to load {pdb_file}: {e}")
        return None

def get_heavy_positions(universe, resid_list=None, resnames=None):
    """Return heavy atom positions for given residue IDs or residue names."""
    if resid_list:
        sel_str = "resid " + " ".join(map(str, resid_list)) + " and not name H*"
    elif resnames:
        sel_str = "resname " + " ".join(resnames) + " and not name H*"
    else:
        return None
    atoms = universe.select_atoms(sel_str)
    return atoms.positions if len(atoms) > 0 else None

def to_rdkit_mol(positions):
    """
    Convert an array of 3D positions into a dummy RDKit molecule.
    Each atom is a carbon; only coordinates are used for RMSD.
    """
    positions = np.asarray(positions, dtype=np.float64)  # ensure proper float type
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Positions array must be Nx3, got shape {positions.shape}")

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(positions))
    for i, pos in enumerate(positions):
        mol.AddAtom(Chem.Atom("C"))
        # Explicitly cast each coordinate to Python float
        conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf)
    return mol

def calculate_rmsd_rdkit(ref_positions, rec_positions):
    """Compute RMSD with RDKit CalcRMS (no alignment)."""
    ref_mol = to_rdkit_mol(ref_positions)
    rec_mol = to_rdkit_mol(rec_positions)
    return AllChem.CalcRMS(ref_mol, rec_mol)

# --------------------
# Main loop
# --------------------
results = []

for rec_file in sorted(glob.glob(os.path.join(query_dir, "*.pdb"))):
    base_name = os.path.basename(rec_file)

    # Extract reference number from "_molX.pdb"
    match = re.search(r"_mol(\d+)\.pdb$", base_name)
    if not match:
        logging.info(f"SKIP (bad name) {base_name}")
        continue
    ref_num = match.group(1)

    # Reference file
    ref_file = os.path.join(ref_dir, f"mers_{ref_num}.pdb")
    if not os.path.exists(ref_file):
        logging.info(f"SKIP (no reference) {base_name}")
        continue

    # Load structures
    ref_uni = load_universe(ref_file)
    rec_uni = load_universe(rec_file)
    if ref_uni is None or rec_uni is None:
        logging.info(f"SKIP (load fail) {base_name}")
        continue

    # Select pocket heavy atoms
    ref_pocket_pos = get_heavy_positions(ref_uni, resid_list=POCKET_RESIDS)
    rec_pocket_pos = get_heavy_positions(rec_uni, resid_list=POCKET_RESIDS)

    # Select ligand heavy atoms
    ref_lig_pos = get_heavy_positions(ref_uni, resnames=[REF_LIG])
    rec_lig_pos = get_heavy_positions(rec_uni, resnames=[REC_LIG])

    # Skip if selection failed
    if ref_pocket_pos is None or rec_pocket_pos is None:
        logging.info(f"SKIP (pocket selection fail) {base_name}")
        continue
    if len(ref_pocket_pos) != len(rec_pocket_pos):
        logging.info(f"SKIP (pocket atom count mismatch) {base_name}, ref={len(ref_pocket_pos)}, rec={len(rec_pocket_pos)}")
        continue

    if ref_lig_pos is None or rec_lig_pos is None:
        logging.info(f"SKIP (ligand selection fail) {base_name}")
        continue
    if len(ref_lig_pos) != len(rec_lig_pos):
        logging.info(f"SKIP (ligand atom count mismatch) {base_name}, ref={len(ref_lig_pos)}, rec={len(rec_lig_pos)}")
        continue

    # Compute RMSDs
    pocket_rmsd = calculate_rmsd_rdkit(ref_pocket_pos, rec_pocket_pos)
    lig_rmsd = calculate_rmsd_rdkit(ref_lig_pos, rec_lig_pos)

    results.append({
        "Reference_File": os.path.basename(ref_file),
        "Docked_Complex": base_name,
        "Ligand_RMSD_A": lig_rmsd,
        "Pocket_RMSD_A": pocket_rmsd
    })

# --------------------
# Save results
# --------------------
df = pd.DataFrame(results)
csv_file = "pocket_ligand_rmsd_rdkit.csv"
df.to_csv(csv_file, index=False)
print(f"Results saved to {csv_file}")

# --------------------
# Plot correlation
# --------------------
plt.figure(figsize=(6, 6))
plt.scatter(df["Ligand_RMSD_A"], df["Pocket_RMSD_A"], alpha=0.7, color="blue")
plt.xlabel("Ligand RMSD (Å, RDKit no alignment)")
plt.ylabel("Pocket RMSD (Å, RDKit no alignment)")
plt.title("Ligand vs Pocket RMSD (RDKit no alignment)")
plt.grid(True)
max_val = max(df["Ligand_RMSD_A"].max(), df["Pocket_RMSD_A"].max())
plt.plot([0, max_val], [0, max_val], "r--")
plt.tight_layout()
plt.savefig("rdkit_ligand_vs_pocket_rmsd.png", dpi=300)
plt.show()

# --------------------
# Correlation
# --------------------
if len(df) > 0:
    corr = np.corrcoef(df["Ligand_RMSD_A"], df["Pocket_RMSD_A"])[0,1]
    print(f"Pearson correlation: {corr:.2f}")

print(f"Skipped files are logged in {LOG_FILE}")