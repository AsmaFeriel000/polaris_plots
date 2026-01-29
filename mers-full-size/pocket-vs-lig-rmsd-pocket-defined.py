import os
import glob
import re
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Compute RMSD values between docked MERS-CoV Mpro complexes and reference structures.
Ligand-only RMSD and pocket+ligand RMSD are computed using **heavy atoms only**.
Docked files rec_y_molx.pdb are matched to reference mers_x.pdb (number after _mol).
Results are saved to a CSV and a correlation plot is generated.
"""

# --------------------
# Paths
# --------------------
query_dir = "fegrow_result/complexes_pdbs"
ref_dir = "released_MERS-CoV_Mpro/mers_files"
REF_LIG = "UNK"
REC_LIG = "UNL"

# Pocket residues
POCKET_RESIDS = [25, 26, 27, 28, 41, 49, 143, 144, 145, 146, 148, 166, 167, 168, 169, 175, 190, 191, 192]

# --------------------
# Functions
# --------------------
def load_universe(pdb_file):
    return mda.Universe(pdb_file)

def get_selection_heavy(universe, resnames=None, resid_list=None):
    """
    Select heavy atoms (non-hydrogen) by residue name or residue ID.
    """
    if resnames:
        sel_str = "resname " + " ".join(resnames) + " and not name H*"
    elif resid_list:
        sel_str = "resid " + " ".join(map(str, resid_list)) + " and not name H*"
    else:
        raise ValueError("Either resnames or resid_list must be provided.")
    return universe.select_atoms(sel_str)

def compute_rmsd(ref_atoms, mob_atoms, align=True):
    if align:
        return rms.rmsd(mob_atoms.positions, ref_atoms.positions, superposition=True)
    else:
        return rms.rmsd(mob_atoms.positions, ref_atoms.positions, superposition=False)

# --------------------
# Loop over docking results
# --------------------
results = []

for rec_file in glob.glob(os.path.join(query_dir, "*.pdb")):
    base_name = os.path.basename(rec_file)
    
    # Extract reference number from "_molX" part
    match = re.search(r"_mol(\d+)\.pdb$", base_name)
    if not match:
        print(f"Cannot parse reference number from {base_name}, skipping")
        continue
    ref_num = match.group(1)
    
    # Reference file = mers_<ref_num>.pdb
    ref_file = os.path.join(ref_dir, f"mers_{ref_num}.pdb")
    if not os.path.exists(ref_file):
        print(f"Reference file not found for {rec_file} -> expected {ref_file}, skipping")
        continue
    
    # Load structures
    ref_uni = load_universe(ref_file)
    rec_uni = load_universe(rec_file)
    
    # Select pocket and ligand heavy atoms
    ref_pocket = get_selection_heavy(ref_uni, resid_list=POCKET_RESIDS)
    rec_pocket = get_selection_heavy(rec_uni, resid_list=POCKET_RESIDS)
    
    ref_ligand = get_selection_heavy(ref_uni, resnames=[REF_LIG])
    rec_ligand = get_selection_heavy(rec_uni, resnames=[REC_LIG])
    
    # Combine pocket + ligand
    ref_combined = ref_pocket + ref_ligand
    rec_combined = rec_pocket + rec_ligand
    
    # Skip if atom counts mismatch
    if len(ref_combined) != len(rec_combined):
        print(f"Skipping {base_name}: atom count mismatch in pocket+ligand")
        print(f"Ref atoms: {len(ref_combined)}, Rec atoms: {len(rec_combined)}")
        continue
    if len(ref_ligand) != len(rec_ligand):
        print(f"Skipping {base_name}: atom count mismatch in ligand")
        continue
    
    # Compute RMSDs
    rmsd_pocket = compute_rmsd(ref_combined, rec_combined, align=True)
    rmsd_ligand = compute_rmsd(ref_ligand, rec_ligand, align=True)
    
    results.append({
        "Reference_File": os.path.basename(ref_file),
        "Docked_Complex": os.path.basename(rec_file),
        "Ligand_RMSD_A": rmsd_ligand,
        "Pocket_Ligand_RMSD_A": rmsd_pocket
    })

# --------------------
# Save CSV
# --------------------
df = pd.DataFrame(results)
csv_file = "pocket-vs-lig-rmsd-results.csv"
df.to_csv(csv_file, index=False)
print(f"Results saved to {csv_file}")

# --------------------
# Plot correlation
# --------------------
plt.figure(figsize=(6,6))
plt.scatter(df["Ligand_RMSD_A"], df["Pocket_Ligand_RMSD_A"], c='blue', alpha=0.7)
plt.xlabel("Ligand RMSD (Å)")
plt.ylabel("Pocket+Ligand RMSD (Å)")
plt.title("Correlation between ligand RMSD and pocket RMSD")
plt.grid(True)
plt.plot([0, max(df["Ligand_RMSD_A"])], [0, max(df["Pocket_Ligand_RMSD_A"])], 'r--', alpha=0.5)
plt.tight_layout()
plt.savefig("ligand_vs_pocket_rmsd.png", dpi=300)  # PNG at 300 dpi
plt.show()

corr = np.corrcoef(df["Ligand_RMSD_A"], df["Pocket_Ligand_RMSD_A"])[0,1]
print(f"Pearson correlation coefficient: {corr:.2f}")