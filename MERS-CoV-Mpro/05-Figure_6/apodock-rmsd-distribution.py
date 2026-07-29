import MDAnalysis as mda
import numpy as np
import pandas as pd
import glob
import os

"""
Compute residue-wise side-chain RMSDs and save them to CSV.
"""

# ----------------------------
# USER INPUT
# ----------------------------

reference_pdb = "../01-ApoDock-side-chain-modelling/inputs/complex-mers.pdb"

apodock_dir = "../01-ApoDock-side-chain-modelling/output/complex-mers-lig-h.sdf/receptors_with_hydrogen/"
xtal_dir = "../../released-receptor-structures/group2_MERS-CoV_Mpro/"

residues = [25, 49, 167, 168, 192]

# ----------------------------
# FILES
# ----------------------------

apodock_files = sorted(glob.glob(os.path.join(apodock_dir, "*.pdb")))
xtal_files = sorted(glob.glob(os.path.join(xtal_dir, "*.pdb")))

# ----------------------------
# AMINO ACID MAP
# ----------------------------

aa_map = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# ----------------------------
# LABELS
# ----------------------------

def get_residue_labels(universe, residues):
    labels = {}

    for res in residues:
        sel = universe.select_atoms(f"resid {res} and name CA")

        if len(sel):
            labels[res] = f"{aa_map.get(sel.resnames[0], 'X')}{res}"
        else:
            labels[res] = f"Res{res}"

    return labels


ref = mda.Universe(reference_pdb)
residue_labels = get_residue_labels(ref, residues)

# ----------------------------
# CORE FUNCTION
# ----------------------------

def compute_residue_distributions(structure_files, reference_pdb, residues):

    ref = mda.Universe(reference_pdb)

    ref_atoms = {
        res: ref.select_atoms(
            f"resid {res} and not name H* and not backbone"
        )
        for res in residues
    }

    results = {res: [] for res in residues}

    for f in structure_files:

        try:

            u = mda.Universe(f)

            for res in residues:

                ref_sel = ref_atoms[res]

                mob_sel = u.select_atoms(
                    f"resid {res} and not name H* and not backbone"
                )

                if len(ref_sel) == 0 or len(mob_sel) == 0:
                    continue

                if len(ref_sel) != len(mob_sel):
                    continue

                diff = mob_sel.positions - ref_sel.positions
                rmsd = np.sqrt((diff**2).mean())

                results[res].append(rmsd)

        except Exception as e:
            print(f"Error with {f}: {e}")

    return results

# ----------------------------
# RUN
# ----------------------------

print("Computing ApoDock RMSDs...")
apodock_res = compute_residue_distributions(
    apodock_files,
    reference_pdb,
    residues,
)

print("Computing experimental RMSDs...")
xtal_res = compute_residue_distributions(
    xtal_files,
    reference_pdb,
    residues,
)

# ----------------------------
# SAVE CSV
# ----------------------------

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

rows = []

for res in residues:

    label = residue_labels[res]

    for value in apodock_res[res]:
        rows.append({
            "residue": res,
            "label": label,
            "dataset": "ApoDock",
            "rmsd": value
        })

    for value in xtal_res[res]:
        rows.append({
            "residue": res,
            "label": label,
            "dataset": "Experimental",
            "rmsd": value
        })

df = pd.DataFrame(rows)

output_csv = os.path.join(output_dir, "residue_rmsd.csv")

df.to_csv(output_csv, index=False)

print(f"\nSaved {len(df)} measurements to {output_csv}")