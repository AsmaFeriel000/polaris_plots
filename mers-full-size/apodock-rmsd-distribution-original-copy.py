import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.stats import ks_2samp

"""
MERS-CoV Mpro residue-wise RMSD distribution comparison
Uses fixed x-range and constant bin width for all histograms.
"""

# ----------------------------
# USER INPUT
# ----------------------------

reference_pdb = "complex-MERS.pdb"

apodock_dir = "receptors_with_hydrogens/"
xtal_dir = "released_MERS-CoV_Mpro"

residues = [25, 49, 167, 168, 192]

# ----------------------------
# GLOBAL HISTOGRAM SETTINGS
# ----------------------------

XMIN = 0.0
XMAX = 3.0
BIN_WIDTH = 0.1
BINS = np.arange(XMIN, XMAX + BIN_WIDTH, BIN_WIDTH)

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

def get_residue_labels(universe, residues):
    labels = {}
    for res in residues:
        sel = universe.select_atoms(f"resid {res} and name CA")
        if len(sel) > 0:
            resname = sel.resnames[0]
            labels[res] = f"{aa_map.get(resname, 'X')}{res}"
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

apodock_res = compute_residue_distributions(apodock_files, reference_pdb, residues)
xtal_res = compute_residue_distributions(xtal_files, reference_pdb, residues)

# ----------------------------
# PLOT
# ----------------------------

fig, axes = plt.subplots(
    1, len(residues),
    figsize=(4*len(residues), 4),
    sharex=True,
    sharey=True
)

if len(residues) == 1:
    axes = [axes]

for i, res in enumerate(residues):

    ax = axes[i]

    apo_vals = np.array(apodock_res[res])
    xtal_vals = np.array(xtal_res[res])

    label = residue_labels[res]

    print(f"\n{label}")
    print(f"ApoDock n = {len(apo_vals)}")
    print(f"Experimental n = {len(xtal_vals)}")

    if len(apo_vals) > 1 and len(xtal_vals) > 1:
        stat, p = ks_2samp(apo_vals, xtal_vals)
        print(f"KS statistic = {stat:.3f}")
        print(f"KS p-value = {p:.3e}")

    sns.histplot(
        xtal_vals,
        bins=BINS,
        binrange=(XMIN, XMAX),
        stat="density",
        alpha=0.3,
        label="Experimental",
        ax=ax
    )

    sns.histplot(
        apo_vals,
        bins=BINS,
        binrange=(XMIN, XMAX),
        stat="density",
        alpha=0.3,
        label="ApoDock",
        ax=ax
    )

    ax.set_xlim(XMIN, XMAX)
    ax.set_title(label)
    ax.set_xlabel("RMSD (Å)")

    if i == 0:
        ax.set_ylabel("Density")

plt.suptitle("MERS-CoV Mpro Residue-wise Pocket RMSD Distributions", y=1.05)
plt.tight_layout()
plt.savefig("pocket_rmsd_distributions-per-residue-histogram-same-x-scale.png", dpi=300)
plt.show()