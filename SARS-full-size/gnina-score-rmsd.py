import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import csv

"""calculate rmsd between top scoring (gnina score, not affinity property in cs_optimised*.sdf) and their corresponding test_sars.sdf (released test molecules) to generate cdf plot"""

def find_best_scores_per_molecule(   # 🔹 renamed function
    rec_dir=".",
    fegrow_dir="fegrow_result",
    test_sdf="test_sars.sdf"
):
    # Load reference molecules
    test_mols = Chem.SDMolSupplier(test_sdf)
    test_mols = [m for m in test_mols if m is not None]
    num_mols = len(test_mols)

    # Store best ligand info per molecule index (<index> value)
    best_per_mol = {}  # mol_index (from <index>) -> (rec_index, score)

    rec_files = sorted(glob.glob(os.path.join(rec_dir, "cs_optimised_molecules_in_rec_*.sdf")))

    skipped = []

    for rec_file in rec_files:
        if os.path.getsize(rec_file) == 0:
            continue

        rec_index = os.path.basename(rec_file).split("_")[-1].split(".")[0]
        supplier = Chem.SDMolSupplier(rec_file)
        if not supplier or all(m is None for m in supplier):
            continue

        for mol in supplier:
            # 🔹 now check for "score" instead of "cnnaffinities"
            if mol is None or not mol.HasProp("score") or not mol.HasProp("index"):
                continue
            try:
                score = float(mol.GetProp("score"))
                mol_index = int(mol.GetProp("index"))
            except:
                continue

            current_best = best_per_mol.get(mol_index)
            # 🔹 pick the best (highest) score
            if current_best is None or score > current_best[1]:
                best_per_mol[mol_index] = (rec_index, score)

    # Collect RMSDs for best ligands
    results = []
    for mol_index, (rec_index, best_score) in best_per_mol.items():
        lig_filename = f"rec_{rec_index}_mol{mol_index}.sdf"
        lig_path = os.path.join(fegrow_dir, lig_filename)

        if not os.path.exists(lig_path):
            skipped.append((f"mol{mol_index}", f"missing {lig_filename}"))
            continue

        lig_supplier = Chem.SDMolSupplier(lig_path)
        if not lig_supplier or not lig_supplier[0]:
            skipped.append((f"mol{mol_index}", f"unreadable ligand {lig_filename}"))
            continue

        if mol_index >= len(test_mols):
            skipped.append((f"mol{mol_index}", "index out of range in test_mers.sdf"))
            continue

        test_mol = test_mols[mol_index]
        try:
            rmsd = rdMolAlign.CalcRMS(test_mol, lig_supplier[0])
            results.append((lig_filename, best_score, rmsd))
        except:
            skipped.append((f"mol{mol_index}", f"RMSD calculation failed for {lig_filename}"))

    if not results:
        print("No results found.")
        return

    # Save RMSD results to CSV
    csv_file = "gnina_scores_rmsds.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ligand", "score", "RMSD"])  # 🔹 updated column name
        writer.writerows(results)

    # Save skipped details to CSV
    if skipped:
        skipped_file = "skipped_details.csv"
        with open(skipped_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Molecule", "Reason"])
            writer.writerows(skipped)
        print(f"Skipped details saved to {skipped_file}")

    # Data for plots
    rmsds = [r[2] for r in results]
    scores = [r[1] for r in results]

    # Plot CDF of RMSDs
    sorted_rmsds = np.sort(rmsds)
    cdf = np.arange(1, len(sorted_rmsds) + 1) / len(sorted_rmsds)

    plt.figure(figsize=(6, 4))
    plt.plot(sorted_rmsds, cdf, marker='.', color='blue')
    plt.title("GNINA scores: RMSD CDF")
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gnina_scores_cdf.png", dpi=300)
    plt.close()

    # Scatter plot Score vs RMSD
    plt.figure(figsize=(6, 4))
    plt.scatter(scores, rmsds, alpha=0.7, color='purple')
    plt.title("GNINA score vs RMSD")
    plt.xlabel("Score")
    plt.ylabel("RMSD (Å)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("score_vs_rmsd_scatter.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    find_best_scores_per_molecule(
        rec_dir=".",
        fegrow_dir="fegrow_result",
        test_sdf="test_sars.sdf"
    )
