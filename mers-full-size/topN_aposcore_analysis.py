import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from collections import defaultdict

"""calculate rmsd between best scoring molecule (apodock score) and corresponding test_mers.sdf (released test poses for mers)""" 

def load_scores(csv_file):
    scores_by_mol = defaultdict(list)
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ligand = row["ligand"]
            score = float(row["score"])
            match = re.search(r"(mol\d+)\.sdf", ligand)
            if match:
                mol_id = match.group(1)
                scores_by_mol[mol_id].append((ligand, score))
    # Sort each mol's ligands by descending score
    for mol_id in scores_by_mol:
        scores_by_mol[mol_id].sort(key=lambda x: x[1], reverse=True)
    return scores_by_mol


def evaluate_rmsd_curve(
    score_csv="fegrow_result/mol_scores_sorted.csv",
    test_sdf="test_mers.sdf",
    ligand_dir="fegrow_result",
    rmsd_threshold=2.0,
    max_N=10,
    summary_csv="topN_rmsd_summary.csv",
    plot_path="topN_rmsd_curve.png"
):
    print("📥 Loading test molecules...")
    test_mols = Chem.SDMolSupplier(test_sdf)
    test_mols = [m for m in test_mols if m is not None]

    print("📈 Loading scored ligands...")
    scores_by_mol = load_scores(score_csv)
    
    summary = []

    for N in range(1, max_N + 1):
        lowest_rmsds = []

        for i, test_mol in enumerate(test_mols):
            mol_id = f"mol{i}"
            if mol_id not in scores_by_mol:
                continue

            ligands = scores_by_mol[mol_id][:N]
            
            best_rmsd = None

            for lig_name, _ in ligands:
                lig_path = os.path.join(ligand_dir, lig_name)
                supplier = Chem.SDMolSupplier(lig_path)
                if not supplier or not supplier[0]:
                    continue
                try:
                    rmsd = rdMolAlign.CalcRMS(test_mol, supplier[0])
                    if best_rmsd is None or rmsd < best_rmsd:
                        best_rmsd = rmsd
                except:
                    continue

            if best_rmsd is not None:
                lowest_rmsds.append(best_rmsd)

        if lowest_rmsds:
            pct_below = 100 * sum(r < rmsd_threshold for r in lowest_rmsds) / len(lowest_rmsds)
        else:
            pct_below = 0.0

        summary.append((N, round(pct_below, 2)))
        print(f"Top-{N}: {pct_below:.2f}% ligands have RMSD < {rmsd_threshold} Å")

    # Save summary CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", f"percent_below_{rmsd_threshold}A"])
        writer.writerows(summary)
    print(f"💾 Summary saved to: {summary_csv}")

    # Plot curve
    Ns, percentages = zip(*summary)
    plt.figure(figsize=(6, 4))
    plt.plot(Ns, percentages, marker='o', color='dodgerblue')
    plt.xlabel("Top N ligands per molX")
    plt.ylabel(f"% with RMSD < {rmsd_threshold} Å")
    plt.title(f"Aposcore's Top N RMSD Analysis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"📊 Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_rmsd_curve(
        score_csv="fegrow_result/mol_scores_sorted.csv",
        test_sdf="test_mers.sdf",
        ligand_dir="fegrow_result",
        rmsd_threshold=2.0,
        max_N=20  # Change to explore deeper
    )
