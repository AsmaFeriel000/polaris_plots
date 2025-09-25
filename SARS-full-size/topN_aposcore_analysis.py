# finally aposcore plot works 
import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from collections import defaultdict

"""calculate rmsd between best scoring molecule (apodock score) and corresponding test_sars.sdf (released test poses for sars)""" 

def load_scores(csv_file):
    """Load scores and sort per molecule."""
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
    test_sdf="test_sars.sdf",
    ligand_dir="fegrow_result",
    rmsd_threshold=2.0,
    max_N=10,
    summary_csv="topN_rmsd_summary.csv",
    skipped_csv="aposcore_skipped_molecules.csv",
    plot_path="topN_rmsd_curve.png",
    exclude_top1=["mol2", "mol8", "mol21", "mol24", "mol32", "mol35", "mol41", "mol47", "mol59", "mol60", "mol83", "mol84", "mol93", "mol95"]   # 🔹 list of mol IDs to skip in Top-1
):
    if exclude_top1 is None:
        exclude_top1 = []

    print("📥 Loading test molecules...")
    test_mols = Chem.SDMolSupplier(test_sdf)
    test_mols = [m for m in test_mols if m is not None]

    print("📈 Loading scored ligands...")
    scores_by_mol = load_scores(score_csv)
    
    summary = []
    skipped_counts = defaultdict(int)

    for N in range(1, max_N + 1):
        all_lowest_rmsds = []

        for i, test_mol in enumerate(test_mols):
            mol_id = f"mol{i}"
            if mol_id not in scores_by_mol:
                continue
            rmsds = []
            ligands = scores_by_mol[mol_id][:N]
            best_rmsd = None

            for lig_name, _ in ligands:
                # 🔹 Skip user-specified molecules only for Top-1
                if N == 1 and mol_id in exclude_top1:
                    continue

                lig_path = os.path.join(ligand_dir, lig_name)
                supplier = Chem.SDMolSupplier(lig_path)
                if not supplier or not supplier[0]:
                    continue

                lig_mol = supplier[0]

                # Check if molecule is built correctly
                if lig_mol.GetNumAtoms() == 0:
                    continue
                conf = lig_mol.GetConformer()
                if conf is None or conf.GetNumAtoms() != lig_mol.GetNumAtoms():
                    continue

                try:
                    rmsd = rdMolAlign.CalcRMS(test_mol, lig_mol)
                    rmsds.append(rmsd)
                except:
                    continue
            if rmsds:
                all_lowest_rmsds.append(min(rmsds))

        pct_below = (
            100 * sum(r < rmsd_threshold for r in all_lowest_rmsds) / len(all_lowest_rmsds)
            if all_lowest_rmsds else 0
        )
        summary.append((N, pct_below))
        print(f"N={N}: {pct_below:.2f}% < {rmsd_threshold} Å")

    # Save summary CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", f"percent_below_{rmsd_threshold}A"])
        writer.writerows(summary)
    print(f"💾 Summary saved to: {summary_csv}")

    # Save skipped molecules report
    with open(skipped_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Molecule", "SkippedCount"])
        for mol_id, count in skipped_counts.items():
            writer.writerow([mol_id, count])
    print(f"💾 Skipped molecules report saved to: {skipped_csv}")

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
        test_sdf="test_sars.sdf",
        ligand_dir="fegrow_result",
        rmsd_threshold=2.0,
        max_N=20,
        exclude_top1=["mol2", "mol8", "mol21", "mol24", "mol32", "mol35", "mol41", "mol47", "mol59", "mol60", "mol83", "mol84", "mol93", "mol95"]  # 🔹 excluded only for Top-1
    )

