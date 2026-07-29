import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from collections import defaultdict

"""for each N top scoring molecules (based on gnina score property), find percentage of molecules with RMSD <2A

"""

def load_gnina_scores(rec_dir="../../02-building-poses-in-ApoDock-receptors/output"):
    scores_by_mol = defaultdict(list)

    rec_files = sorted(glob.glob(os.path.join(rec_dir, "cs_optimised_molecules_in_rec_*.sdf")))
    for rec_file in rec_files:
        if os.path.getsize(rec_file) == 0:
            continue
        supplier = Chem.SDMolSupplier(rec_file)
        if not supplier or all(m is None for m in supplier):
            continue
        for mol in supplier:
            if mol is None or not mol.HasProp("score") or not mol.HasProp("index"):
                continue
            try:
                # 🔹 use <score> instead of <cnnaffinities>
                score = float(mol.GetProp("score"))
                mol_index = int(mol.GetProp("index"))
            except Exception as e:
                print(f"⚠️ Failed parsing in {rec_file}, error={e}")
                continue
            # store ligand filename + score
            rec_index = os.path.basename(rec_file).split("_")[-1].split(".")[0]
            lig_filename = f"rec_{rec_index}_mol{mol_index}.sdf"
            scores_by_mol[mol_index].append((lig_filename, score))

    # sort each mol’s ligands by descending score (higher score = better)
    for mol_index in scores_by_mol:
        scores_by_mol[mol_index].sort(key=lambda x: x[1], reverse=True)

    return scores_by_mol


def gnina_topN_analysis(
    rec_dir="../../02-building-poses-in-ApoDock-receptors/output",
    fegrow_dir="../../02-building-poses-in-ApoDock-receptors/output/resulting_mols",
    test_sdf="../../../released-receptor-structures/released_test_molecules/test_mers.sdf",
    max_N=20,
    rmsd_threshold=2.0,
    summary_csv="gnina_topN_summary.csv"
):
    # load reference poses
    test_mols = Chem.SDMolSupplier(test_sdf)
    test_mols = [m for m in test_mols if m is not None]

    # load gnina scores
    scores_by_mol = load_gnina_scores(rec_dir)

    summary = []

    for N in range(1, max_N + 1):
        all_lowest_rmsds = []
        for mol_index, test_mol in enumerate(test_mols):
            if mol_index not in scores_by_mol:
                continue
            topN = scores_by_mol[mol_index][:N]
            rmsds = []
            for lig_name, score in topN:
                lig_path = os.path.join(fegrow_dir, lig_name)
                if not os.path.exists(lig_path):
                    continue
                lig_supplier = Chem.SDMolSupplier(lig_path, removeHs=False)
                if not lig_supplier or lig_supplier[0] is None:
                    continue
                lig_mol = lig_supplier[0]

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

     # save summary csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, summary_csv)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", f"percent_below_{rmsd_threshold}A"])
        writer.writerows(summary)

    print(f"💾 Saved summary: {summary_csv}")


if __name__ == "__main__":
    gnina_topN_analysis(
        rec_dir="../../02-building-poses-in-ApoDock-receptors/output",
        fegrow_dir="../../02-building-poses-in-ApoDock-receptors/output/resulting_mols",
        test_sdf="../../../released-receptor-structures/released_test_molecules/test_mers.sdf",
        max_N=20
    )
