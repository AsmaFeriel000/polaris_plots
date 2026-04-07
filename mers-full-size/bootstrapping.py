import os
import re
import glob
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolAlign

#############################################
# LOADERS
#############################################

def load_aposcore_scores(csv_file):
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

    for mol_id in scores_by_mol:
        scores_by_mol[mol_id].sort(key=lambda x: x[1], reverse=True)

    return scores_by_mol


def load_gnina_scores(rec_dir="."):
    scores_by_mol = defaultdict(list)

    rec_files = sorted(glob.glob(os.path.join(rec_dir, "cs_optimised_molecules_in_rec_*.sdf")))
    for rec_file in rec_files:
        if os.path.getsize(rec_file) == 0:
            continue

        supplier = Chem.SDMolSupplier(rec_file)
        for mol in supplier:
            if mol is None or not mol.HasProp("score") or not mol.HasProp("index"):
                continue
            try:
                score = float(mol.GetProp("score"))
                mol_index = int(mol.GetProp("index"))
            except:
                continue

            rec_index = os.path.basename(rec_file).split("_")[-1].split(".")[0]
            lig_filename = f"rec_{rec_index}_mol{mol_index}.sdf"

            scores_by_mol[mol_index].append((lig_filename, score))

    for mol_index in scores_by_mol:
        scores_by_mol[mol_index].sort(key=lambda x: x[1], reverse=True)

    return scores_by_mol


#############################################
# CORE RMSD COMPUTATION
#############################################

def compute_curve(test_mols, scores_by_mol, ligand_dir, method, max_N, rmsd_threshold):

    results = []

    for N in range(1, max_N + 1):
        lowest_rmsds = []

        for i, test_mol in enumerate(test_mols):

            if method == "aposcore":
                mol_id = f"mol{i}"
                if mol_id not in scores_by_mol:
                    continue
                ligands = scores_by_mol[mol_id][:N]

            elif method == "gnina":
                if i not in scores_by_mol:
                    continue
                ligands = scores_by_mol[i][:N]

            elif method == "random":
                pattern = os.path.join(ligand_dir, f"rec_*_mol{i}.sdf")
                candidates = glob.glob(pattern)
                if not candidates:
                    continue
                chosen = candidates if len(candidates) < N else random.sample(candidates, N)
                ligands = [(c, 0) for c in chosen]

            best_rmsd = None

            for lig_name, _ in ligands:

                lig_path = lig_name if method == "random" else os.path.join(ligand_dir, lig_name)

                if not os.path.exists(lig_path):
                    continue

                supplier = Chem.SDMolSupplier(lig_path)
                if not supplier or supplier[0] is None:
                    continue

                try:
                    rmsd = rdMolAlign.CalcRMS(test_mol, supplier[0])
                    if best_rmsd is None or rmsd < best_rmsd:
                        best_rmsd = rmsd
                except:
                    continue

            if best_rmsd is not None:
                lowest_rmsds.append(best_rmsd)

        pct = (
            100 * sum(r < rmsd_threshold for r in lowest_rmsds) / len(lowest_rmsds)
            if lowest_rmsds else 0
        )

        results.append(pct)

    return results


#############################################
# BOOTSTRAP ENGINE
#############################################

def bootstrap_method(
    method_name,
    scores_by_mol,
    test_mols,
    ligand_dir,
    n_bootstrap=10000,
    max_N=20,
    rmsd_threshold=2.0,
    out_dir="bootstrap_output"
):
    os.makedirs(out_dir, exist_ok=True)

    all_curves = []

    for b in range(n_bootstrap):
        print(f"{method_name} bootstrap {b+1}/{n_bootstrap}")

        indices = np.random.choice(len(test_mols), len(test_mols), replace=True)
        sampled = [test_mols[i] for i in indices]

        curve = compute_curve(sampled, scores_by_mol, ligand_dir, method_name, max_N, rmsd_threshold)
        all_curves.append(curve)

        # save each bootstrap
        with open(os.path.join(out_dir, f"{method_name}_{b:03d}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "percent"])
            for i, val in enumerate(curve, 1):
                writer.writerow([i, val])

    return np.array(all_curves)


#############################################
# STATS + SAVE
#############################################

def summarize(curves, out_csv):
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "mean", "std"])
        for i in range(len(mean)):
            writer.writerow([i+1, mean[i], std[i]])

    return mean, std


#############################################
# MAIN PIPELINE
#############################################

def run_all():

    test_mols = [m for m in Chem.SDMolSupplier("test_mers.sdf") if m is not None]

    aposcore_scores = load_aposcore_scores("fegrow_result/mol_scores_sorted.csv")
    gnina_scores = load_gnina_scores(".")

    apos_curves = bootstrap_method(
        "aposcore", aposcore_scores, test_mols, "fegrow_result", out_dir="boot_aposcore"
    )

    gnina_curves = bootstrap_method(
        "gnina", gnina_scores, test_mols, "fegrow_result", out_dir="boot_gnina"
    )

    random_curves = bootstrap_method(
        "random", None, test_mols, "fegrow_result", out_dir="boot_random"
    )

    apos_mean, apos_std = summarize(apos_curves, "aposcore_summary.csv")
    gnina_mean, gnina_std = summarize(gnina_curves, "gnina_summary.csv")
    rand_mean, rand_std = summarize(random_curves, "random_summary.csv")

    # PLOT
    N = np.arange(1, len(apos_mean)+1)

    plt.figure(figsize=(7,5))

    plt.errorbar(N, apos_mean, yerr=apos_std, label="Aposcore", marker='o')
    plt.errorbar(N, gnina_mean, yerr=gnina_std, label="GNINA", marker='o')
    plt.errorbar(N, rand_mean, yerr=rand_std, label="Random", marker='o')

    plt.xlabel("Top N")
    plt.ylabel("% RMSD < 2 Å")
    plt.title("Top-N Pose Accuracy (Bootstrapped)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_bootstrap_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_all()