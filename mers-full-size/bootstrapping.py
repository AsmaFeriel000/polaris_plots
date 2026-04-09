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
# PRECOMPUTE RMSDs (FAST)
#############################################

def precompute_rmsds(test_mols, scores_by_mol, ligand_dir, method):
    rmsd_data = {}

    for i, test_mol in enumerate(test_mols):

        if method == "aposcore":
            mol_id = f"mol{i}"
            if mol_id not in scores_by_mol:
                continue
            ligands = scores_by_mol[mol_id]

        elif method == "gnina":
            if i not in scores_by_mol:
                continue
            ligands = scores_by_mol[i]

        elif method == "random":
            pattern = os.path.join(ligand_dir, f"rec_*_mol{i}.sdf")
            lig_files = glob.glob(pattern)
            ligands = [(f, 0) for f in lig_files]

        rmsds = []

        for lig_name, _ in ligands:

            lig_path = lig_name if method == "random" else os.path.join(ligand_dir, lig_name)

            if not os.path.exists(lig_path):
                continue

            supplier = Chem.SDMolSupplier(lig_path)
            if not supplier or supplier[0] is None:
                continue

            try:
                rmsd = rdMolAlign.CalcRMS(test_mol, supplier[0])
                rmsds.append(rmsd)
            except:
                continue

        if rmsds:
            rmsd_data[i] = rmsds

    return rmsd_data


#############################################
# FAST CURVE COMPUTATION
#############################################

def compute_curve_fast(sampled_indices, rmsd_data, method, max_N, rmsd_threshold):

    results = []

    for N in range(1, max_N + 1):
        lowest_rmsds = []

        for idx in sampled_indices:

            if idx not in rmsd_data:
                continue

            rmsds = rmsd_data[idx]

            if method in ["aposcore", "gnina"]:
                selected = rmsds[:N]
            elif method == "random":
                selected = rmsds if len(rmsds) < N else random.sample(rmsds, N)

            if selected:
                lowest_rmsds.append(min(selected))

        pct = (
            100 * sum(r < rmsd_threshold for r in lowest_rmsds) / len(lowest_rmsds)
            if lowest_rmsds else 0
        )

        results.append(pct)

    return results


#############################################
# BOOTSTRAP
#############################################

def bootstrap_method_fast(
    method_name,
    rmsd_data,
    n_mols,
    n_bootstrap=10000,
    max_N=20,
    rmsd_threshold=2.0
):
    all_curves = []

    for b in range(n_bootstrap):
        indices = np.random.choice(n_mols, n_mols, replace=True)

        curve = compute_curve_fast(
            indices,
            rmsd_data,
            method_name,
            max_N,
            rmsd_threshold
        )

        all_curves.append(curve)

    return np.array(all_curves)


#############################################
# SUMMARY + SAVE
#############################################

def summarize(curves):
    mean = np.mean(curves, axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    return mean, lower, upper


def save_summary_csv(filename, mean, lower, upper):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "mean", "lower_95CI", "upper_95CI"])
        for i in range(len(mean)):
            writer.writerow([i+1, mean[i], lower[i], upper[i]])


#############################################
# MAIN
#############################################

def run_all():

    print("Loading test molecules...")
    test_mols = [m for m in Chem.SDMolSupplier("test_mers.sdf") if m is not None]

    print("Loading scores...")
    aposcore_scores = load_aposcore_scores("fegrow_result/mol_scores_sorted.csv")
    gnina_scores = load_gnina_scores(".")

    print("Precomputing RMSDs (slow step)...")

    apos_rmsd = precompute_rmsds(test_mols, aposcore_scores, "fegrow_result", "aposcore")
    gnina_rmsd = precompute_rmsds(test_mols, gnina_scores, "fegrow_result", "gnina")
    random_rmsd = precompute_rmsds(test_mols, None, "fegrow_result", "random")

    print("Bootstrapping (fast)...")

    apos_curves = bootstrap_method_fast("aposcore", apos_rmsd, len(test_mols))
    gnina_curves = bootstrap_method_fast("gnina", gnina_rmsd, len(test_mols))
    rand_curves = bootstrap_method_fast("random", random_rmsd, len(test_mols))

    print("Summarizing...")

    apos_mean, apos_low, apos_up = summarize(apos_curves)
    gnina_mean, gnina_low, gnina_up = summarize(gnina_curves)
    rand_mean, rand_low, rand_up = summarize(rand_curves)

    # Save CSVs
    save_summary_csv("aposcore_summary.csv", apos_mean, apos_low, apos_up)
    save_summary_csv("gnina_summary.csv", gnina_mean, gnina_low, gnina_up)
    save_summary_csv("random_summary.csv", rand_mean, rand_low, rand_up)

    print("Plotting...")

    N = np.arange(1, len(apos_mean)+1)

    plt.figure(figsize=(7,5))

    apos_err = [apos_mean - apos_low, apos_up - apos_mean]
    gnina_err = [gnina_mean - gnina_low, gnina_up - gnina_mean]
    rand_err = [rand_mean - rand_low, rand_up - rand_mean]

    plt.errorbar(N, apos_mean, yerr=apos_err, marker='o', capsize=4, label="Aposcore")
    plt.errorbar(N, gnina_mean, yerr=gnina_err, marker='o', capsize=4, label="GNINA")
    plt.errorbar(N, rand_mean, yerr=rand_err, marker='o', capsize=4, label="Random")

    plt.xlabel("Top N")
    plt.ylabel("% RMSD < 2 Å")
    plt.title("Top-N Pose Accuracy for MERS-CoV Mpro") 
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(1, len(N)+1, 2))
    plt.tight_layout()
    plt.savefig("combined_bootstrap_plot.png", dpi=300)
    plt.show()

    print("\n Done!")
    print("Saved:")
    print("- aposcore_summary.csv")
    print("- gnina_summary.csv")
    print("- random_summary.csv")
    print("- combined_bootstrap_fast.png")


if __name__ == "__main__":
    run_all()