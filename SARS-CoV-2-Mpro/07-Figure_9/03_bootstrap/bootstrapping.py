import os
import re
import glob
import csv
import random
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolAlign

"""
Bootstrap analysis for Aposcore, GNINA and Random ranking.

This script:
1. Loads test molecules.
2. Loads Aposcore and GNINA rankings.
3. Computes RMSDs once.
4. Performs bootstrap resampling.
5. Saves summary CSV files.

No plotting is performed.

"""


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


def load_gnina_scores(rec_dir="../../02-building-poses-in-ApoDock-receptors/output"):
    scores_by_mol = defaultdict(list)

    rec_files = sorted(
        glob.glob(os.path.join(rec_dir, "cs_optimised_molecules_in_rec_*.sdf"))
    )

    for rec_file in rec_files:

        if os.path.getsize(rec_file) == 0:
            continue

        supplier = Chem.SDMolSupplier(rec_file)

        for mol in supplier:

            if mol is None:
                continue

            if not mol.HasProp("score") or not mol.HasProp("index"):
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
# PRECOMPUTE RMSDs
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

        else:
            pattern = os.path.join(ligand_dir, f"rec_*_mol{i}.sdf")
            ligands = [(f, 0) for f in glob.glob(pattern)]

        rmsds = []

        for lig_name, _ in ligands:

            lig_path = (
                lig_name if method == "random"
                else os.path.join(ligand_dir, lig_name)
            )

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
# BOOTSTRAP
#############################################

def compute_curve(sampled_indices, rmsd_data, method, max_N, rmsd_threshold):

    curve = []

    for N in range(1, max_N + 1):

        lowest = []

        for idx in sampled_indices:

            if idx not in rmsd_data:
                continue

            rmsds = rmsd_data[idx]

            if method == "random":
                chosen = rmsds if len(rmsds) < N else random.sample(rmsds, N)
            else:
                chosen = rmsds[:N]

            if chosen:
                lowest.append(min(chosen))

        pct = (
            100 * sum(r < rmsd_threshold for r in lowest) / len(lowest)
            if lowest else 0
        )

        curve.append(pct)

    return curve


def bootstrap(method, rmsd_data, n_molecules,
              n_bootstrap=10000,
              max_N=20,
              rmsd_threshold=2.0):

    curves = []

    for _ in range(n_bootstrap):

        indices = np.random.choice(
            n_molecules,
            n_molecules,
            replace=True
        )

        curves.append(
            compute_curve(
                indices,
                rmsd_data,
                method,
                max_N,
                rmsd_threshold
            )
        )

    return np.array(curves)


#############################################
# SAVE
#############################################

def summarize(curves):
    mean = np.mean(curves, axis=0)
    low = np.percentile(curves, 2.5, axis=0)
    high = np.percentile(curves, 97.5, axis=0)
    return mean, low, high


def save_summary(filename, mean, low, high):

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["N", "mean", "lower_95CI", "upper_95CI"])

        for i in range(len(mean)):
            writer.writerow([i + 1, mean[i], low[i], high[i]])


#############################################
# MAIN
#############################################

def run_all():

    print("Loading molecules...")

    test_mols = [
        m for m in Chem.SDMolSupplier("../../../released-receptor-structures/released_test_molecules/test_sars.sdf")
        if m is not None
    ]

    apos = load_aposcore_scores("../../06-Figure_8/02_ApoScore/output/mol_scores_sorted.csv")
    gnina = load_gnina_scores("../../02-building-poses-in-ApoDock-receptors/output")

    print("Computing RMSDs...")

    apos_rmsd = precompute_rmsds(
        test_mols,
        apos,
        "../../02-building-poses-in-ApoDock-receptors/output/resulting_mols",
        "aposcore",
    )
    gnina_rmsd = precompute_rmsds(
        test_mols,
        gnina,
        "../../02-building-poses-in-ApoDock-receptors/output/resulting_mols",
        "gnina",
    )
    random_rmsd = precompute_rmsds(
        test_mols,
        None,
        "../../02-building-poses-in-ApoDock-receptors/output/resulting_mols",
        "random",
    )

    print("Bootstrapping...")

    apos_curves = bootstrap("aposcore", apos_rmsd, len(test_mols))
    gnina_curves = bootstrap("gnina", gnina_rmsd, len(test_mols))
    random_curves = bootstrap("random", random_rmsd, len(test_mols))

    # Save outputs to scripts/output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    apos_file = os.path.join(output_dir, "aposcore_summary.csv")
    gnina_file = os.path.join(output_dir, "gnina_summary.csv")
    random_file = os.path.join(output_dir, "random_summary.csv")

    save_summary(apos_file, *summarize(apos_curves))
    save_summary(gnina_file, *summarize(gnina_curves))
    save_summary(random_file, *summarize(random_curves))

    print("Done.")
    print("Saved:")
    print(f"  {apos_file}")
    print(f"  {gnina_file}")
    print(f"  {random_file}")


if __name__ == "__main__":
    run_all()