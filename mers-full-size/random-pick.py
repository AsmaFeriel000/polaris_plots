import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from rdkit import Chem
from rdkit.Chem import rdMolAlign

"""if N molecules picked at random and RMSD calculated against test_mers.sdf (released test mol) upto 100 times, how many of those will have rmsd <2 A"""

def random_pick_rmsd_analysis(
    fegrow_dir="fegrow_result",
    test_sdf="test_mers.sdf",
    rmsd_threshold=2.0,
    max_N=20,
    iterations=100
):
    test_mols = Chem.SDMolSupplier(test_sdf)
    test_mols = [m for m in test_mols if m is not None]
    num_mols = len(test_mols)
    print(f"Loaded {num_mols} test molecules from {test_sdf}")

    Ns = list(range(1, max_N + 1))
    avg_percentages = []

    all_rmsd_records = []  # store N, iteration, molX, lowest RMSD

    for N in Ns:
        print(f"\n--- Evaluating N = {N} ---")
        percentages = []

        for it in range(iterations):
            lowest_rmsds = []

            for i, test_mol in enumerate(test_mols):
                mol_id = f"mol{i}"
                pattern = os.path.join(fegrow_dir, f"rec_*_{mol_id}.sdf")
                candidates = glob.glob(pattern)

                if len(candidates) == 0:
                    continue

                chosen = candidates if len(candidates) < N else random.sample(candidates, N)

                best_rmsd = None
                for cand in chosen:
                    lig = Chem.SDMolSupplier(cand)
                    if not lig or not lig[0]:
                        continue
                    try:
                        rmsd = rdMolAlign.CalcRMS(test_mol, lig[0])
                        if best_rmsd is None or rmsd < best_rmsd:
                            best_rmsd = rmsd
                    except:
                        continue

                if best_rmsd is not None:
                    lowest_rmsds.append(best_rmsd)
                    all_rmsd_records.append((N, it, mol_id, best_rmsd))

            if lowest_rmsds:
                pct_below = 100 * sum(r < rmsd_threshold for r in lowest_rmsds) / len(lowest_rmsds)
                percentages.append(pct_below)

        avg_pct = np.mean(percentages) if percentages else 0
        avg_percentages.append(avg_pct)
        print(f"N={N}: Average % RMSD < {rmsd_threshold} Å = {avg_pct:.2f}%")

    # Save summary of % RMSD < threshold
    summary_file = "random_pick_rmsd_summary.csv"
    with open(summary_file, "w") as f:
        f.write("N,avg_percent_below_2A\n")
        for N, pct in zip(Ns, avg_percentages):
            f.write(f"{N},{pct:.2f}\n")
    print(f"Summary saved to {summary_file}")

    # Save all individual RMSDs
    all_rmsd_file = "random_pick_all_rmsds.csv"
    with open(all_rmsd_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Iteration", "Molecule", "LowestRMSD"])
        writer.writerows(all_rmsd_records)
    print(f"All individual RMSDs saved to {all_rmsd_file}")

    # Plot N vs average % below threshold
    plt.figure(figsize=(6, 4))
    plt.plot(Ns, avg_percentages, marker="o", color="green")
    plt.xlabel("N randomly picked ligands")
    plt.ylabel(f"Average % RMSD < {rmsd_threshold} Å")
    plt.title("Random Pick Pose Accuracy vs N")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("random_pick_rmsd_curve.png", dpi=300)
    print("Plot saved to random_pick_rmsd_curve.png")
    plt.show()

if __name__ == "__main__":
    random_pick_rmsd_analysis(
        fegrow_dir="fegrow_result",
        test_sdf="test_mers.sdf",
        rmsd_threshold=2.0,
        max_N=20,
        iterations=100  # use 10000 for final run
    )
