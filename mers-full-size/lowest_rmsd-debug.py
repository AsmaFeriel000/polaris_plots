import os
import csv
from glob import glob
from rdkit import Chem
from rdkit.Chem import rdMolAlign


"""
Find the best (lowest) RMSDs for each test molecule against
corresponding candidate ligands in fegrow-results.

Also saves:
- number of atoms in test molecule
- number of atoms in candidate molecule
"""


def find_best_rmsds(test_sdf_path, fegrow_dir, output_csv="lowest_rmsds.csv"):

    test_mols = Chem.SDMolSupplier(test_sdf_path)

    if not test_mols or len(test_mols) == 0:
        print("No molecules found in", test_sdf_path)
        return

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # added atom count columns
        writer.writerow([
            "ligand_filename",
            "RMSD",
            "test_atoms",
            "candidate_atoms"
        ])

        for mol_index, test_mol in enumerate(test_mols):

            if test_mol is None:
                continue

            pattern = os.path.join(fegrow_dir, f"rec_*_mol{mol_index}.sdf")
            candidate_files = sorted(glob(pattern))

            if not candidate_files:
                print(f"No matching files for mol{mol_index}")
                continue

            rmsd_results = []

            # count atoms once for test molecule
            test_atoms = test_mol.GetNumAtoms()

            for candidate_path in candidate_files:

                candidate_mols = Chem.SDMolSupplier(candidate_path)

                if not candidate_mols or candidate_mols[0] is None:
                    continue

                candidate_mol = candidate_mols[0]
                candidate_atoms = candidate_mol.GetNumAtoms()

                try:
                    # calculate RMSD
                    rmsd = rdMolAlign.CalcRMS(test_mol, candidate_mol)

                    rmsd_results.append(
                        (candidate_path, rmsd, test_atoms, candidate_atoms)
                    )

                except Exception as e:
                    print(f"Error for {candidate_path}: {e}")
                    continue

            if rmsd_results:
                best_path, best_rmsd, test_atoms, candidate_atoms = min(
                    rmsd_results, key=lambda x: x[1]
                )

                writer.writerow([
                    os.path.basename(best_path),
                    f"{best_rmsd:.4f}",
                    test_atoms,
                    candidate_atoms
                ])

                print(
                    f"mol{mol_index}: {os.path.basename(best_path)} | "
                    f"RMSD: {best_rmsd:.4f} | "
                    f"atoms: {test_atoms}/{candidate_atoms}"
                )

            else:
                print(f"mol{mol_index}: No valid matches found")


if __name__ == "__main__":

    test_sdf = "test_mers.sdf"
    fegrow_dir = "fegrow_result"
    output_csv = "lowest_rmsds.csv"

    find_best_rmsds(test_sdf, fegrow_dir, output_csv)