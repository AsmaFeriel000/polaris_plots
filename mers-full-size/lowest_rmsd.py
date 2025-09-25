import os
import re
import csv
from glob import glob
from rdkit import Chem
from rdkit.Chem import rdMolAlign

"""Find the best (lowest) RMSDs for each test molecule against corresponding candidate ligands in fegrow-results.""" 

def find_best_rmsds(test_sdf_path, fegrow_dir, output_csv="lowest_rmsds.csv"):
    #test_mols = Chem.SDMolSupplier(test_sdf_path, removeHs=False)
    test_mols = Chem.SDMolSupplier(test_sdf_path)
    if not test_mols or len(test_mols) == 0:
        print("No molecules found in", test_sdf_path)
        return

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ligand_filename", "RMSD"])  # only 2 columns

        for mol_index, test_mol in enumerate(test_mols):
            if test_mol is None:
                continue

            pattern = os.path.join(fegrow_dir, f"rec_*_mol{mol_index}.sdf")
            candidate_files = sorted(glob(pattern))
            if not candidate_files:
                print(f"No matching files for mol{mol_index}")
                continue

            rmsd_results = []
            for candidate_path in candidate_files:
                #candidate_mols = Chem.SDMolSupplier(candidate_path, removeHs=False)
                candidate_mols = Chem.SDMolSupplier(candidate_path)
                if not candidate_mols or candidate_mols[0] is None:
                    continue
                try:
                    rmsd = rdMolAlign.CalcRMS(test_mol, candidate_mols[0])
                    rmsd_results.append((candidate_path, rmsd))
                except Exception as e:
                    print(f"Error for {candidate_path}: {e}")
                    continue

            if rmsd_results:
                best_path, best_rmsd = min(rmsd_results, key=lambda x: x[1])
                writer.writerow([os.path.basename(best_path), f"{best_rmsd:.4f}"])
                print(f"mol{mol_index}: {os.path.basename(best_path)} | RMSD: {best_rmsd:.4f}")
            else:
                print(f"mol{mol_index}:No valid matches found")

if __name__ == "__main__":
    test_sdf = "test_mers.sdf"
    fegrow_dir = "fegrow_result"
    output_csv = "lowest_rmsds.csv"

    find_best_rmsds(test_sdf, fegrow_dir, output_csv)
