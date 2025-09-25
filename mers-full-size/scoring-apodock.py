import os
from glob import glob
import argparse
import torch
from Aposcore.inference_dataset import get_mdn_score
from Aposcore.Aposcore import Aposcore
import csv

import pymol2
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ApoScore scoring on ligand and receptor files."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="fegrow_result",
        help="Input folder containing ligand and receptor files.",
    )
    parser.add_argument(
        "--ligand_pattern",
        type=str,
        default="cs_optimised*",
        help="Glob pattern for ligand files.",
    )
    parser.add_argument(
        "--receptor_pattern",
        type=str,
        default="rec_final*",
        help="Glob pattern for receptor files.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/ApoScore_time_split_0.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    parser.add_argument(
        "--dis_threshold",
        type=float,
        default=5.0,
        help="Distance threshold for scoring.",
    )
    return parser.parse_args()


def collect_files(folder, pattern):
    files = glob(os.path.join(folder, pattern))
    if not files:
        print(f"Warning: No files found for pattern {pattern} in {folder}")
    # Remove files which have zero size
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        print(f"Warning: All files for pattern {pattern} in {folder} are empty.")
    return files

# remove hydrogen from receptor as required by get_mdn_score(()
def strip_hydrogens_from_receptors(receptor_files):
    for rec_path in receptor_files:
        # Skip if noH version already exists
        if rec_path.endswith("-noH.pdb"):
            continue
        noH_path = rec_path.replace(".pdb", "-noH.pdb")
        with pymol2.PyMOL() as pymol:
            pymol.cmd.load(rec_path, "rec")
            pymol.cmd.remove("hydrogens")
            pymol.cmd.save(noH_path, "rec")

def get_scores():

    args = parse_args()

    model_mdn = Aposcore(
        35,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
        crossAttention=True,
        atten_active_fuc="softmax",
        num_layers=6,
        interact_type="product",
    )

    receptor_files = sorted(collect_files(args.input_folder, args.receptor_pattern))
    ligand_files = sorted(collect_files(args.input_folder, "*.sdf"))

    strip_hydrogens_from_receptors(receptor_files)
    receptor_noH_files = [r.replace(".pdb", "-noH.pdb") for r in receptor_files]

    receptor_map = {}
    for rec in receptor_noH_files:
        match = re.search(r"rec_final_(\d+)-noH\.pdb", os.path.basename(rec))
        if match:
            receptor_map[match.group(1)] = rec

    ligand_map = {}
    for lig in ligand_files:
        match = re.search(r"rec_(\d+)_mol\d+\.sdf", os.path.basename(lig))
        if match:
            rec_index = match.group(1)
            ligand_map.setdefault(rec_index, []).append(lig)

    output_lines = []
    all_scores = []  # To track all scored pairs

    for rec_index, rec_path in receptor_map.items():
        ligands = ligand_map.get(rec_index, [])
        if not ligands:
            continue

        output_lines.append(f"\n=== Scores for {os.path.basename(rec_path)} ===\n")

        for lig_path in sorted(ligands):
            score = get_mdn_score(
                [lig_path],
                [rec_path],
                model_mdn,
                args.ckpt,
                args.device,
                dis_threshold=args.dis_threshold,
            )[0]

            output_lines.append(f"{os.path.basename(lig_path)}: {score:.4f}")
            all_scores.append((lig_path, rec_path, score))

   # Score all receptor-ligand pairs done above, now reduce by molX
    mol_best = {}  # mol_id (e.g. mol0) → (lig_path, rec_path, score)

    for lig_path, rec_path, score in all_scores:
        match = re.search(r"(mol\d+)\.sdf", os.path.basename(lig_path))
        if match:
            mol_id = match.group(1)
            # Keep only highest scoring pair for this molX
            if mol_id not in mol_best or score > mol_best[mol_id][2]:
                mol_best[mol_id] = (lig_path, rec_path, score)

    # Sort by molX numerically (mol0, mol1, mol2, ...)
    def mol_sort_key(item):
        mol_id = item[0]
        return int(re.search(r"mol(\d+)", mol_id).group(1))

    sorted_mol_best = sorted(mol_best.items(), key=mol_sort_key)

    # Write results
    output_file = os.path.join(args.input_folder, "best_score_per_mol.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))  # receptor-wise full scores

        f.write("\n\n=== Best Scoring Pair Per molX ===\n")
        for mol_id, (lig, rec, score) in sorted_mol_best:
            f.write(
                f"{mol_id}: {os.path.basename(lig)} | "
                f"{os.path.basename(rec)} | "
                f"{score:.4f}\n"
            )

    # Print summary
    print(f"\nScoring complete. Output saved to {output_file}")
    print("\n=== Best Scoring Pair Per molX ===")
    for mol_id, (lig, rec, score) in sorted_mol_best:
        print(
            f"{mol_id}: {os.path.basename(lig)} | "
            f"{os.path.basename(rec)} | "
            f"{score:.4f}"

        )
         # === Save all best ligands (one per molX) into a single SDF ===
    try:
        from rdkit import Chem
        from rdkit.Chem import SDWriter
    except ImportError:
        print("RDKit is required to write SDF files. Install it via `pip install rdkit`.")
        return

    output_sdf_path = os.path.join(args.input_folder, "best_score_ligands.sdf")
    with SDWriter(output_sdf_path) as writer:
        for mol_id, (lig_path, _, score) in sorted_mol_best:
            supplier = Chem.SDMolSupplier(lig_path, removeHs=False)
            if not supplier or not supplier[0]:
                print(f"Warning: Could not read ligand from {lig_path}")
                continue
            mol = supplier[0]
            mol.SetProp("_Name", f"{mol_id}_score_{score:.4f}")
            mol.SetProp("ApoScore", f"{score:.4f}")
            writer.write(mol)

    print(f"\n Best-scoring ligands saved to: {output_sdf_path}")

    # === Score all molX individually across receptors and save sorted CSV ===
    print("\nScoring each molX across all receptor matches...")

    mol_scores = {}  # molX → list of (lig, rec, score)
    for lig_path, rec_path, score in all_scores:
        match = re.search(r"(mol\d+)\.sdf", os.path.basename(lig_path))
        if not match:
            continue
        mol_id = match.group(1)
        mol_scores.setdefault(mol_id, []).append((lig_path, rec_path, score))

    output_csv = os.path.join(args.input_folder, "mol_scores_sorted.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_id", "ligand", "receptor", "score"])
        for mol_id in sorted(mol_scores.keys(), key=lambda x: int(x.replace("mol", ""))):
            sorted_scores = sorted(mol_scores[mol_id], key=lambda x: x[2], reverse=True)
            for lig, rec, score in sorted_scores:
                writer.writerow([mol_id, os.path.basename(lig), os.path.basename(rec), f"{score:.4f}"])

    print(f"Full per-molX score list saved to: {output_csv}")

if __name__ == "__main__":
    get_scores()