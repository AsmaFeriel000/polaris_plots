import os
import subprocess
from rdkit import Chem
"""     
Build MERS-CoV Mpro complexes by merging the submitted ligand structures with the provided receptor PDB. The ligand structures are in SDF format and need
to be converted to PDB format using Open Babel. The final output will be a set of PDB files, each containing the receptor and one ligand, named "complex-MERS-mol<ID>.pdb". The receptor is taken from "mers-submission-complex/complex-MERS.pdb" and the ligands are taken from "submitted_structures/cs_optimised_molecules.sdf". 
The script will skip the ligand with index 78 as per instructions. The output complexes will be saved in the "merged_submitted_pdb" directory.
ligand 78 is skipped in the filenames because it wasn't built, so we want to maintain the same numbering for the output files to match the reference mers molecules. The script reads the receptor PDB once and then iterates over each ligand in the SDF, converting it to PDB format, merging it with the receptor, and saving the final complex PDB.
Temporary files for the ligand PDBs are stored in a "_tmp_lig_pdbs" directory which is created if it doesn't exist.
"""
# -----------------------------
# INPUT FILES
# -----------------------------
input_pdb = "mers-submission-complex/complex-MERS.pdb"
input_sdf = "submitted_structures/cs_optimised_molecules.sdf"

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------
output_dir = "merged_submitted_pdb"
tmp_dir = os.path.join(output_dir, "_tmp_lig_pdbs")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# -----------------------------
# Read protein once
# -----------------------------
with open(input_pdb, "r") as f:
    protein_lines = [
        line for line in f if not line.startswith("END")
    ]

# -----------------------------
# Read SDF molecules
# -----------------------------
supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

file_index = 0

for mol_id, mol in enumerate(supplier):

    if mol is None:
        continue

    # Skip filename index 78
    if file_index == 78:
        file_index = 79

    # Write temporary SDF for this molecule
    temp_sdf = os.path.join(tmp_dir, f"lig_{mol_id}.sdf")
    writer = Chem.SDWriter(temp_sdf)
    writer.write(mol)
    writer.close()

    # Convert SDF → PDB using Open Babel
    temp_pdb = os.path.join(tmp_dir, f"lig_{mol_id}.pdb")

    subprocess.run([
        "obabel",
        temp_sdf,
        "-O", temp_pdb,
        "--resname", "LIG"
    ], check=True)

    # Final output filename
    output_name = f"complex-MERS-mol{file_index}.pdb"
    output_path = os.path.join(output_dir, output_name)

    # Merge protein + ligand
    with open(output_path, "w") as out:

        # Write protein
        for line in protein_lines:
            out.write(line)

        # Write ligand ATOM/HETATM lines only
        with open(temp_pdb, "r") as lig:
            for line in lig:
                if line.startswith(("ATOM", "HETATM")):
                    out.write(line)

        out.write("END\n")

    print(f"Created {output_name}")

    file_index += 1

print("All complexes built correctly.")