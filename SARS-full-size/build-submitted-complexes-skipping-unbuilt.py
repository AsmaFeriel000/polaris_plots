import os
import subprocess
from rdkit import Chem
"""     
Build SARS-CoV Mpro complexes by merging the submitted ligand structures with the provided receptor PDB. The ligand structures are in SDF format and need
to be converted to PDB format using Open Babel. The final output will be a set of PDB files, each containing the receptor and one ligand, named "complex-SARS-mol<ID>.pdb". The receptor is taken from "sars-submission-complex/complex-SARS.pdb" and the ligands are taken from "submitted_structures/cs_optimised_molecules.sdf". 
The output complexes will be saved in the "merged_submitted_pdb" directory.

"""
# -----------------------------
# INPUT FILES
# -----------------------------
input_pdb = "sars-submission-complex/complex-SARS.pdb"
input_sdf = "submitted_structures/cs_optimised_molecules.sdf"

# -----------------------------
# OUTPUT DIRECTORIES
# -----------------------------
output_dir = "merged_submitted_pdb"
tmp_dir = os.path.join(output_dir, "_tmp_lig_pdbs")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# -----------------------------
# Indices to skip
# -----------------------------
skip_indices = {22, 33, 42, 94}

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

    # Skip reserved filename indices
    while file_index in skip_indices:
        file_index += 1

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
    output_name = f"complex-SARS-mol{file_index}.pdb"
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